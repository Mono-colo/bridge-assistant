from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO


_UNSET = object()


def default_log_dir() -> Path:
    """Return the per-user directory used for Bridge Assistant run logs."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "BridgeAssistant" / "runs"
    return Path.home() / "AppData" / "Local" / "BridgeAssistant" / "runs"


def _state_value(state: object) -> str:
    value = getattr(state, "value", state)
    return str(value)


class RunDiagnostics:
    """Persist one automatic-run log and its latest machine-readable status."""

    def __init__(
        self,
        *,
        log_dir: str | os.PathLike[str] | None = None,
        argv: Sequence[str] | None = None,
        pid: int | None = None,
        console_stream: TextIO | None = None,
        heartbeat_interval: float = 5.0,
    ) -> None:
        custom_log_dir = log_dir is not None
        self.log_dir = Path(log_dir) if custom_log_dir else default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_path = (
            self.log_dir / "last-run.json"
            if custom_log_dir
            else self.log_dir.parent / "last-run.json"
        )
        self.last_run_path.parent.mkdir(parents=True, exist_ok=True)

        self.pid = os.getpid() if pid is None else pid
        self._started_at = datetime.now().astimezone()
        self._started_monotonic = time.monotonic()
        filename_timestamp = self._started_at.strftime("%Y%m%d-%H%M%S-%f")
        self.log_path = self.log_dir / f"{filename_timestamp}-{self.pid}.log"
        self._lock = threading.RLock()
        self._closed = False
        self._finalized = False
        self._heartbeat_interval = max(0.0, heartbeat_interval)
        self._heartbeat_stop = threading.Event()

        logger_name = f"bridge_assistant.run.{self.pid}.{uuid.uuid4().hex}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._file_handler = logging.FileHandler(
            self.log_path,
            mode="a",
            encoding="utf-8",
        )
        self._file_handler.setFormatter(formatter)
        self._console_handler = logging.StreamHandler(console_stream)
        self._console_handler.setFormatter(formatter)
        self.logger.addHandler(self._file_handler)
        self.logger.addHandler(self._console_handler)
        self._application_logger = logging.getLogger("bridge_assistant")
        self._application_level = self._application_logger.level
        self._application_logger.setLevel(logging.INFO)
        self._application_logger.addHandler(self._file_handler)
        self._application_logger.addHandler(self._console_handler)

        self._record: dict[str, object] = {
            "status": "running",
            "timestamp": self._started_at.isoformat(timespec="seconds"),
            "updated_at": self._started_at.isoformat(timespec="seconds"),
            "pid": self.pid,
            "argv": list(sys.argv if argv is None else argv),
            "state": "idle",
            "completed": 0,
            "newly_solved": 0,
            "log_path": str(self.log_path.resolve()),
            "elapsed": 0.0,
        }
        self.logger.info("Run started: pid=%s argv=%r", self.pid, self._record["argv"])
        self._prune_logs()
        self._persist()
        self._heartbeat_thread: threading.Thread | None = None
        if self._heartbeat_interval > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="bridge-assistant-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def __enter__(self) -> RunDiagnostics:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            if not self._finalized:
                if exc_value is not None and exc_type is not None:
                    self.mark_failed(
                        exc_value,
                        exc_info=(exc_type, exc_value, traceback),
                    )
                else:
                    self.mark_completed()
        finally:
            self.close()
        return False

    @property
    def status(self) -> str:
        return str(self._record["status"])

    def set_state(
        self,
        state: object,
        *,
        completed: int | None = None,
        newly_solved: int | None = None,
        current_puzzle: int | None | object = _UNSET,
        difficulty: str | None | object = _UNSET,
    ) -> None:
        with self._lock:
            self._ensure_open()
            self._record["state"] = _state_value(state)
            self._update_progress(
                completed=completed,
                newly_solved=newly_solved,
                current_puzzle=current_puzzle,
                difficulty=difficulty,
            )
            self.logger.info(
                "State changed: state=%s completed=%s newly_solved=%s",
                self._record["state"],
                self._record["completed"],
                self._record["newly_solved"],
            )
            self._persist()

    def heartbeat(
        self,
        *,
        state: object | None = None,
        completed: int | None = None,
        newly_solved: int | None = None,
        current_puzzle: int | None | object = _UNSET,
        difficulty: str | None | object = _UNSET,
    ) -> None:
        with self._lock:
            self._ensure_open()
            if state is not None:
                self._record["state"] = _state_value(state)
            self._update_progress(
                completed=completed,
                newly_solved=newly_solved,
                current_puzzle=current_puzzle,
                difficulty=difficulty,
            )
            self.logger.info(
                "Heartbeat: state=%s puzzle=%s difficulty=%s elapsed=%.3fs",
                self._record["state"],
                self._record.get("current_puzzle"),
                self._record.get("difficulty"),
                self._elapsed(),
            )
            self._persist()

    def info(self, message: str, *args: object) -> None:
        self.logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        self.logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        self.logger.error(message, *args)

    def mark_completed(
        self,
        completed: int | None = None,
        *,
        newly_solved: int | None = None,
        state: object = "complete",
    ) -> None:
        self._finalize(
            "completed",
            state,
            completed=completed,
            newly_solved=newly_solved,
        )

    def mark_stopped(
        self,
        completed: int | None = None,
        *,
        newly_solved: int | None = None,
        state: object = "stopped",
    ) -> None:
        self._finalize(
            "stopped",
            state,
            completed=completed,
            newly_solved=newly_solved,
        )

    def mark_failed(
        self,
        error: BaseException | str,
        *,
        state: object = "failed",
        completed: int | None = None,
        newly_solved: int | None = None,
        exc_info: (
            tuple[type[BaseException], BaseException, TracebackType | None]
            | bool
            | None
        ) = None,
    ) -> None:
        if exc_info is None and isinstance(error, BaseException):
            exc_info = (type(error), error, error.__traceback__)
        error_text = str(error)
        with self._lock:
            self._ensure_open()
            if self._finalized:
                return
            self._record["error"] = error_text
            self.logger.error("Run failed: %s", error_text, exc_info=exc_info)
            self._finalize_locked(
                "failed",
                state,
                completed=completed,
                newly_solved=newly_solved,
            )

    def close(self) -> None:
        self._heartbeat_stop.set()
        heartbeat_thread = self._heartbeat_thread
        if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join(timeout=1.0)
        with self._lock:
            if self._closed:
                return
            for handler in (self._file_handler, self._console_handler):
                self._application_logger.removeHandler(handler)
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
            self._application_logger.setLevel(self._application_level)
            self._closed = True

    def _finalize(
        self,
        status: str,
        state: object,
        *,
        completed: int | None,
        newly_solved: int | None,
    ) -> None:
        with self._lock:
            self._ensure_open()
            if self._finalized:
                return
            self._finalize_locked(
                status,
                state,
                completed=completed,
                newly_solved=newly_solved,
            )

    def _finalize_locked(
        self,
        status: str,
        state: object,
        *,
        completed: int | None,
        newly_solved: int | None,
    ) -> None:
        self._record["status"] = status
        self._record["state"] = _state_value(state)
        self._update_progress(
            completed=completed,
            newly_solved=newly_solved,
            current_puzzle=_UNSET,
            difficulty=_UNSET,
        )
        self._finalized = True
        self.logger.info(
            "Run %s: completed=%s newly_solved=%s elapsed=%.3fs",
            status,
            self._record["completed"],
            self._record["newly_solved"],
            self._elapsed(),
        )
        self._persist()

    def _update_progress(
        self,
        *,
        completed: int | None,
        newly_solved: int | None,
        current_puzzle: int | None | object,
        difficulty: str | None | object,
    ) -> None:
        if completed is not None:
            self._record["completed"] = completed
        if newly_solved is not None:
            self._record["newly_solved"] = newly_solved
        if current_puzzle is not _UNSET:
            self._set_optional("current_puzzle", current_puzzle)
        if difficulty is not _UNSET:
            self._set_optional("difficulty", difficulty)

    def _set_optional(self, key: str, value: object) -> None:
        if value is None:
            self._record.pop(key, None)
        else:
            self._record[key] = value

    def _persist(self) -> None:
        now = datetime.now().astimezone()
        self._record["updated_at"] = now.isoformat(timespec="seconds")
        self._record["elapsed"] = round(self._elapsed(), 3)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.last_run_path.parent,
                prefix=".last-run-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(self._record, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, self.last_run_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def _prune_logs(self) -> None:
        candidates: list[tuple[int, str, Path]] = []
        for path in self.log_dir.glob("*.log"):
            try:
                candidates.append((path.stat().st_mtime_ns, path.name, path))
            except OSError:
                continue
        candidates.sort(reverse=True)
        for _, _, path in candidates[10:]:
            try:
                path.unlink()
            except OSError as error:
                self.logger.warning("Could not remove old log %s: %s", path, error)

    def _elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._started_monotonic)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            try:
                with self._lock:
                    if self._closed or self._finalized:
                        return
                self.heartbeat()
            except Exception:
                self.logger.exception("Could not persist heartbeat")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("run diagnostics are closed")
