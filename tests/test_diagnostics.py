from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from bridge_assistant.diagnostics import RunDiagnostics, default_log_dir


def read_status(diagnostics: RunDiagnostics) -> dict[str, object]:
    return json.loads(diagnostics.last_run_path.read_text(encoding="utf-8"))


def test_default_log_dir_uses_local_app_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_log_dir() == tmp_path / "BridgeAssistant" / "runs"


def test_default_diagnostics_keep_status_next_to_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    with RunDiagnostics(
        console_stream=io.StringIO(),
        heartbeat_interval=0,
    ) as diagnostics:
        pass

    assert diagnostics.log_dir == tmp_path / "BridgeAssistant" / "runs"
    assert diagnostics.last_run_path == tmp_path / "BridgeAssistant" / "last-run.json"


def test_context_logs_to_utf8_console_and_atomic_status(tmp_path: Path) -> None:
    console = io.StringIO()
    with RunDiagnostics(
        log_dir=tmp_path / "BridgeAssistant" / "runs",
        argv=["bridge-assistant", "--levels", "2"],
        pid=4321,
        console_stream=console,
    ) as diagnostics:
        diagnostics.set_state(
            "inputting",
            completed=1,
            newly_solved=1,
            current_puzzle=27,
            difficulty="EASY",
        )
        diagnostics.info("Processing puzzle %s", 27)

    status = read_status(diagnostics)
    assert status["status"] == "completed"
    assert status["state"] == "complete"
    assert status["completed"] == 1
    assert status["newly_solved"] == 1
    assert status["current_puzzle"] == 27
    assert status["difficulty"] == "EASY"
    assert status["pid"] == 4321
    assert status["argv"] == ["bridge-assistant", "--levels", "2"]
    assert status["log_path"] == str(diagnostics.log_path.resolve())
    assert isinstance(status["elapsed"], float)
    assert diagnostics.last_run_path == tmp_path / "BridgeAssistant" / "runs" / "last-run.json"
    assert diagnostics.log_path.name.endswith("-4321.log")
    assert "Processing puzzle 27" in diagnostics.log_path.read_text(encoding="utf-8")
    assert "Processing puzzle 27" in console.getvalue()
    assert not list(diagnostics.last_run_path.parent.glob(".last-run-*.tmp"))


def test_uncaught_exception_records_failure_and_traceback(tmp_path: Path) -> None:
    diagnostics: RunDiagnostics | None = None
    with pytest.raises(ValueError, match="broken puzzle"):
        with RunDiagnostics(
            log_dir=tmp_path / "runs",
            argv=["runner"],
            console_stream=io.StringIO(),
        ) as active:
            diagnostics = active
            raise ValueError("broken puzzle")

    assert diagnostics is not None
    status = read_status(diagnostics)
    assert status["status"] == "failed"
    assert status["state"] == "failed"
    assert status["error"] == "broken puzzle"
    log = diagnostics.log_path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in log
    assert "ValueError: broken puzzle" in log


def test_heartbeat_updates_running_status_and_manual_stop_is_preserved(
    tmp_path: Path,
) -> None:
    with RunDiagnostics(
        log_dir=tmp_path / "runs",
        console_stream=io.StringIO(),
    ) as diagnostics:
        diagnostics.heartbeat(
            state="waiting_for_change",
            completed=3,
            newly_solved=3,
            current_puzzle=99,
            difficulty="HARD",
        )
        running = read_status(diagnostics)
        assert running["status"] == "running"
        assert running["state"] == "waiting_for_change"
        diagnostics.mark_stopped(completed=3, newly_solved=3)

    stopped = read_status(diagnostics)
    assert stopped["status"] == "stopped"
    assert stopped["state"] == "stopped"
    assert stopped["completed"] == 3
    assert "Heartbeat:" in diagnostics.log_path.read_text(encoding="utf-8")


def test_only_ten_newest_log_files_are_retained(tmp_path: Path) -> None:
    log_dir = tmp_path / "runs"
    log_dir.mkdir()
    old_logs = []
    for index in range(10):
        path = log_dir / f"old-{index:02d}.log"
        path.write_text("old", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))
        old_logs.append(path)

    with RunDiagnostics(log_dir=log_dir, console_stream=io.StringIO()) as diagnostics:
        pass

    retained = list(log_dir.glob("*.log"))
    assert len(retained) == 10
    assert diagnostics.log_path in retained
    assert old_logs[0] not in retained
