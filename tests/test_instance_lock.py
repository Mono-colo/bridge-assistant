from __future__ import annotations

import os
import uuid

import pytest

from bridge_assistant import auto_runner
from bridge_assistant.instance_lock import AnotherInstanceError, SingleInstanceLock


@pytest.mark.skipif(os.name != "nt", reason="Windows named mutex")
def test_named_mutex_rejects_a_second_runner_and_releases_cleanly() -> None:
    name = f"Local\\BridgeAssistant-Test-{uuid.uuid4()}"
    first = SingleInstanceLock(name)
    second = SingleInstanceLock(name)
    first.acquire()
    try:
        with pytest.raises(AnotherInstanceError, match="already active"):
            second.acquire()
    finally:
        first.close()

    with SingleInstanceLock(name):
        pass


def test_automatic_main_rejects_a_second_instance_before_creating_logs(
    monkeypatch,
    capsys,
) -> None:
    class RejectingLock:
        def __enter__(self):
            raise AnotherInstanceError("automatic runner is already active")

        def __exit__(self, *_exc_info):
            return None

    monkeypatch.setattr(auto_runner, "SingleInstanceLock", RejectingLock)
    monkeypatch.setattr(
        auto_runner,
        "RunDiagnostics",
        lambda **_kwargs: pytest.fail("diagnostics must not replace the active run status"),
    )

    assert auto_runner.main(["--once"]) == 1
    assert "already active" in capsys.readouterr().err


def test_automatic_main_closes_diagnostics_when_initial_logging_fails(monkeypatch) -> None:
    events: list[str] = []

    class PassthroughLock:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

    class ExplodingDiagnostics:
        log_path = "unused.log"

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_exc_info):
            events.append("exit")
            return False

        def info(self, *_args):
            raise RuntimeError("logging failed")

    monkeypatch.setattr(auto_runner, "SingleInstanceLock", PassthroughLock)
    monkeypatch.setattr(auto_runner, "RunDiagnostics", ExplodingDiagnostics)

    with pytest.raises(RuntimeError, match="logging failed"):
        auto_runner.main(["--once"])

    assert events == ["enter", "exit"]
