from __future__ import annotations

from pathlib import Path

import pytest

from bridge_assistant.auto_runner import DesktopRuntime, ProgressFileError
from bridge_assistant.game_library import GameLibrary, GameOfficialAnswer, GamePuzzleDefinition


def definition(number: int) -> GamePuzzleDefinition:
    return GamePuzzleDefinition(
        asset_name="libraryVERY_EASY",
        number=number,
        difficulty="VERY_EASY",
        width=2,
        height=1,
        fingerprint=number,
        nodes=(),
        connections=(),
        magic_links=(),
        border_hints=(),
        official_answer=GameOfficialAnswer(()),
    )


def runtime_for(path: Path, monkeypatch) -> DesktopRuntime:
    runtime = DesktopRuntime(GameLibrary((definition(7), definition(8))))
    monkeypatch.setattr(runtime, "_find_solved_files", lambda: (path,))
    monkeypatch.setattr(runtime, "_wait", lambda _duration: None)
    return runtime


def test_solved_snapshot_parses_only_strict_ascii_integer_lines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "solved.dat"
    path.write_bytes(b"7\r\n8\r\n")

    snapshot = runtime_for(path, monkeypatch).solved_snapshot()

    assert snapshot.numbers == frozenset({7, 8})
    assert snapshot.ordered_numbers == (7, 8)
    assert snapshot.files[0][0] == str(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"7\n7\n", "duplicate puzzle number 7"),
        (b"7\n999\n", "unknown puzzle number 999"),
        (b"7\nnot-a-number\n", "invalid line 2"),
        (b"7\n\xff\n", "not ASCII text"),
    ],
)
def test_solved_snapshot_rejects_corrupt_progress(
    tmp_path: Path,
    monkeypatch,
    contents: bytes,
    message: str,
) -> None:
    path = tmp_path / "solved.dat"
    path.write_bytes(contents)

    with pytest.raises(ProgressFileError, match=message):
        runtime_for(path, monkeypatch).solved_snapshot()


def test_known_progress_file_disappearing_is_an_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "solved.dat"
    path.write_text("7\n", encoding="ascii")
    runtime = runtime_for(path, monkeypatch)
    assert runtime.solved_snapshot().numbers == frozenset({7})
    path.unlink()

    with pytest.raises(ProgressFileError, match="disappeared"):
        runtime.solved_snapshot()


def test_solved_snapshot_waits_for_partial_rewrite_to_finish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "solved.dat"
    path.write_text("7\n", encoding="ascii")
    runtime = runtime_for(path, monkeypatch)
    waits = 0

    def finish_rewrite(_duration: float) -> None:
        nonlocal waits
        waits += 1
        if waits == 1:
            path.write_text("7\n8\n", encoding="ascii")

    monkeypatch.setattr(runtime, "_wait", finish_rewrite)

    snapshot = runtime.solved_snapshot()

    assert snapshot.numbers == frozenset({7, 8})
    assert waits >= 2


def test_solved_snapshot_retries_a_temporarily_missing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "solved.dat"
    runtime = runtime_for(path, monkeypatch)
    waits = 0

    def finish_replacement(_duration: float) -> None:
        nonlocal waits
        waits += 1
        if waits == 1:
            path.write_text("7\n8\n", encoding="ascii")

    monkeypatch.setattr(runtime, "_wait", finish_replacement)

    snapshot = runtime.solved_snapshot()

    assert snapshot.numbers == frozenset({7, 8})
    assert waits >= 2
