from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from bridge_assistant.auto_runner import (
    AutomaticRunComplete,
    AutoRunner,
    ControlLayout,
    DesktopRuntime,
    DifficultyMenuError,
    InputConfirmationError,
    MagicLinkSolutionError,
    PreparedLevel,
    RunnerState,
    SolvedSnapshot,
    difficulty_button_centers,
    is_target_window,
    representative_solution,
    run as run_automatic,
    victory_indicator_score,
    victory_indicator_visible,
)
from bridge_assistant.game_library import (
    GameLibrary,
    GameOfficialAnswer,
    GamePuzzleDefinition,
    GamePuzzleMatch,
)
from bridge_assistant.model import (
    AppWindow,
    BridgeAssistantError,
    Edge,
    Island,
    MagicLink,
    Puzzle,
    ScreenRegion,
    Solution,
)


def puzzle() -> Puzzle:
    return Puzzle(
        islands=[Island(0, 10, 20, 2), Island(1, 70, 20, 2), Island(2, 130, 20, 2)],
        edges=[Edge(0, 0, 1, "h"), Edge(1, 1, 2, "h"), Edge(2, 0, 2, "h")],
        magic_links=[MagicLink(0, 1), MagicLink(1, 2)],
    )


def definition(number: int, difficulty: str = "TEST") -> GamePuzzleDefinition:
    return GamePuzzleDefinition(
        asset_name="libraryTEST",
        number=number,
        difficulty=difficulty,
        width=3,
        height=1,
        fingerprint=number,
        nodes=(),
        connections=(),
        magic_links=(),
        border_hints=(),
        official_answer=GameOfficialAnswer(()),
    )


def prepared(number: int = 1, answer: Solution | None = None) -> PreparedLevel:
    board = puzzle()
    match = GamePuzzleMatch(
        definition(number),
        board,
        answer or Solution({0: 1, 1: 1, 2: 1}),
        {},
        {},
        {},
        {},
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    return PreparedLevel(
        AppWindow(1, 2, "BridgeFix 231", "Bridge-Windows64.exe"),
        ScreenRegion(100, 100, 500, 400),
        ScreenRegion(120, 120, 300, 200),
        match,
        np.zeros((400, 500, 3), dtype=np.uint8),
        SolvedSnapshot(),
    )


def test_magic_links_are_transitively_deduplicated_to_one_representative() -> None:
    result = representative_solution(puzzle(), Solution({0: 2, 1: 2, 2: 2}))
    assert result.bridges == {0: 2}


def test_target_window_requires_exact_title_and_process() -> None:
    assert is_target_window(
        AppWindow(1, 2, "BridgeFix 231", r"C:\Games\Bridge-Windows64.exe")
    )
    assert not is_target_window(AppWindow(1, 2, "BridgeFix 231", "notepad.exe"))
    assert not is_target_window(
        AppWindow(1, 2, "Another window", "Bridge-Windows64.exe")
    )


def test_magic_group_requires_equal_answers_before_input() -> None:
    with pytest.raises(MagicLinkSolutionError, match="inconsistent"):
        representative_solution(puzzle(), Solution({0: 1, 1: 2, 2: 1}))


@dataclass
class FakeRuntime:
    confirmations: list[bool]
    victories: list[bool]

    def __post_init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.levels = [prepared(1), prepared(2), prepared(3)]
        self.level_index = 0

    def wait_for_window(self, _stop_event):
        self.events.append(("window", None))
        return self.levels[0].window

    def prepare_level(self, _window, difficulty):
        self.events.append(("prepare", difficulty))
        return self.levels[self.level_index]

    def reset(self, level):
        self.events.append(("reset", level.number))

    def clear_edge(self, level, action):
        self.events.append(("right", (level.number, action.start)))

    def add_bridge(self, level, action):
        self.events.append(("left", (level.number, action.start)))
        return self.confirmations.pop(0)

    def victory_confirmed(self, level):
        self.events.append(("victory", level.number))
        return self.victories.pop(0)

    def click_next(self, level):
        self.events.append(("next", level.number))

    def wait_for_level_change(self, level, difficulty, minimum_delay):
        self.events.append(("wait_change", (level.number, difficulty, minimum_delay)))
        self.level_index += 1
        return self.levels[self.level_index]


class NoChangeRuntime(FakeRuntime):
    def wait_for_level_change(self, level, difficulty, minimum_delay):
        self.events.append(("wait_change", (level.number, difficulty, minimum_delay)))
        raise RuntimeError("level did not change")


class TerminalRuntime(FakeRuntime):
    def click_next(self, level):
        self.events.append(("next", level.number))
        return False


class AlreadyCompleteRuntime(FakeRuntime):
    def prepare_level(self, _window, difficulty):
        self.events.append(("prepare", difficulty))
        raise AutomaticRunComplete("already complete")


class CompleteAfterNextRuntime(FakeRuntime):
    def wait_for_level_change(self, level, difficulty, minimum_delay):
        self.events.append(("wait_change", (level.number, difficulty, minimum_delay)))
        raise AutomaticRunComplete("completed after Next")


def event_names(runtime: FakeRuntime) -> list[str]:
    return [name for name, _payload in runtime.events]


def test_confirmation_failure_resets_and_retries_whole_level_without_blind_clicks() -> None:
    runtime = FakeRuntime([False, True], [True])
    runner = AutoRunner(runtime, level_retries=2)

    assert runner.run(levels=1) == 1

    assert event_names(runtime) == [
        "window",
        "prepare",
        "reset",
        "right",
        "left",
        "reset",
        "right",
        "left",
        "victory",
        "next",
        "wait_change",
    ]
    assert event_names(runtime).count("left") == 2
    assert event_names(runtime).count("reset") == 2


def test_double_bridge_is_cleared_once_and_left_clicked_exactly_twice() -> None:
    runtime = FakeRuntime([True, True], [True])
    runtime.levels[0] = prepared(1, Solution({0: 2, 1: 2, 2: 2}))

    AutoRunner(runtime).run(levels=1)

    assert event_names(runtime).count("right") == 1
    assert event_names(runtime).count("left") == 2


def test_retry_limit_aborts_after_bounded_attempts_and_leaves_board_reset() -> None:
    runtime = FakeRuntime([False, False], [])
    runner = AutoRunner(runtime, level_retries=2)

    with pytest.raises(InputConfirmationError):
        runner.run(levels=1)

    assert event_names(runtime).count("left") == 2
    assert event_names(runtime).count("reset") == 3
    assert "next" not in event_names(runtime)
    assert runner.state == RunnerState.FAILED


def test_next_is_clicked_exactly_once_after_victory_and_change_is_awaited() -> None:
    runtime = FakeRuntime([True], [True])
    states: list[RunnerState] = []

    AutoRunner(runtime, state_changed=states.append).run(levels=1)

    assert event_names(runtime).count("next") == 1
    wait = next(payload for name, payload in runtime.events if name == "wait_change")
    assert wait == (1, None, 1.5)
    assert states[-4:] == [
        RunnerState.VERIFYING_VICTORY,
        RunnerState.ADVANCING,
        RunnerState.WAITING_FOR_CHANGE,
        RunnerState.COMPLETE,
    ]


def test_next_is_never_clicked_without_confirmed_victory() -> None:
    runtime = FakeRuntime([True, True], [False, False])
    runner = AutoRunner(runtime, level_retries=2)

    with pytest.raises(Exception, match="Victory was not confirmed"):
        runner.run(levels=1)

    assert "next" not in event_names(runtime)
    assert event_names(runtime).count("reset") == 3


def test_failed_change_confirmation_does_not_click_next_again() -> None:
    runtime = NoChangeRuntime([True], [True])

    with pytest.raises(RuntimeError, match="did not change"):
        AutoRunner(runtime).run(levels=1)

    assert event_names(runtime).count("next") == 1
    assert event_names(runtime).count("wait_change") == 1


def test_two_levels_reuse_changed_level_and_preserve_state_order() -> None:
    runtime = FakeRuntime([True, True], [True, True])
    states: list[RunnerState] = []

    assert AutoRunner(runtime, state_changed=states.append).run(levels=2, difficulty="test") == 2

    assert event_names(runtime).count("prepare") == 1
    assert event_names(runtime).count("next") == 2
    assert states[0] == RunnerState.WAITING_FOR_WINDOW
    assert states[-1] == RunnerState.COMPLETE


def test_recovered_level_is_advanced_but_not_counted() -> None:
    runtime = FakeRuntime([True], [True, True])
    runtime.levels[0].advance_only = True
    runner = AutoRunner(runtime)

    assert runner.run(levels=1) == 1

    assert event_names(runtime).count("next") == 2
    assert event_names(runtime).count("reset") == 1
    assert runner.completed_levels == 1


def test_public_run_rejects_unknown_difficulty_before_waiting_for_window() -> None:
    runtime = FakeRuntime([], [])

    with pytest.raises(DifficultyMenuError, match="Unknown BridgeFix difficulty"):
        run_automatic(difficulty="not-a-difficulty", runtime=runtime)

    assert runtime.events == []


def test_bridgefix_control_layout_targets_visible_buttons() -> None:
    region = ScreenRegion(0, 0, 1600, 1200)
    controls = ControlLayout()

    assert controls.reset_point(region) == (1485, 42)
    assert controls.menu_point(region) == (1561, 42)
    assert controls.next_point(region) == (1561, 1157)


def test_difficulty_buttons_are_returned_in_two_row_reading_order() -> None:
    image = np.full((1200, 1600, 3), 80, dtype=np.uint8)
    expected = []
    for y in (465, 667):
        for x in (306, 507, 708, 909, 1110):
            cv2.rectangle(image, (x, y), (x + 183, y + 184), (235, 235, 235), 3)
            expected.append((x + 92, y + 92))

    actual = difficulty_button_centers(image)

    assert len(actual) == 10
    assert all(abs(x - ex) <= 2 and abs(y - ey) <= 2 for (x, y), (ex, ey) in zip(actual, expected))


def test_difficulty_grid_is_inferred_from_one_complete_row() -> None:
    image = np.full((1440, 2560, 3), 80, dtype=np.uint8)
    x_positions = (559, 853, 1146, 1439, 1732)
    for x in x_positions:
        cv2.rectangle(image, (x, 817), (x + 268, 817 + 268), (235, 235, 235), 3)
    for x in x_positions[:2]:
        cv2.rectangle(image, (x, 524), (x + 268, 524 + 268), (235, 235, 235), 3)

    actual = difficulty_button_centers(image)

    assert len(actual) == 10
    assert [point[0] for point in actual[:5]] == [693, 987, 1280, 1573, 1866]
    assert all(abs(point[1] - 658) <= 10 for point in actual[:5])
    assert all(abs(point[1] - 951) <= 2 for point in actual[5:])


def test_difficulty_grid_infers_clipped_fifth_column_from_four_by_four() -> None:
    image = np.full((960, 1280, 3), 80, dtype=np.uint8)
    for y in (465, 667):
        for x in (306, 507, 708, 909):
            cv2.rectangle(image, (x, y), (x + 183, y + 184), (235, 235, 235), 3)
        cv2.rectangle(image, (1135, y), (1279, y + 184), (225, 225, 225), -1)

    actual = difficulty_button_centers(image)

    assert len(actual) == 10
    expected_xs = [398, 599, 800, 1001, 1202]
    assert all(abs(point[0] - expected) <= 2 for point, expected in zip(actual[:5], expected_xs))
    assert all(abs(point[0] - expected) <= 2 for point, expected in zip(actual[5:], expected_xs))


def test_visible_win_state_advances_without_resetting_or_inputting() -> None:
    runtime = FakeRuntime([], [True])
    runtime.levels[0].solved_before = SolvedSnapshot(frozenset({1}))
    runtime.levels[0].client_image[352:, 470:] = 255

    assert AutoRunner(runtime).run(levels=1) == 1

    assert event_names(runtime) == [
        "window",
        "prepare",
        "victory",
        "next",
        "wait_change",
    ]


def test_historical_solved_id_without_glow_is_replayed_normally() -> None:
    runtime = FakeRuntime([True], [False, True])
    runtime.levels[0].solved_before = SolvedSnapshot(frozenset({1}))

    assert AutoRunner(runtime).run(levels=1) == 1

    assert "reset" in event_names(runtime)
    assert "left" in event_names(runtime)


def test_recovered_victory_never_resets_or_inputs_if_confirmation_disappears() -> None:
    runtime = FakeRuntime([], [False])
    runtime.levels[0].advance_only = True
    runner = AutoRunner(runtime, level_retries=3)

    with pytest.raises(Exception, match="recovered puzzle"):
        runner.run(levels=1)

    assert event_names(runtime) == ["window", "prepare", "victory"]


def test_victory_indicator_requires_bright_next_glow() -> None:
    image = np.full((400, 500, 3), 190, dtype=np.uint8)
    assert victory_indicator_score(image) == 0.0
    assert not victory_indicator_visible(image)

    image[352:, 470:] = 255
    assert victory_indicator_score(image) == 1.0
    assert victory_indicator_visible(image)


def test_victory_indicator_rejects_partial_bright_menu_panel() -> None:
    image = np.full((400, 500, 3), 120, dtype=np.uint8)
    image[352:357, 470:] = 255

    assert 0.08 <= victory_indicator_score(image) < 0.15
    assert not victory_indicator_visible(image)


def test_prepare_level_recovers_dimmed_victory_from_last_saved_puzzle(monkeypatch) -> None:
    game_definition = definition(7, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((game_definition, definition(8, "VERY_EASY"))),
        driver=RecordingDriver(),
    )
    window = prepared().window
    region = ScreenRegion(100, 100, 500, 400)
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    image[352:, 470:] = 255

    def unreadable_board(_image):
        raise BridgeAssistantError("victory overlay hides the board")

    monkeypatch.setattr(runtime, "_ensure_running", lambda: None)
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr("bridge_assistant.auto_runner.locate_board", unreadable_board)
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({3, 7}), (), (3, 7)),
    )

    level = runtime.prepare_level(window, "VERY_EASY")

    assert level.number == 7
    assert level.client_region == region
    assert level.board_region == region
    assert level.match.puzzle.islands == []
    assert level.solved_before.ordered_numbers == (3, 7)
    assert level.advance_only


def test_victory_recovery_requires_glow_and_uses_last_saved_puzzle() -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(7, "VERY_EASY"),)),
        driver=RecordingDriver(),
    )
    window = prepared().window
    region = ScreenRegion(100, 100, 500, 400)
    snapshot = SolvedSnapshot(frozenset({7}), (), (7,))
    image = np.zeros((400, 500, 3), dtype=np.uint8)

    assert runtime._prepare_victory_recovery(window, region, image, snapshot) is None

    image[352:, 470:] = 255
    recovery = runtime._prepare_victory_recovery(window, region, image, snapshot)
    assert recovery is not None
    assert recovery.advance_only


def test_recovered_victory_clicks_next_without_using_guessed_completion(monkeypatch) -> None:
    game_definition = definition(7, "VERY_EASY")
    runtime = DesktopRuntime(GameLibrary((game_definition,)), driver=RecordingDriver())
    level = prepared(7)
    level.match.definition = game_definition
    level.advance_only = True
    events: list[str] = []
    monkeypatch.setattr(runtime, "_wait_for_next_ready", lambda _level: events.append("ready"))
    monkeypatch.setattr(
        runtime,
        "_ui_click",
        lambda _window, _region, _point: events.append("click"),
    )

    assert runtime.click_next(level)

    assert events == ["ready", "click"]


def test_terminal_completion_does_not_wait_for_another_level() -> None:
    runtime = TerminalRuntime([True], [True])
    runner = AutoRunner(runtime)

    assert runner.run(levels=1) == 1

    assert event_names(runtime)[-2:] == ["victory", "next"]
    assert runner.state == RunnerState.COMPLETE


def test_already_complete_runtime_ends_normally_without_a_level() -> None:
    runtime = AlreadyCompleteRuntime([], [])
    runner = AutoRunner(runtime)

    assert runner.run() == 0

    assert event_names(runtime) == ["window", "prepare"]
    assert runner.state == RunnerState.COMPLETE


def test_completion_discovered_after_next_ends_normally() -> None:
    runtime = CompleteAfterNextRuntime([True], [True])
    runner = AutoRunner(runtime)

    assert runner.run(levels=2) == 1

    assert event_names(runtime)[-2:] == ["next", "wait_change"]
    assert runner.state == RunnerState.COMPLETE


def test_desktop_runtime_rejects_unknown_difficulty_before_capture() -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(7, "VERY_EASY"),)),
        driver=RecordingDriver(),
    )

    with pytest.raises(Exception, match="Unknown BridgeFix difficulty"):
        runtime.prepare_level(prepared().window, "not-a-difficulty")


class RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], str]] = []
        self.FAILSAFE = False
        self.PAUSE = 0.0

    def click(self, x: int, y: int, *, button: str) -> None:
        self.calls.append(((x, y), button))

    def moveTo(self, *_args, **_kwargs) -> None:
        pass


def test_desktop_reset_right_clears_each_magic_group_once(monkeypatch) -> None:
    driver = RecordingDriver()
    runtime = DesktopRuntime(
        SimpleNamespace(puzzles=(definition(1),)),
        driver=driver,
    )
    level = prepared()
    monkeypatch.setattr("bridge_assistant.auto_runner.activate_window", lambda _window: None)
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.grab_screen",
        lambda region: Image.fromarray(np.zeros((region.height, region.width, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        runtime,
        "_level_click",
        lambda _level, point, *, button: driver.click(*point, button=button),
    )

    runtime.reset(level)

    assert driver.calls == [((160, 140), "right")]


def test_click_is_blocked_if_the_bound_client_region_changed(monkeypatch) -> None:
    driver = RecordingDriver()
    runtime = DesktopRuntime(SimpleNamespace(puzzles=(definition(1),)), driver=driver)
    level = prepared()
    monkeypatch.setattr("bridge_assistant.auto_runner.activate_window", lambda _window: None)
    monkeypatch.setattr("bridge_assistant.auto_runner.validate_window", lambda _window: None)
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.window_client_region",
        lambda _window: ScreenRegion(100, 100, 501, 400),
    )

    with pytest.raises(Exception, match="moved, resized"):
        runtime._level_click(level, (160, 140), button="right")

    assert driver.calls == []
