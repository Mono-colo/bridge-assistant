from __future__ import annotations

import logging
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
    DifficultyMenuAction,
    DifficultyMismatchError,
    DifficultyMenuError,
    InputConfirmationError,
    LevelDidNotAdvanceError,
    LevelDisposition,
    MagicLinkSolutionError,
    PreparedLevel,
    ProgressDesynchronizedError,
    VictoryNotConfirmedError,
    RunnerState,
    SolvedSnapshot,
    client_frames_stable,
    difficulty_button_centers,
    difficulty_completion_states,
    difficulty_percentage_digit_counts,
    is_target_window,
    next_glow_pulse_score,
    next_glow_pulse_visible,
    representative_solution,
    run as run_automatic,
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
    Recognition,
    ScreenRegion,
    Solution,
    TargetWindowError,
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


def prepared(
    number: int = 1,
    answer: Solution | None = None,
    *,
    solved_before: SolvedSnapshot | None = None,
    disposition: LevelDisposition = LevelDisposition.UNSOLVED,
) -> PreparedLevel:
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
        solved_before or SolvedSnapshot(),
        disposition,
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


@pytest.mark.parametrize(
    ("timeout_kwargs", "expected"),
    [
        ({}, 30.0),
        ({"change_timeout": 45.0}, 45.0),
        ({"advance_timeout": 12.0}, 12.0),
        ({"change_timeout": 45.0, "advance_timeout": 12.0}, 12.0),
        ({"change_timeout": 0.5}, 1.5),
    ],
    ids=["default", "legacy", "new", "new-wins", "minimum-delay"],
)
def test_transition_timeout_compatibility(
    timeout_kwargs: dict[str, float],
    expected: float,
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"),)),
        driver=RecordingDriver(),
        **timeout_kwargs,
    )

    assert runtime.change_timeout == expected
    assert runtime.advance_timeout == expected


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

    def advance_level(self, level, difficulty, minimum_delay):
        self.events.append(("advance", (level.number, difficulty, minimum_delay)))
        if self.level_index + 1 >= len(self.levels):
            return None
        self.level_index += 1
        return self.levels[self.level_index]


class NoChangeRuntime(FakeRuntime):
    def advance_level(self, level, difficulty, minimum_delay):
        self.events.append(("advance", (level.number, difficulty, minimum_delay)))
        raise RuntimeError("level did not change")


class TerminalRuntime(FakeRuntime):
    def advance_level(self, level, difficulty, minimum_delay):
        self.events.append(("advance", (level.number, difficulty, minimum_delay)))
        return None


class AlreadyCompleteRuntime(FakeRuntime):
    def prepare_level(self, _window, difficulty):
        self.events.append(("prepare", difficulty))
        raise AutomaticRunComplete("already complete")


class CompleteAfterNextRuntime(FakeRuntime):
    def advance_level(self, level, difficulty, minimum_delay):
        self.events.append(("advance", (level.number, difficulty, minimum_delay)))
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
        "advance",
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
    assert "advance" not in event_names(runtime)
    assert runner.state == RunnerState.FAILED


@pytest.mark.parametrize(
    ("error", "outcome", "state"),
    [
        (RuntimeError("boom"), "failed", RunnerState.FAILED),
        (InterruptedError("stopped"), "stopped", RunnerState.STOPPED),
    ],
)
def test_terminal_summary_is_logged_for_failure_and_stop(
    monkeypatch,
    caplog,
    error: Exception,
    outcome: str,
    state: RunnerState,
) -> None:
    runtime = FakeRuntime([], [])
    runner = AutoRunner(runtime)

    def abort(_stop_event):
        raise error

    monkeypatch.setattr(runtime, "wait_for_window", abort)
    with caplog.at_level(logging.INFO, logger="bridge_assistant.auto_runner"):
        with pytest.raises(type(error), match=str(error)):
            runner.run()

    assert any(f"summary outcome={outcome}" in message for message in caplog.messages)
    assert runner.state is state


def test_advance_is_requested_once_after_victory() -> None:
    runtime = FakeRuntime([True], [True])
    states: list[RunnerState] = []

    AutoRunner(runtime, state_changed=states.append).run(levels=1)

    assert event_names(runtime).count("advance") == 1
    advance = next(payload for name, payload in runtime.events if name == "advance")
    assert advance == (1, None, 1.5)
    assert states[-4:] == [
        RunnerState.VERIFYING_VICTORY,
        RunnerState.ADVANCING,
        RunnerState.WAITING_FOR_CHANGE,
        RunnerState.COMPLETE,
    ]


def test_advance_is_never_requested_without_confirmed_victory() -> None:
    runtime = FakeRuntime([True, True], [False, False])
    runner = AutoRunner(runtime, level_retries=2)

    with pytest.raises(Exception, match="Victory was not confirmed"):
        runner.run(levels=1)

    assert "advance" not in event_names(runtime)
    assert event_names(runtime).count("reset") == 3


def test_failed_advance_is_not_reissued_by_the_runner() -> None:
    runtime = NoChangeRuntime([True], [True])

    with pytest.raises(RuntimeError, match="did not change"):
        AutoRunner(runtime).run(levels=1)

    assert event_names(runtime).count("advance") == 1


def test_two_levels_reuse_changed_level_and_preserve_state_order() -> None:
    runtime = FakeRuntime([True, True], [True, True])
    states: list[RunnerState] = []

    assert AutoRunner(runtime, state_changed=states.append).run(levels=2, difficulty="test") == 2

    assert event_names(runtime).count("prepare") == 1
    assert event_names(runtime).count("advance") == 2
    assert states[0] == RunnerState.WAITING_FOR_WINDOW
    assert states[-1] == RunnerState.COMPLETE


def test_recovered_level_is_advanced_but_not_counted() -> None:
    runtime = FakeRuntime([True], [True])
    runtime.levels[0] = prepared(
        1,
        solved_before=SolvedSnapshot(frozenset({1})),
        disposition=LevelDisposition.RECOVERED_EXISTING,
    )
    runner = AutoRunner(runtime)

    assert runner.run(levels=1) == 1

    assert event_names(runtime).count("advance") == 2
    assert event_names(runtime).count("reset") == 1
    assert runner.completed_levels == 1
    assert runner.recovered_levels == 1


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


def rendered_difficulty_percentages(
    labels: tuple[str, ...],
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    step = round(150 * scale)
    image = np.full(
        (round(700 * scale), round(1100 * scale), 3),
        235,
        dtype=np.uint8,
    )
    centers = tuple(
        (round(250 * scale) + column * step, round(250 * scale) + row * step)
        for row in range(2)
        for column in range(5)
    )
    font_scale = 0.72 * scale
    thickness = max(1, round(scale))
    for (center_x, center_y), label in zip(centers, labels):
        (label_width, _label_height), _baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        cv2.putText(
            image,
            label,
            (
                round(center_x + step * 0.37 - label_width),
                round(center_y + step * 0.40),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )
    return image, centers


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_difficulty_percentages_identify_exact_100_at_multiple_scales(
    scale: float,
) -> None:
    labels = (
        "100.0%",
        "99.8%",
        "20.0%",
        "9.8%",
        "0.0%",
        "10.4%",
        "100.0%",
        "28.2%",
        "1.0%",
        "15.4%",
    )
    image, centers = rendered_difficulty_percentages(labels, scale=scale)

    digit_counts = difficulty_percentage_digit_counts(image, centers)
    states = difficulty_completion_states(image, centers)

    assert digit_counts == (3, 2, 2, 1, 1, 2, 3, 2, 1, 2)
    assert states == (
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )


def test_difficulty_percentage_with_missing_decimal_is_unreadable() -> None:
    image, centers = rendered_difficulty_percentages(
        ("100 0%",) + ("20.0%",) * 9
    )

    digit_counts = difficulty_percentage_digit_counts(image, centers)
    states = difficulty_completion_states(image, centers)

    assert digit_counts[0] is None
    assert states[0] is None


def test_difficulty_percentage_ignores_sparse_single_pixel_noise() -> None:
    labels = (
        "100.0%",
        "99.8%",
        "20.0%",
        "9.8%",
        "0.0%",
        "10.4%",
        "100.0%",
        "28.2%",
        "1.0%",
        "15.4%",
    )
    image, centers = rendered_difficulty_percentages(labels)
    for center_x, center_y in centers:
        for offset_x, offset_y in ((-42, 31), (-20, 36), (5, 32), (35, 34)):
            image[center_y + offset_y, center_x + offset_x] = 0

    assert difficulty_completion_states(image, centers) == (
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )


def test_current_incomplete_difficulty_ignores_unrelated_unreadable_labels() -> None:
    decision = DesktopRuntime._difficulty_menu_decision(
        (False,) + (None,) * 9,
        "VERY_EASY",
        stop_after_current=False,
    )

    assert decision.action is DifficultyMenuAction.CONTINUE_CURRENT
    assert decision.target is None


def test_completed_difficulty_does_not_skip_an_unreadable_next_label() -> None:
    decision = DesktopRuntime._difficulty_menu_decision(
        (True, None, False) + (True,) * 7,
        "VERY_EASY",
        stop_after_current=False,
    )

    assert decision.action is DifficultyMenuAction.RETRY
    assert decision.target is None


def test_unreadable_labels_after_first_incomplete_do_not_block_switch() -> None:
    decision = DesktopRuntime._difficulty_menu_decision(
        (True, True, False) + (None,) * 7,
        "VERY_EASY",
        stop_after_current=False,
    )

    assert decision.action is DifficultyMenuAction.SWITCH
    assert decision.target == "MEDIUM"


def test_terminal_completion_requires_every_difficulty_to_be_readable_and_full() -> None:
    complete = DesktopRuntime._difficulty_menu_decision(
        (True,) * 10,
        "VERY_EASY",
        stop_after_current=False,
    )
    unreadable = DesktopRuntime._difficulty_menu_decision(
        (True,) * 9 + (None,),
        "VERY_EASY",
        stop_after_current=False,
    )

    assert complete.action is DifficultyMenuAction.COMPLETE
    assert unreadable.action is DifficultyMenuAction.RETRY


def test_requested_completed_difficulty_ignores_other_unreadable_labels() -> None:
    decision = DesktopRuntime._difficulty_selection_decision(
        (None, True) + (None,) * 8,
        "EASY",
    )

    assert decision.action is DifficultyMenuAction.COMPLETE


def test_startup_selection_stops_at_first_incomplete_before_unreadable_labels() -> None:
    decision = DesktopRuntime._difficulty_selection_decision(
        (False,) + (None,) * 9,
        None,
    )

    assert decision.action is DifficultyMenuAction.SWITCH
    assert decision.target == "VERY_EASY"


def test_requested_current_difficulty_completes_despite_other_unreadable_labels() -> None:
    decision = DesktopRuntime._difficulty_menu_decision(
        (None, True) + (None,) * 8,
        "EASY",
        stop_after_current=True,
    )

    assert decision.action is DifficultyMenuAction.COMPLETE


def test_next_incomplete_difficulty_wraps_from_last_item_to_first() -> None:
    decision = DesktopRuntime._difficulty_menu_decision(
        (False,) + (None,) * 8 + (True,),
        "HUGE_TRICKY",
        stop_after_current=False,
    )

    assert decision.action is DifficultyMenuAction.SWITCH
    assert decision.target == "VERY_EASY"


def test_unsolved_level_is_input_even_if_bottom_right_is_bright() -> None:
    runtime = FakeRuntime([True], [True])
    runtime.levels[0].client_image[352:, 470:] = 255

    assert AutoRunner(runtime).run(levels=1) == 1

    assert event_names(runtime) == [
        "window",
        "prepare",
        "reset",
        "right",
        "left",
        "victory",
        "advance",
    ]


def test_two_consecutive_recovered_levels_fail_as_desynchronized() -> None:
    runtime = FakeRuntime([], [])
    runtime.levels[0] = prepared(
        1,
        solved_before=SolvedSnapshot(frozenset({1, 2})),
        disposition=LevelDisposition.RECOVERED_EXISTING,
    )
    runtime.levels[1] = prepared(
        2,
        solved_before=SolvedSnapshot(frozenset({1, 2})),
        disposition=LevelDisposition.RECOVERED_EXISTING,
    )
    runner = AutoRunner(runtime)

    with pytest.raises(ProgressDesynchronizedError, match="consecutive puzzles"):
        runner.run(levels=1)

    assert event_names(runtime) == ["window", "prepare", "advance"]
    assert runner.completed_levels == 0
    assert runner.recovered_levels == 1
    assert runner.state == RunnerState.FAILED


def test_levels_target_counts_only_newly_solved_levels() -> None:
    runtime = FakeRuntime([True, True], [True, True])
    solved = SolvedSnapshot(frozenset({1, 3}))
    runtime.levels = [
        prepared(1, solved_before=solved, disposition=LevelDisposition.RECOVERED_EXISTING),
        prepared(2),
        prepared(3, solved_before=solved, disposition=LevelDisposition.RECOVERED_EXISTING),
        prepared(4),
        prepared(5),
    ]
    runner = AutoRunner(runtime)

    assert runner.run(levels=2) == 2

    assert [payload for name, payload in runtime.events if name == "reset"] == [2, 4]
    assert event_names(runtime).count("advance") == 4
    assert runner.completed_levels == 2
    assert runner.recovered_levels == 2


def test_fake_advance_returns_changed_level_then_none_at_the_end() -> None:
    runtime = FakeRuntime([], [])

    changed = runtime.advance_level(runtime.levels[0], None, 1.5)
    assert changed is runtime.levels[1]

    runtime.level_index = len(runtime.levels) - 1
    assert runtime.advance_level(runtime.levels[-1], None, 1.5) is None


def test_desktop_advance_retries_only_after_stable_same_level(monkeypatch) -> None:
    game_definition = definition(1, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((game_definition, definition(2, "VERY_EASY"))),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    level = prepared(1)
    level.match.definition = game_definition
    same = prepared(1, disposition=LevelDisposition.RECOVERED_EXISTING)
    same.match.definition = game_definition
    changed = prepared(2)
    changed.match.definition = definition(2, "VERY_EASY")
    captures = iter((same, same, same, same, changed))
    clicks: list[int] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "_wait", lambda duration: clock.__setitem__(0, clock[0] + duration))
    monkeypatch.setattr(runtime, "_capture_prepared_once", lambda _window: next(captures))
    monkeypatch.setattr(runtime, "_ui_click", lambda *_args, **_kwargs: clicks.append(1))
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1}), (), (1,)),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_visual_difficulty",
        lambda *_args, **_kwargs: (False, None),
    )

    result = runtime.advance_level(level, None, 1.5)

    assert result is changed
    assert len(clicks) == 2
    assert runtime.advance_retries == 1


def test_desktop_advance_stops_after_three_stable_failed_clicks(monkeypatch) -> None:
    game_definition = definition(1, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((game_definition, definition(2, "VERY_EASY"))),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    level = prepared(1)
    level.match.definition = game_definition
    same = prepared(1, disposition=LevelDisposition.RECOVERED_EXISTING)
    same.match.definition = game_definition
    clicks: list[int] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "_wait", lambda duration: clock.__setitem__(0, clock[0] + duration))
    monkeypatch.setattr(runtime, "_capture_prepared_once", lambda _window: same)
    monkeypatch.setattr(runtime, "_ui_click", lambda *_args, **_kwargs: clicks.append(1))
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1}), (), (1,)),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_visual_difficulty",
        lambda *_args, **_kwargs: (False, None),
    )

    with pytest.raises(LevelDidNotAdvanceError, match="clicked 3 time"):
        runtime.advance_level(level, None, 1.5)

    assert len(clicks) == 3
    assert runtime.advance_retries == 2


def test_desktop_advance_never_clicks_without_exact_progress_delta(monkeypatch) -> None:
    game_definition = definition(1, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((game_definition, definition(2, "VERY_EASY"))),
        driver=RecordingDriver(),
    )
    level = prepared(1)
    level.match.definition = game_definition
    clicks: list[int] = []
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({2}), (), (2,)),
    )
    monkeypatch.setattr(runtime, "_ui_click", lambda *_args, **_kwargs: clicks.append(1))

    with pytest.raises(ProgressDesynchronizedError, match="unexpected_added"):
        runtime.advance_level(level, None, 1.5)

    assert clicks == []


def test_legacy_change_timeout_controls_the_advance_deadline(monkeypatch) -> None:
    game_definition = definition(1, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((game_definition, definition(2, "EASY"))),
        driver=RecordingDriver(),
        change_timeout=7.0,
    )
    level = prepared(1)
    level.match.definition = game_definition
    target = prepared(2)
    deadlines: list[float] = []

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: 10.0)
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1}), (), (1,)),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_visual_difficulty",
        lambda *_args, deadline, **_kwargs: deadlines.append(deadline) or (False, None),
    )
    monkeypatch.setattr(
        runtime,
        "_click_next_and_confirm",
        lambda _level, _difficulty, _delay, deadline: deadlines.append(deadline) or target,
    )

    assert runtime.advance_level(level, None, 1.5) is target
    assert deadlines == [17.0, 17.0]


def test_visual_completion_switches_to_the_next_visible_incomplete_difficulty(
    monkeypatch,
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "EASY"))),
        driver=RecordingDriver(),
    )
    image, centers = rendered_difficulty_percentages(
        ("100.0%",) + ("20.0%",) * 9
    )
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    selections: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    monkeypatch.setattr(
        runtime,
        "_click_difficulty_button",
        lambda _window, _region, _centers, difficulty, *, deadline: selections.append(
            difficulty
        ),
    )

    complete, target = runtime._handle_visual_difficulty(
        prepared().window,
        "VERY_EASY",
        stop_after_current=False,
        wait_for_percentage_update=False,
        deadline=30.0,
    )

    assert complete
    assert target == "EASY"
    assert selections == ["EASY"]


def test_visual_completion_waits_for_the_percentage_to_refresh(monkeypatch) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "EASY"))),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    initial, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    refreshed, _ = rendered_difficulty_percentages(
        ("100.0%",) + ("20.0%",) * 9
    )
    region = ScreenRegion(100, 100, initial.shape[1], initial.shape[0])
    selections: list[str] = []
    clock = [0.0]
    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, initial, centers),
    )
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, refreshed))
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: centers,
    )
    monkeypatch.setattr(
        runtime,
        "_click_difficulty_button",
        lambda _window, _region, _centers, difficulty, *, deadline: selections.append(
            difficulty
        ),
    )

    result = runtime._handle_visual_difficulty(
        prepared().window,
        "VERY_EASY",
        stop_after_current=False,
        wait_for_percentage_update=True,
        deadline=30.0,
    )

    assert result == (True, "EASY")
    assert selections == ["EASY"]
    assert clock[0] == pytest.approx(0.35)


def test_visual_completion_stops_refresh_wait_after_two_seconds_when_still_incomplete(
    monkeypatch,
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "EASY"))),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    closed: list[bool] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: centers,
    )
    monkeypatch.setattr(
        runtime,
        "_close_difficulty_menu",
        lambda *_args, **_kwargs: closed.append(True),
    )

    result = runtime._handle_visual_difficulty(
        prepared().window,
        "VERY_EASY",
        stop_after_current=False,
        wait_for_percentage_update=True,
        deadline=30.0,
    )

    assert result == (False, None)
    assert closed == [True]
    assert clock[0] == pytest.approx(2.0)


def test_visual_completion_refuses_to_switch_when_menu_is_below_100(
    monkeypatch,
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "EASY"))),
        driver=RecordingDriver(),
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    closed: list[bool] = []
    monkeypatch.setattr(
        runtime,
        "_close_difficulty_menu",
        lambda *_args, **_kwargs: closed.append(True),
    )

    result = runtime._handle_visual_difficulty(
        prepared().window,
        "VERY_EASY",
        stop_after_current=False,
        wait_for_percentage_update=False,
        deadline=30.0,
    )

    assert result == (False, None)
    assert closed == [True]


@pytest.mark.parametrize(
    ("resolved_states", "expected_result", "expected_selection", "should_close"),
    [
        (
            (True, False) + (None,) * 8,
            (True, "EASY"),
            ["EASY"],
            False,
        ),
        (
            (False,) + (None,) * 9,
            (False, None),
            [],
            True,
        ),
    ],
    ids=["unreadable-then-full", "unreadable-then-incomplete"],
)
def test_visual_completion_retries_an_unreadable_frame_then_uses_clear_result(
    monkeypatch,
    resolved_states: tuple[bool | None, ...],
    expected_result: tuple[bool, str | None],
    expected_selection: list[str],
    should_close: bool,
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "EASY"))),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    unreadable = (None,) * 10
    readings = iter((unreadable, resolved_states))
    selections: list[str] = []
    closed: list[bool] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: centers,
    )

    def visible_completion(_image, _centers):
        states = next(readings)
        counts = tuple(
            3 if state is True else 2 if state is False else None for state in states
        )
        return states, counts

    monkeypatch.setattr(runtime, "_visible_difficulty_completion", visible_completion)
    monkeypatch.setattr(
        runtime,
        "_click_difficulty_button",
        lambda _window, _region, _centers, difficulty, *, deadline: selections.append(
            difficulty
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_close_difficulty_menu",
        lambda *_args, **_kwargs: closed.append(True),
    )

    result = runtime._handle_visual_difficulty(
        prepared().window,
        "VERY_EASY",
        stop_after_current=False,
        wait_for_percentage_update=False,
        deadline=30.0,
    )

    assert result == expected_result
    assert selections == expected_selection
    assert closed == ([True] if should_close else [])
    assert clock[0] == pytest.approx(0.35)


def test_advance_uses_native_next_after_persistent_percentage_ambiguity(
    monkeypatch,
) -> None:
    very_easy = definition(1, "VERY_EASY")
    easy = definition(2, "EASY")
    runtime = DesktopRuntime(
        GameLibrary((very_easy, easy)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    level = prepared(1)
    level.match.definition = very_easy
    target = prepared(2)
    target.match.definition = easy
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    closed: list[bool] = []
    native_next_calls: list[bool] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1}), (), (1,)),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: centers,
    )
    monkeypatch.setattr(
        runtime,
        "_visible_difficulty_completion",
        lambda _image, _centers: ((None,) * 10, (None,) * 10),
    )
    monkeypatch.setattr(
        runtime,
        "_close_difficulty_menu",
        lambda *_args, **_kwargs: closed.append(True),
    )
    monkeypatch.setattr(
        runtime,
        "_click_difficulty_button",
        lambda *_args, **_kwargs: pytest.fail("an unreadable percentage must not switch"),
    )
    monkeypatch.setattr(
        runtime,
        "_click_next_and_confirm",
        lambda *_args, **_kwargs: native_next_calls.append(True) or target,
    )

    result = runtime.advance_level(level, None, 1.5)

    assert result is target
    assert closed == [True]
    assert native_next_calls == [True]
    assert clock[0] == pytest.approx(2.0)


def test_expired_outer_deadline_does_not_prevent_menu_close_confirmation(
    monkeypatch,
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"),)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    clicks: list[tuple[int, int]] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    monkeypatch.setattr(
        runtime,
        "_visible_difficulty_completion",
        lambda _image, _centers: ((None,) * 10, (None,) * 10),
    )
    monkeypatch.setattr(
        runtime,
        "_ui_click",
        lambda _window, _region, point: clicks.append(point),
    )
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: (),
    )

    result = runtime._handle_visual_difficulty(
        prepared().window,
        "VERY_EASY",
        stop_after_current=False,
        wait_for_percentage_update=False,
        deadline=0.0,
    )

    assert result == (False, None)
    assert clicks == [(304, 221)]
    assert clock[0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("moved", "expected_error"),
    [(False, DifficultyMenuError), (True, TargetWindowError)],
    ids=["menu-disappeared", "window-moved"],
)
def test_percentage_retry_keeps_menu_and_window_failures_fatal(
    monkeypatch,
    moved: bool,
    expected_error: type[Exception],
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"),)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    captured_region = (
        ScreenRegion(region.left + 1, region.top, region.width, region.height)
        if moved
        else region
    )
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_difficulty_menu",
        lambda _window, *, deadline: (region, image, centers),
    )
    monkeypatch.setattr(
        runtime,
        "_visible_difficulty_completion",
        lambda _image, _centers: ((None,) * 10, (None,) * 10),
    )
    monkeypatch.setattr(
        runtime,
        "_capture_client",
        lambda _window: (captured_region, image),
    )
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: (),
    )

    with pytest.raises(expected_error):
        runtime._handle_visual_difficulty(
            prepared().window,
            "VERY_EASY",
            stop_after_current=False,
            wait_for_percentage_update=False,
            deadline=30.0,
        )


@pytest.mark.parametrize(
    ("moved", "expected_error"),
    [(False, DifficultyMenuError), (True, TargetWindowError)],
    ids=["menu-stayed-open", "window-moved"],
)
def test_close_difficulty_menu_keeps_confirmation_failures_fatal(
    monkeypatch,
    moved: bool,
    expected_error: type[Exception],
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"),)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    captured_region = (
        ScreenRegion(region.left + 1, region.top, region.width, region.height)
        if moved
        else region
    )
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(runtime, "_ui_click", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_capture_client",
        lambda _window: (captured_region, image),
    )
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: centers,
    )

    with pytest.raises(expected_error):
        runtime._close_difficulty_menu(prepared().window, region, centers)


def test_close_difficulty_menu_clicks_the_visible_back_button(monkeypatch) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"),)),
        driver=RecordingDriver(),
    )
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        runtime,
        "_ui_click",
        lambda _window, _region, point: clicks.append(point),
    )
    monkeypatch.setattr(runtime, "_wait", lambda _duration: None)
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: (),
    )

    runtime._close_difficulty_menu(
        prepared().window,
        region,
        centers,
    )

    assert clicks == [(304, 221)]


def test_advance_uses_visual_completion_path_instead_of_next(monkeypatch) -> None:
    very_easy = definition(1, "VERY_EASY")
    easy = definition(2, "EASY")
    runtime = DesktopRuntime(GameLibrary((very_easy, easy)), driver=RecordingDriver())
    level = prepared(1)
    level.match.definition = very_easy
    target = prepared(2)
    target.match.definition = easy
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1}), (), (1,)),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_visual_difficulty",
        lambda _window, _difficulty, **_kwargs: (
            calls.append("visual") or True,
            "EASY",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_wait_for_selected_difficulty",
        lambda _level, _difficulty, _deadline: target,
    )
    monkeypatch.setattr(
        runtime,
        "_click_next_and_confirm",
        lambda *_args, **_kwargs: pytest.fail("Next must not be clicked at visual 100%"),
    )

    result = runtime.advance_level(level, None, 1.5)

    assert result is target
    assert calls == ["visual"]


@pytest.mark.parametrize(
    "ordered_numbers",
    [(1, 2), ()],
    ids=["ordered-delta", "set-fallback"],
)
def test_victory_confirmation_rejects_delta_with_an_extra_puzzle(
    monkeypatch,
    ordered_numbers: tuple[int, ...],
) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "VERY_EASY"))),
        driver=RecordingDriver(),
    )
    level = prepared(1)
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1, 2}), (), ordered_numbers),
    )

    with pytest.raises(ProgressDesynchronizedError):
        runtime.victory_confirmed(level)


def test_victory_confirmation_rejects_removed_progress(monkeypatch) -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(1, "VERY_EASY"), definition(2, "VERY_EASY"))),
        driver=RecordingDriver(),
    )
    level = prepared(1, solved_before=SolvedSnapshot(frozenset({2})))
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1}), (), (1,)),
    )

    with pytest.raises(ProgressDesynchronizedError, match="removed=\\[2\\]"):
        runtime.victory_confirmed(level)


def test_desktop_advance_rejects_delta_with_current_and_extra_puzzle(
    monkeypatch,
) -> None:
    game_definitions = (
        definition(1, "VERY_EASY"),
        definition(2, "VERY_EASY"),
    )
    runtime = DesktopRuntime(GameLibrary(game_definitions), driver=RecordingDriver())
    runtime._requested_difficulty = "VERY_EASY"
    level = prepared(1)
    level.match.definition = game_definitions[0]
    clicks: list[int] = []
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({1, 2}), (), (1, 2)),
    )
    monkeypatch.setattr(runtime, "_ui_click", lambda *_args, **_kwargs: clicks.append(1))

    with pytest.raises(ProgressDesynchronizedError):
        runtime.advance_level(level, None, 1.5)

    assert clicks == []


def test_desktop_advance_does_not_retry_while_same_puzzle_image_changes(
    monkeypatch,
) -> None:
    first_definition = definition(1, "VERY_EASY")
    second_definition = definition(2, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((first_definition, second_definition)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    level = prepared(1)
    level.match.definition = first_definition

    transition_frames = []
    for value in (20, 60, 100, 140, 180):
        frame = prepared(1)
        frame.match.definition = first_definition
        frame.client_image[:] = value
        transition_frames.append(frame)
    changed = prepared(2)
    changed.match.definition = second_definition
    captures = iter((*transition_frames, changed))
    click_times: list[float] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "_wait", lambda duration: clock.__setitem__(0, clock[0] + duration))
    monkeypatch.setattr(runtime, "_capture_prepared_once", lambda _window: next(captures))
    monkeypatch.setattr(runtime, "_ui_click", lambda *_args, **_kwargs: click_times.append(clock[0]))

    result = runtime._click_next_and_confirm(level, None, 1.5, deadline=10.0)

    assert result is changed
    assert click_times == [0.0]


def test_next_glow_pulse_detects_local_temporal_change() -> None:
    still = np.zeros((400, 500, 3), dtype=np.uint8)
    pulse = still.copy()
    cv2.circle(pulse, (488, 385), 42, (255, 255, 255), 18)
    frames = [still, pulse, still, pulse]

    pulse_ratio, outside_ratio = next_glow_pulse_score(frames)

    assert pulse_ratio >= 0.03
    assert outside_ratio == 0.0
    assert next_glow_pulse_visible(frames)


def test_next_glow_pulse_rejects_whole_screen_transition() -> None:
    dark = np.zeros((400, 500, 3), dtype=np.uint8)
    bright = np.full_like(dark, 255)
    frames = [dark, bright, dark]

    pulse_ratio, outside_ratio = next_glow_pulse_score(frames)

    assert pulse_ratio == 1.0
    assert outside_ratio == 1.0
    assert not next_glow_pulse_visible(frames)


def test_frame_stability_allows_sparse_animation_but_rejects_transition() -> None:
    still = np.zeros((400, 500, 3), dtype=np.uint8)
    sparse_animation = still.copy()
    sparse_animation[20:60, 20:120] = 20
    transition = np.full_like(still, 40)

    assert client_frames_stable(still, sparse_animation)
    assert not client_frames_stable(still, transition)


def test_prepare_level_closes_persistently_ambiguous_startup_menu_and_resumes(
    monkeypatch,
) -> None:
    game_definition = definition(1, "VERY_EASY")
    runtime = DesktopRuntime(
        GameLibrary((game_definition,)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    ready = prepared(1)
    ready.match.definition = game_definition
    image, centers = rendered_difficulty_percentages(("20.0%",) * 10)
    region = ScreenRegion(100, 100, image.shape[1], image.shape[0])
    located = SimpleNamespace(region=ScreenRegion(20, 20, 300, 200))
    menu_open = [True]
    closed: list[bool] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runtime,
        "_wait",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )
    monkeypatch.setattr(runtime, "_ensure_running", lambda: None)
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(runtime, "solved_snapshot", SolvedSnapshot)

    def identify_board(_image):
        if menu_open[0]:
            raise DifficultyMenuError("difficulty menu covers the board")
        return located, ready.match

    monkeypatch.setattr(runtime, "_identify_board", identify_board)
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.difficulty_button_centers",
        lambda _image: centers if menu_open[0] else (),
    )
    monkeypatch.setattr(
        runtime,
        "_visible_difficulty_completion",
        lambda _image, _centers: ((None,) * 10, (None,) * 10),
    )

    def close_menu(*_args, **_kwargs):
        closed.append(True)
        menu_open[0] = False

    monkeypatch.setattr(runtime, "_close_difficulty_menu", close_menu)
    monkeypatch.setattr(
        runtime,
        "_handle_visual_difficulty",
        lambda *_args, **_kwargs: (False, None),
    )

    result = runtime.prepare_level(ready.window, None)

    assert result.number == 1
    assert closed == [True]
    assert clock[0] == pytest.approx(2.0)


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
        "_observe_next_glow_pulse",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(frozenset({3, 7}), (), (3, 7)),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_visual_difficulty",
        lambda *_args, **_kwargs: (False, None),
    )

    level = runtime.prepare_level(window, "VERY_EASY")

    assert level.number == 7
    assert level.client_region == region
    assert level.board_region == region
    assert level.match.puzzle.islands == []
    assert level.solved_before.ordered_numbers == (3, 7)
    assert level.disposition is LevelDisposition.RECOVERED_UNREADABLE


@pytest.mark.parametrize(
    ("requested", "solved_numbers"),
    [("EASY", frozenset()), (None, frozenset({1}))],
    ids=["requested-difficulty", "completed-current-difficulty"],
)
def test_prepare_level_waits_for_selected_difficulty_without_reclicking(
    monkeypatch,
    requested: str | None,
    solved_numbers: frozenset[int],
) -> None:
    very_easy = definition(1, "VERY_EASY")
    easy = definition(2, "EASY")
    runtime = DesktopRuntime(
        GameLibrary((very_easy, easy)),
        driver=RecordingDriver(),
        poll_interval=0.35,
    )
    old_level = prepared(1)
    old_level.match.definition = very_easy
    target_level = prepared(2)
    target_level.match.definition = easy
    matches = iter((old_level.match, old_level.match, target_level.match))
    region = ScreenRegion(100, 100, 500, 400)
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    located = SimpleNamespace(region=ScreenRegion(20, 20, 300, 200))
    selections: list[str] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "_wait", lambda duration: clock.__setitem__(0, clock[0] + duration))
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(runtime, "_identify_board", lambda _image: (located, next(matches)))
    monkeypatch.setattr(
        runtime,
        "solved_snapshot",
        lambda: SolvedSnapshot(solved_numbers),
    )
    monkeypatch.setattr(
        runtime,
        "_select_difficulty",
        lambda _window, difficulty, *, deadline: selections.append(difficulty),
    )
    def visual_difficulty(_window, current, **_kwargs):
        if current == "VERY_EASY":
            selections.append("EASY")
            return True, "EASY"
        return False, None

    monkeypatch.setattr(runtime, "_handle_visual_difficulty", visual_difficulty)

    result = runtime.prepare_level(old_level.window, requested)

    assert result.number == 2
    assert selections == ["EASY"]
    assert clock[0] == pytest.approx(0.35)


@pytest.mark.parametrize(
    "timeout_kwargs",
    [{"change_timeout": 1.5}, {"advance_timeout": 1.5}],
    ids=["legacy-change-timeout", "advance-timeout"],
)
def test_prepare_level_uses_configured_deadline_while_old_difficulty_remains(
    monkeypatch,
    timeout_kwargs: dict[str, float],
) -> None:
    very_easy = definition(1, "VERY_EASY")
    easy = definition(2, "EASY")
    runtime = DesktopRuntime(
        GameLibrary((very_easy, easy)),
        driver=RecordingDriver(),
        poll_interval=0.35,
        **timeout_kwargs,
    )
    old_level = prepared(1)
    old_level.match.definition = very_easy
    region = ScreenRegion(100, 100, 500, 400)
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    located = SimpleNamespace(region=ScreenRegion(20, 20, 300, 200))
    selections: list[str] = []
    clock = [0.0]

    monkeypatch.setattr("bridge_assistant.auto_runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "_wait", lambda duration: clock.__setitem__(0, clock[0] + duration))
    monkeypatch.setattr(runtime, "_capture_client", lambda _window: (region, image))
    monkeypatch.setattr(runtime, "_identify_board", lambda _image: (located, old_level.match))
    monkeypatch.setattr(runtime, "solved_snapshot", SolvedSnapshot)
    monkeypatch.setattr(
        runtime,
        "_select_difficulty",
        lambda _window, difficulty, *, deadline: selections.append(difficulty),
    )

    with pytest.raises(DifficultyMismatchError, match="did not become ready"):
        runtime.prepare_level(old_level.window, "EASY")

    assert selections == ["EASY"]
    assert clock[0] > runtime.advance_timeout


def test_victory_recovery_requires_confirmed_pulse_and_uses_last_saved_puzzle() -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(7, "VERY_EASY"),)),
        driver=RecordingDriver(),
    )
    window = prepared().window
    region = ScreenRegion(100, 100, 500, 400)
    snapshot = SolvedSnapshot(frozenset({7}), (), (7,))
    image = np.full((400, 500, 3), 255, dtype=np.uint8)

    assert runtime._prepare_victory_recovery(
        window,
        region,
        image,
        snapshot,
        pulse_confirmed=False,
    ) is None

    recovery = runtime._prepare_victory_recovery(
        window,
        region,
        image,
        snapshot,
        pulse_confirmed=True,
    )
    assert recovery is not None
    assert recovery.disposition is LevelDisposition.RECOVERED_UNREADABLE


def test_terminal_completion_does_not_wait_for_another_level() -> None:
    runtime = TerminalRuntime([True], [True])
    runner = AutoRunner(runtime)

    assert runner.run(levels=1) == 1

    assert event_names(runtime)[-2:] == ["victory", "advance"]
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

    assert event_names(runtime)[-2:] == ["victory", "advance"]
    assert runner.state == RunnerState.COMPLETE


def test_desktop_runtime_rejects_unknown_difficulty_before_capture() -> None:
    runtime = DesktopRuntime(
        GameLibrary((definition(7, "VERY_EASY"),)),
        driver=RecordingDriver(),
    )

    with pytest.raises(Exception, match="Unknown BridgeFix difficulty"):
        runtime.prepare_level(prepared().window, "not-a-difficulty")


def test_full_client_recovery_discards_low_confidence_ui_circles(monkeypatch) -> None:
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    visual_islands = [
        Island(0, 60, 80, 2, 12, 0.95),
        Island(1, 180, 80, 2, 12, 0.94),
        Island(2, 300, 80, 2, 12, 0.93),
        Island(3, 20, 20, 6, 8, 0.10),
    ]

    class RecoveryLibrary:
        puzzles = (
            SimpleNamespace(
                nodes=(None, None, None),
                number=7,
                difficulty="VERY_EASY",
            ),
        )

        def match(self, recovered: Puzzle):
            assert len(recovered.islands) == 3
            assert [island.value for island in recovered.islands] == [2, 2, 2]
            return SimpleNamespace(number=7, difficulty="TEST")

    runtime = DesktopRuntime(RecoveryLibrary(), driver=RecordingDriver())
    monkeypatch.setattr(
        "bridge_assistant.auto_runner.recognize",
        lambda _image: Recognition(visual_islands, image),
    )

    located, match = runtime._recover_full_client_board(image)

    assert match.number == 7
    assert len(located.puzzle.islands) == 3
    assert located.recognition.warnings == [
        "Ignored 1 low-confidence full-client UI circles"
    ]


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
