from __future__ import annotations

import math
import threading

import pytest

from bridge_assistant.automation import DEFAULT_MAX_RETRIES, build_mouse_actions, execute_mouse_actions
from bridge_assistant.graph import build_puzzle
from bridge_assistant.model import Edge, InputMode, Island, MouseInputError, Puzzle, ScreenRegion, Solution


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []
        self.FAILSAFE = False
        self.PAUSE = 0

    def moveTo(self, *args, **kwargs) -> None:
        self.calls.append(("moveTo", args, kwargs))

    def dragTo(self, *args, **kwargs) -> None:
        self.calls.append(("dragTo", args, kwargs))

    def mouseDown(self, *args, **kwargs) -> None:
        self.calls.append(("mouseDown", args, kwargs))

    def mouseUp(self, *args, **kwargs) -> None:
        self.calls.append(("mouseUp", args, kwargs))

    def click(self, *args, **kwargs) -> None:
        self.calls.append(("click", args, kwargs))


def fixture_puzzle():
    return build_puzzle([Island(0, 10, 20, 2), Island(1, 70, 20, 2)])


def test_absolute_coordinate_mapping() -> None:
    puzzle = fixture_puzzle()
    solution = Solution({0: 2})
    actions = build_mouse_actions(puzzle, solution, ScreenRegion(-300, 100, 500, 300), InputMode.DRAG)
    assert actions[0].start == (-290, 120)
    assert actions[0].end == (-230, 120)
    assert actions[0].bridge_count == 2


def test_near_endpoint_click_is_the_default_and_outside_the_island() -> None:
    puzzle = fixture_puzzle()
    region = ScreenRegion(100, 200, 100, 100)
    actions = build_mouse_actions(puzzle, Solution({0: 1}), region)
    action = actions[0]
    local_click = (action.start[0] - region.left, action.start[1] - region.top)
    assert action.mode == InputMode.NEAR_ENDPOINT
    assert action.start == action.end
    assert math.dist(local_click, puzzle.islands[0].point) > puzzle.islands[0].radius
    assert puzzle.islands[0].x < local_click[0] < (puzzle.islands[0].x + puzzle.islands[1].x) / 2
    assert local_click[1] == round(puzzle.islands[0].y)


def test_near_endpoint_click_supports_vertical_edges() -> None:
    puzzle = build_puzzle([Island(0, 30, 10, 1, 12), Island(1, 30, 90, 1, 12)])
    action = build_mouse_actions(
        puzzle,
        Solution({0: 1}),
        ScreenRegion(-50, 300, 100, 100),
    )[0]
    assert action.start[0] == -20
    assert 310 + puzzle.islands[0].radius < action.start[1] < 350


def test_near_endpoint_click_avoids_other_candidate_lines() -> None:
    islands = [
        Island(0, 10, 50, 1, 10),
        Island(1, 150, 50, 1, 10),
        Island(2, 27, 10, 1, 10),
        Island(3, 27, 90, 1, 10),
    ]
    unobstructed = Puzzle(islands[:2], [Edge(0, 0, 1, "h")])
    preferred = build_mouse_actions(
        unobstructed,
        Solution({0: 1}),
        ScreenRegion(0, 0, 180, 120),
    )[0]
    assert preferred.start == (25, 50)

    puzzle = Puzzle(
        islands,
        [Edge(0, 0, 1, "h"), Edge(1, 2, 3, "v")],
    )
    action = build_mouse_actions(
        puzzle,
        Solution({0: 1, 1: 0}),
        ScreenRegion(0, 0, 180, 120),
    )[0]
    assert abs(action.start[0] - 27) > 3
    assert min(abs(action.start[0] - 10), abs(action.start[0] - 150)) < 0.42 * 140
    assert action.start[1] == 50


def test_near_endpoint_click_does_not_land_on_an_unrelated_island() -> None:
    puzzle = Puzzle(
        [
            Island(0, 10, 50, 1, 10),
            Island(1, 150, 50, 1, 10),
            Island(2, 25, 50, 1, 8),
        ],
        [Edge(0, 0, 1, "h")],
    )
    action = build_mouse_actions(
        puzzle,
        Solution({0: 1}),
        ScreenRegion(0, 0, 180, 120),
    )[0]
    assert math.dist(action.start, puzzle.islands[2].point) > puzzle.islands[2].radius
    assert min(abs(action.start[0] - 10), abs(action.start[0] - 150)) < 0.42 * 140


def test_near_endpoint_click_falls_back_to_midpoint_when_no_safe_point_exists() -> None:
    puzzle = Puzzle(
        [
            Island(0, 10, 30, 3, 14),
            Island(1, 50, 30, 3, 14),
        ],
        [Edge(0, 0, 1, "h")],
    )
    action = build_mouse_actions(
        puzzle,
        Solution({0: 1}),
        ScreenRegion(100, 200, 100, 100),
    )[0]
    assert action.mode == InputMode.NEAR_ENDPOINT
    assert action.start == action.end == (130, 230)
    assert action.observation_start == (110, 230)
    assert action.observation_end == (150, 230)


def test_midpoint_coordinate_mapping() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 1}), ScreenRegion(100, 200, 500, 300), InputMode.MIDPOINT)
    assert actions[0].start == actions[0].end == (140, 220)


def test_double_bridge_near_endpoint_clicks_the_same_point_twice() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 2}), ScreenRegion(0, 0, 100, 100))
    driver = FakeDriver()
    execute_mouse_actions(actions, interval=0, driver=driver, escape_check=lambda: False)
    assert [call[0] for call in driver.calls] == ["click", "click"]
    assert driver.calls[0][1] == driver.calls[1][1] == actions[0].start


def test_double_bridge_midpoint_fallback_clicks_the_same_point_twice() -> None:
    puzzle = Puzzle(
        [
            Island(0, 20, 10, 3, 14),
            Island(1, 20, 50, 3, 14),
        ],
        [Edge(0, 0, 1, "v")],
    )
    actions = build_mouse_actions(
        puzzle,
        Solution({0: 2}),
        ScreenRegion(0, 0, 100, 100),
    )
    driver = FakeDriver()
    execute_mouse_actions(actions, interval=0, driver=driver, escape_check=lambda: False)
    assert actions[0].start == (20, 30)
    assert [call[0] for call in driver.calls] == ["click", "click"]
    assert driver.calls[0][1] == driver.calls[1][1] == actions[0].start


def test_double_bridge_executes_twice() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 2}), ScreenRegion(0, 0, 100, 100), InputMode.DRAG)
    driver = FakeDriver()
    completed = execute_mouse_actions(
        actions,
        interval=0,
        drag_duration=0,
        gesture_pause=0,
        driver=driver,
        escape_check=lambda: False,
    )
    assert completed == 2
    assert [call[0] for call in driver.calls] == [
        "moveTo",
        "mouseDown",
        "moveTo",
        "mouseUp",
        "moveTo",
        "mouseDown",
        "moveTo",
        "mouseUp",
    ]


def test_endpoint_mode_clicks_each_endpoint() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 1}), ScreenRegion(0, 0, 100, 100), InputMode.ENDPOINTS)
    driver = FakeDriver()
    execute_mouse_actions(actions, interval=0, driver=driver, escape_check=lambda: False)
    assert [call[1] for call in driver.calls] == [((10, 20)), ((70, 20))]


def test_stop_event_prevents_mouse_input() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 1}), ScreenRegion(0, 0, 100, 100), InputMode.DRAG)
    driver = FakeDriver()
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(InterruptedError):
        execute_mouse_actions(actions, interval=0, driver=driver, stop_event=stopped, escape_check=lambda: False)
    assert driver.calls == []


def test_unconfirmed_operation_is_retried_before_progress() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 1}), ScreenRegion(0, 0, 100, 100), InputMode.DRAG)
    driver = FakeDriver()
    checks = iter((False, True))
    retries: list[int] = []
    progress: list[tuple[int, int]] = []
    completed = execute_mouse_actions(
        actions,
        interval=0,
        drag_duration=0,
        gesture_pause=0,
        driver=driver,
        escape_check=lambda: False,
        operation_confirmed=lambda _action: next(checks),
        prepare_retry=lambda _action, attempt: retries.append(attempt),
        max_retries=1,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert completed == 1
    assert retries == [1]
    assert progress == [(1, 1)]
    assert [call[0] for call in driver.calls].count("mouseDown") == 2


def test_unconfirmed_operation_stops_instead_of_advancing() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 1}), ScreenRegion(0, 0, 100, 100), InputMode.DRAG)
    driver = FakeDriver()
    with pytest.raises(MouseInputError):
        execute_mouse_actions(
            actions,
            interval=0,
            drag_duration=0,
            gesture_pause=0,
            driver=driver,
            escape_check=lambda: False,
            operation_confirmed=lambda _action: False,
            max_retries=1,
        )
    assert [call[0] for call in driver.calls].count("mouseDown") == 2


def test_default_retry_limit_stops_without_an_additional_click() -> None:
    puzzle = fixture_puzzle()
    actions = build_mouse_actions(puzzle, Solution({0: 1}), ScreenRegion(0, 0, 100, 100))
    driver = FakeDriver()
    retries: list[int] = []
    with pytest.raises(MouseInputError):
        execute_mouse_actions(
            actions,
            interval=0,
            driver=driver,
            escape_check=lambda: False,
            operation_confirmed=lambda _action: False,
            prepare_retry=lambda _action, attempt: retries.append(attempt),
        )
    assert DEFAULT_MAX_RETRIES == 0
    assert retries == []
    assert [call[0] for call in driver.calls].count("click") == 1
