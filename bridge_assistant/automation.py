from __future__ import annotations

import ctypes
import math
import threading
import time
from collections.abc import Callable, Iterable
from statistics import median

import pyautogui

from .model import Edge, InputMode, Island, MouseAction, MouseInputError, Puzzle, ScreenRegion, Solution


DEFAULT_MAX_RETRIES = 0


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-9:
        return math.dist(point, start)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(point, closest)


def _safe_near_endpoint_point(
    puzzle: Puzzle,
    edge: Edge,
    by_id: dict[int, Island],
) -> tuple[int, int]:
    """Pick a click on this edge, just outside a safe endpoint's outline."""

    first, second = by_id[edge.a], by_id[edge.b]
    typical_radius = median(max(1.0, island.radius) for island in puzzle.islands)
    other_edges = [candidate for candidate in puzzle.edges if candidate.id != edge.id]
    line_clearance = max(3.0, typical_radius * 0.22)
    candidates: list[tuple[float, int, tuple[int, int]]] = []

    for endpoint_order, (origin, target) in enumerate(((first, second), (second, first))):
        dx, dy = target.x - origin.x, target.y - origin.y
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        unit_x, unit_y = dx / length, dy / length
        safety_gap = max(5.0, origin.radius * 0.35)
        preferred_offset = origin.radius + safety_gap
        far_clearance = max(2.0, target.radius * 0.12)
        maximum_offset = min(length * 0.42, length - target.radius - far_clearance)
        if preferred_offset > maximum_offset:
            continue
        step = max(2.0, origin.radius * 0.14)
        offset = preferred_offset
        while offset <= maximum_offset + 1e-6:
            point = (
                int(round(origin.x + unit_x * offset)),
                int(round(origin.y + unit_y * offset)),
            )
            projection = (point[0] - origin.x) * unit_x + (point[1] - origin.y) * unit_y
            if not 0.0 < projection < length:
                offset += step
                continue
            if any(
                math.dist(point, island.point) <= island.radius + 1.5
                for island in puzzle.islands
            ):
                offset += step
                continue
            overlaps_another_line = False
            for candidate in other_edges:
                line_start = by_id[candidate.a].point
                line_end = by_id[candidate.b].point
                if _point_segment_distance(point, line_start, line_end) <= line_clearance:
                    overlaps_another_line = True
                    break
            if not overlaps_another_line:
                relative_gap = (projection - origin.radius) / max(1.0, origin.radius)
                candidates.append((relative_gap, endpoint_order, point))
                break
            offset += step

    if not candidates:
        raise MouseInputError(
            f"候选连接 {edge.a + 1}-{edge.b + 1} 附近没有避开岛屿和其他连线的安全点击位置"
        )
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def build_mouse_actions(
    puzzle: Puzzle,
    solution: Solution,
    region: ScreenRegion,
    mode: InputMode = InputMode.NEAR_ENDPOINT,
) -> list[MouseAction]:
    by_id = {island.id: island for island in puzzle.islands}
    actions: list[MouseAction] = []
    for edge in puzzle.edges:
        count = solution.bridges.get(edge.id, 0)
        if count <= 0:
            continue
        first, second = by_id[edge.a], by_id[edge.b]
        first_point = (region.left + round(first.x), region.top + round(first.y))
        second_point = (region.left + round(second.x), region.top + round(second.y))
        if mode == InputMode.NEAR_ENDPOINT:
            try:
                local_click = _safe_near_endpoint_point(puzzle, edge, by_id)
            except MouseInputError:
                local_click = (
                    round((first.x + second.x) / 2),
                    round((first.y + second.y) / 2),
                )
            click = (region.left + local_click[0], region.top + local_click[1])
            actions.append(
                MouseAction(
                    mode,
                    click,
                    click,
                    count,
                    first_point,
                    second_point,
                    min(first.radius, second.radius),
                )
            )
        elif mode == InputMode.MIDPOINT:
            midpoint = (
                round((first_point[0] + second_point[0]) / 2),
                round((first_point[1] + second_point[1]) / 2),
            )
            actions.append(
                MouseAction(
                    mode,
                    midpoint,
                    midpoint,
                    count,
                    first_point,
                    second_point,
                    min(first.radius, second.radius),
                )
            )
        else:
            actions.append(
                MouseAction(
                    mode,
                    first_point,
                    second_point,
                    count,
                    first_point,
                    second_point,
                    min(first.radius, second.radius),
                )
            )
    return actions


def escape_is_pressed() -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)


def execute_mouse_actions(
    actions: Iterable[MouseAction],
    *,
    interval: float = 0.18,
    drag_duration: float = 0.24,
    gesture_pause: float = 0.07,
    stop_event: threading.Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    operation_confirmed: Callable[[MouseAction], bool] | None = None,
    prepare_retry: Callable[[MouseAction, int], None] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verification_cursor: tuple[int, int] | None = None,
    driver=pyautogui,
    escape_check: Callable[[], bool] = escape_is_pressed,
) -> int:
    actions = list(actions)
    operation_count = sum(action.bridge_count for action in actions)
    completed = 0
    driver.FAILSAFE = True
    driver.PAUSE = 0.03

    def ensure_running() -> None:
        if (stop_event is not None and stop_event.is_set()) or escape_check():
            raise InterruptedError("Mouse operation cancelled")

    for action in actions:
        for _ in range(action.bridge_count):
            attempt = 0
            while True:
                ensure_running()
                if action.mode == InputMode.DRAG:
                    driver.moveTo(*action.start, duration=min(0.12, drag_duration / 2))
                    time.sleep(max(0.0, gesture_pause))
                    ensure_running()
                    driver.mouseDown(button="left")
                    try:
                        driver.moveTo(*action.end, duration=drag_duration)
                        time.sleep(max(0.0, gesture_pause))
                    finally:
                        driver.mouseUp(button="left")
                elif action.mode in (InputMode.NEAR_ENDPOINT, InputMode.MIDPOINT):
                    driver.click(*action.start, button="left")
                elif action.mode == InputMode.ENDPOINTS:
                    driver.click(*action.start, button="left")
                    ensure_running()
                    driver.click(*action.end, button="left")
                else:
                    raise ValueError(f"Unsupported input mode: {action.mode}")
                ensure_running()
                if operation_confirmed is not None:
                    parking_point = verification_cursor or action.observation_start
                    if parking_point is not None:
                        driver.moveTo(*parking_point, duration=min(0.08, drag_duration / 2))
                if operation_confirmed is None or operation_confirmed(action):
                    break
                if attempt >= max(0, max_retries):
                    raise MouseInputError(
                        "目标应用未显示本次鼠标操作产生的桥；为防止后续输入错位，执行已停止"
                    )
                attempt += 1
                if prepare_retry is not None:
                    prepare_retry(action, attempt)
            completed += 1
            if progress is not None:
                progress(completed, operation_count)
            ensure_running()
            time.sleep(max(0.0, interval))
    return completed
