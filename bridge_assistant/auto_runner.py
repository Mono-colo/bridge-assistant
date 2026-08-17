from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pyautogui

from .automation import build_mouse_actions, escape_is_pressed
from .board_locator import _crop_region, locate_board
from .capture import enable_dpi_awareness, grab_screen, pil_to_bgr
from .diagnostics import RunDiagnostics
from .game_library import (
    LIBRARY_ASSET_NAMES,
    GameLibrary,
    GamePuzzleMatchError,
    GamePuzzleMatch,
    load_game_library,
)
from .graph import build_puzzle
from .instance_lock import AnotherInstanceError, SingleInstanceLock
from .model import (
    AppWindow,
    BridgeAssistantError,
    Island,
    LocatedBoard,
    MouseAction,
    Puzzle,
    Recognition,
    ScreenRegion,
    Solution,
    TargetWindowError,
)
from .scene import bridge_corridor_change
from .vision import recognize
from .windowing import (
    activate_window,
    list_app_windows,
    validate_window,
    window_at_screen_point,
    window_client_region,
)


TARGET_TITLE = "BridgeFix 231"
TARGET_PROCESS = "Bridge-Windows64.exe"
DEFAULT_LEVEL_RETRIES = 3
DEFAULT_NEXT_DELAY = 1.5
DEFAULT_ADVANCE_TIMEOUT = 30.0
DEFAULT_ADVANCE_CLICKS = 3
MAX_FULL_CLIENT_OUTLIERS = 8
DIFFICULTY_PERCENTAGE_SETTLE_TIMEOUT = 2.0
DIFFICULTIES = tuple(name.removeprefix("library") for name in LIBRARY_ASSET_NAMES)
LOGGER = logging.getLogger("bridge_assistant.auto_runner")


class AutoRunnerError(BridgeAssistantError):
    pass


class MagicLinkSolutionError(AutoRunnerError):
    pass


class InputConfirmationError(AutoRunnerError):
    pass


class VictoryNotConfirmedError(AutoRunnerError):
    pass


class LevelDidNotAdvanceError(AutoRunnerError):
    pass


class DifficultyMismatchError(AutoRunnerError):
    pass


class DifficultyMenuError(AutoRunnerError):
    pass


class AutomaticRunComplete(AutoRunnerError):
    pass


class ProgressFileError(AutoRunnerError):
    pass


class ProgressDesynchronizedError(AutoRunnerError):
    pass


class RunnerState(str, Enum):
    IDLE = "idle"
    WAITING_FOR_WINDOW = "waiting_for_window"
    CAPTURING = "capturing"
    RESETTING = "resetting"
    INPUTTING = "inputting"
    VERIFYING_VICTORY = "verifying_victory"
    RECOVERING = "recovering"
    RETRYING = "retrying"
    ADVANCING = "advancing"
    WAITING_FOR_CHANGE = "waiting_for_change"
    COMPLETE = "complete"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class SolvedSnapshot:
    numbers: frozenset[int] = frozenset()
    files: tuple[tuple[str, int, int], ...] = ()
    ordered_numbers: tuple[int, ...] = ()


class LevelDisposition(str, Enum):
    UNSOLVED = "unsolved"
    RECOVERED_EXISTING = "recovered_existing"
    RECOVERED_UNREADABLE = "recovered_unreadable"


class DifficultyMenuAction(str, Enum):
    RETRY = "retry"
    CONTINUE_CURRENT = "continue_current"
    SWITCH = "switch"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class DifficultyMenuDecision:
    action: DifficultyMenuAction
    target: str | None = None


@dataclass(slots=True)
class PreparedLevel:
    window: AppWindow
    client_region: ScreenRegion
    board_region: ScreenRegion
    match: GamePuzzleMatch
    client_image: np.ndarray
    solved_before: SolvedSnapshot = field(default_factory=SolvedSnapshot)
    disposition: LevelDisposition = LevelDisposition.UNSOLVED

    @property
    def number(self) -> int:
        return self.match.number

    @property
    def signature(self) -> tuple[object, ...]:
        definition = self.match.definition
        return (
            definition.number,
            definition.width,
            definition.height,
            tuple((node.x, node.y, node.value) for node in definition.nodes),
        )


class RunnerRuntime(Protocol):
    def wait_for_window(self, stop_event: threading.Event) -> AppWindow: ...

    def prepare_level(self, window: AppWindow, difficulty: str | None) -> PreparedLevel: ...

    def reset(self, level: PreparedLevel) -> None: ...

    def clear_edge(self, level: PreparedLevel, action: MouseAction) -> None: ...

    def add_bridge(self, level: PreparedLevel, action: MouseAction) -> bool: ...

    def victory_confirmed(self, level: PreparedLevel) -> bool: ...

    def advance_level(
        self,
        level: PreparedLevel,
        difficulty: str | None,
        minimum_delay: float,
    ) -> PreparedLevel | None: ...


def _normalized_difficulty(value: str) -> str:
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def is_target_window(window: AppWindow) -> bool:
    return (
        window.title.casefold() == TARGET_TITLE.casefold()
        and Path(window.process_name).name.casefold() == TARGET_PROCESS.casefold()
    )


def representative_solution(puzzle: Puzzle, solution: Solution) -> Solution:
    """Collapse transitive MagicLink groups after proving their answers agree."""

    edge_ids = {edge.id for edge in puzzle.edges}
    parent = {edge_id: edge_id for edge_id in edge_ids}

    def find(edge_id: int) -> int:
        while parent[edge_id] != edge_id:
            parent[edge_id] = parent[parent[edge_id]]
            edge_id = parent[edge_id]
        return edge_id

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for link in puzzle.magic_links:
        if link.edge_a not in edge_ids or link.edge_b not in edge_ids:
            raise MagicLinkSolutionError("MagicLink references an unknown edge")
        union(link.edge_a, link.edge_b)

    groups: dict[int, list[int]] = {}
    for edge_id in edge_ids:
        groups.setdefault(find(edge_id), []).append(edge_id)

    representatives: dict[int, int] = {}
    for group in groups.values():
        counts = {solution.bridges.get(edge_id, 0) for edge_id in group}
        if any(count not in (0, 1, 2) for count in counts):
            raise MagicLinkSolutionError("Bridge answers must be 0, 1, or 2")
        if len(counts) != 1:
            joined = ", ".join(str(edge_id) for edge_id in sorted(group))
            raise MagicLinkSolutionError(
                f"MagicLink group has inconsistent answers for edges {joined}"
            )
        representatives[min(group)] = counts.pop()
    return Solution(representatives)


class AutoRunner:
    def __init__(
        self,
        runtime: RunnerRuntime,
        *,
        level_retries: int = DEFAULT_LEVEL_RETRIES,
        next_delay: float = DEFAULT_NEXT_DELAY,
        state_changed: Callable[[RunnerState], None] | None = None,
        level_changed: Callable[[PreparedLevel, int], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        if level_retries < 1:
            raise ValueError("level_retries must be at least one")
        if next_delay < DEFAULT_NEXT_DELAY:
            raise ValueError(f"next_delay must be at least {DEFAULT_NEXT_DELAY}")
        self.runtime = runtime
        self.level_retries = level_retries
        self.next_delay = next_delay
        self.state_changed = state_changed
        self.level_changed = level_changed
        self.stop_event = stop_event or threading.Event()
        self.state = RunnerState.IDLE
        self.completed_levels = 0
        self.recovered_levels = 0
        self.solve_retries = 0

    def _transition(self, state: RunnerState) -> None:
        self.state = state
        LOGGER.info("state=%s", state.value)
        if self.state_changed is not None:
            self.state_changed(state)

    def _ensure_running(self) -> None:
        if self.stop_event.is_set() or escape_is_pressed():
            self.stop_event.set()
            self._transition(RunnerState.STOPPED)
            raise InterruptedError("Automatic runner stopped")

    def _log_summary(self, outcome: str) -> None:
        LOGGER.info(
            "summary outcome=%s newly_solved=%s recovered=%s solve_retries=%s "
            "advance_retries=%s difficulty_switches=%s recognition_failures=%s",
            outcome,
            self.completed_levels,
            self.recovered_levels,
            self.solve_retries,
            getattr(self.runtime, "advance_retries", 0),
            getattr(self.runtime, "difficulty_switches", 0),
            getattr(self.runtime, "recognition_failures", 0),
        )

    def _complete(self) -> int:
        self._log_summary("completed")
        self._transition(RunnerState.COMPLETE)
        return self.completed_levels

    def _input_actions(self, level: PreparedLevel) -> list[MouseAction]:
        solution = representative_solution(level.match.puzzle, level.match.official_solution)
        return build_mouse_actions(level.match.puzzle, solution, level.board_region)

    def _attempt_level(self, level: PreparedLevel) -> None:
        if level.disposition is not LevelDisposition.UNSOLVED:
            raise ProgressDesynchronizedError(
                f"Recovered puzzle {level.number} must not be replayed as a new puzzle"
            )
        actions = self._input_actions(level)
        already_reset = False
        last_error: AutoRunnerError | None = None
        for attempt in range(1, self.level_retries + 1):
            self._ensure_running()
            self._transition(RunnerState.RESETTING)
            if not already_reset:
                self.runtime.reset(level)
            already_reset = False

            self._transition(RunnerState.INPUTTING)
            try:
                for action in actions:
                    self._ensure_running()
                    self.runtime.clear_edge(level, action)
                    for _ in range(action.bridge_count):
                        if not self.runtime.add_bridge(level, action):
                            raise InputConfirmationError(
                                f"Bridge input was not confirmed on edge {action.start}"
                            )

                self._transition(RunnerState.VERIFYING_VICTORY)
                if not self.runtime.victory_confirmed(level):
                    raise VictoryNotConfirmedError(
                        f"Victory was not confirmed for puzzle {level.number}"
                    )
                return
            except (InputConfirmationError, VictoryNotConfirmedError) as error:
                last_error = error
                self.solve_retries += 1
                self._transition(RunnerState.RETRYING)
                self.runtime.reset(level)
                already_reset = True
                if attempt == self.level_retries:
                    break

        assert last_error is not None
        raise last_error

    def run(self, *, levels: int | None = None, difficulty: str | None = None) -> int:
        if levels is not None and levels < 1:
            raise ValueError("levels must be at least one")
        selected_difficulty = _normalized_difficulty(difficulty) if difficulty else None
        try:
            self._transition(RunnerState.WAITING_FOR_WINDOW)
            window = self.runtime.wait_for_window(self.stop_event)
            current: PreparedLevel | None = None
            existing_streak = 0
            while levels is None or self.completed_levels < levels:
                self._ensure_running()
                if current is None:
                    self._transition(RunnerState.CAPTURING)
                    current = self.runtime.prepare_level(window, selected_difficulty)

                LOGGER.info(
                    "puzzle=%s difficulty=%s disposition=%s completed=%s target=%s",
                    current.number,
                    current.match.difficulty,
                    current.disposition.value,
                    self.completed_levels,
                    levels if levels is not None else "all",
                )
                if self.level_changed is not None:
                    self.level_changed(current, self.completed_levels)
                if current.disposition is LevelDisposition.UNSOLVED:
                    self._attempt_level(current)
                    self.completed_levels += 1
                    if self.level_changed is not None:
                        self.level_changed(current, self.completed_levels)
                    existing_streak = 0
                else:
                    existing_streak += 1
                    if existing_streak > 1:
                        raise ProgressDesynchronizedError(
                            "BridgeFix returned consecutive puzzles that are already present "
                            f"in solved.dat; the game must be restarted (latest puzzle {current.number})"
                        )
                    self.recovered_levels += 1
                    self._transition(RunnerState.RECOVERING)
                self._transition(RunnerState.ADVANCING)
                self._transition(RunnerState.WAITING_FOR_CHANGE)
                current = self.runtime.advance_level(
                    current,
                    selected_difficulty,
                    self.next_delay,
                )
                if current is None:
                    return self._complete()
                if self.level_changed is not None:
                    self.level_changed(current, self.completed_levels)

            return self._complete()
        except InterruptedError:
            self._log_summary("stopped")
            if self.state != RunnerState.STOPPED:
                self._transition(RunnerState.STOPPED)
            raise
        except AutomaticRunComplete:
            return self._complete()
        except Exception:
            self._log_summary("failed")
            self._transition(RunnerState.FAILED)
            raise


@dataclass(frozen=True, slots=True)
class ControlLayout:
    # BridgeFix anchors these controls to the client edges.
    reset_x: float = 0.929
    reset_y: float = 0.035
    menu_x: float = 0.976
    menu_y: float = 0.035
    next_x: float = 0.976
    next_y: float = 0.965

    @staticmethod
    def _point(region: ScreenRegion, x_ratio: float, y_ratio: float) -> tuple[int, int]:
        return (
            region.left + round((region.width - 1) * x_ratio),
            region.top + round((region.height - 1) * y_ratio),
        )

    def reset_point(self, region: ScreenRegion) -> tuple[int, int]:
        return self._point(region, self.reset_x, self.reset_y)

    def menu_point(self, region: ScreenRegion) -> tuple[int, int]:
        return self._point(region, self.menu_x, self.menu_y)

    def next_point(self, region: ScreenRegion) -> tuple[int, int]:
        return self._point(region, self.next_x, self.next_y)


def difficulty_button_centers(image: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return the BridgeFix 2x5 difficulty buttons in reading order."""

    if image is None or image.size == 0:
        return ()
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    edges = cv2.Canny(gray, 20, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (
            width * 0.07 <= box_width <= width * 0.22
            and height * 0.10 <= box_height <= height * 0.28
            and 0.78 <= box_width / max(1, box_height) <= 1.22
            and width * 0.12 <= x <= width * 0.88
            and height * 0.28 <= y <= height * 0.78
        ):
            continue
        fill = cv2.contourArea(contour) / max(1.0, float(box_width * box_height))
        if fill >= 0.90:
            candidates.append((x, y, box_width, box_height))

    deduplicated: list[tuple[int, int, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: item[2] * item[3], reverse=True):
        center = (candidate[0] + candidate[2] // 2, candidate[1] + candidate[3] // 2)
        if any(
            abs(center[0] - (item[0] + item[2] // 2)) < max(6, candidate[2] * 0.08)
            and abs(center[1] - (item[1] + item[3] // 2)) < max(6, candidate[3] * 0.08)
            for item in deduplicated
        ):
            continue
        deduplicated.append(candidate)

    centers = [
        (x + box_width // 2, y + box_height // 2)
        for x, y, box_width, box_height in deduplicated
    ]
    if len(deduplicated) != 10:
        typical_height = float(np.median([item[3] for item in deduplicated])) if deduplicated else 0.0
        rows: list[list[tuple[int, int]]] = []
        for point in sorted(centers, key=lambda item: item[1]):
            if not rows or abs(point[1] - float(np.mean([item[1] for item in rows[-1]]))) > typical_height * 0.25:
                rows.append([point])
            else:
                rows[-1].append(point)
        complete = next((sorted(row) for row in rows if len(row) == 5), None)
        if complete is None:
            partial = sorted(
                (sorted(row) for row in rows if len(row) == 4),
                key=lambda row: float(np.mean([point[1] for point in row])),
            )
            if len(partial) != 2:
                return ()
            typical_width = float(np.median([item[2] for item in deduplicated]))
            if any(
                abs(first[0] - second[0]) > typical_width * 0.20
                for first, second in zip(partial[0], partial[1])
            ):
                return ()
            all_gaps = np.asarray(
                [
                    second[0] - first[0]
                    for row in partial
                    for first, second in zip(row, row[1:])
                ],
                dtype=float,
            )
            gap = float(np.median(all_gaps))
            if gap <= 0 or np.max(np.abs(all_gaps - gap)) > gap * 0.15:
                return ()
            candidate_xs = (
                round(float(np.mean([row[0][0] for row in partial])) - gap),
                round(float(np.mean([row[-1][0] for row in partial])) + gap),
            )
            half_width = max(4, round(typical_width * 0.35))
            half_height = max(4, round(typical_height * 0.35))
            scored: list[tuple[float, int]] = []
            for candidate_x in candidate_xs:
                if not 0 <= candidate_x < width:
                    continue
                scores = []
                for row in partial:
                    row_y = round(float(np.mean([point[1] for point in row])))
                    patch = gray[
                        max(0, row_y - half_height) : min(height, row_y + half_height + 1),
                        max(0, candidate_x - half_width) : min(width, candidate_x + half_width + 1),
                    ]
                    scores.append(float(np.mean(patch >= 170)) if patch.size else 0.0)
                scored.append((float(np.mean(scores)), candidate_x))
            if not scored:
                return ()
            missing_score, missing_x = max(scored)
            if missing_score < 0.45:
                return ()
            completed_rows = [
                sorted(row + [(missing_x, round(float(np.mean([point[1] for point in row]))))])
                for row in partial
            ]
            return tuple(completed_rows[0] + completed_rows[1])
        gaps = np.diff([point[0] for point in complete])
        if np.min(gaps) <= 0 or np.max(gaps) - np.min(gaps) > float(np.median(gaps)) * 0.15:
            return ()
        row_y = round(float(np.mean([point[1] for point in complete])))
        other_y = round(row_y + typical_height * (1.09 if row_y < height / 2 else -1.09))
        inferred = [(point[0], other_y) for point in complete]
        return tuple((complete + inferred) if row_y < height / 2 else (inferred + complete))
    centers.sort(key=lambda point: point[1])
    first_row = sorted(centers[:5])
    second_row = sorted(centers[5:])
    typical_height = float(np.median([item[3] for item in deduplicated]))
    if (
        max(point[1] for point in first_row) - min(point[1] for point in first_row)
        > typical_height * 0.2
        or max(point[1] for point in second_row) - min(point[1] for point in second_row)
        > typical_height * 0.2
        or min(point[1] for point in second_row) - max(point[1] for point in first_row)
        < typical_height * 0.45
    ):
        return ()
    return tuple(first_row + second_row)


def difficulty_percentage_digit_counts(
    image: np.ndarray,
    centers: Sequence[tuple[int, int]],
) -> tuple[int | None, ...]:
    """Count complete integer glyphs before each percentage label's decimal point."""

    if image is None or image.size == 0 or len(centers) != len(DIFFICULTIES):
        return ()
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return ()

    first_row = centers[:5]
    second_row = centers[5:]
    column_gaps = [
        second[0] - first[0]
        for row in (first_row, second_row)
        for first, second in zip(row, row[1:])
    ]
    row_gaps = [second[1] - first[1] for first, second in zip(first_row, second_row)]
    if not column_gaps or min(column_gaps) <= 0 or min(row_gaps) <= 0:
        return ()
    column_step = float(np.median(column_gaps))
    row_step = float(np.median(row_gaps))
    if (
        max(abs(gap - column_step) for gap in column_gaps) > column_step * 0.15
        or max(abs(gap - row_step) for gap in row_gaps) > row_step * 0.15
    ):
        return ()

    height, width = gray.shape
    minimum_area = max(1, int(np.ceil(column_step * row_step * 0.00008)))
    counts: list[int | None] = []
    for center_x, center_y in centers:
        left = max(0, round(center_x - column_step * 0.30))
        right = min(width, round(center_x + column_step * 0.44))
        top = max(0, round(center_y + row_step * 0.20))
        bottom = min(height, round(center_y + row_step * 0.47))
        patch = gray[top:bottom, left:right]
        if (
            patch.size == 0
            or patch.shape[1] < column_step * 0.65
            or patch.shape[0] < row_step * 0.20
        ):
            counts.append(None)
            continue

        threshold, _ = cv2.threshold(
            patch,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ink = (patch < float(np.clip(threshold, 80, 180))).astype(np.uint8)
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ink,
            8,
        )
        components: list[dict[str, float | int]] = []
        for label_id in range(1, component_count):
            local_x, local_y, item_width, item_height, area = (
                int(value) for value in stats[label_id]
            )
            if area < minimum_area:
                continue
            components.append(
                {
                    "label": label_id,
                    "x": left + local_x,
                    "y": top + local_y,
                    "width": item_width,
                    "height": item_height,
                    "area": area,
                    "center_x": left + float(centroids[label_id][0]),
                    "center_y": top + float(centroids[label_id][1]),
                    "local_x": local_x,
                    "local_y": local_y,
                }
            )

        def _right(component: dict[str, float | int]) -> float:
            return float(component["x"]) + float(component["width"])

        def _holes(component: dict[str, float | int]) -> int:
            local_x = int(component["local_x"])
            local_y = int(component["local_y"])
            item_width = int(component["width"])
            item_height = int(component["height"])
            label_id = int(component["label"])
            glyph = np.where(
                labels[
                    local_y : local_y + item_height,
                    local_x : local_x + item_width,
                ]
                == label_id,
                255,
                0,
            ).astype(np.uint8)
            contours, hierarchy = cv2.findContours(
                glyph,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if hierarchy is None:
                return 0
            minimum_hole_area = item_width * item_height * 0.10
            return sum(
                1
                for contour, relation in zip(contours, hierarchy[0])
                if relation[3] >= 0 and cv2.contourArea(contour) >= minimum_hole_area
            )

        tall = [
            component
            for component in components
            if row_step * 0.09 <= float(component["height"]) <= row_step * 0.19
            and column_step * 0.018
            <= float(component["width"])
            <= column_step * 0.13
            and center_y + row_step * 0.20
            <= float(component["y"])
            <= center_y + row_step * 0.34
            and center_y + row_step * 0.35
            <= float(component["y"]) + float(component["height"])
            <= center_y + row_step * 0.46
            and float(component["area"])
            / (float(component["width"]) * float(component["height"]))
            >= 0.15
        ]
        dots = [
            component
            for component in components
            if float(component["width"]) <= column_step * 0.04
            and float(component["height"]) <= row_step * 0.04
            and center_x + column_step * 0.06
            <= float(component["center_x"])
            <= center_x + column_step * 0.24
            and center_y + row_step * 0.35
            <= float(component["center_y"])
            <= center_y + row_step * 0.46
        ]
        readings: list[int] = []
        for dot in dots:
            fractional_candidates = sorted(
                (
                    component
                    for component in tall
                    if 0
                    <= float(component["x"]) - _right(dot)
                    <= column_step * 0.075
                    and float(component["x"]) < center_x + column_step * 0.30
                ),
                key=lambda component: float(component["x"]),
            )
            if not fractional_candidates:
                continue
            fractional = fractional_candidates[0]
            percent_slashes = [
                component
                for component in components
                if center_x + column_step * 0.27
                <= float(component["center_x"])
                <= center_x + column_step * 0.40
                and float(component["x"]) >= _right(fractional)
                and row_step * 0.09
                <= float(component["height"])
                <= row_step * 0.20
                and column_step * 0.018
                <= float(component["width"])
                <= column_step * 0.13
                and center_y + row_step * 0.20
                <= float(component["y"])
                <= center_y + row_step * 0.34
                and center_y + row_step * 0.35
                <= float(component["y"]) + float(component["height"])
                <= center_y + row_step * 0.47
                and float(component["area"])
                / (float(component["width"]) * float(component["height"]))
                >= 0.08
            ]
            if not percent_slashes:
                continue

            integer_components = [
                component
                for component in tall
                if _right(component) <= float(dot["x"]) + column_step * 0.01
                and float(component["center_x"]) >= center_x - column_step * 0.30
            ]
            chain: list[dict[str, float | int]] = []
            cursor = float(dot["x"])
            for component in sorted(
                integer_components,
                key=lambda item: float(item["x"]),
                reverse=True,
            ):
                gap = cursor - _right(component)
                if gap < -column_step * 0.01:
                    continue
                if gap > column_step * 0.075:
                    break
                chain.append(component)
                cursor = float(component["x"])
            if len(chain) not in (1, 2, 3) or len(chain) != len(integer_components):
                continue

            if len(chain) == 3:
                ordered = list(reversed(chain))
                widths = [float(component["width"]) for component in ordered]
                zero_widths = widths[1:] + [float(fractional["width"])]
                zero_width_ratio = max(zero_widths) / min(zero_widths)
                first_width_ratio = widths[0] / float(np.median(widths[1:]))
                if (
                    [_holes(component) for component in ordered] != [0, 1, 1]
                    or _holes(fractional) != 1
                    or not 0.30 <= first_width_ratio <= 0.75
                    or zero_width_ratio > 1.35
                ):
                    continue
            readings.append(len(chain))

        if readings and all(reading == readings[0] for reading in readings):
            counts.append(readings[0])
        else:
            counts.append(None)

    return tuple(counts)


def difficulty_completion_states(
    image: np.ndarray,
    centers: Sequence[tuple[int, int]],
) -> tuple[bool | None, ...]:
    """Identify exact 100.0% labels from their integer glyph structure."""

    counts = difficulty_percentage_digit_counts(image, centers)
    if not counts:
        return ()
    return tuple(
        True if count == 3 else False if count in (1, 2) else None
        for count in counts
    )


def next_glow_pulse_score(images: Sequence[np.ndarray]) -> tuple[float, float]:
    """Measure temporal motion around Next while rejecting whole-screen transitions."""

    usable = [image for image in images if image is not None and image.size]
    if len(usable) < 3 or any(image.shape != usable[0].shape for image in usable[1:]):
        return 0.0, 1.0
    gray_frames = []
    for image in usable:
        if image.ndim == 2:
            gray_frames.append(image)
        elif image.shape[2] == 4:
            gray_frames.append(cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY))
        else:
            gray_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    stack = np.stack(gray_frames).astype(np.int16)
    temporal_range = np.max(stack, axis=0) - np.min(stack, axis=0)
    height, width = temporal_range.shape
    scale = float(min(height, width))
    layout = ControlLayout()
    center_x = round((width - 1) * layout.next_x)
    center_y = round((height - 1) * layout.next_y)
    yy, xx = np.ogrid[:height, :width]
    radius_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
    inner = max(12.0, scale * 0.04)
    outer = max(inner + 8.0, scale * 0.20)
    pulse_mask = (radius_squared >= inner**2) & (radius_squared <= outer**2)
    outside_mask = radius_squared >= (outer * 1.20) ** 2
    changed = temporal_range >= 18
    pulse_ratio = float(np.mean(changed[pulse_mask])) if np.any(pulse_mask) else 0.0
    outside_ratio = (
        float(np.mean(changed[outside_mask])) if np.any(outside_mask) else 1.0
    )
    return pulse_ratio, outside_ratio


def next_glow_pulse_visible(images: Sequence[np.ndarray]) -> bool:
    pulse_ratio, outside_ratio = next_glow_pulse_score(images)
    return pulse_ratio >= 0.03 and pulse_ratio >= max(0.03, outside_ratio * 3.0)


def client_frame_change_score(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    """Measure client changes outside the animated bottom-right Next glow."""

    if (
        first is None
        or second is None
        or first.size == 0
        or second.size == 0
        or first.shape != second.shape
    ):
        return 1.0, 255.0

    def grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            return np.empty((0, 0), dtype=np.uint8)
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, conversion)

    first_gray = grayscale(first)
    second_gray = grayscale(second)
    if first_gray.size == 0 or second_gray.size == 0:
        return 1.0, 255.0

    height, width = first_gray.shape
    layout = ControlLayout()
    center_x = round((width - 1) * layout.next_x)
    center_y = round((height - 1) * layout.next_y)
    radius = max(16.0, min(height, width) * 0.24)
    yy, xx = np.ogrid[:height, :width]
    stable_area = (xx - center_x) ** 2 + (yy - center_y) ** 2 >= radius**2
    if not np.any(stable_area):
        return 1.0, 255.0

    difference = np.abs(first_gray.astype(np.int16) - second_gray.astype(np.int16))
    changed_ratio = float(np.mean(difference[stable_area] >= 8))
    mean_delta = float(np.mean(difference[stable_area]))
    return changed_ratio, mean_delta


def client_frames_stable(first: np.ndarray, second: np.ndarray) -> bool:
    changed_ratio, mean_delta = client_frame_change_score(first, second)
    return changed_ratio <= 0.03 and mean_delta <= 0.75


class DesktopRuntime:
    def __init__(
        self,
        library: GameLibrary,
        *,
        driver=pyautogui,
        controls: ControlLayout = ControlLayout(),
        poll_interval: float = 0.35,
        victory_timeout: float = 5.0,
        change_timeout: float | None = None,
        advance_timeout: float | None = None,
        advance_clicks: int = DEFAULT_ADVANCE_CLICKS,
    ) -> None:
        if advance_clicks < 1:
            raise ValueError("advance_clicks must be at least one")
        self.library = library
        self.driver = driver
        self.controls = controls
        self.poll_interval = max(0.05, poll_interval)
        self.victory_timeout = max(0.0, victory_timeout)
        configured_timeout = (
            advance_timeout if advance_timeout is not None else change_timeout
        )
        if configured_timeout is None:
            configured_timeout = DEFAULT_ADVANCE_TIMEOUT
        effective_timeout = max(DEFAULT_NEXT_DELAY, configured_timeout)
        self.change_timeout = effective_timeout
        self.advance_timeout = effective_timeout
        self.advance_clicks = advance_clicks
        self.driver.FAILSAFE = True
        self.driver.PAUSE = 0.03
        self._edge_baseline: np.ndarray | None = None
        self._solved_paths: tuple[Path, ...] = ()
        self._requested_difficulty: str | None = None
        self._stop_event = threading.Event()
        self.advance_retries = 0
        self.difficulty_switches = 0
        self.recognition_failures = 0
        self._progress_logged = False
        self._window_region_logged = False
        self._numbers_by_difficulty = {
            difficulty: frozenset(
                puzzle.number
                for puzzle in library.puzzles
                if _normalized_difficulty(puzzle.difficulty) == difficulty
            )
            for difficulty in DIFFICULTIES
        }
        self._all_numbers = frozenset(
            puzzle.number for puzzle in library.puzzles
        )

    @staticmethod
    def _find_solved_files() -> tuple[Path, ...]:
        candidates: set[Path] = set()
        bridgefix = (
            Path.home()
            / "AppData"
            / "LocalLow"
            / "MirkoStudio"
            / TARGET_TITLE
            / "solved.dat"
        )
        if bridgefix.is_file():
            candidates.add(bridgefix.resolve())
        return tuple(sorted(candidates, key=str))

    def _percentage_refresh_expected(
        self,
        difficulty: str,
        snapshot: SolvedSnapshot,
    ) -> bool:
        # This only grants the menu time to refresh; it never authorizes a switch.
        numbers = self._numbers_by_difficulty.get(_normalized_difficulty(difficulty))
        return bool(numbers) and numbers <= snapshot.numbers

    def _capture_client(self, window: AppWindow) -> tuple[ScreenRegion, np.ndarray]:
        self._ensure_running()
        activate_window(window)
        region = window_client_region(window)
        if not self._window_region_logged:
            LOGGER.info(
                "game_window handle=%s pid=%s region=%s,%s,%s,%s",
                window.handle,
                window.process_id,
                region.left,
                region.top,
                region.width,
                region.height,
            )
            self._window_region_logged = True
        return region, pil_to_bgr(grab_screen(region))

    def _ensure_running(self) -> None:
        if self._stop_event.is_set() or escape_is_pressed():
            self._stop_event.set()
            raise InterruptedError("Automatic runner stopped")

    def _wait(self, duration: float) -> None:
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            self._ensure_running()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _same_window(first: AppWindow | None, second: AppWindow) -> bool:
        return bool(
            first is not None
            and first.handle == second.handle
            and first.process_id == second.process_id
        )

    def _verify_click_target(
        self,
        window: AppWindow,
        expected_region: ScreenRegion,
        point: tuple[int, int],
    ) -> None:
        validate_window(window)
        current_region = window_client_region(window)
        if current_region != expected_region:
            raise TargetWindowError(
                "BridgeFix window moved, resized, or changed DPI; recapture before clicking"
            )
        owner = window_at_screen_point(*point, exclude_process_id=-1)
        if not self._same_window(owner, window):
            raise TargetWindowError(
                "BridgeFix is not the visible window at the intended click position"
            )

    def _ui_click(
        self,
        window: AppWindow,
        expected_region: ScreenRegion,
        point: tuple[int, int],
    ) -> None:
        self._ensure_running()
        activate_window(window)
        self._verify_click_target(window, expected_region, point)
        self.driver.moveTo(*point, duration=0.10)
        self._wait(0.12)
        self._verify_click_target(window, expected_region, point)
        self.driver.click(*point, button="left")

    def _level_click(
        self,
        level: PreparedLevel,
        point: tuple[int, int],
        *,
        button: str,
    ) -> None:
        self._ensure_running()
        activate_window(level.window)
        self._verify_click_target(level.window, level.client_region, point)
        self.driver.click(*point, button=button)

    def _wait_for_difficulty_menu(
        self,
        window: AppWindow,
        *,
        timeout: float = 8.0,
        deadline: float | None = None,
    ) -> tuple[ScreenRegion, np.ndarray, tuple[tuple[int, int], ...]]:
        local_deadline = time.monotonic() + timeout
        if deadline is not None:
            local_deadline = min(local_deadline, deadline)
        clicked_menu = False
        while time.monotonic() <= local_deadline:
            client_region, image = self._capture_client(window)
            centers = difficulty_button_centers(image)
            if centers:
                return client_region, image, centers
            if not clicked_menu:
                self._ui_click(
                    window,
                    client_region,
                    self.controls.menu_point(client_region),
                )
                clicked_menu = True
            self._wait(self.poll_interval)
        raise DifficultyMenuError("BridgeFix difficulty menu did not become visible")

    @staticmethod
    def _visible_difficulty_completion(
        image: np.ndarray,
        centers: Sequence[tuple[int, int]],
    ) -> tuple[tuple[bool | None, ...], tuple[int | None, ...]]:
        digit_counts = difficulty_percentage_digit_counts(image, centers)
        if len(digit_counts) != len(DIFFICULTIES):
            digit_counts = (None,) * len(DIFFICULTIES)
        states = tuple(
            True if count == 3 else False if count in (1, 2) else None
            for count in digit_counts
        )
        return states, digit_counts

    @staticmethod
    def _scan_difficulty_states(
        states: Sequence[bool | None],
        ordered_difficulties: Sequence[str],
    ) -> DifficultyMenuDecision:
        if len(states) != len(DIFFICULTIES):
            return DifficultyMenuDecision(DifficultyMenuAction.RETRY)
        for difficulty in ordered_difficulties:
            state = states[DIFFICULTIES.index(difficulty)]
            if state is None:
                return DifficultyMenuDecision(DifficultyMenuAction.RETRY)
            if not state:
                return DifficultyMenuDecision(DifficultyMenuAction.SWITCH, difficulty)
        return DifficultyMenuDecision(DifficultyMenuAction.COMPLETE)

    @classmethod
    def _difficulty_menu_decision(
        cls,
        states: Sequence[bool | None],
        current_difficulty: str,
        *,
        stop_after_current: bool,
    ) -> DifficultyMenuDecision:
        if len(states) != len(DIFFICULTIES):
            return DifficultyMenuDecision(DifficultyMenuAction.RETRY)
        current = _normalized_difficulty(current_difficulty)
        current_index = DIFFICULTIES.index(current)
        current_state = states[current_index]
        if current_state is None:
            return DifficultyMenuDecision(DifficultyMenuAction.RETRY)
        if not current_state:
            return DifficultyMenuDecision(DifficultyMenuAction.CONTINUE_CURRENT)
        if stop_after_current:
            return DifficultyMenuDecision(DifficultyMenuAction.COMPLETE)
        ordered = list(DIFFICULTIES[current_index + 1 :] + DIFFICULTIES[:current_index])
        return cls._scan_difficulty_states(states, ordered)

    @classmethod
    def _difficulty_selection_decision(
        cls,
        states: Sequence[bool | None],
        expected_difficulty: str | None,
    ) -> DifficultyMenuDecision:
        if len(states) != len(DIFFICULTIES):
            return DifficultyMenuDecision(DifficultyMenuAction.RETRY)
        if expected_difficulty is None:
            return cls._scan_difficulty_states(states, DIFFICULTIES)
        expected = _normalized_difficulty(expected_difficulty)
        state = states[DIFFICULTIES.index(expected)]
        if state is None:
            return DifficultyMenuDecision(DifficultyMenuAction.RETRY)
        if state:
            return DifficultyMenuDecision(DifficultyMenuAction.COMPLETE)
        return DifficultyMenuDecision(DifficultyMenuAction.SWITCH, expected)

    def _resolve_difficulty_menu_decision(
        self,
        window: AppWindow,
        client_region: ScreenRegion,
        image: np.ndarray,
        centers: tuple[tuple[int, int], ...],
        *,
        decide: Callable[[Sequence[bool | None]], DifficultyMenuDecision],
        wait_for_percentage_update: bool,
        deadline: float,
    ) -> tuple[
        DifficultyMenuDecision,
        tuple[tuple[int, int], ...],
        tuple[bool | None, ...],
        tuple[int | None, ...],
    ]:
        settle_deadline = min(
            deadline,
            time.monotonic() + DIFFICULTY_PERCENTAGE_SETTLE_TIMEOUT,
        )
        while True:
            states, digit_counts = self._visible_difficulty_completion(image, centers)
            decision = decide(states)
            should_wait = decision.action is DifficultyMenuAction.RETRY or (
                wait_for_percentage_update
                and decision.action is DifficultyMenuAction.CONTINUE_CURRENT
            )
            if not should_wait or time.monotonic() >= settle_deadline:
                return decision, centers, states, digit_counts
            self._wait(
                min(
                    self.poll_interval,
                    max(0.0, settle_deadline - time.monotonic()),
                )
            )
            current_region, image = self._capture_client(window)
            if current_region != client_region:
                raise TargetWindowError(
                    "BridgeFix window moved, resized, or changed DPI while reading percentages"
                )
            centers = difficulty_button_centers(image)
            if not centers:
                raise DifficultyMenuError(
                    "BridgeFix difficulty menu disappeared while percentages were read"
                )

    @staticmethod
    def _format_percentage_readings(
        digit_counts: Sequence[int | None],
    ) -> str:
        return ",".join(
            f"{difficulty}={count if count is not None else '?'}"
            for difficulty, count in zip(DIFFICULTIES, digit_counts)
        )

    def _click_difficulty_button(
        self,
        window: AppWindow,
        client_region: ScreenRegion,
        centers: Sequence[tuple[int, int]],
        difficulty: str,
        *,
        deadline: float | None,
    ) -> None:
        normalized = _normalized_difficulty(difficulty)
        try:
            index = DIFFICULTIES.index(normalized)
        except ValueError as error:
            raise DifficultyMenuError(f"Unknown BridgeFix difficulty: {difficulty}") from error
        if deadline is not None and time.monotonic() + DEFAULT_NEXT_DELAY > deadline:
            raise DifficultyMenuError(
                f"Not enough time remained to switch to difficulty {difficulty}"
            )
        local_x, local_y = centers[index]
        self._ui_click(
            window,
            client_region,
            (client_region.left + local_x, client_region.top + local_y),
        )
        self.difficulty_switches += 1
        self._wait(DEFAULT_NEXT_DELAY)

    def _close_difficulty_menu(
        self,
        window: AppWindow,
        client_region: ScreenRegion,
        centers: Sequence[tuple[int, int]],
    ) -> None:
        column_step = float(
            np.median(
                [
                    second[0] - first[0]
                    for row in (centers[:5], centers[5:])
                    for first, second in zip(row, row[1:])
                ]
            )
        )
        row_step = float(
            np.median(
                [
                    centers[index + 5][1] - centers[index][1]
                    for index in range(5)
                ]
            )
        )
        first_x, first_y = centers[0]
        back_point = (
            client_region.left + round(first_x - column_step * 0.31),
            client_region.top + round(first_y - row_step * 0.86),
        )
        self._ui_click(window, client_region, back_point)
        self._wait(max(0.50, self.poll_interval))
        close_deadline = time.monotonic() + 5.0
        while time.monotonic() <= close_deadline:
            current_region, image = self._capture_client(window)
            if current_region != client_region:
                raise TargetWindowError(
                    "BridgeFix window moved, resized, or changed DPI while closing its menu"
                )
            if not difficulty_button_centers(image):
                return
            self._wait(self.poll_interval)
        raise DifficultyMenuError("BridgeFix difficulty menu did not close")

    def _handle_visual_difficulty(
        self,
        window: AppWindow,
        current_difficulty: str,
        *,
        stop_after_current: bool,
        wait_for_percentage_update: bool,
        deadline: float,
    ) -> tuple[bool, str | None]:
        client_region, image, centers = self._wait_for_difficulty_menu(
            window,
            deadline=deadline,
        )
        current = _normalized_difficulty(current_difficulty)
        try:
            current_index = DIFFICULTIES.index(current)
        except ValueError as error:
            raise DifficultyMenuError(
                f"Unknown BridgeFix difficulty: {current_difficulty}"
            ) from error
        decision, centers, states, digit_counts = self._resolve_difficulty_menu_decision(
            window,
            client_region,
            image,
            centers,
            decide=lambda current_states: self._difficulty_menu_decision(
                current_states,
                current,
                stop_after_current=stop_after_current,
            ),
            wait_for_percentage_update=wait_for_percentage_update,
            deadline=deadline,
        )
        LOGGER.info(
            "difficulty_menu current=%s current_100=%s action=%s screen_complete=%s "
            "integer_digits=[%s]",
            current,
            states[current_index],
            decision.action.value,
            ",".join(
                name for name, complete in zip(DIFFICULTIES, states) if complete is True
            )
            or "none",
            self._format_percentage_readings(digit_counts),
        )
        if decision.action is DifficultyMenuAction.RETRY:
            LOGGER.warning(
                "difficulty_menu percentages remained unreadable; closing menu and "
                "continuing current difficulty=%s integer_digits=[%s]",
                current,
                self._format_percentage_readings(digit_counts),
            )
            self._close_difficulty_menu(window, client_region, centers)
            return False, None
        if decision.action is DifficultyMenuAction.CONTINUE_CURRENT:
            self._close_difficulty_menu(window, client_region, centers)
            return False, None
        if decision.action is DifficultyMenuAction.COMPLETE:
            self._close_difficulty_menu(window, client_region, centers)
            return True, None
        target = decision.target
        if target is None:
            raise DifficultyMenuError("Difficulty switch decision did not include a target")
        LOGGER.info("difficulty=%s screen_100 switching_to=%s", current, target)
        self._click_difficulty_button(
            window,
            client_region,
            centers,
            target,
            deadline=deadline,
        )
        return True, target

    def _select_difficulty(
        self,
        window: AppWindow,
        difficulty: str,
        *,
        deadline: float | None = None,
    ) -> None:
        client_region, _image, centers = self._wait_for_difficulty_menu(
            window,
            deadline=deadline,
        )
        self._click_difficulty_button(
            window,
            client_region,
            centers,
            difficulty,
            deadline=deadline,
        )

    def solved_snapshot(self) -> SolvedSnapshot:
        # Unity may create the persistence file only after the first win.
        discovered = set(self._solved_paths)
        discovered.update(self._find_solved_files())
        self._solved_paths = tuple(sorted(discovered, key=str))
        numbers: set[int] = set()
        ordered_numbers: list[int] = []
        files: list[tuple[str, int, int]] = []
        for path in self._solved_paths:
            data: bytes | None = None
            stable_stat = None
            previous_observation: tuple[bytes, int, int] | None = None
            missing_attempts = 0
            last_read_error: OSError | None = None
            for attempt in range(5):
                try:
                    before = path.stat()
                    candidate = path.read_bytes()
                    after = path.stat()
                except FileNotFoundError:
                    missing_attempts += 1
                    previous_observation = None
                except OSError as error:
                    last_read_error = error
                    previous_observation = None
                else:
                    if (
                        before.st_mtime_ns == after.st_mtime_ns
                        and before.st_size == after.st_size == len(candidate)
                    ):
                        observation = (
                            candidate,
                            after.st_mtime_ns,
                            after.st_size,
                        )
                        if observation == previous_observation:
                            data = candidate
                            stable_stat = after
                            break
                        previous_observation = observation
                    else:
                        previous_observation = None
                if attempt < 4:
                    self._wait(0.05)
            if data is None or stable_stat is None:
                if missing_attempts == 5:
                    raise ProgressFileError(
                        f"BridgeFix progress file disappeared: {path}"
                    )
                if last_read_error is not None and previous_observation is None:
                    raise ProgressFileError(
                        f"BridgeFix progress file could not be read: {path}: {last_read_error}"
                    ) from last_read_error
                raise ProgressFileError(
                    f"BridgeFix progress file kept changing while it was read: {path}"
                )
            try:
                text = data.decode("ascii")
            except UnicodeDecodeError as error:
                raise ProgressFileError(
                    f"BridgeFix progress file is not ASCII text: {path}"
                ) from error
            values: list[int] = []
            seen_in_file: set[int] = set()
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if not line.isdecimal():
                    raise ProgressFileError(
                        f"BridgeFix progress file has invalid line {line_number}: {line!r}"
                    )
                value = int(line)
                if value not in self._all_numbers:
                    raise ProgressFileError(
                        f"BridgeFix progress file contains unknown puzzle number {value}"
                    )
                if value in seen_in_file:
                    raise ProgressFileError(
                        f"BridgeFix progress file contains duplicate puzzle number {value}"
                    )
                seen_in_file.add(value)
                values.append(value)
            files.append((str(path), stable_stat.st_mtime_ns, len(data)))
            ordered_numbers.extend(values)
            numbers.update(values)
        return SolvedSnapshot(
            frozenset(numbers),
            tuple(files),
            tuple(ordered_numbers),
        )

    def _prepare_victory_recovery(
        self,
        window: AppWindow,
        client_region: ScreenRegion,
        client_image: np.ndarray,
        snapshot: SolvedSnapshot,
        *,
        pulse_confirmed: bool,
    ) -> PreparedLevel | None:
        if not pulse_confirmed or not snapshot.ordered_numbers:
            return None
        definition = self.library.by_number.get(snapshot.ordered_numbers[-1])
        if definition is None:
            return None
        match = GamePuzzleMatch(
            definition,
            Puzzle([]),
            Solution({}),
            {},
            {},
            {},
            {},
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        return PreparedLevel(
            window,
            client_region,
            client_region,
            match,
            client_image,
            snapshot,
            disposition=LevelDisposition.RECOVERED_UNREADABLE,
        )

    def _observe_next_glow_pulse(
        self,
        window: AppWindow,
        client_region: ScreenRegion,
        first_image: np.ndarray,
        *,
        duration: float = 1.2,
    ) -> bool:
        frames = [first_image]
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            self._wait(min(0.10, max(0.0, deadline - time.monotonic())))
            current_region, image = self._capture_client(window)
            if current_region != client_region:
                return False
            frames.append(image)
        pulse_ratio, outside_ratio = next_glow_pulse_score(frames)
        LOGGER.info(
            "next_glow_pulse pulse_ratio=%.4f outside_ratio=%.4f",
            pulse_ratio,
            outside_ratio,
        )
        return next_glow_pulse_visible(frames)

    def wait_for_window(self, stop_event: threading.Event) -> AppWindow:
        self._stop_event = stop_event
        while not stop_event.is_set():
            self._ensure_running()
            matches = [window for window in list_app_windows() if is_target_window(window)]
            if matches:
                LOGGER.info(
                    "game_found pid=%s handle=%s title=%r process=%r",
                    matches[0].process_id,
                    matches[0].handle,
                    matches[0].title,
                    matches[0].process_name,
                )
                return matches[0]
            self._wait(self.poll_interval)
        raise InterruptedError("Automatic runner stopped")

    def _recover_full_client_board(
        self,
        client_image: np.ndarray,
    ) -> tuple[LocatedBoard, GamePuzzleMatch]:
        recognition = recognize(client_image)
        raw_islands = recognition.islands
        puzzle_sizes = sorted(
            {len(definition.nodes) for definition in self.library.puzzles},
            reverse=True,
        )
        candidate_sizes = [
            size
            for size in puzzle_sizes
            if 0 <= len(raw_islands) - size <= MAX_FULL_CLIENT_OUTLIERS
        ]
        if not candidate_sizes:
            raise GamePuzzleMatchError(
                f"Full-client recognition found {len(raw_islands)} islands, "
                "but no BridgeFix puzzle size is close enough"
            )

        ranked = sorted(raw_islands, key=lambda island: island.confidence, reverse=True)
        last_error: GamePuzzleMatchError | None = None
        for target_size in candidate_sizes:
            selected_ids = {island.id for island in ranked[:target_size]}
            selected = [island for island in raw_islands if island.id in selected_ids]
            region = _crop_region(client_image, selected)
            islands = [
                Island(
                    index,
                    island.x - region.left,
                    island.y - region.top,
                    island.value,
                    island.radius,
                    island.confidence,
                )
                for index, island in enumerate(selected)
            ]
            puzzle = build_puzzle(islands)
            try:
                match = self.library.match(puzzle)
            except GamePuzzleMatchError as error:
                last_error = error
                continue
            crop = client_image[region.top : region.bottom, region.left : region.right].copy()
            removed = len(raw_islands) - target_size
            warnings = list(recognition.warnings)
            if removed:
                warnings.append(
                    f"Ignored {removed} low-confidence full-client UI circles"
                )
            located = LocatedBoard(crop, region, Recognition(islands, crop, warnings), puzzle)
            return located, match

        detail = f": {last_error}" if last_error is not None else ""
        raise GamePuzzleMatchError(
            f"Full-client recovery could not match {len(raw_islands)} visual islands{detail}"
        )

    def _identify_board(
        self,
        client_image: np.ndarray,
    ) -> tuple[LocatedBoard, GamePuzzleMatch]:
        try:
            located = locate_board(client_image)
            return located, self.library.match(located.puzzle)
        except BridgeAssistantError as localized_error:
            try:
                return self._recover_full_client_board(client_image)
            except BridgeAssistantError as recovery_error:
                raise recovery_error from localized_error

    @staticmethod
    def _board_region(
        client_region: ScreenRegion,
        located: LocatedBoard,
    ) -> ScreenRegion:
        return ScreenRegion(
            client_region.left + located.region.left,
            client_region.top + located.region.top,
            located.region.width,
            located.region.height,
        )

    def _capture_prepared_once(self, window: AppWindow) -> PreparedLevel:
        client_region, client_image = self._capture_client(window)
        located, match = self._identify_board(client_image)
        snapshot = self.solved_snapshot()
        disposition = (
            LevelDisposition.RECOVERED_EXISTING
            if match.number in snapshot.numbers
            else LevelDisposition.UNSOLVED
        )
        return PreparedLevel(
            window,
            client_region,
            self._board_region(client_region, located),
            match,
            client_image,
            snapshot,
            disposition,
        )

    def prepare_level(self, window: AppWindow, difficulty: str | None) -> PreparedLevel:
        requested = _normalized_difficulty(difficulty) if difficulty else None
        if requested is not None and requested not in DIFFICULTIES:
            raise DifficultyMenuError(f"Unknown BridgeFix difficulty: {difficulty}")
        progress = self.solved_snapshot()
        if not self._progress_logged:
            counts = ", ".join(
                f"{name}={len(numbers & progress.numbers)}/{len(numbers)}"
                for name, numbers in self._numbers_by_difficulty.items()
            )
            LOGGER.info(
                "progress total=%s file=%s difficulties=[%s]",
                len(progress.numbers),
                progress.files[0][0] if progress.files else "not-created",
                counts,
            )
            self._progress_logged = True
        deadline = time.monotonic() + self.advance_timeout
        self._requested_difficulty = requested
        last_error: Exception | None = None
        expected_difficulty = requested
        selection_started = False
        recovery_checked = False
        while time.monotonic() <= deadline:
            self._ensure_running()
            client_region, client_image = self._capture_client(window)
            try:
                located, match = self._identify_board(client_image)
            except BridgeAssistantError as error:
                last_error = error
                snapshot = self.solved_snapshot()
                centers = difficulty_button_centers(client_image)
                if centers and not selection_started:
                    decision, centers, states, digit_counts = (
                        self._resolve_difficulty_menu_decision(
                            window,
                            client_region,
                            client_image,
                            centers,
                            decide=lambda current_states: (
                                self._difficulty_selection_decision(
                                    current_states,
                                    expected_difficulty,
                                )
                            ),
                            wait_for_percentage_update=False,
                            deadline=deadline,
                        )
                    )
                    LOGGER.info(
                        "difficulty_menu startup action=%s screen_complete=%s "
                        "integer_digits=[%s]",
                        decision.action.value,
                        ",".join(
                            name
                            for name, complete in zip(DIFFICULTIES, states)
                            if complete is True
                        )
                        or "none",
                        self._format_percentage_readings(digit_counts),
                    )
                    if decision.action is DifficultyMenuAction.RETRY:
                        LOGGER.warning(
                            "difficulty_menu startup percentages remained unreadable; "
                            "closing menu and continuing integer_digits=[%s]",
                            self._format_percentage_readings(digit_counts),
                        )
                        self._close_difficulty_menu(window, client_region, centers)
                    elif decision.action is DifficultyMenuAction.COMPLETE:
                        self._close_difficulty_menu(window, client_region, centers)
                        if expected_difficulty is None:
                            raise AutomaticRunComplete(
                                "All BridgeFix difficulties are complete"
                            )
                        raise AutomaticRunComplete(
                            f"BridgeFix difficulty {expected_difficulty} is already complete"
                        )
                    elif decision.action is DifficultyMenuAction.SWITCH:
                        target = decision.target
                        if target is None:
                            raise DifficultyMenuError(
                                "Difficulty selection decision did not include a target"
                            )
                        self._click_difficulty_button(
                            window,
                            client_region,
                            centers,
                            target,
                            deadline=deadline,
                        )
                        expected_difficulty = target
                        selection_started = True
                elif not recovery_checked and snapshot.ordered_numbers:
                    recovery_checked = True
                    pulse_confirmed = self._observe_next_glow_pulse(
                        window,
                        client_region,
                        client_image,
                    )
                    recovery = self._prepare_victory_recovery(
                        window,
                        client_region,
                        client_image,
                        snapshot,
                        pulse_confirmed=pulse_confirmed,
                    )
                    if recovery is not None:
                        actual = _normalized_difficulty(recovery.match.difficulty)
                        if expected_difficulty and actual != expected_difficulty:
                            if not selection_started:
                                self._select_difficulty(
                                    window,
                                    expected_difficulty,
                                    deadline=deadline,
                                )
                                selection_started = True
                            else:
                                self._wait(self.poll_interval)
                            continue
                        visually_complete, target = self._handle_visual_difficulty(
                            window,
                            actual,
                            stop_after_current=requested is not None,
                            wait_for_percentage_update=self._percentage_refresh_expected(
                                actual,
                                snapshot,
                            ),
                            deadline=deadline,
                        )
                        if visually_complete:
                            if requested:
                                raise AutomaticRunComplete(
                                    f"BridgeFix difficulty {requested} is already complete"
                                )
                            if target is None:
                                raise AutomaticRunComplete("All BridgeFix difficulties are complete")
                            expected_difficulty = target
                            selection_started = True
                            continue
                        return recovery
                else:
                    self._wait(self.poll_interval)
                continue

            actual = _normalized_difficulty(match.difficulty)
            if expected_difficulty and actual != expected_difficulty:
                if not selection_started:
                    self._select_difficulty(
                        window,
                        expected_difficulty,
                        deadline=deadline,
                    )
                    selection_started = True
                else:
                    self._wait(self.poll_interval)
                continue
            snapshot = self.solved_snapshot()
            visually_complete, target = self._handle_visual_difficulty(
                window,
                actual,
                stop_after_current=requested is not None,
                wait_for_percentage_update=self._percentage_refresh_expected(
                    actual,
                    snapshot,
                ),
                deadline=deadline,
            )
            if visually_complete:
                if requested:
                    raise AutomaticRunComplete(
                        f"BridgeFix difficulty {requested} is already complete"
                    )
                if target is None:
                    raise AutomaticRunComplete("All BridgeFix difficulties are complete")
                expected_difficulty = target
                selection_started = True
                continue
            disposition = (
                LevelDisposition.RECOVERED_EXISTING
                if match.number in snapshot.numbers
                else LevelDisposition.UNSOLVED
            )
            return PreparedLevel(
                window,
                client_region,
                self._board_region(client_region, located),
                match,
                client_image,
                snapshot,
                disposition,
            )

        detail = f": {last_error}" if last_error else ""
        if expected_difficulty and selection_started:
            raise DifficultyMismatchError(
                f"Difficulty {expected_difficulty} did not become ready before the deadline"
                f"{detail}"
            )
        raise AutoRunnerError(f"BridgeFix puzzle did not become ready{detail}")

    def reset(self, level: PreparedLevel) -> None:
        representatives = representative_solution(
            level.match.puzzle,
            level.match.official_solution,
        )
        clear_actions = build_mouse_actions(
            level.match.puzzle,
            Solution({edge_id: 1 for edge_id in representatives.bridges}),
            level.board_region,
        )
        for action in clear_actions:
            self._level_click(level, action.start, button="right")
            self._wait(0.04)
        self._wait(0.20)
        level.client_image = pil_to_bgr(grab_screen(level.client_region))

    def _capture_board(self, level: PreparedLevel) -> np.ndarray:
        return pil_to_bgr(grab_screen(level.board_region))

    def clear_edge(self, level: PreparedLevel, action: MouseAction) -> None:
        self._level_click(level, action.start, button="right")
        self._wait(0.08)
        self._edge_baseline = self._capture_board(level)

    def add_bridge(self, level: PreparedLevel, action: MouseAction) -> bool:
        if self._edge_baseline is None:
            raise AutoRunnerError("clear_edge must be called before add_bridge")
        self._level_click(level, action.start, button="left")
        self._wait(0.12)
        current = self._capture_board(level)
        if action.observation_start is None or action.observation_end is None:
            return False
        start = (
            action.observation_start[0] - level.board_region.left,
            action.observation_start[1] - level.board_region.top,
        )
        end = (
            action.observation_end[0] - level.board_region.left,
            action.observation_end[1] - level.board_region.top,
        )
        score = bridge_corridor_change(
            self._edge_baseline,
            current,
            start,
            end,
            action.observation_radius,
        )
        if score < 0.025:
            solved = self.solved_snapshot()
            if not self._has_exact_progress_delta(level, solved):
                return False
        self._edge_baseline = current
        return True

    def victory_confirmed(self, level: PreparedLevel) -> bool:
        deadline = time.monotonic() + self.victory_timeout
        while time.monotonic() <= deadline:
            self._ensure_running()
            current_solved = self.solved_snapshot()
            if self._has_exact_progress_delta(level, current_solved):
                LOGGER.info("puzzle=%s saved_to_solved_dat", level.number)
                return True
            self._wait(self.poll_interval)
        return False

    @staticmethod
    def _has_exact_progress_delta(
        level: PreparedLevel,
        current: SolvedSnapshot,
    ) -> bool:
        before = level.solved_before.numbers
        removed = before - current.numbers
        expected = (
            frozenset({level.number})
            if level.disposition is LevelDisposition.UNSOLVED
            else frozenset()
        )
        added = current.numbers - before
        unexpected = added - expected
        if removed or unexpected:
            details = []
            if removed:
                details.append(f"removed={sorted(removed)}")
            if unexpected:
                details.append(f"unexpected_added={sorted(unexpected)}")
            raise ProgressDesynchronizedError(
                "BridgeFix solved.dat changed outside the current puzzle: "
                + ", ".join(details)
            )
        if (
            level.disposition is not LevelDisposition.UNSOLVED
            and level.number not in before
        ):
            raise ProgressDesynchronizedError(
                f"Recovered puzzle {level.number} was not present in the baseline solved.dat"
            )
        return added == expected

    @staticmethod
    def _same_level(first: PreparedLevel, second: PreparedLevel) -> bool:
        return first.number == second.number and first.signature == second.signature

    def _wait_for_selected_difficulty(
        self,
        level: PreparedLevel,
        difficulty: str,
        deadline: float,
    ) -> PreparedLevel:
        last_error: BridgeAssistantError | None = None
        while time.monotonic() <= deadline:
            self._ensure_running()
            try:
                current = self._capture_prepared_once(level.window)
            except BridgeAssistantError as error:
                self.recognition_failures += 1
                last_error = error
                self._wait(self.poll_interval)
                continue
            if _normalized_difficulty(current.match.difficulty) == difficulty:
                return current
            self._wait(self.poll_interval)
        detail = f"; last capture failed: {last_error}" if last_error else ""
        raise DifficultyMismatchError(
            f"Difficulty {difficulty} did not become ready before the deadline{detail}"
        )

    def _click_next_and_confirm(
        self,
        level: PreparedLevel,
        difficulty: str | None,
        minimum_delay: float,
        deadline: float,
    ) -> PreparedLevel:
        click_count = 0
        stable_same_count = 0
        stable_same_since: float | None = None
        previous_same_image: np.ndarray | None = None
        last_error: BridgeAssistantError | None = None
        latest_same = level

        while time.monotonic() <= deadline:
            self._ensure_running()
            if click_count == 0:
                self._ui_click(
                    latest_same.window,
                    latest_same.client_region,
                    self.controls.next_point(latest_same.client_region),
                )
                click_count = 1
                LOGGER.info("puzzle=%s next_click=%s", level.number, click_count)
                self._wait(max(DEFAULT_NEXT_DELAY, minimum_delay))

            try:
                current = self._capture_prepared_once(level.window)
            except BridgeAssistantError as error:
                self.recognition_failures += 1
                last_error = error
                stable_same_count = 0
                stable_same_since = None
                previous_same_image = None
                self._wait(self.poll_interval)
                continue

            actual = _normalized_difficulty(current.match.difficulty)
            if difficulty is not None and actual != _normalized_difficulty(difficulty):
                raise DifficultyMismatchError(
                    f"Current puzzle difficulty is {current.match.difficulty}, expected {difficulty}"
                )
            if not self._same_level(level, current):
                LOGGER.info(
                    "puzzle=%s advanced_to=%s next_clicks=%s",
                    level.number,
                    current.number,
                    click_count,
                )
                return current

            now = time.monotonic()
            latest_same = current
            frame_stable = (
                previous_same_image is not None
                and client_frames_stable(previous_same_image, current.client_image)
            )
            previous_same_image = current.client_image
            if not frame_stable:
                stable_same_since = now
                stable_same_count = 1
            else:
                stable_same_count += 1
            if stable_same_count >= 3 and now - stable_same_since >= 0.75:
                if click_count >= self.advance_clicks:
                    break
                click_count += 1
                self.advance_retries += 1
                LOGGER.warning(
                    "puzzle=%s next_retry=%s reason=stable_unchanged",
                    level.number,
                    click_count,
                )
                self._ui_click(
                    latest_same.window,
                    latest_same.client_region,
                    self.controls.next_point(latest_same.client_region),
                )
                self._wait(max(DEFAULT_NEXT_DELAY, minimum_delay))
                stable_same_count = 0
                stable_same_since = None
                previous_same_image = None
            else:
                self._wait(self.poll_interval)

        detail = f"; last capture failed: {last_error}" if last_error else ""
        raise LevelDidNotAdvanceError(
            f"Next was clicked {click_count} time(s), but a different puzzle could not be "
            f"confirmed after puzzle {level.number}{detail}"
        )

    def advance_level(
        self,
        level: PreparedLevel,
        difficulty: str | None,
        minimum_delay: float,
    ) -> PreparedLevel | None:
        deadline = time.monotonic() + self.advance_timeout
        progress = self.solved_snapshot()
        if level.disposition is LevelDisposition.UNSOLVED:
            if not self._has_exact_progress_delta(level, progress):
                raise VictoryNotConfirmedError(
                    f"Puzzle {level.number} was not newly recorded in solved.dat"
                )
        elif not self._has_exact_progress_delta(level, progress):
            raise ProgressDesynchronizedError(
                f"Recovered puzzle {level.number} is missing from solved.dat"
            )

        visually_complete, next_difficulty = self._handle_visual_difficulty(
            level.window,
            level.match.difficulty,
            stop_after_current=self._requested_difficulty is not None,
            wait_for_percentage_update=self._percentage_refresh_expected(
                level.match.difficulty,
                progress,
            ),
            deadline=deadline,
        )
        if visually_complete:
            if next_difficulty is None:
                return None
            return self._wait_for_selected_difficulty(
                level,
                next_difficulty,
                deadline,
            )
        return self._click_next_and_confirm(
            level,
            difficulty,
            minimum_delay,
            deadline,
        )


def run(
    *,
    once: bool = False,
    levels: int | None = None,
    difficulty: str | None = None,
    resources_path: str | None = None,
    runtime: RunnerRuntime | None = None,
    state_changed: Callable[[RunnerState], None] | None = None,
    level_changed: Callable[[PreparedLevel, int], None] | None = None,
) -> int:
    if once and levels is not None:
        raise ValueError("--once and --levels cannot be used together")
    if difficulty is not None and _normalized_difficulty(difficulty) not in DIFFICULTIES:
        raise DifficultyMenuError(f"Unknown BridgeFix difficulty: {difficulty}")
    requested_levels = 1 if once else levels
    if runtime is None:
        enable_dpi_awareness()
        runtime = DesktopRuntime(load_game_library(resources_path))
    return AutoRunner(
        runtime,
        state_changed=state_changed,
        level_changed=level_changed,
    ).run(
        levels=requested_levels,
        difficulty=difficulty,
    )


def _parser() -> argparse.ArgumentParser:
    def positive_int(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be at least one")
        return parsed

    parser = argparse.ArgumentParser(description="Run BridgeFix 231 puzzles unattended")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="solve one puzzle")
    group.add_argument("--levels", type=positive_int, help="number of puzzles to solve")
    group.add_argument("--manual", action="store_true", help="open the legacy manual GUI")
    parser.add_argument("--difficulty", help="require this BridgeFix difficulty")
    parser.add_argument("--resources-path", help="BridgeFix executable or resources.assets path")
    parser.add_argument("--log-dir", help="directory for persistent automatic-run logs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.manual:
        if args.difficulty or args.resources_path or args.log_dir:
            parser.error("--manual cannot be combined with automatic-runner options")
        from .app import main as manual_main

        manual_main()
        return 0
    try:
        with SingleInstanceLock():
            with RunDiagnostics(
                log_dir=args.log_dir,
                argv=list(argv) if argv is not None else sys.argv[1:],
            ) as diagnostics:
                diagnostics.info("Persistent log: %s", diagnostics.log_path)
                try:
                    completed = run(
                        once=args.once,
                        levels=args.levels,
                        difficulty=args.difficulty,
                        resources_path=args.resources_path,
                        state_changed=diagnostics.set_state,
                        level_changed=lambda level, completed: diagnostics.heartbeat(
                            completed=completed,
                            newly_solved=completed,
                            current_puzzle=level.number,
                            difficulty=level.match.difficulty,
                        ),
                    )
                except (KeyboardInterrupt, InterruptedError):
                    diagnostics.mark_stopped()
                    return 130
                except Exception as error:
                    diagnostics.mark_failed(error)
                    return 1
                diagnostics.mark_completed(completed, newly_solved=completed)
                return 0
    except AnotherInstanceError as error:
        print(f"Bridge Assistant: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
