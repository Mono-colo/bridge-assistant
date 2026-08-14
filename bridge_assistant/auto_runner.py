from __future__ import annotations

import argparse
import re
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
from .board_locator import locate_board
from .capture import enable_dpi_awareness, grab_screen, pil_to_bgr
from .game_library import (
    LIBRARY_ASSET_NAMES,
    GameLibrary,
    GamePuzzleMatch,
    load_game_library,
)
from .model import (
    AppWindow,
    BridgeAssistantError,
    MouseAction,
    Puzzle,
    ScreenRegion,
    Solution,
    TargetWindowError,
)
from .scene import bridge_corridor_change
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
DIFFICULTIES = tuple(name.removeprefix("library") for name in LIBRARY_ASSET_NAMES)


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


class RunnerState(str, Enum):
    IDLE = "idle"
    WAITING_FOR_WINDOW = "waiting_for_window"
    CAPTURING = "capturing"
    RESETTING = "resetting"
    INPUTTING = "inputting"
    VERIFYING_VICTORY = "verifying_victory"
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


@dataclass(slots=True)
class PreparedLevel:
    window: AppWindow
    client_region: ScreenRegion
    board_region: ScreenRegion
    match: GamePuzzleMatch
    client_image: np.ndarray
    solved_before: SolvedSnapshot = field(default_factory=SolvedSnapshot)
    advance_only: bool = False

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

    def click_next(self, level: PreparedLevel) -> bool | None: ...

    def wait_for_level_change(
        self,
        level: PreparedLevel,
        difficulty: str | None,
        minimum_delay: float,
    ) -> PreparedLevel: ...


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
        self.stop_event = stop_event or threading.Event()
        self.state = RunnerState.IDLE
        self.completed_levels = 0

    def _transition(self, state: RunnerState) -> None:
        self.state = state
        if self.state_changed is not None:
            self.state_changed(state)

    def _ensure_running(self) -> None:
        if self.stop_event.is_set() or escape_is_pressed():
            self.stop_event.set()
            self._transition(RunnerState.STOPPED)
            raise InterruptedError("Automatic runner stopped")

    def _input_actions(self, level: PreparedLevel) -> list[MouseAction]:
        solution = representative_solution(level.match.puzzle, level.match.official_solution)
        return build_mouse_actions(level.match.puzzle, solution, level.board_region)

    def _attempt_level(self, level: PreparedLevel) -> None:
        if level.advance_only:
            self._transition(RunnerState.VERIFYING_VICTORY)
            if self.runtime.victory_confirmed(level):
                return
            raise VictoryNotConfirmedError(
                f"Victory was not confirmed for recovered puzzle {level.number}"
            )
        if (
            victory_indicator_visible(level.client_image)
            or level.number in level.solved_before.numbers
        ):
            self._transition(RunnerState.VERIFYING_VICTORY)
            if self.runtime.victory_confirmed(level):
                return
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
            while levels is None or self.completed_levels < levels:
                self._ensure_running()
                if current is None:
                    self._transition(RunnerState.CAPTURING)
                    current = self.runtime.prepare_level(window, selected_difficulty)

                self._attempt_level(current)
                if not current.advance_only:
                    self.completed_levels += 1
                self._transition(RunnerState.ADVANCING)
                advanced = self.runtime.click_next(current)
                if advanced is False:
                    self._transition(RunnerState.COMPLETE)
                    return self.completed_levels
                self._transition(RunnerState.WAITING_FOR_CHANGE)
                current = self.runtime.wait_for_level_change(
                    current,
                    selected_difficulty,
                    self.next_delay,
                )

            self._transition(RunnerState.COMPLETE)
            return self.completed_levels
        except InterruptedError:
            if self.state != RunnerState.STOPPED:
                self._transition(RunnerState.STOPPED)
            raise
        except AutomaticRunComplete:
            self._transition(RunnerState.COMPLETE)
            return self.completed_levels
        except Exception:
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


def victory_indicator_score(image: np.ndarray) -> float:
    """Measure the bright NextGlow behind BridgeFix's bottom-right button."""

    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    patch = gray[round(height * 0.88) :, round(width * 0.94) :]
    return float(np.mean(patch >= 220)) if patch.size else 0.0


def victory_indicator_visible(image: np.ndarray) -> bool:
    return victory_indicator_score(image) >= 0.15


class DesktopRuntime:
    def __init__(
        self,
        library: GameLibrary,
        *,
        driver=pyautogui,
        controls: ControlLayout = ControlLayout(),
        poll_interval: float = 0.35,
        victory_timeout: float = 5.0,
        change_timeout: float = 10.0,
    ) -> None:
        self.library = library
        self.driver = driver
        self.controls = controls
        self.poll_interval = max(0.05, poll_interval)
        self.victory_timeout = max(0.0, victory_timeout)
        self.change_timeout = max(DEFAULT_NEXT_DELAY, change_timeout)
        self.driver.FAILSAFE = True
        self.driver.PAUSE = 0.03
        self._edge_baseline: np.ndarray | None = None
        self._solved_paths: tuple[Path, ...] = ()
        self._requested_difficulty: str | None = None
        self._stop_event = threading.Event()
        self._numbers_by_difficulty = {
            difficulty: frozenset(
                puzzle.number
                for puzzle in library.puzzles
                if _normalized_difficulty(puzzle.difficulty) == difficulty
            )
            for difficulty in DIFFICULTIES
        }

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

    def _difficulty_complete(
        self,
        difficulty: str,
        snapshot: SolvedSnapshot | None = None,
    ) -> bool:
        numbers = self._numbers_by_difficulty.get(_normalized_difficulty(difficulty))
        return bool(numbers) and numbers <= (snapshot or self.solved_snapshot()).numbers

    def _next_incomplete_difficulty(
        self,
        *,
        after: str | None = None,
        snapshot: SolvedSnapshot | None = None,
    ) -> str | None:
        progress = snapshot or self.solved_snapshot()
        ordered = list(DIFFICULTIES)
        if after is not None and _normalized_difficulty(after) in ordered:
            pivot = ordered.index(_normalized_difficulty(after)) + 1
            ordered = ordered[pivot:] + ordered[:pivot]
        return next(
            (
                difficulty
                for difficulty in ordered
                if not self._difficulty_complete(difficulty, progress)
            ),
            None,
        )

    def _capture_client(self, window: AppWindow) -> tuple[ScreenRegion, np.ndarray]:
        self._ensure_running()
        activate_window(window)
        region = window_client_region(window)
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
    ) -> tuple[ScreenRegion, tuple[tuple[int, int], ...]]:
        deadline = time.monotonic() + timeout
        clicked_menu = False
        while time.monotonic() <= deadline:
            client_region, image = self._capture_client(window)
            centers = difficulty_button_centers(image)
            if centers:
                return client_region, centers
            if not clicked_menu:
                self._ui_click(
                    window,
                    client_region,
                    self.controls.menu_point(client_region),
                )
                clicked_menu = True
            self._wait(self.poll_interval)
        raise DifficultyMenuError("BridgeFix difficulty menu did not become visible")

    def _select_difficulty(self, window: AppWindow, difficulty: str) -> None:
        normalized = _normalized_difficulty(difficulty)
        try:
            index = DIFFICULTIES.index(normalized)
        except ValueError as error:
            raise DifficultyMenuError(f"Unknown BridgeFix difficulty: {difficulty}") from error
        client_region, centers = self._wait_for_difficulty_menu(window)
        local_x, local_y = centers[index]
        self._ui_click(
            window,
            client_region,
            (client_region.left + local_x, client_region.top + local_y),
        )
        self._wait(DEFAULT_NEXT_DELAY)

    def solved_snapshot(self) -> SolvedSnapshot:
        # Unity may create the persistence file only after the first win.
        discovered = set(self._solved_paths)
        discovered.update(self._find_solved_files())
        self._solved_paths = tuple(sorted(discovered, key=str))
        numbers: set[int] = set()
        ordered_numbers: list[int] = []
        files: list[tuple[str, int, int]] = []
        for path in self._solved_paths:
            try:
                data = path.read_bytes()
                stat = path.stat()
            except OSError:
                continue
            files.append((str(path), stat.st_mtime_ns, len(data)))
            values = [
                int(value)
                for value in re.findall(rb"(?<!\d)\d{1,7}(?!\d)", data)
            ]
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
    ) -> PreparedLevel | None:
        if not victory_indicator_visible(client_image) or not snapshot.ordered_numbers:
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
            advance_only=True,
        )

    def wait_for_window(self, stop_event: threading.Event) -> AppWindow:
        self._stop_event = stop_event
        while not stop_event.is_set():
            self._ensure_running()
            matches = [window for window in list_app_windows() if is_target_window(window)]
            if matches:
                return matches[0]
            self._wait(self.poll_interval)
        raise InterruptedError("Automatic runner stopped")

    def prepare_level(self, window: AppWindow, difficulty: str | None) -> PreparedLevel:
        requested = _normalized_difficulty(difficulty) if difficulty else None
        if requested is not None and requested not in DIFFICULTIES:
            raise DifficultyMenuError(f"Unknown BridgeFix difficulty: {difficulty}")
        progress = self.solved_snapshot()
        if requested is not None and self._difficulty_complete(requested, progress):
            raise AutomaticRunComplete(
                f"BridgeFix difficulty {requested} is already complete"
            )
        if requested is None and self._next_incomplete_difficulty(snapshot=progress) is None:
            raise AutomaticRunComplete("All BridgeFix difficulties are complete")
        self._requested_difficulty = requested
        deadline = time.monotonic() + self.change_timeout
        last_error: Exception | None = None
        selected = False
        while time.monotonic() <= deadline:
            self._ensure_running()
            client_region, client_image = self._capture_client(window)
            try:
                located = locate_board(client_image)
                match = self.library.match(located.puzzle)
            except BridgeAssistantError as error:
                last_error = error
                snapshot = self.solved_snapshot()
                recovery = self._prepare_victory_recovery(
                    window,
                    client_region,
                    client_image,
                    snapshot,
                )
                if recovery is not None:
                    return recovery
                centers = difficulty_button_centers(client_image)
                if centers and not selected:
                    target = requested or self._next_incomplete_difficulty()
                    if target is None:
                        raise DifficultyMenuError("All BridgeFix difficulties are complete")
                    local_x, local_y = centers[DIFFICULTIES.index(target)]
                    self._ui_click(
                        window,
                        client_region,
                        (client_region.left + local_x, client_region.top + local_y),
                    )
                    selected = True
                    self._wait(DEFAULT_NEXT_DELAY)
                    deadline = time.monotonic() + self.change_timeout
                else:
                    self._wait(self.poll_interval)
                continue

            actual = _normalized_difficulty(match.difficulty)
            if requested and actual != requested:
                if selected:
                    raise DifficultyMismatchError(
                        f"Current puzzle difficulty is {match.difficulty}, expected {requested}"
                    )
                self._select_difficulty(window, requested)
                selected = True
                deadline = time.monotonic() + self.change_timeout
                continue
            board_region = ScreenRegion(
                client_region.left + located.region.left,
                client_region.top + located.region.top,
                located.region.width,
                located.region.height,
            )
            return PreparedLevel(
                window,
                client_region,
                board_region,
                match,
                client_image,
                self.solved_snapshot(),
            )

        detail = f": {last_error}" if last_error else ""
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
            if level.number not in solved.numbers - level.solved_before.numbers:
                return False
        self._edge_baseline = current
        return True

    @staticmethod
    def _next_glow_change(level: PreparedLevel, current: np.ndarray) -> float:
        baseline = level.client_image
        if baseline.shape != current.shape or baseline.size == 0:
            return 0.0
        difference = np.max(
            np.abs(current.astype(np.int16) - baseline.astype(np.int16)),
            axis=2,
        )
        height, width = difference.shape
        mask = np.zeros(difference.shape, dtype=bool)
        mask[round(height * 0.70) :, round(width * 0.76) :] = True
        left = max(0, level.board_region.left - level.client_region.left)
        top = max(0, level.board_region.top - level.client_region.top)
        right = min(mask.shape[1], left + level.board_region.width)
        bottom = min(mask.shape[0], top + level.board_region.height)
        mask[top:bottom, left:right] = False
        pixels = difference[mask]
        return float(np.mean(pixels >= 18)) if pixels.size else 0.0

    def victory_confirmed(self, level: PreparedLevel) -> bool:
        deadline = time.monotonic() + self.victory_timeout
        visual_hits = 0
        while time.monotonic() <= deadline:
            self._ensure_running()
            current_solved = self.solved_snapshot()
            if level.number in current_solved.numbers - level.solved_before.numbers:
                return True
            current = pil_to_bgr(grab_screen(level.client_region))
            if (
                victory_indicator_visible(current)
                or self._next_glow_change(level, current) >= 0.015
            ):
                visual_hits += 1
                if visual_hits >= 2:
                    return True
            else:
                visual_hits = 0
            self._wait(self.poll_interval)
        return False

    def _wait_for_next_ready(self, level: PreparedLevel, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() <= deadline:
            self._ensure_running()
            self._verify_click_target(
                level.window,
                level.client_region,
                self.controls.next_point(level.client_region),
            )
            current = pil_to_bgr(grab_screen(level.client_region))
            if victory_indicator_visible(current):
                return
            self._wait(0.10)
        raise VictoryNotConfirmedError(
            f"Puzzle {level.number} was recorded as solved, but NextGlow did not appear"
        )

    def click_next(self, level: PreparedLevel) -> bool:
        if level.advance_only:
            self._wait_for_next_ready(level)
            self._ui_click(
                level.window,
                level.client_region,
                self.controls.next_point(level.client_region),
            )
            return True
        progress = self.solved_snapshot()
        if self._difficulty_complete(level.match.difficulty, progress):
            if self._requested_difficulty:
                return False
            next_difficulty = self._next_incomplete_difficulty(
                after=level.match.difficulty,
                snapshot=progress,
            )
            if next_difficulty is None:
                return False
            self._select_difficulty(level.window, next_difficulty)
            return True
        self._wait_for_next_ready(level)
        self._ui_click(
            level.window,
            level.client_region,
            self.controls.next_point(level.client_region),
        )
        return True

    def wait_for_level_change(
        self,
        level: PreparedLevel,
        difficulty: str | None,
        minimum_delay: float,
    ) -> PreparedLevel:
        self._wait(max(DEFAULT_NEXT_DELAY, minimum_delay))
        deadline = time.monotonic() + self.change_timeout
        while time.monotonic() <= deadline:
            self._ensure_running()
            try:
                current = self.prepare_level(level.window, difficulty)
            except (AutomaticRunComplete, DifficultyMismatchError):
                raise
            except BridgeAssistantError:
                self._wait(self.poll_interval)
                continue
            if current.number != level.number or current.signature != level.signature:
                return current
            self._wait(self.poll_interval)
        raise LevelDidNotAdvanceError(
            f"Next was clicked once, but puzzle {level.number} did not change"
        )


def run(
    *,
    once: bool = False,
    levels: int | None = None,
    difficulty: str | None = None,
    resources_path: str | None = None,
    runtime: RunnerRuntime | None = None,
    state_changed: Callable[[RunnerState], None] | None = None,
) -> int:
    if once and levels is not None:
        raise ValueError("--once and --levels cannot be used together")
    if difficulty is not None and _normalized_difficulty(difficulty) not in DIFFICULTIES:
        raise DifficultyMenuError(f"Unknown BridgeFix difficulty: {difficulty}")
    requested_levels = 1 if once else levels
    if runtime is None:
        enable_dpi_awareness()
        runtime = DesktopRuntime(load_game_library(resources_path))
    return AutoRunner(runtime, state_changed=state_changed).run(
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.manual:
        if args.difficulty or args.resources_path:
            parser.error("--manual cannot be combined with automatic-runner options")
        from .app import main as manual_main

        manual_main()
        return 0
    try:
        completed = run(
            once=args.once,
            levels=args.levels,
            difficulty=args.difficulty,
            resources_path=args.resources_path,
            state_changed=lambda state: print(f"[auto] {state.value}", flush=True),
        )
    except KeyboardInterrupt:
        return 130
    except InterruptedError:
        print("BridgeFix auto runner stopped.")
        return 130
    except BridgeAssistantError as error:
        print(f"BridgeFix auto runner failed: {error}")
        return 1
    print(f"Completed {completed} level(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
