from __future__ import annotations

import time
from collections.abc import Callable
from statistics import median

import cv2
import numpy as np

from .model import Island, MouseAction, SceneChangedError, ScreenRegion


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def scene_similarity(reference: np.ndarray, current: np.ndarray, islands: list[Island]) -> float:
    if reference.shape[:2] != current.shape[:2] or not islands:
        return 0.0
    reference_gray = _gray(reference)
    current_gray = _gray(current)
    height, width = reference_gray.shape
    scores: list[float] = []
    for island in islands:
        radius = max(5, int(round(island.radius * 0.82)))
        margin = max(3, int(round(radius * 0.22)))
        cx, cy = int(round(island.x)), int(round(island.y))
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(width, cx + radius + 1), min(height, cy + radius + 1)
        sx1, sy1 = max(0, x1 - margin), max(0, y1 - margin)
        sx2, sy2 = min(width, x2 + margin), min(height, y2 + margin)
        template = reference_gray[y1:y2, x1:x2]
        search = current_gray[sy1:sy2, sx1:sx2]
        if min(template.shape) < 5 or search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            continue
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        score = float(np.nanmax(result)) if result.size else 0.0
        scores.append(max(0.0, min(1.0, score)))
    if not scores:
        return 0.0
    # A cursor can obscure one island, so favor the median while retaining a
    # small penalty when multiple patches changed.
    return float(0.72 * median(scores) + 0.28 * np.mean(scores))


def verify_scene(
    reference: np.ndarray,
    current: np.ndarray,
    islands: list[Island],
    threshold: float = 0.72,
) -> float:
    similarity = scene_similarity(reference, current, islands)
    if similarity < threshold:
        raise SceneChangedError(
            f"Target board changed or moved (similarity {similarity:.0%}); capture it again"
        )
    return similarity


def bridge_corridor_change(
    reference: np.ndarray,
    current: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    island_radius: float,
) -> float:
    """Return the fraction of visibly changed pixels between two islands."""

    if reference.shape != current.shape or reference.size == 0:
        return 0.0
    x1, y1 = start
    x2, y2 = end
    height, width = reference.shape[:2]
    trim = max(4, int(round(island_radius * 1.15)))
    half_width = max(4, int(round(island_radius * 0.3)))

    if abs(y1 - y2) <= abs(x1 - x2):
        left = max(0, int(round(min(x1, x2))) + trim)
        right = min(width, int(round(max(x1, x2))) - trim + 1)
        center = int(round((y1 + y2) / 2))
        top, bottom = max(0, center - half_width), min(height, center + half_width + 1)
    else:
        top = max(0, int(round(min(y1, y2))) + trim)
        bottom = min(height, int(round(max(y1, y2))) - trim + 1)
        center = int(round((x1 + x2) / 2))
        left, right = max(0, center - half_width), min(width, center + half_width + 1)
    if right <= left or bottom <= top:
        return 0.0

    before = reference[top:bottom, left:right].astype(np.int16)
    after = current[top:bottom, left:right].astype(np.int16)
    if before.ndim == 3:
        difference = np.max(np.abs(after - before), axis=2)
    else:
        difference = np.abs(after - before)
    return float(np.mean(difference >= 14))


class VisualActionVerifier:
    """Confirm mouse operations from pixel changes in their bridge corridor."""

    def __init__(
        self,
        baseline: np.ndarray,
        region: ScreenRegion,
        capture: Callable[[], np.ndarray],
        *,
        threshold: float = 0.025,
        settle_delay: float = 0.10,
        sample_count: int = 3,
    ) -> None:
        self.baseline = baseline.copy()
        self.region = region
        self.capture = capture
        self.threshold = threshold
        self.settle_delay = max(0.0, settle_delay)
        self.sample_count = max(1, sample_count)
        self.last_score = 0.0

    def __call__(self, action: MouseAction) -> bool:
        if action.observation_start is None or action.observation_end is None:
            return False
        start = (
            action.observation_start[0] - self.region.left,
            action.observation_start[1] - self.region.top,
        )
        end = (
            action.observation_end[0] - self.region.left,
            action.observation_end[1] - self.region.top,
        )
        best_score = 0.0
        for _ in range(self.sample_count):
            if self.settle_delay:
                time.sleep(self.settle_delay)
            current = self.capture()
            score = bridge_corridor_change(
                self.baseline,
                current,
                start,
                end,
                action.observation_radius,
            )
            best_score = max(best_score, score)
            if score >= self.threshold:
                self.baseline = current.copy()
                self.last_score = score
                return True
        self.last_score = best_score
        return False
