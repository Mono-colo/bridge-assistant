from __future__ import annotations

from collections import deque
from statistics import median

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import build_puzzle
from .model import Island, LocatedBoard, RecognitionError, ScreenRegion
from .vision import (
    _circle_candidates,
    _classify_digit_scales,
    _extract_digit_scales,
    _invalid_glyph_distance,
    _local_sharpness,
    _MAX_VALID_GLYPH_DISTANCE,
    _MIN_SHARPNESS_FOR_INVALID_GLYPH_CHECK,
    _MIN_VALID_GLYPH_ADVANTAGE,
    recognize,
)


def _gray_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _localization_candidates(image: np.ndarray) -> list[Island]:
    gray = _gray_image(image)
    height, width = gray.shape
    # Work in original physical pixels.  Downscaling a full 4K client area
    # makes small island digits unreadable before a board crop is known.
    scale = 1.0
    scanning = gray

    # The regular recognizer deliberately ties its minimum circle size to a
    # tight board crop.  Tile the client area here so a small board in a large
    # or 4K application window is still examined at a useful scale.
    tile_size = 1400
    overlap = 220
    candidates: list[tuple[float, float, float, float]] = []
    step = tile_size - overlap
    for top in range(0, scanning.shape[0], step):
        for left in range(0, scanning.shape[1], step):
            bottom = min(scanning.shape[0], top + tile_size)
            right = min(scanning.shape[1], left + tile_size)
            tile = scanning[top:bottom, left:right]
            if min(tile.shape) < 30:
                continue
            for x, y, radius, quality in _circle_candidates(
                tile,
                minimum_diameter=8,
                maximum_diameter=180,
                adaptive_threshold=True,
            ):
                candidates.append((x + left, y + top, radius, quality))

    candidates.sort(key=lambda item: (item[3], item[2]), reverse=True)
    deduplicated: list[tuple[float, float, float, float]] = []
    for candidate in candidates:
        x, y, radius, _ = candidate
        if any(
            np.hypot(x - other_x, y - other_y) < min(radius, other_radius) * 0.65
            for other_x, other_y, other_radius, _ in deduplicated
        ):
            continue
        deduplicated.append(candidate)

    islands: list[Island] = []
    for x, y, candidate_radius, _ in deduplicated:
        nearby_radii = [
            item[2]
            for item in deduplicated
            if 0.62 <= item[2] / candidate_radius <= 1.62
        ]
        if len(nearby_radii) < 2:
            continue
        typical_radius = float(np.percentile(nearby_radii, 20))
        ocr_scale = max(1.0, 16.0 / typical_radius)
        local_half = max(8, int(np.ceil(max(typical_radius, candidate_radius) * 1.25)))
        center_x, center_y = int(round(x)), int(round(y))
        local_left = max(0, center_x - local_half)
        local_top = max(0, center_y - local_half)
        local_right = min(scanning.shape[1], center_x + local_half + 1)
        local_bottom = min(scanning.shape[0], center_y + local_half + 1)
        local_gray = scanning[local_top:local_bottom, local_left:local_right]
        ocr_gray = (
            cv2.resize(local_gray, None, fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_CUBIC)
            if ocr_scale > 1.0
            else local_gray
        )
        try:
            masks = _extract_digit_scales(
                ocr_gray,
                (x - local_left) * ocr_scale,
                (y - local_top) * ocr_scale,
                typical_radius * ocr_scale,
            )
            value, confidence, scores = _classify_digit_scales(masks)
            valid_distance = min(scores.values())
            sharpness = _local_sharpness(scanning, x, y, candidate_radius)
            invalid = False
            if sharpness >= _MIN_SHARPNESS_FOR_INVALID_GLYPH_CHECK:
                invalid = _invalid_glyph_distance(masks) - valid_distance < _MIN_VALID_GLYPH_ADVANTAGE
            if valid_distance > _MAX_VALID_GLYPH_DISTANCE or invalid:
                continue
        except RecognitionError:
            continue
        inverse = 1.0 / scale
        islands.append(
            Island(
                len(islands),
                x * inverse,
                y * inverse,
                value,
                candidate_radius * inverse,
                confidence,
            )
        )
    return islands


def _aligned_components(islands: list[Island]) -> list[list[Island]]:
    if len(islands) < 2:
        return []
    radius_min = min(item.radius for item in islands)
    radius_max = max(item.radius for item in islands)
    if radius_max / max(1.0, radius_min) <= 1.7:
        spatial = _spatial_clusters(islands)
        if len(spatial) > 1 and sum(len(group) for group in spatial[:2]) >= len(islands) * 0.8:
            return spatial
    return _aligned_size_components(islands)


def _spatial_clusters(islands: list[Island]) -> list[list[Island]]:
    if len(islands) < 4:
        return [islands]
    nearest: list[float] = []
    for index, item in enumerate(islands):
        nearest.append(
            min(
                float(np.hypot(item.x - other.x, item.y - other.y))
                for other_index, other in enumerate(islands)
                if other_index != index
            )
        )
    threshold = max(median(nearest) * 5.5, median(item.radius for item in islands) * 10.0)
    adjacency = [set() for _ in islands]
    for first, item in enumerate(islands):
        for second in range(first + 1, len(islands)):
            other = islands[second]
            if np.hypot(item.x - other.x, item.y - other.y) <= threshold:
                adjacency[first].add(second)
                adjacency[second].add(first)
    clusters: list[list[Island]] = []
    unseen = set(range(len(islands)))
    while unseen:
        start = unseen.pop()
        group = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        clusters.append([islands[index] for index in sorted(group)])
    return [cluster for cluster in clusters if len(cluster) >= 2]


def _aligned_size_components(islands: list[Island]) -> list[list[Island]]:
    typical_radius = median(item.radius for item in islands)
    alignment = max(3.0, typical_radius * 0.85)
    adjacency = [set() for _ in islands]
    row_groups: list[list[int]] = []
    for index in sorted(range(len(islands)), key=lambda item: islands[item].y):
        if not row_groups:
            row_groups.append([index])
            continue
        center = sum(islands[item].y for item in row_groups[-1]) / len(row_groups[-1])
        if abs(islands[index].y - center) <= alignment:
            row_groups[-1].append(index)
        else:
            row_groups.append([index])
    column_groups: list[list[int]] = []
    for index in sorted(range(len(islands)), key=lambda item: islands[item].x):
        if not column_groups:
            column_groups.append([index])
            continue
        center = sum(islands[item].x for item in column_groups[-1]) / len(column_groups[-1])
        if abs(islands[index].x - center) <= alignment:
            column_groups[-1].append(index)
        else:
            column_groups.append([index])
    gap_samples: list[float] = []
    for group, axis in ((row_groups, "x"), (column_groups, "y")):
        for indices in group:
            ordered = sorted(indices, key=lambda item: getattr(islands[item], axis))
            gap_samples.extend(
                getattr(islands[second], axis) - getattr(islands[first], axis)
                for first, second in zip(ordered, ordered[1:])
            )
    positive_gaps = sorted(gap for gap in gap_samples if gap > typical_radius * 2.0)
    typical_gap = median(positive_gaps[: max(1, len(positive_gaps) // 2)]) if positive_gaps else typical_radius * 8
    maximum_gap = max(typical_radius * 8.0, typical_gap * 12.0)
    for group, axis in ((row_groups, "x"), (column_groups, "y")):
        for indices in group:
            ordered = sorted(indices, key=lambda item: getattr(islands[item], axis))
            for position, first in enumerate(ordered):
                for second in ordered[position + 1 :]:
                    radius_ratio = islands[second].radius / max(1.0, islands[first].radius)
                    if not 0.62 <= radius_ratio <= 1.62:
                        continue
                    if (
                        getattr(islands[second], axis) - getattr(islands[first], axis)
                        <= maximum_gap
                    ):
                        adjacency[first].add(second)
                        adjacency[second].add(first)
                    break

    groups: list[list[Island]] = []
    unseen = set(range(len(islands)))
    while unseen:
        start = unseen.pop()
        indices = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    indices.add(neighbor)
                    queue.append(neighbor)
        if len(indices) >= 2:
            groups.append([islands[index] for index in sorted(indices)])
    return groups


def _crop_region(image: np.ndarray, islands: list[Island]) -> ScreenRegion:
    typical_radius = median(item.radius for item in islands)
    margin = max(8, int(round(typical_radius * 2.0)))
    left = max(0, int(np.floor(min(item.x - item.radius for item in islands))) - margin)
    top = max(0, int(np.floor(min(item.y - item.radius for item in islands))) - margin)
    right = min(image.shape[1], int(np.ceil(max(item.x + item.radius for item in islands))) + margin)
    bottom = min(image.shape[0], int(np.ceil(max(item.y + item.radius for item in islands))) + margin)
    return ScreenRegion(left, top, right - left, bottom - top)


def _matches_localization(recognized: list[Island], localized: list[Island], region: ScreenRegion) -> bool:
    # The coarse full-client scan can miss a few islands whose rings overlap
    # Tricky hint boxes, or treat the hint decoration as an island. The tighter
    # crop usually recovers the real set, so tolerate a small symmetric count
    # difference while requiring the smaller set to align one-to-one.
    larger_count = max(len(recognized), len(localized))
    allowed_difference = max(2, (larger_count + 9) // 10)
    if abs(len(recognized) - len(localized)) > allowed_difference:
        return False
    expected_points = np.asarray(
        [[item.x - region.left, item.y - region.top] for item in localized],
        dtype=float,
    )
    actual_points = np.asarray([[item.x, item.y] for item in recognized], dtype=float)
    distances = np.linalg.norm(expected_points[:, None, :] - actual_points[None, :, :], axis=2)
    expected_indices, actual_indices = linear_sum_assignment(distances)
    if len(expected_indices) != min(len(localized), len(recognized)):
        return False
    for expected_index, actual_index in zip(expected_indices, actual_indices):
        expected = localized[expected_index]
        actual = recognized[actual_index]
        if distances[expected_index, actual_index] > max(4.0, expected.radius * 0.8):
            return False
    return True


def locate_board(image: np.ndarray) -> LocatedBoard:
    if image is None or image.size == 0:
        raise RecognitionError("截图为空")

    candidates = _localization_candidates(image)
    accepted: list[tuple[LocatedBoard, float]] = []
    for component in _aligned_components(candidates):
        region = _crop_region(image, component)
        if region.width < 30 or region.height < 30:
            continue
        crop = image[region.top:region.bottom, region.left:region.right].copy()
        try:
            recognition = recognize(crop)
            if not _matches_localization(recognition.islands, component, region):
                continue
            puzzle = build_puzzle(recognition.islands)
        except Exception:
            continue
        confidence = float(
            median(
                [item.confidence for item in recognition.islands]
                + [item.confidence for item in component]
            )
        )
        accepted.append((LocatedBoard(crop, region, recognition, puzzle), confidence))

    if not accepted:
        raise RecognitionError("未能在应用窗口中定位到可信的数桥题面，请改用“框选题面”")
    accepted.sort(
        key=lambda item: (len(item[0].puzzle.islands), item[1]),
        reverse=True,
    )
    best = accepted[0][0]
    if len(accepted) > 1 and len(accepted[1][0].puzzle.islands) >= len(best.puzzle.islands) * 0.75:
        raise RecognitionError("应用窗口中存在多个可信题面，请改用“框选题面”指定目标")
    return best
