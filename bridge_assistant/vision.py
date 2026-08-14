from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import Island, Recognition, RecognitionError


_FONT_CANDIDATES = (
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
)
_VALID_DIGITS = tuple("12345678")
_INVALID_GLYPHS = tuple("09ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_MIN_VALID_GLYPH_ADVANTAGE = -0.02
_MAX_VALID_GLYPH_DISTANCE = 0.52
_MIN_SHARPNESS_FOR_INVALID_GLYPH_CHECK = 5000.0


def _normalize_digit(mask: np.ndarray, size: int = 28) -> np.ndarray | None:
    binary = (mask > 0).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None
    minimum_area = max(2, int(binary.size * 0.003))
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    points = cv2.findNonZero(cleaned)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width < 1 or height < 3:
        return None
    glyph = cleaned[y : y + height, x : x + width]
    usable = size - 6
    scale = min(usable / width, usable / height)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    resized = cv2.resize(glyph, (target_width, target_height), interpolation=cv2.INTER_AREA)
    result = np.zeros((size, size), dtype=np.uint8)
    left = (size - target_width) // 2
    top = (size - target_height) // 2
    result[top : top + target_height, left : left + target_width] = resized
    return result


def _font_paths() -> list[Path]:
    roots = [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts"]
    paths = [root / name for root in roots for name in _FONT_CANDIDATES]
    return [path for path in paths if path.exists()]


@lru_cache(maxsize=2)
def _character_templates(characters: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {character: [] for character in characters}
    for path in _font_paths():
        for font_size in (34, 40, 46):
            font = ImageFont.truetype(str(path), font_size)
            for stroke in (0, 1):
                for character in characters:
                    image = Image.new("L", (64, 64), 0)
                    draw = ImageDraw.Draw(image)
                    box = draw.textbbox((0, 0), character, font=font, stroke_width=stroke)
                    width, height = box[2] - box[0], box[3] - box[1]
                    position = ((64 - width) / 2 - box[0], (64 - height) / 2 - box[1])
                    draw.text(position, character, font=font, fill=255, stroke_width=stroke)
                    normalized = _normalize_digit(np.asarray(image))
                    if normalized is not None:
                        templates[character].append(normalized)

    for font_face in (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ):
        for thickness in (1, 2, 3):
            for character in characters:
                image = np.zeros((64, 64), dtype=np.uint8)
                scale = 1.45
                (width, height), baseline = cv2.getTextSize(character, font_face, scale, thickness)
                origin = ((64 - width) // 2, (64 + height - baseline) // 2)
                cv2.putText(image, character, origin, font_face, scale, 255, thickness, cv2.LINE_AA)
                normalized = _normalize_digit(image)
                if normalized is not None:
                    templates[character].append(normalized)
    if not all(templates.values()):
        raise RecognitionError("No local fonts are available for character recognition")
    return templates


@lru_cache(maxsize=1)
def _digit_templates() -> dict[int, list[np.ndarray]]:
    return {
        int(character): templates
        for character, templates in _character_templates(_VALID_DIGITS).items()
    }


def _mask_score(mask: np.ndarray) -> float:
    normalized = _normalize_digit(mask)
    if normalized is None:
        return -1.0
    fraction = float(np.count_nonzero(mask)) / mask.size
    if not 0.015 <= fraction <= 0.48:
        return -1.0
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return -1.0
    x, y, width, height = cv2.boundingRect(points)
    center_x = x + width / 2
    center_y = y + height / 2
    center_penalty = abs(center_x - mask.shape[1] / 2) / mask.shape[1]
    center_penalty += abs(center_y - mask.shape[0] / 2) / mask.shape[0]
    return height / mask.shape[0] - 0.55 * center_penalty - 0.15 * abs(fraction - 0.17)


def _extract_digit_at_scale(gray: np.ndarray, x: float, y: float, half: int) -> np.ndarray:
    cx, cy = int(round(x)), int(round(y))
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(gray.shape[1], cx + half + 1), min(gray.shape[0], cy + half + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0 or min(patch.shape) < 7:
        raise RecognitionError("An island is too close to the image boundary")
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    _, dark = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, light = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    candidates = [(score, mask) for mask in (dark, light) if (score := _mask_score(mask)) >= 0.0]
    if not candidates:
        raise RecognitionError("Could not isolate an island digit at this scale")
    _, best = max(candidates, key=lambda item: item[0])
    normalized = _normalize_digit(best)
    if normalized is None:
        raise RecognitionError("Could not isolate an island digit")
    return normalized


def _extract_digit_scales(
    gray: np.ndarray,
    x: float,
    y: float,
    typical_radius: float,
) -> list[np.ndarray]:
    # Thresholding can trace either side of the anti-aliased circle rim, so
    # per-contour radii are not reliable OCR crop sizes.  Use the board-wide
    # typical radius and combine several crops that all stay inside the rim.
    half_sizes = sorted(
        {
            max(5, int(round(typical_radius * ratio)))
            for ratio in (0.50, 0.60, 0.70)
        }
    )
    masks: list[np.ndarray] = []
    for half in half_sizes:
        try:
            masks.append(_extract_digit_at_scale(gray, x, y, half))
        except RecognitionError:
            continue
    if not masks:
        raise RecognitionError("Could not isolate an island digit")
    return masks


@lru_cache(maxsize=1)
def _hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor((32, 32), (8, 8), (4, 4), (4, 4), 9)


def _hog_vector(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_AREA)
    return _hog().compute(resized).reshape(-1)


@lru_cache(maxsize=1)
def _template_hog_vectors() -> dict[int, np.ndarray]:
    return {
        digit: _normalized_hog_matrix(templates)
        for digit, templates in _digit_templates().items()
    }


def _normalized_hog_matrix(templates: list[np.ndarray]) -> np.ndarray:
    vectors = np.stack([_hog_vector(template) for template in templates])
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)


@lru_cache(maxsize=1)
def _invalid_template_hog_vectors() -> dict[str, np.ndarray]:
    return {
        character: _normalized_hog_matrix(templates)
        for character, templates in _character_templates(_INVALID_GLYPHS).items()
    }


def _normalized_hog_vector(mask: np.ndarray) -> np.ndarray:
    vector = _hog_vector(mask)
    return vector / (float(np.linalg.norm(vector)) + 1e-9)


def _template_scores(
    vector: np.ndarray,
    templates: dict[int | str, np.ndarray],
) -> dict[int | str, float]:
    return {
        label: 1.0 - float(np.max(matrix @ vector))
        for label, matrix in templates.items()
    }


def _rank_digit_scores(scores: dict[int, float]) -> tuple[int, float, dict[int, float]]:
    ranked = sorted(scores.items(), key=lambda item: item[1])
    best_digit, best_distance = ranked[0]
    second_distance = ranked[1][1]
    shape_quality = np.clip((0.62 - best_distance) / 0.52, 0.0, 1.0)
    margin_quality = np.clip((second_distance - best_distance) / 0.12, 0.0, 1.0)
    confidence = float(0.55 * shape_quality + 0.45 * margin_quality)
    return best_digit, confidence, scores


def classify_digit(mask: np.ndarray) -> tuple[int, float, dict[int, float]]:
    scores = _template_scores(_normalized_hog_vector(mask), _template_hog_vectors())
    return _rank_digit_scores(scores)


def _classify_digit_scales(masks: list[np.ndarray]) -> tuple[int, float, dict[int, float]]:
    vectors = [_normalized_hog_vector(mask) for mask in masks]
    scale_scores = [_template_scores(vector, _template_hog_vectors()) for vector in vectors]
    combined = {
        digit: float(np.mean([scores[digit] for scores in scale_scores]))
        for digit in range(1, 9)
    }
    return _rank_digit_scores(combined)


def _invalid_glyph_distance(masks: list[np.ndarray]) -> float:
    scale_scores = [
        _template_scores(_normalized_hog_vector(mask), _invalid_template_hog_vectors())
        for mask in masks
    ]
    return min(
        float(np.mean([scores[character] for scores in scale_scores]))
        for character in _INVALID_GLYPHS
    )


def _local_sharpness(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> float:
    half = max(4, int(round(radius * 0.72)))
    center_x, center_y = int(round(x)), int(round(y))
    patch = gray[
        max(0, center_y - half) : min(gray.shape[0], center_y + half + 1),
        max(0, center_x - half) : min(gray.shape[1], center_x + half + 1),
    ]
    if patch.size == 0:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def _circle_candidates(
    gray: np.ndarray,
    *,
    minimum_diameter: int | None = None,
    maximum_diameter: int | None = None,
    adaptive_threshold: bool = False,
) -> list[tuple[float, float, float, float]]:
    height, width = gray.shape
    min_diameter = (
        max(8, int(minimum_diameter))
        if minimum_diameter is not None
        else max(12, int(min(height, width) * 0.012))
    )
    max_diameter = (
        max(min_diameter + 2, int(maximum_diameter))
        if maximum_diameter is not None
        else max(min_diameter + 2, min(180, int(min(height, width) * 0.28)))
    )
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates: list[tuple[float, float, float, float]] = []

    binaries: list[np.ndarray] = []
    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, binary = cv2.threshold(blurred, 0, 255, polarity | cv2.THRESH_OTSU)
        binaries.append(binary)
    if adaptive_threshold:
        block_size = max(15, min(81, (min(height, width) // 18) | 1))
        for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            binaries.append(
                cv2.adaptiveThreshold(
                    blurred,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    polarity,
                    block_size,
                    5,
                )
            )

    for binary in binaries:
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if not (min_diameter <= box_width <= max_diameter and min_diameter <= box_height <= max_diameter):
                continue
            aspect = box_width / box_height
            if not 0.78 <= aspect <= 1.28:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            fill = area / (box_width * box_height)
            if circularity < 0.68 or not 0.48 <= fill <= 0.88:
                continue
            center, radius = cv2.minEnclosingCircle(contour)
            if radius <= 0:
                continue
            quality = float(circularity - abs(aspect - 1.0) * 0.35)
            candidates.append((float(center[0]), float(center[1]), float(radius), quality))

    if len(candidates) < 2:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, min_diameter),
            param1=100,
            param2=22,
            minRadius=max(5, min_diameter // 2),
            maxRadius=max_diameter // 2,
        )
        if circles is not None:
            candidates.extend((float(x), float(y), float(radius), 0.65) for x, y, radius in circles[0])

    candidates.sort(key=lambda item: (item[3], item[2]), reverse=True)
    deduplicated: list[tuple[float, float, float, float]] = []
    for candidate in candidates:
        x, y, radius, _ = candidate
        if any(np.hypot(x - ox, y - oy) < min(radius, other_radius) * 0.55 for ox, oy, other_radius, _ in deduplicated):
            continue
        deduplicated.append(candidate)
    return deduplicated


def _typical_inner_radius(candidates: list[tuple[float, float, float, float]]) -> float:
    return float(np.percentile([candidate[2] for candidate in candidates], 20))


def _hough_circle_candidates(
    gray: np.ndarray,
    typical_radius: float,
) -> list[tuple[float, float, float, float]]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(7, int(round(typical_radius * 1.2))),
        param1=80,
        param2=max(12, int(round(typical_radius * 1.65))),
        minRadius=max(4, int(typical_radius * 0.62)),
        maxRadius=max(5, int(np.ceil(typical_radius * 1.55))),
    )
    if circles is None:
        return []
    return [
        (float(x), float(y), float(radius), 0.65)
        for x, y, radius in circles[0]
    ]


def _ring_support(
    gray_shape: tuple[int, int],
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    magnitude: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> float:
    x1 = max(0, int(x - radius * 1.5))
    y1 = max(0, int(y - radius * 1.5))
    x2 = min(gray_shape[1], int(x + radius * 1.5 + 1))
    y2 = min(gray_shape[0], int(y + radius * 1.5 + 1))
    local = magnitude[y1:y2, x1:x2]
    threshold = max(12.0, float(np.percentile(local, 62)) * 0.55) if local.size else 12.0
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radial_samples = np.linspace(0.68, 1.36, 15)
    supported = 0

    for angle in angles:
        cosine, sine = np.cos(angle), np.sin(angle)
        xs = np.rint(x + cosine * radius * radial_samples).astype(int)
        ys = np.rint(y + sine * radius * radial_samples).astype(int)
        valid = (
            (xs >= 0)
            & (xs < gray_shape[1])
            & (ys >= 0)
            & (ys < gray_shape[0])
        )
        if not np.any(valid):
            continue
        valid_xs, valid_ys = xs[valid], ys[valid]
        strengths = magnitude[valid_ys, valid_xs]
        strongest = int(np.argmax(strengths))
        strength = float(strengths[strongest])
        dx = float(gradient_x[valid_ys[strongest], valid_xs[strongest]])
        dy = float(gradient_y[valid_ys[strongest], valid_xs[strongest]])
        radial_alignment = abs(dx * cosine + dy * sine) / (strength + 1e-6)
        if strength >= threshold and radial_alignment >= 0.42:
            supported += 1
    return supported / len(angles)


def _merge_circle_candidates(
    gray: np.ndarray,
    contour_candidates: list[tuple[float, float, float, float]],
    typical_radius: float,
) -> list[tuple[float, float, float, float]]:
    merged = list(contour_candidates)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)

    for candidate in _hough_circle_candidates(gray, typical_radius):
        x, y, radius, _ = candidate
        if any(
            np.hypot(x - other_x, y - other_y) < min(radius, other_radius) * 0.65
            for other_x, other_y, other_radius, _ in merged
        ):
            continue
        if _ring_support(
            gray.shape,
            gradient_x,
            gradient_y,
            magnitude,
            x,
            y,
            radius,
        ) >= 0.78:
            merged.append(candidate)
    return merged


def _dominant_size(candidates: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    if len(candidates) < 2:
        return candidates
    best_group: list[tuple[float, float, float, float]] = []
    best_score = -1.0
    for seed in candidates:
        radius = seed[2]
        # Anti-aliased rims can alternate between inner and outer contours.
        # Both radius modes belong to the same physical circle, especially
        # after downscaling or blur, where the modes can differ by about 35%.
        group = [
            candidate
            for candidate in candidates
            if 0.62 <= candidate[2] / radius <= 1.62
        ]
        if len(group) < 2:
            continue
        typical = median(candidate[2] for candidate in group)
        score = len(group) * (typical ** 0.32) * (0.75 + np.mean([item[3] for item in group]))
        if score > best_score:
            best_group, best_score = group, score
    return best_group or candidates


def _recognize_at_scale(gray: np.ndarray) -> tuple[list[tuple[float, float, float, int, float]], int]:
    contour_candidates = _dominant_size(_circle_candidates(gray))
    if len(contour_candidates) < 2:
        raise RecognitionError("Fewer than two circular islands were found; crop tightly around the board")

    typical_radius = _typical_inner_radius(contour_candidates)
    candidates = _merge_circle_candidates(gray, contour_candidates, typical_radius)
    ocr_scale = max(1.0, 16.0 / typical_radius)
    ocr_gray = (
        cv2.resize(gray, None, fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_CUBIC)
        if ocr_scale > 1.0
        else gray
    )
    raw: list[tuple[float, float, float, int, float]] = []
    failures = 0
    for x, y, radius, _ in candidates:
        try:
            masks = _extract_digit_scales(
                ocr_gray,
                x * ocr_scale,
                y * ocr_scale,
                typical_radius * ocr_scale,
            )
            value, confidence, scores = _classify_digit_scales(masks)
            valid_distance = min(scores.values())
            sharpness = _local_sharpness(gray, x, y, radius)
            invalid_glyph_is_closer = False
            if sharpness >= _MIN_SHARPNESS_FOR_INVALID_GLYPH_CHECK:
                invalid_distance = _invalid_glyph_distance(masks)
                invalid_glyph_is_closer = (
                    invalid_distance - valid_distance < _MIN_VALID_GLYPH_ADVANTAGE
                )
            if valid_distance > _MAX_VALID_GLYPH_DISTANCE or invalid_glyph_is_closer:
                failures += 1
                continue
            raw.append((x, y, radius, value, confidence))
        except RecognitionError:
            failures += 1
    return raw, failures


def _sort_islands_by_rows(
    raw: list[tuple[float, float, float, int, float]],
) -> list[tuple[float, float, float, int, float]]:
    if not raw:
        return []
    row_tolerance = max(3.0, median(item[2] for item in raw) * 0.48)
    rows: list[list[tuple[float, float, float, int, float]]] = []
    for item in sorted(raw, key=lambda candidate: candidate[1]):
        if not rows:
            rows.append([item])
            continue
        row_center = sum(candidate[1] for candidate in rows[-1]) / len(rows[-1])
        if abs(item[1] - row_center) <= row_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [
        item
        for row in rows
        for item in sorted(row, key=lambda candidate: candidate[0])
    ]


def recognize(image: np.ndarray) -> Recognition:
    if image is None or image.size == 0:
        raise RecognitionError("The screenshot is empty")
    if image.ndim == 2:
        gray = image.copy()
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
        color = image.copy()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    initial_candidates = _dominant_size(_circle_candidates(gray))
    if len(initial_candidates) < 2:
        raise RecognitionError("Fewer than two circular islands were found; crop tightly around the board")

    initial_radius = _typical_inner_radius(initial_candidates)
    board_scale = min(3.0, 18.0 / initial_radius) if initial_radius < 10.5 else 1.0
    working_gray = (
        cv2.resize(gray, None, fx=board_scale, fy=board_scale, interpolation=cv2.INTER_CUBIC)
        if board_scale > 1.0
        else gray
    )
    scaled_raw, failures = _recognize_at_scale(working_gray)
    raw = [
        (
            x / board_scale,
            y / board_scale,
            radius / board_scale,
            value,
            confidence,
        )
        for x, y, radius, value, confidence in scaled_raw
    ]

    if len(raw) < 2:
        raise RecognitionError("Island circles were found, but their digits could not be read")
    raw = _sort_islands_by_rows(raw)
    islands = [
        Island(index, x, y, value, radius, confidence)
        for index, (x, y, radius, value, confidence) in enumerate(raw)
    ]
    warnings: list[str] = []
    low = [island.id + 1 for island in islands if island.confidence < 0.58]
    if low:
        warnings.append("Low-confidence digits: " + ", ".join(map(str, low)))
    if failures:
        warnings.append(f"Ignored {failures} circle candidates without readable digits")
    return Recognition(islands, color, warnings)
