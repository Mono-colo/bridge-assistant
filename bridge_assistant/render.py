from __future__ import annotations

import cv2
import numpy as np

from .model import Puzzle, Solution


def render_overlay(
    image: np.ndarray,
    puzzle: Puzzle | None = None,
    solution: Solution | None = None,
) -> np.ndarray:
    canvas = image.copy()
    if puzzle is None:
        return canvas
    by_id = {island.id: island for island in puzzle.islands}

    if solution is not None:
        for edge in puzzle.edges:
            count = solution.bridges.get(edge.id, 0)
            if count <= 0:
                continue
            first, second = by_id[edge.a], by_id[edge.b]
            start = (round(first.x), round(first.y))
            end = (round(second.x), round(second.y))
            color = (32, 118, 224)
            if count == 1:
                cv2.line(canvas, start, end, color, max(2, round(first.radius * 0.13)), cv2.LINE_AA)
            else:
                offset = max(2, round(first.radius * 0.16))
                if edge.orientation == "h":
                    starts = ((start[0], start[1] - offset), (start[0], start[1] + offset))
                    ends = ((end[0], end[1] - offset), (end[0], end[1] + offset))
                else:
                    starts = ((start[0] - offset, start[1]), (start[0] + offset, start[1]))
                    ends = ((end[0] - offset, end[1]), (end[0] + offset, end[1]))
                for line_start, line_end in zip(starts, ends):
                    cv2.line(canvas, line_start, line_end, color, max(2, round(first.radius * 0.1)), cv2.LINE_AA)

    for island in puzzle.islands:
        center = (round(island.x), round(island.y))
        radius = max(7, round(island.radius))
        color = (37, 170, 75) if island.confidence >= 0.58 else (0, 148, 255)
        cv2.circle(canvas, center, radius + 3, color, 2, cv2.LINE_AA)
        label = str(island.value)
        scale = max(0.45, radius / 24.0)
        thickness = max(1, round(scale * 2))
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        origin = (center[0] - width // 2, center[1] + height // 2)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (25, 25, 25), thickness, cv2.LINE_AA)
    return canvas

