from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from bridge_assistant.automation import build_mouse_actions
from bridge_assistant.graph import build_puzzle
from bridge_assistant.model import InputMode, ScreenRegion
from bridge_assistant.solver import solve
from bridge_assistant.vision import recognize


def synthetic_board() -> tuple[np.ndarray, list[int]]:
    image = np.full((250, 250, 3), 248, dtype=np.uint8)
    items = [
        (50, 50, 2),
        (200, 50, 2),
        (50, 200, 2),
        (200, 200, 2),
    ]
    for x, y, value in items:
        cv2.circle(image, (x, y), 19, (10, 10, 10), 2, cv2.LINE_AA)
        text = str(value)
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(
            image,
            text,
            (x - width // 2, y + height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
    return image, [2, 2, 2, 2]


def assert_spatially_equal(
    reference,
    actual,
    *,
    coordinate_scale: float = 1.0,
) -> None:
    assert len(actual) == len(reference)
    expected_points = np.asarray(
        [[island.x * coordinate_scale, island.y * coordinate_scale] for island in reference]
    )
    actual_points = np.asarray([[island.x, island.y] for island in actual])
    distances = np.linalg.norm(expected_points[:, None, :] - actual_points[None, :, :], axis=2)
    expected_indices, actual_indices = linear_sum_assignment(distances)
    maximum_distance = max(
        distances[expected_index, actual_index]
        for expected_index, actual_index in zip(expected_indices, actual_indices)
    )
    typical_radius = float(np.median([island.radius for island in actual]))
    assert maximum_distance < max(3.0, typical_radius * 0.55)
    assert [reference[index].value for index in expected_indices] == [
        actual[index].value for index in actual_indices
    ]


def test_recognize_and_solve_synthetic_board() -> None:
    image, values = synthetic_board()
    recognition = recognize(image)
    assert len(recognition.islands) == 4
    assert [island.value for island in recognition.islands] == values
    puzzle = build_puzzle(recognition.islands)
    solution = solve(puzzle)
    assert solution.active_count == 4


def test_recognize_requires_a_circle_with_a_valid_digit() -> None:
    image = np.full((360, 900, 3), 245, dtype=np.uint8)
    circled = [(70, "1"), (170, "3"), (270, "8"), (390, ""), (490, "0"), (590, "9"), (690, "X"), (790, "A")]
    for x, character in circled:
        cv2.circle(image, (x, 100), 28, (10, 10, 10), 2, cv2.LINE_AA)
        if character:
            (width, height), _ = cv2.getTextSize(character, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cv2.putText(
                image,
                character,
                (x - width // 2, 100 + height // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )

    for x, character in ((100, "2"), (250, "5"), (400, "6"), (550, "7"), (700, "8")):
        (width, height), _ = cv2.getTextSize(character, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.putText(
            image,
            character,
            (x - width // 2, 260 + height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

    recognition = recognize(image)
    assert [(round(island.x), island.value) for island in recognition.islands] == [
        (70, 1),
        (170, 3),
        (270, 8),
    ]
    assert recognition.warnings == ["Ignored 5 circle candidates without readable digits"]


def test_recognize_real_puzzle_bridges_capture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "puzzle_bridges_7x7.png"
    image = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
    recognition = recognize(image)
    assert len(recognition.islands) == 14
    assert [island.value for island in recognition.islands] == [
        3,
        4,
        2,
        2,
        4,
        5,
        3,
        1,
        6,
        3,
        1,
        3,
        4,
        1,
    ]
    puzzle = build_puzzle(recognition.islands)
    solution = solve(puzzle)
    assert solution.active_count == 13
    assert sum(solution.bridges.values()) == 21

    actions = build_mouse_actions(
        puzzle,
        solution,
        ScreenRegion(0, 0, image.shape[1], image.shape[0]),
    )
    active_edges = [edge for edge in puzzle.edges if solution.bridges[edge.id] > 0]
    by_id = {island.id: island for island in puzzle.islands}
    assert len(actions) == len(active_edges) == 13
    assert sum(action.bridge_count for action in actions) == 21
    for action, edge in zip(actions, active_edges):
        first, second = by_id[edge.a], by_id[edge.b]
        distance_to_first = math.dist(action.start, first.point)
        distance_to_second = math.dist(action.start, second.point)
        edge_length = math.dist(first.point, second.point)
        assert action.mode == InputMode.NEAR_ENDPOINT
        assert action.start == action.end
        assert distance_to_first + distance_to_second == pytest.approx(edge_length, abs=1.0)
        assert min(distance_to_first, distance_to_second) < edge_length * 0.43
        assert all(
            math.dist(action.start, island.point) > island.radius
            for island in puzzle.islands
        )


@pytest.mark.parametrize(
    ("filename", "values", "candidate_edges", "active_edges", "bridge_count"),
    [
        (
            "66ce9e45-39ff-49a8-b427-34fa883d98a6.png",
            [
                3, 5, 1, 2, 1, 2, 6, 4, 3, 2, 4, 3, 1, 3, 4, 5, 6, 2, 3, 1,
                2, 1, 5, 2, 1, 3, 2, 2, 5, 3, 3, 1, 3, 4, 7, 4, 3, 3, 5, 4,
                3, 7, 5, 3, 2, 3, 7, 6, 3, 3, 2, 3, 1, 6, 5, 2, 2, 2, 3, 2,
            ],
            77,
            63,
            97,
        ),
        (
            "b2e81ea3-a3d6-4513-87d6-8df3169faef0.png",
            [
                2, 4, 1, 2, 4, 3, 1, 3, 2, 4, 3, 2, 3, 5, 3, 3, 4, 7, 5, 3,
                2, 1, 4, 3, 4, 3, 4, 4, 3, 4, 3, 2, 4, 3, 5, 4, 3, 7, 4, 6,
                4, 1, 2, 5, 3, 4, 1, 2, 4, 3, 3, 2, 5, 2, 2, 2, 4, 1, 3, 2,
            ],
            79,
            61,
            96,
        ),
        (
            "cb476f14-c682-4b98-80d5-627605a71474.png",
            [
                4, 5, 4, 2, 3, 2, 2, 5, 1, 2, 3, 4, 7, 3, 1, 4, 4, 3, 4, 3,
                2, 2, 5, 8, 7, 2, 2, 2, 2, 3, 1, 3, 3, 2, 2, 3, 3, 4, 1, 3,
                1, 4, 3, 7, 6, 4, 2, 2, 3, 4, 2, 3, 3, 3, 3, 4, 4, 1, 3, 2,
            ],
            79,
            61,
            95,
        ),
    ],
)
def test_recognize_large_small_font_boards(
    filename: str,
    values: list[int],
    candidate_edges: int,
    active_edges: int,
    bridge_count: int,
) -> None:
    fixture = Path(__file__).parents[1] / "test_image" / filename
    image = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
    recognition = recognize(image)
    assert len(recognition.islands) == len(values) == 60
    assert [island.value for island in recognition.islands] == values
    puzzle = build_puzzle(recognition.islands)
    assert len(puzzle.edges) == candidate_edges
    solution = solve(puzzle)
    assert solution.active_count == active_edges
    assert sum(solution.bridges.values()) == bridge_count


@pytest.mark.parametrize(
    "filename",
    [
        "66ce9e45-39ff-49a8-b427-34fa883d98a6.png",
        "b2e81ea3-a3d6-4513-87d6-8df3169faef0.png",
        "cb476f14-c682-4b98-80d5-627605a71474.png",
        "4.png",
    ],
)
def test_recognize_circle_and_digit_under_image_degradation(filename: str) -> None:
    fixture = Path(__file__).parents[1] / "test_image" / filename
    image = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
    reference = recognize(image).islands

    downscaled = cv2.resize(image, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA)
    assert_spatially_equal(reference, recognize(downscaled).islands, coordinate_scale=0.65)

    blurred = cv2.GaussianBlur(image, (0, 0), 1.4)
    assert_spatially_equal(reference, recognize(blurred).islands)

    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 45])
    assert encoded
    compressed = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    assert_spatially_equal(reference, recognize(compressed).islands)


def test_recognize_and_build_actions_for_midpoint_fallback_board() -> None:
    fixture = Path(__file__).parents[1] / "test_image" / "4.png"
    image = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
    recognition = recognize(image)
    assert len(recognition.islands) == 60
    assert [island.value for island in recognition.islands] == [
        2, 3, 4, 5, 4, 3, 2, 3, 3, 3, 4, 2, 2, 2, 4, 3, 5, 3, 4, 1,
        3, 4, 4, 4, 4, 3, 3, 4, 3, 4, 4, 1, 4, 2, 1, 3, 2, 2, 5, 2,
        4, 2, 2, 2, 4, 5, 4, 3, 2, 3, 3, 2, 2, 2, 1, 3, 4, 3, 2, 3,
    ]

    puzzle = build_puzzle(recognition.islands)
    assert len(puzzle.edges) == 77
    solution = solve(puzzle)
    assert solution.active_count == 62
    assert sum(solution.bridges.values()) == 90

    actions = build_mouse_actions(
        puzzle,
        solution,
        ScreenRegion(0, 0, image.shape[1], image.shape[0]),
    )
    active_edges = [edge for edge in puzzle.edges if solution.bridges[edge.id] > 0]
    assert len(actions) == len(active_edges) == 62

    by_id = {island.id: island for island in puzzle.islands}
    fallback = next(
        action
        for action, edge in zip(actions, active_edges)
        if by_id[edge.a].value == by_id[edge.b].value == 3
        and max(by_id[edge.a].y, by_id[edge.b].y) < 120
    )
    assert fallback.mode == InputMode.NEAR_ENDPOINT
    assert fallback.start == fallback.end == (437, 101)
