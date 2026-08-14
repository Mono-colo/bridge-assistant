from __future__ import annotations

import pytest

from bridge_assistant.graph import build_puzzle, crossing_pairs
from bridge_assistant.model import (
    BorderHint,
    BorderHintType,
    Island,
    MagicLink,
    Puzzle,
    Solution,
    SolveError,
)
from bridge_assistant.solver import solve, validate_solution


def puzzle_from(items: list[tuple[float, float, int]]):
    islands = [Island(index, x, y, value, 10) for index, (x, y, value) in enumerate(items)]
    return build_puzzle(islands)


def test_two_islands_get_double_bridge() -> None:
    puzzle = puzzle_from([(10, 10, 2), (60, 10, 2)])
    solution = solve(puzzle)
    assert list(solution.bridges.values()) == [2]
    validate_solution(puzzle, solution)


def test_solver_returns_connected_square() -> None:
    puzzle = puzzle_from(
        [
            (10, 10, 2),
            (60, 10, 2),
            (10, 60, 2),
            (60, 60, 2),
        ]
    )
    solution = solve(puzzle)
    assert solution.active_count == 4
    assert sum(solution.bridges.values()) == 4
    validate_solution(puzzle, solution)


def test_crossing_candidates_are_detected() -> None:
    puzzle = puzzle_from(
        [
            (10, 40, 1),
            (70, 40, 1),
            (40, 10, 1),
            (40, 70, 1),
        ]
    )
    assert len(crossing_pairs(puzzle)) == 1
    with pytest.raises(SolveError):
        solve(puzzle)


def test_disconnected_degree_solution_is_rejected() -> None:
    # Degree equations alone permit two disconnected double-bridge pairs.
    puzzle = puzzle_from(
        [
            (10, 10, 2),
            (60, 10, 2),
            (10, 60, 2),
            (60, 60, 2),
        ]
    )
    # Reduce candidate graph to the two horizontal pairs, preserving a
    # deliberately disconnected instance.
    puzzle.edges = [edge for edge in puzzle.edges if edge.orientation == "h"]
    for index, edge in enumerate(puzzle.edges):
        object.__setattr__(edge, "id", index)
    with pytest.raises(SolveError):
        solve(puzzle)


def test_impossible_clue_is_reported() -> None:
    puzzle = puzzle_from([(10, 10, 3), (60, 10, 3)])
    with pytest.raises(SolveError, match="impossible degree"):
        solve(puzzle)


def test_puzzle_constraint_fields_preserve_positional_construction() -> None:
    puzzle = Puzzle([], [], 1.25)
    assert puzzle.alignment_tolerance == 1.25
    assert puzzle.magic_links == []
    assert puzzle.border_hints == []
    assert [item.value for item in BorderHintType] == [0, 1, 2]


def test_magic_link_is_enforced_by_solver_and_validator() -> None:
    puzzle = puzzle_from(
        [
            (10, 10, 1),
            (60, 10, 1),
            (110, 10, 2),
            (10, 60, 2),
            (60, 60, 3),
            (110, 60, 3),
        ]
    )
    puzzle.magic_links.append(MagicLink(1, 2))

    solution = solve(puzzle)

    assert solution.bridges == {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 2}
    validate_solution(puzzle, solution)

    unequal = Solution({0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 1, 6: 1})
    with pytest.raises(SolveError, match="Magic link"):
        validate_solution(puzzle, unequal)


@pytest.mark.parametrize(
    ("hint_type", "hint", "expected_horizontal_sum"),
    [
        (BorderHintType.EQ, 4, 4),
        (BorderHintType.GTE, 4, 4),
        (BorderHintType.LTE, 2, 2),
    ],
)
def test_border_hint_is_enforced_by_solver(
    hint_type: BorderHintType,
    hint: int,
    expected_horizontal_sum: int,
) -> None:
    puzzle = puzzle_from(
        [
            (10, 10, 3),
            (60, 10, 3),
            (10, 60, 3),
            (60, 60, 3),
        ]
    )
    puzzle.border_hints.append(BorderHint((0, 3), hint, hint_type))

    solution = solve(puzzle)

    assert solution.bridges[0] + solution.bridges[3] == expected_horizontal_sum
    validate_solution(puzzle, solution)


@pytest.mark.parametrize(
    ("hint_type", "hint", "valid", "invalid"),
    [
        (
            BorderHintType.EQ,
            4,
            {0: 2, 1: 1, 2: 1, 3: 2},
            {0: 1, 1: 2, 2: 2, 3: 1},
        ),
        (
            BorderHintType.GTE,
            4,
            {0: 2, 1: 1, 2: 1, 3: 2},
            {0: 1, 1: 2, 2: 2, 3: 1},
        ),
        (
            BorderHintType.LTE,
            2,
            {0: 1, 1: 2, 2: 2, 3: 1},
            {0: 2, 1: 1, 2: 1, 3: 2},
        ),
    ],
)
def test_border_hint_is_enforced_by_validator(
    hint_type: BorderHintType,
    hint: int,
    valid: dict[int, int],
    invalid: dict[int, int],
) -> None:
    puzzle = puzzle_from(
        [
            (10, 10, 3),
            (60, 10, 3),
            (10, 60, 3),
            (60, 60, 3),
        ]
    )
    puzzle.border_hints.append(BorderHint((0, 3), hint, hint_type))

    validate_solution(puzzle, Solution(valid))
    with pytest.raises(SolveError, match="Border hint"):
        validate_solution(puzzle, Solution(invalid))


@pytest.mark.parametrize(
    "border_hint",
    [
        BorderHint((), 0),
        BorderHint((0, 0), 0),
        BorderHint((99,), 0),
    ],
)
def test_invalid_border_hint_edges_are_rejected(border_hint: BorderHint) -> None:
    puzzle = puzzle_from([(10, 10, 2), (60, 10, 2)])
    puzzle.border_hints.append(border_hint)
    with pytest.raises(SolveError, match="Border hint"):
        solve(puzzle)
