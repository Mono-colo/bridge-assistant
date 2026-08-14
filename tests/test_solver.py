from __future__ import annotations

import pytest

from bridge_assistant.graph import build_puzzle, crossing_pairs
from bridge_assistant.model import Island, SolveError
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

