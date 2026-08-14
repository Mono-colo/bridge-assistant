from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .graph import crossing_pairs
from .model import BorderHint, BorderHintType, Puzzle, Solution, SolveError


def _border_hint_type(border_hint: BorderHint) -> BorderHintType:
    try:
        return BorderHintType(border_hint.type)
    except (TypeError, ValueError) as error:
        raise SolveError(f"Unsupported border hint type: {border_hint.type!r}") from error


def _validate_input(puzzle: Puzzle) -> None:
    if len(puzzle.islands) < 2:
        raise SolveError("At least two islands are required")
    if any(not 1 <= island.value <= 8 for island in puzzle.islands):
        raise SolveError("Island values must be between 1 and 8")
    island_ids = {island.id for island in puzzle.islands}
    if len(island_ids) != len(puzzle.islands):
        raise SolveError("Island ids must be unique")
    edge_ids = {edge.id for edge in puzzle.edges}
    if edge_ids != set(range(len(puzzle.edges))):
        raise SolveError("Edge ids must be contiguous from zero")
    incident = {island.id: 0 for island in puzzle.islands}
    for edge in puzzle.edges:
        if edge.a not in island_ids or edge.b not in island_ids:
            raise SolveError(f"Edge {edge.id} references an unknown island")
        incident[edge.a] += 1
        incident[edge.b] += 1
    impossible = [
        island.id
        for island in puzzle.islands
        if island.value > 2 * incident[island.id]
    ]
    if impossible:
        joined = ", ".join(str(item + 1) for item in impossible)
        raise SolveError(f"Islands with impossible degree: {joined}")

    for index, magic_link in enumerate(puzzle.magic_links):
        if magic_link.edge_a == magic_link.edge_b:
            raise SolveError(f"Magic link {index + 1} must reference two different edges")
        if magic_link.edge_a not in edge_ids or magic_link.edge_b not in edge_ids:
            raise SolveError(f"Magic link {index + 1} references an unknown edge")

    for index, border_hint in enumerate(puzzle.border_hints):
        if not border_hint.edge_ids:
            raise SolveError(f"Border hint {index + 1} must reference at least one edge")
        if len(set(border_hint.edge_ids)) != len(border_hint.edge_ids):
            raise SolveError(f"Border hint {index + 1} contains duplicate edges")
        if any(edge_id not in edge_ids for edge_id in border_hint.edge_ids):
            raise SolveError(f"Border hint {index + 1} references an unknown edge")
        if border_hint.hint < 0:
            raise SolveError(f"Border hint {index + 1} cannot be negative")
        _border_hint_type(border_hint)


def solve(puzzle: Puzzle) -> Solution:
    """Solve Hashi as a mixed-integer flow model.

    Bridge variables enforce the island clues. Binary edge variables enforce
    crossings, while a single-commodity flow proves that all islands belong
    to one connected component.
    """

    _validate_input(puzzle)
    islands = puzzle.islands
    edges = puzzle.edges
    edge_count = len(edges)
    island_count = len(islands)
    if edge_count == 0:
        raise SolveError("No aligned neighboring islands were found")

    # [bridge_count(E), active(E), flow_a_to_b(E), flow_b_to_a(E)]
    variable_count = 4 * edge_count
    bridge_offset = 0
    active_offset = edge_count
    flow_ab_offset = 2 * edge_count
    flow_ba_offset = 3 * edge_count

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: dict[int, float], low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(low)
        upper.append(high)

    for island in islands:
        coefficients = {
            bridge_offset + edge.id: 1.0
            for edge in edges
            if island.id in (edge.a, edge.b)
        }
        add(coefficients, float(island.value), float(island.value))

    for edge in edges:
        x = bridge_offset + edge.id
        y = active_offset + edge.id
        add({x: 1.0, y: -1.0}, 0.0, np.inf)
        add({x: 1.0, y: -2.0}, -np.inf, 0.0)
        capacity = float(island_count - 1)
        add({flow_ab_offset + edge.id: 1.0, y: -capacity}, -np.inf, 0.0)
        add({flow_ba_offset + edge.id: 1.0, y: -capacity}, -np.inf, 0.0)

    for first, second in crossing_pairs(puzzle):
        add(
            {active_offset + first: 1.0, active_offset + second: 1.0},
            -np.inf,
            1.0,
        )

    for magic_link in puzzle.magic_links:
        add(
            {
                bridge_offset + magic_link.edge_a: 1.0,
                bridge_offset + magic_link.edge_b: -1.0,
            },
            0.0,
            0.0,
        )

    for border_hint in puzzle.border_hints:
        coefficients = {
            bridge_offset + edge_id: 1.0
            for edge_id in border_hint.edge_ids
        }
        hint = float(border_hint.hint)
        hint_type = _border_hint_type(border_hint)
        if hint_type == BorderHintType.EQ:
            add(coefficients, hint, hint)
        elif hint_type == BorderHintType.GTE:
            add(coefficients, hint, np.inf)
        else:
            add(coefficients, -np.inf, hint)

    root_id = islands[0].id
    for island in islands:
        coefficients: dict[int, float] = {}
        for edge in edges:
            if edge.a == island.id:
                coefficients[flow_ba_offset + edge.id] = 1.0
                coefficients[flow_ab_offset + edge.id] = -1.0
            elif edge.b == island.id:
                coefficients[flow_ab_offset + edge.id] = 1.0
                coefficients[flow_ba_offset + edge.id] = -1.0
        balance = -(island_count - 1) if island.id == root_id else 1.0
        add(coefficients, float(balance), float(balance))

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value

    objective = np.zeros(variable_count, dtype=float)
    objective[active_offset : active_offset + edge_count] = 1e-4
    integrality = np.zeros(variable_count, dtype=int)
    integrality[: 2 * edge_count] = 1
    lb = np.zeros(variable_count, dtype=float)
    ub = np.empty(variable_count, dtype=float)
    ub[:edge_count] = 2.0
    ub[active_offset : active_offset + edge_count] = 1.0
    ub[flow_ab_offset:] = float(island_count - 1)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        message = result.message if result.message else "No solution"
        raise SolveError(f"Puzzle has no valid solution ({message})")

    bridges = {
        edge.id: int(round(float(result.x[bridge_offset + edge.id])))
        for edge in edges
    }
    solution = Solution(bridges)
    validate_solution(puzzle, solution)
    return solution


def validate_solution(puzzle: Puzzle, solution: Solution) -> None:
    _validate_input(puzzle)
    by_island = {island.id: 0 for island in puzzle.islands}
    for edge in puzzle.edges:
        count = solution.bridges.get(edge.id, 0)
        if count not in (0, 1, 2):
            raise SolveError("A bridge count is outside 0..2")
        by_island[edge.a] += count
        by_island[edge.b] += count
    for island in puzzle.islands:
        if by_island[island.id] != island.value:
            raise SolveError(f"Island {island.id + 1} does not match its clue")

    for first, second in crossing_pairs(puzzle):
        if solution.bridges.get(first, 0) and solution.bridges.get(second, 0):
            raise SolveError("The solution contains crossing bridges")

    for index, magic_link in enumerate(puzzle.magic_links):
        first = solution.bridges.get(magic_link.edge_a, 0)
        second = solution.bridges.get(magic_link.edge_b, 0)
        if first != second:
            raise SolveError(f"Magic link {index + 1} does not have equal bridge counts")

    for index, border_hint in enumerate(puzzle.border_hints):
        total = sum(solution.bridges.get(edge_id, 0) for edge_id in border_hint.edge_ids)
        hint_type = _border_hint_type(border_hint)
        if hint_type == BorderHintType.EQ:
            satisfied = total == border_hint.hint
        elif hint_type == BorderHintType.GTE:
            satisfied = total >= border_hint.hint
        else:
            satisfied = total <= border_hint.hint
        if not satisfied:
            raise SolveError(
                f"Border hint {index + 1} is not satisfied "
                f"(sum {total}, {hint_type.name} {border_hint.hint})"
            )

    adjacency: dict[int, list[int]] = {island.id: [] for island in puzzle.islands}
    for edge in puzzle.edges:
        if solution.bridges.get(edge.id, 0):
            adjacency[edge.a].append(edge.b)
            adjacency[edge.b].append(edge.a)
    seen = {puzzle.islands[0].id}
    queue = deque(seen)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(seen) != len(puzzle.islands):
        raise SolveError("The solution is not connected")
