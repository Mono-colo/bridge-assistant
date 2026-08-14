from __future__ import annotations

from collections import defaultdict
from statistics import median

from .model import Edge, Island, Puzzle


def _cluster_axis(islands: list[Island], axis: str, tolerance: float) -> dict[int, int]:
    ordered = sorted(islands, key=lambda island: getattr(island, axis))
    groups: list[list[Island]] = []
    for island in ordered:
        coordinate = getattr(island, axis)
        if not groups:
            groups.append([island])
            continue
        center = sum(getattr(item, axis) for item in groups[-1]) / len(groups[-1])
        if abs(coordinate - center) <= tolerance:
            groups[-1].append(island)
        else:
            groups.append([island])
    return {
        island.id: group_index
        for group_index, group in enumerate(groups)
        for island in group
    }


def build_puzzle(islands: list[Island], tolerance: float | None = None) -> Puzzle:
    if not islands:
        return Puzzle([])
    ids = [island.id for island in islands]
    if len(set(ids)) != len(ids):
        raise ValueError("Island ids must be unique")

    typical_radius = median(max(1.0, island.radius) for island in islands)
    tolerance = tolerance if tolerance is not None else max(3.0, typical_radius * 0.48)
    x_groups = _cluster_axis(islands, "x", tolerance)
    y_groups = _cluster_axis(islands, "y", tolerance)

    by_row: dict[int, list[Island]] = defaultdict(list)
    by_column: dict[int, list[Island]] = defaultdict(list)
    for island in islands:
        by_row[y_groups[island.id]].append(island)
        by_column[x_groups[island.id]].append(island)

    pairs: dict[tuple[int, int], str] = {}
    for row in by_row.values():
        row.sort(key=lambda island: island.x)
        for left, right in zip(row, row[1:]):
            pairs[tuple(sorted((left.id, right.id)))] = "h"
    for column in by_column.values():
        column.sort(key=lambda island: island.y)
        for top, bottom in zip(column, column[1:]):
            pairs[tuple(sorted((top.id, bottom.id)))] = "v"

    edges = [
        Edge(index, pair[0], pair[1], orientation)
        for index, (pair, orientation) in enumerate(sorted(pairs.items()))
    ]
    return Puzzle(list(islands), edges, tolerance)


def crossing_pairs(puzzle: Puzzle) -> list[tuple[int, int]]:
    by_id = {island.id: island for island in puzzle.islands}
    horizontal = [edge for edge in puzzle.edges if edge.orientation == "h"]
    vertical = [edge for edge in puzzle.edges if edge.orientation == "v"]
    crossings: list[tuple[int, int]] = []
    eps = max(1.0, puzzle.alignment_tolerance * 0.25)

    for h_edge in horizontal:
        h_a, h_b = by_id[h_edge.a], by_id[h_edge.b]
        hx1, hx2 = sorted((h_a.x, h_b.x))
        hy = (h_a.y + h_b.y) / 2
        for v_edge in vertical:
            v_a, v_b = by_id[v_edge.a], by_id[v_edge.b]
            vx = (v_a.x + v_b.x) / 2
            vy1, vy2 = sorted((v_a.y, v_b.y))
            if hx1 + eps < vx < hx2 - eps and vy1 + eps < hy < vy2 - eps:
                crossings.append((h_edge.id, v_edge.id))
    return crossings

