"""Read and match the puzzle library embedded in the BridgeFix Unity game."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import build_puzzle
from .model import (
    BorderHint,
    BorderHintType,
    BridgeAssistantError,
    Edge,
    Island,
    MagicLink,
    Puzzle,
    Solution,
    SolveError,
)
from .solver import validate_solution


LIBRARY_ASSET_NAMES = (
    "libraryVERY_EASY",
    "libraryEASY",
    "libraryMEDIUM",
    "libraryBIG",
    "libraryHUGE",
    "libraryVERY_EASY_TRICKY",
    "libraryEASY_TRICKY",
    "libraryMEDIUM_TRICKY",
    "libraryBIG_TRICKY",
    "libraryHUGE_TRICKY",
)
_LIBRARY_ASSET_NAME_SET = frozenset(LIBRARY_ASSET_NAMES)


class GameLibraryError(BridgeAssistantError):
    pass


class GameResourceNotFoundError(GameLibraryError):
    pass


class GameLibraryFormatError(GameLibraryError):
    pass


class GamePuzzleMatchError(GameLibraryError):
    pass


@dataclass(frozen=True, slots=True)
class GameNode:
    id: int
    x: int
    y: int
    value: int


@dataclass(frozen=True, slots=True)
class GameConnection:
    id: int
    a: int
    b: int
    orientation: str


@dataclass(frozen=True, slots=True)
class GameMagicLink:
    first_connection_id: int
    second_connection_id: int


@dataclass(frozen=True, slots=True)
class GameBorderHint:
    row: bool
    index: int
    hint: int
    type: BorderHintType
    connection_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GameOfficialAnswer:
    bridge_counts: tuple[int, ...]

    def bridge_count(self, connection_id: int) -> int:
        if connection_id < 0:
            raise KeyError(connection_id)
        try:
            return self.bridge_counts[connection_id]
        except IndexError as exc:
            raise KeyError(connection_id) from exc

    def to_solution(self, connection_to_edge: Mapping[int, int]) -> Solution:
        if set(connection_to_edge) != set(range(len(self.bridge_counts))):
            raise GamePuzzleMatchError(
                "Official answer cannot be mapped because the visual edge mapping is incomplete"
            )
        visual_edge_ids = list(connection_to_edge.values())
        if len(set(visual_edge_ids)) != len(visual_edge_ids):
            raise GamePuzzleMatchError(
                "Official answer cannot be mapped because visual edge ids are not unique"
            )
        return Solution(
            {
                connection_to_edge[connection_id]: bridge_count
                for connection_id, bridge_count in enumerate(self.bridge_counts)
            }
        )


@dataclass(frozen=True, slots=True)
class GamePuzzleDefinition:
    asset_name: str
    number: int
    difficulty: str
    width: int
    height: int
    fingerprint: int
    nodes: tuple[GameNode, ...]
    connections: tuple[GameConnection, ...]
    magic_links: tuple[GameMagicLink, ...]
    border_hints: tuple[GameBorderHint, ...]
    official_answer: GameOfficialAnswer


@dataclass(slots=True)
class GamePuzzleMatch:
    definition: GamePuzzleDefinition
    puzzle: Puzzle
    official_solution: Solution
    visual_to_game_nodes: dict[int, int]
    game_to_visual_nodes: dict[int, int]
    visual_to_game_edges: dict[int, int]
    game_to_visual_edges: dict[int, int]
    scale: float
    offset_x: float
    offset_y: float
    rms_error: float
    maximum_error: float

    @property
    def number(self) -> int:
        return self.definition.number

    @property
    def difficulty(self) -> str:
        return self.definition.difficulty


def _format_error(asset_name: str, line_number: int, message: str) -> GameLibraryFormatError:
    return GameLibraryFormatError(f"{asset_name} line {line_number}: {message}")


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _nested_json(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} contains invalid nested JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object or an encoded JSON object")
    return value


def _required_list(record: Mapping[str, Any], key: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _parse_record(
    asset_name: str,
    line_number: int,
    record: Mapping[str, Any],
) -> GamePuzzleDefinition:
    width = _require_int(record.get("sizex"), "sizex", minimum=1)
    height = _require_int(record.get("sizey"), "sizey", minimum=1)
    number = _require_int(record.get("num"), "num", minimum=0)
    fingerprint = _require_int(record.get("fingerprint"), "fingerprint")
    difficulty = record.get("_difficulty")
    if not isinstance(difficulty, str) or not difficulty:
        raise ValueError("_difficulty must be a non-empty string")

    raw_nodes = _required_list(record, "_nodes")
    nodes: list[GameNode] = []
    for node_id, raw_node in enumerate(raw_nodes):
        node = _nested_json(raw_node, f"_nodes[{node_id}]")
        x = _require_int(node.get("x"), f"_nodes[{node_id}].x", minimum=0)
        y = _require_int(node.get("y"), f"_nodes[{node_id}].y", minimum=0)
        value = _require_int(node.get("solution"), f"_nodes[{node_id}].solution", minimum=1)
        if x >= width or y >= height:
            raise ValueError(f"_nodes[{node_id}] lies outside the declared board size")
        if value > 8:
            raise ValueError(f"_nodes[{node_id}].solution must be in 1..8")
        nodes.append(GameNode(node_id, x, y, value))
    if not nodes:
        raise ValueError("_nodes must not be empty")
    if len({(node.x, node.y) for node in nodes}) != len(nodes):
        raise ValueError("_nodes contains duplicate coordinates")

    links_a = _required_list(record, "_linksA")
    links_b = _required_list(record, "_linksB")
    solution = _required_list(record, "solution")
    if not (len(links_a) == len(links_b) == len(solution)):
        raise ValueError("solution, _linksA, and _linksB must have identical lengths")

    connections: list[GameConnection] = []
    seen_pairs: set[tuple[int, int]] = set()
    for connection_id, (raw_a, raw_b, raw_count) in enumerate(
        zip(links_a, links_b, solution)
    ):
        a = _require_int(raw_a, f"_linksA[{connection_id}]", minimum=0)
        b = _require_int(raw_b, f"_linksB[{connection_id}]", minimum=0)
        bridge_count = _require_int(
            raw_count,
            f"solution[{connection_id}]",
            minimum=0,
        )
        if a >= len(nodes) or b >= len(nodes) or a == b:
            raise ValueError(f"connection {connection_id} has invalid node indices")
        if bridge_count > 2:
            raise ValueError(f"solution[{connection_id}] must be in 0..2")
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            raise ValueError(f"connection {connection_id} duplicates node pair {pair}")
        seen_pairs.add(pair)
        first, second = nodes[a], nodes[b]
        if first.y == second.y:
            orientation = "h"
            skipped = next(
                (
                    node.id
                    for node in nodes
                    if node.id not in pair
                    and node.y == first.y
                    and min(first.x, second.x) < node.x < max(first.x, second.x)
                ),
                None,
            )
        elif first.x == second.x:
            orientation = "v"
            skipped = next(
                (
                    node.id
                    for node in nodes
                    if node.id not in pair
                    and node.x == first.x
                    and min(first.y, second.y) < node.y < max(first.y, second.y)
                ),
                None,
            )
        else:
            raise ValueError(f"connection {connection_id} is not horizontal or vertical")
        if skipped is not None:
            raise ValueError(
                f"connection {connection_id} skips intermediate node {skipped}"
            )
        connections.append(GameConnection(connection_id, a, b, orientation))

    magic_a = _required_list(record, "_magicLinksA")
    magic_b = _required_list(record, "_magicLinksB")
    if len(magic_a) != len(magic_b):
        raise ValueError("_magicLinksA and _magicLinksB must have identical lengths")
    magic_links: list[GameMagicLink] = []
    for magic_id, (raw_first, raw_second) in enumerate(zip(magic_a, magic_b)):
        first = _require_int(raw_first, f"_magicLinksA[{magic_id}]", minimum=0)
        second = _require_int(raw_second, f"_magicLinksB[{magic_id}]", minimum=0)
        if first >= len(connections) or second >= len(connections) or first == second:
            raise ValueError(f"MagicLink {magic_id} has invalid connection indices")
        magic_links.append(GameMagicLink(first, second))

    border_hints: list[GameBorderHint] = []
    for hint_id, raw_hint in enumerate(_required_list(record, "_borderHints")):
        hint_data = _nested_json(raw_hint, f"_borderHints[{hint_id}]")
        row = hint_data.get("row")
        if not isinstance(row, bool):
            raise ValueError(f"_borderHints[{hint_id}].row must be a boolean")
        index = _require_int(hint_data.get("index"), f"_borderHints[{hint_id}].index", minimum=0)
        hint = _require_int(hint_data.get("hint"), f"_borderHints[{hint_id}].hint", minimum=0)
        try:
            hint_type = BorderHintType(
                _require_int(hint_data.get("type"), f"_borderHints[{hint_id}].type")
            )
        except ValueError as exc:
            raise ValueError(f"_borderHints[{hint_id}].type is unknown") from exc
        if index >= (height if row else width):
            raise ValueError(f"_borderHints[{hint_id}].index lies outside the board")
        connection_ids = tuple(
            connection.id
            for connection in connections
            if (
                nodes[connection.a].y == index
                and nodes[connection.b].y == index
                if row
                else nodes[connection.a].x == index
                and nodes[connection.b].x == index
            )
        )
        if not connection_ids:
            raise ValueError(f"_borderHints[{hint_id}] does not select any connections")
        border_hints.append(GameBorderHint(row, index, hint, hint_type, connection_ids))

    definition = GamePuzzleDefinition(
        asset_name=asset_name,
        number=number,
        difficulty=difficulty,
        width=width,
        height=height,
        fingerprint=fingerprint,
        nodes=tuple(nodes),
        connections=tuple(connections),
        magic_links=tuple(magic_links),
        border_hints=tuple(border_hints),
        official_answer=GameOfficialAnswer(
            tuple(
                _require_int(value, f"solution[{index}]", minimum=0)
                for index, value in enumerate(solution)
            )
        ),
    )
    validation_puzzle = Puzzle(
        islands=[Island(node.id, node.x, -node.y, node.value) for node in nodes],
        edges=[
            Edge(connection.id, connection.a, connection.b, connection.orientation)
            for connection in connections
        ],
        alignment_tolerance=0.01,
        magic_links=[
            MagicLink(link.first_connection_id, link.second_connection_id)
            for link in magic_links
        ],
        border_hints=[
            BorderHint(hint.connection_ids, hint.hint, hint.type)
            for hint in border_hints
        ],
    )
    try:
        validate_solution(
            validation_puzzle,
            Solution(dict(enumerate(definition.official_answer.bridge_counts))),
        )
    except SolveError as exc:
        raise ValueError(f"official solution is inconsistent: {exc}") from exc
    return definition


def parse_library_text(asset_name: str, text: str | bytes) -> tuple[GamePuzzleDefinition, ...]:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise GameLibraryFormatError(f"{asset_name}: TextAsset is not UTF-8") from exc
    if not isinstance(text, str):
        raise TypeError("TextAsset content must be str or bytes")
    text = text.removeprefix("\ufeff")

    puzzles: list[GamePuzzleDefinition] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _format_error(asset_name, line_number, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise _format_error(asset_name, line_number, "record must be a JSON object")
        try:
            puzzles.append(_parse_record(asset_name, line_number, record))
        except (KeyError, TypeError, ValueError) as exc:
            raise _format_error(asset_name, line_number, str(exc)) from exc
    if not puzzles:
        raise GameLibraryFormatError(f"{asset_name}: TextAsset contains no puzzle records")
    numbers = [puzzle.number for puzzle in puzzles]
    if len(set(numbers)) != len(numbers):
        raise GameLibraryFormatError(f"{asset_name}: duplicate puzzle numbers")
    return tuple(puzzles)


def _candidate_resources_paths(source: Path) -> list[Path]:
    source = source.expanduser()
    candidates: list[Path] = []
    if source.name.lower() == "resources.assets":
        candidates.append(source)
    if source.suffix.lower() == ".exe":
        candidates.append(source.with_name(f"{source.stem}_Data") / "resources.assets")
    candidates.extend(
        [
            source / "resources.assets",
            source / "Bridge-Windows64_Data" / "resources.assets",
        ]
    )
    if source.is_dir():
        for executable in source.glob("Bridge-Windows64.exe"):
            candidates.append(
                executable.with_name(f"{executable.stem}_Data") / "resources.assets"
            )
        for game_dir in source.glob("BridgeFix*"):
            candidates.append(game_dir / "Bridge-Windows64_Data" / "resources.assets")
    return candidates


def _steam_roots() -> list[Path]:
    roots: list[Path] = []
    for variable in ("STEAM_PATH", "STEAM_INSTALL_PATH"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]))
    for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]) / "Steam")
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
                roots.append(Path(winreg.QueryValueEx(key, "SteamPath")[0]))
        except (ImportError, OSError):
            pass

    library_roots = list(roots)
    for root in tuple(roots):
        library_file = root / "steamapps" / "libraryfolders.vdf"
        try:
            content = library_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(r'"path"\s+"([^"]+)"', content):
            library_roots.append(Path(match.group(1).replace(r"\\", "\\")))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in library_roots:
        key = os.path.normcase(os.path.abspath(root))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def locate_resources_assets(path: str | os.PathLike[str] | None = None) -> Path:
    sources: list[Path] = []
    if path is not None:
        sources.append(Path(path))
    else:
        configured = os.environ.get("BRIDGEFIX_PATH")
        if configured:
            sources.append(Path(configured))
        executable = shutil.which("Bridge-Windows64.exe")
        if executable:
            sources.append(Path(executable))
        sources.extend([Path.cwd(), Path(sys.executable)])
        sources.extend(root / "steamapps" / "common" for root in _steam_roots())

    attempted: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        for candidate in _candidate_resources_paths(source):
            key = os.path.normcase(os.path.abspath(candidate))
            if key in seen:
                continue
            seen.add(key)
            attempted.append(candidate)
            if candidate.is_file():
                return candidate.resolve()

    attempted_text = "\n".join(f"  - {candidate}" for candidate in attempted)
    qualifier = f" from {path!s}" if path is not None else ""
    raise GameResourceNotFoundError(
        "BridgeFix resources.assets was not found"
        f"{qualifier}. Pass the game directory, Bridge-Windows64.exe, or resources.assets "
        "explicitly, or set BRIDGEFIX_PATH. Checked:\n"
        f"{attempted_text or '  - no candidate paths were available'}"
    )


def read_library_text_assets(
    resources_path: str | os.PathLike[str],
) -> dict[str, str]:
    path = locate_resources_assets(resources_path)
    try:
        import UnityPy
    except ImportError as exc:
        raise GameLibraryError(
            "UnityPy is required to read BridgeFix resources.assets; install project dependencies"
        ) from exc
    try:
        environment = UnityPy.load(str(path))
    except Exception as exc:
        raise GameLibraryError(f"Failed to load {path} with UnityPy: {exc}") from exc

    assets: dict[str, str] = {}
    try:
        for obj in environment.objects:
            if obj.type.name != "TextAsset":
                continue
            data = obj.read()
            name = data.m_Name
            if name not in _LIBRARY_ASSET_NAME_SET:
                continue
            script = data.m_Script
            if isinstance(script, bytes):
                script = script.decode("utf-8-sig")
            if not isinstance(script, str):
                raise TypeError(f"{name} has an unsupported m_Script value")
            if name in assets:
                raise ValueError(f"duplicate TextAsset {name}")
            assets[name] = script
    except (AttributeError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise GameLibraryFormatError(
            f"Could not read BridgeFix TextAssets from {path}: {exc}"
        ) from exc

    missing = sorted(_LIBRARY_ASSET_NAME_SET - assets.keys())
    if missing:
        raise GameLibraryFormatError(
            f"{path} is missing BridgeFix library TextAssets: {', '.join(missing)}"
        )
    return assets


def _value_signature(nodes: Sequence[GameNode] | Sequence[Island]) -> tuple[int, ...]:
    counts = [0] * 8
    for node in nodes:
        if not 1 <= node.value <= 8:
            raise GamePuzzleMatchError("Visual island values must be in 1..8")
        counts[node.value - 1] += 1
    return tuple(counts)


def _alignment(
    visual_islands: Sequence[Island],
    definition: GamePuzzleDefinition,
) -> tuple[dict[int, int], float, float, float, float, float] | None:
    visual_points = np.asarray([[island.x, island.y] for island in visual_islands], dtype=float)
    game_points = np.asarray([[node.x, -node.y] for node in definition.nodes], dtype=float)
    visual_centered = visual_points - visual_points.mean(axis=0)
    game_centered = game_points - game_points.mean(axis=0)
    visual_norm = math.sqrt(float(np.mean(np.sum(visual_centered**2, axis=1))))
    game_norm = math.sqrt(float(np.mean(np.sum(game_centered**2, axis=1))))
    if visual_norm <= 0.0 or game_norm <= 0.0:
        return None

    normalized_visual = visual_centered / visual_norm
    normalized_game = game_centered / game_norm
    distances = np.linalg.norm(
        normalized_visual[:, None, :] - normalized_game[None, :, :], axis=2
    )
    visual_values = np.asarray([island.value for island in visual_islands])
    game_values = np.asarray([node.value for node in definition.nodes])
    distances[visual_values[:, None] != game_values[None, :]] = 1_000_000.0
    visual_indices, game_indices = linear_sum_assignment(distances)
    if np.any(visual_values[visual_indices] != game_values[game_indices]):
        return None

    ordered_game_points = np.empty_like(visual_points)
    game_id_by_visual_index: dict[int, int] = {}
    for visual_index, game_index in zip(visual_indices, game_indices):
        ordered_game_points[visual_index] = game_points[game_index]
        game_id_by_visual_index[int(visual_index)] = definition.nodes[int(game_index)].id
    ordered_centered = ordered_game_points - ordered_game_points.mean(axis=0)
    denominator = float(np.sum(ordered_centered**2))
    if denominator <= 0.0:
        return None
    scale = float(np.sum(ordered_centered * visual_centered) / denominator)
    if scale <= 0.0:
        return None
    offset = (visual_points - ordered_game_points * scale).mean(axis=0)
    residuals = np.linalg.norm(ordered_game_points * scale + offset - visual_points, axis=1)
    visual_to_game = {
        visual_islands[index].id: game_id_by_visual_index[index]
        for index in range(len(visual_islands))
    }
    return (
        visual_to_game,
        scale,
        float(offset[0]),
        float(offset[1]),
        float(math.sqrt(np.mean(residuals**2))),
        float(np.max(residuals)),
    )


def _edge_mapping(
    visual_puzzle: Puzzle,
    definition: GamePuzzleDefinition,
    game_to_visual_nodes: Mapping[int, int],
) -> tuple[dict[int, int], dict[int, int]] | None:
    visual_edge_ids = [edge.id for edge in visual_puzzle.edges]
    if len(set(visual_edge_ids)) != len(visual_edge_ids):
        raise GamePuzzleMatchError("Visual edge ids must be unique")
    visual_by_pair: dict[tuple[int, int], int] = {}
    for edge in visual_puzzle.edges:
        pair = tuple(sorted((edge.a, edge.b)))
        if pair in visual_by_pair:
            raise GamePuzzleMatchError(f"Visual puzzle has duplicate edge endpoints {pair}")
        visual_by_pair[pair] = edge.id

    game_to_visual_edges: dict[int, int] = {}
    for connection in definition.connections:
        pair = tuple(
            sorted(
                (
                    game_to_visual_nodes[connection.a],
                    game_to_visual_nodes[connection.b],
                )
            )
        )
        visual_edge_id = visual_by_pair.get(pair)
        if visual_edge_id is None:
            return None
        game_to_visual_edges[connection.id] = visual_edge_id
    if len(visual_puzzle.edges) != len(game_to_visual_edges):
        return None
    return (
        game_to_visual_edges,
        {visual_edge_id: game_id for game_id, visual_edge_id in game_to_visual_edges.items()},
    )


class GameLibrary:
    def __init__(self, puzzles: Sequence[GamePuzzleDefinition]):
        if not puzzles:
            raise GameLibraryFormatError("Game library contains no puzzles")
        numbers = [puzzle.number for puzzle in puzzles]
        if len(set(numbers)) != len(numbers):
            raise GameLibraryFormatError("Game library contains duplicate puzzle numbers")
        self.puzzles = tuple(puzzles)
        self.by_number = {puzzle.number: puzzle for puzzle in puzzles}
        by_signature: defaultdict[
            tuple[int, tuple[int, ...]], list[GamePuzzleDefinition]
        ] = defaultdict(list)
        for puzzle in puzzles:
            by_signature[(len(puzzle.nodes), _value_signature(puzzle.nodes))].append(puzzle)
        self._by_signature = dict(by_signature)

    @classmethod
    def from_text_assets(
        cls,
        assets: Mapping[str, str | bytes],
        *,
        require_complete: bool = False,
    ) -> GameLibrary:
        if require_complete:
            missing = sorted(_LIBRARY_ASSET_NAME_SET - assets.keys())
            unexpected = sorted(assets.keys() - _LIBRARY_ASSET_NAME_SET)
            if missing or unexpected:
                details = []
                if missing:
                    details.append(f"missing: {', '.join(missing)}")
                if unexpected:
                    details.append(f"unexpected: {', '.join(unexpected)}")
                raise GameLibraryFormatError(
                    "Invalid BridgeFix TextAsset set (" + "; ".join(details) + ")"
                )
        puzzles = [
            puzzle
            for asset_name, text in assets.items()
            for puzzle in parse_library_text(asset_name, text)
        ]
        if require_complete:
            counts: defaultdict[str, int] = defaultdict(int)
            for puzzle in puzzles:
                counts[puzzle.asset_name] += 1
            invalid = {
                name: counts[name]
                for name in LIBRARY_ASSET_NAMES
                if counts[name] != 500
            }
            if invalid:
                detail = ", ".join(f"{name}={count}" for name, count in invalid.items())
                raise GameLibraryFormatError(
                    f"Expected 500 puzzles in each BridgeFix TextAsset; got {detail}"
                )
        return cls(puzzles)

    @classmethod
    def from_resources(
        cls,
        path: str | os.PathLike[str] | None = None,
    ) -> GameLibrary:
        resources_path = locate_resources_assets(path)
        return cls.from_text_assets(
            read_library_text_assets(resources_path),
            require_complete=True,
        )

    def match(
        self,
        visual: Puzzle | Sequence[Island],
        *,
        maximum_error: float | None = None,
    ) -> GamePuzzleMatch:
        if isinstance(visual, Puzzle):
            visual_puzzle = visual
        else:
            visual_puzzle = build_puzzle(list(visual))
        islands = visual_puzzle.islands
        if len(islands) < 2:
            raise GamePuzzleMatchError("At least two visual islands are required")
        visual_ids = [island.id for island in islands]
        if len(set(visual_ids)) != len(visual_ids):
            raise GamePuzzleMatchError("Visual island ids must be unique")
        if any(
            not math.isfinite(value)
            for island in islands
            for value in (island.x, island.y, island.radius)
        ):
            raise GamePuzzleMatchError("Visual island coordinates and radii must be finite")
        threshold = maximum_error
        if threshold is None:
            threshold = max(3.0, median(max(1.0, island.radius) for island in islands) * 0.55)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("maximum_error must be a positive finite number")

        signature = (len(islands), _value_signature(islands))
        candidates = self._by_signature.get(signature, ())
        accepted: list[
            tuple[
                GamePuzzleDefinition,
                dict[int, int],
                dict[int, int],
                dict[int, int],
                float,
                float,
                float,
                float,
                float,
            ]
        ] = []
        for definition in candidates:
            alignment = _alignment(islands, definition)
            if alignment is None:
                continue
            visual_to_game, scale, offset_x, offset_y, rms_error, max_error = alignment
            if max_error > threshold:
                continue
            game_to_visual = {game_id: visual_id for visual_id, game_id in visual_to_game.items()}
            mapped_edges = _edge_mapping(visual_puzzle, definition, game_to_visual)
            if mapped_edges is None:
                continue
            game_to_visual_edges, visual_to_game_edges = mapped_edges
            accepted.append(
                (
                    definition,
                    visual_to_game,
                    game_to_visual,
                    game_to_visual_edges,
                    scale,
                    offset_x,
                    offset_y,
                    rms_error,
                    max_error,
                )
            )

        if not accepted:
            raise GamePuzzleMatchError(
                f"No BridgeFix puzzle matches {len(islands)} visual islands "
                f"within {threshold:.2f} pixels"
            )
        if len(accepted) > 1:
            numbers = ", ".join(str(item[0].number) for item in accepted[:8])
            suffix = "..." if len(accepted) > 8 else ""
            raise GamePuzzleMatchError(
                f"Visual puzzle is ambiguous; {len(accepted)} BridgeFix puzzles match: "
                f"{numbers}{suffix}"
            )

        (
            definition,
            visual_to_game,
            game_to_visual,
            game_to_visual_edges,
            scale,
            offset_x,
            offset_y,
            rms_error,
            max_error,
        ) = accepted[0]
        visual_to_game_edges = {
            visual_edge_id: game_edge_id
            for game_edge_id, visual_edge_id in game_to_visual_edges.items()
        }
        puzzle = Puzzle(
            islands=list(visual_puzzle.islands),
            edges=list(visual_puzzle.edges),
            alignment_tolerance=visual_puzzle.alignment_tolerance,
            magic_links=[
                MagicLink(
                    game_to_visual_edges[link.first_connection_id],
                    game_to_visual_edges[link.second_connection_id],
                )
                for link in definition.magic_links
            ],
            border_hints=[
                BorderHint(
                    tuple(game_to_visual_edges[edge_id] for edge_id in hint.connection_ids),
                    hint.hint,
                    hint.type,
                )
                for hint in definition.border_hints
            ],
        )
        return GamePuzzleMatch(
            definition=definition,
            puzzle=puzzle,
            official_solution=definition.official_answer.to_solution(game_to_visual_edges),
            visual_to_game_nodes=visual_to_game,
            game_to_visual_nodes=game_to_visual,
            visual_to_game_edges=visual_to_game_edges,
            game_to_visual_edges=game_to_visual_edges,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
            rms_error=rms_error,
            maximum_error=max_error,
        )


BridgeFixLibrary = GameLibrary


def load_game_library(
    path: str | os.PathLike[str] | None = None,
) -> GameLibrary:
    return GameLibrary.from_resources(path)
