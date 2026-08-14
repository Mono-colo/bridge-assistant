from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridge_assistant.game_library import (
    GameLibrary,
    GameLibraryFormatError,
    GamePuzzleMatchError,
    GameResourceNotFoundError,
    locate_resources_assets,
    parse_library_text,
)
from bridge_assistant.model import BorderHintType, Island, MagicLink


def encoded(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


def puzzle_record(number: int = 42) -> dict:
    return {
        "solution": [2, 1, 2, 1],
        "sizex": 3,
        "sizey": 3,
        "fingerprint": 12345,
        "num": number,
        "_difficulty": "TEST_TRICKY",
        "_nodes": [
            encoded({"x": 0, "y": 0, "solution": 3}),
            encoded({"x": 2, "y": 0, "solution": 3}),
            encoded({"x": 0, "y": 2, "solution": 3}),
            encoded({"x": 2, "y": 2, "solution": 3}),
        ],
        "_borderHints": [encoded({"row": True, "index": 0, "hint": 2, "type": 1})],
        "_magicLinksA": [0],
        "_magicLinksB": [2],
        "_linksA": [0, 0, 2, 1],
        "_linksB": [1, 2, 3, 3],
    }


def library_text(*records: dict) -> str:
    return "\n".join(json.dumps(record, separators=(",", ":")) for record in records)


def visual_islands() -> list[Island]:
    # screen_x = 10 * game_x + 100; screen_y = -10 * game_y + 80
    return [
        Island(12, 120.1, 60.0, 3, 4),
        Island(41, 99.9, 80.1, 3, 4),
        Island(99, 100.1, 59.9, 3, 4),
        Island(7, 120.0, 80.0, 3, 4),
    ]


def test_parse_json_lines_and_nested_schema() -> None:
    definition = parse_library_text("libraryTEST", library_text(puzzle_record()))[0]

    assert definition.number == 42
    assert definition.difficulty == "TEST_TRICKY"
    assert [(node.id, node.x, node.y, node.value) for node in definition.nodes] == [
        (0, 0, 0, 3),
        (1, 2, 0, 3),
        (2, 0, 2, 3),
        (3, 2, 2, 3),
    ]
    assert [connection.orientation for connection in definition.connections] == ["h", "v", "h", "v"]
    assert definition.magic_links[0].first_connection_id == 0
    assert definition.magic_links[0].second_connection_id == 2
    assert definition.border_hints[0].type == BorderHintType.GTE
    assert definition.border_hints[0].connection_ids == (0,)
    assert definition.official_answer.bridge_counts == (2, 1, 2, 1)
    assert definition.official_answer.bridge_count(2) == 2
    with pytest.raises(KeyError):
        definition.official_answer.bridge_count(-1)


def test_parse_accepts_decoded_utf8_bom() -> None:
    parsed = parse_library_text("libraryTEST", "\ufeff" + library_text(puzzle_record()))

    assert parsed[0].number == 42


def test_match_preserves_visual_ids_and_maps_constraints_and_answer() -> None:
    library = GameLibrary.from_text_assets({"libraryTEST": library_text(puzzle_record())})

    match = library.match(visual_islands())

    assert match.number == 42
    assert match.scale == pytest.approx(10.0, abs=0.05)
    assert match.offset_x == pytest.approx(100.0, abs=0.1)
    assert match.offset_y == pytest.approx(80.0, abs=0.1)
    assert match.maximum_error < 0.2
    assert match.visual_to_game_nodes == {12: 3, 41: 0, 99: 2, 7: 1}
    assert [island.id for island in match.puzzle.islands] == [12, 41, 99, 7]

    assert match.game_to_visual_edges == {0: 1, 1: 3, 2: 2, 3: 0}
    assert match.visual_to_game_edges == {1: 0, 3: 1, 2: 2, 0: 3}
    assert match.puzzle.magic_links == [MagicLink(1, 2)]
    assert len(match.puzzle.border_hints) == 1
    assert match.puzzle.border_hints[0].edge_ids == (1,)
    assert match.puzzle.border_hints[0].hint == 2
    assert match.puzzle.border_hints[0].type == BorderHintType.GTE
    assert match.official_solution.bridges == {1: 2, 3: 1, 2: 2, 0: 1}


def test_match_rejects_ambiguous_layout() -> None:
    library = GameLibrary.from_text_assets(
        {"libraryTEST": library_text(puzzle_record(42), puzzle_record(43))}
    )

    with pytest.raises(GamePuzzleMatchError, match="ambiguous"):
        library.match(visual_islands())


def test_match_requires_one_uniform_scale() -> None:
    library = GameLibrary.from_text_assets({"libraryTEST": library_text(puzzle_record())})
    distorted = visual_islands()
    for island in distorted:
        if island.y < 70:
            island.y -= 10

    with pytest.raises(GamePuzzleMatchError, match="No BridgeFix puzzle matches"):
        library.match(distorted, maximum_error=1.0)


def test_parse_reports_asset_and_line_for_bad_schema() -> None:
    record = puzzle_record()
    record["_magicLinksB"] = []

    with pytest.raises(GameLibraryFormatError, match=r"libraryTEST line 1.*identical lengths"):
        parse_library_text("libraryTEST", library_text(record))


def test_parse_rejects_inconsistent_official_answer() -> None:
    record = puzzle_record()
    record["solution"] = [1, 1, 1, 1]

    with pytest.raises(GameLibraryFormatError, match="official solution is inconsistent"):
        parse_library_text("libraryTEST", library_text(record))


def test_parse_rejects_connection_that_skips_an_island() -> None:
    record = puzzle_record()
    record["_nodes"].append(encoded({"x": 1, "y": 0, "solution": 1}))

    with pytest.raises(GameLibraryFormatError, match="skips intermediate node"):
        parse_library_text("libraryTEST", library_text(record))


def test_official_answer_rejects_duplicate_visual_edge_ids() -> None:
    definition = parse_library_text("libraryTEST", library_text(puzzle_record()))[0]

    with pytest.raises(GamePuzzleMatchError, match="visual edge ids are not unique"):
        definition.official_answer.to_solution({0: 7, 1: 7, 2: 8, 3: 9})


def test_explicit_executable_path_locates_resources(tmp_path: Path) -> None:
    executable = tmp_path / "Bridge-Windows64.exe"
    executable.touch()
    resources = tmp_path / "Bridge-Windows64_Data" / "resources.assets"
    resources.parent.mkdir()
    resources.touch()

    assert locate_resources_assets(executable) == resources.resolve()


def test_missing_resources_error_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "Bridge-Windows64.exe"

    with pytest.raises(GameResourceNotFoundError) as caught:
        locate_resources_assets(missing)

    assert "resources.assets was not found" in str(caught.value)
    assert "BRIDGEFIX_PATH" in str(caught.value)
    assert "Bridge-Windows64_Data" in str(caught.value)
