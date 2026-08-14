from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bridge_assistant.board_locator import locate_board
from bridge_assistant.model import RecognitionError, SolveError
from bridge_assistant.solver import solve


def fixture_board() -> np.ndarray:
    path = Path(__file__).parent / "fixtures" / "puzzle_bridges_7x7.png"
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def embed_board(
    board: np.ndarray,
    *,
    width: int = 1280,
    height: int = 800,
    left: int = 420,
    top: int = 230,
) -> np.ndarray:
    image = np.full((height, width, 3), 242, dtype=np.uint8)
    image[top : top + board.shape[0], left : left + board.shape[1]] = board
    return image


def unsolvable_board() -> np.ndarray:
    image = np.full((250, 250, 3), 248, dtype=np.uint8)
    for x, y in ((50, 50), (200, 50), (50, 200), (200, 200)):
        cv2.circle(image, (x, y), 19, (10, 10, 10), 2, cv2.LINE_AA)
        text = "8"
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
    return image


def test_locate_board_inside_full_client_area() -> None:
    board = fixture_board()
    located = locate_board(embed_board(board))
    assert len(located.puzzle.islands) == 14
    assert len(located.puzzle.edges) == 17
    assert located.region.left < 420
    assert located.region.top < 230
    assert located.region.right > 420 + board.shape[1]
    assert located.region.bottom > 230 + board.shape[0]


def test_locator_ignores_circled_number_controls() -> None:
    board = fixture_board()
    image = embed_board(board)
    for index, value in enumerate((1, 2, 3, 4)):
        center = (60 + index * 85, 60)
        cv2.circle(image, center, 21, (20, 20, 20), 2, cv2.LINE_AA)
        text = str(value)
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(
            image,
            text,
            (center[0] - width // 2, center[1] + height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    located = locate_board(image)
    assert len(located.puzzle.islands) == 14
    assert located.region.left > 300


def test_locate_small_board_in_4k_client_area() -> None:
    board = cv2.resize(fixture_board(), None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
    located = locate_board(embed_board(board, width=3840, height=2160, left=2730, top=1510))
    assert len(located.puzzle.islands) == 14
    assert len(located.puzzle.edges) == 17
    assert located.region.left < 2730
    assert located.region.top < 1510


def test_locate_board_does_not_require_a_solution() -> None:
    located = locate_board(embed_board(unsolvable_board(), left=420, top=230))

    assert [island.value for island in located.puzzle.islands] == [8, 8, 8, 8]
    assert len(located.puzzle.edges) == 4
    with pytest.raises(SolveError, match="impossible degree"):
        solve(located.puzzle)


def test_two_equally_credible_boards_are_rejected() -> None:
    board = fixture_board()
    image = np.full((700, 1300, 3), 242, dtype=np.uint8)
    image[170 : 170 + board.shape[0], 100 : 100 + board.shape[1]] = board
    image[170 : 170 + board.shape[0], 820 : 820 + board.shape[1]] = board
    with pytest.raises(RecognitionError, match="多个可信题面"):
        locate_board(image)


def test_no_board_requests_region_fallback() -> None:
    image = np.full((600, 900, 3), 242, dtype=np.uint8)
    with pytest.raises(RecognitionError, match="框选题面"):
        locate_board(image)


@pytest.mark.parametrize(
    "filename",
    [
        "66ce9e45-39ff-49a8-b427-34fa883d98a6.png",
        "b2e81ea3-a3d6-4513-87d6-8df3169faef0.png",
        "cb476f14-c682-4b98-80d5-627605a71474.png",
        "4.png",
    ],
)
def test_locate_large_real_board_in_client_area(filename: str) -> None:
    path = Path(__file__).parents[1] / "test_image" / filename
    board = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = np.full((board.shape[0] + 140, board.shape[1] + 260, 3), 242, dtype=np.uint8)
    image[80 : 80 + board.shape[0], 150 : 150 + board.shape[1]] = board
    located = locate_board(image)
    assert len(located.puzzle.islands) == 60
    assert 0 <= located.region.left < image.shape[1]
    assert 0 <= located.region.top < image.shape[0]
    assert located.region.right <= image.shape[1]
    assert located.region.bottom <= image.shape[0]
