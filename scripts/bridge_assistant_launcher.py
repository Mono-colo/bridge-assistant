import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from bridge_assistant.auto_runner import main
from bridge_assistant.game_library import load_game_library
from bridge_assistant.graph import build_puzzle
from bridge_assistant.solver import solve
from bridge_assistant.vision import recognize
from bridge_assistant.windowing import list_app_windows


def self_test() -> None:
    image = np.full((240, 240, 3), 248, dtype=np.uint8)
    for x, y in ((45, 45), (195, 45), (45, 195), (195, 195)):
        cv2.circle(image, (x, y), 19, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "2",
            (x - 8, y + 9),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
    recognition = recognize(image)
    puzzle = build_puzzle(recognition.islands)
    solution = solve(puzzle)
    if len(puzzle.islands) != 4 or solution.active_count != 4:
        raise RuntimeError("Bundled recognition or solver self-test failed")
    list_app_windows()
    library = load_game_library()
    if len(library.puzzles) != 5000:
        raise RuntimeError("Bundled BridgeFix library adapter self-test failed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        raise SystemExit(main())
