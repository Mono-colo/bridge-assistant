from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyautogui

from bridge_assistant.automation import DEFAULT_MAX_RETRIES, build_mouse_actions, execute_mouse_actions
from bridge_assistant.board_locator import locate_board
from bridge_assistant.capture import enable_dpi_awareness, grab_screen, pil_to_bgr
from bridge_assistant.model import InputMode, ScreenRegion
from bridge_assistant.scene import VisualActionVerifier, verify_scene
from bridge_assistant.solver import solve
from bridge_assistant.windowing import activate_window, list_app_windows, window_client_region

from desktop_e2e_target import TITLE


def wait_for_client_region(process_id: int, timeout: float = 10.0) -> tuple[int, ScreenRegion]:
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hwnd = int(user32.FindWindowW(None, TITLE) or 0)
        if hwnd:
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(owner))
            rect = wintypes.RECT()
            origin = wintypes.POINT(0, 0)
            if owner.value == process_id and user32.GetClientRect(hwnd, ctypes.byref(rect)) and user32.ClientToScreen(hwnd, ctypes.byref(origin)):
                width = int(rect.right - rect.left)
                height = int(rect.bottom - rect.top)
                if width >= 350 and height >= 350:
                    return hwnd, ScreenRegion(int(origin.x), int(origin.y), width, height)
        time.sleep(0.1)
    raise TimeoutError("The desktop target window did not become ready")


def wait_for_solved(path: Path, timeout: float = 5.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            last = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
            continue
        if last.get("solved"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"Target did not reach the solved state: {last}")


def run_once(iteration: int) -> dict[str, object]:
    if not hasattr(ctypes, "windll"):
        raise SystemExit("This end-to-end check requires Windows")
    enable_dpi_awareness()
    output = Path(tempfile.gettempdir()) / f"bridge-assistant-desktop-e2e-{iteration}.json"
    output.unlink(missing_ok=True)
    target = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("desktop_e2e_target.py")), "--output", str(output)],
        cwd=ROOT,
    )
    hwnd = 0
    try:
        hwnd, expected_client_region = wait_for_client_region(target.pid)
        matching = [
            window
            for window in list_app_windows(exclude_process_id=-1)
            if window.handle == hwnd and window.process_id == target.pid
        ]
        if len(matching) != 1:
            raise AssertionError(f"Window enumeration did not return the target exactly once: {matching}")
        target_window = matching[0]
        activate_window(target_window, timeout=5.0)
        time.sleep(0.35)
        client_region = window_client_region(target_window)
        if client_region != expected_client_region:
            raise AssertionError(
                f"Unexpected client region: expected={expected_client_region}, actual={client_region}"
            )

        # Capture only obtains the full client image. Location/recognition and
        # solving are deliberately invoked as separate stages below.
        first_client = pil_to_bgr(grab_screen(client_region))
        first_located = locate_board(first_client)
        second_client = pil_to_bgr(grab_screen(window_client_region(target_window)))
        located = locate_board(second_client)
        if first_located.region != located.region:
            raise AssertionError(
                f"Repeated captures located different board regions: {first_located.region} != {located.region}"
            )
        region = ScreenRegion(
            client_region.left + located.region.left,
            client_region.top + located.region.top,
            located.region.width,
            located.region.height,
        )
        reference = located.image
        puzzle = located.puzzle
        solution = solve(puzzle)
        if len(puzzle.islands) != 3 or [island.value for island in puzzle.islands] != [1, 3, 2]:
            raise AssertionError(f"Unexpected recognition: {[island.value for island in puzzle.islands]}")
        if solution.active_count != 2 or sum(solution.bridges.values()) != 3:
            raise AssertionError(f"Unexpected solution: {solution.bridges}")

        current = pil_to_bgr(grab_screen(region))
        similarity = verify_scene(reference, current, puzzle.islands)
        actions = build_mouse_actions(puzzle, solution, region, InputMode.NEAR_ENDPOINT)
        if any(action.start != action.end for action in actions):
            raise AssertionError("Primary input mode generated a non-click gesture")
        print(
            json.dumps(
                {
                    "region": [region.left, region.top, region.width, region.height],
                    "islands": [
                        [island.id, round(island.x), round(island.y), island.value]
                        for island in puzzle.islands
                    ],
                    "actions": [
                        [list(action.start), list(action.end), action.bridge_count]
                        for action in actions
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        verifier = VisualActionVerifier(
            current,
            region,
            lambda: pil_to_bgr(grab_screen(region)),
        )
        parking_point = (region.right - 8, region.bottom - 8)
        pyautogui.moveTo(*parking_point, duration=0.05)
        completed = execute_mouse_actions(
            actions,
            interval=0.18,
            operation_confirmed=verifier,
            prepare_retry=lambda _action, _attempt: activate_window(target_window, timeout=5.0),
            max_retries=DEFAULT_MAX_RETRIES,
            verification_cursor=parking_point,
        )
        state = wait_for_solved(output)
        expected_points = [
            action.start
            for action in actions
            for _ in range(action.bridge_count)
        ]
        events = state["events"]
        actual_points = [tuple(event["press"]["screen"]) for event in events]
        if len(events) != len(expected_points) or actual_points != expected_points:
            raise AssertionError(f"Unexpected click events: expected={expected_points}, actual={events}")
        if any(
            not event["accepted"]
            or event["press"]["screen"] != event["release"]["screen"]
            or event["press"]["foreground"] != hwnd
            or event["release"]["foreground"] != hwnd
            or not event["press"]["focus"]
            or not event["release"]["focus"]
            for event in events
        ):
            raise AssertionError(f"A system click was rejected or delivered without focus: {events}")
        result = {
            "iteration": iteration,
            "recognized_islands": len(puzzle.islands),
            "candidate_edges": len(puzzle.edges),
            "mouse_operations": completed,
            "accepted_events": sum(bool(event["accepted"]) for event in events),
            "input_mode": InputMode.NEAR_ENDPOINT.value,
            "bound_process_id": target_window.process_id,
            "scene_similarity": round(similarity, 4),
            "solved": state["solved"],
        }
        return result
    finally:
        if hwnd:
            user32 = ctypes.windll.user32
            user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.PostMessageW.restype = wintypes.BOOL
            user32.PostMessageW(wintypes.HWND(hwnd), 0x0010, 0, 0)
        try:
            target.wait(timeout=2)
        except subprocess.TimeoutExpired:
            target.terminate()
            target.wait(timeout=2)
        output.unlink(missing_ok=True)


def main() -> None:
    if not hasattr(ctypes, "windll"):
        raise SystemExit("This end-to-end check requires Windows")
    enable_dpi_awareness()
    results = []
    for iteration in range(1, 6):
        result = run_once(iteration)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    print(
        json.dumps(
            {
                "runs": len(results),
                "all_solved": all(result["solved"] for result in results),
                "accepted_events": sum(int(result["accepted_events"]) for result in results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
