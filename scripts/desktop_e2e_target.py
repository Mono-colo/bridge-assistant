from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_assistant.capture import enable_dpi_awareness


TITLE = "Bridge Assistant E2E Target"
ISLANDS = (
    (70, 70, 1),
    (290, 70, 3),
    (290, 290, 2),
)
EDGES = ((0, 1), (1, 2))
ISLAND_RADIUS = 25
BOARD_OFFSET = (70, 90)


class TargetWindow:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.counts = {edge: 0 for edge in EDGES}
        self.events: list[dict[str, object]] = []
        self.pressed: dict[str, object] | None = None

        self.user32 = ctypes.windll.user32
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetFocus.argtypes = []
        self.user32.GetFocus.restype = wintypes.HWND

        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.geometry("500x440+1380+180")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.root, width=500, height=440, background="#f8fafc", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.draw()
        self.save_state()

    def event_snapshot(self, event: tk.Event) -> dict[str, object]:
        return {
            "local": [int(event.x), int(event.y)],
            "screen": [int(event.x_root), int(event.y_root)],
            "foreground": int(self.user32.GetForegroundWindow() or 0),
            "focus": int(self.user32.GetFocus() or 0),
        }

    def edge_at_point(self, x: int, y: int) -> tuple[int, int] | None:
        x -= BOARD_OFFSET[0]
        y -= BOARD_OFFSET[1]
        for edge in EDGES:
            first, second = ISLANDS[edge[0]], ISLANDS[edge[1]]
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            along = (x - first[0]) * ux + (y - first[1]) * uy
            across = abs((x - first[0]) * uy - (y - first[1]) * ux)
            if ISLAND_RADIUS + 2 < along < length - ISLAND_RADIUS - 2 and across <= 11:
                return edge
        return None

    def on_press(self, event: tk.Event) -> None:
        self.pressed = self.event_snapshot(event)
        self.pressed["edge"] = self.edge_at_point(event.x, event.y)
        self.save_state()

    def on_release(self, event: tk.Event) -> None:
        released = self.event_snapshot(event)
        released_edge = self.edge_at_point(event.x, event.y)
        pressed = self.pressed
        self.pressed = None
        pressed_edge = pressed.get("edge") if pressed is not None else None
        press_local = pressed.get("local") if pressed is not None else None
        moved = (
            math.dist(press_local, released["local"])
            if isinstance(press_local, list)
            else float("inf")
        )
        edge = tuple(pressed_edge) if pressed_edge is not None else None
        accepted = edge is not None and edge == released_edge and moved <= 4.0
        self.events.append(
            {
                "press": pressed,
                "release": released,
                "edge": list(edge) if edge is not None else None,
                "accepted": accepted,
            }
        )
        if accepted:
            self.counts[edge] = (self.counts[edge] + 1) % 3
            self.draw()
        self.save_state()

    def solved(self) -> bool:
        degrees = [0] * len(ISLANDS)
        adjacency = [[] for _ in ISLANDS]
        for (first, second), count in self.counts.items():
            degrees[first] += count
            degrees[second] += count
            if count:
                adjacency[first].append(second)
                adjacency[second].append(first)
        if degrees != [value for _, _, value in ISLANDS]:
            return False
        seen = {0}
        pending = [0]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        return len(seen) == len(ISLANDS)

    def save_state(self) -> None:
        state = {
            "solved": self.solved(),
            "events": self.events,
            "edges": {f"{first}-{second}": count for (first, second), count in self.counts.items()},
            "pending_press": self.pressed,
            "window": int(self.root.winfo_id()),
            "canvas": int(self.canvas.winfo_id()),
        }
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
        temporary.replace(self.output)

    def draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 500, 54, fill="#e2e8f0", outline="")
        self.canvas.create_text(18, 27, text="Puzzle workspace", anchor="w", fill="#334155", font=("Segoe UI", 12, "bold"))
        for index, value in enumerate((1, 2, 3, 4)):
            x = 300 + index * 45
            self.canvas.create_oval(x - 15, 12, x + 15, 42, fill="white", outline="#64748b", width=2)
            self.canvas.create_text(x, 27, text=str(value), fill="#334155", font=("Segoe UI", 11, "bold"))
        ox, oy = BOARD_OFFSET
        self.canvas.create_rectangle(ox + 18, oy + 18, ox + 342, oy + 342, outline="#cbd5e1", width=1)
        for (first, second), count in self.counts.items():
            if not count:
                continue
            x1, y1, _ = ISLANDS[first]
            x2, y2, _ = ISLANDS[second]
            x1, y1 = x1 + ox, y1 + oy
            x2, y2 = x2 + ox, y2 + oy
            offsets = (0,) if count == 1 else (-4, 4)
            for offset in offsets:
                if y1 == y2:
                    self.canvas.create_line(x1, y1 + offset, x2, y2 + offset, fill="#2563eb", width=3)
                else:
                    self.canvas.create_line(x1 + offset, y1, x2 + offset, y2, fill="#2563eb", width=3)
        for x, y, value in ISLANDS:
            x, y = x + ox, y + oy
            self.canvas.create_oval(
                x - ISLAND_RADIUS,
                y - ISLAND_RADIUS,
                x + ISLAND_RADIUS,
                y + ISLAND_RADIUS,
                fill="white",
                outline="#111827",
                width=3,
            )
            self.canvas.create_text(x, y, text=str(value), fill="#111827", font=("Segoe UI", 20, "bold"))
        if self.solved():
            self.canvas.create_rectangle(ox + 3, oy + 3, ox + 357, oy + 357, outline="#16a34a", width=6)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enable_dpi_awareness()
    TargetWindow(args.output).run()


if __name__ == "__main__":
    main()
