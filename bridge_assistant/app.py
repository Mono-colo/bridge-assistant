from __future__ import annotations

import statistics
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import ImageTk

from .automation import DEFAULT_MAX_RETRIES, build_mouse_actions, execute_mouse_actions
from .board_locator import locate_board
from .capture import bgr_to_pil, enable_dpi_awareness, grab_screen, pil_to_bgr, virtual_screen_region
from .graph import build_puzzle
from .model import (
    BridgeAssistantError,
    AppWindow,
    InputMode,
    Island,
    Puzzle,
    Recognition,
    ScreenRegion,
    Solution,
)
from .render import render_overlay
from .scene import VisualActionVerifier, verify_scene
from .solver import solve
from .vision import recognize
from .windowing import activate_window, list_app_windows, window_at_screen_point, window_client_region


class RegionSelector:
    def __init__(self, parent: tk.Tk, callback) -> None:
        self.parent = parent
        self.callback = callback
        self.screen_region = virtual_screen_region()
        self.screenshot = grab_screen()
        self.start: tuple[int, int] | None = None
        self.rectangle: int | None = None

        self.window = tk.Toplevel(parent)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        geometry = (
            f"{self.screen_region.width}x{self.screen_region.height}"
            f"{self.screen_region.left:+d}{self.screen_region.top:+d}"
        )
        self.window.geometry(geometry)
        self.window.configure(cursor="crosshair")
        self.canvas = tk.Canvas(self.window, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.photo = ImageTk.PhotoImage(self.screenshot)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.create_rectangle(
            0,
            0,
            self.screen_region.width,
            self.screen_region.height,
            fill="#111827",
            stipple="gray50",
            outline="",
        )
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.window.bind("<Escape>", lambda _event: self._cancel())
        self.window.focus_force()
        self.window.grab_set()

    def _press(self, event: tk.Event) -> None:
        self.start = (int(event.x), int(event.y))
        self.rectangle = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#22c55e",
            width=3,
            fill="white",
            stipple="gray75",
        )

    def _move(self, event: tk.Event) -> None:
        if self.start is not None and self.rectangle is not None:
            self.canvas.coords(self.rectangle, self.start[0], self.start[1], event.x, event.y)

    def _release(self, event: tk.Event) -> None:
        if self.start is None:
            return
        x1, x2 = sorted((self.start[0], int(event.x)))
        y1, y2 = sorted((self.start[1], int(event.y)))
        x1 = max(0, min(self.screen_region.width, x1))
        x2 = max(0, min(self.screen_region.width, x2))
        y1 = max(0, min(self.screen_region.height, y1))
        y2 = max(0, min(self.screen_region.height, y2))
        if x2 - x1 < 30 or y2 - y1 < 30:
            self._cancel()
            return
        region = ScreenRegion(
            self.screen_region.left + x1,
            self.screen_region.top + y1,
            x2 - x1,
            y2 - y1,
        )
        crop = self.screenshot.crop((x1, y1, x2, y2))
        self.window.grab_release()
        self.window.withdraw()
        self.window.update_idletasks()
        target = window_at_screen_point(region.left + region.width // 2, region.top + region.height // 2)
        self.window.destroy()
        self.callback(region, crop, target)

    def _cancel(self) -> None:
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.window.destroy()
        self.callback(None, None, None)


class WindowSelector:
    def __init__(self, parent: tk.Tk, callback) -> None:
        self.callback = callback
        self.windows: dict[str, AppWindow] = {}
        self.window = tk.Toplevel(parent)
        self.window.title("选择应用窗口")
        self.window.geometry("760x430")
        self.window.minsize(620, 320)
        self.window.transient(parent)

        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        columns = ("process", "pid", "title")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("process", text="进程")
        self.tree.heading("pid", text="PID")
        self.tree.heading("title", text="窗口标题")
        self.tree.column("process", width=150, stretch=False)
        self.tree.column("pid", width=75, stretch=False, anchor="center")
        self.tree.column("title", width=460)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        buttons = ttk.Frame(self.window, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="刷新", command=self.refresh).pack(side="left")
        ttk.Button(buttons, text="取消", command=self.cancel).pack(side="right")
        ttk.Button(buttons, text="选择", command=self.select).pack(side="right", padx=(0, 6))
        self.tree.bind("<Double-Button-1>", lambda _event: self.select())
        self.window.bind("<Escape>", lambda _event: self.cancel())
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)
        self.refresh()
        self.window.grab_set()
        self.window.focus_force()

    def refresh(self) -> None:
        selection = self.tree.selection()
        selected_handle = self.windows.get(selection[0]).handle if selection and selection[0] in self.windows else None
        self.tree.delete(*self.tree.get_children())
        self.windows.clear()
        for target in list_app_windows():
            item_id = str(target.handle)
            self.windows[item_id] = target
            self.tree.insert("", "end", iid=item_id, values=(target.process_name, target.process_id, target.title))
        if selected_handle is not None and self.tree.exists(str(selected_handle)):
            self.tree.selection_set(str(selected_handle))
        elif self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)

    def select(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        target = self.windows.get(selection[0])
        if target is None:
            return
        self._finish(target)

    def cancel(self) -> None:
        self._finish(None)

    def _finish(self, target: AppWindow | None) -> None:
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.window.destroy()
        self.callback(target)


class PreviewCanvas(tk.Canvas):
    def __init__(self, parent, select_callback, edit_callback, **kwargs) -> None:
        super().__init__(parent, background="#161b22", highlightthickness=0, **kwargs)
        self.image = None
        self.puzzle: Puzzle | None = None
        self.solution: Solution | None = None
        self.photo = None
        self.scale = 1.0
        self.offset = (0.0, 0.0)
        self.selected_id: int | None = None
        self.select_callback = select_callback
        self.edit_callback = edit_callback
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._select)
        self.bind("<Double-Button-1>", self._edit)

    def set_content(self, image, puzzle: Puzzle | None, solution: Solution | None) -> None:
        self.image = image
        self.puzzle = puzzle
        self.solution = solution
        self.redraw()

    def _image_point(self, event: tk.Event) -> tuple[float, float] | None:
        if self.image is None:
            return None
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            return x, y
        return None

    def _nearest(self, point: tuple[float, float]) -> Island | None:
        if self.puzzle is None:
            return None
        x, y = point
        candidates = sorted(
            self.puzzle.islands,
            key=lambda island: (island.x - x) ** 2 + (island.y - y) ** 2,
        )
        if not candidates:
            return None
        nearest = candidates[0]
        distance = ((nearest.x - x) ** 2 + (nearest.y - y) ** 2) ** 0.5
        return nearest if distance <= max(12.0, nearest.radius * 1.5) else None

    def _select(self, event: tk.Event) -> None:
        point = self._image_point(event)
        if point is None:
            return
        island = self._nearest(point)
        self.selected_id = island.id if island is not None else None
        self.select_callback(self.selected_id)
        self.redraw()

    def _edit(self, event: tk.Event) -> None:
        point = self._image_point(event)
        if point is None:
            return
        self.edit_callback(self._nearest(point), point)

    def redraw(self) -> None:
        self.delete("all")
        if self.image is None:
            return
        overlay = render_overlay(self.image, self.puzzle, self.solution)
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        image_height, image_width = overlay.shape[:2]
        self.scale = min(width / image_width, height / image_height, 1.8)
        target_width = max(1, round(image_width * self.scale))
        target_height = max(1, round(image_height * self.scale))
        interpolation = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        display = cv2.resize(overlay, (target_width, target_height), interpolation=interpolation)
        self.offset = ((width - target_width) / 2, (height - target_height) / 2)
        self.photo = ImageTk.PhotoImage(bgr_to_pil(display))
        self.create_image(*self.offset, image=self.photo, anchor="nw")
        if self.selected_id is not None and self.puzzle is not None:
            island = next((item for item in self.puzzle.islands if item.id == self.selected_id), None)
            if island is not None:
                x = self.offset[0] + island.x * self.scale
                y = self.offset[1] + island.y * self.scale
                radius = (island.radius + 7) * self.scale
                self.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    outline="#facc15",
                    width=3,
                )


class BridgeAssistantApp:
    MODE_LABELS = {
        "靠近端点点击（必要时中点）": InputMode.NEAR_ENDPOINT,
        "拖拽岛到岛（兼容）": InputMode.DRAG,
        "点击桥中点（兼容）": InputMode.MIDPOINT,
        "依次点击两岛（兼容）": InputMode.ENDPOINTS,
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("数桥识别助手")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)
        self.region: ScreenRegion | None = None
        self.client_region: ScreenRegion | None = None
        self.board_offset: ScreenRegion | None = None
        self.target_window: AppWindow | None = None
        self.capture_image = None
        self.capture_screen_region: ScreenRegion | None = None
        self.source_image = None
        self.recognition: Recognition | None = None
        self.puzzle: Puzzle | None = None
        self.solution: Solution | None = None
        self.stop_event = threading.Event()
        self.capture_events: queue.Queue[object] = queue.Queue()
        self.execution_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.capturing = False
        self.executing = False
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        toolbar = ttk.Frame(self.root, padding=(10, 9))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="选择窗口", command=self.select_window).pack(side="left")
        self.capture_button = ttk.Button(toolbar, text="采集画面", command=self.capture_window)
        self.capture_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="框选题面", command=self.capture_region).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="打开图片", command=self.open_image).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="识别画面", command=self.run_recognition).pack(side="left", padx=(14, 0))
        ttk.Button(toolbar, text="求解", command=self.run_solver).pack(side="left", padx=(6, 0))
        self.execute_button = ttk.Button(toolbar, text="执行到应用", command=self.execute)
        self.execute_button.pack(side="left", padx=(14, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Label(toolbar, text="操作方式").pack(side="left")
        self.mode_var = tk.StringVar(value="靠近端点点击（必要时中点）")
        mode_box = ttk.Combobox(
            toolbar,
            state="readonly",
            width=19,
            textvariable=self.mode_var,
            values=list(self.MODE_LABELS),
        )
        mode_box.pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, text="间隔").pack(side="left", padx=(12, 0))
        self.interval_var = tk.DoubleVar(value=0.18)
        ttk.Spinbox(toolbar, from_=0.05, to=2.0, increment=0.05, width=6, textvariable=self.interval_var).pack(
            side="left", padx=(6, 0)
        )

        separator = ttk.Separator(self.root, orient="horizontal")
        separator.pack(fill="x")
        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True)
        preview_frame = ttk.Frame(pane)
        side = ttk.Frame(pane, width=310, padding=(10, 10))
        pane.add(preview_frame, weight=4)
        pane.add(side, weight=1)

        self.preview = PreviewCanvas(preview_frame, self.select_island, self.edit_at_point)
        self.preview.pack(fill="both", expand=True)

        ttk.Label(side, text="识别结果", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        columns = ("value", "confidence", "position")
        self.tree = ttk.Treeview(side, columns=columns, show="tree headings", height=18, selectmode="browse")
        self.tree.heading("#0", text="#")
        self.tree.heading("value", text="数字")
        self.tree.heading("confidence", text="置信度")
        self.tree.heading("position", text="坐标")
        self.tree.column("#0", width=42, stretch=False, anchor="center")
        self.tree.column("value", width=52, stretch=False, anchor="center")
        self.tree.column("confidence", width=70, stretch=False, anchor="center")
        self.tree.column("position", width=105, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=(8, 8))
        self.tree.tag_configure("low", foreground="#c05a00")
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        self.tree.bind("<Double-Button-1>", lambda _event: self.edit_selected())

        editor = ttk.Frame(side)
        editor.pack(fill="x")
        ttk.Button(editor, text="修改数字", command=self.edit_selected).pack(side="left")
        ttk.Button(editor, text="添加岛", command=self.add_island_dialog).pack(side="left", padx=5)
        ttk.Button(editor, text="删除岛", command=self.delete_selected).pack(side="left")
        ttk.Button(side, text="保存预览", command=self.save_preview).pack(fill="x", pady=(12, 0))

        status_frame = ttk.Frame(self.root, padding=(10, 6))
        status_frame.pack(fill="x")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        self.source_var = tk.StringVar(value="未载入题面")
        ttk.Label(status_frame, textvariable=self.source_var).pack(side="right")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def select_window(self) -> None:
        if self.executing or self.capturing:
            return
        WindowSelector(self.root, self._window_selected)

    def _window_selected(self, target: AppWindow | None) -> None:
        if target is None:
            self.set_status("已取消选择窗口")
            return
        self.target_window = target
        self.capture_image = None
        self.capture_screen_region = None
        self.region = None
        self.client_region = None
        self.board_offset = None
        self.source_image = None
        self.recognition = None
        self.puzzle = None
        self.solution = None
        self.preview.selected_id = None
        self.refresh()
        self.source_var.set(f"{target.process_name} · {target.title[:36]}")
        self.set_status("已绑定窗口，请点击“采集画面”获取当前画面")

    def capture_window(self) -> None:
        if self.executing or self.capturing:
            return
        if self.target_window is None:
            messagebox.showinfo("尚未选择窗口", "请先选择要采集的应用窗口。", parent=self.root)
            return
        target = self.target_window
        self.capture_image = None
        self.capture_screen_region = None
        self.region = None
        self.client_region = None
        self.board_offset = None
        self.source_image = None
        self.recognition = None
        self.puzzle = None
        self.solution = None
        self.preview.selected_id = None
        self.refresh()
        self.root.withdraw()
        self.root.update_idletasks()

        def work() -> None:
            try:
                time.sleep(0.2)
                activate_window(target)
                time.sleep(0.35)
                client_region = window_client_region(target)
                client_image = pil_to_bgr(grab_screen(client_region))
                self.capture_events.put(("success", (client_image, client_region)))
            except Exception as error:
                self.capture_events.put(("error", error))

        self.capturing = True
        self.capture_button.configure(state="disabled")
        self.set_status("正在激活窗口并采集客户区画面...")
        threading.Thread(target=work, name="bridge-capture-worker", daemon=True).start()
        self.root.after(40, self._poll_capture_events)

    def _poll_capture_events(self) -> None:
        try:
            payload = self.capture_events.get_nowait()
        except queue.Empty:
            self.root.after(40, self._poll_capture_events)
            return
        self.capturing = False
        self.capture_button.configure(state="normal")
        self.root.deiconify()
        self.root.lift()
        outcome, detail = payload
        if outcome == "error":
            self.set_status(str(detail))
            messagebox.showerror("采集失败", str(detail), parent=self.root)
            return
        client_image, client_region = detail
        self.capture_image = client_image
        self.capture_screen_region = client_region
        self.client_region = client_region
        self.region = None
        self.board_offset = None
        self.source_image = client_image
        self.recognition = None
        self.puzzle = None
        self.solution = None
        self.preview.selected_id = None
        self.refresh()
        self.source_var.set(
            f"{self.target_window.process_name} · {self.target_window.title[:28]} · "
            f"{client_region.width}x{client_region.height}"
        )
        self.set_status("画面已采集，请点击“识别画面”自动定位题面")

    def capture_region(self) -> None:
        if self.executing or self.capturing:
            return
        self.root.withdraw()

        def begin() -> None:
            try:
                RegionSelector(self.root, self._region_selected)
            except Exception as error:
                self.root.deiconify()
                messagebox.showerror("截图失败", str(error), parent=self.root)

        self.root.after(350, begin)

    def _region_selected(self, region: ScreenRegion | None, image, target: AppWindow | None) -> None:
        self.root.deiconify()
        self.root.lift()
        if region is None or image is None:
            self.set_status("已取消框选")
            return
        captured = pil_to_bgr(image)
        self.capture_image = captured
        self.capture_screen_region = region
        self.region = None
        self.client_region = None
        self.board_offset = None
        self.target_window = target
        self.source_image = captured
        self.recognition = None
        self.puzzle = None
        self.solution = None
        self.preview.selected_id = None
        self.refresh()
        target_label = target.title if target is not None else "未识别应用"
        self.source_var.set(f"{target_label[:32]} · {region.width}x{region.height}")
        self.set_status("题面已框选，请点击“识别画面”")

    def open_image(self) -> None:
        if self.executing or self.capturing:
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="打开题面截图",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("打开失败", "无法读取该图片", parent=self.root)
            return
        self.capture_image = image
        self.capture_screen_region = None
        self.region = None
        self.client_region = None
        self.board_offset = None
        self.target_window = None
        self.source_image = image
        self.recognition = None
        self.puzzle = None
        self.solution = None
        self.preview.selected_id = None
        self.refresh()
        self.source_var.set(Path(path).name + "（仅预览）")
        self.set_status("图片已打开，请点击“识别画面”")

    def run_recognition(self) -> None:
        if self.capturing:
            return
        if self.capture_image is None:
            messagebox.showinfo("没有题面", "请先采集画面、框选题面或打开截图", parent=self.root)
            return
        self.set_status("正在识别...")
        self.region = None
        self.board_offset = None
        self.source_image = self.capture_image
        self.recognition = None
        self.puzzle = None
        self.solution = None
        self.preview.selected_id = None
        self.refresh()
        try:
            if self.client_region is not None:
                located = locate_board(self.capture_image)
                self.source_image = located.image
                self.recognition = located.recognition
                self.puzzle = located.puzzle
                self.board_offset = located.region
                self.region = ScreenRegion(
                    self.client_region.left + located.region.left,
                    self.client_region.top + located.region.top,
                    located.region.width,
                    located.region.height,
                )
            else:
                self.source_image = self.capture_image
                self.recognition = recognize(self.capture_image)
                self.puzzle = build_puzzle(self.recognition.islands)
                self.board_offset = None
                self.region = self.capture_screen_region
            self.solution = None
            self.preview.selected_id = None
            self.refresh()
            warning = "；".join(self.recognition.warnings)
            suffix = f"；{warning}" if warning else ""
            self.set_status(f"识别到 {len(self.puzzle.islands)} 个岛、{len(self.puzzle.edges)} 条候选连接{suffix}")
        except BridgeAssistantError as error:
            self.source_image = self.capture_image
            self.recognition = None
            self.puzzle = None
            self.solution = None
            self.region = None
            self.board_offset = None
            self.preview.selected_id = None
            self.refresh()
            self.set_status(str(error))
            messagebox.showerror("识别失败", str(error), parent=self.root)

    def run_solver(self) -> None:
        if self.capturing:
            return
        if self.puzzle is None:
            messagebox.showinfo("没有题面", "请先识别题面", parent=self.root)
            return
        self.set_status("正在求解...")
        try:
            self.solution = solve(self.puzzle)
            self.refresh()
            bridge_total = sum(self.solution.bridges.values())
            self.set_status(f"已求解：{self.solution.active_count} 条连接，共 {bridge_total} 座桥")
        except BridgeAssistantError as error:
            self.solution = None
            self.refresh()
            self.set_status(str(error))
            messagebox.showerror("无法求解", str(error) + "\n\n请检查橙色低置信数字。", parent=self.root)

    def refresh(self) -> None:
        self.preview.set_content(self.source_image, self.puzzle, self.solution)
        selected = self.preview.selected_id
        self.tree.delete(*self.tree.get_children())
        if self.puzzle is None:
            return
        for island in self.puzzle.islands:
            tags = ("low",) if island.confidence < 0.58 else ()
            self.tree.insert(
                "",
                "end",
                iid=str(island.id),
                text=str(island.id + 1),
                values=(island.value, f"{island.confidence:.0%}", f"{island.x:.0f}, {island.y:.0f}"),
                tags=tags,
            )
        if selected is not None and self.tree.exists(str(selected)):
            self.tree.selection_set(str(selected))
            self.tree.see(str(selected))

    def select_island(self, island_id: int | None) -> None:
        if island_id is None:
            self.tree.selection_remove(self.tree.selection())
        elif self.tree.exists(str(island_id)):
            self.tree.selection_set(str(island_id))
            self.tree.see(str(island_id))

    def _tree_selected(self, _event=None) -> None:
        selection = self.tree.selection()
        self.preview.selected_id = int(selection[0]) if selection else None
        self.preview.redraw()

    def _ask_value(self, initial: int = 1) -> int | None:
        return simpledialog.askinteger(
            "岛屿数字",
            "数字（1-8）",
            parent=self.root,
            minvalue=1,
            maxvalue=8,
            initialvalue=initial,
        )

    def edit_at_point(self, island: Island | None, point: tuple[float, float]) -> None:
        if self.puzzle is None:
            return
        if island is None:
            value = self._ask_value()
            if value is not None:
                self._add_island(point, value)
            return
        value = self._ask_value(island.value)
        if value is not None:
            island.value = value
            island.confidence = 1.0
            self._rebuild_after_edit()

    def edit_selected(self) -> None:
        if self.puzzle is None:
            return
        selection = self.tree.selection()
        if not selection:
            return
        island = next(item for item in self.puzzle.islands if item.id == int(selection[0]))
        value = self._ask_value(island.value)
        if value is not None:
            island.value = value
            island.confidence = 1.0
            self._rebuild_after_edit()

    def add_island_dialog(self) -> None:
        messagebox.showinfo("添加岛", "在题面预览的空白位置双击，然后输入数字。", parent=self.root)

    def _add_island(self, point: tuple[float, float], value: int) -> None:
        if self.puzzle is None:
            return
        radius = statistics.median(item.radius for item in self.puzzle.islands) if self.puzzle.islands else 16.0
        self.puzzle.islands.append(Island(len(self.puzzle.islands), point[0], point[1], value, radius, 1.0))
        self._rebuild_after_edit()

    def delete_selected(self) -> None:
        if self.puzzle is None:
            return
        selection = self.tree.selection()
        if not selection:
            return
        selected_id = int(selection[0])
        self.puzzle.islands = [item for item in self.puzzle.islands if item.id != selected_id]
        for index, island in enumerate(sorted(self.puzzle.islands, key=lambda item: (item.y, item.x))):
            island.id = index
        self.preview.selected_id = None
        self._rebuild_after_edit()

    def _rebuild_after_edit(self) -> None:
        if self.puzzle is None:
            return
        self.puzzle = build_puzzle(self.puzzle.islands)
        self.solution = None
        self.refresh()
        self.set_status("题面已修改，请重新求解")

    def save_preview(self) -> None:
        preview_image = self.source_image if self.source_image is not None else self.capture_image
        if preview_image is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存预览",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png")],
        )
        if not path:
            return
        rendered = render_overlay(preview_image, self.puzzle, self.solution)
        success, encoded = cv2.imencode(".png", rendered)
        if not success:
            messagebox.showerror("保存失败", "无法编码预览图", parent=self.root)
            return
        encoded.tofile(path)
        self.set_status(f"预览已保存：{Path(path).name}")

    def execute(self) -> None:
        if self.capturing:
            return
        if self.executing:
            self.stop_event.set()
            return
        if self.region is None:
            messagebox.showinfo("没有屏幕坐标", "请从已选窗口采集画面，或使用“框选题面”。", parent=self.root)
            return
        if self.target_window is None:
            messagebox.showinfo("没有目标应用", "请重新选择并采集窗口，或重新框选题面。", parent=self.root)
            return
        if self.puzzle is None or self.solution is None or self.source_image is None:
            messagebox.showinfo("尚未求解", "请先识别并求解题面。", parent=self.root)
            return
        mode = self.MODE_LABELS[self.mode_var.get()]
        try:
            interval = max(0.0, float(self.interval_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("无效间隔", "操作间隔必须是数字。", parent=self.root)
            return
        try:
            preview_actions = build_mouse_actions(self.puzzle, self.solution, self.region, mode)
        except BridgeAssistantError as error:
            messagebox.showerror("无法生成鼠标操作", str(error), parent=self.root)
            return
        if not preview_actions:
            messagebox.showinfo("无需操作", "解答中没有需要输入的桥。", parent=self.root)
            return
        operation_count = sum(self.solution.bridges.values())
        low_count = sum(island.confidence < 0.58 for island in self.puzzle.islands)
        warning = f"\n有 {low_count} 个低置信数字，请确认预览。" if low_count else ""
        if self.recognition is not None and self.recognition.warnings:
            warning += "\n识别器警告：" + "；".join(self.recognition.warnings)
        protocol_warning = ""
        if any(action.bridge_count == 2 for action in preview_actions):
            protocol_warning = "\n双桥会对同一连接重复操作两次，请确认目标应用按 0→1→2 增加桥数。"
        confirmed = messagebox.askyesno(
            "执行到目标应用",
            f"目标：{self.target_window.title}\n将执行 {operation_count} 次鼠标操作。{warning}{protocol_warning}"
            "\n\nEsc 可中止；移动鼠标到屏幕左上角也会触发保护。",
            parent=self.root,
        )
        if not confirmed:
            return
        self.executing = True
        self.stop_event.clear()
        self.execute_button.configure(text="停止")
        self.set_status("正在切换到目标应用...")
        self.root.withdraw()
        region = self.region
        source_image = self.source_image
        islands = list(self.puzzle.islands)
        stop_event = self.stop_event
        target_window = self.target_window
        captured_client_region = self.client_region
        board_offset = self.board_offset
        puzzle = self.puzzle
        solution = self.solution

        def work() -> None:
            try:
                time.sleep(0.35)
                activate_window(target_window)
                time.sleep(0.35)
                execution_region = region
                if captured_client_region is not None and board_offset is not None:
                    current_client = window_client_region(target_window)
                    if (
                        current_client.width != captured_client_region.width
                        or current_client.height != captured_client_region.height
                    ):
                        raise BridgeAssistantError(
                            "目标窗口大小或 DPI 已改变，请重新采集画面后再执行"
                        )
                    execution_region = ScreenRegion(
                        current_client.left + board_offset.left,
                        current_client.top + board_offset.top,
                        board_offset.width,
                        board_offset.height,
                    )
                actions = build_mouse_actions(puzzle, solution, execution_region, mode)
                if not actions:
                    raise BridgeAssistantError("解答中没有需要输入的桥")
                current = pil_to_bgr(grab_screen(execution_region))
                similarity = verify_scene(source_image, current, islands)
                verifier = VisualActionVerifier(
                    current,
                    execution_region,
                    lambda: pil_to_bgr(grab_screen(execution_region)),
                )

                def progress(done: int, total: int) -> None:
                    self.execution_events.put(
                        ("progress", (done, total, similarity))
                    )

                execute_mouse_actions(
                    actions,
                    interval=interval,
                    stop_event=stop_event,
                    progress=progress,
                    operation_confirmed=verifier,
                    prepare_retry=lambda _action, _attempt: activate_window(target_window),
                    max_retries=DEFAULT_MAX_RETRIES,
                    verification_cursor=(execution_region.right - 4, execution_region.bottom - 4),
                )
                self.execution_events.put(("finished", None))
            except Exception as error:
                self.execution_events.put(("finished", error))

        threading.Thread(target=work, name="bridge-mouse-worker", daemon=True).start()
        self.root.after(40, self._poll_execution_events)

    def _poll_execution_events(self) -> None:
        finished = False
        while True:
            try:
                kind, payload = self.execution_events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                done, total, similarity = payload
                self.set_status(f"正在执行 {done}/{total}（校验 {similarity:.0%}）")
            elif kind == "finished":
                finished = True
                self._execution_finished(payload)
        if self.executing and not finished:
            self.root.after(40, self._poll_execution_events)

    def _execution_finished(self, error: Exception | None) -> None:
        self.executing = False
        self.execute_button.configure(text="执行到应用")
        self.root.deiconify()
        self.root.lift()
        if error is None:
            self.set_status("鼠标操作已完成")
        elif isinstance(error, InterruptedError):
            self.set_status("鼠标操作已中止")
        else:
            self.set_status(str(error))
            messagebox.showerror("执行中止", str(error), parent=self.root)


def main() -> None:
    enable_dpi_awareness()
    root = tk.Tk()
    BridgeAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
