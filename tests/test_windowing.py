from __future__ import annotations

from ctypes import wintypes

import pytest

from bridge_assistant.model import AppWindow, ScreenRegion, TargetWindowError
from bridge_assistant import windowing


class FakeListUser32:
    def __init__(self, *, hidden: set[int] | None = None, tools: set[int] | None = None) -> None:
        self.hidden = hidden or set()
        self.tools = tools or set()

    def IsWindow(self, handle) -> bool:
        return int(handle.value) != 99

    def IsWindowVisible(self, handle) -> bool:
        return int(handle.value) not in self.hidden

    def GetWindowLongW(self, handle, _index: int) -> int:
        return windowing.WS_EX_TOOLWINDOW if int(handle.value) in self.tools else 0


class FakeGeometryUser32:
    def __init__(self, client_rect=(0, 0, 800, 600), origin=(-100, 40), virtual=(-1920, 0, 3840, 1080)) -> None:
        self.client_rect = client_rect
        self.origin = origin
        self.virtual = virtual
        self.client_to_screen_calls = 0

    def GetClientRect(self, _handle, output) -> bool:
        rect = output._obj
        rect.left, rect.top, rect.right, rect.bottom = self.client_rect
        return True

    def ClientToScreen(self, _handle, output) -> bool:
        point = output._obj
        point.x, point.y = self.origin
        self.client_to_screen_calls += 1
        return True

    def GetSystemMetrics(self, metric: int) -> int:
        values = dict(zip(
            (
                windowing.SM_XVIRTUALSCREEN,
                windowing.SM_YVIRTUALSCREEN,
                windowing.SM_CXVIRTUALSCREEN,
                windowing.SM_CYVIRTUALSCREEN,
            ),
            self.virtual,
        ))
        return values[metric]


def test_app_window_keeps_three_argument_construction_compatible() -> None:
    target = AppWindow(10, 20, "Puzzle")
    assert target.process_name == ""


def test_process_name_uses_windows_basename_and_pid_fallback(monkeypatch) -> None:
    monkeypatch.setattr(windowing, "_query_process_image_path", lambda _pid: r"C:\Games\hashi.exe")
    assert windowing._window_process_name(42) == "hashi.exe"
    monkeypatch.setattr(windowing, "_query_process_image_path", lambda _pid: None)
    assert windowing._window_process_name(42) == "PID 42"


def test_list_app_windows_filters_and_sorts_stably(monkeypatch) -> None:
    user32 = FakeListUser32(hidden={60}, tools={40})
    handles = [70, 30, 50, 20, 40, 60, 80, 90, 99, 30]
    process_ids = {20: 200, 30: 200, 40: 400, 50: 500, 60: 600, 70: 700, 80: 800, 90: 900, 99: 990}
    titles = {
        20: "Second puzzle",
        30: "First puzzle",
        40: "Tool",
        50: "Cloaked",
        60: "Hidden",
        70: "Own window",
        80: "   ",
        90: "No client",
        99: "Invalid",
    }
    names = {200: "game.exe", 400: "tool.exe", 500: "cloak.exe", 600: "hidden.exe", 900: "zero.exe"}

    monkeypatch.setattr(windowing, "_user32", lambda: user32)
    monkeypatch.setattr(windowing, "_enumerate_window_handles", lambda _user32: handles)
    monkeypatch.setattr(windowing, "_window_process_id", lambda handle: process_ids[handle])
    monkeypatch.setattr(windowing, "_window_title", lambda handle: " ".join(titles[handle].split()))
    monkeypatch.setattr(windowing, "_window_process_name", lambda pid: names.get(pid, f"PID {pid}"))
    monkeypatch.setattr(windowing, "_is_window_cloaked", lambda handle: handle == 50)
    monkeypatch.setattr(windowing, "_has_positive_client_area", lambda handle, _user32: handle != 90)
    monkeypatch.setattr(windowing.os, "getpid", lambda: 700)

    windows = windowing.list_app_windows()

    assert [(item.handle, item.process_id, item.process_name) for item in windows] == [
        (30, 200, "game.exe"),
        (20, 200, "game.exe"),
    ]


def test_explicit_process_exclusion_replaces_default(monkeypatch) -> None:
    user32 = FakeListUser32()
    monkeypatch.setattr(windowing, "_user32", lambda: user32)
    monkeypatch.setattr(windowing, "_enumerate_window_handles", lambda _user32: [1, 2])
    monkeypatch.setattr(windowing, "_window_process_id", lambda handle: {1: 100, 2: 200}[handle])
    monkeypatch.setattr(windowing, "_window_title", lambda handle: f"Window {handle}")
    monkeypatch.setattr(windowing, "_window_process_name", lambda pid: f"app-{pid}.exe")
    monkeypatch.setattr(windowing, "_is_window_cloaked", lambda _handle: False)
    monkeypatch.setattr(windowing, "_has_positive_client_area", lambda _handle, _user32: True)
    monkeypatch.setattr(windowing.os, "getpid", lambda: 100)

    assert [item.handle for item in windowing.list_app_windows(exclude_process_id=200)] == [1]


def test_window_client_region_uses_client_origin_and_virtual_desktop(monkeypatch) -> None:
    user32 = FakeGeometryUser32()
    target = AppWindow(123, 456, "Puzzle", "game.exe")
    monkeypatch.setattr(windowing, "_user32", lambda: user32)
    monkeypatch.setattr(windowing, "validate_window", lambda _target: None)

    assert windowing.window_client_region(target) == ScreenRegion(-100, 40, 800, 600)
    assert user32.client_to_screen_calls == 1


@pytest.mark.parametrize(
    ("client_rect", "origin"),
    [
        ((0, 0, 0, 600), (10, 10)),
        ((0, 0, 800, 0), (10, 10)),
    ],
)
def test_window_client_region_rejects_empty_client_area(monkeypatch, client_rect, origin) -> None:
    user32 = FakeGeometryUser32(client_rect=client_rect, origin=origin)
    monkeypatch.setattr(windowing, "_user32", lambda: user32)
    monkeypatch.setattr(windowing, "validate_window", lambda _target: None)

    with pytest.raises(TargetWindowError, match="客户区为空"):
        windowing.window_client_region(AppWindow(1, 2, "Puzzle"))


@pytest.mark.parametrize(
    "origin",
    [
        (-1921, 20),
        (1121, 20),
        (-100, -1),
        (-100, 481),
    ],
)
def test_window_client_region_rejects_any_part_outside_virtual_desktop(monkeypatch, origin) -> None:
    user32 = FakeGeometryUser32(origin=origin)
    monkeypatch.setattr(windowing, "_user32", lambda: user32)
    monkeypatch.setattr(windowing, "validate_window", lambda _target: None)

    with pytest.raises(TargetWindowError, match="完整位于屏幕内"):
        windowing.window_client_region(AppWindow(1, 2, "Puzzle"))


def test_validate_window_rejects_closed_or_reused_handle(monkeypatch) -> None:
    class FakeValidateUser32:
        def __init__(self, valid: bool) -> None:
            self.valid = valid

        def IsWindow(self, _handle) -> bool:
            return self.valid

    target = AppWindow(11, 22, "Puzzle")
    monkeypatch.setattr(windowing, "_user32", lambda: FakeValidateUser32(False))
    with pytest.raises(TargetWindowError, match="已关闭"):
        windowing.validate_window(target)

    monkeypatch.setattr(windowing, "_user32", lambda: FakeValidateUser32(True))
    monkeypatch.setattr(windowing, "_window_process_id", lambda _handle: 23)
    with pytest.raises(TargetWindowError, match="其他进程替换"):
        windowing.validate_window(target)
