from __future__ import annotations

import ctypes
import ntpath
import os
import time
from contextlib import contextmanager
from ctypes import wintypes
from typing import Iterator

from .model import AppWindow, ScreenRegion, TargetWindowError


GA_ROOT = 2
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SW_RESTORE = 9
SW_SHOW = 5
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4

_USER32_CONFIGURED = False
_KERNEL32_CONFIGURED = False
_DWMAPI_CONFIGURED = False


def _user32():
    global _USER32_CONFIGURED
    if not hasattr(ctypes, "windll"):
        return None
    user32 = ctypes.windll.user32
    if not _USER32_CONFIGURED:
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetClientRect.restype = wintypes.BOOL
        user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        user32.ClientToScreen.restype = wintypes.BOOL
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.GetSystemMetrics.restype = ctypes.c_int
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        user32.AttachThreadInput.restype = wintypes.BOOL
        user32.BringWindowToTop.argtypes = [wintypes.HWND]
        user32.BringWindowToTop.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.SetActiveWindow.argtypes = [wintypes.HWND]
        user32.SetActiveWindow.restype = wintypes.HWND
        user32.SetFocus.argtypes = [wintypes.HWND]
        user32.SetFocus.restype = wintypes.HWND
        if hasattr(user32, "SetThreadDpiAwarenessContext"):
            user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.EnumWindows.restype = wintypes.BOOL
        _USER32_CONFIGURED = True
    return user32


def _kernel32():
    global _KERNEL32_CONFIGURED
    if not hasattr(ctypes, "windll"):
        return None
    kernel32 = ctypes.windll.kernel32
    if not _KERNEL32_CONFIGURED:
        kernel32.GetCurrentThreadId.argtypes = []
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        _KERNEL32_CONFIGURED = True
    return kernel32


def _dwmapi():
    global _DWMAPI_CONFIGURED
    if not hasattr(ctypes, "windll"):
        return None
    try:
        dwmapi = ctypes.windll.dwmapi
    except (AttributeError, OSError):
        return None
    if not _DWMAPI_CONFIGURED:
        dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
        _DWMAPI_CONFIGURED = True
    return dwmapi


@contextmanager
def _physical_coordinate_context(user32) -> Iterator[None]:
    setter = getattr(user32, "SetThreadDpiAwarenessContext", None)
    previous = None
    if setter is not None:
        try:
            previous = setter(ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
        except (AttributeError, OSError):
            previous = None
    try:
        yield
    finally:
        if setter is not None and previous:
            try:
                setter(previous)
            except (AttributeError, OSError):
                pass


def _root_window(handle: int) -> int:
    user32 = _user32()
    if user32 is None or not handle:
        return 0
    return int(user32.GetAncestor(wintypes.HWND(handle), GA_ROOT) or handle)


def _window_process_id(handle: int) -> int:
    user32 = _user32()
    if user32 is None or not handle:
        return 0
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(process_id))
    return int(process_id.value)


def _window_title(handle: int) -> str:
    user32 = _user32()
    if user32 is None or not handle:
        return ""
    length = int(user32.GetWindowTextLengthW(wintypes.HWND(handle)))
    buffer = ctypes.create_unicode_buffer(max(1, length + 1))
    user32.GetWindowTextW(wintypes.HWND(handle), buffer, len(buffer))
    return " ".join(buffer.value.split())


def _query_process_image_path(process_id: int) -> str | None:
    kernel32 = _kernel32()
    if kernel32 is None or process_id <= 0:
        return None
    process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not process:
        return None
    try:
        capacity = 32768
        size = wintypes.DWORD(capacity)
        buffer = ctypes.create_unicode_buffer(capacity)
        if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return None
        return buffer.value[: size.value]
    finally:
        kernel32.CloseHandle(process)


def _window_process_name(process_id: int) -> str:
    path = _query_process_image_path(process_id)
    if path:
        basename = ntpath.basename(path.rstrip("\\/"))
        if basename:
            return basename
    return f"PID {process_id}"


def _is_window_cloaked(handle: int) -> bool:
    dwmapi = _dwmapi()
    if dwmapi is None:
        return False
    cloaked = wintypes.DWORD()
    try:
        result = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(handle),
            DWMWA_CLOAKED,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
    except (AttributeError, OSError):
        return False
    return result == 0 and bool(cloaked.value)


def _enumerate_window_handles(user32) -> list[int]:
    callback_type = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    handles: list[int] = []

    @callback_type(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(handle, _parameter) -> bool:
        handles.append(int(handle))
        return True

    user32.EnumWindows(collect, 0)
    return handles


def _has_positive_client_area(handle: int, user32) -> bool:
    rect = wintypes.RECT()
    if not user32.GetClientRect(wintypes.HWND(handle), ctypes.byref(rect)):
        return False
    return rect.right > rect.left and rect.bottom > rect.top


def _window_sort_key(window: AppWindow) -> tuple[str, str, int, int]:
    return (
        window.process_name.casefold(),
        window.title.casefold(),
        window.process_id,
        window.handle,
    )


def list_app_windows(*, exclude_process_id: int | None = None) -> list[AppWindow]:
    """List selectable top-level application windows in a stable order."""

    user32 = _user32()
    if user32 is None:
        return []
    excluded = os.getpid() if exclude_process_id is None else exclude_process_id
    windows: list[AppWindow] = []
    seen: set[int] = set()
    for handle in _enumerate_window_handles(user32):
        if handle in seen:
            continue
        seen.add(handle)
        native_handle = wintypes.HWND(handle)
        if not user32.IsWindow(native_handle) or not user32.IsWindowVisible(native_handle):
            continue
        if int(user32.GetWindowLongW(native_handle, GWL_EXSTYLE)) & WS_EX_TOOLWINDOW:
            continue
        if _is_window_cloaked(handle) or not _has_positive_client_area(handle, user32):
            continue
        process_id = _window_process_id(handle)
        if process_id <= 0 or process_id == excluded:
            continue
        title = _window_title(handle)
        if not title:
            continue
        windows.append(AppWindow(handle, process_id, title, _window_process_name(process_id)))
    return sorted(windows, key=_window_sort_key)


def _read_client_region(handle: int, user32) -> ScreenRegion:
    rect = wintypes.RECT()
    native_handle = wintypes.HWND(handle)
    if not user32.GetClientRect(native_handle, ctypes.byref(rect)):
        raise TargetWindowError("无法读取目标窗口的客户区。")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        raise TargetWindowError("目标窗口的客户区为空，请先还原窗口。")
    origin = wintypes.POINT(rect.left, rect.top)
    if not user32.ClientToScreen(native_handle, ctypes.byref(origin)):
        raise TargetWindowError("无法将目标窗口客户区转换为屏幕坐标。")
    return ScreenRegion(int(origin.x), int(origin.y), width, height)


def _virtual_screen_region(user32) -> ScreenRegion:
    return ScreenRegion(
        int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)),
    )


def _region_inside(region: ScreenRegion, bounds: ScreenRegion) -> bool:
    return (
        bounds.width > 0
        and bounds.height > 0
        and region.left >= bounds.left
        and region.top >= bounds.top
        and region.right <= bounds.right
        and region.bottom <= bounds.bottom
    )


def window_client_region(target: AppWindow) -> ScreenRegion:
    """Return the target client area in physical virtual-screen coordinates."""

    user32 = _user32()
    if user32 is None:
        raise TargetWindowError("窗口采集仅支持 Windows。")
    validate_window(target)
    with _physical_coordinate_context(user32):
        region = _read_client_region(target.handle, user32)
        virtual_screen = _virtual_screen_region(user32)
    if not _region_inside(region, virtual_screen):
        raise TargetWindowError("目标窗口未完整位于屏幕内，请移动或调整窗口大小后重试。")
    return region


def window_at_screen_point(x: int, y: int, *, exclude_process_id: int | None = None) -> AppWindow | None:
    """Return the top-level app window under a physical screen coordinate."""

    user32 = _user32()
    if user32 is None:
        return None
    point = wintypes.POINT(int(x), int(y))
    handle = _root_window(int(user32.WindowFromPoint(point) or 0))
    if not handle or not user32.IsWindow(wintypes.HWND(handle)):
        return None
    process_id = _window_process_id(handle)
    excluded = os.getpid() if exclude_process_id is None else exclude_process_id
    if process_id == excluded:
        return None
    return AppWindow(
        handle,
        process_id,
        _window_title(handle) or f"PID {process_id}",
        _window_process_name(process_id),
    )


def validate_window(target: AppWindow) -> None:
    user32 = _user32()
    if user32 is None:
        return
    handle = wintypes.HWND(target.handle)
    if not user32.IsWindow(handle):
        raise TargetWindowError("目标应用窗口已关闭，请重新选择窗口。")
    if _window_process_id(target.handle) != target.process_id:
        raise TargetWindowError("目标窗口已被其他进程替换，请重新选择窗口。")


def activate_window(target: AppWindow, timeout: float = 2.0) -> None:
    """Restore and foreground the captured application window on Windows."""

    user32 = _user32()
    if user32 is None:
        return
    validate_window(target)
    handle = wintypes.HWND(target.handle)
    if user32.IsIconic(handle):
        user32.ShowWindow(handle, SW_RESTORE)
    else:
        user32.ShowWindow(handle, SW_SHOW)

    foreground = _root_window(int(user32.GetForegroundWindow() or 0))
    kernel32 = _kernel32()
    current_thread = int(kernel32.GetCurrentThreadId()) if kernel32 is not None else 0
    foreground_thread = int(user32.GetWindowThreadProcessId(wintypes.HWND(foreground), None)) if foreground else 0
    target_thread = int(user32.GetWindowThreadProcessId(handle, None))
    attached: list[int] = []
    try:
        for thread_id in {foreground_thread, target_thread}:
            if thread_id and thread_id != current_thread:
                if user32.AttachThreadInput(current_thread, thread_id, True):
                    attached.append(thread_id)
        user32.BringWindowToTop(handle)
        user32.SetForegroundWindow(handle)
        user32.SetActiveWindow(handle)
        user32.SetFocus(handle)
    finally:
        for thread_id in reversed(attached):
            user32.AttachThreadInput(current_thread, thread_id, False)

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() <= deadline:
        if _root_window(int(user32.GetForegroundWindow() or 0)) == target.handle:
            return
        user32.BringWindowToTop(handle)
        user32.SetForegroundWindow(handle)
        time.sleep(0.05)
    raise TargetWindowError(f"无法激活目标应用“{target.title}”，请重新选择窗口后再试。")
