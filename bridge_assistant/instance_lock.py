from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


MUTEX_NAME = "Local\\BridgeAssistant-BridgeFix231-AutoRunner"
ERROR_ALREADY_EXISTS = 183


class AnotherInstanceError(RuntimeError):
    pass


class SingleInstanceLock:
    """Hold a per-user Windows mutex for one automatic runner."""

    def __init__(self, name: str = MUTEX_NAME) -> None:
        self.name = name
        self._handle: int | None = None

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def acquire(self) -> None:
        if self._handle is not None:
            return
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        create_mutex.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        ctypes.set_last_error(0)
        handle = create_mutex(None, False, self.name)
        error = ctypes.get_last_error()
        if not handle:
            raise OSError(error, "CreateMutexW failed")
        if error == ERROR_ALREADY_EXISTS:
            close_handle(handle)
            raise AnotherInstanceError(
                "Another Bridge Assistant automatic runner is already active"
            )
        self._handle = int(handle)

    def close(self) -> None:
        if self._handle is None or os.name != "nt":
            self._handle = None
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        handle = self._handle
        self._handle = None
        if not close_handle(handle):
            raise OSError(ctypes.get_last_error(), "CloseHandle failed")
