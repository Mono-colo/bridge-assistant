from __future__ import annotations

import ctypes

import numpy as np
from PIL import Image, ImageGrab

from .model import ScreenRegion


def enable_dpi_awareness() -> None:
    if not hasattr(ctypes, "windll"):
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def virtual_screen_region() -> ScreenRegion:
    if hasattr(ctypes, "windll"):
        user32 = ctypes.windll.user32
        return ScreenRegion(
            int(user32.GetSystemMetrics(76)),
            int(user32.GetSystemMetrics(77)),
            int(user32.GetSystemMetrics(78)),
            int(user32.GetSystemMetrics(79)),
        )
    image = ImageGrab.grab(all_screens=True)
    return ScreenRegion(0, 0, image.width, image.height)


def grab_screen(region: ScreenRegion | None = None) -> Image.Image:
    if region is None:
        return ImageGrab.grab(all_screens=True).convert("RGB")
    return ImageGrab.grab(bbox=region.as_bbox(), all_screens=True).convert("RGB")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(image[:, :, ::-1])

