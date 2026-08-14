from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bridge_assistant.app import BridgeAssistantApp
from bridge_assistant.model import AppWindow, ScreenRegion


class StubValue:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class StubPreview:
    selected_id = 5


def bare_app() -> BridgeAssistantApp:
    app = BridgeAssistantApp.__new__(BridgeAssistantApp)
    app.target_window = AppWindow(10, 20, "Target", "target.exe")
    app.capture_image = np.full((80, 120, 3), 240, dtype=np.uint8)
    app.capture_screen_region = ScreenRegion(10, 20, 120, 80)
    app.region = ScreenRegion(30, 40, 50, 60)
    app.client_region = ScreenRegion(10, 20, 120, 80)
    app.board_offset = ScreenRegion(20, 20, 50, 60)
    app.source_image = np.zeros((60, 50, 3), dtype=np.uint8)
    app.recognition = object()
    app.puzzle = object()
    app.solution = object()
    app.preview = StubPreview()
    app.source_var = StubValue()
    app.status_var = StubValue()
    app.root = SimpleNamespace(update_idletasks=lambda: None)
    app.refresh = lambda: None
    return app


def test_selecting_a_window_binds_only_and_clears_previous_capture() -> None:
    app = bare_app()
    target = AppWindow(11, 21, "New target", "new.exe")

    app._window_selected(target)

    assert app.target_window == target
    assert app.capture_image is None
    assert app.source_image is None
    assert app.recognition is None
    assert app.puzzle is None
    assert app.solution is None
    assert app.region is None
    assert "采集画面" in app.status_var.value


def test_capture_event_keeps_full_client_image_without_recognition() -> None:
    app = bare_app()
    image = np.full((90, 140, 3), 233, dtype=np.uint8)
    client = ScreenRegion(101, 202, 140, 90)
    app.capture_events = SimpleNamespace(get_nowait=lambda: ("success", (image, client)))
    app.capture_button = SimpleNamespace(configure=lambda **_kwargs: None)
    app.root = SimpleNamespace(
        after=lambda *_args: None,
        deiconify=lambda: None,
        lift=lambda: None,
        update_idletasks=lambda: None,
    )
    app.capturing = True

    app._poll_capture_events()

    assert app.capture_image is image
    assert app.source_image is image
    assert app.client_region == client
    assert app.region is None
    assert app.board_offset is None
    assert app.recognition is None
    assert app.puzzle is None
    assert app.solution is None
    assert "识别画面" in app.status_var.value
