from __future__ import annotations

import cv2
import numpy as np
import pytest

from bridge_assistant.model import InputMode, Island, MouseAction, SceneChangedError, ScreenRegion
from bridge_assistant.scene import VisualActionVerifier, bridge_corridor_change, scene_similarity, verify_scene


def test_same_scene_matches() -> None:
    image = np.full((100, 120, 3), 245, dtype=np.uint8)
    cv2.circle(image, (30, 40), 12, (20, 20, 20), 2)
    cv2.putText(image, "3", (24, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    islands = [Island(0, 30, 40, 3, 12)]
    assert scene_similarity(image, image.copy(), islands) > 0.99
    verify_scene(image, image.copy(), islands)


def test_changed_scene_is_blocked() -> None:
    reference = np.full((100, 120, 3), 255, dtype=np.uint8)
    cv2.circle(reference, (30, 40), 12, (0, 0, 0), 2)
    current = np.zeros_like(reference)
    islands = [Island(0, 30, 40, 3, 12)]
    with pytest.raises(SceneChangedError):
        verify_scene(reference, current, islands)


def test_bridge_corridor_change_detects_a_new_line() -> None:
    reference = np.full((100, 180, 3), 248, dtype=np.uint8)
    current = reference.copy()
    cv2.line(current, (30, 50), (150, 50), (235, 99, 37), 3)
    score = bridge_corridor_change(reference, current, (30, 50), (150, 50), 16)
    assert score > 0.05


def test_bridge_corridor_change_ignores_an_island_only_change() -> None:
    reference = np.full((100, 180, 3), 248, dtype=np.uint8)
    current = reference.copy()
    cv2.circle(current, (30, 50), 15, (0, 0, 0), -1)
    score = bridge_corridor_change(reference, current, (30, 50), (150, 50), 16)
    assert score == 0.0


def test_visual_action_verifier_updates_its_baseline() -> None:
    empty = np.full((100, 180, 3), 248, dtype=np.uint8)
    single = empty.copy()
    double = empty.copy()
    cv2.line(single, (30, 50), (150, 50), (235, 99, 37), 3)
    cv2.line(double, (30, 46), (150, 46), (235, 99, 37), 3)
    cv2.line(double, (30, 54), (150, 54), (235, 99, 37), 3)
    frames = iter((single, double))
    verifier = VisualActionVerifier(
        empty,
        ScreenRegion(100, 200, 180, 100),
        lambda: next(frames),
        settle_delay=0,
        sample_count=1,
    )
    action = MouseAction(InputMode.DRAG, (130, 250), (250, 250), 2, (130, 250), (250, 250), 16)
    assert verifier(action)
    assert verifier(action)
