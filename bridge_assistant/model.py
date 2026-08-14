from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True, slots=True)
class AppWindow:
    handle: int
    process_id: int
    title: str
    process_name: str = ""


@dataclass(slots=True)
class Island:
    id: int
    x: float
    y: float
    value: int
    radius: float = 16.0
    confidence: float = 1.0

    @property
    def point(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True, slots=True)
class Edge:
    id: int
    a: int
    b: int
    orientation: str


class BorderHintType(IntEnum):
    EQ = 0
    GTE = 1
    LTE = 2


@dataclass(frozen=True, slots=True)
class MagicLink:
    edge_a: int
    edge_b: int


@dataclass(frozen=True, slots=True)
class BorderHint:
    edge_ids: tuple[int, ...]
    hint: int
    type: BorderHintType = BorderHintType.EQ


@dataclass(slots=True)
class Puzzle:
    islands: list[Island]
    edges: list[Edge] = field(default_factory=list)
    alignment_tolerance: float = 0.0
    magic_links: list[MagicLink] = field(default_factory=list)
    border_hints: list[BorderHint] = field(default_factory=list)


@dataclass(slots=True)
class Solution:
    bridges: dict[int, int]

    @property
    def active_count(self) -> int:
        return sum(value > 0 for value in self.bridges.values())


@dataclass(slots=True)
class Recognition:
    islands: list[Island]
    image: np.ndarray
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocatedBoard:
    image: np.ndarray
    region: ScreenRegion
    recognition: Recognition
    puzzle: Puzzle


class InputMode(str, Enum):
    NEAR_ENDPOINT = "near_endpoint"
    DRAG = "drag"
    MIDPOINT = "midpoint"
    ENDPOINTS = "endpoints"


@dataclass(frozen=True, slots=True)
class MouseAction:
    mode: InputMode
    start: tuple[int, int]
    end: tuple[int, int]
    bridge_count: int
    observation_start: tuple[int, int] | None = None
    observation_end: tuple[int, int] | None = None
    observation_radius: float = 16.0


class BridgeAssistantError(RuntimeError):
    """Base error suitable for displaying to the user."""


class RecognitionError(BridgeAssistantError):
    pass


class SolveError(BridgeAssistantError):
    pass


class SceneChangedError(BridgeAssistantError):
    pass


class TargetWindowError(BridgeAssistantError):
    pass


class MouseInputError(BridgeAssistantError):
    pass
