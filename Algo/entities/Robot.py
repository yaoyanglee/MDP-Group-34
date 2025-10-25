from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from consts import Direction
from entities.Entity import CellState


@dataclass
class _StateLedger:
    """Tracks the robot's positional history."""

    _entries: List[CellState]

    @classmethod
    def bootstrap(cls, initial_state: CellState) -> "_StateLedger":
        return cls([initial_state])

    def expose(self) -> List[CellState]:
        return self._entries

    def origin(self) -> CellState:
        return self._entries[0]


class Robot:
    """Encapsulates mutable robot state while keeping legacy attributes available."""

    __slots__ = ("_ledger", "states")

    def __init__(self, center_x: int, center_y: int, start_direction: Direction) -> None:
        anchor = CellState(center_x, center_y, start_direction)
        self._ledger = _StateLedger.bootstrap(anchor)
        self.states = self._ledger.expose()

    def get_start_state(self) -> CellState:
        return self._ledger.origin()
