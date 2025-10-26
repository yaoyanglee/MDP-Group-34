from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from consts import Direction, EXPANDED_CELL, SCREENSHOT_COST
from helper import is_position_within_arena


SAFETY_RING = EXPANDED_CELL * 2


@dataclass(eq=False)
class CellState:
    """Normalized representation of a board position with orientation."""

    x: int
    y: int
    direction: Direction = Direction.NORTH
    screenshot_id: int = -1
    penalty: int = 0

    def cmp_position(self, x: int, y: int) -> bool:
        return self.x == x and self.y == y

    def is_eq(self, x: int, y: int, direction: Direction) -> bool:
        return self.x == x and self.y == y and self.direction == direction

    def set_screenshot(self, screenshot_id: int) -> None:
        self.screenshot_id = screenshot_id

    def get_dict(self) -> Dict[str, int | Direction]:
        return {"x": self.x, "y": self.y, "d": self.direction, "s": self.screenshot_id}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"CellState(x={self.x}, y={self.y}, direction={self.direction}, "
            f"screenshot={self.screenshot_id}, penalty={self.penalty})"
        )


def _compile_view_offsets() -> Dict[Direction, Dict[bool, Tuple[Tuple[int, int, Direction, int], ...]]]:
    vertical_front = (
        (0, 1 + SAFETY_RING, Direction.SOUTH, 10),
        (0, 2 + SAFETY_RING, Direction.SOUTH, 5),
        (0, 3 + SAFETY_RING, Direction.SOUTH, 0),
        (1, 2 + SAFETY_RING, Direction.SOUTH, SCREENSHOT_COST),
        (-1, 2 + SAFETY_RING, Direction.SOUTH, SCREENSHOT_COST),
    )
    vertical_retry = (
        (0, 2 + SAFETY_RING, Direction.SOUTH, 0),
        (0, 3 + SAFETY_RING, Direction.SOUTH, 0),
        (1, 2 + SAFETY_RING, Direction.SOUTH, SCREENSHOT_COST),
        (-1, 2 + SAFETY_RING, Direction.SOUTH, SCREENSHOT_COST),
    )
    horizontal_front = (
        (1 + SAFETY_RING, 0, Direction.WEST, 10),
        (2 + SAFETY_RING, 0, Direction.WEST, 5),
        (3 + SAFETY_RING, 0, Direction.WEST, 0),
        (2 + SAFETY_RING, 1, Direction.WEST, SCREENSHOT_COST),
        (2 + SAFETY_RING, -1, Direction.WEST, SCREENSHOT_COST),
    )
    horizontal_retry = (
        (2 + SAFETY_RING, 0, Direction.WEST, 0),
        (3 + SAFETY_RING, 0, Direction.WEST, 0),
        (2 + SAFETY_RING, 1, Direction.WEST, SCREENSHOT_COST),
        (2 + SAFETY_RING, -1, Direction.WEST, SCREENSHOT_COST),
    )

    def mirror_vertical(entries: Sequence[Tuple[int, int, Direction, int]]) -> Tuple[Tuple[int, int, Direction, int], ...]:
        return tuple((dx, -dy, Direction.NORTH, cost) for dx, dy, _facing, cost in entries)

    def mirror_horizontal(entries: Sequence[Tuple[int, int, Direction, int]]) -> Tuple[Tuple[int, int, Direction, int], ...]:
        return tuple((-dx, dy, Direction.EAST, cost) for dx, dy, _facing, cost in entries)

    return {
        Direction.NORTH: {False: vertical_front, True: vertical_retry},
        Direction.SOUTH: {False: mirror_vertical(vertical_front), True: mirror_vertical(vertical_retry)},
        Direction.EAST: {False: horizontal_front, True: horizontal_retry},
        Direction.WEST: {False: mirror_horizontal(horizontal_front), True: mirror_horizontal(horizontal_retry)},
    }


VIEW_OFFSETS = _compile_view_offsets()


class Obstacle(CellState):
    """Concrete obstacle that reuses CellState behaviour."""

    def __init__(self, x: int, y: int, direction: Direction, obstacle_id: int):
        super().__init__(x, y, direction)
        self.obstacle_id = obstacle_id

    def __eq__(self, other: object) -> bool:  # pragma: no cover - direct coord comparison
        if not isinstance(other, Obstacle):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.direction == other.direction

    def get_view_state(self, retrying) -> List[CellState]:
        relative_positions = VIEW_OFFSETS.get(self.direction, {}).get(bool(retrying), ())
        return [
            _build_state(self, offset_x, offset_y, facing, cost)
            for offset_x, offset_y, facing, cost in relative_positions
            if is_position_within_arena(self.x + offset_x, self.y + offset_y)
        ]


def _build_state(
    origin: Obstacle,
    offset_x: int,
    offset_y: int,
    facing: Direction,
    penalty: int,
) -> CellState:
    return CellState(origin.x + offset_x, origin.y + offset_y, facing, origin.obstacle_id, penalty)


class Grid:
    """Arena abstraction responsible for obstacle bookkeeping and reachability checks."""

    def __init__(self, size_x: int, size_y: int):
        self.size_x = size_x
        self.size_y = size_y
        self._obstacles: List[Obstacle] = []

    @property
    def obstacles(self) -> List[Obstacle]:
        return self._obstacles

    def add_obstacle(self, obstacle: Obstacle) -> None:
        if not any(existing == obstacle for existing in self._obstacles):
            self._obstacles.append(obstacle)

    def reset_obstacles(self) -> None:
        self._obstacles.clear()

    def get_obstacles(self) -> List[Obstacle]:
        return self._obstacles

    def reachable(self, x: int, y: int, turn: bool = False, preTurn: bool = False) -> bool:
        if not self.is_valid_coord(x, y):
            return False

        for obstacle in self._obstacles:
            if _skip_corner_case(obstacle, x, y):
                continue

            taxicab, max_gap = _distance_metrics(obstacle, x, y)
            if taxicab >= 4:
                continue

            if _violates_padding(max_gap, turn, preTurn):
                return False

        return True

    def is_valid_coord(self, x: int, y: int) -> bool:
        return 1 <= x < self.size_x - 1 and 1 <= y < self.size_y - 1

    def is_valid_cell_state(self, state: CellState) -> bool:
        return self.is_valid_coord(state.x, state.y)

    def get_view_obstacle_positions(self, retrying) -> List[List[CellState]]:
        positions: List[List[CellState]] = []
        for obstacle in self._obstacles:
            if obstacle.direction == Direction.SKIP:
                continue
            states = [candidate for candidate in obstacle.get_view_state(retrying) if self.reachable(candidate.x, candidate.y)]
            positions.append(states)
        return positions


def _skip_corner_case(obstacle: Obstacle, target_x: int, target_y: int) -> bool:
    return obstacle.x == 4 and obstacle.y <= 4 and target_x < 4 and target_y < 4


def _distance_metrics(obstacle: Obstacle, x: int, y: int) -> Tuple[int, int]:
    delta_x = abs(obstacle.x - x)
    delta_y = abs(obstacle.y - y)
    return delta_x + delta_y, max(delta_x, delta_y)


def _violates_padding(max_gap: int, turn: bool, preTurn: bool) -> bool:
    padded_gap = SAFETY_RING + 1
    if turn and max_gap < padded_gap:
        return True
    if preTurn and max_gap < padded_gap:
        return True
    if not preTurn and max_gap < 2:
        return True
    return False
