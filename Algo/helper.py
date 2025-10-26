from __future__ import annotations

from dataclasses import dataclass
from itertools import tee
from typing import Iterable, Iterator, List, Mapping, MutableSequence, Sequence

from consts import HEIGHT, WIDTH, Direction

# Movement
MOVEMENT_STEP = 10
TURN_ROTATION = "090"


@dataclass(frozen=True)
class TurnDirective:
    primary: str
    fallback: str

    def command(self, dy: int) -> str:
        template = self.primary if dy > 0 else self.fallback
        return f"{template}{TURN_ROTATION}"


@dataclass(frozen=True)
class SnapshotDirective:
    axis: str
    positive: str
    aligned: str
    negative: str

    def suffix(self, delta: int) -> str:
        if delta > 0:
            return self.positive
        if delta == 0:
            return self.aligned
        return self.negative


TURN_LOOKUP: Mapping[tuple[Direction, Direction], TurnDirective] = {
    (Direction.NORTH, Direction.EAST): TurnDirective("FR", "BL"),
    (Direction.NORTH, Direction.WEST): TurnDirective("FL", "BR"),
    (Direction.EAST, Direction.NORTH): TurnDirective("FL", "BR"),
    (Direction.EAST, Direction.SOUTH): TurnDirective("BL", "FR"),
    (Direction.SOUTH, Direction.EAST): TurnDirective("BR", "FL"),
    (Direction.SOUTH, Direction.WEST): TurnDirective("BL", "FR"),
    (Direction.WEST, Direction.NORTH): TurnDirective("FR", "BL"),
    (Direction.WEST, Direction.SOUTH): TurnDirective("BR", "FL"),
}


SNAP_LOOKUP: Mapping[tuple[Direction, Direction], SnapshotDirective] = {
    (Direction.WEST, Direction.EAST): SnapshotDirective("y", "L", "C", "R"),
    (Direction.EAST, Direction.WEST): SnapshotDirective("y", "R", "C", "L"),
    (Direction.NORTH, Direction.SOUTH): SnapshotDirective("x", "L", "C", "R"),
    (Direction.SOUTH, Direction.NORTH): SnapshotDirective("x", "R", "C", "L"),
}


DIRECTION_UNIT: Mapping[Direction, tuple[int, int]] = {
    Direction.NORTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, -1),
    Direction.WEST: (-1, 0),
}


def is_position_within_arena(center_x: int, center_y: int) -> bool:
    """Return True when both coordinates lie strictly within the arena walls."""

    return _within_bounds(center_x, WIDTH) and _within_bounds(center_y, HEIGHT)


def is_valid(center_x: int, center_y: int) -> bool:
    """Provide backwards-compatible validation for historical imports."""

    return is_position_within_arena(center_x, center_y)


def command_generator(states: Sequence, obstacles: Iterable[Mapping]) -> List[str]:
    """Construct the condensed list of robot commands for the given route."""

    planner = _CommandPlanner(states, obstacles)
    return planner.build()


class _CommandPlanner:
    __slots__ = ("_states", "_obstacles", "_commands")

    def __init__(self, states: Sequence, obstacles: Iterable[Mapping]) -> None:
        self._states = list(states)
        self._obstacles = {payload["id"]: payload for payload in obstacles}
        self._commands: MutableSequence[str] = []

    def build(self) -> List[str]:
        if not self._states:
            return ["FIN"]

        for previous, current in _pairwise(self._states):
            self._record_transition(previous, current)
            self._record_snapshot(current)

        self._commands.append("FIN")
        return _compress_commands(self._commands)

    def _record_transition(self, previous, current) -> None:
        hop = _Transition(previous, current)
        if hop.same_heading:
            command = _linear_motion(hop)
        else:
            command = _turn_command(hop)

        if command:
            self._commands.append(command)

    def _record_snapshot(self, state) -> None:
        if getattr(state, "screenshot_id", -1) == -1:
            return

        self._commands.append(_snapshot_command(state, self._obstacles))


@dataclass(frozen=True)
class _Transition:
    previous: object
    current: object

    @property
    def same_heading(self) -> bool:
        return self.previous.direction == self.current.direction

    @property
    def dx(self) -> int:
        return self.current.x - self.previous.x

    @property
    def dy(self) -> int:
        return self.current.y - self.previous.y

    def projection_along_heading(self) -> int:
        vector = DIRECTION_UNIT[self.previous.direction]
        return self.dx * vector[0] + self.dy * vector[1]


def _linear_motion(transition: _Transition) -> str:
    displacement = transition.projection_along_heading()
    if displacement > 0:
        return _format_distance_command("FW", MOVEMENT_STEP)
    if displacement < 0:
        return _format_distance_command("BW", MOVEMENT_STEP)
    return _format_distance_command("BW", MOVEMENT_STEP)


def _turn_command(transition: _Transition) -> str:
    key = (transition.previous.direction, transition.current.direction)
    directive = TURN_LOOKUP.get(key)
    if directive is None:
        raise ValueError(f"Invalid turning direction: {key}")
    return directive.command(transition.dy)


def _snapshot_command(state, obstacles: Mapping[int, Mapping]) -> str:
    obstacle = obstacles.get(state.screenshot_id)
    base = f"SNAP{state.screenshot_id}"
    if obstacle is None:
        return base

    directive = SNAP_LOOKUP.get((Direction(obstacle["d"]), state.direction))
    if directive is None:
        return base

    obstacle_value = obstacle[directive.axis]
    robot_value = getattr(state, directive.axis)
    suffix = directive.suffix(obstacle_value - robot_value)
    return f"{base}_{suffix}"


def _pairwise(collection: Sequence) -> Iterator[tuple[object, object]]:
    first, second = tee(collection)
    next(second, None)
    return zip(first, second)


def _format_distance_command(prefix: str, step: int) -> str:
    return f"{prefix}{step:03d}"


def _compress_commands(commands: Sequence[str]) -> List[str]:
    if not commands:
        return []

    merged: List[str] = []
    for command in commands:
        if merged and _mergeable(merged[-1], command):
            merged[-1] = _merge_distance(merged[-1], command)
        else:
            merged.append(command)
    return merged


def _mergeable(first: str, second: str) -> bool:
    return first[:2] in {"FW", "BW"} and first[:2] == second[:2]


def _merge_distance(first: str, second: str) -> str:
    total = int(first[2:]) + int(second[2:])
    return f"{first[:2]}{total:03d}"


def _within_bounds(coordinate: int, upper_bound: int) -> bool:
    return 0 < coordinate < upper_bound - 1