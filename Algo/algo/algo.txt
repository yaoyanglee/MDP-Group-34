from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from consts import Direction, ITERATIONS, MOVE_DIRECTION, SAFE_COST, TURN_FACTOR, TURN_RADIUS
from entities.Entity import CellState, Grid, Obstacle
from entities.Robot import Robot
from python_tsp.exact import solve_tsp_dynamic_programming


HEADING_VECTORS: Dict[Direction, Tuple[int, int]] = {
    direction: vector
    for direction, vector in (
        (Direction.NORTH, (0, 1)),
        (Direction.EAST, (1, 0)),
        (Direction.SOUTH, (0, -1)),
        (Direction.WEST, (-1, 0)),
    )
}


TURN_KINEMATICS: Dict[Tuple[str, str], Tuple[Tuple[float, float], Tuple[float, float]]] = {
    ("forward", "right"): (
        (4 * TURN_RADIUS, 2.5 * TURN_RADIUS),
        (4 * TURN_RADIUS, 2.5 * TURN_RADIUS),
    ),
    ("forward", "left"): (
        (3 * TURN_RADIUS, 1.5 * TURN_RADIUS),
        (4 * TURN_RADIUS, 2 * TURN_RADIUS),
    ),
    ("backward", "right"): (
        (2 * TURN_RADIUS, 3 * TURN_RADIUS),
        (2 * TURN_RADIUS, 4 * TURN_RADIUS),
    ),
    ("backward", "left"): (
        (3 * TURN_RADIUS, 2 * TURN_RADIUS),
        (4 * TURN_RADIUS, 2 * TURN_RADIUS),
    ),
}


@dataclass
class CandidateMatrix:
    states: List[CellState]
    groups: List[Sequence[CellState]]
    offsets: List[int]


@dataclass
class RouteCache:
    costs: Dict[Tuple[CellState, CellState], float]
    paths: Dict[Tuple[CellState, CellState], List[Tuple[int, int, Direction]]]

    def __init__(self) -> None:
        self.costs = {}
        self.paths = {}


def _pairwise(sequence: Sequence[int]) -> Iterator[Tuple[int, int]]:
    iterator = iter(sequence)
    try:
        previous = next(iterator)
    except StopIteration:
        return

    for current in iterator:
        yield previous, current
        previous = current


class MazeSolver:
    """Generates robot paths while keeping the original external behaviour intact."""

    def __init__(
        self,
        size_x: int,
        size_y: int,
        robot_x: int,
        robot_y: int,
        robot_direction: Direction,
        big_turn=None,
    ):
        self._grid = Grid(size_x, size_y)
        self._robot = Robot(robot_x, robot_y, robot_direction)
        self._cache = RouteCache()
        self._turn_profile = int(big_turn or 0)

        # Preserve backwards compatibility with legacy callers.
        self.path_table = self._cache.paths
        self.cost_table = self._cache.costs

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def robot(self) -> Robot:
        return self._robot

    @property
    def big_turn(self) -> int:
        return self._turn_profile

    def add_obstacle(self, x: int, y: int, direction: Direction, obstacle_id: int) -> None:
        self._grid.add_obstacle(Obstacle(x, y, direction, obstacle_id))

    def reset_obstacles(self) -> None:
        self._grid.reset_obstacles()

    @staticmethod
    def compute_coord_distance(x1: int, y1: int, x2: int, y2: int, level: int = 1) -> float:
        delta_x, delta_y = x1 - x2, y1 - y2
        if level == 2:
            return math.hypot(delta_x, delta_y)
        return abs(delta_x) + abs(delta_y)

    @staticmethod
    def compute_state_distance(start_state: CellState, end_state: CellState, level: int = 1) -> float:
        return MazeSolver.compute_coord_distance(start_state.x, start_state.y, end_state.x, end_state.y, level)

    @staticmethod
    def get_visit_options(count: int) -> List[str]:
        if count == 0:
            return ["0"]

        masks = [format(idx, f"0{count}b") for idx in range(2**count)]
        masks.sort(key=lambda bitmask: bitmask.count("1"), reverse=True)
        return masks

    def get_optimal_order_dp(self, retrying) -> Tuple[List[CellState], float]:
        vantage_catalog = self._grid.get_view_obstacle_positions(retrying)
        if not vantage_catalog:
            return [self._robot.get_start_state()], 0.0

        for visit_mask in self.get_visit_options(len(vantage_catalog)):
            bundle = self._build_candidate_matrix(vantage_catalog, visit_mask)
            if bundle is None:
                continue

            if not bundle.groups:
                return [bundle.states[0]], 0.0

            if any(len(group) == 0 for group in bundle.groups):
                continue

            self.path_cost_generator(bundle.states)

            best_path_for_mask: List[CellState] | None = None
            best_cost_for_mask = float("inf")

            for selection, penalty in self._selection_generator(bundle.groups):
                route_indices = self._translate_indices(selection, bundle.offsets)
                permutation, travel_cost = self._solve_route(bundle.states, route_indices)
                total_cost = travel_cost + penalty

                if total_cost >= best_cost_for_mask:
                    continue

                candidate_path = self._materialize_path(bundle.states, route_indices, permutation)
                if not candidate_path:
                    continue

                best_cost_for_mask = total_cost
                best_path_for_mask = candidate_path

            if best_path_for_mask is not None:
                return best_path_for_mask, best_cost_for_mask

        return [], float("inf")

    def path_cost_generator(self, states: List[CellState]):
        for start_idx in range(len(states) - 1):
            for end_idx in range(start_idx + 1, len(states)):
                self._run_astar(states[start_idx], states[end_idx])

    @staticmethod
    def generate_combination(view_positions, _index, _current, result, iteration_left):
        limit = iteration_left[0]
        generated = []
        for selection, _ in MazeSolver._selection_generator(view_positions, limit=limit):
            generated.append(selection)

        result.extend(generated)
        iteration_left[0] = max(iteration_left[0] - len(generated), 0)

    def _build_candidate_matrix(
        self,
        vantage_catalog: Sequence[Sequence[CellState]],
        mask: str,
    ) -> CandidateMatrix | None:
        states = [self._robot.get_start_state()]
        groups: List[Sequence[CellState]] = []
        offsets: List[int] = []
        cursor = 1

        for include_flag, viewpoints in zip(mask, vantage_catalog):
            if include_flag == "1":
                groups.append(viewpoints)
                offsets.append(cursor)
                states.extend(viewpoints)
                cursor += len(viewpoints)

        return CandidateMatrix(states, groups, offsets)

    def _translate_indices(self, selection: Sequence[int], offsets: Sequence[int]) -> List[int]:
        indices = [0]
        for base_index, chosen in zip(offsets, selection):
            indices.append(base_index + chosen)
        return indices

    def _solve_route(self, states: Sequence[CellState], indices: Sequence[int]):
        size = len(indices)
        if size <= 1:
            return [0], 0.0

        matrix = np.full((size, size), 1e9, dtype=float)
        np.fill_diagonal(matrix, 0)

        for i in range(size - 1):
            for j in range(i + 1, size):
                start_state = states[indices[i]]
                end_state = states[indices[j]]
                cost = self._cache.costs.get((start_state, end_state))
                if cost is not None:
                    matrix[i, j] = cost
                    matrix[j, i] = cost

        matrix[:, 0] = 0
        return solve_tsp_dynamic_programming(matrix)

    def _materialize_path(
        self,
        states: Sequence[CellState],
        indices: Sequence[int],
        permutation: Sequence[int],
    ) -> List[CellState] | None:
        if not permutation:
            return None

        ordered_indices = [indices[idx] for idx in permutation]
        route: List[CellState] = [states[ordered_indices[0]]]

        for start_idx, end_idx in _pairwise(ordered_indices):
            start_state = states[start_idx]
            end_state = states[end_idx]
            segment = self._cache.paths.get((start_state, end_state))
            if not segment:
                return None

            for x_coord, y_coord, heading in segment[1:]:
                route.append(CellState(x_coord, y_coord, heading))
            route[-1].set_screenshot(end_state.screenshot_id)

        return route

    @staticmethod
    def _selection_generator(
        groups: Sequence[Sequence[CellState]],
        limit: int = ITERATIONS,
    ) -> Iterator[Tuple[List[int], int]]:
        if not groups:
            yield [], 0
            return

        for selection in product(*(range(len(group)) for group in groups)):
            if limit <= 0:
                break
            penalty = sum(group[idx].penalty for group, idx in zip(groups, selection))
            yield list(selection), penalty
            limit -= 1

    def _run_astar(self, start: CellState, goal: CellState) -> None:
        if (start, goal) in self._cache.paths:
            return

        frontier: List[Tuple[float, int, int, Direction]] = [
            (self.compute_state_distance(start, goal), start.x, start.y, start.direction)
        ]
        best_cost = {(start.x, start.y, start.direction): 0.0}
        parents: Dict[Tuple[int, int, Direction], Tuple[int, int, Direction]] = {}
        visited: set[Tuple[int, int, Direction]] = set()

        while frontier:
            _, cur_x, cur_y, heading = heapq.heappop(frontier)
            signature = (cur_x, cur_y, heading)

            if signature in visited:
                continue

            if goal.is_eq(cur_x, cur_y, heading):
                self._record_route(start, goal, parents, best_cost[signature])
                return

            visited.add(signature)
            base_cost = best_cost[signature]

            for next_x, next_y, next_heading, safety in self._neighbors(cur_x, cur_y, heading):
                neighbour_signature = (next_x, next_y, next_heading)
                if neighbour_signature in visited:
                    continue

                move_cost = (
                    Direction.rotation_cost(next_heading, heading) * TURN_FACTOR
                    + 1
                    + safety
                )
                tentative = base_cost + move_cost
                heuristic = self.compute_coord_distance(next_x, next_y, goal.x, goal.y)
                score = tentative + heuristic

                if neighbour_signature not in best_cost or tentative < best_cost[neighbour_signature]:
                    best_cost[neighbour_signature] = tentative
                    parents[neighbour_signature] = signature
                    heapq.heappush(frontier, (score, next_x, next_y, next_heading))

    def _record_route(
        self,
        start: CellState,
        goal: CellState,
        parents: Dict[Tuple[int, int, Direction], Tuple[int, int, Direction]],
        cost: float,
    ) -> None:
        self._cache.costs[(start, goal)] = cost
        self._cache.costs[(goal, start)] = cost

        path: List[Tuple[int, int, Direction]] = []
        cursor = (goal.x, goal.y, goal.direction)

        while cursor in parents:
            path.append(cursor)
            cursor = parents[cursor]

        path.append(cursor)

        forward_path = list(reversed(path))
        reverse_path = list(path)

        self._cache.paths[(start, goal)] = forward_path
        self._cache.paths[(goal, start)] = reverse_path

    def _neighbors(self, x: int, y: int, heading: Direction) -> Iterable[Tuple[float, float, Direction, int]]:
        yield from self._linear_moves(x, y, heading)
        yield from self._turn_moves(x, y, heading)

    def get_neighbors(self, x: int, y: int, direction: Direction):
        return list(self._neighbors(x, y, direction))

    def _linear_moves(self, x: int, y: int, heading: Direction) -> Iterable[Tuple[float, float, Direction, int]]:
        for step_x, step_y, direction in MOVE_DIRECTION:
            if direction != heading:
                continue
            for multiplier in (1, -1):
                candidate_x = x + step_x * multiplier
                candidate_y = y + step_y * multiplier
                if self._grid.reachable(candidate_x, candidate_y):
                    yield candidate_x, candidate_y, heading, self._safety_penalty(candidate_x, candidate_y)

    def _turn_moves(self, x: int, y: int, heading: Direction) -> Iterable[Tuple[float, float, Direction, int]]:
        right_heading = self._rotate(heading, 2)
        left_heading = self._rotate(heading, -2)

        for target_heading, side in ((right_heading, "right"), (left_heading, "left")):
            for motion in ("forward", "backward"):
                delta_x, delta_y = self._turn_delta(motion, side, heading, target_heading)
                target_x = x + delta_x
                target_y = y + delta_y

                if self._grid.reachable(target_x, target_y, turn=True) and self._grid.reachable(x, y, preTurn=True):
                    penalty = self._safety_penalty(target_x, target_y) + 10
                    yield target_x, target_y, target_heading, penalty

    def _turn_delta(
        self,
        motion: str,
        side: str,
        start_heading: Direction,
        target_heading: Direction,
    ) -> Tuple[float, float]:
        offsets = TURN_KINEMATICS[(motion, side)][self._turn_profile]
        next_vector = HEADING_VECTORS[target_heading]
        current_vector = HEADING_VECTORS[start_heading]
        sign = 1 if motion == "forward" else -1

        offset_new, offset_old = offsets
        delta_x = sign * (offset_new * next_vector[0] + offset_old * current_vector[0])
        delta_y = sign * (offset_new * next_vector[1] + offset_old * current_vector[1])
        return delta_x, delta_y

    def _safety_penalty(self, x: float, y: float) -> int:
        for obstacle in self._grid.obstacles:
            if abs(obstacle.x - x) == 2 and abs(obstacle.y - y) == 2:
                return SAFE_COST
            if abs(obstacle.x - x) == 1 and abs(obstacle.y - y) == 2:
                return SAFE_COST
            if abs(obstacle.x - x) == 2 and abs(obstacle.y - y) == 1:
                return SAFE_COST
        return 0

    @staticmethod
    def _rotate(base: Direction, delta: int) -> Direction:
        return Direction((int(base) + delta) % 8)


if __name__ == "__main__":
    pass
