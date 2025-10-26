from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from flask import Flask, jsonify, request
from flask_cors import CORS

from algo.algo import MazeSolver
from entities.Entity import CellState
from helper import command_generator
from model import predict_image_week_9, stitch_image, stitch_image_own


LOGGER = logging.getLogger(__name__)
UPLOAD_DIRECTORY = Path("uploads")
MODEL_REFERENCE = None  # Placeholder for future model loading if needed.


def create_app() -> Flask:
    application = Flask(__name__)
    CORS(application)
    _register_routes(application)
    return application


app = create_app()


@dataclass
class ObstacleForm:
    x: int
    y: int
    d: int
    id: int

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ObstacleForm":
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            d=int(payload["d"]),
            id=int(payload["id"]),
        )


@dataclass
class PlannerRequest:
    robot_x: int
    robot_y: int
    robot_dir: int
    retrying: Any
    obstacles: Sequence[ObstacleForm]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannerRequest":
        obstacles = [ObstacleForm.from_payload(item) for item in payload.get("obstacles", [])]
        return cls(
            robot_x=int(payload["robot_x"]),
            robot_y=int(payload["robot_y"]),
            robot_dir=int(payload["robot_dir"]),
            retrying=payload.get("retrying"),
            obstacles=obstacles,
        )

    def as_solver(self) -> MazeSolver:
        solver = MazeSolver(20, 20, self.robot_x, self.robot_y, self.robot_dir, big_turn=None)
        for obstacle in self.obstacles:
            solver.add_obstacle(obstacle.x, obstacle.y, obstacle.d, obstacle.id)
        return solver


def _register_routes(application: Flask) -> None:
    application.add_url_rule("/status", view_func=_status_handler, methods=["GET"])
    application.add_url_rule("/path", view_func=_path_handler, methods=["POST"])
    application.add_url_rule("/image", view_func=_image_handler, methods=["POST"])
    application.add_url_rule("/stitch", view_func=_stitch_handler, methods=["GET"])


def _status_handler():
    return jsonify({"result": "ok"})


def _path_handler():
    payload = request.get_json(force=True)
    planner_request = PlannerRequest.from_dict(payload)
    solver = planner_request.as_solver()

    start = time.perf_counter()
    planned_states, travel_cost = solver.get_optimal_order_dp(retrying=planner_request.retrying)
    elapsed = time.perf_counter() - start
    LOGGER.info("Path planning finished in %.3fs with cost %.3f", elapsed, travel_cost)

    original_obstacles = [vars(obstacle) for obstacle in planner_request.obstacles]
    commands = command_generator(planned_states, original_obstacles)
    trace = _render_state_trace(planned_states, commands)

    return jsonify(
        {
            "data": {
                "distance": travel_cost,
                "path": trace,
                "commands": commands,
            },
            "error": None,
        }
    )


def _image_handler():
    ensure_upload_directory()

    upload = request.files["file"]
    stored_path = _store_upload(upload.filename, upload.stream.read())
    constituents = upload.filename.split("_")
    obstacle_id = constituents[1]

    image_id = predict_image_week_9(stored_path.name, MODEL_REFERENCE)

    return jsonify({
        "obstacle_id": obstacle_id,
        "image_id": image_id,
    })


def _stitch_handler():
    first = stitch_image()
    first.show()
    second = stitch_image_own()
    second.show()
    return jsonify({"result": "ok"})


def ensure_upload_directory() -> None:
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _store_upload(filename: str, data: bytes) -> Path:
    target = UPLOAD_DIRECTORY / filename
    with target.open("wb") as file_handle:
        file_handle.write(data)
    return target


def _render_state_trace(states: Sequence[CellState], commands: Sequence[str]) -> List[Dict[str, Any]]:
    if not states:
        return []

    cursor = 0
    indices = [cursor]
    for command in commands:
        cursor = _advance_index(cursor, command)
        if cursor >= len(states):
            cursor = len(states) - 1
        indices.append(cursor)

    return [states[index].get_dict() for index in indices]


def _advance_index(current: int, command: str) -> int:
    head = command[:2]
    if command.startswith("SNAP") or command.startswith("FIN"):
        return current
    if head in {"FW", "FS", "BW", "BS"}:
        magnitude = int(command[2:]) // 10
        return current + magnitude
    return current + 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5002, debug=True)
