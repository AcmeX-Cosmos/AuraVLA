"""File-backed MoveIt 2 planning client for the Isaac Sim process.

Isaac Sim and the ROS 2 graph intentionally remain separate processes.  The
MoveIt node consumes the small JSON protocol implemented here and returns a
joint trajectory; no ROS imports are required inside Isaac Python.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np


class MoveItPlanningError(RuntimeError):
    """Raised when MoveIt cannot produce a valid arm trajectory."""


def write_moveit_collision_scene(objects, request_directory="/tmp/aura-vla-control") -> None:
    """Publish static Isaac collision proxies through the file bridge.

    ``objects`` contains dictionaries with ``id``, ``center`` and ``size`` in
    world coordinates.  Keeping this protocol independent of ROS lets the
    Isaac runtime update the scene without importing MoveIt Python modules.
    """
    directory = Path(request_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "objects": list(objects)}
    temporary = directory / "moveit_scene.tmp"
    target = directory / "moveit_scene.json"
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, target)


class MoveItFilePlanner:
    """Synchronous client for ``aura_moveit_planner``.

    The client is deliberately small and deterministic: one request is in
    flight at a time, responses are matched by UUID, and stale responses are
    never accepted as a new plan.
    """

    def __init__(
        self,
        *,
        request_directory: str | os.PathLike[str] = "/tmp/aura-vla-control",
        timeout_sec: float = 5.0,
        poll_interval_sec: float = 0.02,
    ) -> None:
        self.directory = Path(request_directory).expanduser().resolve()
        self.timeout_sec = max(float(timeout_sec), 0.1)
        self.poll_interval_sec = max(float(poll_interval_sec), 0.005)
        self.request_path = self.directory / "moveit_plan_request.json"
        self.response_path = self.directory / "moveit_plan_response.json"
        self.ready_path = self.directory / "moveit_planner_ready.json"

    def is_ready(self) -> bool:
        """Return true only when the current planner node advertises readiness."""
        if not self.ready_path.is_file():
            return False
        try:
            payload = json.loads(self.ready_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
            if pid <= 0:
                return False
            os.kill(pid, 0)
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def plan_pose_waypoints(
        self,
        *,
        group_name: str,
        end_effector_link: str,
        waypoints,
        target_orientation=None,
        orientations=None,
        start_joint_positions,
        joint_names,
    ) -> list[np.ndarray] | None:
        points = [np.asarray(point, dtype=float).reshape(3) for point in waypoints]
        if not points:
            return []
        if orientations is None:
            orientations = [target_orientation] * len(points)
        if len(orientations) != len(points):
            raise ValueError("MoveIt waypoints and orientations must have equal length")

        current = np.asarray(start_joint_positions, dtype=float).reshape(-1)
        result: list[np.ndarray] = []
        for point, orientation in zip(points, orientations):
            planned = self.plan_to_pose(
                group_name=group_name,
                end_effector_link=end_effector_link,
                target_position=point,
                target_orientation=orientation,
                start_joint_positions=current,
                joint_names=joint_names,
            )
            if planned is None:
                return None
            if result and len(planned):
                planned = planned[1:]
            result.extend(planned)
            if planned:
                current = planned[-1].copy()
        return result

    def plan_to_pose(
        self,
        *,
        group_name: str,
        end_effector_link: str,
        target_position,
        target_orientation,
        start_joint_positions,
        joint_names,
    ) -> list[np.ndarray] | None:
        if not self.is_ready():
            raise MoveItPlanningError("MoveIt planner node is not running")
        position = np.asarray(target_position, dtype=float).reshape(-1)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("MoveIt target_position must contain three finite values")
        orientation = None if target_orientation is None else np.asarray(target_orientation, dtype=float).reshape(-1)
        if orientation is not None and (orientation.shape != (4,) or not np.all(np.isfinite(orientation))):
            raise ValueError("MoveIt target_orientation must contain four finite values")
        start = np.asarray(start_joint_positions, dtype=float).reshape(-1)
        if start.shape != (len(joint_names),) or not np.all(np.isfinite(start)):
            raise ValueError("MoveIt start_joint_positions do not match joint_names")

        request_id = uuid.uuid4().hex
        payload = {
            "schema_version": "1.0",
            "request_id": request_id,
            "group_name": str(group_name),
            "end_effector_link": str(end_effector_link),
            "joint_names": [str(name) for name in joint_names],
            "start_joint_positions": start.tolist(),
            "target_position": position.tolist(),
            "target_orientation": None if orientation is None else orientation.tolist(),
            "allowed_planning_time_sec": self.timeout_sec,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.request_path.with_suffix(f".{request_id}.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.request_path)

        deadline = time.monotonic() + self.timeout_sec + 1.0
        while time.monotonic() < deadline:
            if self.response_path.is_file():
                try:
                    response = json.loads(self.response_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    response = None
                if isinstance(response, dict) and response.get("request_id") == request_id:
                    if not response.get("success", False):
                        raise MoveItPlanningError(str(response.get("message", "MoveIt planning failed")))
                    trajectory = np.asarray(response.get("trajectory_positions", []), dtype=float)
                    if trajectory.ndim != 2 or trajectory.shape[1] != len(joint_names) or trajectory.shape[0] == 0:
                        raise MoveItPlanningError("MoveIt returned an invalid arm trajectory")
                    if not np.all(np.isfinite(trajectory)):
                        raise MoveItPlanningError("MoveIt returned non-finite joint positions")
                    return [row.copy() for row in trajectory]
            time.sleep(self.poll_interval_sec)
        raise MoveItPlanningError(
            f"MoveIt planner did not respond within {self.timeout_sec:.2f}s; "
            "start aura_moveit_config/moveit.launch.py"
        )
