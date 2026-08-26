#!/usr/bin/env python3
"""ROS 2 file-protocol adapter from AuraVLA to the MoveIt 2 move action."""

from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    PlanningScene,
    RobotState,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


class MoveItFilePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("aura_moveit_planner")
        self.declare_parameter("request_directory", "/tmp/aura-vla-control")
        self.declare_parameter("poll_period_sec", 0.02)
        self.declare_parameter("position_tolerance_m", 0.006)
        self.declare_parameter("orientation_tolerance_rad", 0.12)
        directory = Path(self.get_parameter("request_directory").value).expanduser()
        self.request_path = directory / "moveit_plan_request.json"
        self.response_path = directory / "moveit_plan_response.json"
        self.ready_path = directory / "moveit_planner_ready.json"
        self.scene_path = directory / "moveit_scene.json"
        directory.mkdir(parents=True, exist_ok=True)
        self.position_tolerance = float(self.get_parameter("position_tolerance_m").value)
        self.orientation_tolerance = float(self.get_parameter("orientation_tolerance_rad").value)
        self.action_client = ActionClient(self, MoveGroup, "/move_action")
        scene_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.scene_publisher = self.create_publisher(
            PlanningScene, "/planning_scene", scene_qos
        )
        self._scene_signature = None
        self._active_request_id = None
        self._active_goal = None
        self.timer = self.create_timer(float(self.get_parameter("poll_period_sec").value), self._poll)
        self.scene_timer = self.create_timer(0.25, self._publish_scene)
        self.ready_path.write_text(
            json.dumps({"schema_version": "1.0", "pid": os.getpid()}),
            encoding="utf-8",
        )
        self.get_logger().info("Aura MoveIt file planner initialized")

    def _poll(self) -> None:
        if self._active_request_id is not None or not self.request_path.is_file():
            return
        try:
            payload = json.loads(self.request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid MoveIt request: {exc}")
            return
        request_id = payload.get("request_id")
        if not request_id:
            return
        self._active_request_id = str(request_id)
        self._active_goal = self._build_goal(payload)
        if not self.action_client.wait_for_server(timeout_sec=0.05):
            self._write_failure(request_id, "MoveIt /move_action server is unavailable")
            self._finish_request()
            return
        future = self.action_client.send_goal_async(self._active_goal)
        future.add_done_callback(self._goal_response)

    def _publish_scene(self) -> None:
        if not self.scene_path.is_file():
            return
        try:
            stat = self.scene_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self._scene_signature:
                return
            payload = json.loads(self.scene_path.read_text(encoding="utf-8"))
            scene = PlanningScene()
            scene.is_diff = True
            for item in payload.get("objects", []):
                center = [float(value) for value in item["center"]]
                size = [float(value) for value in item["size"]]
                primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size)
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = center
                collision = CollisionObject()
                collision.id = str(item["id"])
                collision.header.frame_id = str(item.get("frame_id", "world"))
                collision.operation = CollisionObject.ADD
                collision.primitives = [primitive]
                collision.primitive_poses = [pose]
                scene.world.collision_objects.append(collision)
            self.scene_publisher.publish(scene)
            self._scene_signature = signature
            self.get_logger().info("MoveIt planning scene updated from Isaac proxies")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid MoveIt scene update: {exc}")

    def _build_goal(self, payload: dict) -> MoveGroup.Goal:
        start = [float(value) for value in payload["start_joint_positions"]]
        names = [str(name) for name in payload["joint_names"]]
        request = MotionPlanRequest()
        request.group_name = str(payload["group_name"])
        request.num_planning_attempts = 3
        request.allowed_planning_time = float(payload.get("allowed_planning_time_sec", 5.0))
        request.max_velocity_scaling_factor = 0.35
        request.max_acceleration_scaling_factor = 0.35
        request.planner_id = "RRTConnectkConfigDefault"
        request.start_state = RobotState(joint_state=JointState(name=names, position=start))

        constraint = Constraints()
        position = [float(value) for value in payload["target_position"]]
        region = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[
            self.position_tolerance * 2.0,
            self.position_tolerance * 2.0,
            self.position_tolerance * 2.0,
        ])
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "world"
        position_constraint.link_name = str(payload["end_effector_link"])
        position_constraint.constraint_region = BoundingVolume(primitives=[region], primitive_poses=[Pose()])
        position_constraint.constraint_region.primitive_poses[0].position.x = position[0]
        position_constraint.constraint_region.primitive_poses[0].position.y = position[1]
        position_constraint.constraint_region.primitive_poses[0].position.z = position[2]
        position_constraint.weight = 1.0
        constraint.position_constraints.append(position_constraint)

        orientation = payload.get("target_orientation")
        if orientation is not None:
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = "world"
            orientation_constraint.link_name = str(payload["end_effector_link"])
            orientation_constraint.orientation.x = float(orientation[1])
            orientation_constraint.orientation.y = float(orientation[2])
            orientation_constraint.orientation.z = float(orientation[3])
            orientation_constraint.orientation.w = float(orientation[0])
            orientation_constraint.absolute_x_axis_tolerance = self.orientation_tolerance
            orientation_constraint.absolute_y_axis_tolerance = self.orientation_tolerance
            orientation_constraint.absolute_z_axis_tolerance = self.orientation_tolerance
            orientation_constraint.weight = 1.0
            constraint.orientation_constraints.append(orientation_constraint)
        request.goal_constraints.append(constraint)
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        return goal

    def _goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._write_failure(self._active_request_id, "MoveIt rejected the planning goal")
            self._finish_request()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._planning_result)

    def _planning_result(self, future) -> None:
        request_id = self._active_request_id
        try:
            result = future.result().result
            trajectory = result.planned_trajectory.joint_trajectory
            request = json.loads(self.request_path.read_text(encoding="utf-8"))
            requested_names = list(request.get("joint_names", []))
            trajectory_names = list(trajectory.joint_names)
            indices = [trajectory_names.index(name) for name in requested_names]
            positions = [
                [float(point.positions[index]) for index in indices]
                for point in trajectory.points
            ]
            expected = len(requested_names)
            if not positions or any(len(point) != expected for point in positions):
                self._write_failure(request_id, "MoveIt returned no compatible joint trajectory")
            else:
                self._write_response(request_id, {
                    "success": True,
                    "joint_names": requested_names,
                    "trajectory_positions": positions,
                })
        except Exception as exc:
            self._write_failure(request_id, f"MoveIt planning result error: {type(exc).__name__}: {exc}")
        self._finish_request()

    def _write_failure(self, request_id, message: str) -> None:
        self._write_response(request_id, {"success": False, "message": message})

    def _write_response(self, request_id, payload: dict) -> None:
        payload = {"schema_version": "1.0", "request_id": str(request_id), **payload}
        temporary = self.response_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.response_path)

    def _finish_request(self) -> None:
        self._active_request_id = None
        self._active_goal = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveItFilePlannerNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.ready_path.unlink(missing_ok=True)
        except OSError:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
