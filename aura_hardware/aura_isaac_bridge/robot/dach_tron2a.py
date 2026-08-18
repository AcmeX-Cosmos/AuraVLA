"""Isaac Sim adapter for either arm of the DACH TRON2A robot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
    ArticulationKinematicsSolver,
)
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.path_planners import RRT


LEFT_ARM_JOINT_NAMES = (
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "elbow_L_Joint",
    "wrist_yaw_L_Joint",
    "wrist_pitch_L_Joint",
    "wrist_roll_L_Joint",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("_L_", "_R_") for name in LEFT_ARM_JOINT_NAMES)
LEFT_ARM_HOME = np.array(
    [0.8477, 0.124, -0.1424, -2.3204, 0.0, 0.0, 0.0], dtype=float
)
RIGHT_ARM_HOME = np.array(
    [0.8477, -0.124, 0.1424, -2.3204, 0.0, 0.0, 0.0], dtype=float
)
GRIPPER_OPEN = np.array([0.0375, 0.0375], dtype=float)
GRIPPER_CLOSED = np.array([0.0, 0.0], dtype=float)
GRIPPER_LIMITS = (-0.0045, 0.0375)


def _arm_spec(side: str) -> dict:
    normalized = str(side).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"DACH arm_side must be left or right, got {side!r}")
    suffix = "L" if normalized == "left" else "R"
    return {
        "side": normalized,
        "suffix": suffix,
        "arm_joints": LEFT_ARM_JOINT_NAMES if normalized == "left" else RIGHT_ARM_JOINT_NAMES,
        "home": LEFT_ARM_HOME if normalized == "left" else RIGHT_ARM_HOME,
        "gripper_joints": (
            f"grasper_{suffix}_jaw_left_Joint",
            f"grasper_{suffix}_jaw_right_Joint",
        ),
        "visual_joints": {
            f"grasper_{suffix}_drive_Joint": 20.4114958,
            f"grasper_{suffix}_crank_right_Joint": 20.4114958,
            f"grasper_{suffix}_bar_left_Joint": 20.4114958,
            f"grasper_{suffix}_bar_right_Joint": 20.4114958,
            f"grasper_{suffix}_nut_Joint": 0.4082299,
        },
        "end_effector_frame": f"tcp_{suffix}_Link",
    }


class DACHTron2AArm:
    """Name-based view over one arm and gripper of the 32-DOF articulation."""

    def __init__(
        self,
        prim_path: str = "/World/DACH_TRON2A/root_joint",
        name: str = "dach_tron2a",
        arm_side: str = "right",
    ) -> None:
        self.prim_path = prim_path
        self.name = name
        self.spec = _arm_spec(arm_side)
        self.arm_side = self.spec["side"]
        self.arm_joint_names = self.spec["arm_joints"]
        self.home_positions = self.spec["home"].copy()
        self.end_effector_frame = self.spec["end_effector_frame"]
        self.articulation = SingleArticulation(prim_path=prim_path, name=name)
        self._arm_indices = np.array([], dtype=np.int64)
        self._gripper_indices = np.array([], dtype=np.int64)
        self._gripper_command_indices = np.array([], dtype=np.int64)
        self._gripper_command_scales = np.array([], dtype=float)
        self.gripper = DACHTron2AGripper(self)

    def initialize(self, physics_sim_view=None) -> None:
        self.articulation.initialize(physics_sim_view)
        self._arm_indices = self._indices_for(self.arm_joint_names)
        self._gripper_indices = self._indices_for(self.spec["gripper_joints"])
        visual_joints = self.spec["visual_joints"]
        command_names = (*self.spec["gripper_joints"], *visual_joints)
        self._gripper_command_indices = self._indices_for(command_names)
        self._gripper_command_scales = np.array(
            [1.0, 1.0, *visual_joints.values()], dtype=float
        )
        print(
            f"DACH TRON2A {self.arm_side} arm ready: "
            f"arm_indices={self._arm_indices.tolist()}, "
            f"gripper_indices={self._gripper_indices.tolist()}"
        )

    def _indices_for(self, names) -> np.ndarray:
        available = set(self.articulation.dof_names)
        missing = [name for name in names if name not in available]
        if missing:
            raise RuntimeError(f"DACH TRON2A missing DOFs: {missing}")
        return np.array(
            [self.articulation.get_dof_index(name) for name in names], dtype=np.int64
        )

    def is_valid(self) -> bool:
        return self.articulation.is_valid()

    def apply_action(self, action: ArticulationAction) -> None:
        self.articulation.apply_action(action)

    def get_world_pose(self):
        return self.articulation.get_world_pose()

    def get_all_joint_positions(self) -> np.ndarray:
        positions = self.articulation.get_joint_positions()
        if positions is None:
            return None
        return np.asarray(positions, dtype=float).copy()

    def get_arm_joint_positions(self) -> np.ndarray:
        positions = self.articulation.get_joint_positions(self._arm_indices)
        if positions is None:
            return None
        positions = np.asarray(positions, dtype=float).reshape(-1)
        if positions.size != len(self.arm_joint_names) or not np.all(np.isfinite(positions)):
            return None
        return positions.copy()

    def get_joint_positions(self) -> np.ndarray:
        return self.get_arm_joint_positions()

    def make_arm_action(self, positions) -> ArticulationAction:
        values = np.asarray(positions, dtype=float).reshape(-1)
        if values.size != len(self.arm_joint_names):
            raise ValueError(f"Expected 7 {self.arm_side}-arm positions, got {values.size}")
        return ArticulationAction(
            joint_positions=values,
            joint_indices=self._arm_indices.copy(),
        )

    def set_arm_joint_positions(self, positions) -> None:
        self.apply_action(self.make_arm_action(positions))

    def teleport_arm_joint_positions(self, positions) -> None:
        values = np.asarray(positions, dtype=float).reshape(-1)
        if values.size != len(self.arm_joint_names):
            raise ValueError(
                f"Expected 7 {self.arm_side}-arm positions, got {values.size}"
            )
        self.articulation.set_joint_positions(
            values,
            joint_indices=self._arm_indices.copy(),
        )
        # Teleporting the position leaves the pre-teleport joint velocities in
        # place, so the next physics step integrates from a velocity that no
        # longer matches the pose -- the arm lurches away from where it was just
        # placed.  Zero them, and re-issue the position target so the drive does
        # not immediately pull back toward the old command.
        self.articulation.set_joint_velocities(
            np.zeros(values.size, dtype=float),
            joint_indices=self._arm_indices.copy(),
        )
        self.set_arm_joint_positions(values)


class DACHTron2AGripper:
    joint_opened_positions = GRIPPER_OPEN
    joint_open_positions = GRIPPER_OPEN
    joint_closed_positions = GRIPPER_CLOSED

    def __init__(self, arm: DACHTron2AArm) -> None:
        self._arm = arm

    def set_joint_positions(self, positions) -> None:
        jaw_positions = np.asarray(positions, dtype=float).reshape(-1)[:2]
        if jaw_positions.size != 2:
            raise ValueError("DACH gripper requires two jaw positions")
        jaw_positions = np.clip(jaw_positions, *GRIPPER_LIMITS)
        base_value = float(np.mean(jaw_positions))
        command_values = self._arm._gripper_command_scales * base_value
        command_values[:2] = jaw_positions
        self._arm.apply_action(
            ArticulationAction(
                joint_positions=command_values,
                joint_indices=self._arm._gripper_command_indices.copy(),
            )
        )

    def teleport_joint_positions(self, positions) -> None:
        jaw_positions = np.asarray(positions, dtype=float).reshape(-1)[:2]
        if jaw_positions.size != 2:
            raise ValueError("DACH gripper requires two jaw positions")
        jaw_positions = np.clip(jaw_positions, *GRIPPER_LIMITS)
        base_value = float(np.mean(jaw_positions))
        command_values = self._arm._gripper_command_scales * base_value
        command_values[:2] = jaw_positions
        self._arm.articulation.set_joint_positions(
            command_values,
            joint_indices=self._arm._gripper_command_indices.copy(),
        )

    def get_joint_positions(self) -> np.ndarray:
        positions = self._arm.articulation.get_joint_positions(
            self._arm._gripper_indices
        )
        if positions is None:
            return None
        positions = np.asarray(positions, dtype=float).reshape(-1)
        if positions.size != 2 or not np.all(np.isfinite(positions)):
            return None
        return positions.copy()


class DACHTron2AIKController:
    """Lula IK controller that commands only the selected seven arm joints."""

    def __init__(
        self,
        robot: DACHTron2AArm,
        robot_description_path: str | Path,
        urdf_path: str | Path,
        end_effector_frame: str | None = None,
        rrt_config_path: str | Path | None = None,
        max_joint_step: float = 0.035,
    ) -> None:
        self.robot = robot
        self.max_joint_step = float(max_joint_step)
        self.lula = LulaKinematicsSolver(
            robot_description_path=str(Path(robot_description_path).resolve()),
            urdf_path=str(Path(urdf_path).resolve()),
        )
        self.kinematics = ArticulationKinematicsSolver(
            robot.articulation,
            self.lula,
            end_effector_frame or robot.end_effector_frame,
        )
        self.rrt = None
        if rrt_config_path is not None:
            try:
                self.rrt = RRT(
                    robot_description_path=str(
                        Path(robot_description_path).resolve()
                    ),
                    urdf_path=str(Path(urdf_path).resolve()),
                    rrt_config_path=str(Path(rrt_config_path).resolve()),
                    end_effector_frame_name=(
                        end_effector_frame or robot.end_effector_frame
                    ),
                )
            except Exception as exc:
                print(f"DACH Lula RRT unavailable, using IK fallback: {exc}")
        self.last_ik_success = False
        self.reset()

    def reset(self) -> None:
        position, orientation = self.robot.get_world_pose()
        self.lula.set_robot_base_pose(
            np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)
        )
        if self.rrt is not None:
            self.rrt.set_robot_base_pose(
                np.asarray(position, dtype=float),
                np.asarray(orientation, dtype=float),
            )

    def add_rrt_obstacle(self, obstacle, static: bool = True) -> bool:
        if self.rrt is None:
            return False
        added = bool(self.rrt.add_obstacle(obstacle, static=static))
        if added:
            self.rrt.update_world()
        return added

    def enable_rrt_obstacle(self, obstacle) -> bool:
        if self.rrt is None:
            return False
        enabled = bool(self.rrt.enable_obstacle(obstacle))
        if enabled:
            self.rrt.update_world()
        return enabled

    def disable_rrt_obstacle(self, obstacle) -> bool:
        if self.rrt is None:
            return False
        disabled = bool(self.rrt.disable_obstacle(obstacle))
        if disabled:
            self.rrt.update_world()
        return disabled

    def plan_collision_free_pose_path(
        self,
        target_position,
        target_orientation=None,
        start_joint_positions=None,
    ) -> list[np.ndarray] | None:
        if self.rrt is None:
            return None
        self.reset()
        self.rrt.set_end_effector_target(
            np.asarray(target_position, dtype=float),
            None
            if target_orientation is None
            else np.asarray(target_orientation, dtype=float),
        )
        self.rrt.update_world()
        if start_joint_positions is None:
            start_joint_positions = self.get_active_joint_positions()
        path = self.rrt.compute_path(
            np.asarray(start_joint_positions, dtype=float),
            np.array([], dtype=float),
        )
        if path is None:
            return None
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] == 0:
            return None
        return [row.copy() for row in path]

    def plan_collision_free_cspace_path(
        self,
        target_joint_positions,
    ) -> list[np.ndarray] | None:
        if self.rrt is None:
            return None
        self.reset()
        self.rrt.set_cspace_target(
            np.asarray(target_joint_positions, dtype=float)
        )
        self.rrt.update_world()
        path = self.rrt.compute_path(
            self.get_active_joint_positions(),
            np.array([], dtype=float),
        )
        if path is None:
            return None
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] == 0:
            return None
        return [row.copy() for row in path]

    def get_active_joint_positions(self) -> np.ndarray:
        return self.robot.get_arm_joint_positions()

    def _valid_ik_seed(self, seed) -> np.ndarray:
        """Return a finite seven-DOF IK seed, falling back to this arm's HOME."""
        candidate = np.asarray(seed, dtype=float).reshape(-1)
        if candidate.shape == (len(self.robot.arm_joint_names),) and np.all(np.isfinite(candidate)):
            return candidate.copy()
        fallback = np.asarray(self.robot.home_positions, dtype=float).reshape(-1)
        if fallback.shape != (len(self.robot.arm_joint_names),) or not np.all(np.isfinite(fallback)):
            raise RuntimeError("DACH IK HOME seed is invalid")
        print(
            f"⚠️ DACH {self.robot.arm_side} IK seed 无效，"
            "使用该侧 HOME 关节 seed"
        )
        return fallback.copy()

    def get_end_effector_pose(self):
        self.reset()
        return self.kinematics.compute_end_effector_pose()

    def plan_pose_path(
        self,
        target_position,
        target_orientation=None,
        clearance: float = 0.08,
    ) -> list[np.ndarray] | None:
        """Plan through a high midpoint, keeping the Lula seed continuous."""
        self.reset()
        current_position, _ = self.kinematics.compute_end_effector_pose()
        current_position = np.asarray(current_position, dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        midpoint = np.array(
            [
                0.5 * (current_position[0] + target_position[0]),
                0.5 * (current_position[1] + target_position[1]),
                max(current_position[2], target_position[2]) + float(clearance),
            ],
            dtype=float,
        )
        return self.plan_pose_waypoints(
            [midpoint, target_position],
            target_orientation=target_orientation,
        )

    def plan_pose_waypoints(
        self,
        waypoints,
        target_orientation=None,
        warm_start=None,
        # An orientation request is a hard constraint by default.  Relaxing it
        # lets Lula choose a different wrist branch at an intermediate point,
        # which appears as a twisted transport path.
        allow_orientation_fallback: bool = False,
    ) -> list[np.ndarray] | None:
        """Solve a complete Cartesian waypoint sequence with a continuous seed."""
        points = [np.asarray(point, dtype=float).reshape(3) for point in waypoints]
        if not points:
            return []
        self.reset()
        if warm_start is None:
            warm_start = self.get_active_joint_positions()
        warm_start = self._valid_ik_seed(warm_start)
        joint_targets: list[np.ndarray] = []
        frame_name = self.kinematics.get_end_effector_frame()
        for point in points:
            solution, success = self.lula.compute_inverse_kinematics(
                frame_name,
                point,
                None
                if target_orientation is None
                else np.asarray(target_orientation, dtype=float),
                warm_start,
            )
            if (
                not success
                and target_orientation is not None
                and allow_orientation_fallback
            ):
                solution, success = self.lula.compute_inverse_kinematics(
                    frame_name, point, None, warm_start
                )
            if not success:
                return None
            solution = np.asarray(solution, dtype=float).reshape(-1)
            if (
                solution.shape != (len(self.robot.arm_joint_names),)
                or not np.all(np.isfinite(solution))
            ):
                print(f"⚠️ DACH {self.robot.arm_side} IK 返回无效关节解")
                return None
            warm_start = solution
            joint_targets.append(warm_start.copy())
        return joint_targets

    def plan_pose_waypoints_with_orientations(
        self,
        waypoints,
        orientations,
        warm_start=None,
    ) -> list[np.ndarray] | None:
        """Solve a Cartesian path with an explicit orientation per waypoint."""
        points = [np.asarray(point, dtype=float).reshape(3) for point in waypoints]
        quaternions = [
            np.asarray(orientation, dtype=float).reshape(4)
            for orientation in orientations
        ]
        if len(points) != len(quaternions):
            raise ValueError("waypoints and orientations must have equal length")
        if not points:
            return []
        self.reset()
        if warm_start is None:
            warm_start = self.get_active_joint_positions()
        warm_start = self._valid_ik_seed(warm_start)
        joint_targets: list[np.ndarray] = []
        frame_name = self.kinematics.get_end_effector_frame()
        for point, orientation in zip(points, quaternions):
            solution, success = self.lula.compute_inverse_kinematics(
                frame_name,
                point,
                orientation,
                warm_start,
            )
            if not success:
                return None
            solution = np.asarray(solution, dtype=float).reshape(-1)
            if (
                solution.shape != (len(self.robot.arm_joint_names),)
                or not np.all(np.isfinite(solution))
            ):
                print(f"⚠️ DACH {self.robot.arm_side} IK 返回无效关节解")
                return None
            warm_start = solution
            joint_targets.append(warm_start.copy())
        return joint_targets

    def forward(
        self,
        target_end_effector_position,
        target_end_effector_orientation=None,
    ) -> ArticulationAction:
        self.reset()
        action, success = self.kinematics.compute_inverse_kinematics(
            np.asarray(target_end_effector_position, dtype=float),
            None
            if target_end_effector_orientation is None
            else np.asarray(target_end_effector_orientation, dtype=float),
        )
        self.last_ik_success = bool(success)
        current = self.get_active_joint_positions()
        if not success or action.joint_positions is None:
            return self.robot.make_arm_action(current)
        desired = np.asarray(action.joint_positions, dtype=float).reshape(-1)
        limited = current + np.clip(
            desired - current, -self.max_joint_step, self.max_joint_step
        )
        return self.robot.make_arm_action(limited)


LEFT_GRIPPER_JOINT_NAMES = _arm_spec("left")["gripper_joints"]
RIGHT_GRIPPER_JOINT_NAMES = _arm_spec("right")["gripper_joints"]
LEFT_GRIPPER_OPEN = GRIPPER_OPEN
LEFT_GRIPPER_CLOSED = GRIPPER_CLOSED
DACH_Tron2A_Arm = DACHTron2AArm
