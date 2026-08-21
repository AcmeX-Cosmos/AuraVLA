"""AuraVLA 运动控制模块：IK、RRT、轨迹规划、夹爪控制、避障运动。"""

from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import (
    quat_to_euler_angles,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
)
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from aura_isaac_bridge.core.state import state
from aura_isaac_bridge.core.state import (
    DACH_ARM_SIDE,
    DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET,
    DACH_JAW_COLLISION_LOCAL_BOUNDS,
    MAX_GRASP_APPROACH_TILT_RAD, TARGET_GRASP_APPROACH_TILT_RAD,
    GRIPPER_CONTACT_RESIDUAL, GRIPPER_CONTACT_FORCE_THRESHOLD,
    GRIPPER_CONTACT_PRELOAD_RESIDUAL, GRIPPER_CONTACT_HOLD_PRELOAD,
    GRIPPER_CONTACT_SETTLE_FRAMES, GRIPPER_PRELOAD_CONFIRM_FRAMES,
    MIN_GRIPPER_TABLE_CLEARANCE, TABLE_CLEARANCE_ABORT_MARGIN,
    PHYSX_CONTACT_OFFSET, PHYSX_REST_OFFSET,
    TRAJECTORY_MAX_JOINT_STEP, TRAJECTORY_MIN_FRAMES, TRAJECTORY_SETTLE_FRAMES,
    GRASP_APPROACH_MAX_JOINT_STEP, GRASP_APPROACH_MIN_FRAMES,
    ACTION_WAYPOINT_LIMIT, CARTESIAN_WAYPOINT_SPACING,
    DUAL_ARM_MIN_TCP_SEPARATION,
)
from aura_isaac_bridge.core.physics import step_app
from aura_isaac_bridge.robot.dach_tron2a import LEFT_ARM_HOME, RIGHT_ARM_HOME
from aura_isaac_bridge.robot.motion_planner import minimum_jerk, SparseKeyposeDiffuser, DiffusionConfig
from aura_isaac_bridge.core.perception import (
    get_bbox_center,
    get_current_bbox_center,
    get_sim_pose,
    quat_rotate,
)
from aura_isaac_bridge.utils.path_visualization import render_joint_path


def quat_normalize(quat):
    quat = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError("不能归一化零四元数")
    return quat / norm


def rotate_quat_about_world_z(quat, angle_rad):
    """Rotate a scalar-first quaternion around world Z."""
    w, x, y, z = quat_normalize(quat)
    half_angle = 0.5 * float(angle_rad)
    cosine = math.cos(half_angle)
    sine = math.sin(half_angle)
    return quat_normalize(
        np.array(
            [
                cosine * w - sine * z,
                cosine * x - sine * y,
                cosine * y + sine * x,
                cosine * z + sine * w,
            ],
            dtype=float,
        )
    )


def _max_path_orientation_error(joint_targets, desired_orientation):
    """Return the largest TCP orientation deviation along a joint path."""
    if desired_orientation is None or not joint_targets:
        return 0.0
    desired = quat_normalize(desired_orientation)
    frame_name = state.controller.kinematics.get_end_effector_frame()
    max_error = 0.0
    for joints in joint_targets:
        _, rotation = state.controller.lula.compute_forward_kinematics(
            frame_name, np.asarray(joints, dtype=float)
        )
        actual = quat_normalize(rot_matrix_to_quat(rotation))
        dot = abs(float(np.dot(actual, desired)))
        max_error = max(max_error, 2.0 * math.acos(np.clip(dot, -1.0, 1.0)))
    return max_error


def _max_path_approach_axis_error(joint_targets, desired_orientation):
    """Return path tilt error while allowing harmless wrist yaw."""
    if desired_orientation is None or not joint_targets:
        return 0.0
    desired_axis = quat_to_rot_matrix(
        quat_normalize(desired_orientation)
    )[:, 0]
    desired_axis /= np.linalg.norm(desired_axis)
    frame_name = state.controller.kinematics.get_end_effector_frame()
    max_error = 0.0
    for joints in joint_targets:
        _, rotation = state.controller.lula.compute_forward_kinematics(
            frame_name,
            np.asarray(joints, dtype=float),
        )
        actual_axis = np.asarray(rotation, dtype=float)[:, 0]
        actual_axis /= np.linalg.norm(actual_axis)
        max_error = max(
            max_error,
            math.acos(np.clip(float(np.dot(actual_axis, desired_axis)), -1.0, 1.0)),
        )
    return max_error


def get_object_gripper_containment(prim_path, *, target_prim=None):
    """校验物体几何中心是否位于两片夹指的共同工作空间。

    Do not use the arithmetic mean of mesh vertices here.  The MasterChef
    can's cap and label contain a denser vertex distribution on one side, so
    that mean is 24 mm away from the visible/collision volume center even
    though the can is correctly between the fingers.  Planning already uses
    the world bounding-box center; using the same geometric reference keeps
    the strict containment check aligned with the grasp target.
    """
    if target_prim is None:
        object_center, _, _ = get_bbox_center(get_current_stage(), prim_path)
    else:
        object_center, _, _ = get_current_bbox_center(
            get_current_stage(), target_prim, prim_path
        )
    left_corners = get_finger_collision_world_corners(state.left_finger, "left")
    right_corners = get_finger_collision_world_corners(state.right_finger, "right")
    left_center = np.mean(left_corners, axis=0)
    right_center = np.mean(right_corners, axis=0)
    finger_delta = right_center - left_center
    finger_separation = float(np.linalg.norm(finger_delta))
    if finger_separation < 1e-6:
        raise RuntimeError("双指碰撞中心重合，无法验证抓取包含关系")
    closing_axis = finger_delta / finger_separation
    gripper_center = (left_center + right_center) * 0.5
    center_delta = object_center - gripper_center
    axial_error = float(np.dot(center_delta, closing_axis))
    _, ee_rotation = state.controller.get_end_effector_pose()
    approach_axis = np.asarray(ee_rotation, dtype=float)[:, 0]
    approach_axis -= np.dot(approach_axis, closing_axis) * closing_axis
    approach_axis /= max(float(np.linalg.norm(approach_axis)), 1e-9)
    lateral_axis = np.cross(closing_axis, approach_axis)
    lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1e-9)

    def shared_projection_interval(axis):
        left_projection = left_corners @ axis
        right_projection = right_corners @ axis
        return (
            max(float(np.min(left_projection)), float(np.min(right_projection))),
            min(float(np.max(left_projection)), float(np.max(right_projection))),
        )

    projection_margin = 0.003
    approach_interval = shared_projection_interval(approach_axis)
    lateral_interval = shared_projection_interval(lateral_axis)
    object_approach = float(np.dot(object_center, approach_axis))
    object_lateral = float(np.dot(object_center, lateral_axis))
    approach_inside = bool(
        approach_interval[0] - projection_margin
        <= object_approach
        <= approach_interval[1] + projection_margin
    )
    lateral_inside = bool(
        lateral_interval[0] - projection_margin
        <= object_lateral
        <= lateral_interval[1] + projection_margin
    )
    axial_inset = min(0.002, finger_separation * 0.1)
    axial_limit = max(finger_separation * 0.5 - axial_inset, 0.0)
    approach_error = float(np.dot(center_delta, approach_axis))
    lateral_error = float(np.dot(center_delta, lateral_axis))
    radial_error = float(np.hypot(approach_error, lateral_error))
    radial_limit = float(
        np.hypot(
            max(abs(approach_interval[0] - np.dot(gripper_center, approach_axis)),
                abs(approach_interval[1] - np.dot(gripper_center, approach_axis)))
            + projection_margin,
            max(abs(lateral_interval[0] - np.dot(gripper_center, lateral_axis)),
                abs(lateral_interval[1] - np.dot(gripper_center, lateral_axis)))
            + projection_margin,
        )
    )
    return {
        "contained": bool(
            abs(axial_error) <= axial_limit
            and approach_inside
            and lateral_inside
        ),
        "object_center": object_center.tolist(),
        "object_center_source": "world_bbox_center",
        "gripper_center": gripper_center.tolist(),
        "finger_separation_m": finger_separation,
        "axial_error_m": axial_error,
        "axial_limit_m": axial_limit,
        "radial_error_m": radial_error,
        "radial_limit_m": radial_limit,
        "approach_error_m": approach_error,
        "approach_interval_m": list(approach_interval),
        "approach_inside": approach_inside,
        "lateral_error_m": lateral_error,
        "lateral_interval_m": list(lateral_interval),
        "lateral_inside": lateral_inside,
    }


def get_fabric_usd_pose_error(target_prim):
    fabric_positions, fabric_orientations = target_prim._prim_view.get_world_poses(
        usd=False
    )
    usd_positions, usd_orientations = target_prim._prim_view.get_world_poses(
        usd=True
    )
    fabric_position = np.asarray(fabric_positions[0], dtype=float)
    usd_position = np.asarray(usd_positions[0], dtype=float)
    fabric_orientation = quat_normalize(fabric_orientations[0])
    usd_orientation = quat_normalize(usd_orientations[0])
    orientation_error = min(
        float(np.linalg.norm(fabric_orientation - usd_orientation)),
        float(np.linalg.norm(fabric_orientation + usd_orientation)),
    )
    return (
        float(np.linalg.norm(fabric_position - usd_position)),
        orientation_error,
    )



def set_planning_basket_obstacle_enabled(enabled):
    requested = bool(enabled)
    if state.planning_basket_obstacle is None:
        return True
    if state.planning_basket_obstacle_enabled == requested:
        return True
    method_name = "enable_rrt_obstacle" if requested else "disable_rrt_obstacle"
    controllers = []
    for controller in (
        state.controller,
        getattr(state, "_left_ik", None),
        getattr(state, "_right_ik", None),
    ):
        if controller is None or any(controller is item for item in controllers):
            continue
        controllers.append(controller)
    active_changed = False
    for controller in controllers:
        changed = bool(
            getattr(controller, method_name)(state.planning_basket_obstacle)
        )
        if controller is state.controller:
            active_changed = changed
    if not active_changed:
        print(
            "⚠️ 收纳箱 RRT 障碍状态切换失败: "
            f"requested={'enabled' if requested else 'disabled'}"
        )
        return False
    state.planning_basket_obstacle_enabled = requested
    print(
        "🧱 收纳箱 RRT 障碍已"
        + ("启用" if requested else "临时关闭，仅允许垂直放置")
    )
    return True

def get_finger_collision_world_corners(finger, side):
    local_min, local_max = DACH_JAW_COLLISION_LOCAL_BOUNDS[side]
    local_corners = np.asarray(
        [
            [x, y, z]
            for x in (local_min[0], local_max[0])
            for y in (local_min[1], local_max[1])
            for z in (local_min[2], local_max[2])
        ],
        dtype=float,
    )
    position, orientation, _ = get_sim_pose(finger)
    position = np.asarray(position, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    return np.asarray(
        [position + quat_rotate(orientation, corner) for corner in local_corners],
        dtype=float,
    )


def get_gripper_finger_midpoint():
    return get_gripper_collision_center()


def get_gripper_collision_center():
    left_center = np.mean(
        get_finger_collision_world_corners(state.left_finger, "left"), axis=0
    )
    right_center = np.mean(
        get_finger_collision_world_corners(state.right_finger, "right"), axis=0
    )
    return (left_center + right_center) * 0.5


def get_gripper_collision_diagnostics():
    diagnostics = {}
    for side, finger in (("left", state.left_finger), ("right", state.right_finger)):
        corners = get_finger_collision_world_corners(finger, side)
        diagnostics[side] = {
            "center": np.mean(corners, axis=0).tolist(),
            "min": np.min(corners, axis=0).tolist(),
            "max": np.max(corners, axis=0).tolist(),
        }
    return diagnostics


def get_gripper_table_clearance():
    if state.planning_table_surface_z is None:
        return float("inf")
    finger_bottoms = [
        float(np.min(get_finger_collision_world_corners(state.left_finger, "left")[:, 2])),
        float(np.min(get_finger_collision_world_corners(state.right_finger, "right")[:, 2])),
    ]
    return min(finger_bottoms) - state.planning_table_surface_z


def get_gripper_table_contact_safe_clearance():
    """Return a clearance outside the PhysX table contact generation range."""
    configured_clearance = (
        float(MIN_GRIPPER_TABLE_CLEARANCE)
        + float(TABLE_CLEARANCE_ABORT_MARGIN)
    )
    physx_clearance = (
        2.0 * max(float(PHYSX_CONTACT_OFFSET), 0.0)
        + max(float(PHYSX_REST_OFFSET), 0.0)
        + 0.001
    )
    return max(configured_clearance, physx_clearance)


def get_gripper_closing_axis():
    left_position, _, _ = get_sim_pose(state.left_finger)
    right_position, _, _ = get_sim_pose(state.right_finger)
    closing_axis = (
        np.asarray(right_position, dtype=float)
        - np.asarray(left_position, dtype=float)
    )
    closing_axis[2] = 0.0
    closing_axis_norm = np.linalg.norm(closing_axis)
    if closing_axis_norm < 1e-6:
        raise RuntimeError("无法从指尖位姿计算夹爪闭合轴")
    return closing_axis / closing_axis_norm


def get_gripper_inner_opening_width():
    """Measure the free gap between the two live jaw collision boxes."""
    left_corners = get_finger_collision_world_corners(state.left_finger, "left")
    right_corners = get_finger_collision_world_corners(state.right_finger, "right")
    closing_axis = get_gripper_closing_axis()
    left_projection = left_corners @ closing_axis
    right_projection = right_corners @ closing_axis
    left_center = float(np.mean(left_projection))
    right_center = float(np.mean(right_projection))
    if left_center <= right_center:
        return max(float(np.min(right_projection) - np.max(left_projection)), 0.0)
    return max(float(np.min(left_projection) - np.max(right_projection)), 0.0)


def get_gripper_center_local_offset():
    return DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET.copy()


def verify_gripper_center_local_offset():
    """Assert the hardcoded TCP->finger-center offset matches live geometry.

    Diagnostic only: measured on 2026-08-08 as deviation=0.0000 m, so the
    constant is trustworthy. Kept because a silent drift here misplaces every
    grasp, and the failure looks like a planning bug rather than a bad constant.
    """
    tcp_position, tcp_rotation = state.controller.get_end_effector_pose()
    world_delta = get_gripper_collision_center() - np.asarray(
        tcp_position, dtype=float
    )
    measured_offset = np.asarray(tcp_rotation, dtype=float).T @ world_delta
    deviation = float(
        np.linalg.norm(measured_offset - DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET)
    )
    print(
        f"📐 夹指中心局部偏移校验: measured={measured_offset}, "
        f"constant={DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET}, "
        f"deviation={deviation:.4f} m"
    )
    return deviation


def get_tcp_target_for_gripper_center(target_center, orientation):
    target_rotation = quat_to_rot_matrix(orientation)
    local_finger_offset = get_gripper_center_local_offset()
    tcp_target = np.asarray(target_center, dtype=float) - (
        np.asarray(target_rotation, dtype=float) @ local_finger_offset
    )
    print(
        f"🎯 夹指中心反算 TCP: center={target_center}, "
        f"local_offset={local_finger_offset}, tcp={tcp_target}"
    )
    return tcp_target


def clear_legacy_grasp_joints():
    """Remove attachment joints left by an older Aura/S5 runtime."""
    stage = get_current_stage()
    for joint_path in (
        "/World/AuraBananaGraspFixedJoint",
        "/World/S5BananaGraspFixedJoint",
    ):
        if stage.GetPrimAtPath(joint_path).IsValid():
            delete_prim(joint_path)
            print(f"🧹 已清理旧版抓取绑定: {joint_path}")


def get_active_joint_positions():
    active_joint_positions = state.controller.get_active_joint_positions()
    if active_joint_positions is None:
        return None
    if hasattr(active_joint_positions, "cpu"):
        active_joint_positions = active_joint_positions.cpu().numpy()
    active_joint_positions = np.asarray(active_joint_positions, dtype=float).reshape(-1)
    if (
        active_joint_positions.size != len(state.dach_arm.arm_joint_names)
        or not np.all(np.isfinite(active_joint_positions))
    ):
        return None
    return active_joint_positions


def get_left_joint_positions():
    left_joint_positions = state.dach_left.get_joint_positions()
    if left_joint_positions is None:
        return None
    left_joint_positions = np.asarray(left_joint_positions, dtype=float).reshape(-1)
    if (
        left_joint_positions.size != len(state.dach_left.arm_joint_names)
        or not np.all(np.isfinite(left_joint_positions))
    ):
        return None
    return left_joint_positions


def ensure_robot_control_ready(max_attempts=30):
    if not state.sim_context.is_playing():
        state.sim_context.play()
    for _ in range(max_attempts):
        step_app()
        if (
            get_active_joint_positions() is not None
            and get_left_joint_positions() is not None
        ):
            return

    print("⚠️ DACH 双臂关节状态不完整，重新初始化 physics view 和 Lula IK...")
    SimulationManager.initialize_physics()
    step_app(3)
    state.dach_arm.initialize()
    state.dach_left.initialize()
    state.controller.reset()
    state._left_ik.reset()
    step_app(10)
    for _ in range(max_attempts):
        step_app()
        if (
            get_active_joint_positions() is not None
            and get_left_joint_positions() is not None
        ):
            print("✅ DACH 双臂/Lula IK 控制状态已恢复")
            return
    raise RuntimeError(
        "DACH 双臂关节状态仍不完整，请确认时间线正在播放并重新加载 AuraVLA Isaac 执行器。"
    )


def get_rmp_ee_position():
    active_joint_positions = get_active_joint_positions()
    if active_joint_positions is None:
        raise RuntimeError(f"Lula IK 无法读取 DACH {DACH_ARM_SIDE} 臂主动关节状态")
    ee_position, _ = state.controller.get_end_effector_pose()
    return np.array(ee_position, dtype=float)

def move_ee_to(
    label,
    target_position,
    max_steps=300,
    tolerance=0.035,
    orientation=None,
    retry_position_only=True,
    horizontal_tolerance=None,
    gripper_positions=None,
    strict_orientation=False,
):
    print(f"➡️ {label}: target={target_position}")
    target_position = np.asarray(target_position, dtype=float)
    planning_started = time.perf_counter()
    if state.controller.rrt is not None:
        print(f"🧭 {label}: 使用 Lula RRT 碰撞规划")
        start_joints = get_active_joint_positions()
        joint_targets = state.controller.plan_collision_free_pose_path(
            target_position,
            target_orientation=orientation,
            start_joint_positions=start_joints,
        )
        if joint_targets is not None and orientation is not None:
            path_orientation_error = _max_path_orientation_error(
                joint_targets, orientation
            )
            if path_orientation_error > math.radians(15.0):
                print(
                    f"↪️ {label}: RRT 中途姿态偏差 "
                    f"{math.degrees(path_orientation_error):.1f}° > 15°，"
                    "拒绝执行扭转路径"
                )
                joint_targets = None
        if joint_targets is None and orientation is not None:
            # Task-space RRT may reject a reachable hover because it tries to
            # preserve the final wrist orientation throughout the whole path.
            # Resolve the exact endpoint with IK, then let RRT reach that joint
            # target while still checking every robot link against obstacles.
            endpoint_targets = state.controller.plan_pose_waypoints(
                [target_position],
                target_orientation=orientation,
                warm_start=start_joints,
                allow_orientation_fallback=False,
            )
            if endpoint_targets:
                joint_targets = state.controller.plan_collision_free_cspace_path(
                    endpoint_targets[-1]
                )
                if joint_targets is not None:
                    print(
                        f"🧭 {label}: 使用精确 IK 终点的关节空间 RRT 路径"
                    )
        if joint_targets is None:
            print(
                f"🛑 {label}: RRT 未找到无碰撞路径；拒绝笛卡尔 IK 回退"
            )
    else:
        print(f"🛑 {label}: Lula RRT 不可用；拒绝执行未验证碰撞的路径")
        joint_targets = None
    print(
        f"   {label} planning time: "
        f"{time.perf_counter() - planning_started:.2f} s"
    )
    if joint_targets is not None:
        _execute_joint_trajectory(
            label,
            joint_targets,
            gripper_positions=gripper_positions,
        )

        ee_pos = get_rmp_ee_position()
        dist = float(np.linalg.norm(ee_pos - target_position))
        horizontal_dist = float(np.linalg.norm(ee_pos[:2] - target_position[:2]))
        horizontal_reached = (
            horizontal_tolerance is None or horizontal_dist < horizontal_tolerance
        )
        print(
            f"   {label} planned path: ee={ee_pos}, dist={dist:.3f}, "
            f"xy_dist={horizontal_dist:.3f}"
        )
        if dist < tolerance and horizontal_reached:
            print(f"✅ {label} reached, dist={dist:.3f}, xy_dist={horizontal_dist:.3f}")
            return True

    print(f"⚠️ {label} planned path failed or missed target")
    if retry_position_only and orientation is not None:
        print(
            f"⚠️ {label}: 固定末端姿态不可达，拒绝 position-only fallback，"
            "避免腕部切换到扭曲 IK 分支"
        )
    return False


def _tcp_trace(ik_controller, joint_positions):
    ik_controller.reset()
    frame_name = ik_controller.kinematics.get_end_effector_frame()
    return np.asarray(
        [
            ik_controller.lula.compute_forward_kinematics(
                frame_name,
                np.asarray(joints, dtype=float),
            )[0]
            for joints in joint_positions
        ],
        dtype=float,
    )


def _execute_dual_joint_trajectory(
    label,
    left_joint_targets,
    right_joint_targets,
    left_gripper_positions=None,
    right_gripper_positions=None,
    monitor_table_clearance=False,
    max_joint_step_rad=None,
    minimum_frames=None,
):
    if left_joint_targets is None:
        raise RuntimeError(f"{label}: left joint trajectory planner returned None")
    if right_joint_targets is None:
        raise RuntimeError(f"{label}: right joint trajectory planner returned None")
    execution_started = time.perf_counter()
    ensure_robot_control_ready()
    left_start = get_left_joint_positions()
    right_start = get_active_joint_positions()
    if left_start is None or right_start is None:
        raise RuntimeError("DACH 双臂轨迹执行前无法读取有效关节状态")
    left_targets = [
        np.asarray(target, dtype=float).reshape(-1)
        for target in left_joint_targets
    ]
    right_targets = [
        np.asarray(target, dtype=float).reshape(-1)
        for target in right_joint_targets
    ]
    trajectory_diffuser = state._diffuser
    if max_joint_step_rad is not None or minimum_frames is not None:
        trajectory_diffuser = SparseKeyposeDiffuser(
            DiffusionConfig(
                cartesian_spacing_m=CARTESIAN_WAYPOINT_SPACING,
                max_joint_step_rad=float(
                    max_joint_step_rad
                    if max_joint_step_rad is not None
                    else TRAJECTORY_MAX_JOINT_STEP
                ),
                min_frames=int(
                    minimum_frames
                    if minimum_frames is not None
                    else TRAJECTORY_MIN_FRAMES
                ),
                dual_arm_min_tcp_separation_m=DUAL_ARM_MIN_TCP_SEPARATION,
            )
        )
    trajectory = trajectory_diffuser.synchronize_dual_arm_paths(
        left_start,
        left_targets,
        right_start,
        right_targets,
    )
    left_tcp_trace = _tcp_trace(state._left_ik, trajectory.left_positions)
    right_tcp_trace = _tcp_trace(state.controller, trajectory.right_positions)
    # ``state.dach_left`` is also the active controller when the scene is
    # configured for the left arm (for example, the can is left of the base).
    # In that mode both views address the same articulation/DOFs, so comparing
    # their traces reports a false 0 m separation and blocks every motion.
    same_controlled_arm = (
        getattr(state.dach_left, "arm_side", None)
        == getattr(state.dach_arm, "arm_side", None)
    )
    if same_controlled_arm:
        minimum_tcp_distance = float("inf")
        print(
            f"🧭 {label}: 单臂模式（{getattr(state.dach_arm, 'arm_side', 'unknown')}），"
            "跳过同一 articulation 的双臂 TCP 间距校验"
        )
    else:
        minimum_tcp_distance = trajectory_diffuser.validate_tcp_clearance(
            left_tcp_trace,
            right_tcp_trace,
        )
    print(
        f"🧭 {label}: sparse-keypose diffusion frames={trajectory.frame_count}, "
        f"min_tcp_distance={minimum_tcp_distance:.3f} m"
    )
    # This is debug-only: it converts the already planned active-arm joint
    # samples to TCP points and creates USD markers before commands are sent.
    # A rendering error must never alter or block the planned robot motion.
    try:
        path_visualization = render_joint_path(
            state.controller,
            [right_start, *trajectory.right_positions],
            # RRT may need extra internal joint nodes to remain collision-free.
            # Show at most three representative semantic waypoints; the blue
            # polyline still renders the complete executed trajectory.
            _cap_action_waypoints(right_targets),
        )
        if path_visualization["rendered"]:
            print(
                f"🎨 {label}: 路径可视化已更新: "
                f"points={path_visualization['continuous_point_count']}, "
                f"waypoints={path_visualization['waypoint_count']}"
            )
    except Exception as exc:
        print(f"⚠️ {label}: 路径可视化失败（不影响执行）: {exc}")
    state._task_motion_started = True

    clearance_abort_threshold = (
        float(os.environ.get("AURA_MIN_GRIPPER_TABLE_CLEARANCE", "0.012"))
        + float(os.environ.get("AURA_TABLE_CLEARANCE_ABORT_MARGIN", "0.006"))
    )
    clearance_violation_streak = 0
    clearance_violation_tolerance = int(
        os.environ.get("AURA_CLEARANCE_VIOLATION_TOLERANCE_FRAMES", "3")
    )
    minimum_table_clearance = float("inf")
    last_safe_left = left_start.copy()
    last_safe_right = right_start.copy()

    def verify_table_clearance(progress_label):
        nonlocal last_safe_left, last_safe_right, minimum_table_clearance
        nonlocal clearance_violation_streak
        if not monitor_table_clearance:
            return
        clearance = float(get_gripper_table_clearance())
        minimum_table_clearance = min(minimum_table_clearance, clearance)
        if clearance < clearance_abort_threshold:
            clearance_violation_streak += 1
            if clearance_violation_streak < clearance_violation_tolerance:
                return
            state.dach_left.teleport_arm_joint_positions(last_safe_left)
            state.dach_arm.teleport_arm_joint_positions(last_safe_right)
            state.dach_left.gripper.set_joint_positions(
                state.GRIPPER_OPEN_POSITIONS
                if left_gripper_positions is None
                else left_gripper_positions
            )
            state.dach_arm.gripper.set_joint_positions(
                state.GRIPPER_OPEN_POSITIONS
                if right_gripper_positions is None
                else right_gripper_positions
            )
            step_app(3)
            raise RuntimeError(
                f"{label} gripper table clearance safety stop "
                f"at {progress_label}: "
                f"observed={clearance:.4f} m, "
                f"abort_threshold={clearance_abort_threshold:.4f} m"
            )
        clearance_violation_streak = 0
        left_feedback = get_left_joint_positions()
        right_feedback = get_active_joint_positions()
        if left_feedback is not None and right_feedback is not None:
            last_safe_left = left_feedback.copy()
            last_safe_right = right_feedback.copy()

    verify_table_clearance("start")

    for frame_index, (left_joints, right_joints) in enumerate(
        zip(
            trajectory.left_positions,
            trajectory.right_positions,
        ),
        start=1,
    ):
        if left_gripper_positions is not None:
            state.dach_left.gripper.set_joint_positions(left_gripper_positions)
        if right_gripper_positions is not None:
            state.dach_arm.gripper.set_joint_positions(right_gripper_positions)
        state.dach_left.set_arm_joint_positions(left_joints)
        state.dach_arm.set_arm_joint_positions(right_joints)
        step_app()
        verify_table_clearance(f"trajectory_frame_{frame_index}")

    left_final = trajectory.left_positions[-1]
    right_final = trajectory.right_positions[-1]
    settle_limit = max(
        TRAJECTORY_SETTLE_FRAMES,
        min(max(TRAJECTORY_SETTLE_FRAMES * 3, 12), 36),
    )
    terminal_joint_error = float("inf")
    for settle_index in range(settle_limit):
        if left_gripper_positions is not None:
            state.dach_left.gripper.set_joint_positions(left_gripper_positions)
        if right_gripper_positions is not None:
            state.dach_arm.gripper.set_joint_positions(right_gripper_positions)
        state.dach_left.set_arm_joint_positions(left_final)
        state.dach_arm.set_arm_joint_positions(right_final)
        step_app()
        verify_table_clearance(f"settle_frame_{settle_index + 1}")
        left_feedback = get_left_joint_positions()
        right_feedback = get_active_joint_positions()
        if left_feedback is None or right_feedback is None:
            continue
        terminal_joint_error = max(
            float(np.max(np.abs(left_feedback - left_final))),
            float(np.max(np.abs(right_feedback - right_final))),
        )
        if (
            settle_index + 1 >= TRAJECTORY_SETTLE_FRAMES
            and terminal_joint_error <= 0.005
        ):
            break
    print(
        f"   {label} terminal joint error: "
        f"{terminal_joint_error:.5f} rad"
    )
    print(
        f"   {label} execution time: "
        f"{time.perf_counter() - execution_started:.2f} s"
    )
    if monitor_table_clearance:
        print(
            f"🛡️ {label} 最小夹指桌面净空: "
            f"{minimum_table_clearance:.4f} m"
        )
    return trajectory


def _execute_joint_trajectory(
    label,
    joint_targets,
    gripper_positions=None,
    monitor_table_clearance=False,
    max_joint_step_rad=None,
    minimum_frames=None,
):
    if joint_targets is None:
        raise RuntimeError(f"{label}: joint trajectory planner returned None")
    targets = [
        np.asarray(target_joints, dtype=float).reshape(-1)
        for target_joints in joint_targets
    ]
    if not targets:
        return
    trajectory = _execute_dual_joint_trajectory(
        label,
        [],
        targets,
        left_gripper_positions=state.GRIPPER_OPEN_POSITIONS,
        right_gripper_positions=gripper_positions,
        monitor_table_clearance=monitor_table_clearance,
        max_joint_step_rad=max_joint_step_rad,
        minimum_frames=minimum_frames,
    )
    print(f"   {label} trajectory executed: {len(targets)} planner waypoints")
    return trajectory


def _cap_action_waypoints(points):
    """Cap Cartesian action samples while preserving their first and last point."""
    samples = [np.asarray(point, dtype=float) for point in points]
    if len(samples) <= ACTION_WAYPOINT_LIMIT:
        return samples
    selected_indices = np.rint(
        np.linspace(0, len(samples) - 1, ACTION_WAYPOINT_LIMIT)
    ).astype(int)
    return [samples[index] for index in selected_indices]


def plan_ee_waypoints(waypoints, orientation=None):
    route_points = [np.asarray(point, dtype=float) for point in waypoints]
    if not route_points:
        return [], []
    sparse_keyposes = [get_rmp_ee_position(), *route_points]
    points = state._diffuser.diffuse_cartesian_keyposes(
        sparse_keyposes,
        max_spacing=CARTESIAN_WAYPOINT_SPACING,
    )[1:]
    points = _cap_action_waypoints(points)
    # Lula's continuous warm-start IK is the stable solver for this robot.
    # Cartesian points are dense; the controller keeps the requested terminal
    # orientation without forcing an exact quaternion at every sample, which
    # can jump to a different wrist branch near the basket.
    joint_targets = state.controller.plan_pose_waypoints(
        points,
        target_orientation=orientation,
        allow_orientation_fallback=orientation is None,
    )
    if joint_targets is None:
        return None
    return list(points), joint_targets


def plan_ee_waypoints_with_spacing(
    waypoints,
    orientation=None,
    max_spacing= CARTESIAN_WAYPOINT_SPACING,
):
    """Plan dense Cartesian waypoints with a caller-selected spacing."""
    route_points = [np.asarray(point, dtype=float) for point in waypoints]
    if not route_points:
        return [], []
    sparse_keyposes = [get_rmp_ee_position(), *route_points]
    points = state._diffuser.diffuse_cartesian_keyposes(
        sparse_keyposes,
        max_spacing=float(max_spacing),
    )[1:]
    points = _cap_action_waypoints(points)
    joint_targets = state.controller.plan_pose_waypoints(
        points,
        target_orientation=orientation,
        allow_orientation_fallback=orientation is None,
    )
    if joint_targets is None:
        return None
    return list(points), joint_targets


def plan_collision_free_keyposes(keyposes, orientation=None):
    state.last_collision_free_keypose_diagnostics = []
    if state.controller.rrt is None:
        print("🛑 Lula RRT 不可用；拒绝执行未验证碰撞的关键姿态路径")
        return None
    sparse_keyposes = [np.asarray(point, dtype=float) for point in keyposes]
    start_joints = get_active_joint_positions()
    if start_joints is None:
        return None
    joint_targets = []
    current_orientation = (
        None if orientation is None else quat_normalize(orientation)
    )
    for keypose_index, keypose in enumerate(sparse_keyposes, start=1):
        diagnostic = {
            "keypose_index": keypose_index,
            "keypose": keypose.tolist(),
            "task_space_rrt": False,
            "cspace_rrt": False,
        }
        state.last_collision_free_keypose_diagnostics.append(diagnostic)
        segment = state.controller.plan_collision_free_pose_path(
            keypose,
            target_orientation=current_orientation,
            start_joint_positions=start_joints,
        )
        diagnostic["task_space_rrt"] = segment is not None
        if segment is not None and current_orientation is not None:
            path_tilt_error = _max_path_approach_axis_error(
                segment, current_orientation
            )
            diagnostic["task_space_tilt_error_deg"] = math.degrees(
                path_tilt_error
            )
            if path_tilt_error > math.radians(20.0):
                print(
                    f"↪️ sparse keypose {keypose_index}: RRT 中途倾斜偏差 "
                    f"{math.degrees(path_tilt_error):.1f}° > 20°，"
                    "拒绝执行该路径"
                )
                segment = None
        if segment is None and current_orientation is not None:
            endpoint_targets = state.controller.plan_pose_waypoints(
                [keypose],
                target_orientation=current_orientation,
                warm_start=start_joints,
                allow_orientation_fallback=False,
            )
            diagnostic["endpoint_ik"] = bool(endpoint_targets)
            if endpoint_targets:
                candidate_segment = (
                    state.controller.plan_collision_free_cspace_path(
                        endpoint_targets[-1],
                        start_joint_positions=start_joints,
                    )
                )
                if candidate_segment is not None:
                    path_tilt_error = _max_path_approach_axis_error(
                        candidate_segment,
                        current_orientation,
                    )
                    diagnostic["cspace_tilt_error_deg"] = math.degrees(
                        path_tilt_error
                    )
                    if path_tilt_error <= math.radians(20.0):
                        segment = candidate_segment
                        diagnostic["cspace_rrt"] = True
                        print(
                            f"🧭 sparse keypose {keypose_index}: "
                            "使用精确 IK 终点的关节空间 RRT 路径"
                        )
        if segment is None and current_orientation is not None:
            # A fixed wrist yaw can make an otherwise reachable overhead
            # target fail IK. Rotate only around world Z at payload-safe
            # height; this preserves a top-down approach axis and every
            # candidate still requires a collision-aware RRT path.
            yaw_offsets_deg = [
                value
                for magnitude in range(15, 91, 15)
                for value in (magnitude, -magnitude)
            ]
            for yaw_offset_deg in yaw_offsets_deg:
                candidate_orientation = rotate_quat_about_world_z(
                    current_orientation,
                    math.radians(yaw_offset_deg),
                )
                candidate_segment = state.controller.plan_collision_free_pose_path(
                    keypose,
                    target_orientation=candidate_orientation,
                    start_joint_positions=start_joints,
                )
                candidate_strategy = "task_space_rrt"
                if candidate_segment is None:
                    candidate_endpoint = state.controller.plan_pose_waypoints(
                        [keypose],
                        target_orientation=candidate_orientation,
                        warm_start=start_joints,
                        allow_orientation_fallback=False,
                    )
                    if candidate_endpoint:
                        candidate_segment = (
                            state.controller.plan_collision_free_cspace_path(
                                candidate_endpoint[-1],
                                start_joint_positions=start_joints,
                            )
                        )
                        candidate_strategy = "cspace_rrt"
                if candidate_segment is None:
                    continue
                path_tilt_error = _max_path_approach_axis_error(
                    candidate_segment,
                    current_orientation,
                )
                if path_tilt_error > math.radians(20.0):
                    continue
                segment = candidate_segment
                current_orientation = candidate_orientation
                diagnostic["yaw_offset_deg"] = yaw_offset_deg
                diagnostic["yaw_strategy"] = candidate_strategy
                diagnostic["yaw_path_tilt_error_deg"] = math.degrees(
                    path_tilt_error
                )
                print(
                    f"🧭 sparse keypose {keypose_index}: "
                    f"采用碰撞感知的世界 Z 轴转向 "
                    f"{yaw_offset_deg:+d}°"
                )
                break
        if segment is None:
            diagnostic["success"] = False
            print(
                f"⚠️ sparse keypose {keypose_index}/{len(sparse_keyposes)} "
                "RRT 规划失败；未执行非碰撞感知回退"
            )
            return None
        diagnostic["success"] = True
        diagnostic["joint_waypoints"] = len(segment)
        joint_targets.extend(segment)
        start_joints = np.asarray(segment[-1], dtype=float)
    return sparse_keyposes, joint_targets, current_orientation


def move_ee_waypoints(
    label,
    waypoints,
    tolerance=0.05,
    orientation=None,
    gripper_positions=None,
    max_joint_step_rad=None,
    minimum_frames=None,
    max_spacing_m=None,
):
    planned_path = (
        plan_ee_waypoints_with_spacing(
            waypoints,
            orientation=orientation,
            max_spacing=max_spacing_m,
        )
        if max_spacing_m is not None
        else plan_ee_waypoints(waypoints, orientation=orientation)
    )
    if planned_path is None:
        print(f"⚠️ {label} 连续 waypoint IK 失败")
        return False
    points, joint_targets = planned_path
    if not points:
        return True
    print(
        f"🧭 安全路径 {label}: route_points={len(waypoints)}, "
        f"dense_waypoints={len(points)}"
    )
    _execute_joint_trajectory(
        label,
        joint_targets,
        gripper_positions=gripper_positions,
        max_joint_step_rad=max_joint_step_rad,
        minimum_frames=minimum_frames,
    )
    ee_position = get_rmp_ee_position()
    distance = float(np.linalg.norm(ee_position - points[-1]))
    print(f"   {label} reached: ee={ee_position}, dist={distance:.3f}")
    return distance <= tolerance

def move_ee_smooth(
    label,
    start_position,
    end_position,
    segments=6,
    max_steps_per_segment=180,
    tolerance=0.05,
    orientation=None,
    gripper_positions=None,
    monitor_table_clearance=False,
    max_joint_step_rad=None,
    minimum_frames=None,
    cartesian_waypoint_limit=None,
):
    actual_start = get_rmp_ee_position()
    end_position = np.asarray(end_position, dtype=float)
    waypoint_limit = (
        ACTION_WAYPOINT_LIMIT
        if cartesian_waypoint_limit is None
        else max(int(cartesian_waypoint_limit), 1)
    )
    waypoint_count = min(max(int(segments), 1), waypoint_limit)
    waypoints = [
        actual_start + (index / waypoint_count) * (end_position - actual_start)
        for index in range(1, waypoint_count + 1)
    ]
    print(
        f"🧭 平滑移动 {label}: cartesian_waypoints={waypoint_count}, "
        f"distance={np.linalg.norm(end_position - actual_start):.3f} m"
    )
    joint_targets = state.controller.plan_pose_waypoints(
        waypoints,
        target_orientation=orientation,
        allow_orientation_fallback=orientation is None,
    )
    if joint_targets is None:
        print(f"⚠️ {label} 连续 waypoint IK 失败")
        return False
    _execute_joint_trajectory(
        label,
        joint_targets,
        gripper_positions=gripper_positions,
        monitor_table_clearance=monitor_table_clearance,
        max_joint_step_rad=max_joint_step_rad,
        minimum_frames=minimum_frames,
    )
    ee_position = get_rmp_ee_position()
    distance = float(np.linalg.norm(ee_position - end_position))
    print(f"   {label} reached: ee={ee_position}, dist={distance:.3f}")
    if distance > tolerance and orientation is None:
        for refinement_index in range(2):
            correction_targets = state.controller.plan_pose_waypoints(
                [end_position],
                target_orientation=None,
            )
            if correction_targets is None:
                break
            print(
                f"↪️ {label}: TCP 终点闭环校正 "
                f"{refinement_index + 1}/2, error={distance:.3f} m"
            )
            _execute_joint_trajectory(
                f"{label}_terminal_refine_{refinement_index + 1}",
                correction_targets,
                gripper_positions=gripper_positions,
                monitor_table_clearance=monitor_table_clearance,
            )
            ee_position = get_rmp_ee_position()
            distance = float(np.linalg.norm(ee_position - end_position))
            print(
                f"   {label} refined: ee={ee_position}, "
                f"dist={distance:.3f}"
            )
            if distance <= tolerance:
                break
    return distance <= tolerance


def move_ee_collision_aware_approach(
    label,
    target_position,
    *,
    tolerance=0.035,
    orientation=None,
    gripper_positions=None,
    desired_closing_axis=None,
    minimum_closing_alignment=0.0,
    maximum_approach_tilt_rad=None,
):
    if state.controller.rrt is None:
        return {
            "success": move_ee_smooth(
                label,
                get_rmp_ee_position(),
                target_position,
                tolerance=tolerance,
                orientation=orientation,
                gripper_positions=gripper_positions,
                monitor_table_clearance=True,
            ),
            "orientation_constrained": orientation is not None,
            "planner": "cartesian_ik",
        }

    planning_started = time.perf_counter()
    start_joints = get_active_joint_positions()
    initial_orientation = None if desired_closing_axis is not None else orientation
    joint_targets = None
    planner_name = "lula_rrt"
    if label == "grasp_approach" and orientation is not None:
        current_position = get_rmp_ee_position()
        target_position_array = np.asarray(target_position, dtype=float)
        waypoint_count = min(
            max(
                int(
                    np.ceil(
                        np.linalg.norm(target_position_array - current_position)
                        / 0.01
                    )
                ),
                1,
            ),
            ACTION_WAYPOINT_LIMIT,
        )
        cartesian_waypoints = [
            current_position
            + (index / waypoint_count)
            * (target_position_array - current_position)
            for index in range(1, waypoint_count + 1)
        ]
        joint_targets = state.controller.plan_pose_waypoints(
            cartesian_waypoints,
            target_orientation=orientation,
            warm_start=start_joints,
            allow_orientation_fallback=False,
        )
        if joint_targets is not None:
            planner_name = "fixed_orientation_cartesian_ik"
            print(
                f"🧭 {label}: 使用单调笛卡尔下降，"
                f"waypoints={len(cartesian_waypoints)}"
            )
    if joint_targets is None:
        joint_targets = state.controller.plan_collision_free_pose_path(
            target_position,
            target_orientation=initial_orientation,
            start_joint_positions=start_joints,
        )
    orientation_constrained = (
        joint_targets is not None and initial_orientation is not None
    )
    hold_orientation = (
        np.asarray(orientation, dtype=float).copy()
        if orientation_constrained
        else None
    )
    if joint_targets is None and orientation is not None:
        print(
            f"⚠️ {label}: 固定末端姿态 RRT 不可达，"
            "不再使用 position-only 路径，避免腕部扭曲"
        )
    if joint_targets is None:
        tcp_table_clearance = (
            float(target_position[2]) - float(state.planning_table_surface_z)
            if state.planning_table_surface_z is not None
            else float("inf")
        )
        if tcp_table_clearance >= 0.20:
            print(
                f"↪️ {label}: RRT 下降失败，使用逐点 IK 直线下降，"
                f"tcp_table_clearance={tcp_table_clearance:.3f} m"
            )
            waypoint_count = min(
                max(
                    int(
                        np.ceil(
                            np.linalg.norm(
                                np.asarray(target_position, dtype=float)
                                - get_rmp_ee_position()
                            )
                            / 0.02
                        )
                    ),
                    1,
                ),
                ACTION_WAYPOINT_LIMIT,
            )
            current_position = get_rmp_ee_position()
            cartesian_waypoints = [
                current_position
                + (index / waypoint_count)
                * (np.asarray(target_position, dtype=float) - current_position)
                for index in range(1, waypoint_count + 1)
            ]
            joint_targets = state.controller.plan_pose_waypoints(
                cartesian_waypoints,
                target_orientation=orientation,
                warm_start=start_joints,
                allow_orientation_fallback=False,
            )
            orientation_constrained = orientation is not None
        if joint_targets is None:
            print(f"⚠️ {label}: RRT 和逐点 IK 均未找到安全下降路径")
            return {
                "success": False,
                "orientation_constrained": False,
                "planner": "lula_rrt+cartesian_ik",
            }

    print(
        f"🧭 {label}: 执行 {planner_name}，planner_waypoints={len(joint_targets)}, "
        f"planning_time={time.perf_counter() - planning_started:.2f} s"
    )
    try:
        _execute_joint_trajectory(
            label,
            joint_targets,
            gripper_positions=gripper_positions,
            monitor_table_clearance=True,
            max_joint_step_rad=(
                GRASP_APPROACH_MAX_JOINT_STEP
                if label == "grasp_approach"
                else None
            ),
            minimum_frames=(
                GRASP_APPROACH_MIN_FRAMES
                if label == "grasp_approach"
                else None
            ),
        )
        orientation_refined = False
        if not orientation_constrained and orientation is not None:
            frame_name = state.controller.kinematics.get_end_effector_frame()
            current_joints = get_active_joint_positions()
            _, actual_rotation = state.controller.lula.compute_forward_kinematics(
                frame_name,
                current_joints,
            )
            actual_orientation = np.asarray(
                rot_matrix_to_quat(actual_rotation),
                dtype=float,
            )
            desired_orientation = np.asarray(orientation, dtype=float).copy()
            if np.dot(actual_orientation, desired_orientation) < 0.0:
                desired_orientation = -desired_orientation

            refinement_targets = []
            refined_orientation = None
            warm_start = current_joints.copy()
            for alpha in np.linspace(0.01, 1.0, 100):
                candidate_orientation = (
                    (1.0 - alpha) * actual_orientation
                    + alpha * desired_orientation
                )
                candidate_orientation /= np.linalg.norm(candidate_orientation)
                solution, success = state.controller.lula.compute_inverse_kinematics(
                    frame_name,
                    np.asarray(target_position, dtype=float),
                    candidate_orientation,
                    warm_start,
                )
                if not success:
                    refinement_targets = []
                    break
                warm_start = np.asarray(solution, dtype=float).reshape(-1)
                refinement_targets.append(warm_start.copy())
                refined_orientation = candidate_orientation.copy()
                refined_approach = quat_rotate(
                    refined_orientation,
                    np.array([1.0, 0.0, 0.0]),
                )
                refined_tilt = float(
                    np.arccos(
                        np.clip(
                            np.dot(
                                refined_approach,
                                np.array([0.0, 0.0, -1.0]),
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                refined_rotation = quat_to_rot_matrix(refined_orientation)
                refined_closing_axis = np.asarray(
                    refined_rotation[:2, 1], dtype=float
                )
                refined_closing_axis /= np.linalg.norm(refined_closing_axis)
                closing_alignment = (
                    float(
                        abs(
                            np.dot(
                                refined_closing_axis,
                                np.asarray(desired_closing_axis, dtype=float)[:2],
                            )
                        )
                    )
                    if desired_closing_axis is not None
                    else 1.0
                )
                orientation_ready = (
                    closing_alignment >= float(minimum_closing_alignment)
                    and refined_tilt <= TARGET_GRASP_APPROACH_TILT_RAD
                )
                if orientation_ready:
                    break

            if refinement_targets:
                refinement_goal = (
                    "香蕉短轴闭合姿态"
                    if desired_closing_axis is not None
                    else "目标向下姿态"
                )
                print(
                    f"🧭 {label}: 固定 TCP 渐进收敛到{refinement_goal}，"
                    f"keyposes={len(refinement_targets)}"
                )
                _execute_joint_trajectory(
                    f"{label}_orientation_refine",
                    refinement_targets,
                    gripper_positions=gripper_positions,
                    monitor_table_clearance=True,
                )
                orientation_refined = True
                orientation_constrained = True
                hold_orientation = refined_orientation
    except RuntimeError as exc:
        print(f"🛑 {label}: {exc}")
        return {
            "success": False,
            "orientation_constrained": orientation_constrained,
            "planner": planner_name,
            "error": str(exc),
        }

    ee_position = get_rmp_ee_position()
    distance = float(
        np.linalg.norm(ee_position - np.asarray(target_position, dtype=float))
    )
    clearance = float(get_gripper_table_clearance())
    _, ee_rotation = state.controller.get_end_effector_pose()
    actual_approach = np.asarray(ee_rotation, dtype=float)[:, 0]
    actual_approach /= np.linalg.norm(actual_approach)
    downward_alignment = float(
        np.clip(np.dot(actual_approach, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    )
    actual_tilt = float(np.arccos(downward_alignment))
    actual_closing_axis = np.asarray(ee_rotation, dtype=float)[:2, 1]
    actual_closing_axis /= np.linalg.norm(actual_closing_axis)
    closing_alignment = (
        float(
            abs(
                np.dot(
                    actual_closing_axis,
                    np.asarray(desired_closing_axis, dtype=float)[:2],
                )
            )
        )
        if desired_closing_axis is not None
        else 1.0
    )
    print(
        f"   {label} reached: ee={ee_position}, dist={distance:.3f}, "
        f"finger_clearance={clearance:.4f} m, "
        f"downward_tilt={np.degrees(actual_tilt):.1f}°, "
        f"closing_alignment={closing_alignment:.3f}"
    )
    tilt_limit = (
        np.radians(float(os.environ.get("AURA_MAX_GRASP_APPROACH_TILT_DEG", "60.0")))
        if maximum_approach_tilt_rad is None
        else float(maximum_approach_tilt_rad)
    )
    return {
        "success": bool(
            distance <= tolerance
            and clearance >= float(os.environ.get("AURA_MIN_GRIPPER_TABLE_CLEARANCE", "0.012"))
            # Lula/PhysX feedback around an exact limit can differ by a few
            # thousandths of a degree (for example 45.006° for a 45° target).
            and actual_tilt <= tilt_limit + math.radians(0.25)
            and closing_alignment >= float(minimum_closing_alignment)
        ),
        "orientation_constrained": orientation_constrained,
        "planner": planner_name,
        "distance_m": distance,
        "finger_clearance_m": clearance,
        "downward_tilt_deg": float(np.degrees(actual_tilt)),
        "closing_alignment": closing_alignment,
        "orientation_refined": orientation_refined,
        "hold_orientation": (
            None if hold_orientation is None else hold_orientation.tolist()
        ),
    }

def hold_ee_target(target_position, orientation=None):
    action = state.controller.forward(
        target_end_effector_position=target_position,
        target_end_effector_orientation=orientation,
    )
    state.dach_arm.apply_action(action)


def get_gripper_joint_efforts():
    try:
        measured_efforts = np.asarray(
            state.dach_arm.articulation.get_measured_joint_efforts(), dtype=float
        )
        return np.abs(measured_efforts[state.dach_arm._gripper_indices])
    except (AttributeError, IndexError, TypeError, ValueError):
        return np.full(2, np.nan, dtype=float)

# 【修改】close_gripper_slowly 增加 target_close 参数，允许自定义闭合位置
def close_gripper_slowly(
    hold_position,
    orientation=None,
    frames=30,
    target_close=None,
    monitor_table_clearance=False,
):
    print("🤏 双指独立力反馈闭合夹爪...")
    start_positions = np.array(state.dach_arm.gripper.get_joint_positions(), dtype=float)
    if target_close is None:
        target_close = np.array(state.dach_arm.gripper.joint_closed_positions, dtype=float)
    else:
        target_close = np.array(target_close, dtype=float)
    # Servo lag must stay well under GRIPPER_CONTACT_RESIDUAL, or the ramp
    # outruns the jaw and the resulting position error is latched as contact.
    # minimum_jerk peaks at 1.875x the mean velocity.
    peak_commanded_step = 1.875 * float(
        np.max(np.abs(target_close - start_positions))
    ) / max(int(frames), 1)
    print(
        f"📐 闭合斜坡: frames={frames}, "
        f"峰值步长={peak_commanded_step * 1000:.3f} mm, "
        f"接触残差阈值={GRIPPER_CONTACT_RESIDUAL * 1000:.3f} mm, "
        f"占比={peak_commanded_step / max(GRIPPER_CONTACT_RESIDUAL, 1e-9):.0%}"
    )
    hold_target = target_close.copy()
    finger_count = len(start_positions)
    contact_streaks = np.zeros(finger_count, dtype=int)
    contacted = np.zeros(finger_count, dtype=bool)
    previous_feedback = start_positions.copy()
    blocked_residuals = np.zeros_like(start_positions)
    commanded_positions = start_positions.copy()
    for i in range(frames):
        linear_progress = min(1.0, (i + 1) / frames)
        alpha = float(minimum_jerk(linear_progress))
        closing_positions = start_positions + alpha * (
            target_close - start_positions
        )
        # The physical DACH linkage closes both jaws symmetrically. Do not
        # freeze one simulated jaw on position lag; that creates an asymmetric
        # gripper and was repeatedly stopping the left jaw at 26 mm with no
        # measured contact force.
        commanded_positions = closing_positions
        state.dach_arm.gripper.set_joint_positions(commanded_positions)
        hold_ee_target(hold_position, orientation)
        step_app()
        feedback = np.asarray(
            state.dach_arm.gripper.get_joint_positions(), dtype=float
        )
        blocked_residuals = np.maximum(feedback - commanded_positions, 0.0)
        measured_efforts = get_gripper_joint_efforts()
        feedback_steps = np.abs(feedback - previous_feedback)
        requested_steps = np.abs(
            commanded_positions - previous_feedback
        )
        effort_feedback_available = np.isfinite(measured_efforts)
        contact_candidates = (
            i >= max(4, frames // 5)
        ) & (
            blocked_residuals >= GRIPPER_CONTACT_RESIDUAL
        ) & (
            feedback_steps <= np.maximum(requested_steps * 0.5, 0.00025)
        ) & (
            (~effort_feedback_available)
            | (measured_efforts >= GRIPPER_CONTACT_FORCE_THRESHOLD)
        )
        contact_streaks = np.where(
            contacted,
            contact_streaks,
            np.where(contact_candidates, contact_streaks + 1, 0),
        )
        newly_contacted = (~contacted) & (contact_streaks >= 3)
        if np.any(newly_contacted):
            contacted[newly_contacted] = True
            print(
                "✅ 夹指持续受力接触: "
                f"step={i + 1}, contacted={contacted.tolist()}, "
                f"commands={commanded_positions}, "
                f"feedback={feedback}, residuals={blocked_residuals} m, "
                f"efforts={measured_efforts} N"
            )
        previous_feedback = feedback.copy()
        if monitor_table_clearance:
            clearance = float(get_gripper_table_clearance())
            abort_threshold = get_gripper_table_contact_safe_clearance()
            if clearance < abort_threshold:
                state.dach_arm.gripper.set_joint_positions(start_positions)
                step_app(2)
                raise RuntimeError(
                    "gripper table clearance safety stop while closing: "
                    f"observed={clearance:.4f} m, "
                    f"abort_threshold={abort_threshold:.4f} m"
                )
        if np.all(contacted):
            hold_target = np.maximum(
                commanded_positions - GRIPPER_CONTACT_HOLD_PRELOAD,
                target_close,
            )
            print(
                "✅ 双指均已接触，开始对称渐进预紧: "
                f"step={i + 1}, contact_commands={commanded_positions}, "
                f"hold_target={hold_target}, "
                f"feedback={feedback}, residuals={blocked_residuals} m"
            )
            break
        if i % 15 == 0:
            print(
                f"   close step={i}, command={commanded_positions}, "
                f"feedback={feedback}, residuals={blocked_residuals} m"
            )
    settle_start_target = commanded_positions.copy()
    hold_target = settle_start_target.copy()
    preload_limit = np.maximum(
        settle_start_target - GRIPPER_CONTACT_HOLD_PRELOAD,
        target_close,
    )
    preload_step = GRIPPER_CONTACT_HOLD_PRELOAD / max(
        GRIPPER_CONTACT_SETTLE_FRAMES,
        1,
    )
    preload_streak = 0
    for settle_index in range(GRIPPER_CONTACT_SETTLE_FRAMES):
        feedback = np.asarray(
            state.dach_arm.gripper.get_joint_positions(), dtype=float
        )
        residuals = np.maximum(feedback - hold_target, 0.0)
        measured_efforts = get_gripper_joint_efforts()
        effort_feedback_available = np.all(np.isfinite(measured_efforts))
        needs_preload = (
            measured_efforts < GRIPPER_CONTACT_FORCE_THRESHOLD
            if effort_feedback_available
            else residuals < GRIPPER_CONTACT_PRELOAD_RESIDUAL
        )
        hold_target = np.where(
            needs_preload,
            np.maximum(hold_target - preload_step, preload_limit),
            hold_target,
        )
        state.dach_arm.gripper.set_joint_positions(hold_target)
        hold_ee_target(hold_position, orientation)
        step_app()
        feedback = np.asarray(
            state.dach_arm.gripper.get_joint_positions(), dtype=float
        )
        residuals = np.maximum(feedback - hold_target, 0.0)
        measured_efforts = get_gripper_joint_efforts()
        effort_feedback_available = np.all(np.isfinite(measured_efforts))
        preload_confirmed = (
            np.all(measured_efforts >= GRIPPER_CONTACT_FORCE_THRESHOLD)
            if effort_feedback_available
            else np.all(residuals >= GRIPPER_CONTACT_PRELOAD_RESIDUAL)
        )
        if preload_confirmed:
            preload_streak += 1
        else:
            preload_streak = 0
        if settle_index % 5 == 0 or preload_streak >= 1:
            print(
                "   adaptive preload "
                f"step={settle_index + 1}, target={hold_target}, "
                f"feedback={feedback}, residuals={residuals} m, "
                f"efforts={measured_efforts} N"
            )
        if monitor_table_clearance:
            clearance = float(get_gripper_table_clearance())
            abort_threshold = get_gripper_table_contact_safe_clearance()
            if clearance < abort_threshold:
                state.dach_arm.gripper.set_joint_positions(start_positions)
                step_app(2)
                raise RuntimeError(
                    "gripper table clearance safety stop while settling: "
                    f"observed={clearance:.4f} m, "
                    f"abort_threshold={abort_threshold:.4f} m"
                )
        # One qualifying frame is not a grip -- effort spikes transiently as the
        # drive accelerates.  Require the threshold to hold for several frames
        # so the preload loop cannot exit before real force is established.
        if preload_streak >= GRIPPER_PRELOAD_CONFIRM_FRAMES:
            break
    final_feedback = np.asarray(
        state.dach_arm.gripper.get_joint_positions(), dtype=float
    )
    final_residuals = np.maximum(final_feedback - hold_target, 0.0)
    final_residual = float(np.min(final_residuals))
    final_efforts = get_gripper_joint_efforts()
    effort_feedback_available = np.all(np.isfinite(final_efforts))
    residual_contact_confirmed = bool(
        np.all(final_residuals >= GRIPPER_CONTACT_PRELOAD_RESIDUAL)
    )
    effort_contact_confirmed = bool(
        effort_feedback_available
        and np.all(final_efforts >= GRIPPER_CONTACT_FORCE_THRESHOLD)
    )
    final_finger_contacts = (
        (final_efforts >= GRIPPER_CONTACT_FORCE_THRESHOLD)
        if effort_feedback_available
        else (final_residuals >= GRIPPER_CONTACT_PRELOAD_RESIDUAL)
    )
    contact_confirmed = bool(
        np.all(final_finger_contacts)
        and (effort_contact_confirmed or residual_contact_confirmed)
    )
    return {
        "contact_confirmed": contact_confirmed,
        "finger_contacts": np.asarray(final_finger_contacts, dtype=bool),
        "hold_target": hold_target.copy(),
        "feedback": final_feedback.copy(),
        "blocked_residual_m": final_residual,
        "blocked_residuals_m": final_residuals.copy(),
        "measured_efforts_n": final_efforts.copy(),
    }

def open_gripper_slowly(
    hold_position,
    orientation=None,
    frames=30,
    target_open=None,
):
    print("✋ 慢速打开夹爪...")
    start_positions = np.array(state.dach_arm.gripper.get_joint_positions(), dtype=float)
    target_positions = (
        state.GRIPPER_OPEN_POSITIONS.copy()
        if target_open is None
        else np.asarray(target_open, dtype=float)
    )
    for i in range(frames):
        alpha = min(1.0, (i + 1) / frames)
        finger_positions = start_positions + alpha * (target_positions - start_positions)
        state.dach_arm.gripper.set_joint_positions(finger_positions)
        hold_ee_target(hold_position, orientation)
        step_app()
        if i % 15 == 0:
            print(f"   open step={i}, finger_pos={state.dach_arm.gripper.get_joint_positions()}")
    # The interpolation duration controls command smoothness, not physical
    # convergence.  With the finite-effort jaw drives, 18-30 ramp frames can
    # end while the fingers are still only half open.  Hold the terminal
    # command until the measured joints settle so later geometry calibration
    # cannot drift during descent.
    settled_frames = 0
    final_feedback = np.asarray(
        state.dach_arm.gripper.get_joint_positions(), dtype=float
    )
    for settle_index in range(max(30, int(frames) * 3)):
        state.dach_arm.gripper.set_joint_positions(target_positions)
        hold_ee_target(hold_position, orientation)
        step_app()
        final_feedback = np.asarray(
            state.dach_arm.gripper.get_joint_positions(), dtype=float
        )
        open_error = float(np.max(np.abs(final_feedback - target_positions)))
        settled_frames = settled_frames + 1 if open_error <= 0.001 else 0
        if settled_frames >= 3:
            break
    final_error = float(np.max(np.abs(final_feedback - target_positions)))
    print(
        f"✋ 夹爪张开收敛: feedback={final_feedback}, "
        f"target={target_positions}, error={final_error:.4f} m"
    )
    return {
        "feedback": final_feedback.copy(),
        "target": target_positions.copy(),
        "error_m": final_error,
        "converged": final_error <= 0.001,
    }

def move_robot_home(frames=30, joint_tolerance=0.01):
    ensure_robot_control_ready()
    right_start = get_active_joint_positions()
    left_start = get_left_joint_positions()
    if right_start is None or left_start is None:
        raise RuntimeError("DACH 双臂归位前无法读取有效关节状态")
    right_target = (
        LEFT_ARM_HOME.copy()
        if getattr(state.dach_arm, "arm_side", None) == "left"
        else RIGHT_ARM_HOME.copy()
    )
    left_target = LEFT_ARM_HOME.copy()
    if right_start.shape != right_target.shape or left_start.shape != left_target.shape:
        raise RuntimeError(
            "DACH 归位关节维度不匹配: "
            f"left={left_start.shape}/{left_target.shape}, "
            f"right={right_start.shape}/{right_target.shape}"
        )
    right_error = float(np.max(np.abs(right_start - right_target)))
    left_error = float(np.max(np.abs(left_start - left_target)))
    max_joint_error = max(left_error, right_error)
    state.dach_arm.gripper.set_joint_positions(state.GRIPPER_OPEN_POSITIONS)
    state.dach_left.gripper.set_joint_positions(state.GRIPPER_OPEN_POSITIONS)
    if max_joint_error <= float(joint_tolerance):
        step_app(3)
        print(
            "🏠 双臂已在初始姿态，跳过重复归位: "
            f"max_joint_error={max_joint_error:.5f} rad"
        )
        return True

    print(
        "🏠 双臂同步归位... "
        f"left_error={left_error:.5f}, right_error={right_error:.5f} rad"
    )
    left_targets = state._left_ik.plan_collision_free_cspace_path(left_target)
    right_targets = state.controller.plan_collision_free_cspace_path(right_target)
    if left_targets is None:
        print("⚠️ left home: RRT 未找到路径，使用稀疏关节关键姿态")
        left_targets = [left_target]
    if right_targets is None:
        print("⚠️ right home: RRT 未找到路径，使用稀疏关节关键姿态")
        right_targets = [right_target]
    _execute_dual_joint_trajectory(
        "dual_home",
        left_targets,
        right_targets,
        left_gripper_positions=state.GRIPPER_OPEN_POSITIONS,
        right_gripper_positions=state.GRIPPER_OPEN_POSITIONS,
    )
    left_feedback = get_left_joint_positions()
    right_feedback = get_active_joint_positions()
    if left_feedback is None or right_feedback is None:
        ensure_robot_control_ready()
        left_feedback = get_left_joint_positions()
        right_feedback = get_active_joint_positions()
    if left_feedback is None or right_feedback is None:
        raise RuntimeError("DACH 双臂归位后无法读取有效关节反馈")
    left_residual = float(np.max(np.abs(left_feedback - left_target)))
    right_residual = float(np.max(np.abs(right_feedback - right_target)))
    if max(left_residual, right_residual) > float(joint_tolerance):
        print(
            "⚠️ 双臂驱动归位未完全收敛，执行终点状态对齐: "
            f"left={left_residual:.5f}, right={right_residual:.5f} rad"
        )
        state.dach_left.teleport_arm_joint_positions(left_target)
        state.dach_arm.teleport_arm_joint_positions(right_target)
        step_app(5)
        left_residual = float(
            np.max(np.abs(state.dach_left.get_joint_positions() - left_target))
        )
        right_residual = float(
            np.max(np.abs(state.dach_arm.get_joint_positions() - right_target))
        )
    same_controlled_arm = (
        getattr(state.dach_left, "arm_side", None)
        == getattr(state.dach_arm, "arm_side", None)
    )
    home_residual_limit = 0.05 if same_controlled_arm else 0.02
    if max(left_residual, right_residual) > home_residual_limit:
        raise RuntimeError(
            "DACH 双臂归位反馈超差: "
            f"left={left_residual:.5f}, right={right_residual:.5f} rad"
        )
    return True


def reset_robot_after_task():
    """Best-effort cleanup that must run after every bridge task."""
    cleanup_errors = []
    try:
        clear_legacy_grasp_joints()
    except Exception as exc:
        cleanup_errors.append(f"legacy grasp cleanup: {type(exc).__name__}: {exc}")

    try:
        ensure_robot_control_ready()
        state.dach_arm.gripper.set_joint_positions(state.GRIPPER_OPEN_POSITIONS)
        state.dach_left.gripper.set_joint_positions(state.GRIPPER_OPEN_POSITIONS)
        step_app(3)
        basket_retreat_ok = True
        if (
            state.planning_basket_obstacle is not None
            and not state.planning_basket_obstacle_enabled
        ):
            retreat_start = get_rmp_ee_position()
            basket_retreat_ok = move_ee_smooth(
                "basket_cleanup_vertical_retreat",
                retreat_start,
                retreat_start + np.array([0.0, 0.0, 0.20]),
                segments=3,
                tolerance=0.05,
                gripper_positions=state.GRIPPER_OPEN_POSITIONS,
            )
            if basket_retreat_ok:
                basket_retreat_ok = set_planning_basket_obstacle_enabled(True)
        if basket_retreat_ok:
            move_robot_home(frames=90)
        else:
            cleanup_errors.append(
                "home: skipped because basket vertical retreat failed"
            )
    except Exception as exc:
        cleanup_errors.append(f"home: {type(exc).__name__}: {exc}")

    if cleanup_errors:
        print("⚠️ 任务结束复位未完全成功: " + "; ".join(cleanup_errors))
        return {"success": False, "errors": cleanup_errors}
    print("✅ 任务结束复位完成：夹爪已打开，机械臂已回初始姿态")
    return {"success": True}
