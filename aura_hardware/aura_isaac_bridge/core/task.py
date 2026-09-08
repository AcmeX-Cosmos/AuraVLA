"""AuraVLA 任务执行模块：抓取位姿调整与 Pick-and-Place 流程。"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import (
    euler_angles_to_quat,
    quat_to_euler_angles,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
)
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from aura_isaac_bridge.core.state import state
from aura_isaac_bridge.core.state import (
    DACH_ARM_SIDE, DACH_BASE_XY,
    GRASP_MIN_SHORT_AXIS_ALIGNMENT, GRASP_APERTURE_MARGIN,
    GRASP_PLANAR_REFINEMENT_STEPS, GRASP_PLANAR_CENTER_TOLERANCE,
    GRASP_MAX_PLANAR_CORRECTION,
    GRASP_APPROACH_ROLL_CANDIDATES_DEG,
    GRASP_CONTACT_CONFIRMATION_MARGIN,
    GRASP_MINIMUM_PHYSICAL_OPENING_MARGIN,
    GRASP_LOW_POSE_CORRECTION_LIMIT,
    GRASP_CLEARANCE_GUARD_PAD,
    DACH_GRASP_HEIGHT_OFFSET, DACH_GRASP_YAW_OFFSET_RAD,
    MAX_GRASP_APPROACH_TILT_RAD,
    TARGET_GRASP_APPROACH_TILT_RAD,
    GRASP_REFINEMENT_STEPS, GRIPPER_CLOSE_FRAMES,
    GRIPPER_MAX_EFFORT, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    GRIPPER_CONTACT_RESIDUAL, GRIPPER_CONTACT_FORCE_THRESHOLD,
    GRIPPER_CONTACT_PRELOAD_RESIDUAL, GRIPPER_CONTACT_HOLD_PRELOAD,
    GRIPPER_CONTACT_SETTLE_FRAMES,
    DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET,
    DACH_JAW_COLLISION_LOCAL_BOUNDS,
    DIRECTIONAL_PLACE_DISTANCE,
    GRASP_MIN_HEIGHT_FRACTION, MINIMUM_OBJECT_LIFT,
    PLACE_SUCCESS_TOLERANCE, PLACE_MIN_HEIGHT_ABOVE_TABLE,
    DACH_PATH_CLEARANCE,
    TRAJECTORY_MAX_JOINT_STEP, TRAJECTORY_MIN_FRAMES, TRAJECTORY_SETTLE_FRAMES,
    GRASP_APPROACH_MAX_JOINT_STEP, GRASP_APPROACH_MIN_FRAMES,
    GRASP_LIFT_MAX_JOINT_STEP, GRASP_LIFT_MIN_FRAMES,
    CARTESIAN_WAYPOINT_SPACING, CARRY_APEX_CLEARANCE,
    TRANSPORT_LIFT_HEIGHT, CARRY_CARTESIAN_WAYPOINT_SPACING,
    CARRY_MAX_JOINT_STEP, CARRY_MIN_FRAMES,
    CARRY_REPLAN_CHECK_WAYPOINTS, CARRY_REPLAN_POSITION_TOLERANCE_M,
    CARRY_REPLAN_MAX_ATTEMPTS, CARRY_REPLAN_FRAME_COUNT,
    SHOW_GRASP_DEBUG, USE_ANYGRASP, USE_ANYGRASP_ORIENTATION,
    USE_GRASPNET,
    GRASP_BACKEND,
    GRASP_POSITION_OFFSET, GRASP_INSERT_DEPTH,
    MIN_GRIPPER_TABLE_CLEARANCE, TABLE_CLEARANCE_ABORT_MARGIN,
    BASKET_RESET_POSITION, BASKET_RESET_ORIENTATION,
    BASKET_PLANNING_MARGIN, BASKET_PLACE_TABLE_CLEARANCE,
    DUAL_ARM_MIN_TCP_SEPARATION,
)
from aura_isaac_bridge.core.physics import step_app, ensure_pickable_object
from aura_isaac_bridge.core.gripper_contact import (
    classify_finger_contacts,
    evaluate_grasp_stability,
)
from aura_isaac_bridge.core.perception import (
    get_sim_pose,
    quat_rotate,
    resolve_scene_prim_path,
    get_bbox_center,
    get_current_bbox_center,
    get_mesh_center,
    get_mesh_extent_along_axis,
    get_mesh_horizontal_min_width_axis,
    get_current_mesh_horizontal_min_width_axis,
    get_current_mesh_horizontal_cross_section_center,
    get_current_mesh_horizontal_cross_section_geometry,
    get_mesh_horizontal_principal_axes,
    show_red_grasp_point,
    release_cuda_inference_cache,
    infer_grasp_fused_world_pose,
)
from aura_isaac_bridge.core.motion import (
    clear_legacy_grasp_joints,
    get_active_joint_positions,
    get_left_joint_positions,
    ensure_robot_control_ready,
    get_rmp_ee_position,
    move_ee_to,
    move_ee_smooth,
    plan_ee_waypoints,
    plan_collision_free_keyposes,
    move_ee_collision_aware_approach,
    hold_ee_target,
    get_gripper_joint_efforts,
    close_gripper_slowly,
    open_gripper_slowly,
    move_robot_home,
    set_planning_basket_obstacle_enabled,
    get_finger_collision_world_corners,
    get_gripper_finger_midpoint,
    get_gripper_collision_center,
    get_gripper_collision_diagnostics,
    get_object_gripper_containment,
    get_gripper_table_clearance,
    get_gripper_table_contact_safe_clearance,
    get_gripper_closing_axis, get_gripper_closing_axis_3d,
    get_gripper_inner_opening_width,
    get_gripper_center_local_offset,
    get_tcp_target_for_gripper_center,
    verify_gripper_center_local_offset,
    _execute_joint_trajectory,
)
from aura_isaac_bridge.robot.motion_planner import (
    compute_payload_tracking_state,
    container_place_candidates,
)
from aura_isaac_bridge.core.telemetry import publish_transport_tracking




def resolve_place_position(target_name, object_position):
    normalized_target = str(target_name).strip().lower()
    directional_sign = {
        "right": 1.0,
        "right_zone": 1.0,
        "右边": 1.0,
        "右侧": 1.0,
        "left": -1.0,
        "left_zone": -1.0,
        "左边": -1.0,
        "左侧": -1.0,
    }.get(normalized_target)
    if directional_sign is not None:
        _, camera_orientation = SingleXFormPrim(
            name="aura_direction_camera",
            prim_path=CAMERA_PRIM_PATH,
        ).get_world_pose()
        image_right = quat_rotate(camera_orientation, np.array([1.0, 0.0, 0.0]))
        image_right[2] = 0.0
        direction_norm = np.linalg.norm(image_right)
        if direction_norm < 1e-6:
            image_right = np.array([0.0, 1.0, 0.0])
        else:
            image_right /= direction_norm
        place_position = np.asarray(object_position, dtype=float).copy()
        place_position += image_right * directional_sign * DIRECTIONAL_PLACE_DISTANCE
        place_position[2] += 0.05
        return place_position

    target_prim_path = resolve_scene_prim_path(target_name)
    target_center, target_bbox_min, target_bbox_max = get_bbox_center(
        get_current_stage(),
        target_prim_path,
    )
    place_position = np.asarray(target_center, dtype=float)
    place_position[2] = max(
        place_position[2],
        float(target_bbox_min[2]) + 0.05,
    )
    if state.SCENE_NAME_RESOLVER.canonicalize(target_name) == "basket":
        # The previous fixed 50 mm lift left the can's lower mesh bounds
        # grazing the basket floor.  A dynamic can then receives a lateral
        # collision impulse and rolls out after the initial containment check.
        # Put its mesh center above the floor by half its actual height plus a
        # small settling margin, while keeping the basket center in XY.
        object_path = getattr(state, "TARGET_OBJECT_PRIM_PATH", None)
        if object_path:
            try:
                _, object_bbox_min, object_bbox_max = get_bbox_center(
                    get_current_stage(), object_path
                )
                object_half_height = 0.5 * float(
                    object_bbox_max[2] - object_bbox_min[2]
                )
                place_position[2] = max(
                    place_position[2],
                    float(target_bbox_min[2])
                    + object_half_height
                    + 0.015,
                )
                # Use the target container's measured floor for every payload.
                # Object-specific drop heights create different collision
                # impulses and make the same controller behave inconsistently.
                basket_floor_z = float(target_bbox_min[2])
                place_position[2] = max(
                    place_position[2],
                    basket_floor_z + object_half_height + 0.015,
                )
            except Exception as exc:
                print(f"⚠️ 无法计算篮筐物体高度余量，使用默认放置高度: {exc}")
    print(
        f"📦 放置目标 {target_name}: path={target_prim_path}, "
        f"bbox=({target_bbox_min}, {target_bbox_max})"
    )
    return place_position


def get_human_tabletop_approach_direction(object_position):
    """Approach a tabletop object from the base side with a downward tool."""
    if DACH_BASE_XY is None:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    inward = np.asarray(object_position, dtype=float)[:2] - np.asarray(
        DACH_BASE_XY,
        dtype=float,
    )
    inward_norm = float(np.linalg.norm(inward))
    if inward_norm <= 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    inward /= inward_norm
    tilt = min(TARGET_GRASP_APPROACH_TILT_RAD, MAX_GRASP_APPROACH_TILT_RAD)
    direction = np.array(
        [inward[0] * math.sin(tilt), inward[1] * math.sin(tilt), -math.cos(tilt)],
        dtype=float,
    )
    return direction / np.linalg.norm(direction)


def get_top_down_grasp_orientation(object_name, target_prim, tilt_override=None):
    canonical_name = state.SCENE_NAME_RESOLVER.canonicalize(object_name)
    maximum_tilt = MAX_GRASP_APPROACH_TILT_RAD
    requested_tilt = (
        TARGET_GRASP_APPROACH_TILT_RAD
        if tilt_override is None
        else float(tilt_override)
    )
    tilt = float(np.clip(requested_tilt, 0.0, maximum_tilt))
    if abs(tilt - requested_tilt) > 1e-9:
        print(
            f"🛡️ {canonical_name} 抓取倾角限制为 "
            f"{np.degrees(maximum_tilt):.1f}°，"
            f"requested={np.degrees(requested_tilt):.1f}°"
        )

    # GraspNet supplies the grasp center. The physical jaw axis is hard-bound
    # to the narrowest live mesh projection for every object class.
    object_short_axis = -get_current_mesh_horizontal_min_width_axis(
        get_current_stage(), target_prim, target_prim.prim_path
    )
    object_long_axis = np.array(
        [-object_short_axis[1], object_short_axis[0]], dtype=float
    )
    inward_axis = None
    if DACH_BASE_XY is not None:
        object_position, _ = target_prim.get_world_pose()
        inward_axis = np.asarray(object_position, dtype=float)[:2] - np.asarray(
            DACH_BASE_XY,
            dtype=float,
        )
        inward_norm = float(np.linalg.norm(inward_axis))
        if inward_norm > 1e-6:
            inward_axis /= inward_norm
        else:
            inward_axis = None
    # Keep the approach direction in the object's long-axis plane. This makes
    # the same orientation construction valid for curved, round, and box-like
    # payloads while preserving the narrowest closing axis.
    tilt_axis = np.asarray(object_long_axis, dtype=float).copy()
    if inward_axis is not None and np.dot(tilt_axis, inward_axis) < 0.0:
        tilt_axis *= -1.0
    approach_axis = np.array(
        [
            tilt_axis[0] * np.sin(tilt),
            tilt_axis[1] * np.sin(tilt),
            -np.cos(tilt),
        ],
        dtype=float,
    )
    closing_axis = np.array(
        [-object_short_axis[0], -object_short_axis[1], 0.0],
        dtype=float,
    )
    print(
        f"🧭 {canonical_name} 抓取轴: long={object_long_axis}, "
        f"short={object_short_axis}; 夹爪沿最窄方向闭合, "
        f"inward_tilt_axis={tilt_axis}"
    )
    closing_axis -= np.dot(closing_axis, approach_axis) * approach_axis
    closing_axis /= np.linalg.norm(closing_axis)
    approach_axis -= np.dot(approach_axis, closing_axis) * closing_axis
    approach_axis /= np.linalg.norm(approach_axis)
    lateral_axis = np.cross(approach_axis, closing_axis)
    rotation = np.column_stack((approach_axis, closing_axis, lateral_axis))
    if tilt > 0.0:
        print(
            f"🧭 {canonical_name} 向内倾斜抓取: {np.degrees(tilt):.1f}°, "
            f"approach={approach_axis}"
        )
    return rot_matrix_to_quat(rotation)


def rotate_grasp_about_approach_axis(orientation, angle_rad):
    """Rotate tool roll while preserving its TCP approach axis."""
    rotation = quat_to_rot_matrix(np.asarray(orientation, dtype=float)).copy()
    closing = rotation[:, 1].copy()
    lateral = rotation[:, 2].copy()
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    rotation[:, 1] = cosine * closing + sine * lateral
    rotation[:, 2] = -sine * closing + cosine * lateral
    return rot_matrix_to_quat(rotation)


def flip_grasp_about_approach_axis(orientation):
    """Return the equivalent opposite tool branch for the same jaw line."""
    return rotate_grasp_about_approach_axis(orientation, math.pi)


def interpolate_quaternions(start, end, sample_count=12):
    start = np.asarray(start, dtype=float)
    start /= np.linalg.norm(start)
    end = np.asarray(end, dtype=float)
    end /= np.linalg.norm(end)
    if np.dot(start, end) < 0.0:
        end = -end
    values = []
    for alpha in np.linspace(1.0 / sample_count, 1.0, sample_count):
        value = (1.0 - alpha) * start + alpha * end
        value /= np.linalg.norm(value)
        values.append(value)
    return values


def align_physical_closing_axis_at_hover(
    desired_axis,
    gripper_positions,
    minimum_alignment,
    *,
    max_attempts=3,
):
    """Close the TCP-to-physical-jaw calibration loop above the table."""
    desired = np.asarray(desired_axis, dtype=float)[:2]
    desired /= max(float(np.linalg.norm(desired)), 1e-9)
    diagnostics = []
    for attempt_index in range(max_attempts):
        physical = np.asarray(get_gripper_closing_axis(), dtype=float)[:2]
        physical /= max(float(np.linalg.norm(physical)), 1e-9)
        if np.dot(physical, desired) < 0.0:
            desired = -desired
        alignment = float(np.clip(np.dot(physical, desired), -1.0, 1.0))
        if alignment >= minimum_alignment:
            return {
                "success": True,
                "alignment": alignment,
                "attempts": diagnostics,
            }

        signed_error = math.atan2(
            physical[0] * desired[1] - physical[1] * desired[0],
            np.dot(physical, desired),
        )
        tcp_position, tcp_rotation = state.controller.get_end_effector_pose()
        current_orientation = rot_matrix_to_quat(tcp_rotation)
        tcp_axis = np.asarray(tcp_rotation, dtype=float)[:2, 1]
        tcp_axis /= max(float(np.linalg.norm(tcp_axis)), 1e-9)

        best = None
        for roll_delta in (signed_error, -signed_error):
            candidate = rotate_grasp_about_approach_axis(
                current_orientation, roll_delta
            )
            candidate_axis = quat_to_rot_matrix(candidate)[:2, 1]
            candidate_axis /= max(float(np.linalg.norm(candidate_axis)), 1e-9)
            tcp_shift = math.atan2(
                tcp_axis[0] * candidate_axis[1]
                - tcp_axis[1] * candidate_axis[0],
                np.dot(tcp_axis, candidate_axis),
            )
            predicted_angle = signed_error - tcp_shift
            score = abs(math.atan2(math.sin(predicted_angle), math.cos(predicted_angle)))
            if best is None or score < best[0]:
                best = (score, roll_delta, candidate)
        _, roll_delta, target_orientation = best
        orientation_path = interpolate_quaternions(
            current_orientation, target_orientation, sample_count=16
        )
        joint_targets = state.controller.plan_pose_waypoints_with_orientations(
            [np.asarray(tcp_position, dtype=float)] * len(orientation_path),
            orientation_path,
            warm_start=get_active_joint_positions(),
        )
        diagnostic = {
            "attempt": attempt_index + 1,
            "alignment_before": alignment,
            "signed_error_deg": math.degrees(signed_error),
            "roll_delta_deg": math.degrees(roll_delta),
            "ik_reachable": joint_targets is not None,
        }
        diagnostics.append(diagnostic)
        if joint_targets is None:
            break
        _execute_joint_trajectory(
            f"physical_jaw_axis_refine_{attempt_index + 1}",
            joint_targets,
            gripper_positions=gripper_positions,
            monitor_table_clearance=True,
            max_joint_step_rad=0.01,
            minimum_frames=24,
        )
    physical = np.asarray(get_gripper_closing_axis(), dtype=float)[:2]
    physical /= max(float(np.linalg.norm(physical)), 1e-9)
    final_alignment = float(abs(np.dot(physical, desired)))
    return {
        "success": final_alignment >= minimum_alignment,
        "alignment": final_alignment,
        "attempts": diagnostics,
    }


def get_gripper_close_target(object_name):
    configured_target = getattr(
        getattr(state.dach_arm, "gripper", None),
        "joint_closed_positions",
        None,
    )
    if configured_target is not None:
        return np.asarray(configured_target, dtype=float).copy()
    # Keep a deterministic fallback for unit-test and pre-initialization paths.
    return np.zeros(2, dtype=float)


def get_gripper_open_target(object_name):
    return state.GRIPPER_OPEN_POSITIONS.copy()


def get_gripper_close_frames(object_name):
    return GRIPPER_CLOSE_FRAMES


def adjust_object_grasp_position(
    object_name,
    object_position,
    bbox_min,
    bbox_max,
):
    adjusted_position = np.asarray(object_position, dtype=float).copy()
    bbox_center = (
        np.asarray(bbox_min, dtype=float) + np.asarray(bbox_max, dtype=float)
    ) * 0.5
    source_xy = adjusted_position[:2].copy()
    object_prim_path = state.SCENE_NAME_RESOLVER.prim_candidates(object_name)[0]
    object_short_axis = get_mesh_horizontal_min_width_axis(
        get_current_stage(), object_prim_path
    )
    object_short_axis = np.asarray(object_short_axis, dtype=float)
    object_short_axis /= max(float(np.linalg.norm(object_short_axis)), 1e-9)
    object_long_axis = np.array(
        [-object_short_axis[1], object_short_axis[0]], dtype=float
    )
    mesh_center = get_mesh_center(get_current_stage(), object_prim_path)
    long_axis_offset = float(
        np.dot(source_xy - mesh_center[:2], object_long_axis)
    )
    mesh_center_long = float(
        np.dot(mesh_center[:2], object_long_axis)
    )
    source_long = mesh_center_long + float(
        np.clip(long_axis_offset, -0.02, 0.02)
    )
    collision_short_center = float(
        np.dot(mesh_center[:2], object_short_axis)
    )
    adjusted_position[:2] = (
        object_long_axis * source_long
        + object_short_axis * collision_short_center
    )
    xy_margin = np.minimum(
        np.maximum(
            (np.asarray(bbox_max[:2]) - np.asarray(bbox_min[:2])) * 0.08,
            0.003,
        ),
        0.012,
    )
    adjusted_position[:2] = np.clip(
        adjusted_position[:2],
        np.asarray(bbox_min[:2]) + xy_margin,
        np.asarray(bbox_max[:2]) - xy_margin,
    )
    print(
        f"🎯 {object_name} 感知/场景抓取点: "
        f"mesh_center_xy={mesh_center[:2]}, "
        f"bbox_center_xy={bbox_center[:2]}, "
        f"source_xy={source_xy}, "
        f"target_xy={adjusted_position[:2]}, "
        f"long_axis_offset={long_axis_offset:.4f} m, "
        f"short_axis={object_short_axis}"
    )
    return adjusted_position


def _activate_arm(side):
    """Switch the legacy active-arm aliases used by motion.py."""
    normalized = str(side).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"invalid DACH arm side: {side!r}")
    controller = (getattr(state, "arm_controllers", None) or {}).get(normalized)
    arm = (getattr(state, "arm_views", None) or {}).get(normalized)
    fingers = (getattr(state, "arm_fingers", None) or {}).get(normalized)
    if controller is None or arm is None or fingers is None:
        raise RuntimeError(
            f"DACH {normalized} arm is not initialized; reload the AuraVLA runtime"
        )
    state.active_arm_side = normalized
    state.dach_arm = arm
    state.controller = controller
    state.left_finger = fingers["left"]
    state.right_finger = fingers["right"]
    state.right_gripper = fingers["tcp"]
    print(f"🦾 自动选择 DACH {normalized} 臂执行本次任务")


def _preferred_arm_side_for_position(position):
    """Choose the arm on the geometric side of the target.

    The configured arm is only a fallback/tie-breaker.  Preferring it whenever
    both IK gates pass makes a right-side banana get routed through the left
    arm after automatic selection was introduced, which is a path-planning
    error rather than an IK failure.
    """
    if DACH_BASE_XY is None:
        return DACH_ARM_SIDE
    x_delta = float(np.asarray(position, dtype=float)[0]) - float(DACH_BASE_XY[0])
    # Keep the configured side for targets essentially on the center line to
    # avoid branch flapping from perception noise.
    if abs(x_delta) < 0.05:
        return DACH_ARM_SIDE
    return "right" if x_delta > 0.0 else "left"


def evaluate_dual_arm_hover_reachability(object_name, target_prim, bbox_min, bbox_max):
    object_position, _, _ = get_sim_pose(target_prim)
    minimum_grasp_height = bbox_min[2] + (
        bbox_max[2] - bbox_min[2]
    ) * GRASP_MIN_HEIGHT_FRACTION
    object_position[2] = max(object_position[2], minimum_grasp_height)
    object_position = adjust_object_grasp_position(
        object_name,
        object_position,
        bbox_min,
        bbox_max,
    )
    object_position[2] += DACH_GRASP_HEIGHT_OFFSET
    orientation = get_top_down_grasp_orientation(object_name, target_prim)
    approach_direction = quat_rotate(
        orientation,
        np.array([1.0, 0.0, 0.0]),
    )
    approach_direction /= np.linalg.norm(approach_direction)
    grasp_position = get_tcp_target_for_gripper_center(
        object_position,
        orientation,
    )

    candidates = []
    reachable_by_side = {}
    hover_clearances = (0.08, 0.12, 0.16, 0.20, 0.24)

    # This is a fast kinematic gate only. Collision-aware RRT is run once later,
    # immediately before commanding the arm.
    gate_orientation = orientation
    for side in ("right", "left"):
        controller = (getattr(state, "arm_controllers", None) or {}).get(side)
        arm = (getattr(state, "arm_views", None) or {}).get(side)
        if controller is None or arm is None:
            reachable_by_side[side] = False
            continue
        start_joints = arm.get_joint_positions()
        side_reachable = False
        side_candidates = []
        for clearance in hover_clearances:
            hover_position = grasp_position - approach_direction * clearance
            side_ok = controller.plan_pose_waypoints(
                [hover_position, grasp_position],
                target_orientation=gate_orientation,
                warm_start=start_joints,
            ) is not None
            side_reachable = side_reachable or side_ok
            side_candidates.append(
                {
                    "side": side,
                    "clearance_m": clearance,
                    "hover_position": hover_position.tolist(),
                    "grasp_position": grasp_position.tolist(),
                    "reachable": side_ok,
                }
            )
            if side_ok:
                break
        reachable_by_side[side] = side_reachable
        candidates.extend(side_candidates)

    preferred_side = _preferred_arm_side_for_position(object_position)
    if not any(reachable_by_side.values()):
        selected_side = None
    elif reachable_by_side.get(preferred_side, False):
        selected_side = preferred_side
    else:
        selected_side = "left" if reachable_by_side.get("left") else "right"
    return {
        "base_xy": list(DACH_BASE_XY) if DACH_BASE_XY is not None else None,
        "object_position": object_position.tolist(),
        "right_reachable": bool(reachable_by_side.get("right", False)),
        "left_reachable": bool(reachable_by_side.get("left", False)),
        "selected_arm": selected_side,
        "preferred_arm": preferred_side,
        "candidates": candidates,
        "planner": "fast_lula_ik_gate",
    }


def _evaluate_arm_pose_reachability(object_name, grasp_position, orientation):
    """Check the exact post-perception pose that will be sent to the arm."""
    canonical_name = state.SCENE_NAME_RESOLVER.canonicalize(object_name)
    approach_direction = quat_rotate(
        orientation,
        np.array([1.0, 0.0, 0.0]),
    )
    approach_direction = np.asarray(approach_direction, dtype=float)
    approach_direction /= max(float(np.linalg.norm(approach_direction)), 1e-9)
    gate_orientation = orientation
    by_side = {}
    strict_by_side = {}
    hover_by_side = {}
    hover_plan_by_side = {}
    strict_hover_by_side = {}
    strict_hover_plan_by_side = {}
    attempts = []
    for side in ("right", "left"):
        controller = (getattr(state, "arm_controllers", None) or {}).get(side)
        arm = (getattr(state, "arm_views", None) or {}).get(side)
        side_ok = False
        strict_reachable_for_side = False
        fallback_hover = None
        fallback_plan = None
        if controller is not None and arm is not None:
            start = arm.get_joint_positions()
            for clearance in (0.08, 0.12, 0.16, 0.20, 0.24):
                hover = np.asarray(grasp_position, dtype=float) - approach_direction * clearance
                lift = np.asarray(grasp_position, dtype=float) + np.array(
                    [0.0, 0.0, TRANSPORT_LIFT_HEIGHT], dtype=float
                )
                joint_targets = controller.plan_pose_waypoints(
                    [hover, grasp_position, lift],
                    target_orientation=gate_orientation,
                    warm_start=start,
                    allow_orientation_fallback=False,
                )
                strict_diagnostics = dict(
                    getattr(controller, "last_pose_waypoint_diagnostics", {})
                )
                strict_reachable = joint_targets is not None
                grasp_only_reachable = None
                lift_only_reachable = None
                position_only_reachable = None
                position_only_targets = None
                if joint_targets is None:
                    grasp_only_reachable = (
                        controller.plan_pose_waypoints(
                            [grasp_position],
                            target_orientation=gate_orientation,
                            warm_start=start,
                            allow_orientation_fallback=False,
                        )
                        is not None
                    )
                    lift_only_reachable = (
                        controller.plan_pose_waypoints(
                            [lift],
                            target_orientation=gate_orientation,
                            warm_start=start,
                            allow_orientation_fallback=False,
                        )
                        is not None
                    )
                    position_only_targets = controller.plan_pose_waypoints(
                        [hover],
                        target_orientation=None,
                        warm_start=start,
                    )
                    position_only_reachable = position_only_targets is not None
                reachable_for_entry = (
                    joint_targets is not None or position_only_reachable is True
                )
                attempts.append(
                    {
                        "side": side,
                        "clearance_m": clearance,
                        "reachable": reachable_for_entry,
                        "orientation_constrained": strict_reachable,
                        "grasp_only_reachable": grasp_only_reachable,
                        "lift_only_reachable": lift_only_reachable,
                        "position_only_reachable": position_only_reachable,
                        "ik_diagnostics": strict_diagnostics,
                    }
                )
                if strict_reachable:
                    strict_reachable_for_side = True
                    side_ok = True
                    hover_by_side[side] = hover.copy()
                    hover_plan_by_side[side] = [
                        np.asarray(joint_targets[0], dtype=float).copy()
                    ]
                    strict_hover_by_side[side] = hover.copy()
                    strict_hover_plan_by_side[side] = [
                        np.asarray(joint_targets[0], dtype=float).copy()
                    ]
                    break
                if position_only_reachable is True and fallback_hover is None:
                    fallback_hover = hover.copy()
                    fallback_plan = [
                        np.asarray(position_only_targets[0], dtype=float).copy()
                    ]
        if not strict_reachable_for_side and fallback_hover is not None:
            side_ok = True
            hover_by_side[side] = fallback_hover
            hover_plan_by_side[side] = fallback_plan
        by_side[side] = side_ok
        strict_by_side[side] = strict_reachable_for_side
    preferred_side = _preferred_arm_side_for_position(grasp_position)
    selected = None
    if by_side.get(preferred_side, False):
        selected = preferred_side
    elif by_side.get("right", False):
        selected = "right"
    elif by_side.get("left", False):
        selected = "left"
    strict_selected = None
    if strict_by_side.get(preferred_side, False):
        strict_selected = preferred_side
    elif strict_by_side.get("right", False):
        strict_selected = "right"
    elif strict_by_side.get("left", False):
        strict_selected = "left"
    return {
        "right": by_side.get("right", False),
        "left": by_side.get("left", False),
        "selected": selected,
        "strict_right": strict_by_side.get("right", False),
        "strict_left": strict_by_side.get("left", False),
        "strict_selected": strict_selected,
        "preferred": preferred_side,
        "hover_position": None if selected is None else hover_by_side[selected],
        "hover_plan": None if selected is None else hover_plan_by_side[selected],
        "strict_hover_position": (
            None if strict_selected is None else strict_hover_by_side[strict_selected]
        ),
        "strict_hover_plan": (
            None if strict_selected is None else strict_hover_plan_by_side[strict_selected]
        ),
        "attempts": attempts,
    }


def _evaluate_complete_task_pose_chain(
    object_name,
    target_name,
    object_position,
    bbox_min,
    bbox_max,
    orientation,
    grasp_reachability,
    nominal_goal_position,
):
    """Fast IK gate for one fixed grasp orientation over the complete task."""
    grasp_tcp = get_tcp_target_for_gripper_center(object_position, orientation)
    approach_direction = quat_rotate(orientation, np.array([1.0, 0.0, 0.0]))
    approach_direction /= max(float(np.linalg.norm(approach_direction)), 1e-9)
    hover_tcp = grasp_tcp - approach_direction * 0.12
    lift_tcp = grasp_tcp + np.array([0.0, 0.0, TRANSPORT_LIFT_HEIGHT])

    goal_candidates = [np.asarray(nominal_goal_position, dtype=float).copy()]
    minimum_safe_hover_z = None
    if state.SCENE_NAME_RESOLVER.canonicalize(target_name) == "basket":
        target_path = resolve_scene_prim_path(target_name)
        _, target_min, target_max = get_bbox_center(
            get_current_stage(), target_path
        )
        payload_drop_below_tcp = max(
            float(grasp_tcp[2] - np.asarray(bbox_min, dtype=float)[2]),
            0.0,
        )
        minimum_safe_hover_z = (
            float(target_max[2])
            + payload_drop_below_tcp
            + max(BASKET_PLANNING_MARGIN, 0.02)
        )
        payload_half_extents = 0.5 * (
            np.asarray(bbox_max, dtype=float)[:2]
            - np.asarray(bbox_min, dtype=float)[:2]
        )
        candidate_xy_values = container_place_candidates(
            target_min,
            target_max,
            payload_half_extents,
            DACH_BASE_XY,
            wall_margin_m=max(0.01, 0.5 * BASKET_PLANNING_MARGIN),
        )
        goal_candidates = []
        for candidate_xy in candidate_xy_values:
            candidate_goal = np.asarray(nominal_goal_position, dtype=float).copy()
            candidate_goal[:2] = candidate_xy
            goal_candidates.append(candidate_goal)

    selected_side = grasp_reachability.get("selected")
    candidate_sides = []
    for side in (selected_side, "right", "left"):
        if (
            side in {"right", "left"}
            and side not in candidate_sides
            and grasp_reachability.get(side, False)
        ):
            candidate_sides.append(side)
    if not candidate_sides:
        return {"reachable": False, "reason": "no grasp-reachable arm is available"}

    attempts = []
    # Keep the same bounded transport search for every payload.  The object
    # geometry determines the grasp axes; object names must not select a
    # different motion policy.
    yaw_offsets = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90]
    for candidate_side in candidate_sides:
        controller = (getattr(state, "arm_controllers", None) or {}).get(
            candidate_side
        )
        arm = (getattr(state, "arm_views", None) or {}).get(candidate_side)
        if controller is None or arm is None:
            attempts.append(
                {
                    "arm": candidate_side,
                    "grasp_chain_reachable": False,
                    "reason": "arm controller is unavailable",
                }
            )
            continue
        grasp_chain = controller.plan_pose_waypoints(
            [hover_tcp, grasp_tcp, lift_tcp],
            target_orientation=orientation,
            warm_start=arm.get_joint_positions(),
            allow_orientation_fallback=False,
        )
        if grasp_chain is None:
            attempts.append(
                {
                    "arm": candidate_side,
                    "grasp_chain_reachable": False,
                    "reason": "grasp chain IK failed",
                }
            )
            continue
        for yaw_offset_deg in yaw_offsets:
            transport_orientation = rotate_grasp_about_approach_axis(
                orientation, math.radians(yaw_offset_deg)
            )
            orientation_path = interpolate_quaternions(
                orientation, transport_orientation
            )
            reorientation_chain = controller.plan_pose_waypoints_with_orientations(
                [lift_tcp] * len(orientation_path),
                orientation_path,
                warm_start=grasp_chain[-1],
            )
            if reorientation_chain is None:
                attempts.append(
                    {
                        "arm": candidate_side,
                        "grasp_chain_reachable": True,
                        "transport_yaw_offset_deg": yaw_offset_deg,
                        "reorientation_reachable": False,
                    }
                )
                continue
            for candidate_goal in goal_candidates:
                place_tcp = get_tcp_target_for_gripper_center(
                    candidate_goal, transport_orientation
                )
                hover_clearances = (0.22, 0.20, 0.18, 0.16, 0.14, 0.12)
                attempted_hover_z = []
                for hover_clearance in hover_clearances:
                    place_hover = place_tcp + np.array(
                        [0.0, 0.0, hover_clearance]
                    )
                    place_hover[2] = max(
                        float(place_hover[2]),
                        float(lift_tcp[2]),
                        float(minimum_safe_hover_z or -np.inf),
                    )
                    if any(
                        abs(float(place_hover[2]) - previous_z) < 1e-6
                        for previous_z in attempted_hover_z
                    ):
                        continue
                    attempted_hover_z.append(float(place_hover[2]))
                    # The execution path uses a collision-aware sparse RRT for
                    # the transport corridor. The endpoint gate preserves the
                    # fixed wrist pose while allowing the lowest payload-safe
                    # basket entry height near the edge of the arm workspace.
                    hover_chain = controller.plan_pose_waypoints(
                        [place_hover],
                        target_orientation=transport_orientation,
                        warm_start=reorientation_chain[-1],
                        allow_orientation_fallback=False,
                    )
                    place_chain = None
                    if hover_chain is not None:
                        place_chain = controller.plan_pose_waypoints(
                            [place_tcp],
                            target_orientation=transport_orientation,
                            warm_start=hover_chain[-1],
                            allow_orientation_fallback=False,
                        )
                    attempts.append(
                        {
                            "arm": candidate_side,
                            "grasp_chain_reachable": True,
                            "transport_yaw_offset_deg": yaw_offset_deg,
                            "reorientation_reachable": True,
                            "goal_position": candidate_goal.tolist(),
                            "place_tcp": place_tcp.tolist(),
                            "place_hover": place_hover.tolist(),
                            "place_hover_clearance_m": float(
                                place_hover[2] - place_tcp[2]
                            ),
                            "minimum_safe_hover_z_m": minimum_safe_hover_z,
                            "transport_gate": "payload_safe_sparse_endpoint_ik",
                            "hover_reachable": hover_chain is not None,
                            "reachable": place_chain is not None,
                            "ik_diagnostics": dict(
                                getattr(controller, "last_pose_waypoint_diagnostics", {})
                            ),
                        }
                    )
                    if place_chain is not None:
                        return {
                            "reachable": True,
                            "selected_arm": candidate_side,
                            "selected_goal_position": candidate_goal.tolist(),
                            "place_hover_clearance_m": float(
                                place_hover[2] - place_tcp[2]
                            ),
                            "transport_yaw_offset_deg": yaw_offset_deg,
                            "transport_orientation": transport_orientation.tolist(),
                            "attempts": attempts,
                        }
    return {
        "reachable": False,
        "reason": "no fixed-orientation placement candidate is reachable",
        "attempts": attempts,
    }


def _select_geometry_constrained_grasp_candidate(
    object_name,
    target_name,
    target_prim,
    object_prim_path,
    object_position,
    bbox_min,
    bbox_max,
    nominal_goal_position,
    initial_orientation,
    reachability,
    gripper_opening_width,
    aperture_margin,
):
    """Select one complete-task grasp pose using the same policy for all objects."""
    stage = get_current_stage()
    desired_short_axis = -get_current_mesh_horizontal_min_width_axis(
        stage, target_prim, object_prim_path
    )
    desired_short_axis = np.asarray(desired_short_axis, dtype=float)[:2]
    desired_short_axis /= max(float(np.linalg.norm(desired_short_axis)), 1e-9)
    desired_long_axis = np.array(
        [-desired_short_axis[1], desired_short_axis[0]],
        dtype=float,
    )
    _, _, short_width = get_mesh_extent_along_axis(
        stage,
        object_prim_path,
        np.array([desired_short_axis[0], desired_short_axis[1], 0.0]),
    )
    _, _, long_width = get_mesh_extent_along_axis(
        stage,
        object_prim_path,
        np.array([desired_long_axis[0], desired_long_axis[1], 0.0]),
    )
    shape_is_axisymmetric = bool(
        long_width <= max(short_width * 1.05, short_width + 0.002)
    )

    def closing_alignment(orientation):
        axis = np.asarray(quat_to_rot_matrix(orientation)[:2, 1], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return float(abs(np.dot(axis, desired_short_axis)))

    # For elongated meshes, preserve the short-axis requirement. For nearly
    # axisymmetric meshes, the PCA direction is arbitrary, so the same
    # geometric policy must allow an equivalent wrist roll instead of
    # rejecting a valid grasp because of an unstable principal-axis sign.
    minimum_alignment = (
        0.0 if shape_is_axisymmetric else float(GRASP_MIN_SHORT_AXIS_ALIGNMENT)
    )
    configured_tilt_deg = float(np.degrees(TARGET_GRASP_APPROACH_TILT_RAD))
    tilt_candidates = []
    for tilt_deg in (
        0.0,
        configured_tilt_deg,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
    ):
        tilt_deg = float(tilt_deg)
        if tilt_deg > np.degrees(MAX_GRASP_APPROACH_TILT_RAD) + 1e-6:
            continue
        if not any(abs(tilt_deg - existing) < 1e-6 for existing in tilt_candidates):
            tilt_candidates.append(tilt_deg)

    # Prefer a level approach, then the configured inward tilt, while still
    # evaluating every candidate so selection is based on the full task gate.
    tilt_candidates.sort(key=lambda value: (value != 0.0, abs(value - configured_tilt_deg)))
    attempts = []
    best = None
    for tilt_deg in tilt_candidates:
        base_orientation = get_top_down_grasp_orientation(
            object_name,
            target_prim,
            tilt_override=np.radians(tilt_deg),
        )
        for roll_deg in GRASP_APPROACH_ROLL_CANDIDATES_DEG:
            rolled_orientation = rotate_grasp_about_approach_axis(
                base_orientation,
                np.radians(float(roll_deg)),
            )
            for opposite_tool_branch in (False, True):
                candidate_orientation = (
                    flip_grasp_about_approach_axis(rolled_orientation)
                    if opposite_tool_branch
                    else rolled_orientation
                )
                candidate_alignment = closing_alignment(candidate_orientation)
                attempt = {
                    "tilt_deg": tilt_deg,
                    "roll_deg": float(roll_deg),
                    "opposite_tool_branch": opposite_tool_branch,
                    "short_axis_alignment": candidate_alignment,
                }
                if candidate_alignment < minimum_alignment:
                    attempt["reason"] = "short-axis alignment below threshold"
                    attempts.append(attempt)
                    continue

                candidate_axis = np.asarray(
                    quat_to_rot_matrix(candidate_orientation)[:, 1],
                    dtype=float,
                )
                _, _, candidate_width = get_mesh_extent_along_axis(
                    stage,
                    object_prim_path,
                    candidate_axis,
                )
                attempt["object_width_on_closing_axis_m"] = float(candidate_width)
                if candidate_width + aperture_margin > gripper_opening_width:
                    attempt["reason"] = "closing-axis width exceeds live opening"
                    attempts.append(attempt)
                    continue

                candidate_tcp = get_tcp_target_for_gripper_center(
                    object_position,
                    candidate_orientation,
                )
                candidate_reachability = _evaluate_arm_pose_reachability(
                    object_name,
                    candidate_tcp,
                    candidate_orientation,
                )
                strict_selected = candidate_reachability.get("strict_selected")
                if strict_selected is None:
                    attempt["grasp_reachability"] = {
                        "selected": None,
                        "right": bool(candidate_reachability.get("strict_right")),
                        "left": bool(candidate_reachability.get("strict_left")),
                    }
                    attempt["reason"] = "strict grasp/lift IK unavailable"
                    attempts.append(attempt)
                    continue

                strict_reachability = dict(candidate_reachability)
                strict_reachability.update(
                    {
                        "selected": strict_selected,
                        "right": bool(candidate_reachability.get("strict_right")),
                        "left": bool(candidate_reachability.get("strict_left")),
                        "hover_position": candidate_reachability.get(
                            "strict_hover_position"
                        ),
                        "hover_plan": candidate_reachability.get(
                            "strict_hover_plan"
                        ),
                    }
                )
                task_gate = _evaluate_complete_task_pose_chain(
                    object_name,
                    target_name,
                    object_position,
                    bbox_min,
                    bbox_max,
                    candidate_orientation,
                    strict_reachability,
                    nominal_goal_position,
                )
                attempt["grasp_reachability"] = {
                    "selected": strict_selected,
                    "right": bool(candidate_reachability.get("strict_right")),
                    "left": bool(candidate_reachability.get("strict_left")),
                }
                attempt["complete_task_gate"] = task_gate
                attempts.append(attempt)
                if not task_gate.get("reachable"):
                    continue

                score = (
                    candidate_alignment,
                    -float(candidate_width),
                    -float(tilt_deg),
                    -abs(float(roll_deg)),
                    -int(opposite_tool_branch),
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "tilt_deg": tilt_deg,
                        "roll_deg": float(roll_deg),
                        "opposite_tool_branch": opposite_tool_branch,
                        "orientation": candidate_orientation,
                        "tcp": candidate_tcp,
                        "reachability": strict_reachability,
                        "alignment": candidate_alignment,
                        "task_gate": task_gate,
                    }

    if best is None:
        return {
            "selected": False,
            "planned_alignment": closing_alignment(initial_orientation),
            "attempts": attempts,
            "minimum_alignment": minimum_alignment,
        }
    task_gate = best["task_gate"]
    return {
        "selected": True,
        "tilt_deg": best["tilt_deg"],
        "roll_deg": best["roll_deg"],
        "opposite_tool_branch": best["opposite_tool_branch"],
        "orientation": best["orientation"],
        "tcp": best["tcp"],
        "reachability": best["reachability"],
        "planned_alignment": best["alignment"],
        "task_gate": task_gate,
        "attempts": attempts,
        "minimum_alignment": minimum_alignment,
    }


def execute_pick_place(object_name, target_name, _reacquire_attempt=0):
    state._task_motion_started = False
    task_started = time.perf_counter()

    canonical_object_name = state.SCENE_NAME_RESOLVER.canonicalize(object_name)
    maximum_grasp_tilt = MAX_GRASP_APPROACH_TILT_RAD
    object_prim_path = resolve_scene_prim_path(object_name)
    state.TARGET_OBJECT_PRIM_PATH = object_prim_path
    target_prim = SingleXFormPrim(
        name=f"aura_target_{str(object_name).lower()}",
        prim_path=object_prim_path,
    )
    if not target_prim.is_valid():
        raise RuntimeError(f"抓取目标无效: {object_prim_path}")

    canonical_target_name = state.SCENE_NAME_RESOLVER.canonicalize(target_name)
    clear_legacy_grasp_joints()
    if not set_planning_basket_obstacle_enabled(True):
        return {
            "success": False,
            "message": "failed to enable basket collision obstacle",
            "object_name": str(object_name),
            "target_name": str(target_name),
        }

    if not state.sim_context.is_playing():
        state.sim_context.play()
        step_app(3)
    ensure_robot_control_ready()
    gripper_open_target = get_gripper_open_target(object_name)
    _, precheck_bbox_min, precheck_bbox_max = get_current_bbox_center(
        get_current_stage(),
        target_prim,
        object_prim_path,
    )
    precheck_started = time.perf_counter()
    reachability = evaluate_dual_arm_hover_reachability(
        object_name,
        target_prim,
        precheck_bbox_min,
        precheck_bbox_max,
    )
    print(
        f"⏱️ 可达性预检: "
        f"{time.perf_counter() - precheck_started:.2f} s"
    )
    selected_arm = reachability.get("selected_arm")
    if selected_arm is not None:
        _activate_arm(selected_arm)
    else:
        print("⚠️ 左右臂快速连续 IK 均未找到解；继续交给 Lula RRT 做最终可达性判断")
        _activate_arm(reachability.get("preferred_arm", DACH_ARM_SIDE))
    ensure_pickable_object(get_current_stage(), object_prim_path)
    # A position command is not an instantaneous jaw reset.  The previous
    # 20-frame wait could leave the finite-effort gripper nearly closed, so
    # the live aperture check rejected the next object before motion started.
    open_result = open_gripper_slowly(
        get_rmp_ee_position(),
        frames=18,
        target_open=gripper_open_target,
    )
    if not open_result["converged"]:
        return {
            "success": False,
            "message": "gripper did not reach the open target before task precheck",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "gripper_feedback": open_result["feedback"].tolist(),
            "gripper_target": open_result["target"].tolist(),
            "gripper_open_error_m": open_result["error_m"],
        }

    # Reject geometrically impossible grasps while the arm is still at its
    # safe starting pose. The later live-axis check remains the final guard.
    object_short_axis = -get_current_mesh_horizontal_min_width_axis(
        get_current_stage(), target_prim, object_prim_path
    )
    minimum_closing_axis = np.array(
        [object_short_axis[0], object_short_axis[1], 0.0],
        dtype=float,
    )
    _, _, minimum_object_width = get_mesh_extent_along_axis(
        get_current_stage(),
        object_prim_path,
        minimum_closing_axis,
    )
    gripper_opening_width = float(get_gripper_inner_opening_width())
    aperture_margin = GRASP_APERTURE_MARGIN
    required_opening_width = float(minimum_object_width + aperture_margin)
    print(
        "📏 运动前最小抓取宽度校验: "
        f"jaw_gap={gripper_opening_width:.4f} m, "
        f"minimum_object_width={minimum_object_width:.4f} m, "
        f"required={required_opening_width:.4f} m"
    )
    if required_opening_width > gripper_opening_width:
        message = (
            f"object width {minimum_object_width * 1000:.1f} mm exceeds "
            f"single-arm gripper opening {gripper_opening_width * 1000:.1f} mm; "
            "use a bimanual grasp or a wider tool"
        )
        return {
            "success": False,
            "message": message,
            "object_name": str(object_name),
            "target_name": str(target_name),
            "physical_constraint": {
                "gripper_opening_m": gripper_opening_width,
                "object_width_on_closing_axis_m": float(minimum_object_width),
                "required_opening_m": required_opening_width,
                "safety_margin_m": aperture_margin,
                "checked_before_motion": True,
            },
        }

    top_down_orientation = get_top_down_grasp_orientation(object_name, target_prim)
    gripper_close_target = get_gripper_close_target(object_name)
    perception_source = f"{GRASP_BACKEND}_required"
    grasp_position_active = False
    grasp_fusion = None
    perception_started = time.perf_counter()
    backend_enabled = USE_ANYGRASP if GRASP_BACKEND == "anygrasp" else USE_GRASPNET
    if not backend_enabled:
        return {
            "success": False,
            "message": f"{GRASP_BACKEND} perception is mandatory but disabled",
            "error_code": f"{GRASP_BACKEND.upper()}_REQUIRED",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "perception_source": perception_source,
        }
    try:
        grasp_fusion = infer_grasp_fused_world_pose(
            get_current_stage(),
            state.grasp_camera,
            target_prim,
        )
    except Exception as exc:
        release_cuda_inference_cache()
        print(f"❌ {GRASP_BACKEND} 视觉抓取不可用，强制任务终止")
        return {
            "success": False,
            "message": f"{GRASP_BACKEND} perception is required but unavailable",
            "error_code": f"{GRASP_BACKEND.upper()}_UNAVAILABLE",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "perception_source": perception_source,
            "details": {
                "stage": f"{GRASP_BACKEND}_temporal_fusion",
                "backend": GRASP_BACKEND,
                "error": str(exc),
            },
        }
    else:
        perception_source = str(grasp_fusion.get("source", f"{GRASP_BACKEND}_temporal_fusion"))
        # The selected backend's calibrated position is the sole visual
        # grasp-center source; geometry remains a bounded validation gate.
        # Keep the top-down/PCA orientation because its physical jaw-axis
        # constraint is independent of the detector's camera-frame wrist roll.
        object_position = np.asarray(grasp_fusion["position"], dtype=float).copy()
        grasp_orientation = top_down_orientation
        grasp_position_active = True
        grasp_strategy = f"{GRASP_BACKEND}_temporal_fusion_position_top_down"
        publish_transport_tracking({
            "event": f"{GRASP_BACKEND}_perception",
            "state": "grasp_pose_fused",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "observed_position": object_position.tolist(),
            "backend": GRASP_BACKEND,
            "orientation": grasp_fusion.get("orientation"),
            "fusion": grasp_fusion,
            "replan": False,
        })
        print(
            f"🎯 使用 {GRASP_BACKEND} 多帧融合 + 相机标定抓取点: "
            f"position={object_position}, "
            f"confidence={grasp_fusion['confidence']:.3f}"
        )
    print(
        f"⏱️ SAM + {GRASP_BACKEND}: "
        f"{time.perf_counter() - perception_started:.2f} s"
    )

    bbox_center, bbox_min, bbox_max = get_current_bbox_center(
        get_current_stage(), target_prim, object_prim_path
    )
    if not grasp_position_active:
        object_position = np.asarray(bbox_center, dtype=float).copy()
        print(
            "🎯 使用 USD 几何包围盒中心: "
            f"center={object_position}"
        )
    if not grasp_position_active:
        minimum_grasp_height = bbox_min[2] + (
            bbox_max[2] - bbox_min[2]
        ) * GRASP_MIN_HEIGHT_FRACTION
        if object_position[2] < minimum_grasp_height:
            print(
                f"⚠️ 场景抓取高度 {object_position[2]:.4f} 过低，"
                f"提升到物体包围盒内部 {minimum_grasp_height:.4f}"
            )
            object_position[2] = minimum_grasp_height
        object_position = adjust_object_grasp_position(
            object_name,
            object_position,
            bbox_min,
            bbox_max,
        )
        object_position[2] += DACH_GRASP_HEIGHT_OFFSET
    grasp_target_center = np.asarray(object_position, dtype=float).copy()
    # The selected backend and its camera-frame calibration remain the grasp source.
    # Its selected pinch point can still carry a small object-specific lateral
    # residual.  Use the live collision geometry only as a bounded final
    # containment guard, so closing never begins with the object beside a jaw.
    physical_alignment_center = grasp_target_center.copy()
    if grasp_position_active:
        bbox_center_array = np.asarray(bbox_center, dtype=float)
        bbox_min_array = np.asarray(bbox_min, dtype=float)
        bbox_max_array = np.asarray(bbox_max, dtype=float)
        geometry_margin = 0.01
        grasp_xy_in_bounds = bool(
            np.all(
                physical_alignment_center[:2]
                >= bbox_min_array[:2] - geometry_margin
            )
            and np.all(
                physical_alignment_center[:2]
                <= bbox_max_array[:2] + geometry_margin
            )
        )
        grasp_vertical_margin = max(
            0.01,
            0.15 * float(bbox_max_array[2] - bbox_min_array[2]),
        )
        grasp_z_in_bounds = (
            bbox_min_array[2] - grasp_vertical_margin
            <= physical_alignment_center[2]
            <= bbox_max_array[2] + grasp_vertical_margin
        )
        if grasp_xy_in_bounds and grasp_z_in_bounds:
            section_center = get_current_mesh_horizontal_cross_section_center(
                get_current_stage(),
                target_prim,
                physical_alignment_center[:2],
                object_prim_path,
            )
            section_correction = (
                np.asarray(section_center, dtype=float)
                - physical_alignment_center[:2]
            )
            section_correction_norm = float(np.linalg.norm(section_correction))
            if section_correction_norm <= GRASP_MAX_PLANAR_CORRECTION:
                physical_alignment_center[:2] = section_center
            containment_xy_error = (
                bbox_center_array[:2] - physical_alignment_center[:2]
            )
            containment_xy_error_norm = float(np.linalg.norm(containment_xy_error))
            physical_alignment_center[2] = grasp_target_center[2]
            grasp_target_center = physical_alignment_center.copy()
            object_position = grasp_target_center.copy()
            grasp_strategy += "+geometry_validated"
            print(
                f"📐 {GRASP_BACKEND} 抓取点局部截面校正: "
                f"correction={section_correction_norm:.4f} m, "
                f"applied={section_correction_norm <= GRASP_MAX_PLANAR_CORRECTION}"
            )
            print(
                f"📐 {GRASP_BACKEND} 抓取点物理包含校正: "
                f"xy_error={containment_xy_error_norm:.4f} m"
            )
        else:
            print(
                f"⛔ {GRASP_BACKEND} 点未通过目标几何一致性检查: "
                f"xy_in_bounds={grasp_xy_in_bounds}, "
                f"z_in_bounds={grasp_z_in_bounds}"
            )
            return {
                "success": False,
                "message": f"{GRASP_BACKEND} point is outside the target geometry",
                "error_code": f"{GRASP_BACKEND.upper()}_GEOMETRY_MISMATCH",
                "object_name": str(object_name),
                "target_name": str(target_name),
                "grasp_position": object_position.tolist(),
                "bbox_min": bbox_min_array.tolist(),
                "bbox_max": bbox_max_array.tolist(),
                "perception_source": perception_source,
            }
    preplanned_goal_position = resolve_place_position(
        target_name, object_position
    )
    final_grasp_tcp = get_tcp_target_for_gripper_center(
        object_position,
        grasp_orientation,
    )
    final_pose_reachability = _evaluate_arm_pose_reachability(
        object_name,
        final_grasp_tcp,
        grasp_orientation,
    )
    strict_selected = final_pose_reachability.get("strict_selected")
    if strict_selected is None:
        final_pose_reachability = dict(final_pose_reachability)
        final_pose_reachability["selected"] = None
    else:
        final_pose_reachability = dict(final_pose_reachability)
        final_pose_reachability.update(
            {
                "selected": strict_selected,
                "right": bool(final_pose_reachability.get("strict_right")),
                "left": bool(final_pose_reachability.get("strict_left")),
                "hover_position": final_pose_reachability.get("strict_hover_position"),
                "hover_plan": final_pose_reachability.get("strict_hover_plan"),
            }
        )
    planned_transport_yaw_deg = 0.0
    planned_place_hover_clearance_m = 0.22
    candidate_selection = _select_geometry_constrained_grasp_candidate(
        object_name,
        target_name,
        target_prim,
        object_prim_path,
        object_position,
        bbox_min,
        bbox_max,
        preplanned_goal_position,
        grasp_orientation,
        reachability,
        gripper_opening_width,
        aperture_margin,
    )
    if not candidate_selection["selected"]:
        return {
            "success": False,
            "message": "no strictly reachable geometry-constrained grasp pose",
            "planned_short_axis_alignment": candidate_selection[
                "planned_alignment"
            ],
            "reachability_precheck": reachability,
            "grasp_pose_attempts": candidate_selection["attempts"],
            "grasp_fusion": grasp_fusion,
        }
    grasp_orientation = candidate_selection["orientation"]
    final_grasp_tcp = candidate_selection["tcp"]
    final_pose_reachability = candidate_selection["reachability"]
    planned_alignment = candidate_selection["planned_alignment"]
    tilt_deg = candidate_selection["tilt_deg"]
    roll_deg = candidate_selection.get("roll_deg", 0.0)
    opposite_tool_branch = candidate_selection["opposite_tool_branch"]
    task_pose_gate = candidate_selection["task_gate"]
    grasp_strategy += f"+geometry_constrained_short_axis_{tilt_deg:g}deg"
    if abs(roll_deg) > 1e-6:
        grasp_strategy += f"+geometry_constrained_roll_{roll_deg:g}deg"
    if opposite_tool_branch:
        grasp_strategy += "+equivalent_tool_branch_180deg"
    selected_preplanned_goal = task_pose_gate.get("selected_goal_position")
    if selected_preplanned_goal is not None:
        preplanned_goal_position = np.asarray(
            selected_preplanned_goal, dtype=float
        )
    planned_transport_yaw_deg = float(
        task_pose_gate.get("transport_yaw_offset_deg", 0.0)
    )
    planned_place_hover_clearance_m = float(
        task_pose_gate.get("place_hover_clearance_m", 0.22)
    )
    print(
        "🧭 统一几何约束抓取候选已选定: "
        f"arm={final_pose_reachability['selected']}, "
        f"tilt={tilt_deg:.1f}°, alignment={planned_alignment:.3f}, "
        f"opposite_branch={opposite_tool_branch}"
    )
    if final_pose_reachability["selected"] is not None:
        _activate_arm(final_pose_reachability["selected"])
        reachability["selected_arm"] = final_pose_reachability["selected"]
        reachability["final_pose_right_reachable"] = final_pose_reachability["right"]
        reachability["final_pose_left_reachable"] = final_pose_reachability["left"]
    else:
        return {
            "success": False,
            "message": "no arm can reach the complete grasp and lift pose",
            "reachability_precheck": reachability,
            "strict_pose_attempts": final_pose_reachability.get("attempts", []),
            "grasp_fusion": grasp_fusion,
        }
    print(
        f"🎯 DACH 抓取中心上移 {DACH_GRASP_HEIGHT_OFFSET:.3f} m: "
        f"z={object_position[2]:.4f}"
    )
    initial_object_position, _, _ = get_sim_pose(target_prim)
    grasp_center_z_offset = float(object_position[2] - initial_object_position[2])

    goal_position = preplanned_goal_position.copy()
    show_red_grasp_point(get_current_stage(), object_position)
    step_app()
    ensure_robot_control_ready()
    print(f"🎯 最终抓取位置: {object_position}")
    print(f"🧭 最终抓取姿态(wxyz): {grasp_orientation}")
    print(f"📍 语义目标 {target_name} -> 放置位置: {goal_position}")

    print(f"🚀 执行 Pick & Place: {object_name} -> {target_name}")
    # grasp_orientation 已由 AnyGrasp（6DoF）或 top-down fallback 确定。
    # 不要在这里重新调用 get_top_down_grasp_orientation()，
    # 否则 AnyGrasp 推理出的侧向/斜向位姿会被丢弃，
    # 导致 hover 始终沿竖直 [0,0,-1] 方向接近，在 arm 工作空间边界处 IK 失败。
    hover_reference_orientation = grasp_orientation
    hover_orientation = hover_reference_orientation
    grasp_approach_direction = quat_rotate(
        hover_reference_orientation,
        np.array([1.0, 0.0, 0.0]),
    )
    # Desktop grasps must keep the TCP approach orientation constrained from
    # the first hover waypoint. A position-only RRT can reach the point by
    # flipping the wrist far away from the tabletop pose, so it is forbidden
    # for non-banana objects as well.
    hover_motion_orientation = hover_orientation
    grasp_approach_direction /= np.linalg.norm(grasp_approach_direction)
    grasp_position = get_tcp_target_for_gripper_center(
        object_position,
        hover_reference_orientation,
    )
    place_gripper_position = get_tcp_target_for_gripper_center(
        goal_position,
        hover_reference_orientation,
    )

    hover_position = None
    attempted_hover_positions = []
    pre_hover_object_position, pre_hover_object_orientation, _ = get_sim_pose(
        target_prim
    )
    # The reachability result is a kinematic gate for arm selection only.
    # Executing its first IK waypoint directly bypasses the Lula world model
    # and can sweep through the basket. Every commanded hover must therefore
    # be replanned below with the table and basket obstacles enabled.
    print("🛡️ 选臂 IK 仅用于预检；悬停移动统一交给 Lula RRT 碰撞规划")
    # Always enter from the highest hover candidate. The target object is not
    # an RRT obstacle because the final grasp must contact it, so routing
    # directly to a near hover can sweep a finger through the object before
    # the controlled vertical approach starts.
    hover_clearances = (0.24, 0.20, 0.16, 0.12, 0.08)
    for hover_clearance in hover_clearances:
        if hover_position is not None:
            break
        candidate_hover = (
            grasp_position - grasp_approach_direction * hover_clearance
        )
        attempted_hover_positions.append(candidate_hover.tolist())
        if move_ee_to(
            f"hover_{hover_clearance:.2f}",
            candidate_hover,
            max_steps=240,
            tolerance=0.025,
            orientation=hover_motion_orientation,
            retry_position_only=False,
            horizontal_tolerance=None,
            gripper_positions=gripper_open_target,
            strict_orientation=True,
        ):
            hover_position = candidate_hover
            print(f"✅ 选用可达悬停间距: {hover_clearance:.2f} m")
            break
    if hover_position is None:
        return {
            "success": False,
            "message": "failed to reach hover position",
            "attempted_hover_positions": attempted_hover_positions,
            "reachability_precheck": reachability,
        }

    post_hover_object_position, post_hover_object_orientation, _ = get_sim_pose(
        target_prim
    )
    hover_object_translation = float(
        np.linalg.norm(
            np.asarray(post_hover_object_position, dtype=float)
            - np.asarray(pre_hover_object_position, dtype=float)
        )
    )
    pre_hover_quat = np.asarray(pre_hover_object_orientation, dtype=float)
    post_hover_quat = np.asarray(post_hover_object_orientation, dtype=float)
    pre_hover_quat /= max(float(np.linalg.norm(pre_hover_quat)), 1e-9)
    post_hover_quat /= max(float(np.linalg.norm(post_hover_quat)), 1e-9)
    hover_object_rotation_deg = float(
        np.degrees(
            2.0
            * np.arccos(
                np.clip(abs(float(np.dot(pre_hover_quat, post_hover_quat))), 0.0, 1.0)
            )
        )
    )
    object_moved_during_hover = bool(
        hover_object_translation > 0.008 or hover_object_rotation_deg > 5.0
    )
    if object_moved_during_hover:
        print(
            "🛑 悬停路径推动了目标，废弃旧抓取点: "
            f"translation={hover_object_translation:.4f} m, "
            f"rotation={hover_object_rotation_deg:.2f}°"
        )
        move_robot_home(frames=90)
        if _reacquire_attempt < 1:
            print(f"🔄 重新运行 {GRASP_BACKEND} 感知并规划安全抓取")
            return execute_pick_place(
                object_name,
                target_name,
                _reacquire_attempt=_reacquire_attempt + 1,
            )
        return {
            "success": False,
            "message": "object moved during hover; fresh grasp pose was invalidated",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "hover_object_translation_m": hover_object_translation,
            "hover_object_rotation_deg": hover_object_rotation_deg,
            "grasp_fusion": grasp_fusion,
        }

    if grasp_position_active:
        object_position = grasp_target_center.copy()
    else:
        live_bbox_center, live_bbox_min, live_bbox_max = get_current_bbox_center(
            get_current_stage(),
            target_prim,
            object_prim_path,
        )
        live_object_position, _, _ = get_sim_pose(target_prim)
        live_minimum_grasp_height = live_bbox_min[2] + (
            live_bbox_max[2] - live_bbox_min[2]
        ) * GRASP_MIN_HEIGHT_FRACTION
        live_object_position[2] = max(
            live_object_position[2],
            live_minimum_grasp_height,
        )
        object_position = live_object_position.copy()
        object_position[:2] = live_bbox_center[:2]
        object_position[2] += DACH_GRASP_HEIGHT_OFFSET
    # Keep the exact pose selected by strict IK. Rebuilding the configured
    # default here can overwrite an adaptive banana tilt candidate.
    grasp_orientation = hover_reference_orientation
    desired_closing_axis = -get_current_mesh_horizontal_min_width_axis(
        get_current_stage(), target_prim, object_prim_path
    )
    desired_short_axis = np.asarray(desired_closing_axis, dtype=float)[:2]
    desired_short_axis /= max(float(np.linalg.norm(desired_short_axis)), 1e-9)
    desired_long_axis = np.array(
        [-desired_short_axis[1], desired_short_axis[0], 0.0],
        dtype=float,
    )
    _, _, live_short_width = get_mesh_extent_along_axis(
        get_current_stage(),
        object_prim_path,
        np.array([desired_short_axis[0], desired_short_axis[1], 0.0]),
    )
    _, _, live_long_width = get_mesh_extent_along_axis(
        get_current_stage(),
        object_prim_path,
        desired_long_axis,
    )
    alignment_threshold = (
        0.0
        if live_long_width <= max(live_short_width * 1.05, live_short_width + 0.002)
        else float(GRASP_MIN_SHORT_AXIS_ALIGNMENT)
    )
    # Every payload uses the same fixed-orientation convergence stage. The
    # selected geometry candidate determines the target; the live jaw-axis
    # check only validates that PhysX reached that target without a branch flip.
    orientation_result = move_ee_collision_aware_approach(
        "hover_orientation",
        get_rmp_ee_position(),
        tolerance=0.02,
        orientation=hover_reference_orientation,
        gripper_positions=gripper_open_target,
        desired_closing_axis=desired_closing_axis,
        minimum_closing_alignment=alignment_threshold,
        maximum_approach_tilt_rad=maximum_grasp_tilt,
    )
    if orientation_result["success"]:
        jaw_axis_result = align_physical_closing_axis_at_hover(
            desired_closing_axis,
            gripper_open_target,
            alignment_threshold,
        )
        orientation_result["physical_jaw_axis_refinement"] = jaw_axis_result
        orientation_result["success"] = jaw_axis_result["success"]
        if jaw_axis_result["success"]:
            _, refined_rotation = state.controller.get_end_effector_pose()
            orientation_result["hold_orientation"] = rot_matrix_to_quat(
                refined_rotation
            ).tolist()
    # Validate the pose actually reached by PhysX, not the stale alignment
    # measured before the hover-orientation trajectory ran.
    physical_alignment = float(
        abs(
            np.dot(
                get_gripper_closing_axis()[:2],
                np.asarray(desired_closing_axis, dtype=float)[:2],
            )
        )
    )
    if (
        not orientation_result["success"]
        or physical_alignment < alignment_threshold
    ):
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "failed to align physical fingers with object short axis",
            "physical_closing_alignment": physical_alignment,
            "approach": orientation_result,
            "grasp_fusion": grasp_fusion,
        }
    _, aligned_rotation = state.controller.get_end_effector_pose()
    grasp_motion_orientation = rot_matrix_to_quat(aligned_rotation)
    print(
        f"✅ {canonical_object_name} 物理短轴闭合姿态已在悬停高度完成: "
        f"alignment={physical_alignment:.3f}"
    )
    # Open the jaw before measuring the finger-center offset: the offset must
    # describe the same settled open-jaw geometry used for hover alignment and
    # the following vertical descent.
    open_result = open_gripper_slowly(
        get_rmp_ee_position(),
        orientation=grasp_motion_orientation,
        frames=18,
        target_open=gripper_open_target,
    )
    if not open_result["converged"]:
        print(
            "⚠️ 夹爪未完全达到标称开口，"
            f"继续执行几何开口检查: error={open_result['error_m']:.4f} m"
        )
    print(
        "✅ 下探前夹爪已张开到物理最大开口: "
        f"jaw={state.dach_arm.gripper.get_joint_positions()}"
    )
    live_closing_axis_3d = get_gripper_closing_axis_3d()
    gripper_opening_width = float(get_gripper_inner_opening_width())
    _, _, object_closing_width = get_mesh_extent_along_axis(
        get_current_stage(),
        object_prim_path,
        live_closing_axis_3d,
    )
    aperture_margin = GRASP_APERTURE_MARGIN
    required_opening_width = float(object_closing_width + aperture_margin)
    print(
        "📏 抓取前开口尺寸校验: "
        f"jaw_gap={gripper_opening_width:.4f} m, "
        f"object_width={object_closing_width:.4f} m, "
        f"required={required_opening_width:.4f} m"
    )
    if required_opening_width > gripper_opening_width:
        move_robot_home(frames=90)
        message = (
            f"object minimum grasp width {minimum_object_width * 1000:.1f} mm exceeds "
            f"single-arm gripper opening {gripper_opening_width * 1000:.1f} mm; "
            "use a bimanual grasp or a wider tool"
        )
        return {
            "success": False,
            "message": message,
            "object_name": str(object_name),
            "target_name": str(target_name),
            "physical_constraint": {
                "gripper_opening_m": gripper_opening_width,
                "object_width_on_closing_axis_m": float(object_closing_width),
                "minimum_object_grasp_width_m": float(minimum_object_width),
                "required_opening_m": required_opening_width,
                "safety_margin_m": aperture_margin,
            },
            "reachability_precheck": reachability,
            "grasp_fusion": grasp_fusion,
        }
    verify_gripper_center_local_offset()
    grasp_position = get_tcp_target_for_gripper_center(
        object_position,
        grasp_motion_orientation,
    )
    place_gripper_position = get_tcp_target_for_gripper_center(
        goal_position,
        grasp_motion_orientation,
    )

    # Align XY while the open fingers are still at hover height.  The old
    # post-descent refinement swept a tilted jaw sideways next to the table;
    # that could cross the table collider and make the wrist jump between IK
    # branches.  Once aligned here, the grasp approach remains vertical.
    planar_center_tolerance = GRASP_PLANAR_CENTER_TOLERANCE
    planar_refinement_steps = GRASP_PLANAR_REFINEMENT_STEPS
    desired_grasp_tcp_z = float(grasp_position[2])
    for refinement in range(planar_refinement_steps):
        live_finger_midpoint = get_gripper_finger_midpoint()
        desired_planar_center = np.asarray(
            physical_alignment_center[:2],
            dtype=float,
        )
        planar_error = np.asarray(
            desired_planar_center - live_finger_midpoint[:2],
            dtype=float,
        )
        planar_error_norm = float(np.linalg.norm(planar_error))
        print(
            f"🎯 悬停高度夹指中点 XY 校正 "
            f"{refinement + 1}/{planar_refinement_steps}: "
            f"target={desired_planar_center}, "
            f"finger={live_finger_midpoint[:2]}, "
            f"error={planar_error}, norm={planar_error_norm:.4f}m"
        )
        if planar_error_norm <= planar_center_tolerance:
            break
        correction_scale = min(
            1.0,
            GRASP_MAX_PLANAR_CORRECTION / max(planar_error_norm, 1e-9),
        )
        corrected_center = np.asarray(
            [
                desired_planar_center[0],
                desired_planar_center[1],
                live_finger_midpoint[2],
            ],
            dtype=float,
        )
        corrected_hover_tcp = get_tcp_target_for_gripper_center(
            corrected_center,
            grasp_motion_orientation,
        )
        current_tcp = get_rmp_ee_position()
        corrected_hover_tcp = current_tcp + correction_scale * (
            corrected_hover_tcp - current_tcp
        )
        hover_alignment_reached = move_ee_smooth(
            f"pregrasp_planar_align_{refinement + 1}",
            get_rmp_ee_position(),
            corrected_hover_tcp,
            segments=1,
            tolerance=0.008,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
            monitor_table_clearance=True,
        )
        if not hover_alignment_reached:
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "failed to align finger midpoint at hover height",
                "planar_error": planar_error.tolist(),
            }
    live_finger_midpoint = get_gripper_finger_midpoint()
    desired_planar_center = np.asarray(
        physical_alignment_center[:2],
        dtype=float,
    )
    planar_error = np.asarray(
        desired_planar_center - live_finger_midpoint[:2],
        dtype=float,
    )
    planar_error_norm = float(np.linalg.norm(planar_error))
    if planar_error_norm > planar_center_tolerance:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "finger midpoint is not centered at hover height",
            "planar_error": planar_error.tolist(),
            "planar_error_norm": planar_error_norm,
        }
    grasp_position[:2] = get_rmp_ee_position()[:2]
    grasp_position[2] = desired_grasp_tcp_z

    # Safety guard: predict finger table clearance at the grasp position and
    # lift the grasp center if the tilted jaw would dip below the safety
    # threshold. Use the live finger-center-to-bottom geometry directly;
    # inferring it from TCP descent is inaccurate for the offset DACH jaws.
    if state.planning_table_surface_z is not None:
        min_finger_clearance = float(
            os.environ.get("AURA_MIN_GRIPPER_TABLE_CLEARANCE", "0.0")
        ) + float(os.environ.get("AURA_TABLE_CLEARANCE_ABORT_MARGIN", "0.0"))
        # Safety margin: aim slightly higher than the abort threshold so
        # one-frame IK overshoot cannot trip the safety stop.  This pad, not
        # the banana geometry, is what sets the final descent depth -- the
        # guard lifts the grasp center until predicted clearance equals it,
        # so shrinking it is the only way to get the jaw lower.
        guard_pad = GRASP_CLEARANCE_GUARD_PAD
        guard_target_clearance = max(
            min_finger_clearance + guard_pad,
            get_gripper_table_contact_safe_clearance(),
        )
        current_finger_bottom = min(
            float(np.min(get_finger_collision_world_corners(state.left_finger, "left")[:, 2])),
            float(np.min(get_finger_collision_world_corners(state.right_finger, "right")[:, 2])),
        )
        current_finger_center_z = float(get_gripper_collision_center()[2])
        finger_bottom_offset = current_finger_bottom - current_finger_center_z
        predicted_finger_bottom = (
            float(physical_alignment_center[2]) + finger_bottom_offset
        )
        predicted_clearance = (
            predicted_finger_bottom - state.planning_table_surface_z
        )
        if predicted_clearance < guard_target_clearance:
            required_lift = guard_target_clearance - predicted_clearance
            print(
                f"⚠️ Grasp finger clearance unsafe: "
                f"finger_bottom_offset={finger_bottom_offset:.4f} m, "
                f"predicted_bottom={predicted_finger_bottom:.4f} m, "
                f"predicted={predicted_clearance:.4f} m < "
                f"target={guard_target_clearance:.4f} m, "
                f"lifting grasp center by {required_lift:.4f} m"
            )
            object_position[2] += required_lift
            grasp_position[2] += required_lift
            physical_alignment_center[2] += required_lift
            grasp_center_z_offset += required_lift
            # Keep the closed-loop target at the same safety-adjusted height.
            # Its XY remains the calibrated AnyGrasp point; without this the
            # later center refinement could undo the table-clearance guard.
            grasp_target_center[2] += required_lift
            print(f"🎯 调整后抓取中心 z={object_position[2]:.4f}, tcp_z={grasp_position[2]:.4f}")

    print(
        f"🎯 DACH 稀疏关键姿态抓取: tcp_xy={grasp_position[:2]}, "
        f"hover_z={hover_position[2]:.4f}, "
        f"grasp_xy={grasp_position[:2]}, grasp_z={grasp_position[2]:.4f}"
    )

    lift_position = grasp_position + np.array(
        [0.0, 0.0, TRANSPORT_LIFT_HEIGHT]
    )
    place_hover_position = place_gripper_position + np.array(
        [0.0, 0.0, planned_place_hover_clearance_m]
    )
    pre_descent_object_center, _, _ = get_current_bbox_center(
        get_current_stage(), target_prim, object_prim_path
    )
    approach_result = move_ee_collision_aware_approach(
        "grasp_approach",
        grasp_position,
        # Use one tightly tracked descent for every payload. Live geometry
        # and table-clearance monitors remain the final safety authority.
        tolerance=0.008,
        # The base is positioned so both banana and can TCP targets remain in
        # Lula's fixed top-down orientation manifold.  Keeping this rotation
        # constrained is essential: an unconstrained RRT solution can move the
        # wrist while leaving the finger center far from the USD object.
        orientation=grasp_motion_orientation,
        gripper_positions=gripper_open_target,
        maximum_approach_tilt_rad=maximum_grasp_tilt,
        terminal_joint_tolerance_rad=0.003,
    )
    if not approach_result["success"]:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "failed to reach grasp position safely",
            "approach": approach_result,
        }
    grasp_motion_orientation = (
        np.asarray(approach_result["hold_orientation"], dtype=float)
        if approach_result.get("hold_orientation") is not None
        else grasp_motion_orientation
    )
    grasp_position = get_rmp_ee_position().copy()

    post_descent_object_center, _, _ = get_current_bbox_center(
        get_current_stage(), target_prim, object_prim_path
    )
    descent_object_displacement = float(
        np.linalg.norm(
            np.asarray(post_descent_object_center, dtype=float)
            - np.asarray(pre_descent_object_center, dtype=float)
        )
    )
    if descent_object_displacement > 0.003:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "grasp descent moved the target object; closing was blocked",
            "object_displacement_m": descent_object_displacement,
            "pre_descent_object_center": np.asarray(
                pre_descent_object_center, dtype=float
            ).tolist(),
            "post_descent_object_center": np.asarray(
                post_descent_object_center, dtype=float
            ).tolist(),
            "grasp_fusion": grasp_fusion,
        }

    refinement_steps = GRASP_REFINEMENT_STEPS
    for refinement in range(refinement_steps):
        live_finger_center = get_gripper_collision_center()
        if grasp_position_active:
            desired_finger_center = physical_alignment_center.copy()
            # Keep AnyGrasp's table-safe height. This final loop corrects the
            # lateral containment residual only.
            desired_finger_center[2] = live_finger_center[2]
        else:
            live_object_position, _, _ = get_sim_pose(target_prim)
            desired_finger_center = live_object_position.copy()
            desired_finger_center[2] += grasp_center_z_offset
        center_error = desired_finger_center - live_finger_center
        center_error_norm = float(np.linalg.norm(center_error))
        print(
            f"🎯 夹爪中心闭环校正 {refinement + 1}/{refinement_steps}: "
            f"error={center_error}, norm={center_error_norm:.4f}"
        )
        if center_error_norm <= 0.01:
            break
        grasp_position = get_rmp_ee_position() + center_error
        refinement_reached = move_ee_smooth(
            f"grasp_refine_{refinement + 1}",
            get_rmp_ee_position(),
            grasp_position,
            segments=1,
            max_steps_per_segment=40,
            tolerance=0.008,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
            monitor_table_clearance=True,
        )
        if not refinement_reached:
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "failed to center DACH gripper on object",
                "center_error": center_error.tolist(),
            }
    # At grasp height do not sweep horizontally.  A miss is safer to report
    # than trying to correct beside the table and distorting the wrist path.
    live_finger_midpoint = get_gripper_finger_midpoint()
    desired_planar_center = np.asarray(
        physical_alignment_center[:2],
        dtype=float,
    )
    planar_error = np.asarray(
        desired_planar_center - live_finger_midpoint[:2],
        dtype=float,
    )
    planar_error_norm = float(np.linalg.norm(planar_error))
    if planar_error_norm > planar_center_tolerance:
        low_pose_clearance = float(get_gripper_table_clearance())
        correction_limit = GRASP_LOW_POSE_CORRECTION_LIMIT
        correction_clearance = max(
            MIN_GRIPPER_TABLE_CLEARANCE + TABLE_CLEARANCE_ABORT_MARGIN,
            get_gripper_table_contact_safe_clearance(),
        )
        if (
            correction_limit > 0.0
            and planar_error_norm <= correction_limit
            and low_pose_clearance >= correction_clearance
        ):
            print(
                "🧭 低位夹指中点固定姿态平面对中: "
                f"error={planar_error_norm:.4f} m, "
                f"clearance={low_pose_clearance:.4f} m"
            )
            corrected_grasp_tcp = get_rmp_ee_position().copy()
            corrected_grasp_tcp[:2] += planar_error
            if move_ee_smooth(
                "low_grasp_planar_align",
                get_rmp_ee_position(),
                corrected_grasp_tcp,
                segments=1,
                tolerance=0.008,
                orientation=grasp_motion_orientation,
                gripper_positions=gripper_open_target,
                monitor_table_clearance=True,
            ):
                live_finger_midpoint = get_gripper_finger_midpoint()
                desired_planar_center = np.asarray(
                    physical_alignment_center[:2],
                    dtype=float,
                )
                planar_error = np.asarray(
                    desired_planar_center - live_finger_midpoint[:2],
                    dtype=float,
                )
                planar_error_norm = float(np.linalg.norm(planar_error))
        if planar_error_norm > planar_center_tolerance:
            live_object_center, _, _ = get_current_bbox_center(
                get_current_stage(), target_prim, object_prim_path
            )
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "finger midpoint is not centered on target object",
                "planar_error": planar_error.tolist(),
                "planar_error_norm": planar_error_norm,
                "gripper_table_clearance_m": low_pose_clearance,
                "planned_object_center": physical_alignment_center.tolist(),
                "live_object_center": np.asarray(
                    live_object_center, dtype=float
                ).tolist(),
                "finger_midpoint": np.asarray(
                    live_finger_midpoint, dtype=float
                ).tolist(),
                "tcp_position": np.asarray(
                    get_rmp_ee_position(), dtype=float
                ).tolist(),
            }

    # Resolve final vertical insertion from live collision geometry for every
    # payload. This corrects conservative IK height without lateral sweeping.
    safe_clearance = get_gripper_table_contact_safe_clearance()
    target_clearance = safe_clearance + 0.002
    live_clearance = float(get_gripper_table_clearance())
    if live_clearance > target_clearance + 0.004:
        insertion_distance = live_clearance - target_clearance
        insertion_target = get_rmp_ee_position().copy()
        insertion_target[2] -= insertion_distance
        print(
            "⬇️ 最终垂直抓取插入: "
            f"clearance={live_clearance:.4f} m -> "
            f"target={target_clearance:.4f} m, "
            f"descent={insertion_distance:.4f} m"
        )
        insertion_reached = move_ee_smooth(
            "final_vertical_grasp_insertion",
            get_rmp_ee_position(),
            insertion_target,
            segments=1,
            max_steps_per_segment=50,
            tolerance=0.006,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
            monitor_table_clearance=True,
        )
        live_clearance = float(get_gripper_table_clearance())
        if not insertion_reached or live_clearance > target_clearance + 0.006:
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "failed to reach physical grasp depth",
                "gripper_table_clearance_m": live_clearance,
                "target_clearance_m": target_clearance,
            }
    grasp_position = get_rmp_ee_position().copy()
    physical_alignment_center[2] = get_gripper_collision_center()[2]

    gripper_table_clearance = get_gripper_table_clearance()
    print(
        f"🛡️ 夹爪前段桌面间隙: {gripper_table_clearance:.4f} m "
        f"(minimum={MIN_GRIPPER_TABLE_CLEARANCE:.4f} m)"
    )
    if gripper_table_clearance < MIN_GRIPPER_TABLE_CLEARANCE:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "gripper table clearance is below safety minimum",
            "gripper_table_clearance_m": gripper_table_clearance,
            "minimum_clearance_m": MIN_GRIPPER_TABLE_CLEARANCE,
        }

    lift_position = grasp_position + np.array(
        [0.0, 0.0, TRANSPORT_LIFT_HEIGHT]
    )
    place_hover_position = place_gripper_position + np.array(
        [0.0, 0.0, planned_place_hover_clearance_m]
    )

    live_closing_axis = get_gripper_closing_axis()[:2]
    validated_closing_axis = np.asarray(desired_closing_axis, dtype=float)[:2]
    validated_closing_axis /= max(
        float(np.linalg.norm(validated_closing_axis)), 1e-9
    )
    axis_alignment = float(
        abs(np.dot(live_closing_axis, validated_closing_axis))
    )
    print(
        f"🧭 {canonical_object_name} 闭合轴校验: "
        f"gripper={live_closing_axis}, "
        f"object_short={validated_closing_axis}, "
        f"alignment={axis_alignment:.3f}"
    )
    if axis_alignment < alignment_threshold:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "gripper is not aligned with object closing axis",
            "closing_axis_alignment": axis_alignment,
            "minimum_alignment": alignment_threshold,
        }

    # Keep the dynamic target settled while the open gripper holds its final
    # physical pose. No kinematic freeze or attachment is used.
    for _ in range(5):
        hold_ee_target(grasp_position, grasp_motion_orientation)
        state.dach_arm.gripper.set_joint_positions(gripper_open_target)
        step_app()
    preclose_table_clearance = float(get_gripper_table_clearance())
    safe_preclose_clearance = get_gripper_table_contact_safe_clearance()
    preclose_target_clearance = safe_preclose_clearance + 0.002
    if preclose_table_clearance < preclose_target_clearance:
        clearance_correction = (
            preclose_target_clearance - preclose_table_clearance
        )
        corrected_preclose_position = get_rmp_ee_position().copy()
        corrected_preclose_position[2] += clearance_correction
        print(
            "🛡️ 闭爪前桌面净空校正: "
            f"observed={preclose_table_clearance:.4f} m, "
            f"target={preclose_target_clearance:.4f} m, "
            f"raise={clearance_correction:.4f} m"
        )
        correction_reached = move_ee_smooth(
            "preclose_vertical_clearance_correction",
            get_rmp_ee_position(),
            corrected_preclose_position,
            segments=1,
            max_steps_per_segment=40,
            tolerance=0.001,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
            # The measured pose is already below the conservative abort
            # threshold. This correction is strictly vertical and moves away
            # from the table, so checking the pre-correction violation inside
            # the trajectory would abort before the recovery can run.
            monitor_table_clearance=False,
            max_joint_step_rad=GRASP_APPROACH_MAX_JOINT_STEP,
            minimum_frames=GRASP_APPROACH_MIN_FRAMES,
        )
        grasp_position = get_rmp_ee_position().copy()
        preclose_table_clearance = float(get_gripper_table_clearance())
        if preclose_table_clearance < safe_preclose_clearance:
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "failed to establish safe table clearance before closing",
                "gripper_table_clearance_m": preclose_table_clearance,
                "target_clearance_m": preclose_target_clearance,
            }
    nominal_gripper_close_target = np.asarray(
        gripper_close_target, dtype=float
    ).copy()
    object_position_before_close, _, _ = get_sim_pose(target_prim)
    contact_confirmation_margin = GRASP_CONTACT_CONFIRMATION_MARGIN
    close_result = close_gripper_slowly(
        grasp_position,
        orientation=grasp_motion_orientation,
        frames=get_gripper_close_frames(object_name),
        target_close=nominal_gripper_close_target,
        monitor_table_clearance=True,
        maximum_contact_opening_width=(
            float(object_closing_width + contact_confirmation_margin)
        ),
        minimum_physical_opening_width=max(
            float(object_closing_width)
            - GRASP_MINIMUM_PHYSICAL_OPENING_MARGIN,
            0.0,
        ),
    )
    gripper_close_target = np.asarray(
        close_result["hold_target"], dtype=float
    )
    gripper_feedback = np.asarray(
        state.dach_arm.gripper.get_joint_positions(), dtype=float
    )
    closure_residual = float(
        np.mean(np.maximum(gripper_feedback - gripper_close_target, 0.0))
    )
    object_position_after_close, _, _ = get_sim_pose(target_prim)
    close_displacement = float(
        np.linalg.norm(object_position_after_close - object_position_before_close)
    )
    contact_confirmed = bool(close_result["contact_confirmed"])
    measured_efforts = np.asarray(
        close_result["measured_efforts_n"], dtype=float
    )
    print(
        "🧪 夹持接触校验: "
        f"feedback={gripper_feedback}, target={gripper_close_target}, "
        f"efforts={measured_efforts}N, "
        f"opening={close_result['opening_width_m']:.4f}m, "
        f"blocked_residual={closure_residual:.4f}m, "
        f"object_displacement={close_displacement:.4f}m, "
        f"confirmed={contact_confirmed}"
    )
    collision_diagnostics = get_gripper_collision_diagnostics()
    print(
        "🧪 左右夹指碰撞体诊断: "
        f"left={collision_diagnostics['left']}, "
        f"right={collision_diagnostics['right']}"
    )
    if not contact_confirmed:
        open_gripper_slowly(
            grasp_position,
            orientation=grasp_motion_orientation,
            target_open=gripper_open_target,
        )
        return {
            "success": False,
            "message": "grasp contact was not confirmed; lift was not commanded",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "gripper_feedback": gripper_feedback.tolist(),
            "gripper_target": gripper_close_target.tolist(),
            "gripper_efforts_n": measured_efforts.tolist(),
            "finger_contacts": close_result["finger_contacts"].tolist(),
            "blocked_residuals_m": close_result[
                "blocked_residuals_m"
            ].tolist(),
            "gripper_opening_width_m": close_result["opening_width_m"],
            "maximum_contact_opening_width_m": close_result[
                "maximum_contact_opening_width_m"
            ],
            "closure_residual_m": closure_residual,
            "object_displacement_m": close_displacement,
            "object_position_before_close": object_position_before_close.tolist(),
            "object_position_after_close": object_position_after_close.tolist(),
            "grasp_target_position": grasp_target_center.tolist(),
            "physical_alignment_center": physical_alignment_center.tolist(),
            "preclose_table_clearance_m": preclose_table_clearance,
            "gripper_table_clearance_m": get_gripper_table_clearance(),
            "finger_colliders": collision_diagnostics,
        }
    grasp_containment = get_object_gripper_containment(
        object_prim_path,
        target_prim=target_prim,
        reference_center=physical_alignment_center,
    )
    print(
        "🧪 闭合后双指空间包含校验: "
        f"contained={grasp_containment['contained']}, "
        f"axial={grasp_containment['axial_error_m']:.4f} m, "
        f"approach={grasp_containment['approach_error_m']:.4f} m, "
        f"lateral={grasp_containment['lateral_error_m']:.4f} m"
    )
    if not grasp_containment["contained"]:
        open_gripper_slowly(
            grasp_position,
            orientation=grasp_motion_orientation,
            target_open=gripper_open_target,
        )
        return {
            "success": False,
            "message": "object is not physically centered between both fingers",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "grasp_containment": grasp_containment,
        }
    # Keep the wrist orientation constrained throughout lift/carry. Object
    # motion is governed only by finger contact, friction, and gravity.
    transport_orientation = grasp_motion_orientation
    live_object_before_lift, live_object_orientation, _ = get_sim_pose(target_prim)
    grasp_reference_local = quat_to_rot_matrix(live_object_orientation).T @ (
        np.asarray(physical_alignment_center, dtype=float)
        - live_object_before_lift
    )

    def current_grasp_reference_center():
        live_position, live_orientation, _ = get_sim_pose(target_prim)
        return live_position + quat_to_rot_matrix(live_orientation) @ (
            grasp_reference_local
        )

    print(
        f"🧪 闭合后抓取诊断: object={live_object_before_lift}, "
        f"finger_center={get_gripper_collision_center()}, "
        f"finger_midpoint={get_gripper_finger_midpoint()}, "
        f"jaw={state.dach_arm.gripper.get_joint_positions()}"
    )
    for _ in range(5):
        hold_ee_target(grasp_position, grasp_motion_orientation)
        step_app()

    # Build one vertical lift waypoint from the settled live TCP, not the
    # pre-contact target. Keep the lift as a single waypoint so the arm does
    # not introduce an intermediate horizontal bend near the tabletop.
    lift_start_position = get_rmp_ee_position().copy()
    lift_position = lift_start_position + np.array(
        [0.0, 0.0, TRANSPORT_LIFT_HEIGHT],
        dtype=float,
    )
    lift_ok = move_ee_smooth(
        "lift",
        lift_start_position,
        lift_position,
        segments=1,
        max_steps_per_segment=50,
        tolerance=0.065,
        orientation=transport_orientation,
        gripper_positions=gripper_close_target,
        max_joint_step_rad=GRASP_LIFT_MAX_JOINT_STEP,
        minimum_frames=GRASP_LIFT_MIN_FRAMES,
        cartesian_waypoint_limit=1,
        cartesian_path_samples=20,
    )
    lifted_object_position, _, _ = get_sim_pose(target_prim)
    lifted_gripper_position = get_rmp_ee_position().copy()
    object_lift_distance = float(
        lifted_object_position[2] - initial_object_position[2]
    )
    object_horizontal_displacement = float(
        np.linalg.norm(lifted_object_position[:2] - initial_object_position[:2])
    )
    grasp_acquired = object_lift_distance >= MINIMUM_OBJECT_LIFT
    if not grasp_acquired:
        print(
            f"❌ 夹取验证失败: 物体仅抬升 {object_lift_distance:.3f} m，"
            f"要求至少 {MINIMUM_OBJECT_LIFT:.3f} m"
        )
        open_gripper_slowly(
            lift_position,
            orientation=transport_orientation,
            frames=20,
            target_open=gripper_open_target,
        )
        move_robot_home(frames=90)
        failure_result = {
            "success": False,
            "message": "grasp verification failed: object was not lifted",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "grasp_strategy": grasp_strategy,
            "phases": {
                "lift_motion": lift_ok,
                "grasp_acquired": False,
            },
            "object_lift_distance": object_lift_distance,
            "object_horizontal_displacement": object_horizontal_displacement,
            "object_position_before_close": object_position_before_close.tolist(),
            "object_position_after_close": object_position_after_close.tolist(),
            "object_position_before_lift": live_object_before_lift.tolist(),
            "object_position_after_lift": lifted_object_position.tolist(),
            "close_displacement_m": close_displacement,
            "gripper_feedback": gripper_feedback.tolist(),
            "gripper_target": gripper_close_target.tolist(),
            "gripper_efforts_n": measured_efforts.tolist(),
            "gripper_opening_width_m": close_result["opening_width_m"],
            "maximum_contact_opening_width_m": close_result[
                "maximum_contact_opening_width_m"
            ],
            "finger_contacts": close_result["finger_contacts"].tolist(),
            "grasp_containment": grasp_containment,
            "preclose_table_clearance_m": preclose_table_clearance,
            "planned_lift_position": lift_position.tolist(),
            "actual_lift_position": lifted_gripper_position.tolist(),
            "lift_minimum_frames": GRASP_LIFT_MIN_FRAMES,
            "lift_max_joint_step_rad": GRASP_LIFT_MAX_JOINT_STEP,
            "grasp_position": object_position.tolist(),
            "place_position": goal_position.tolist(),
        }
        return failure_result

    if abs(planned_transport_yaw_deg) > 1e-6:
        reorientation_start = get_rmp_ee_position().copy()
        desired_transport_orientation = rotate_grasp_about_approach_axis(
            transport_orientation,
            math.radians(planned_transport_yaw_deg),
        )
        orientation_path = interpolate_quaternions(
            transport_orientation,
            desired_transport_orientation,
            sample_count=16,
        )
        reorientation_targets = (
            state.controller.plan_pose_waypoints_with_orientations(
                [reorientation_start] * len(orientation_path),
                orientation_path,
                warm_start=get_active_joint_positions(),
            )
        )
        if reorientation_targets is None:
            move_ee_smooth(
                "transport_reorientation_abort_lower",
                reorientation_start,
                lift_start_position,
                segments=1,
                orientation=transport_orientation,
                gripper_positions=gripper_close_target,
                cartesian_waypoint_limit=1,
                cartesian_path_samples=20,
            )
            open_gripper_slowly(
                lift_start_position,
                orientation=transport_orientation,
                target_open=gripper_open_target,
            )
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "planned payload reorientation is unreachable",
                "transport_yaw_offset_deg": planned_transport_yaw_deg,
            }
        _execute_joint_trajectory(
            "payload_reorientation_at_clearance",
            reorientation_targets,
            gripper_positions=gripper_close_target,
            monitor_table_clearance=True,
            payload_prim_path=object_prim_path,
            payload_prim=target_prim,
            minimum_payload_table_clearance=0.02,
            max_joint_step_rad=0.012,
            minimum_frames=32,
        )
        transport_orientation = desired_transport_orientation
        post_rotation_containment = get_object_gripper_containment(
            object_prim_path,
            target_prim=target_prim,
            reference_center=current_grasp_reference_center(),
        )
        if not post_rotation_containment["contained"]:
            return {
                "success": False,
                "message": "payload slipped during planned wrist reorientation",
                "transport_yaw_offset_deg": planned_transport_yaw_deg,
                "grasp_containment": post_rotation_containment,
            }
        print(
            "🧭 高位负载原地转向完成: "
            f"yaw_offset={planned_transport_yaw_deg:+.1f}°"
        )
    # The tool-center offset rotates with the wrist. Recompute placement TCP
    # from the selected live transport orientation before candidate planning.
    carried_reference_center = np.asarray(
        current_grasp_reference_center(), dtype=float
    )
    carried_gripper_center = np.asarray(
        get_gripper_collision_center(), dtype=float
    )
    carried_object_gripper_offset = (
        carried_reference_center - carried_gripper_center
    )
    planned_place_gripper_center = (
        np.asarray(goal_position, dtype=float) - carried_object_gripper_offset
    )
    place_gripper_position = get_tcp_target_for_gripper_center(
        planned_place_gripper_center, transport_orientation
    )
    place_hover_position = place_gripper_position + np.array(
        [0.0, 0.0, planned_place_hover_clearance_m]
    )
    print(
        "📐 按实时负载偏移反算放置中心: "
        f"payload_offset={carried_object_gripper_offset}, "
        f"gripper_center={planned_place_gripper_center}"
    )
    base_carry_height = max(lift_position[2], place_hover_position[2])
    _, carried_bbox_min, carried_bbox_max = get_current_bbox_center(
        get_current_stage(),
        target_prim,
        object_prim_path,
    )
    target_prim_path = resolve_scene_prim_path(target_name)
    _, target_bbox_min, target_bbox_max = get_bbox_center(
        get_current_stage(),
        target_prim_path,
    )
    payload_drop_below_tcp = max(
        float(lifted_gripper_position[2] - carried_bbox_min[2]),
        0.0,
    )
    payload_carry_margin = max(BASKET_PLANNING_MARGIN, 0.02)
    payload_safe_tcp_z = (
        float(target_bbox_max[2])
        + payload_drop_below_tcp
        + payload_carry_margin
    )
    base_carry_height = max(base_carry_height, payload_safe_tcp_z)
    # Enter the placement corridor only from directly above the basket. The
    # payload is not part of Lula's robot collision model, so its live lower
    # extent must clear the basket before any horizontal crossing.
    place_hover_position[2] = max(
        float(place_hover_position[2]),
        payload_safe_tcp_z,
    )
    print(
        "📦 负载包络安全高度: "
        f"payload_bottom={carried_bbox_min[2]:.4f} m, "
        f"tcp={lifted_gripper_position[2]:.4f} m, "
        f"basket_top={target_bbox_max[2]:.4f} m, "
        f"required_tcp_z={payload_safe_tcp_z:.4f} m"
    )
    clearance_candidates = []
    for clearance in (
        0.0,
        min(CARRY_APEX_CLEARANCE, DACH_PATH_CLEARANCE),
    ):
        clearance = max(float(clearance), 0.0)
        if not any(
            abs(clearance - existing) < 1e-6
            for existing in clearance_candidates
        ):
            clearance_candidates.append(clearance)

    carry_ok = False
    carry_clearance_used = None
    carry_planned_path = None
    carry_transport_orientation = transport_orientation
    carry_path_strategy = "rrt_keypose"
    carry_planning_attempts = []
    carry_payload_containment = None
    carry_replan_count = 0
    carry_replan_events = []
    carry_max_joint_step = CARRY_MAX_JOINT_STEP
    carry_min_frames = CARRY_MIN_FRAMES
    placing_in_basket = canonical_target_name == "basket"
    placement_candidates = [goal_position.copy()]
    if placing_in_basket:
        payload_half_extents = 0.5 * (
            carried_bbox_max[:2] - carried_bbox_min[:2]
        )
        candidate_xy_values = container_place_candidates(
            target_bbox_min,
            target_bbox_max,
            payload_half_extents,
            DACH_BASE_XY,
            wall_margin_m=max(0.01, 0.5 * BASKET_PLANNING_MARGIN),
        )
        placement_candidates = []
        for candidate_xy in candidate_xy_values:
            candidate_goal = goal_position.copy()
            candidate_goal[:2] = candidate_xy
            placement_candidates.append(candidate_goal)
        print(
            "🧭 篮内固定姿态候选: "
            f"count={len(placement_candidates)}, "
            f"xy={[point[:2].tolist() for point in placement_candidates]}"
        )

    # Every transport candidate is validated against the live table and
    # basket obstacles before execution. Cartesian IK alone proves only
    # reachability and must not be used as a collision-planning substitute.
    for clearance in clearance_candidates:
        apex_z_height = base_carry_height + clearance
        source_clear_position = np.array(
            [lift_position[0], lift_position[1], apex_z_height],
            dtype=float,
        )
        for candidate_index, candidate_goal in enumerate(placement_candidates):
            candidate_offset = candidate_goal[:2] - goal_position[:2]
            candidate_place_gripper = place_gripper_position.copy()
            candidate_place_gripper[:2] += candidate_offset
            candidate_hover = place_hover_position.copy()
            candidate_hover[:2] += candidate_offset
            target_clear_position = np.array(
                [candidate_hover[0], candidate_hover[1], apex_z_height],
                dtype=float,
            )
            print(
                "🧭 尝试篮内运输候选: "
                f"index={candidate_index + 1}/{len(placement_candidates)}, "
                f"xy={candidate_goal[:2]}, "
                f"extra_clearance={clearance:.3f} m"
            )
            fallback_keyposes = []
            previous_point = lift_position
            for candidate_point in (
                source_clear_position,
                target_clear_position,
            ):
                if np.linalg.norm(candidate_point - previous_point) <= 1e-4:
                    continue
                fallback_keyposes.append(candidate_point)
                previous_point = candidate_point
            attempt = {
                "candidate_index": candidate_index,
                "candidate_goal_position": candidate_goal.tolist(),
                "extra_clearance_m": clearance,
                "keyposes": [point.tolist() for point in fallback_keyposes],
                "transport_gate": "collision_free_sparse_keyposes",
                "success": False,
            }

            planned_path = plan_collision_free_keyposes(
                fallback_keyposes,
                orientation=transport_orientation,
                maximum_approach_tilt_rad=maximum_grasp_tilt,
            )
            attempt["segments"] = list(
                getattr(state, "last_collision_free_keypose_diagnostics", [])
            )
            if planned_path is not None:
                descent_seed = planned_path[1][-1]
                descent_ik = state.controller.plan_pose_waypoints(
                    [candidate_hover, candidate_place_gripper],
                    target_orientation=transport_orientation,
                    warm_start=descent_seed,
                    allow_orientation_fallback=False,
                )
                attempt["descent_pose_ik"] = descent_ik is not None
                if descent_ik is None:
                    planned_path = None
            attempt["success"] = planned_path is not None
            carry_planning_attempts.append(attempt)
            if planned_path is None:
                continue

            carry_clearance_used = clearance
            carry_planned_path = planned_path
            goal_position = candidate_goal
            place_gripper_position = candidate_place_gripper
            place_hover_position = candidate_hover
            break
        if carry_planned_path is not None:
            break
    if carry_planned_path is not None:
        (
            carry_points,
            carry_joint_targets,
            carry_transport_orientation,
        ) = carry_planned_path
        transport_orientation = carry_transport_orientation
        print(
            "🧭 执行稀疏关键姿态 RRT 安全层: "
            f"extra_clearance={carry_clearance_used:.3f} m, "
            f"keyposes={len(carry_points)}"
        )
        # Track the fused visual object coordinate relative to the live
        # gripper-center anchor. This detects slip or a route-induced object
        # displacement without assuming a rigid attachment in simulation.
        carry_reference_object_position = np.asarray(
            current_grasp_reference_center(), dtype=float
        ).copy()
        carry_reference_gripper_center = np.asarray(
            get_gripper_collision_center(), dtype=float
        ).copy()
        carry_terminal_position = np.asarray(carry_points[-1], dtype=float)
        carry_joint_index = 0
        carry_continuation_start = None
        while carry_joint_index < len(carry_joint_targets):
            next_joint_index = min(
                carry_joint_index + CARRY_REPLAN_CHECK_WAYPOINTS,
                len(carry_joint_targets),
            )
            final_carry_chunk = next_joint_index >= len(carry_joint_targets)
            try:
                executed_chunk = _execute_joint_trajectory(
                    f"carry_clearance_route_segment_{carry_joint_index + 1}",
                    carry_joint_targets[carry_joint_index:next_joint_index],
                    gripper_positions=gripper_close_target,
                    monitor_table_clearance=True,
                    payload_prim_path=object_prim_path,
                    payload_prim=target_prim,
                    minimum_payload_table_clearance=0.02,
                    max_joint_step_rad=carry_max_joint_step,
                    minimum_frames=carry_min_frames,
                    settle_at_end=final_carry_chunk,
                    continuation_start=carry_continuation_start,
                )
            except RuntimeError as exc:
                carry_replan_events.append(
                    {
                        "event": "carry_payload_safety_stop",
                        "state": "aborted",
                        "error": str(exc),
                        "replan": False,
                    }
                )
                carry_ok = False
                break
            carry_continuation_start = np.asarray(
                executed_chunk.right_positions[-1], dtype=float
            ).copy()
            carry_joint_index = next_joint_index
            current_gripper_center = np.asarray(
                get_gripper_collision_center(), dtype=float
            )
            try:
                observed_object_position = np.asarray(
                    current_grasp_reference_center(), dtype=float
                )
                (
                    expected_object_position,
                    tracking_error,
                    tracked_object_offset,
                ) = compute_payload_tracking_state(
                    carry_reference_object_position,
                    carry_reference_gripper_center,
                    current_gripper_center,
                    observed_object_position,
                )
                tracking_error_vector = (
                    observed_object_position - expected_object_position
                )
                tracking_event = {
                    "event": "payload_transport_tracking",
                    "state": "observation",
                    "object_name": str(object_name),
                    "target_name": str(target_name),
                    "segment_end_index": carry_joint_index,
                    "segment_count": len(carry_joint_targets),
                    "observed_position": observed_object_position.tolist(),
                    "expected_position": expected_object_position.tolist(),
                    "error_m": tracking_error,
                    "vertical_error_m": float(tracking_error_vector[2]),
                    "threshold_m": CARRY_REPLAN_POSITION_TOLERANCE_M,
                    "tracking_source": "isaac_live_geometry",
                    "initial_perception_source": GRASP_BACKEND,
                    "inference_blocking_control": False,
                    "replan": False,
                }
                publish_transport_tracking(tracking_event)
            except Exception as exc:
                tracking_event = {
                    "event": "payload_transport_tracking",
                    "state": "observation_unavailable",
                    "object_name": str(object_name),
                    "target_name": str(target_name),
                    "segment_end_index": carry_joint_index,
                    "error": str(exc),
                    "tracking_source": "isaac_live_geometry",
                    "initial_perception_source": GRASP_BACKEND,
                    "inference_blocking_control": False,
                    "replan": False,
                }
                carry_replan_events.append(tracking_event)
                publish_transport_tracking(tracking_event)
                continue

            if tracking_error <= CARRY_REPLAN_POSITION_TOLERANCE_M:
                carry_replan_events.append(tracking_event)
                continue

            if carry_replan_count >= CARRY_REPLAN_MAX_ATTEMPTS:
                tracking_event["state"] = "replan_limit_reached"
                tracking_event["reason"] = "replan_attempt_limit_reached"
                carry_replan_events.append(tracking_event)
                publish_transport_tracking(tracking_event)
                carry_ok = False
                break

            carry_replan_count += 1
            tracking_event["replan"] = True
            tracking_event["state"] = "replan_requested"
            tracking_event["replan_index"] = carry_replan_count
            carry_replan_events.append(tracking_event)
            publish_transport_tracking(tracking_event)
            corrected_goal_gripper_center = (
                np.asarray(goal_position, dtype=float) - tracked_object_offset
            )
            replanned_place_gripper = get_tcp_target_for_gripper_center(
                corrected_goal_gripper_center,
                transport_orientation,
            )
            replanned_place_hover = replanned_place_gripper + np.array(
                [0.0, 0.0, planned_place_hover_clearance_m], dtype=float
            )
            replanned_place_hover[2] = max(
                float(replanned_place_hover[2]),
                float(payload_safe_tcp_z),
            )
            current_tcp = np.asarray(get_rmp_ee_position(), dtype=float).copy()
            replan_apex_z = max(
                float(payload_safe_tcp_z),
                float(current_tcp[2]),
                float(replanned_place_hover[2]),
            ) + float(carry_clearance_used or 0.0)
            replanned_target_clear = np.array(
                [
                    replanned_place_hover[0],
                    replanned_place_hover[1],
                    replan_apex_z,
                ],
                dtype=float,
            )
            replanned_route = plan_collision_free_keyposes(
                [replanned_target_clear],
                orientation=transport_orientation,
                maximum_approach_tilt_rad=maximum_grasp_tilt,
            )
            if replanned_route is None:
                tracking_event["state"] = "replan_failed"
                tracking_event["replan_failure"] = "collision_free_route_unreachable"
                publish_transport_tracking(tracking_event)
                carry_ok = False
                break
            descent_seed = replanned_route[1][-1]
            replanned_descent = state.controller.plan_pose_waypoints(
                [replanned_place_hover, replanned_place_gripper],
                target_orientation=transport_orientation,
                warm_start=descent_seed,
                allow_orientation_fallback=False,
            )
            if replanned_descent is None:
                tracking_event["state"] = "replan_failed"
                tracking_event["replan_failure"] = "placement_descent_unreachable"
                publish_transport_tracking(tracking_event)
                carry_ok = False
                break
            # Carry only to the collision-cleared point above the basket.
            # The independently monitored basket corridor performs the hover
            # and placement descent after the basket obstacle is disabled.
            carry_joint_targets = list(replanned_route[1])
            carry_joint_index = 0
            current_joint_positions = get_active_joint_positions()
            carry_continuation_start = (
                None
                if current_joint_positions is None
                else np.asarray(current_joint_positions, dtype=float).copy()
            )
            carry_terminal_position = np.asarray(
                replanned_target_clear, dtype=float
            )
            place_gripper_position = replanned_place_gripper
            place_hover_position = replanned_place_hover
            carry_points = [
                current_tcp,
                replanned_target_clear,
            ]
            carry_reference_object_position = observed_object_position.copy()
            carry_reference_gripper_center = current_gripper_center.copy()
            carry_path_strategy = "rrt_keypose+isaac_live_geometry_replan"
            tracking_event["state"] = "replan_succeeded"
            tracking_event["replanned_waypoint_count"] = len(carry_joint_targets)
            publish_transport_tracking(tracking_event)

        if carry_joint_index >= len(carry_joint_targets):
            carry_error = float(
                np.linalg.norm(get_rmp_ee_position() - carry_terminal_position)
            )
            carry_ok = carry_error <= 0.08
        if carry_ok:
            carry_payload_containment = get_object_gripper_containment(
                object_prim_path,
                target_prim=target_prim,
            )
            carry_feedback = np.asarray(
                state.dach_arm.gripper.get_joint_positions(),
                dtype=float,
            )
            carry_residuals = np.maximum(
                carry_feedback - gripper_close_target,
                0.0,
            )
            carry_efforts = get_gripper_joint_efforts()
            carry_finger_contacts = classify_finger_contacts(
                carry_feedback,
                gripper_close_target,
                carry_efforts,
                residual_threshold=GRIPPER_CONTACT_PRELOAD_RESIDUAL,
                force_threshold=GRIPPER_CONTACT_FORCE_THRESHOLD,
            )
            carry_contact_stability = evaluate_grasp_stability(
                carry_finger_contacts,
                carry_efforts,
                carry_residuals,
                force_threshold=GRIPPER_CONTACT_FORCE_THRESHOLD,
                residual_threshold=GRIPPER_CONTACT_PRELOAD_RESIDUAL,
            )
            carry_contact_confirmed = bool(carry_contact_stability["stable"])
            carry_geometry_retained = bool(
                abs(carry_payload_containment["axial_error_m"])
                <= carry_payload_containment["axial_limit_m"]
                and carry_payload_containment["radial_error_m"]
                <= carry_payload_containment["radial_limit_m"]
            )
            carry_payload_containment.update(
                {
                    "retained": bool(
                        carry_geometry_retained and carry_contact_confirmed
                    ),
                    "geometry_retained": carry_geometry_retained,
                    "contact_confirmed": carry_contact_confirmed,
                    "gripper_feedback": carry_feedback.tolist(),
                    "gripper_target": gripper_close_target.tolist(),
                    "gripper_residuals_m": carry_residuals.tolist(),
                    "gripper_efforts_n": carry_efforts.tolist(),
                    "finger_contacts": carry_finger_contacts.tolist(),
                    "contact_stability": carry_contact_stability,
                }
            )
            carry_ok = bool(carry_payload_containment["retained"])
            if not carry_ok:
                print(
                    "🛑 持物运输后负载已脱离夹指："
                    f"{carry_payload_containment}"
                )
    if not carry_ok:
        abort_position = get_rmp_ee_position()
        returned_to_source = move_ee_smooth(
            "carry_abort_vertical_return",
            abort_position,
            lift_start_position,
            segments=1,
            tolerance=0.05,
            orientation=transport_orientation,
            gripper_positions=gripper_close_target,
            monitor_table_clearance=True,
            max_joint_step_rad=GRASP_LIFT_MAX_JOINT_STEP,
            minimum_frames=GRASP_LIFT_MIN_FRAMES,
            cartesian_waypoint_limit=1,
            cartesian_path_samples=20,
        )
        if returned_to_source:
            open_gripper_slowly(
                lift_start_position,
                orientation=transport_orientation,
                frames=20,
                target_open=gripper_open_target,
            )
            abort_retreat_position = lift_start_position + np.array(
                [0.0, 0.0, 0.15]
            )
            move_ee_smooth(
                "carry_abort_retreat",
                lift_start_position,
                abort_retreat_position,
                segments=1,
                tolerance=0.05,
                orientation=transport_orientation,
                gripper_positions=gripper_open_target,
            )
        move_robot_home(frames=90)
        if carry_payload_containment is not None:
            carry_failure_message = "payload retention check failed during carry"
        elif carry_planned_path is not None:
            carry_failure_message = "carry trajectory execution failed"
        else:
            carry_failure_message = "collision-clearance carry path is unreachable"
        return {
            "success": False,
            "message": carry_failure_message,
            "object_name": str(object_name),
            "target_name": str(target_name),
            "returned_to_source_before_release": returned_to_source,
            "payload_safe_tcp_z_m": payload_safe_tcp_z,
            "carry_planning_attempts": carry_planning_attempts,
            "carry_payload_containment": carry_payload_containment,
            "carry_replan_count": carry_replan_count,
            "carry_replan_events": carry_replan_events,
            "grasp_fusion": grasp_fusion,
        }
    # Re-measure the friction-held payload after transport. Small settling
    # offsets are expected for a near-aperture-width can and must be included
    # when converting the desired object center into the final gripper TCP.
    settled_payload_center = np.asarray(
        current_grasp_reference_center(), dtype=float
    )
    settled_gripper_center = np.asarray(
        get_gripper_collision_center(), dtype=float
    )
    settled_payload_offset = settled_payload_center - settled_gripper_center
    settled_place_gripper_center = (
        np.asarray(goal_position, dtype=float) - settled_payload_offset
    )
    place_gripper_position = get_tcp_target_for_gripper_center(
        settled_place_gripper_center,
        transport_orientation,
    )
    place_hover_position = place_gripper_position + np.array(
        [0.0, 0.0, planned_place_hover_clearance_m],
        dtype=float,
    )
    place_hover_position[2] = max(
        float(place_hover_position[2]),
        float(payload_safe_tcp_z),
    )
    if placing_in_basket and not set_planning_basket_obstacle_enabled(False):
        return {
            "success": False,
            "message": "failed to open basket placement corridor",
            "object_name": str(object_name),
            "target_name": str(target_name),
        }
    if placing_in_basket:
        corridor_entry_ok = move_ee_smooth(
            "basket_vertical_corridor_entry",
            get_rmp_ee_position(),
            place_hover_position,
            segments=1,
            max_steps_per_segment=45,
            tolerance=0.025,
            orientation=transport_orientation,
            gripper_positions=gripper_close_target,
            monitor_table_clearance=True,
            max_joint_step_rad=0.016,
            minimum_frames=30,
            cartesian_waypoint_limit=1,
        )
        if not corridor_entry_ok:
            set_planning_basket_obstacle_enabled(True)
            return {
                "success": False,
                "message": "failed to enter basket vertical placement corridor",
                "object_name": str(object_name),
                "target_name": str(target_name),
            }
    if placing_in_basket and state.planning_table_surface_z is not None:
        current_tcp_z = float(get_rmp_ee_position()[2])
        current_clearance = float(get_gripper_table_clearance())
        planned_descent = current_tcp_z - float(place_gripper_position[2])
        predicted_clearance = current_clearance - planned_descent
        if predicted_clearance < BASKET_PLACE_TABLE_CLEARANCE:
            clearance_lift = (
                BASKET_PLACE_TABLE_CLEARANCE - predicted_clearance
            )
            place_gripper_position[2] += clearance_lift
            print(
                "🛡️ 篮筐放置点桌面净空保护: "
                f"predicted={predicted_clearance:.4f} m, "
                f"target={BASKET_PLACE_TABLE_CLEARANCE:.4f} m, "
                f"raise={clearance_lift:.4f} m"
            )
    lower_ok = move_ee_smooth(
        "lower",
        place_hover_position,
        place_gripper_position,
        segments=1,
        max_steps_per_segment=45,
        tolerance=0.025,
        orientation=transport_orientation,
        gripper_positions=gripper_close_target,
        monitor_table_clearance=True,
        max_joint_step_rad=0.016,
        minimum_frames=30,
    )
    if not lower_ok:
        # Retry the low-speed descent with the same transport orientation. A
        # position-only retry would let Lula flip the wrist at the basket.
        print("↪️ 放置下探固定姿态未收敛，使用同姿态低速下探重试")
        lower_ok = move_ee_smooth(
            "lower_position_first_fallback",
            get_rmp_ee_position(),
            place_gripper_position,
            segments=1,
            max_steps_per_segment=55,
            tolerance=0.025,
            orientation=transport_orientation,
            gripper_positions=gripper_close_target,
            monitor_table_clearance=True,
            max_joint_step_rad=0.016,
            minimum_frames=30,
        )
    # Releasing while the descent missed its target drops the object wherever
    # the gripper happens to be.  Report the miss instead of opening the jaw --
    # a silent release here is what made a 17.6 cm placement error look like a
    # success.
    if not lower_ok:
        lower_error = float(
            np.linalg.norm(get_rmp_ee_position() - place_gripper_position)
        )
        print(
            f"❌ 下降到放置点失败: error={lower_error:.4f} m, "
            f"未松爪以避免把物体丢在半路"
        )
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "failed to lower object into target region",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "lower_error_m": lower_error,
        }
    release_position = get_rmp_ee_position().copy()
    lower_error = float(np.linalg.norm(release_position - place_gripper_position))
    if placing_in_basket:
        payload_center, payload_bbox_min, payload_bbox_max = (
            get_current_bbox_center(
                get_current_stage(), target_prim, object_prim_path
            )
        )
        _, basket_bbox_min, basket_bbox_max = get_bbox_center(
            get_current_stage(), resolve_scene_prim_path(target_name)
        )
        basket_inset = 0.005
        payload_inside_opening = bool(
            np.all(payload_bbox_min[:2] >= basket_bbox_min[:2] + basket_inset)
            and np.all(payload_bbox_max[:2] <= basket_bbox_max[:2] - basket_inset)
        )
        if not payload_inside_opening:
            return {
                "success": False,
                "message": "payload is not inside basket opening; release blocked",
                "object_name": str(object_name),
                "target_name": str(target_name),
                "lower_error_m": lower_error,
                "payload_center": payload_center.tolist(),
                "payload_bbox_min": payload_bbox_min.tolist(),
                "payload_bbox_max": payload_bbox_max.tolist(),
                "basket_bbox_min": basket_bbox_min.tolist(),
                "basket_bbox_max": basket_bbox_max.tolist(),
            }
    # Release at the validated placement pose. Raising a still-gripped object
    # before opening removes it from the basket and turns placement into an
    # uncontrolled drop. The selected basket candidate includes wall margin
    # for opening the physical fingers in place.
    release_clear_position = release_position + np.array([0.0, 0.0, 0.12])
    open_gripper_slowly(
        release_position,
        orientation=transport_orientation,
        frames=30,
        target_open=gripper_open_target,
    )
    # Make the release observable in PhysX as well as in the command stream:
    # the slow controller can finish one frame short of its target when the
    # wrist is settling. Hold the jaws at their calibrated open limit for a
    # few physics frames before measuring placement.
    state.dach_arm.gripper.set_joint_positions(gripper_open_target)
    step_app(5)
    release_clear_ok = move_ee_smooth(
        "release_vertical_retreat",
        get_rmp_ee_position(),
        release_clear_position,
        segments=1,
        max_steps_per_segment=35,
        tolerance=0.04,
        orientation=transport_orientation,
        gripper_positions=gripper_open_target,
        monitor_table_clearance=True,
        max_joint_step_rad=0.016,
        minimum_frames=30,
        cartesian_waypoint_limit=1,
        cartesian_path_samples=20,
    )
    if not release_clear_ok:
        print("⚠️ 释放后垂直撤离未完全收敛")
    # The object remains a free dynamic rigid body while the robot retreats.
    # Let it settle against the basket floor before measuring containment.
    step_app(120)
    release_feedback = np.asarray(
        state.dach_arm.gripper.get_joint_positions(), dtype=float
    )
    release_error = float(np.max(np.abs(release_feedback - gripper_open_target)))
    release_verified = bool(release_error <= 0.005)
    print(
        f"👐 放置释放校验: feedback={release_feedback}, "
        f"target={gripper_open_target}, error={release_error:.4f}, "
        f"verified={release_verified}"
    )
    final_object_position, _, _ = get_sim_pose(target_prim)
    placement_error = float(
        np.linalg.norm(final_object_position[:2] - goal_position[:2])
    )
    placement_height_ok = True
    basket_containment = None
    if placing_in_basket:
        final_mesh_center, _, _ = get_current_bbox_center(
            get_current_stage(), target_prim, object_prim_path
        )
        basket_prim_path = resolve_scene_prim_path(target_name)
        _, basket_bbox_min, basket_bbox_max = get_bbox_center(
            get_current_stage(),
            basket_prim_path,
        )
        xy_inset = np.minimum(
            0.01,
            0.1 * (basket_bbox_max[:2] - basket_bbox_min[:2]),
        )
        xy_inside = bool(
            np.all(final_mesh_center[:2] >= basket_bbox_min[:2] + xy_inset)
            and np.all(final_mesh_center[:2] <= basket_bbox_max[:2] - xy_inset)
        )
        z_inside = bool(
            basket_bbox_min[2] - 0.01
            <= final_mesh_center[2]
            <= basket_bbox_max[2] + 0.05
        )
        basket_containment = xy_inside and z_inside
        placement_height_ok = basket_containment
        print(
            f"🧪 篮筐三维包含校验: object_mesh_center={final_mesh_center}, "
            f"basket_min={basket_bbox_min}, basket_max={basket_bbox_max}, "
            f"xy_inside={xy_inside}, z_inside={z_inside}, "
            f"contained={basket_containment}"
        )
    elif state.planning_table_surface_z is not None:
        # For non-container targets, reject objects that merely landed beside
        # the requested point at their original table height.
        object_height_above_table = float(
            final_object_position[2] - state.planning_table_surface_z
        )
        placement_height_ok = (
            object_height_above_table >= PLACE_MIN_HEIGHT_ABOVE_TABLE
        )
        print(
            f"🧪 放置高度校验: object_z={final_object_position[2]:.5f}, "
            f"table_z={state.planning_table_surface_z:.5f}, "
            f"above_table={object_height_above_table * 1000:.1f} mm, "
            f"要求>={PLACE_MIN_HEIGHT_ABOVE_TABLE * 1000:.1f} mm, "
            f"ok={placement_height_ok}"
        )
    placement_ok = (
        placement_error <= PLACE_SUCCESS_TOLERANCE
    ) and placement_height_ok
    retreat_position = release_position + np.array([0.0, 0.0, 0.15])
    retreat_ok = move_ee_smooth(
        "retreat_after_place",
        get_rmp_ee_position(),
        retreat_position,
        segments=1,
        max_steps_per_segment=50,
        tolerance=0.02,
        orientation=transport_orientation,
        gripper_positions=gripper_open_target,
        max_joint_step_rad=0.016,
        minimum_frames=30,
    )
    if placing_in_basket:
        if not retreat_ok:
            return {
                "success": False,
                "message": "failed to retreat above basket after placement",
                "object_name": str(object_name),
                "target_name": str(target_name),
            }
        if not set_planning_basket_obstacle_enabled(True):
            return {
                "success": False,
                "message": "failed to restore basket collision obstacle",
                "object_name": str(object_name),
                "target_name": str(target_name),
            }
    move_robot_home(frames=90)

    # Re-check after the arm has fully retreated and returned HOME.  The
    # released can must remain contained after all robot motion, not only at
    # the instant immediately following jaw opening.
    if placing_in_basket:
        step_app(60)
        final_object_position, _, _ = get_sim_pose(target_prim)
        final_mesh_center, _, _ = get_current_bbox_center(
            get_current_stage(), target_prim, object_prim_path
        )
        _, basket_bbox_min, basket_bbox_max = get_bbox_center(
            get_current_stage(), resolve_scene_prim_path(target_name)
        )
        xy_inset = np.minimum(
            0.01,
            0.1 * (basket_bbox_max[:2] - basket_bbox_min[:2]),
        )
        basket_containment = bool(
            np.all(final_mesh_center[:2] >= basket_bbox_min[:2] + xy_inset)
            and np.all(final_mesh_center[:2] <= basket_bbox_max[:2] - xy_inset)
            and basket_bbox_min[2] - 0.01
            <= final_mesh_center[2]
            <= basket_bbox_max[2] + 0.05
        )
        placement_error = float(
            np.linalg.norm(final_object_position[:2] - goal_position[:2])
        )
        placement_ok = (
            placement_error <= PLACE_SUCCESS_TOLERANCE
            and basket_containment
        )
        print(
            f"🧪 归位后篮筐保持校验: object_mesh_center={final_mesh_center}, "
            f"placement_error={placement_error:.4f}m, "
            f"contained={basket_containment}"
        )

    success = grasp_acquired and placement_ok and release_verified
    result = {
        "success": success,
        "message": "pick and place completed" if success else "object did not reach target region",
        "object_name": str(object_name),
        "target_name": str(target_name),
        "selected_arm": reachability.get("selected_arm"),
        "grasp_strategy": grasp_strategy,
        "grasp_approach": approach_result,
        "path_strategy": carry_path_strategy,
        "cartesian_waypoint_spacing_m": CARTESIAN_WAYPOINT_SPACING,
        "carry_cartesian_waypoint_spacing_m": CARRY_CARTESIAN_WAYPOINT_SPACING,
        "carry_max_joint_step_rad": carry_max_joint_step,
        "carry_min_frames": carry_min_frames,
        "transport_lift_height_m": TRANSPORT_LIFT_HEIGHT,
        "carry_extra_clearance_m": carry_clearance_used,
        "carry_replan_count": carry_replan_count,
        "carry_replan_events": carry_replan_events,
        "transport_yaw_offset_deg": planned_transport_yaw_deg,
        "phases": {
            "lift_motion": lift_ok,
            "grasp_acquired": grasp_acquired,
            "carry_motion": carry_ok,
            "lower_motion": lower_ok,
            "placement": placement_ok,
            "release": release_verified,
        },
        "release_verified": release_verified,
        "object_lift_distance": object_lift_distance,
        "object_horizontal_displacement": object_horizontal_displacement,
        "placement_error": placement_error,
        "basket_containment": basket_containment,
        "final_object_position": final_object_position.tolist(),
        "grasp_position": object_position.tolist(),
        "gripper_grasp_position": grasp_position.tolist(),
        "place_position": goal_position.tolist(),
        "task_duration_sec": round(time.perf_counter() - task_started, 3),
        "perception_source": perception_source,
        "grasp_fusion": grasp_fusion,
    }
    return result


def snapshot_scene_object_pose(object_name):
    prim_path = resolve_scene_prim_path(object_name)
    target_prim = SingleXFormPrim(
        name=f"aura_snapshot_{state.SCENE_NAME_RESOLVER.canonicalize(object_name)}",
        prim_path=prim_path,
    )
    position, orientation, _ = get_sim_pose(target_prim)
    return {
        "prim_path": prim_path,
        "position": position.copy(),
        "orientation": orientation.copy(),
    }


def restore_scene_object_pose(snapshot):
    if snapshot is None:
        return
    prim_path = snapshot["prim_path"]
    root = get_current_stage().GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    rigid_body = UsdPhysics.RigidBodyAPI(root)
    kinematic_attr = rigid_body.CreateKinematicEnabledAttr()
    previous_kinematic = bool(kinematic_attr.Get())
    kinematic_attr.Set(True)
    target_prim = SingleXFormPrim(
        name=f"aura_restore_{root.GetName()}",
        prim_path=prim_path,
    )
    target_prim.set_world_pose(
        position=snapshot["position"],
        orientation=snapshot["orientation"],
    )
    step_app(3)
    kinematic_attr.Set(previous_kinematic)
    step_app(2)
    print(f"↩️ 失败任务已恢复目标物体位姿: {prim_path}")
