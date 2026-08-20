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
    BANANA_GRASP_TILT_RAD, BANANA_NEAR_SIDE_OFFSET,
    BANANA_MIN_SHORT_AXIS_ALIGNMENT,
    DACH_GRASP_HEIGHT_OFFSET, DACH_GRASP_YAW_OFFSET_RAD,
    MAX_GRASP_APPROACH_TILT_RAD, TARGET_GRASP_APPROACH_TILT_RAD,
    BANANA_USE_SIMULATED_ATTACHMENT,
    ALLOW_OVERSIZED_CAN_GRASP, LARGE_CAN_USE_SIMULATED_ATTACHMENT,
    LARGE_CAN_CLOSED_JAW_CLEARANCE_LIFT,
    GRASP_REFINEMENT_STEPS, GRIPPER_CLOSE_FRAMES,
    GRIPPER_MAX_EFFORT, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    GRIPPER_CONTACT_RESIDUAL, GRIPPER_CONTACT_FORCE_THRESHOLD,
    GRIPPER_CONTACT_PRELOAD_RESIDUAL, GRIPPER_CONTACT_HOLD_PRELOAD,
    GRIPPER_CONTACT_SETTLE_FRAMES,
    BANANA_GRIPPER_CLOSE_POSITION,
    BANANA_PLANAR_REFINEMENT_STEPS, BANANA_PLANAR_CENTER_TOLERANCE,
    BANANA_MAX_PLANAR_CORRECTION,
    DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET,
    DACH_JAW_COLLISION_LOCAL_BOUNDS,
    DIRECTIONAL_PLACE_DISTANCE,
    GRASP_MIN_HEIGHT_FRACTION, MINIMUM_OBJECT_LIFT,
    PLACE_SUCCESS_TOLERANCE, PLACE_MIN_HEIGHT_ABOVE_TABLE,
    DACH_PATH_CLEARANCE,
    TRAJECTORY_MAX_JOINT_STEP, TRAJECTORY_MIN_FRAMES, TRAJECTORY_SETTLE_FRAMES,
    GRASP_APPROACH_MAX_JOINT_STEP, GRASP_APPROACH_MIN_FRAMES,
    CARTESIAN_WAYPOINT_SPACING, CARRY_APEX_CLEARANCE,
    TRANSPORT_LIFT_HEIGHT, CARRY_CARTESIAN_WAYPOINT_SPACING,
    SHOW_GRASP_DEBUG, USE_GRASPNET, USE_GRASPNET_ORIENTATION,
    GRASP_POSITION_OFFSET, GRASP_INSERT_DEPTH,
    MIN_GRIPPER_TABLE_CLEARANCE, TABLE_CLEARANCE_ABORT_MARGIN,
    BASKET_RESET_POSITION, BASKET_RESET_ORIENTATION,
    BASKET_PLANNING_MARGIN, BASKET_PLACE_TABLE_CLEARANCE,
    DUAL_ARM_MIN_TCP_SEPARATION,
)
from aura_isaac_bridge.core.physics import step_app, ensure_pickable_object
from aura_isaac_bridge.core.perception import (
    get_sim_pose,
    quat_rotate,
    resolve_scene_prim_path,
    get_bbox_center,
    get_mesh_center,
    get_mesh_extent_along_axis,
    get_mesh_horizontal_principal_axes,
    show_red_grasp_point,
    release_cuda_inference_cache,
    infer_graspnet_world_pose,
)
from aura_isaac_bridge.core.motion import (
    freeze_object_for_pregrasp,
    release_pregrasp_object,
    attach_simulated_object,
    detach_simulated_object,
    release_detached_object,
    update_simulated_attachment,
    get_active_joint_positions,
    get_left_joint_positions,
    ensure_robot_control_ready,
    get_rmp_ee_position,
    move_ee_to,
    move_ee_waypoints,
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
    get_gripper_closing_axis,
    get_gripper_inner_opening_width,
    get_gripper_center_local_offset,
    get_tcp_target_for_gripper_center,
    verify_gripper_center_local_offset,
    _execute_joint_trajectory,
)




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
                if str(object_path).rstrip("/").endswith("/banana"):
                    basket_floor_z = (
                        float(state.planning_table_surface_z)
                        if state.planning_table_surface_z is not None
                        else float(target_bbox_min[2])
                    )
                    place_position[2] = (
                        basket_floor_z + object_half_height + 0.003
                    )
                    print(
                        "📦 香蕉使用低落差篮筐放置高度: "
                        f"floor_z={basket_floor_z:.4f}, "
                        f"half_height={object_half_height:.4f}, "
                        f"center_z={place_position[2]:.4f}"
                    )
            except Exception as exc:
                print(f"⚠️ 无法计算篮筐物体高度余量，使用默认放置高度: {exc}")
    print(
        f"📦 放置目标 {target_name}: path={target_prim_path}, "
        f"bbox=({target_bbox_min}, {target_bbox_max})"
    )
    return place_position


def get_top_down_grasp_orientation(object_name, target_prim, tilt_override=None):
    canonical_name = state.SCENE_NAME_RESOLVER.canonicalize(object_name)
    tilt = (
        BANANA_GRASP_TILT_RAD
        if tilt_override is None and canonical_name == "banana"
        else float(tilt_override or 0.0)
    )

    object_long_axis, object_short_axis = get_mesh_horizontal_principal_axes(
        get_current_stage(),
        target_prim.prim_path,
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
    if canonical_name == "banana":
        # The physical jaw closing axis is TCP -Y.  Tilting around an arbitrary
        # base-to-object vector mixes the requested short axis into TCP Y; the
        # IK fallback can then reach the point with a visibly rolled wrist but
        # the fingers no longer straddle the banana.  Use the PCA long axis as
        # the only horizontal tilt direction and select its outward-facing
        # sign so the long TCP offset remains on the robot side of the object.
        tilt_axis = np.asarray(object_long_axis, dtype=float).copy()
        if inward_axis is not None and np.dot(tilt_axis, inward_axis) < 0.0:
            tilt_axis *= -1.0
    else:
        # Preserve the already validated can/non-banana approach geometry.
        tilt_axis = object_long_axis if inward_axis is None else inward_axis
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
        f"🧭 {canonical_name} 网格 PCA: long={object_long_axis}, "
        f"short={object_short_axis}; 夹爪沿短轴闭合, "
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


def get_gripper_close_target(object_name):
    return np.array(
        [BANANA_GRIPPER_CLOSE_POSITION, BANANA_GRIPPER_CLOSE_POSITION],
        dtype=float,
    )


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
    canonical_name = state.SCENE_NAME_RESOLVER.canonicalize(object_name)
    object_prim_path = state.SCENE_NAME_RESOLVER.prim_candidates(object_name)[0]
    object_long_axis, object_short_axis = get_mesh_horizontal_principal_axes(
        get_current_stage(),
        object_prim_path,
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
    if (
        canonical_name == "banana"
        and DACH_BASE_XY is not None
        and BANANA_NEAR_SIDE_OFFSET > 0.0
    ):
        direction_to_base = np.asarray(DACH_BASE_XY, dtype=float) - mesh_center[:2]
        direction_norm = float(np.linalg.norm(direction_to_base))
        if direction_norm > 1e-9:
            adjusted_position[:2] += (
                direction_to_base / direction_norm * BANANA_NEAR_SIDE_OFFSET
            )
    print(
        f"🎯 {canonical_name} GraspNet/场景抓取点: "
        f"mesh_center_xy={mesh_center[:2]}, "
        f"bbox_center_xy={bbox_center[:2]}, "
        f"source_xy={source_xy}, "
        f"target_xy={adjusted_position[:2]}, "
        f"long_axis_offset={long_axis_offset:.4f} m, "
        f"near_side_offset={BANANA_NEAR_SIDE_OFFSET:.3f} m"
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
    approach_direction = (
        quat_rotate(orientation, np.array([1.0, 0.0, 0.0]))
        if canonical_name == "banana"
        else np.array([0.0, 0.0, -1.0], dtype=float)
    )
    approach_direction = np.asarray(approach_direction, dtype=float)
    approach_direction /= max(float(np.linalg.norm(approach_direction)), 1e-9)
    gate_orientation = orientation
    by_side = {}
    hover_by_side = {}
    hover_plan_by_side = {}
    for side in ("right", "left"):
        controller = (getattr(state, "arm_controllers", None) or {}).get(side)
        arm = (getattr(state, "arm_views", None) or {}).get(side)
        side_ok = False
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
                if joint_targets is not None:
                    side_ok = True
                    hover_by_side[side] = hover.copy()
                    hover_plan_by_side[side] = [
                        np.asarray(joint_targets[0], dtype=float).copy()
                    ]
                    break
        by_side[side] = side_ok
    preferred_side = _preferred_arm_side_for_position(grasp_position)
    selected = None
    if by_side.get(preferred_side, False):
        selected = preferred_side
    elif by_side.get("right", False):
        selected = "right"
    elif by_side.get("left", False):
        selected = "left"
    return {
        "right": by_side.get("right", False),
        "left": by_side.get("left", False),
        "selected": selected,
        "preferred": preferred_side,
        "hover_position": None if selected is None else hover_by_side[selected],
        "hover_plan": None if selected is None else hover_plan_by_side[selected],
    }


def execute_pick_place(object_name, target_name):
    state._task_motion_started = False
    task_started = time.perf_counter()

    canonical_object_name = state.SCENE_NAME_RESOLVER.canonicalize(object_name)
    large_can_mode = (
        canonical_object_name in {"master_chef_can", "tomato_soup_can"}
        and ALLOW_OVERSIZED_CAN_GRASP
    )
    object_prim_path = resolve_scene_prim_path(object_name)
    state.TARGET_OBJECT_PRIM_PATH = object_prim_path
    target_prim = SingleXFormPrim(
        name=f"aura_target_{str(object_name).lower()}",
        prim_path=object_prim_path,
    )
    if not target_prim.is_valid():
        raise RuntimeError(f"抓取目标无效: {object_prim_path}")

    canonical_target_name = state.SCENE_NAME_RESOLVER.canonicalize(target_name)
    release_pregrasp_object()
    detach_simulated_object()
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
    _, precheck_bbox_min, precheck_bbox_max = get_bbox_center(
        get_current_stage(),
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
    state.dach_arm.gripper.set_joint_positions(gripper_open_target)
    step_app(10)
    ensure_pickable_object(get_current_stage(), object_prim_path)
    freeze_object_for_pregrasp(object_prim_path)
    state.dach_arm.gripper.set_joint_positions(gripper_open_target)
    step_app(10)

    top_down_orientation = get_top_down_grasp_orientation(object_name, target_prim)
    gripper_close_target = get_gripper_close_target(object_name)
    graspnet_inference_error = None
    graspnet_position_active = False
    perception_started = time.perf_counter()
    if USE_GRASPNET:
        try:
            graspnet_position, graspnet_orientation = infer_graspnet_world_pose(
                get_current_stage(),
                state.grasp_camera,
                target_prim,
            )
        except Exception as exc:
            release_cuda_inference_cache()
            graspnet_inference_error = f"{type(exc).__name__}: {exc}"
            print(f"⚠️ GraspNet 推理失败，直接使用场景位姿顶视抓取: {graspnet_inference_error}")
            object_position, _, _ = get_sim_pose(target_prim)
            grasp_orientation = top_down_orientation
            grasp_strategy = "scene_pose_top_down"
        else:
            release_cuda_inference_cache()
            # GraspNet's calibrated position is now the sole grasp-center
            # source for every object.  Keep the top-down/PCA orientation
            # because its physical jaw-axis constraint is independent of the
            # detector's camera-frame wrist roll.
            object_position = np.asarray(graspnet_position, dtype=float).copy()
            grasp_orientation = top_down_orientation
            graspnet_position_active = True
            grasp_strategy = "graspnet_calibrated_position_top_down"
            print(
                "🎯 使用 GraspNet + 相机标定 offset 抓取点: "
                f"position={object_position}"
            )
    else:
        object_position, _, _ = get_sim_pose(target_prim)
        grasp_orientation = top_down_orientation
        grasp_strategy = "scene_pose_top_down"
    print(
        f"⏱️ SAM + GraspNet: "
        f"{time.perf_counter() - perception_started:.2f} s"
    )

    bbox_center, bbox_min, bbox_max = get_bbox_center(get_current_stage(), object_prim_path)
    if not graspnet_position_active and canonical_object_name != "banana":
        object_position = np.asarray(bbox_center, dtype=float).copy()
        print(
            "🎯 非香蕉物体使用 USD 几何包围盒中心: "
            f"center={object_position}"
        )
    if not graspnet_position_active:
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
        if canonical_object_name == "banana":
            banana_center_offset = np.zeros(2, dtype=float)
            if DACH_BASE_XY is not None and BANANA_NEAR_SIDE_OFFSET > 0.0:
                direction_to_base = (
                    np.asarray(DACH_BASE_XY, dtype=float)
                    - np.asarray(bbox_center[:2], dtype=float)
                )
                direction_norm = float(np.linalg.norm(direction_to_base))
                if direction_norm > 1e-9:
                    banana_center_offset = (
                        direction_to_base / direction_norm * BANANA_NEAR_SIDE_OFFSET
                    )
            object_position[:2] = (
                np.asarray(bbox_center[:2], dtype=float) + banana_center_offset
            )
        object_position[2] += DACH_GRASP_HEIGHT_OFFSET
    grasp_target_center = np.asarray(object_position, dtype=float).copy()
    # GraspNet + the one global camera-frame offset remains the grasp source.
    # Its selected pinch point can still carry a small object-specific lateral
    # residual.  Use the live collision geometry only as a bounded final
    # containment guard, so closing never begins with the object beside a jaw.
    physical_alignment_center = grasp_target_center.copy()
    if graspnet_position_active:
        containment_xy_error = (
            np.asarray(bbox_center[:2], dtype=float)
            - physical_alignment_center[:2]
        )
        containment_xy_error_norm = float(np.linalg.norm(containment_xy_error))
        bbox_center_array = np.asarray(bbox_center, dtype=float)
        bbox_min_array = np.asarray(bbox_min, dtype=float)
        bbox_max_array = np.asarray(bbox_max, dtype=float)
        graspnet_vertical_margin = max(
            0.01,
            0.15 * float(bbox_max_array[2] - bbox_min_array[2]),
        )
        graspnet_z_in_bounds = (
            bbox_min_array[2] - graspnet_vertical_margin
            <= physical_alignment_center[2]
            <= bbox_max_array[2] + graspnet_vertical_margin
        )
        if containment_xy_error_norm <= 0.04 and graspnet_z_in_bounds:
            physical_alignment_center[:2] = bbox_center[:2]
            print(
                "📐 GraspNet 抓取点物理包含校正: "
                f"xy_error={containment_xy_error}, "
                f"norm={containment_xy_error_norm:.4f} m"
            )
        else:
            print(
                "⚠️ GraspNet 点未通过几何一致性检查，"
                f"xy_error={containment_xy_error_norm:.4f} m, "
                f"z_in_bounds={graspnet_z_in_bounds}; "
                "回退到 USD 包围盒中心"
            )
            # A detector point outside the object's collision geometry is not
            # a valid grasp center. Use the bounded scene fallback, including
            # the configured lift offset used by the Aura scene-pose path.
            graspnet_position_active = False
            object_position = bbox_center_array.copy()
            minimum_grasp_height = bbox_min_array[2] + (
                bbox_max_array[2] - bbox_min_array[2]
            ) * GRASP_MIN_HEIGHT_FRACTION
            object_position[2] = max(object_position[2], minimum_grasp_height)
            object_position = adjust_object_grasp_position(
                object_name,
                object_position,
                bbox_min_array,
                bbox_max_array,
            )
            object_position[2] += DACH_GRASP_HEIGHT_OFFSET
            grasp_target_center = object_position.copy()
    final_grasp_tcp = get_tcp_target_for_gripper_center(
        object_position,
        grasp_orientation,
    )
    final_pose_reachability = _evaluate_arm_pose_reachability(
        object_name,
        final_grasp_tcp,
        grasp_orientation,
    )
    if canonical_object_name == "banana":
        desired_short_axis = get_mesh_horizontal_principal_axes(
            get_current_stage(),
            object_prim_path,
        )[1]

        def banana_orientation_alignment(orientation):
            tcp_closing_axis = quat_to_rot_matrix(orientation)[:2, 1]
            axis_norm = float(np.linalg.norm(tcp_closing_axis))
            if axis_norm < 1e-9:
                return 0.0
            return float(
                abs(
                    np.dot(
                        tcp_closing_axis / axis_norm,
                        np.asarray(desired_short_axis, dtype=float)[:2],
                    )
                )
            )

        planned_alignment = banana_orientation_alignment(grasp_orientation)
        # Prefer the zero-tilt branch whenever it is strictly reachable. It
        # keeps both jaw collision bottoms level and avoids the table-induced
        # lateral impulse observed at the 7° edge approach. Tilt is retained
        # only as a reachability fallback for scenes where top-down IK fails.
        if canonical_object_name == "banana":
            configured_tilt_deg = float(np.degrees(BANANA_GRASP_TILT_RAD))
            tilt_candidates = []
            for tilt_deg in (0.0, configured_tilt_deg, 5.0, 10.0, 15.0):
                if not any(abs(tilt_deg - existing) < 1e-6 for existing in tilt_candidates):
                    tilt_candidates.append(tilt_deg)
            selected_banana_candidate = None
            for tilt_deg in tilt_candidates:
                candidate_orientation = get_top_down_grasp_orientation(
                    object_name,
                    target_prim,
                    tilt_override=np.radians(tilt_deg),
                )
                candidate_alignment = banana_orientation_alignment(
                    candidate_orientation
                )
                if candidate_alignment < BANANA_MIN_SHORT_AXIS_ALIGNMENT:
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
                if candidate_reachability["selected"] is None:
                    continue
                selected_banana_candidate = (
                    tilt_deg,
                    candidate_orientation,
                    candidate_tcp,
                    candidate_reachability,
                    candidate_alignment,
                )
                break
            if selected_banana_candidate is not None:
                (
                    tilt_deg,
                    grasp_orientation,
                    final_grasp_tcp,
                    final_pose_reachability,
                    planned_alignment,
                ) = selected_banana_candidate
                grasp_strategy += f"+strict_short_axis_ik_{tilt_deg:g}deg"
                print(
                    "🧭 香蕉改用严格可达的短轴抓取候选: "
                    f"arm={final_pose_reachability['selected']}, "
                    f"tilt={tilt_deg:.1f}°, alignment={planned_alignment:.3f}"
                )
            else:
                return {
                    "success": False,
                    "message": "no strictly reachable banana short-axis grasp pose",
                    "planned_short_axis_alignment": planned_alignment,
                    "reachability_precheck": reachability,
                }
    if final_pose_reachability["selected"] is None and canonical_object_name != "banana":
        for inward_tilt_deg in (
            5.0, 8.0, 10.0, 12.0, 15.0, 20.0,
            25.0, 30.0, 35.0, 40.0, 45.0,
        ):
            candidate_orientation = get_top_down_grasp_orientation(
                object_name,
                target_prim,
                tilt_override=np.radians(inward_tilt_deg),
            )
            candidate_tcp = get_tcp_target_for_gripper_center(
                object_position,
                candidate_orientation,
            )
            candidate_reachability = _evaluate_arm_pose_reachability(
                object_name,
                candidate_tcp,
                candidate_orientation,
            )
            if candidate_reachability["selected"] is None:
                continue
            grasp_orientation = candidate_orientation
            final_grasp_tcp = candidate_tcp
            final_pose_reachability = candidate_reachability
            grasp_strategy += f"+adaptive_inward_tilt_{inward_tilt_deg:g}deg"
            print(
                "🧭 顶视抓取位于工作空间边缘，自动采用向内倾斜姿态: "
                f"{inward_tilt_deg:.1f}°"
            )
            break
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
        }
    print(
        f"🎯 DACH 抓取中心上移 {DACH_GRASP_HEIGHT_OFFSET:.3f} m: "
        f"z={object_position[2]:.4f}"
    )
    initial_object_position, _, _ = get_sim_pose(target_prim)
    grasp_center_z_offset = float(object_position[2] - initial_object_position[2])

    goal_position = resolve_place_position(target_name, object_position)
    show_red_grasp_point(get_current_stage(), object_position)
    step_app()
    ensure_robot_control_ready()
    print(f"🎯 最终抓取位置: {object_position}")
    print(f"🧭 最终抓取姿态(wxyz): {grasp_orientation}")
    print(f"📍 语义目标 {target_name} -> 放置位置: {goal_position}")

    print(f"🚀 执行 Pick & Place: {object_name} -> {target_name}")
    # grasp_orientation 已由 GraspNet（6DoF）或 top-down fallback 确定。
    # 不要在这里重新调用 get_top_down_grasp_orientation()，
    # 否则 GraspNet 推理出的侧向/斜向位姿会被丢弃，
    # 导致 hover 始终沿竖直 [0,0,-1] 方向接近，在 arm 工作空间边界处 IK 失败。
    hover_reference_orientation = grasp_orientation
    grasp_approach_direction = (
        quat_rotate(
            hover_reference_orientation,
            np.array([1.0, 0.0, 0.0]),
        )
        if canonical_object_name == "banana"
        else np.array([0.0, 0.0, -1.0], dtype=float)
    )
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
    hover_orientation = hover_reference_orientation
    # The reachability result is a kinematic gate for arm selection only.
    # Executing its first IK waypoint directly bypasses the Lula world model
    # and can sweep through the basket. Every commanded hover must therefore
    # be replanned below with the table and basket obstacles enabled.
    print("🛡️ 选臂 IK 仅用于预检；悬停移动统一交给 Lula RRT 碰撞规划")
    for hover_clearance in (0.08, 0.12, 0.16, 0.20, 0.24):
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
            orientation=hover_orientation,
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

    if graspnet_position_active:
        object_position = grasp_target_center.copy()
    else:
        live_bbox_center, live_bbox_min, live_bbox_max = get_bbox_center(
            get_current_stage(),
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
    desired_closing_axis = get_mesh_horizontal_principal_axes(
        get_current_stage(),
        object_prim_path,
    )[1]
    alignment_threshold = (
        BANANA_MIN_SHORT_AXIS_ALIGNMENT
        if canonical_object_name == "banana"
        # Cans are nearly circular.  Require the reached jaw axis to stay
        # within about 2.6 degrees of the PCA short axis; a looser threshold
        # projects the 92 mm long axis into the opening and makes a feasible
        # grasp look physically too wide.
        else 0.999
    )
    physical_alignment = float(
        abs(
            np.dot(
                get_gripper_closing_axis()[:2],
                np.asarray(desired_closing_axis, dtype=float)[:2],
            )
        )
    )
    _, current_hover_rotation = state.controller.get_end_effector_pose()
    current_hover_rotation = np.asarray(current_hover_rotation, dtype=float)
    current_hover_approach = current_hover_rotation[:, 0]
    current_hover_approach /= np.linalg.norm(current_hover_approach)
    current_hover_tilt = float(
        np.arccos(
            np.clip(
                np.dot(
                    current_hover_approach,
                    np.array([0.0, 0.0, -1.0]),
                ),
                -1.0,
                1.0,
            )
        )
    )
    if canonical_object_name != "banana":
        orientation_result = move_ee_collision_aware_approach(
            "nonbanana_hover_orientation",
            get_rmp_ee_position(),
            tolerance=0.02,
            orientation=hover_reference_orientation,
            gripper_positions=gripper_open_target,
            desired_closing_axis=desired_closing_axis,
            minimum_closing_alignment=alignment_threshold,
            maximum_approach_tilt_rad=np.radians(45.0),
        )
        print(
            "🧭 非香蕉物体在悬停高度收敛到已选抓取姿态: "
            f"success={orientation_result['success']}"
        )
    elif (
        physical_alignment >= alignment_threshold
        and current_hover_tilt <= np.radians(float(os.environ.get("AURA_MAX_GRASP_APPROACH_TILT_DEG", "60.0")))
    ):
        current_hover_orientation = np.asarray(
            rot_matrix_to_quat(current_hover_rotation),
            dtype=float,
        )
        orientation_result = {
            "success": True,
            "orientation_constrained": True,
            "planner": "validated_hover_orientation",
            "distance_m": 0.0,
            "finger_clearance_m": float(get_gripper_table_clearance()),
            "downward_tilt_deg": float(np.degrees(current_hover_tilt)),
            "closing_alignment": physical_alignment,
            "orientation_refined": False,
            "hold_orientation": current_hover_orientation.tolist(),
        }
        print(
            "✅ 悬停姿态已满足香蕉抓取约束，跳过重复 RRT: "
            f"tilt={np.degrees(current_hover_tilt):.1f}°, "
            f"alignment={physical_alignment:.3f}"
        )
    else:
        # The selected banana pose already passed strict IK for both hover and
        # grasp.  A fixed-TCP yaw correction here used to jump wrist branches
        # and create the visible twist.  Fail cleanly if execution did not
        # realize the prevalidated pose instead of introducing a new pose.
        orientation_result = {
            "success": False,
            "orientation_constrained": True,
            "planner": "strict_prevalidated_hover",
            "distance_m": 0.0,
            "finger_clearance_m": float(get_gripper_table_clearance()),
            "downward_tilt_deg": float(np.degrees(current_hover_tilt)),
            "closing_alignment": physical_alignment,
            "orientation_refined": False,
            "hold_orientation": None,
        }
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
    # The can's asymmetric jaw drive can settle several millimetres short of
    # the nominal open target while still providing its measured clearance.
    # Keep the strict 1 mm criterion for bananas; use the validated 12 mm
    # mechanical tolerance for non-banana objects and let geometry checks below
    # decide whether the opening is actually sufficient.
    # Non-banana cans use the measured free-opening geometry below as the
    # authoritative feasibility check. Their asymmetric jaw drives can settle
    # short of the nominal joint target even when the physical gap is usable.
    if canonical_object_name == "banana" and open_result["error_m"] > 0.001:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "gripper did not reach the open target before descent",
            "gripper_feedback": open_result["feedback"].tolist(),
            "gripper_target": open_result["target"].tolist(),
            "gripper_open_error_m": open_result["error_m"],
        }
    if not open_result["converged"]:
        print(
            "⚠️ 非香蕉物体夹爪未完全达到标称开口，"
            f"继续执行几何开口检查: error={open_result['error_m']:.4f} m"
        )
    print(
        "✅ 下探前夹爪已张开到物理最大开口: "
        f"jaw={state.dach_arm.gripper.get_joint_positions()}"
    )
    live_closing_axis_3d = get_gripper_closing_axis()
    gripper_opening_width = float(get_gripper_inner_opening_width())
    _, _, object_closing_width = get_mesh_extent_along_axis(
        get_current_stage(),
        object_prim_path,
        live_closing_axis_3d,
    )
    # Preserve the 4 mm margin for banana, but use a 0.5 mm contact margin
    # for the DACH can.  Its measured PCA width is about 79.8 mm and the live
    # gripper opening is about 81.0 mm; a uniform 4 mm margin rejects the only
    # physically feasible short-axis grasp before planning can run.
    aperture_margin = 0.004 if canonical_object_name == "banana" else 0.0005
    required_opening_width = float(object_closing_width + aperture_margin)
    print(
        "📏 抓取前开口尺寸校验: "
        f"jaw_gap={gripper_opening_width:.4f} m, "
        f"object_width={object_closing_width:.4f} m, "
        f"required={required_opening_width:.4f} m"
    )
    if required_opening_width > gripper_opening_width:
        move_robot_home(frames=90)
        release_pregrasp_object()
        message = "object is wider than the physical gripper opening"
        if (
            canonical_object_name in {"master_chef_can", "tomato_soup_can"}
            and ALLOW_OVERSIZED_CAN_GRASP
        ):
            message = (
                "Isaac oversized-can mode is enabled, but the live gripper opening "
                "was not reloaded with the extended jaw limit"
            )
        return {
            "success": False,
            "message": message,
            "object_name": str(object_name),
            "target_name": str(target_name),
            "physical_constraint": {
                "gripper_opening_m": gripper_opening_width,
                "object_width_on_closing_axis_m": float(object_closing_width),
                "required_opening_m": required_opening_width,
                "safety_margin_m": aperture_margin,
                "oversized_can_mode_enabled": bool(ALLOW_OVERSIZED_CAN_GRASP),
            },
        }
    banana_closed_center_offset = np.zeros(2, dtype=float)
    if canonical_object_name == "banana":
        print(
            "📐 香蕉预抓取不再执行开闭探测；"
            "夹指中点直接对准校正后的 GraspNet 抓取点"
        )
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
    planar_center_tolerance = (
        BANANA_PLANAR_CENTER_TOLERANCE
        if canonical_object_name == "banana"
        else 0.008
    )
    planar_refinement_steps = (
        BANANA_PLANAR_REFINEMENT_STEPS
        if canonical_object_name == "banana"
        else max(BANANA_PLANAR_REFINEMENT_STEPS, 5)
    )
    desired_grasp_tcp_z = float(grasp_position[2])
    for refinement in range(planar_refinement_steps):
        live_finger_midpoint = get_gripper_finger_midpoint()
        desired_planar_center = (
            np.asarray(physical_alignment_center[:2], dtype=float)
            - banana_closed_center_offset
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
            BANANA_MAX_PLANAR_CORRECTION / max(planar_error_norm, 1e-9),
        )
        corrected_hover_tcp = get_rmp_ee_position().copy()
        corrected_hover_tcp[:2] += planar_error * correction_scale
        if not move_ee_smooth(
            f"pregrasp_planar_align_{refinement + 1}",
            get_rmp_ee_position(),
            corrected_hover_tcp,
            segments=2,
            tolerance=0.008,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
            monitor_table_clearance=True,
        ):
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "failed to align finger midpoint at hover height",
                "planar_error": planar_error.tolist(),
            }
    live_finger_midpoint = get_gripper_finger_midpoint()
    desired_planar_center = (
        np.asarray(physical_alignment_center[:2], dtype=float)
        - banana_closed_center_offset
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
    # threshold.  We use the *current* finger world positions (at hover) and
    # the planned vertical descent to predict the finger bottom Z at grasp.
    if state.planning_table_surface_z is not None:
        min_finger_clearance = float(
            os.environ.get("AURA_MIN_GRIPPER_TABLE_CLEARANCE", "0.0")
        ) + float(os.environ.get("AURA_TABLE_CLEARANCE_ABORT_MARGIN", "0.0"))
        # Safety margin: aim slightly higher than the abort threshold so
        # one-frame IK overshoot cannot trip the safety stop.  This pad, not
        # the banana geometry, is what sets the final descent depth -- the
        # guard lifts the grasp center until predicted clearance equals it,
        # so shrinking it is the only way to get the jaw lower.
        configured_guard_pad = float(
            os.environ.get("AURA_GRASP_CLEARANCE_GUARD_PAD", "0.0005")
        )
        guard_pad = max(
            configured_guard_pad,
            0.003 if canonical_object_name == "banana" else configured_guard_pad,
        )
        closed_jaw_clearance_lift = (
            LARGE_CAN_CLOSED_JAW_CLEARANCE_LIFT
            if large_can_mode
            else 0.0
        )
        guard_target_clearance = (
            min_finger_clearance
            + guard_pad
            + closed_jaw_clearance_lift
        )
        current_finger_bottom = min(
            float(np.min(get_finger_collision_world_corners(state.left_finger, "left")[:, 2])),
            float(np.min(get_finger_collision_world_corners(state.right_finger, "right")[:, 2])),
        )
        current_clearance = current_finger_bottom - state.planning_table_surface_z
        current_tcp_z = float(get_rmp_ee_position()[2])
        predicted_descent = current_tcp_z - float(grasp_position[2])
        predicted_clearance = current_clearance - predicted_descent
        if predicted_clearance < guard_target_clearance:
            required_lift = guard_target_clearance - predicted_clearance
            print(
                f"⚠️ Grasp finger clearance unsafe: "
                f"current_clearance={current_clearance:.4f} m, "
                f"predicted_descent={predicted_descent:.4f} m, "
                f"predicted={predicted_clearance:.4f} m < "
                f"target={guard_target_clearance:.4f} m, "
                f"lifting grasp center by {required_lift:.4f} m"
            )
            object_position[2] += required_lift
            grasp_position[2] += required_lift
            physical_alignment_center[2] += required_lift
            grasp_center_z_offset += required_lift
            # Keep the closed-loop target at the same safety-adjusted height.
            # Its XY remains the calibrated GraspNet point; without this the
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
    place_hover_position = place_gripper_position + np.array([0.0, 0.0, 0.22])
    approach_result = move_ee_collision_aware_approach(
        "grasp_approach",
        grasp_position,
        tolerance=0.035,
        # The base is positioned so both banana and can TCP targets remain in
        # Lula's fixed top-down orientation manifold.  Keeping this rotation
        # constrained is essential: an unconstrained RRT solution can move the
        # wrist while leaving the finger center far from the USD object.
        orientation=grasp_motion_orientation,
        gripper_positions=gripper_open_target,
        maximum_approach_tilt_rad=(
            None
            if canonical_object_name == "banana"
            else np.radians(45.0)
        ),
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

    refinement_steps = (
        max(GRASP_REFINEMENT_STEPS, 3)
        if large_can_mode
        else GRASP_REFINEMENT_STEPS
    )
    for refinement in range(refinement_steps):
        live_finger_center = get_gripper_collision_center()
        if graspnet_position_active:
            desired_finger_center = physical_alignment_center.copy()
            # Keep GraspNet's table-safe height. This final loop corrects the
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
            segments=3,
            max_steps_per_segment=40,
            tolerance=0.008,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
            monitor_table_clearance=True,
        )
        if not refinement_reached and not large_can_mode:
            move_robot_home(frames=90)
            return {
                "success": False,
                "message": "failed to center DACH gripper on object",
                "center_error": center_error.tolist(),
            }
        if not refinement_reached:
            print(
                "↪️ 扩展罐头模式固定姿态闭环尚未收敛，"
                "重新测量夹爪中心后继续修正"
            )

    # At grasp height do not sweep horizontally.  A miss is safer to report
    # than trying to correct beside the table and distorting the wrist path.
    live_finger_midpoint = get_gripper_finger_midpoint()
    desired_planar_center = (
        np.asarray(physical_alignment_center[:2], dtype=float)
        - banana_closed_center_offset
    )
    planar_error = np.asarray(
        desired_planar_center - live_finger_midpoint[:2],
        dtype=float,
    )
    planar_error_norm = float(np.linalg.norm(planar_error))
    if planar_error_norm > planar_center_tolerance:
        low_pose_clearance = float(get_gripper_table_clearance())
        correction_limit = (
            0.03
            if canonical_object_name == "banana" or large_can_mode
            else 0.0
        )
        correction_clearance = max(
            float(os.environ.get("AURA_MIN_GRIPPER_TABLE_CLEARANCE", "0.0"))
            + float(os.environ.get("AURA_TABLE_CLEARANCE_ABORT_MARGIN", "0.0")),
            0.005 if canonical_object_name == "banana" else 0.012,
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
                segments=2,
                tolerance=0.008,
                orientation=grasp_motion_orientation,
                gripper_positions=gripper_open_target,
                monitor_table_clearance=True,
            ):
                live_finger_midpoint = get_gripper_finger_midpoint()
                desired_planar_center = (
                    np.asarray(physical_alignment_center[:2], dtype=float)
                    - banana_closed_center_offset
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
                "message": "finger midpoint is not centered on target object",
                "planar_error": planar_error.tolist(),
                "planar_error_norm": planar_error_norm,
                "gripper_table_clearance_m": low_pose_clearance,
            }

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
    place_hover_position = place_gripper_position + np.array([0.0, 0.0, 0.22])

    banana_short_axis = get_mesh_horizontal_principal_axes(
        get_current_stage(),
        object_prim_path,
    )[1]
    live_closing_axis = get_gripper_closing_axis()[:2]
    axis_alignment = float(abs(np.dot(live_closing_axis, banana_short_axis)))
    print(
        f"🧭 香蕉闭合轴校验: gripper={live_closing_axis}, "
        f"banana_short={banana_short_axis}, alignment={axis_alignment:.3f}"
    )
    if axis_alignment < alignment_threshold:
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "gripper is not aligned with banana short axis",
            "closing_axis_alignment": axis_alignment,
            "minimum_alignment": alignment_threshold,
        }

    use_simulated_attachment = (
        LARGE_CAN_USE_SIMULATED_ATTACHMENT
        if canonical_object_name in {"master_chef_can", "tomato_soup_can"}
        else BANANA_USE_SIMULATED_ATTACHMENT
    )

    # Keep the lightweight object kinematic while the jaws close.  It still
    # participates in collision, but the first contacting finger cannot shove
    # it out of the second finger's path.  attach_simulated_object() releases
    # the kinematic lock and creates the FixedJoint in the same control tick.
    for _ in range(5):
        hold_ee_target(grasp_position, grasp_motion_orientation)
        state.dach_arm.gripper.set_joint_positions(gripper_open_target)
        step_app()
    nominal_gripper_close_target = np.asarray(
        gripper_close_target, dtype=float
    ).copy()
    object_position_before_close, _, _ = get_sim_pose(target_prim)
    close_result = close_gripper_slowly(
        grasp_position,
        orientation=grasp_motion_orientation,
        frames=get_gripper_close_frames(object_name),
        target_close=nominal_gripper_close_target,
        monitor_table_clearance=False,
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
    # Fallback: if the gripper closed to its target without detecting contact
    # but the object barely moved, do a test lift to verify grasp
    fallback_triggered = False
    if (
        not contact_confirmed
        and close_displacement < 0.002
        and state._pregrasp_frozen_prim_path is None
    ):
        fallback_triggered = True
        print(
            "⚠️ 接触校验未触发但物体位移极小，尝试测试抬升验证夹持: "
            f"displacement={close_displacement:.4f}m"
        )
        # Test lift: raise 0.02m to verify object is actually grasped
        test_lift_position = grasp_position.copy()
        test_lift_position[2] += 0.02
        test_lift_ok = move_ee_smooth(
            "test_lift",
            grasp_position,
            test_lift_position,
            segments=1,
            max_steps_per_segment=30,
            tolerance=0.03,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_close_target,
        )
        test_object_position, _, _ = get_sim_pose(target_prim)
        test_lift_distance = float(test_object_position[2] - object_position_before_close[2])
        if test_lift_ok and test_lift_distance >= 0.015:
            contact_confirmed = True
            print(
                f"✅ 测试抬升成功: lift={test_lift_distance:.4f}m，确认夹持"
            )
            # Lower back to grasp position before full lift
            move_ee_smooth(
                "test_lower",
                test_lift_position,
                grasp_position,
                segments=1,
                max_steps_per_segment=30,
                tolerance=0.03,
                orientation=grasp_motion_orientation,
                gripper_positions=gripper_close_target,
            )
        else:
            print(
                f"❌ 测试抬升失败: lift={test_lift_distance:.4f}m，未夹住物体"
            )
    print(
        "🧪 夹持接触校验: "
        f"feedback={gripper_feedback}, target={gripper_close_target}, "
        f"efforts={measured_efforts}N, "
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
        release_pregrasp_object()
        return {
            "success": False,
            "message": "grasp contact was not confirmed; lift was not commanded",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "gripper_feedback": gripper_feedback.tolist(),
            "gripper_efforts_n": measured_efforts.tolist(),
            "closure_residual_m": closure_residual,
            "object_displacement_m": close_displacement,
            "gripper_table_clearance_m": get_gripper_table_clearance(),
            "finger_colliders": collision_diagnostics,
        }
    grasp_containment = get_object_gripper_containment(object_prim_path)
    print(
        "🧪 闭合后双指空间包含校验: "
        f"contained={grasp_containment['contained']}, "
        f"axial={grasp_containment['axial_error_m']:.4f} m, "
        f"approach={grasp_containment['approach_error_m']:.4f} m, "
        f"lateral={grasp_containment['lateral_error_m']:.4f} m"
    )
    containment_retry = None
    if not grasp_containment["contained"] and large_can_mode:
        first_containment = grasp_containment
        planar_correction = np.asarray(
            first_containment["object_center"], dtype=float
        ) - np.asarray(first_containment["gripper_center"], dtype=float)
        planar_correction[2] = 0.0
        planar_correction_norm = float(np.linalg.norm(planar_correction))
        retry_clearance = float(get_gripper_table_clearance())
        retry_clearance_minimum = (
            MIN_GRIPPER_TABLE_CLEARANCE
            + TABLE_CLEARANCE_ABORT_MARGIN
            + float(os.environ.get("AURA_GRASP_CLEARANCE_GUARD_PAD", "0.0005"))
        )
        retry_eligible = bool(
            abs(first_containment["axial_error_m"])
            <= first_containment["axial_limit_m"]
            and first_containment["approach_inside"]
            and planar_correction_norm <= 0.03
            and retry_clearance >= retry_clearance_minimum
        )
        containment_retry = {
            "attempted": retry_eligible,
            "planar_correction_m": planar_correction.tolist(),
            "planar_correction_norm_m": planar_correction_norm,
            "maximum_correction_m": 0.03,
            "clearance_before_retry_m": retry_clearance,
            "minimum_clearance_m": retry_clearance_minimum,
            "first_containment": first_containment,
        }
        if retry_eligible:
            print(
                "↪️ 扩展罐头闭合后平面对中重试: "
                f"correction={planar_correction}, "
                f"norm={planar_correction_norm:.4f} m, "
                f"clearance={retry_clearance:.4f} m"
            )
            retry_start_tcp = get_rmp_ee_position().copy()
            open_gripper_slowly(
                retry_start_tcp,
                orientation=grasp_motion_orientation,
                target_open=gripper_open_target,
            )
            retry_target_tcp = get_rmp_ee_position().copy()
            retry_target_tcp[:2] += planar_correction[:2]
            retry_reached = move_ee_smooth(
                "large_can_post_close_recenter",
                get_rmp_ee_position(),
                retry_target_tcp,
                segments=2,
                max_steps_per_segment=40,
                tolerance=0.008,
                orientation=grasp_motion_orientation,
                gripper_positions=gripper_open_target,
                monitor_table_clearance=True,
            )
            grasp_position = get_rmp_ee_position().copy()
            containment_retry["motion_reached"] = bool(retry_reached)
            containment_retry["clearance_after_motion_m"] = float(
                get_gripper_table_clearance()
            )
            if retry_reached:
                for _ in range(5):
                    hold_ee_target(grasp_position, grasp_motion_orientation)
                    state.dach_arm.gripper.set_joint_positions(gripper_open_target)
                    step_app()
                object_position_before_close, _, _ = get_sim_pose(target_prim)
                close_result = close_gripper_slowly(
                    grasp_position,
                    orientation=grasp_motion_orientation,
                    frames=get_gripper_close_frames(object_name),
                    target_close=nominal_gripper_close_target,
                    monitor_table_clearance=True,
                )
                gripper_close_target = np.asarray(
                    close_result["hold_target"], dtype=float
                )
                gripper_feedback = np.asarray(
                    state.dach_arm.gripper.get_joint_positions(), dtype=float
                )
                closure_residual = float(
                    np.mean(
                        np.maximum(
                            gripper_feedback - gripper_close_target,
                            0.0,
                        )
                    )
                )
                object_position_after_close, _, _ = get_sim_pose(target_prim)
                close_displacement = float(
                    np.linalg.norm(
                        object_position_after_close - object_position_before_close
                    )
                )
                contact_confirmed = bool(close_result["contact_confirmed"])
                measured_efforts = np.asarray(
                    close_result["measured_efforts_n"], dtype=float
                )
                containment_retry["contact_confirmed"] = contact_confirmed
                grasp_containment = get_object_gripper_containment(
                    object_prim_path
                )
                containment_retry["final_containment"] = grasp_containment
                print(
                    "🧪 扩展罐头重闭合校验: "
                    f"contact={contact_confirmed}, "
                    f"contained={grasp_containment['contained']}, "
                    f"axial={grasp_containment['axial_error_m']:.4f} m, "
                    f"approach={grasp_containment['approach_error_m']:.4f} m, "
                    f"lateral={grasp_containment['lateral_error_m']:.4f} m"
                )
        else:
            print(
                "⛔ 扩展罐头闭合后重试被安全边界拒绝: "
                f"correction={planar_correction_norm:.4f} m, "
                f"clearance={retry_clearance:.4f} m, "
                f"axial_ok={abs(first_containment['axial_error_m']) <= first_containment['axial_limit_m']}, "
                f"approach_ok={first_containment['approach_inside']}"
            )
    if not contact_confirmed:
        open_gripper_slowly(
            grasp_position,
            orientation=grasp_motion_orientation,
            target_open=gripper_open_target,
        )
        release_pregrasp_object()
        return {
            "success": False,
            "message": "grasp contact was not confirmed after containment retry",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "gripper_feedback": gripper_feedback.tolist(),
            "gripper_efforts_n": measured_efforts.tolist(),
            "closure_residual_m": closure_residual,
            "object_displacement_m": close_displacement,
            "containment_retry": containment_retry,
        }
    if not grasp_containment["contained"]:
        open_gripper_slowly(
            grasp_position,
            orientation=grasp_motion_orientation,
            target_open=gripper_open_target,
        )
        release_pregrasp_object()
        return {
            "success": False,
            "message": "object is not physically centered between both fingers",
            "object_name": str(object_name),
            "target_name": str(target_name),
            "grasp_containment": grasp_containment,
            "containment_retry": containment_retry,
        }
    if use_simulated_attachment and contact_confirmed:
        attach_simulated_object(target_prim, object_prim_path)
        grasp_strategy += "+contact_confirmed_fixed_joint"
        print("🧲 夹爪闭合并通过几何校验后建立 PhysX FixedJoint")
    elif use_simulated_attachment:
        use_simulated_attachment = False
        print("⚠️ 未检测到夹持阻力，禁止仿真随动附着")
    else:
        release_pregrasp_object()
    # Keep the wrist orientation constrained throughout lift/carry. The old
    # per-frame teleport attachment needed a free orientation and allowed the
    # IK/RRT solution to twist the arm; a rigid PhysX joint does not.
    transport_orientation = grasp_motion_orientation
    live_object_before_lift, _, _ = get_sim_pose(target_prim)
    print(
        f"🧪 闭合后抓取诊断: object={live_object_before_lift}, "
        f"finger_center={get_gripper_collision_center()}, "
        f"finger_midpoint={get_gripper_finger_midpoint()}, "
        f"jaw={state.dach_arm.gripper.get_joint_positions()}"
    )
    for _ in range(5):
        hold_ee_target(grasp_position, grasp_motion_orientation)
        step_app()

    lift_ok = move_ee_smooth(
        "lift",
        grasp_position,
        lift_position,
        segments=3,
        max_steps_per_segment=50,
        tolerance=0.065,
        orientation=transport_orientation,
        gripper_positions=gripper_close_target,
    )
    lifted_object_position, _, _ = get_sim_pose(target_prim)
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
        if state._simulated_attachment is not None:
            detach_simulated_object()
            step_app(2)
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
            "grasp_position": object_position.tolist(),
            "place_position": goal_position.tolist(),
        }
        return failure_result
    base_carry_height = max(lift_position[2], place_hover_position[2])
    clearance_candidates = []
    for clearance in (
        0.0,
        min(CARRY_APEX_CLEARANCE, 0.05),
        min(CARRY_APEX_CLEARANCE, DACH_PATH_CLEARANCE),
        CARRY_APEX_CLEARANCE,
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
    carry_path_strategy = "cartesian_fixed_orientation"
    # Carry directly at the already reached lift height.  The old main route
    # added a redundant 5 cm up/down detour before every horizontal transport.
    # Extra height is now reserved for the collision-planning fallback below.
    direct_clearance = 0.0
    direct_apex_z = base_carry_height + direct_clearance
    direct_source_clear = np.array(
        [lift_position[0], lift_position[1], direct_apex_z], dtype=float
    )
    direct_target_clear = np.array(
        [place_hover_position[0], place_hover_position[1], direct_apex_z],
        dtype=float,
    )
    direct_route = []
    previous_point = lift_position
    for candidate_point in (
        direct_source_clear,
        direct_target_clear,
        place_hover_position,
    ):
        if np.linalg.norm(candidate_point - previous_point) <= 1e-4:
            continue
        direct_route.append(candidate_point)
        previous_point = candidate_point
    print(
        "🧭 主运输直达路径: "
        f"keyposes={len(direct_route) + 1}, "
        f"extra_clearance={direct_clearance:.3f} m"
    )
    carry_ok = move_ee_waypoints(
        "carry_cartesian_fixed_orientation",
        direct_route,
        tolerance=0.08,
        orientation=transport_orientation,
        gripper_positions=gripper_close_target,
        max_joint_step_rad=0.016,
        minimum_frames=30,
        max_spacing_m=CARRY_CARTESIAN_WAYPOINT_SPACING,
    )
    if carry_ok:
        carry_clearance_used = direct_clearance
        carry_transport_orientation = transport_orientation
    for clearance in clearance_candidates:
        if carry_ok:
            break
        apex_z_height = base_carry_height + clearance
        source_clear_position = np.array(
            [lift_position[0], lift_position[1], apex_z_height],
            dtype=float,
        )
        target_clear_position = np.array(
            [place_hover_position[0], place_hover_position[1], apex_z_height],
            dtype=float,
        )
        print(f"🧭 尝试搬运安全层: extra_clearance={clearance:.3f} m")
        fallback_keyposes = []
        previous_point = lift_position
        for candidate_point in (
            source_clear_position,
            target_clear_position,
            place_hover_position,
        ):
            if np.linalg.norm(candidate_point - previous_point) <= 1e-4:
                continue
            fallback_keyposes.append(candidate_point)
            previous_point = candidate_point
        planned_path = plan_collision_free_keyposes(
            fallback_keyposes,
            orientation=transport_orientation,
        )
        if planned_path is not None:
            carry_clearance_used = clearance
            carry_planned_path = planned_path
            carry_path_strategy = "rrt_keypose_fallback"
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
        _execute_joint_trajectory(
            "carry_clearance_route",
            carry_joint_targets,
            gripper_positions=gripper_close_target,
        )
        carry_error = float(
            np.linalg.norm(get_rmp_ee_position() - place_hover_position)
        )
        carry_ok = carry_error <= 0.08
    if not carry_ok:
        abort_position = get_rmp_ee_position()
        if state._simulated_attachment is not None:
            detach_simulated_object()
            step_app(2)
        open_gripper_slowly(
            abort_position,
            orientation=grasp_motion_orientation,
            frames=20,
            target_open=gripper_open_target,
        )
        abort_retreat_position = abort_position + np.array([0.0, 0.0, 0.15])
        move_ee_smooth(
            "carry_abort_retreat",
            abort_position,
            abort_retreat_position,
            segments=3,
            tolerance=0.05,
            orientation=grasp_motion_orientation,
            gripper_positions=gripper_open_target,
        )
        move_robot_home(frames=90)
        return {
            "success": False,
            "message": "collision-clearance carry path is unreachable",
            "object_name": str(object_name),
            "target_name": str(target_name),
        }
    placing_in_basket = canonical_target_name == "basket"
    if placing_in_basket and not set_planning_basket_obstacle_enabled(False):
        return {
            "success": False,
            "message": "failed to open basket placement corridor",
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
        segments=3,
        max_steps_per_segment=45,
        tolerance=0.07,
        orientation=transport_orientation,
        gripper_positions=gripper_close_target,
        monitor_table_clearance=True,
        max_joint_step_rad=0.016,
        minimum_frames=30,
    )
    if not lower_ok and canonical_object_name != "banana":
        # Retry the low-speed descent with the same transport orientation. A
        # position-only retry would let Lula flip the wrist at the basket.
        print("↪️ 放置下探固定姿态未收敛，使用同姿态低速下探重试")
        lower_ok = move_ee_smooth(
            "lower_position_first_fallback",
            get_rmp_ee_position(),
            place_gripper_position,
            segments=4,
            max_steps_per_segment=55,
            tolerance=0.07,
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
    release_attachment = None
    if state._simulated_attachment is not None:
        release_attachment = detach_simulated_object(hold_kinematic=True)
        step_app(2)
    # Do not open the jaws inside the basket.  First move the still-closed
    # gripper vertically clear of the can and rim; opening in place can sweep
    # a finger through the can and pull it back out with the wrist.
    release_clear_position = place_gripper_position + np.array([0.0, 0.0, 0.12])
    release_clear_ok = move_ee_smooth(
        "release_vertical_clear",
        place_gripper_position,
        release_clear_position,
        segments=3,
        max_steps_per_segment=35,
        tolerance=0.04,
        orientation=transport_orientation,
        gripper_positions=gripper_close_target,
        monitor_table_clearance=True,
        max_joint_step_rad=0.016,
        minimum_frames=30,
    )
    if not release_clear_ok:
        print("⚠️ 释放前垂直撤离未完全收敛，仍在高位执行张爪")
    open_gripper_slowly(
        release_clear_position,
        orientation=transport_orientation,
        frames=30,
        target_open=gripper_open_target,
    )
    # Make the release observable in PhysX as well as in the command stream:
    # the slow controller can finish one frame short of its target when the
    # wrist is settling.  Hold the jaws at their calibrated open limit for a
    # few physics frames before freeing the kinematic hand-off state.
    state.dach_arm.gripper.set_joint_positions(gripper_open_target)
    step_app(5)
    # Keep the released object kinematic while the robot retreats.  If it is
    # made dynamic immediately, the retreat/home trajectory can brush the can
    # through the basket wall before the arm is clear.
    # Let the free rigid body settle against the basket floor before measuring
    # containment.  During this first window it remains fixed in the basket.
    step_app(120)
    release_feedback = np.asarray(
        state.dach_arm.gripper.get_joint_positions(), dtype=float
    )
    release_error = float(np.max(np.abs(release_feedback - gripper_open_target)))
    release_verified = bool(
        state._simulated_attachment is None and release_error <= 0.005
    )
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
        final_mesh_center, _, _ = get_bbox_center(
            get_current_stage(),
            object_prim_path,
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
    retreat_position = place_gripper_position + np.array([0.0, 0.0, 0.15])
    retreat_ok = move_ee_smooth(
        "retreat_after_place",
        get_rmp_ee_position(),
        retreat_position,
        segments=3,
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

    if release_attachment is not None:
        release_detached_object(release_attachment)
        # Now that the robot is clear, let PhysX settle the can as a free body.
        step_app(120)

    # Re-check after the arm has fully retreated and returned HOME.  The
    # released can must remain contained after all robot motion, not only at
    # the instant immediately following jaw opening.
    if placing_in_basket:
        step_app(60)
        final_object_position, _, _ = get_sim_pose(target_prim)
        final_mesh_center, _, _ = get_bbox_center(
            get_current_stage(), object_prim_path
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
        "grasp_strategy": grasp_strategy,
        "grasp_approach": approach_result,
        "path_strategy": carry_path_strategy,
        "cartesian_waypoint_spacing_m": CARTESIAN_WAYPOINT_SPACING,
        "carry_cartesian_waypoint_spacing_m": CARRY_CARTESIAN_WAYPOINT_SPACING,
        "transport_lift_height_m": TRANSPORT_LIFT_HEIGHT,
        "carry_extra_clearance_m": carry_clearance_used,
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
    }
    if graspnet_inference_error is not None:
        result["graspnet_inference_error"] = graspnet_inference_error
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
