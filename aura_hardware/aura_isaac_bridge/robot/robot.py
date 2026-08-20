# S5 DACH TRON2A robot runtime — 模块化入口
# 模块拆分: core/state(配置+状态) / core/physics(物理) / core/perception(感知)
#           core/motion(运动控制) / core/task(任务执行)

import importlib
import importlib.util
import json
import os
import site
import sys
import time
import types
from pathlib import Path

import omni.timeline
import cv2
import numpy as np
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.types import ArticulationAction

# ── Bootstrap: 确保 S5 包可导入 ──────────────────────────────────
_BOOTSTRAP_PROJECT_ROOT = Path(
    os.environ.get("EVA_AGENT_ROOT", "/home/acmex/Code/learning/courses/Eva-Agent")
).expanduser().resolve()
_BOOTSTRAP_STUDY_DIR = _BOOTSTRAP_PROJECT_ROOT / "Study"
if not (_BOOTSTRAP_STUDY_DIR / "S5").is_dir():
    _BOOTSTRAP_STUDY_DIR = _BOOTSTRAP_PROJECT_ROOT / "src" / "Study"
_BOOTSTRAP_S5_DIR = _BOOTSTRAP_STUDY_DIR / "S5"
_AURA_ISAAC_BRIDGE_DIR = Path(
    os.environ.get("AURA_ISAAC_BRIDGE_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()


def _load_s5_runtime_config() -> None:
    """Apply S5's config.yaml values before importing state constants."""
    config_path = _BOOTSTRAP_PROJECT_ROOT / "Study" / "S5" / "config" / "config.yaml"
    if not config_path.is_file():
        config_path = _BOOTSTRAP_PROJECT_ROOT / "src" / "Study" / "S5" / "config" / "config.yaml"
    if not config_path.is_file():
        return
    try:
        import yaml

        settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"⚠️ 无法加载 S5 config.yaml，使用环境变量/默认值: {exc}")
        return
    robot = settings.get("robot") or {}
    camera = settings.get("camera") or {}
    perception = settings.get("perception") or {}
    calibration = perception.get("graspnet_calibration") or {}
    scene = settings.get("scene") or {}
    basket = ((scene.get("objects") or {}).get("basket") or {})

    def set_value(env_name, value):
        if value is not None:
            os.environ[env_name] = str(value)

    set_value("S5_CAMERA_PRIM_PATH", camera.get("prim_path"))
    set_value("S5_DACH_ARM_SIDE", str(robot.get("arm_side", "right")).lower())
    set_value("S5_DACH_BASE_XY_JSON", json.dumps(robot.get("base_xy")))
    set_value("S5_GRASPNET_CALIBRATION_ENABLED", str(bool(calibration.get("enabled", False))).lower())
    set_value("S5_GRASPNET_CAMERA_OFFSET_JSON", json.dumps(calibration.get("camera_offset_m", [0.0, 0.0, 0.0])))
    set_value("S5_GRASPNET_CALIBRATION_MAX_CORRECTION_M", calibration.get("max_correction_m", 0.06))
    set_value("S5_BASKET_RESET_POSITION_JSON", json.dumps(basket.get("reset_position")))
    set_value("S5_BASKET_RESET_ORIENTATION_JSON", json.dumps(basket.get("reset_orientation")))

    # Keep this mapping aligned with Study/S5/main.py. Values from config.yaml
    # take precedence over module defaults, while explicit shell overrides win.
    config_env = {
        "S5_DACH_GRASP_HEIGHT_OFFSET": ("grasp_height_offset", 0.020),
        "S5_DACH_GRASP_YAW_OFFSET_DEG": ("grasp_yaw_offset_deg", 0.0),
        "S5_BANANA_GRASP_TILT_DEG": ("banana_grasp_tilt_deg", 30.0),
        "S5_BANANA_NEAR_SIDE_OFFSET": ("banana_near_side_offset_m", 0.0),
        "S5_BANANA_MIN_SHORT_AXIS_ALIGNMENT": ("banana_min_short_axis_alignment", 0.92),
        "S5_MIN_GRIPPER_TABLE_CLEARANCE": ("min_gripper_table_clearance_m", 0.0),
        "S5_TABLE_CLEARANCE_ABORT_MARGIN": ("table_clearance_abort_margin_m", 0.0),
        "S5_GRASP_CLEARANCE_GUARD_PAD": ("grasp_clearance_guard_pad_m", 0.0005),
        "S5_GRASP_MIN_HEIGHT_FRACTION": ("grasp_min_height_fraction", 0.10),
        "S5_BASKET_PLANNING_MARGIN": ("basket_planning_margin_m", 0.03),
        "S5_BASKET_PLACE_TABLE_CLEARANCE": ("basket_place_table_clearance_m", 0.003),
        "S5_MAX_GRASP_APPROACH_TILT_DEG": ("max_grasp_approach_tilt_deg", 60.0),
        "S5_TARGET_GRASP_APPROACH_TILT_DEG": ("target_grasp_approach_tilt_deg", 55.0),
        "S5_ACTION_WAYPOINT_LIMIT": ("action_waypoint_limit", 3),
        "S5_GRIPPER_CLOSE_FRAMES": ("gripper_close_frames", 20),
        "S5_GRIPPER_MAX_EFFORT": ("gripper_max_effort_n", 3.0),
        "S5_GRIPPER_STIFFNESS": ("gripper_stiffness", 250.0),
        "S5_GRIPPER_DAMPING": ("gripper_damping", 8.0),
        "S5_BANANA_STATIC_FRICTION": ("banana_static_friction", 1.2),
        "S5_BANANA_DYNAMIC_FRICTION": ("banana_dynamic_friction", 1.0),
        "S5_GRIPPER_STATIC_FRICTION": ("gripper_static_friction", 6.0),
        "S5_GRIPPER_DYNAMIC_FRICTION": ("gripper_dynamic_friction", 5.0),
        "S5_PHYSX_CONTACT_OFFSET": ("physx_contact_offset_m", 0.02),
        "S5_PHYSX_REST_OFFSET": ("physx_rest_offset_m", 0.001),
        "S5_PHYSX_SOLVER_POSITION_ITERATIONS": ("physx_solver_position_iterations", 32),
        "S5_PHYSX_SOLVER_VELOCITY_ITERATIONS": ("physx_solver_velocity_iterations", 8),
        "S5_PHYSX_MAX_DEPENETRATION_VELOCITY": ("physx_max_depenetration_velocity", 0.2),
        "S5_GRIPPER_CONTACT_RESIDUAL": ("gripper_contact_residual_m", 0.0015),
        "S5_GRIPPER_CONTACT_FORCE_THRESHOLD": ("gripper_contact_force_threshold_n", 0.25),
        "S5_GRIPPER_CONTACT_PRELOAD_RESIDUAL": ("gripper_contact_preload_residual_m", 0.0015),
        "S5_GRIPPER_CONTACT_HOLD_PRELOAD": ("gripper_contact_hold_preload_m", 0.003),
        "S5_GRIPPER_CONTACT_SETTLE_FRAMES": ("gripper_contact_settle_frames", 15),
        "S5_GRIPPER_PRELOAD_CONFIRM_FRAMES": ("gripper_preload_confirm_frames", 3),
        "S5_BANANA_GRIPPER_CLOSE_POSITION": ("banana_gripper_close_position", 0.0),
        "S5_BANANA_PLANAR_REFINEMENT_STEPS": ("banana_planar_refinement_steps", 2),
        "S5_BANANA_PLANAR_CENTER_TOLERANCE": ("banana_planar_center_tolerance_m", 0.008),
        "S5_BANANA_MAX_PLANAR_CORRECTION": ("banana_max_planar_correction_m", 0.04),
        "S5_PATH_CLEARANCE": ("path_clearance_m", 0.10),
        "S5_TRAJECTORY_MAX_JOINT_STEP": ("trajectory_max_joint_step_rad", 0.008),
        "S5_TRAJECTORY_MIN_FRAMES": ("trajectory_min_frames", 8),
        "S5_TRAJECTORY_SETTLE_FRAMES": ("trajectory_settle_frames", 12),
        "S5_GRASP_APPROACH_MAX_JOINT_STEP": ("grasp_approach_max_joint_step_rad", 0.018),
        "S5_GRASP_APPROACH_MIN_FRAMES": ("grasp_approach_min_frames", 24),
        "S5_CARTESIAN_WAYPOINT_SPACING": ("cartesian_waypoint_spacing_m", 0.04),
        "S5_TRANSPORT_LIFT_HEIGHT": ("transport_lift_height_m", 0.32),
        "S5_CARRY_CARTESIAN_WAYPOINT_SPACING": ("carry_cartesian_waypoint_spacing_m", 0.12),
        "S5_GRASP_REFINEMENT_STEPS": ("grasp_refinement_steps", 0),
        "S5_CARRY_APEX_CLEARANCE": ("carry_apex_clearance_m", 0.15),
        "S5_DUAL_ARM_MIN_TCP_SEPARATION": ("dual_arm_min_tcp_separation_m", 0.18),
    }
    for env_name, (key, default) in config_env.items():
        if env_name not in os.environ and key in robot:
            set_value(env_name, robot.get(key, default))
    set_value("S5_BANANA_USE_SIMULATED_ATTACHMENT", str(bool(robot.get("banana_use_simulated_attachment", True))).lower())
    set_value("S5_USE_GRASPNET", str(bool(robot.get("use_graspnet", True))).lower())
    set_value("S5_USE_GRASPNET_ORIENTATION", str(bool(robot.get("use_graspnet_orientation", False))).lower())
    set_value("S5_PHYSX_ENABLE_CCD", str(bool(robot.get("physx_enable_ccd", False))).lower())
    set_value("S5_PHYSX_CONVEX_SHRINK_WRAP", str(bool(robot.get("physx_convex_shrink_wrap", True))).lower())
    set_value("S5_SCENE_OBJECTS_JSON", json.dumps(scene.get("objects") or {}, ensure_ascii=False))


_load_s5_runtime_config()
if str(_BOOTSTRAP_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_STUDY_DIR))
_s5_package = sys.modules.get("S5")
if _s5_package is None:
    _s5_package = types.ModuleType("S5")
    _s5_package.__file__ = str(_BOOTSTRAP_S5_DIR / "__init__.py")
    _s5_package.__package__ = "S5"
    sys.modules["S5"] = _s5_package
_s5_package.__path__ = [
    str(_AURA_ISAAC_BRIDGE_DIR),
    str(_BOOTSTRAP_S5_DIR),
]

from S5.robot.dach_tron2a import (
    DACHTron2AArm,
    DACHTron2AIKController,
    GRIPPER_OPEN,
    LEFT_ARM_HOME,
    RIGHT_ARM_HOME,
)
from S5.robot.motion_planner import (
    DiffusionConfig,
    SparseKeyposeDiffuser,
    minimum_jerk,
)
from omni.kit.app import get_app
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

# ── 导入全局状态与配置 ────────────────────────────────────────────
from S5.core.state import state
from S5.core.state import (
    PROJECT_ROOT, ISAAC_SIM_ROOT, ISAAC_SITE_PACKAGES, STUDY_DIR, S5_DIR, SECTION3_DIR,
    DEFAULT_PROJECT_ROOT, DEFAULT_ISAAC_SIM_ROOT, DEFAULT_ISAAC_SITE_PACKAGES,
    GRASPNET_DIR, GRASPNET_CHECKPOINT_PATH, SAM_MODEL_PATH,
    DEFAULT_GRASPNET_DIR, DEFAULT_SAM_MODEL_PATH,
    CAMERA_PRIM_PATH, CAMERA_RESOLUTION, CAMERA_PREVIEW_RESOLUTION,
    SHOW_GRASP_DEBUG, USE_GRASPNET, USE_GRASPNET_ORIENTATION,
    DACH_SCENE_ROOT_PATH, DACH_ARTICULATION_ROOT_PATH,
    TRON2_URDF_PATH, DACH_ROBOT_DESCRIPTION_PATH, DACH_RIGHT_ROBOT_DESCRIPTION_PATH,
    DACH_LEFT_RRT_CONFIG_PATH, DACH_RIGHT_RRT_CONFIG_PATH,
    DACH_ARM_SIDE, DACH_BASE_XY,
    DACH_GRASP_HEIGHT_OFFSET, DACH_GRASP_YAW_OFFSET_RAD,
    DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET, DACH_JAW_COLLISION_LOCAL_BOUNDS,
    BANANA_GRASP_TILT_RAD, BANANA_NEAR_SIDE_OFFSET, BANANA_MIN_SHORT_AXIS_ALIGNMENT,
    GRASP_POSITION_OFFSET, GRASP_INSERT_DEPTH,
    MAX_GRASP_APPROACH_TILT_RAD, TARGET_GRASP_APPROACH_TILT_RAD,
    BANANA_USE_SIMULATED_ATTACHMENT, GRASP_REFINEMENT_STEPS,
    BANANA_GRIPPER_CLOSE_POSITION, BANANA_PLANAR_REFINEMENT_STEPS,
    BANANA_PLANAR_CENTER_TOLERANCE, BANANA_MAX_PLANAR_CORRECTION,
    DIRECTIONAL_PLACE_DISTANCE, GRASP_MIN_HEIGHT_FRACTION,
    MINIMUM_OBJECT_LIFT, PLACE_SUCCESS_TOLERANCE,
    BASKET_RESET_POSITION, BASKET_RESET_ORIENTATION, BASKET_PLANNING_MARGIN,
    DACH_PATH_CLEARANCE, TRAJECTORY_MAX_JOINT_STEP, TRAJECTORY_MIN_FRAMES,
    TRAJECTORY_SETTLE_FRAMES, GRASP_APPROACH_MAX_JOINT_STEP,
    GRASP_APPROACH_MIN_FRAMES, CARTESIAN_WAYPOINT_SPACING, CARRY_APEX_CLEARANCE,
    MIN_GRIPPER_TABLE_CLEARANCE, TABLE_CLEARANCE_ABORT_MARGIN,
    DUAL_ARM_MIN_TCP_SEPARATION,
    GRIPPER_CLOSE_FRAMES, GRIPPER_MAX_EFFORT, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    GRIPPER_CONTACT_RESIDUAL, GRIPPER_CONTACT_FORCE_THRESHOLD,
    GRIPPER_CONTACT_PRELOAD_RESIDUAL, GRIPPER_CONTACT_HOLD_PRELOAD,
    GRIPPER_CONTACT_SETTLE_FRAMES,
    BANANA_STATIC_FRICTION, BANANA_DYNAMIC_FRICTION,
    GRIPPER_STATIC_FRICTION, GRIPPER_DYNAMIC_FRICTION,
    PHYSX_CONTACT_OFFSET, PHYSX_REST_OFFSET,
    PHYSX_SOLVER_POSITION_ITERATIONS, PHYSX_SOLVER_VELOCITY_ITERATIONS,
    PHYSX_MAX_DEPENETRATION_VELOCITY,
)

# The VS Code executor can retain an older S5 module object between runs. Keep
# the hardware asset explicit in this entrypoint so Lula never falls back to
# the obsolete Eva-Agent troncamp-mani-main path.
if os.environ.get("S5_TRON2_URDF_PATH"):
    TRON2_URDF_PATH = Path(os.environ["S5_TRON2_URDF_PATH"]).expanduser().resolve()

# ── 导入功能模块 ──────────────────────────────────────────────────
from S5.core.physics import (
    step_app, cleanup_debug_markers,
    create_grasp_physics_material, bind_grasp_physics_material,
    ensure_pickable_object, get_dach_finger_paths,
    prepare_dach_finger_collision_instances, configure_dach_contact_physics,
    configure_physics_scene,
)
from S5.core.perception import (
    get_bbox_center,
    get_mesh_horizontal_principal_axes,
    get_mesh_horizontal_cross_section_center,
    get_mesh_center,
    quat_rotate, get_sim_pose, show_red_grasp_point, get_transform,
    initialize_grasp_camera, capture_camera_data, show_camera_preview,
    get_current_object_center, create_sam_prompt_points,
    segment_target_with_sam,
    ensure_graspnet_python_dependencies, load_graspnet_demo,
    release_cuda_inference_cache, infer_graspnet_world_pose,
    resolve_scene_prim_path,
)
from S5.core.motion import (
    set_planning_basket_obstacle_enabled,
    get_finger_collision_world_corners, get_gripper_finger_midpoint,
    get_gripper_collision_center, get_gripper_collision_diagnostics,
    get_gripper_table_clearance, get_gripper_closing_axis,
    get_gripper_center_local_offset, get_tcp_target_for_gripper_center,
    freeze_object_for_pregrasp, release_pregrasp_object,
    attach_simulated_object, update_simulated_attachment,
    detach_simulated_object,
    get_active_joint_positions, get_left_joint_positions,
    ensure_robot_control_ready, get_rmp_ee_position,
    move_ee_to, move_ee_waypoints, move_ee_smooth,
    plan_ee_waypoints, plan_collision_free_keyposes,
    move_ee_collision_aware_approach,
    hold_ee_target, get_gripper_joint_efforts,
    close_gripper_slowly, open_gripper_slowly,
    move_robot_home, reset_robot_after_task,
)
from S5.core.task import (
    resolve_place_position,
    get_top_down_grasp_orientation,
    get_gripper_close_target, get_gripper_open_target,
    get_gripper_close_frames,
    adjust_object_grasp_position,
    evaluate_dual_arm_hover_reachability,
    execute_pick_place,
    snapshot_scene_object_pose, restore_scene_object_pose,
)

# 包装函数：pick + reset，供 restore_isaac_runtime 检测
def execute_pick_place_with_reset(object_name, target_name):
    object_snapshot = snapshot_scene_object_pose(object_name)
    result = None
    try:
        result = execute_pick_place(object_name, target_name)
        return result
    finally:
        reset_robot_after_task()
        if result is None or not bool(result.get("success", False)):
            restore_scene_object_pose(object_snapshot)

# ── 初始化配置 ────────────────────────────────────────────────────
# 设置 HOME / GRIPPER_OPEN 常量（写入 state 对象，供其他模块读取）
state.DACH_HOME_JOINT_POSITIONS = (
    LEFT_ARM_HOME.copy() if DACH_ARM_SIDE == "left" else RIGHT_ARM_HOME.copy()
)
state.GRIPPER_OPEN_POSITIONS = GRIPPER_OPEN.copy()

from S5.perception.scene_names import SceneNameResolver
state.SCENE_NAME_RESOLVER = SceneNameResolver.from_mapping(
    json.loads(os.environ.get("S5_SCENE_OBJECTS_JSON", "{}"))
)
SCENE_NAME_RESOLVER = state.SCENE_NAME_RESOLVER

print("=== 开始执行脚本 ===")
print(f"📁 Eva-Agent 项目目录: {PROJECT_ROOT}")
print(f"📁 GraspNet baseline 目录: {GRASPNET_DIR}")

# ── 1. SimulationContext ──────────────────────────────────────────
print(f"📁 GraspNet baseline 目录: {GRASPNET_DIR}")

# 1. 获取/创建 SimulationContext（物理引擎核心）
state.sim_context = SimulationContext.instance()
if state.sim_context is None:
    state.sim_context = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,          # 物理步长 60Hz
        physics_prim_path="/PhysicsScene",
    )
    print("✅ 创建了新的 SimulationContext")
else:
    print("✅ 复用了现有的 SimulationContext")

# 2. 确保物理场景 Prim 存在（否则物理引擎无法工作）
stage = get_current_stage()
state.stage = stage
physx_prim = stage.GetPrimAtPath("/PhysicsScene")
if not physx_prim.IsValid():
    print("⚠️ 物理场景不存在，正在创建...")
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_prim = stage.GetPrimAtPath("/PhysicsScene")
    print("✅ 物理场景已创建")
else:
    print("✅ 物理场景已存在")

configure_physics_scene(physx_prim)

# ── 3. 创建 DACH TRON2A 双臂实例 ──────────────────────────────────
state._preserved_arm_state = {}
state._existing_dach_arm = globals().get("state.dach_arm")
state._existing_dach_left = globals().get("state.dach_left")
state._existing_dach_right = globals().get("state.dach_right")
for _state_name in ("state.dach_arm", "state.dach_left", "state.dach_right"):
    _existing_arm = globals().get(_state_name)
    if _existing_arm is None:
        continue
    try:
        _joint_positions = _existing_arm.get_joint_positions()
        _gripper_positions = _existing_arm.gripper.get_joint_positions()
    except Exception:
        continue
    if _joint_positions is not None:
        state._preserved_arm_state[_state_name] = {
            "joints": np.asarray(_joint_positions, dtype=float).reshape(-1).copy(),
            "gripper": (
                None
                if _gripper_positions is None
                else np.asarray(_gripper_positions, dtype=float).reshape(-1).copy()
            ),
        }

state._reuse_dach_articulations = (
    state._existing_dach_arm is not None
    and state._existing_dach_left is not None
    and getattr(state._existing_dach_arm, "arm_side", None) == DACH_ARM_SIDE
    and getattr(state._existing_dach_left, "arm_side", None) == "left"
    and getattr(state._existing_dach_arm, "prim_path", None) == DACH_ARTICULATION_ROOT_PATH
    and getattr(state._existing_dach_left, "prim_path", None) == DACH_ARTICULATION_ROOT_PATH
    and "state.dach_arm" in state._preserved_arm_state
    and "state.dach_left" in state._preserved_arm_state
)

if state.sim_context.is_playing():
    print("▶️ 检测到仿真正运行，热重载保留现有 physics view")

cleanup_debug_markers(stage)

# 3. 创建 DACH TRON2A 左臂实例
if stage.GetPrimAtPath(DACH_SCENE_ROOT_PATH).IsValid():
    print(f"✅ 复用已有的 {DACH_SCENE_ROOT_PATH} Prim")
else:
    raise RuntimeError("未找到 /World/DACH_TRON2A 机器人，请先在 Isaac Sim 中添加该机器人")
if DACH_BASE_XY is not None:
    dach_root = SingleXFormPrim(
        name="s5_dach_root",
        prim_path=DACH_SCENE_ROOT_PATH,
    )
    current_base_position, current_base_orientation = dach_root.get_world_pose()
    configured_base_position = np.array(
        [
            float(DACH_BASE_XY[0]),
            float(DACH_BASE_XY[1]),
            float(current_base_position[2]),
        ],
        dtype=float,
    )
    if np.allclose(
        np.asarray(current_base_position, dtype=float),
        configured_base_position,
        atol=1e-6,
    ):
        print(f"✅ DACH 根节点位置已符合配置: {configured_base_position}")
    else:
        dach_root.set_world_pose(
            position=configured_base_position,
            orientation=current_base_orientation,
        )
        print(f"✅ DACH 根节点位置已配置: {configured_base_position}")
prepare_dach_finger_collision_instances(stage)
if state._reuse_dach_articulations:
    state.dach_arm = state._existing_dach_arm
    state.dach_left = state._existing_dach_left
    state.dach_right = (
        state._existing_dach_right
        if state._existing_dach_right is not None
        else (state.dach_arm if getattr(state.dach_arm, "arm_side", None) == "right" else None)
    )
    if state.dach_right is None:
        state.dach_right = DACHTron2AArm(
            prim_path=DACH_ARTICULATION_ROOT_PATH,
            name="dach_tron2a_right",
            arm_side="right",
        )
        state.dach_right.initialize(
            physics_sim_view=SimulationManager.get_physics_sim_view()
        )
    print("✅ 热重载复用已初始化的 DACH 双臂 articulation")
else:
    print("♻️ DACH articulation 句柄无效，重建 physics view...")
    if state.sim_context.is_playing():
        state.sim_context.stop()
        step_app(5)
    SimulationManager._on_stop(None)
    step_app(2)
    state.dach_left = DACHTron2AArm(
        prim_path=DACH_ARTICULATION_ROOT_PATH,
        name="dach_tron2a_left",
        arm_side="left",
    )
    state.dach_right = DACHTron2AArm(
        prim_path=DACH_ARTICULATION_ROOT_PATH,
        name="dach_tron2a_right",
        arm_side="right",
    )
    state.dach_arm = state.dach_left if DACH_ARM_SIDE == "left" else state.dach_right
    print("▶️ 启动物理并初始化 physics view...")
    state.sim_context.play()
    physics_sim_view = None
    articulation_metadata = None
    articulation_metadata_ready = False
    for warmup_attempt in range(1, 4):
        SimulationManager._warmup_needed = True
        SimulationManager.initialize_physics()
        for _ in range(40):
            step_app()
            physics_sim_view = SimulationManager.get_physics_sim_view()
            if (
                physics_sim_view is None
                or getattr(physics_sim_view, "_backend", None) is None
            ):
                continue
            articulation_probe = physics_sim_view.create_articulation_view(
                [DACH_ARTICULATION_ROOT_PATH]
            )
            articulation_metadata = getattr(
                articulation_probe, "shared_metatype", None
            )
            if (
                articulation_metadata is not None
                and getattr(articulation_metadata, "link_names", None)
                and getattr(articulation_metadata, "dof_names", None)
            ):
                articulation_metadata_ready = True
                break
        if articulation_metadata_ready:
            print(
                "✅ DACH tensor articulation metadata 已就绪: "
                f"warmup_attempt={warmup_attempt}"
            )
            break
    if not articulation_metadata_ready:
        raise RuntimeError(
            "DACH tensor articulation metadata 在 3 轮 PhysX warmup 后仍未就绪"
        )
    state.dach_left.initialize(physics_sim_view=physics_sim_view)
    state.dach_right.initialize(physics_sim_view=physics_sim_view)
    print("✅ DACH TRON2A 双臂初始化完成")

state._left_ik = DACHTron2AIKController(
    robot=state.dach_left,
    robot_description_path=DACH_ROBOT_DESCRIPTION_PATH,
    urdf_path=TRON2_URDF_PATH,
    rrt_config_path=DACH_LEFT_RRT_CONFIG_PATH,
)
print("✅ 左臂 IK 控制器创建完成")
state._right_ik = DACHTron2AIKController(
    robot=state.dach_right,
    robot_description_path=DACH_RIGHT_ROBOT_DESCRIPTION_PATH,
    urdf_path=TRON2_URDF_PATH,
    rrt_config_path=DACH_RIGHT_RRT_CONFIG_PATH,
)
print("✅ 右臂 IK 控制器创建完成")
_left_initial_state = state._preserved_arm_state.get("state.dach_left")
_right_initial_state = state._preserved_arm_state.get("state.dach_right")
if _left_initial_state is None and getattr(state._existing_dach_arm, "arm_side", None) == "left":
    _left_initial_state = state._preserved_arm_state.get("state.dach_arm")
if _right_initial_state is None and getattr(state._existing_dach_arm, "arm_side", None) == "right":
    _right_initial_state = state._preserved_arm_state.get("state.dach_arm")
if not state._reuse_dach_articulations:
    if _left_initial_state is not None:
        state.dach_left.teleport_arm_joint_positions(_left_initial_state["joints"])
        if _left_initial_state["gripper"] is not None:
            state.dach_left.gripper.teleport_joint_positions(_left_initial_state["gripper"])
    else:
        state.dach_left.teleport_arm_joint_positions(LEFT_ARM_HOME.copy())
        state.dach_left.gripper.teleport_joint_positions(state.GRIPPER_OPEN_POSITIONS)
    if _right_initial_state is not None:
        state.dach_right.teleport_arm_joint_positions(_right_initial_state["joints"])
        if _right_initial_state["gripper"] is not None:
            state.dach_right.gripper.teleport_joint_positions(_right_initial_state["gripper"])
    else:
        state.dach_right.teleport_arm_joint_positions(RIGHT_ARM_HOME.copy())
        state.dach_right.gripper.teleport_joint_positions(state.GRIPPER_OPEN_POSITIONS)
    print("🏠 已按左右臂分别恢复重载前关节姿态，未执行 HOME 归位")
elif (
    state.dach_left.get_joint_positions() is None
    or state.dach_right.get_joint_positions() is None
):
    state.dach_left.teleport_arm_joint_positions(LEFT_ARM_HOME.copy())
    state.dach_right.teleport_arm_joint_positions(RIGHT_ARM_HOME.copy())
    state.dach_left.gripper.teleport_joint_positions(state.GRIPPER_OPEN_POSITIONS)
    state.dach_right.gripper.teleport_joint_positions(state.GRIPPER_OPEN_POSITIONS)
    print("🏠 首次初始化关节状态无效，已使用双臂 HOME 姿态")
else:
    print("🏠 保留场景中的双臂当前姿态，未执行 HOME 归位")
for _ in range(3):
    step_app()
configure_dach_contact_physics(stage)

# Diffuser for dual-arm smooth trajectory synchronisation
DUAL_ARM_MIN_TCP_SEPARATION = float(
    os.environ.get("S5_DUAL_ARM_MIN_TCP_SEPARATION", "0.18")
)
state._diffuser = SparseKeyposeDiffuser(
    DiffusionConfig(
        cartesian_spacing_m=0.04,
        max_joint_step_rad=TRAJECTORY_MAX_JOINT_STEP,
        min_frames=TRAJECTORY_MIN_FRAMES,
        dual_arm_min_tcp_separation_m=DUAL_ARM_MIN_TCP_SEPARATION,
    )
)
print(f"✅ 双臂轨迹扩散器已初始化（最小TCP间距 {DUAL_ARM_MIN_TCP_SEPARATION} m）")

# 6. 创建仅控制左臂七轴的 Lula IK 控制器
if not TRON2_URDF_PATH.is_file():
    raise RuntimeError(f"DACH TRON2A URDF 不存在: {TRON2_URDF_PATH}")
print(f"📄 DACH TRON2A URDF: {TRON2_URDF_PATH}")
# The task layer may switch this alias after checking both arms.  Keep the
# configured side as the initial/default selection until the first task.
if DACH_ARM_SIDE == "left":
    state.controller = state._left_ik
    state.dach_arm = state.dach_left
else:
    state.controller = state._right_ik
    state.dach_arm = state.dach_right
print(
    "✅ DACH TRON2A Lula IK 控制器创建完成 "
    f"({state.dach_arm.end_effector_frame})"
)

state.planning_table_obstacle = None
state.planning_table_surface_z = None
state.planning_basket_obstacle = None

state.planning_basket_obstacle_enabled = False
if state.controller.rrt is not None:
    if BASKET_RESET_POSITION is not None:
        basket_prim_path = resolve_scene_prim_path("basket")
        basket_prim = SingleXFormPrim(
            name="s5_static_basket",
            prim_path=basket_prim_path,
        )
        basket_prim.set_world_pose(
            position=np.asarray(BASKET_RESET_POSITION, dtype=float),
            orientation=(
                None
                if BASKET_RESET_ORIENTATION is None
                else np.asarray(BASKET_RESET_ORIENTATION, dtype=float)
            ),
        )
        basket_root = stage.GetPrimAtPath(basket_prim_path)
        basket_rigid_body = UsdPhysics.RigidBodyAPI(basket_root)
        basket_rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        basket_rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        basket_rigid_body.CreateKinematicEnabledAttr().Set(True)
        get_app().update()
        print(
            "✅ 收纳箱已恢复并固定为运动学刚体: "
            f"position={BASKET_RESET_POSITION}"
        )
    planning_proxy_path = "/World/S5PlanningTableProxy"
    if stage.GetPrimAtPath(planning_proxy_path).IsValid():
        delete_prim(planning_proxy_path)
        get_app().update()
    table_center, table_min, table_max = get_bbox_center(
        stage,
        "/SimpleRoom/table_low_327",
    )
    state.planning_table_surface_z = float(table_max[2])
    planning_table_min = np.asarray(table_min, dtype=float)
    planning_table_max = np.asarray(table_max, dtype=float)
    planning_table_max[2] += (
        MIN_GRIPPER_TABLE_CLEARANCE + TABLE_CLEARANCE_ABORT_MARGIN
    )
    planning_table_center = (planning_table_min + planning_table_max) * 0.5
    state.planning_table_obstacle = VisualCuboid(
        prim_path=planning_proxy_path,
        name="s5_planning_table_proxy",
        position=planning_table_center,
        size=1.0,
        scale=np.maximum(planning_table_max - planning_table_min, 1e-3),
        color=np.array([0.0, 0.0, 0.0]),
    )
    UsdGeom.Imageable(stage.GetPrimAtPath(planning_proxy_path)).MakeInvisible()
    if state.controller.add_rrt_obstacle(state.planning_table_obstacle, static=True):
        print(
            "✅ DACH Lula RRT 已启用桌面碰撞障碍: "
            f"physical_top={state.planning_table_surface_z:.4f}, "
            f"planning_top={planning_table_max[2]:.4f}"
        )
    if state._left_ik.add_rrt_obstacle(state.planning_table_obstacle, static=True):
        print("✅ DACH 左臂 Lula RRT 已启用桌面碰撞障碍")
    if state._right_ik is not None and state._right_ik.add_rrt_obstacle(state.planning_table_obstacle, static=True):
        print("✅ DACH 右臂 Lula RRT 已启用桌面碰撞障碍")

    basket_proxy_path = "/World/S5PlanningBasketProxy"
    if stage.GetPrimAtPath(basket_proxy_path).IsValid():
        delete_prim(basket_proxy_path)
        get_app().update()
    _, basket_min, basket_max = get_bbox_center(
        stage,
        "/World/small_KLT_visual",
    )
    basket_min = np.asarray(basket_min, dtype=float) - BASKET_PLANNING_MARGIN
    basket_max = np.asarray(basket_max, dtype=float) + BASKET_PLANNING_MARGIN
    basket_center = (basket_min + basket_max) * 0.5
    state.planning_basket_obstacle = VisualCuboid(
        prim_path=basket_proxy_path,
        name="s5_planning_basket_proxy",
        position=basket_center,
        size=1.0,
        scale=np.maximum(basket_max - basket_min, 1e-3),
        color=np.array([0.0, 0.0, 0.0]),
    )
    UsdGeom.Imageable(stage.GetPrimAtPath(basket_proxy_path)).MakeInvisible()
    basket_added_right = state.controller.add_rrt_obstacle(
        state.planning_basket_obstacle,
        static=True,
    )
    basket_added_left = state._left_ik.add_rrt_obstacle(
        state.planning_basket_obstacle,
        static=True,
    )
    basket_added_right = bool(
        state._right_ik is not None
        and state._right_ik.add_rrt_obstacle(
            state.planning_basket_obstacle,
            static=True,
        )
    )
    # Lula enables an obstacle when it is added.  Preserve that state even if
    # a hot-reloaded controller reports that the same proxy already exists;
    # otherwise the first task retries enable_obstacle and raises
    # ``Attempted to enable an already-enabled obstacle``.
    state.planning_basket_obstacle_enabled = bool(
        basket_added_right or basket_added_left
    )
    if basket_added_right:
        print(
            "✅ DACH Lula RRT 已启用收纳箱碰撞障碍: "
            f"margin={BASKET_PLANNING_MARGIN:.3f} m, "
            f"min={basket_min}, max={basket_max}"
        )
    if basket_added_left:
        print("✅ DACH 左臂 Lula RRT 已启用收纳箱碰撞障碍")
    if basket_added_right:
        print("✅ DACH 右臂 Lula RRT 已启用收纳箱碰撞障碍")

# ── 7. 时间线 ────────────────────────────────────────────────────
state.timeline = omni.timeline.get_timeline_interface()
print("⏯️ 时间线已启动")

# 8. 稳定物理
print("🔄 稳定物理...")
step_app(50)
print("✅ 稳定完成")

# 9. 初始化相机；夹爪只在任务通过可达性预检后才动作。
state.grasp_camera = initialize_grasp_camera(stage)

state.arm_views = {
    "left": state.dach_left,
    "right": state.dach_right,
}
state.arm_controllers = {
    "left": state._left_ik,
    "right": state._right_ik or state.controller,
}
state.arm_fingers = {}
for _side, _suffix in (("left", "L"), ("right", "R")):
    state.arm_fingers[_side] = {
        "tcp": SingleXFormPrim(
            name=f"dach_{_side}_tcp",
            prim_path=f"/World/DACH_TRON2A/tcp_{_suffix}_Link",
        ),
        "left": SingleXFormPrim(
            name=f"s5_{_side}_gripper_left_finger",
            prim_path=f"/World/DACH_TRON2A/grasper_{_suffix}_jaw_left_Link",
        ),
        "right": SingleXFormPrim(
            name=f"s5_{_side}_gripper_right_finger",
            prim_path=f"/World/DACH_TRON2A/grasper_{_suffix}_jaw_right_Link",
        ),
    }
state.active_arm_side = DACH_ARM_SIDE
state.right_gripper = state.arm_fingers[DACH_ARM_SIDE]["tcp"]
state.left_finger = state.arm_fingers[DACH_ARM_SIDE]["left"]
state.right_finger = state.arm_fingers[DACH_ARM_SIDE]["right"]


# ── 10. 启动 Camera Bridge 与 Task Bridge ─────────────────────────
# Load AuraVLA bridges by file path. The VS Code Edition process can retain
# Eva-Agent's S5.communication modules across injections, which would silently
# write to /tmp/eva-agent-s5-control instead of AuraVLA's control directory.
_aura_camera_bridge_path = _AURA_ISAAC_BRIDGE_DIR / "start_camera_bridge.py"
_aura_task_bridge_path = (
    _AURA_ISAAC_BRIDGE_DIR.parent.parent
    / "aura_execution"
    / "aura_execution"
    / "task_bridge.py"
)
_aura_project_root = _AURA_ISAAC_BRIDGE_DIR.parent.parent
for _aura_python_path in (
    _aura_project_root / "aura_execution",
    _aura_project_root / "aura_orchestration",
    _aura_project_root / "aura_planning",
):
    if str(_aura_python_path) not in sys.path:
        sys.path.insert(0, str(_aura_python_path))
_camera_bridge_spec = importlib.util.spec_from_file_location(
    "aura_vla_camera_bridge", _aura_camera_bridge_path
)
if _camera_bridge_spec is None or _camera_bridge_spec.loader is None:
    raise RuntimeError(
        f"Unable to load AuraVLA camera bridge: {_aura_camera_bridge_path}"
    )
_camera_bridge_module = importlib.util.module_from_spec(_camera_bridge_spec)
sys.modules[_camera_bridge_spec.name] = _camera_bridge_module
_camera_bridge_spec.loader.exec_module(_camera_bridge_module)
start_camera_bridge = _camera_bridge_module.start_camera_bridge

_task_bridge_spec = importlib.util.spec_from_file_location(
    "aura_vla_task_bridge", _aura_task_bridge_path
)
if _task_bridge_spec is None or _task_bridge_spec.loader is None:
    raise RuntimeError(
        f"Unable to load AuraVLA task bridge: {_aura_task_bridge_path}"
    )
_task_bridge_module = importlib.util.module_from_spec(_task_bridge_spec)
sys.modules[_task_bridge_spec.name] = _task_bridge_module
_task_bridge_spec.loader.exec_module(_task_bridge_module)
start_task_bridge = _task_bridge_module.start_task_bridge

release_cuda_inference_cache()
state.camera_bridge = start_camera_bridge(
    camera_prim_path=CAMERA_PRIM_PATH,
    output_directory=os.environ.get("AURA_CAMERA_DIR", "/tmp/aura-vla-camera"),
)
print("✅ Camera Bridge 已启动")
state.task_bridge = start_task_bridge(execute_pick_place_with_reset)
print("✅ Task Bridge 已启动")
print("=== S5 机器人运行时就绪 ===")
