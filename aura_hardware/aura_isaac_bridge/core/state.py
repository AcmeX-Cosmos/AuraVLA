"""AuraVLA Isaac runtime configuration and mutable state."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ── 环境变量 / 路径 ──────────────────────────────────────────────
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("AURA_VLA_ROOT", DEFAULT_PROJECT_ROOT)).expanduser().resolve()
DEFAULT_ISAAC_SIM_ROOT = Path("/home/acmex/Code/learning/isaacsim")
ISAAC_SIM_ROOT = Path(os.environ.get("ISAAC_SIM_ROOT", DEFAULT_ISAAC_SIM_ROOT)).expanduser().resolve()
DEFAULT_ISAAC_SITE_PACKAGES = (
    ISAAC_SIM_ROOT / "kit" / "python" / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
ISAAC_SITE_PACKAGES = Path(
    os.environ.get("ISAAC_PYTHON_SITE_PACKAGES", DEFAULT_ISAAC_SITE_PACKAGES)
).expanduser().resolve()
STUDY_DIR = PROJECT_ROOT
AURA_DIR = PROJECT_ROOT / "aura_bringup"
AURA_ASSET_ROOT = Path(
    os.environ.get("AURA_ASSET_ROOT", "/home/acmex/Code/learning/isaac_assets")
).expanduser().resolve()
SECTION3_DIR = AURA_ASSET_ROOT

_DEFAULT_TRON2_URDF_CANDIDATES = (
    AURA_ASSET_ROOT / "tron2_v5_DACH_validing" / "robot.urdf",
    PROJECT_ROOT / "aura_description" / "urdf" / "tron2_v5_DACH_validing" / "robot.urdf",
)
TRON2_URDF_PATH = Path(os.environ.get("AURA_TRON2_URDF_PATH", next((str(path) for path in _DEFAULT_TRON2_URDF_CANDIDATES if path.is_file()), str(_DEFAULT_TRON2_URDF_CANDIDATES[-1])))).expanduser().resolve()
DACH_ROBOT_DESCRIPTION_PATH = AURA_DIR / "config" / "aura_dach_tron2a_robot_description.yaml"
DACH_RIGHT_ROBOT_DESCRIPTION_PATH = AURA_DIR / "config" / "aura_dach_tron2a_right_robot_description.yaml"
DACH_LEFT_RRT_CONFIG_PATH = AURA_DIR / "config" / "aura_dach_tron2a_left_rrt.yaml"
DACH_RIGHT_RRT_CONFIG_PATH = AURA_DIR / "config" / "aura_dach_tron2a_right_rrt.yaml"

GRASP_BACKEND = os.environ.get("AURA_GRASP_BACKEND", "graspnet").strip().lower()
if GRASP_BACKEND not in {"anygrasp", "graspnet"}:
    raise RuntimeError(
        "AURA_GRASP_BACKEND 必须是 anygrasp 或 graspnet: "
        f"{GRASP_BACKEND}"
    )

DEFAULT_ANYGRASP_DIR = (
    PROJECT_ROOT / "aura_hardware" / "aura_isaac_bridge" / "thirdparty" / "anygrasp" / "sdk"
)
ANYGRASP_DIR = Path(
    os.environ.get(
        "ANYGRASP_DIR",
        DEFAULT_ANYGRASP_DIR,
    )
).expanduser().resolve()
DEFAULT_ANYGRASP_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "aura_hardware"
    / "aura_isaac_bridge"
    / "thirdparty"
    / "anygrasp"
    / "checkpoint_detection.tar"
)
ANYGRASP_CHECKPOINT_PATH = Path(
    os.environ.get(
        "ANYGRASP_CHECKPOINT_PATH",
        DEFAULT_ANYGRASP_CHECKPOINT_PATH,
    )
).expanduser().resolve()

DEFAULT_GRASPNET_DIR = SECTION3_DIR / "thirdparty" / "graspnet-baseline"
GRASPNET_DIR = Path(
    os.environ.get("GRASPNET_BASELINE_DIR", DEFAULT_GRASPNET_DIR)
).expanduser().resolve()
GRASPNET_CHECKPOINT_PATH = Path(
    os.environ.get("GRASPNET_CHECKPOINT_PATH", GRASPNET_DIR / "checkpoint-rs.tar")
).expanduser().resolve()
GRASPNET_NUM_POINT = max(
    int(os.environ.get("AURA_GRASPNET_NUM_POINT", "8000")), 2048
)
DEFAULT_SAM_MODEL_PATH = SECTION3_DIR / "sam2.1_b.pt"
SAM_MODEL_PATH = os.environ.get(
    "SAM_MODEL_PATH", str(DEFAULT_SAM_MODEL_PATH) if DEFAULT_SAM_MODEL_PATH.is_file() else "sam2.1_b.pt"
)

# ── 场景 / 相机 ──────────────────────────────────────────────────
CAMERA_PRIM_PATH = os.environ.get("AURA_CAMERA_PRIM_PATH", "/World/DACH_TRON2A/head_pitch_Link/camera")
CAMERA_RESOLUTION = (640, 360)
CAMERA_PREVIEW_RESOLUTION = (640, 360)
SHOW_GRASP_DEBUG = False
ANYGRASP_REQUIRED = os.environ.get(
    "AURA_ANYGRASP_REQUIRED",
    "true" if GRASP_BACKEND == "anygrasp" else "false",
).strip().lower() in {"true", "1", "yes", "on"}
USE_ANYGRASP = ANYGRASP_REQUIRED or (
    os.environ.get(
        "AURA_USE_ANYGRASP",
        "true" if GRASP_BACKEND == "anygrasp" else "false",
    ).strip().lower()
    in ("true", "1", "yes", "on")
)
USE_ANYGRASP_ORIENTATION = os.environ.get(
    "AURA_USE_ANYGRASP_ORIENTATION",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
GRASPNET_REQUIRED = os.environ.get(
    "AURA_GRASPNET_REQUIRED", "true" if GRASP_BACKEND == "graspnet" else "false"
).strip().lower() in {"true", "1", "yes", "on"}
USE_GRASPNET = GRASPNET_REQUIRED or (
    os.environ.get(
        "AURA_USE_GRASPNET",
        "true" if GRASP_BACKEND == "graspnet" else "false",
    ).strip().lower()
    in {"true", "1", "yes", "on"}
)
USE_GRASPNET_ORIENTATION = os.environ.get(
    "AURA_USE_GRASPNET_ORIENTATION", "false"
).strip().lower() in {"1", "true", "yes", "on"}
DACH_SCENE_ROOT_PATH = "/World/DACH_TRON2A"
DACH_ARTICULATION_ROOT_PATH = "/World/DACH_TRON2A/root_joint"

# ── 机械臂 ───────────────────────────────────────────────────────
DACH_ARM_SIDE = os.environ.get("AURA_DACH_ARM_SIDE", "right").strip().lower()
if DACH_ARM_SIDE not in {"left", "right"}:
    raise RuntimeError(f"AURA_DACH_ARM_SIDE 必须是 left 或 right: {DACH_ARM_SIDE}")

# 从 DACH TRON2A 模块导入的 HOME / GRIPPER_OPEN
# 这些在 robot/robot.py 初始化时通过 from aura_isaac_bridge.robot.dach_tron2a import ... 获取
# 这里先设默认值，初始化时会被覆盖
# 注意: GRIPPER_OPEN_POSITIONS 和 DACH_HOME_JOINT_POSITIONS
# 在运行时由 robot/robot.py 设置到 state 对象上 (state.GRIPPER_OPEN_POSITIONS)

DACH_BASE_XY = json.loads(os.environ.get("AURA_DACH_BASE_XY_JSON", "null"))
if DACH_BASE_XY is not None and len(DACH_BASE_XY) != 2:
    raise RuntimeError("robot.base_xy 必须是 [x, y] 或 null")

DACH_GRASP_HEIGHT_OFFSET = float(os.environ.get("AURA_DACH_GRASP_HEIGHT_OFFSET", "0.05"))
DACH_GRASP_YAW_OFFSET_RAD = np.radians(float(os.environ.get("AURA_DACH_GRASP_YAW_OFFSET_DEG", "0.0")))
DACH_OPEN_GRIPPER_CENTER_LOCAL_OFFSET = np.array([0.22457, 0.0, -0.03170], dtype=float)
DACH_JAW_COLLISION_LOCAL_BOUNDS = {
    "left": (
        np.array([-0.015001, 0.004043, -0.182661], dtype=float),
        np.array([0.015001, 0.031479, -0.104075], dtype=float),
    ),
    "right": (
        np.array([-0.015001, -0.031479, -0.182661], dtype=float),
        np.array([0.015001, -0.004043, -0.104075], dtype=float),
    ),
}

# ── 抓取参数 ─────────────────────────────────────────────────────
BANANA_GRASP_TILT_RAD = np.radians(float(os.environ.get("AURA_BANANA_GRASP_TILT_DEG", "30.0")))
BANANA_NEAR_SIDE_OFFSET = float(os.environ.get("AURA_BANANA_NEAR_SIDE_OFFSET", "0.0"))
BANANA_MIN_SHORT_AXIS_ALIGNMENT = float(os.environ.get("AURA_BANANA_MIN_SHORT_AXIS_ALIGNMENT", "0.92"))
GRASP_POSITION_OFFSET = np.array([0.0, 0.0, -0.01])
GRASP_INSERT_DEPTH = 0.015


def _perception_calibration_vec3(name: str, default) -> np.ndarray:
    try:
        vector = np.asarray(json.loads(os.environ.get(name, json.dumps(default))), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{name} 必须是三个数值组成的 JSON 数组") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"{name} 必须是三个有限数值组成的 JSON 数组")
    return vector


ANYGRASP_CALIBRATION_ENABLED = os.environ.get(
    "AURA_ANYGRASP_CALIBRATION_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
ANYGRASP_CAMERA_OFFSET = _perception_calibration_vec3(
    "AURA_ANYGRASP_CAMERA_OFFSET_JSON", [0.0, 0.0, 0.0]
)
ANYGRASP_CALIBRATION_MAX_CORRECTION = max(
    float(
        os.environ.get(
            "AURA_ANYGRASP_CALIBRATION_MAX_CORRECTION_M",
            "0.06",
        )
    ),
    0.0,
)
ANYGRASP_FUSION_FRAME_COUNT = max(
    int(os.environ.get("AURA_ANYGRASP_FUSION_FRAME_COUNT", "3")), 1
)
ANYGRASP_FUSION_FRAME_INTERVAL_SEC = max(
    float(os.environ.get("AURA_ANYGRASP_FUSION_FRAME_INTERVAL_SEC", "0.12")), 0.0
)
ANYGRASP_FUSION_MAX_POSITION_DISPERSION_M = max(
    float(os.environ.get("AURA_ANYGRASP_FUSION_MAX_POSITION_DISPERSION_M", "0.025")),
    0.001,
)
ANYGRASP_FUSION_MAX_ORIENTATION_DISPERSION_DEG = max(
    float(os.environ.get("AURA_ANYGRASP_FUSION_MAX_ORIENTATION_DISPERSION_DEG", "25.0")),
    0.1,
)
ANYGRASP_FUSION_POSITION_OUTLIER_FLOOR_M = max(
    float(os.environ.get("AURA_ANYGRASP_FUSION_POSITION_OUTLIER_FLOOR_M", "0.012")),
    0.001,
)
ANYGRASP_FUSION_MIN_CONFIDENCE = float(
    np.clip(float(os.environ.get("AURA_ANYGRASP_FUSION_MIN_CONFIDENCE", "0.03")), 0.0, 1.0)
)

GRASPNET_CALIBRATION_ENABLED = os.environ.get(
    "AURA_GRASPNET_CALIBRATION_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
GRASPNET_CAMERA_OFFSET = _perception_calibration_vec3(
    "AURA_GRASPNET_CAMERA_OFFSET_JSON", [0.0, 0.0, 0.0]
)
GRASPNET_CALIBRATION_MAX_CORRECTION = max(
    float(os.environ.get("AURA_GRASPNET_CALIBRATION_MAX_CORRECTION_M", "0.06")),
    0.0,
)
GRASPNET_FUSION_FRAME_COUNT = max(
    int(os.environ.get("AURA_GRASPNET_FUSION_FRAME_COUNT", "3")), 1
)
GRASPNET_FUSION_FRAME_INTERVAL_SEC = max(
    float(os.environ.get("AURA_GRASPNET_FUSION_FRAME_INTERVAL_SEC", "0.12")), 0.0
)
GRASPNET_FUSION_MAX_POSITION_DISPERSION_M = max(
    float(os.environ.get("AURA_GRASPNET_FUSION_MAX_POSITION_DISPERSION_M", "0.025")),
    0.001,
)
GRASPNET_FUSION_MAX_ORIENTATION_DISPERSION_DEG = max(
    float(os.environ.get("AURA_GRASPNET_FUSION_MAX_ORIENTATION_DISPERSION_DEG", "25.0")),
    0.1,
)
GRASPNET_FUSION_POSITION_OUTLIER_FLOOR_M = max(
    float(os.environ.get("AURA_GRASPNET_FUSION_POSITION_OUTLIER_FLOOR_M", "0.012")),
    0.001,
)
GRASPNET_FUSION_MIN_CONFIDENCE = float(
    np.clip(float(os.environ.get("AURA_GRASPNET_FUSION_MIN_CONFIDENCE", "0.10")), 0.0, 1.0)
)

# Desktop pick-and-place should remain predominantly top-down. The launcher
# may override these values from config, but standalone runtime defaults must
# keep the same conservative posture constraint.
MAX_GRASP_APPROACH_TILT_RAD = np.radians(float(os.environ.get("AURA_MAX_GRASP_APPROACH_TILT_DEG", "15.0")))
CAN_MAX_GRASP_APPROACH_TILT_RAD = np.radians(
    float(os.environ.get("AURA_CAN_MAX_GRASP_APPROACH_TILT_DEG", "30.0"))
)
TARGET_GRASP_APPROACH_TILT_RAD = np.radians(float(os.environ.get("AURA_TARGET_GRASP_APPROACH_TILT_DEG", "10.0")))
GRASP_REFINEMENT_STEPS = max(int(os.environ.get("AURA_GRASP_REFINEMENT_STEPS", "0")), 0)
BANANA_GRIPPER_CLOSE_POSITION = float(os.environ.get("AURA_BANANA_GRIPPER_CLOSE_POSITION", "0.0"))
BANANA_PLANAR_REFINEMENT_STEPS = max(int(os.environ.get("AURA_BANANA_PLANAR_REFINEMENT_STEPS", "2")), 0)
BANANA_PLANAR_CENTER_TOLERANCE = float(os.environ.get("AURA_BANANA_PLANAR_CENTER_TOLERANCE", "0.008"))
BANANA_MAX_PLANAR_CORRECTION = float(os.environ.get("AURA_BANANA_MAX_PLANAR_CORRECTION", "0.04"))

# ── 放置参数 ─────────────────────────────────────────────────────
DIRECTIONAL_PLACE_DISTANCE = float(os.environ.get("AURA_DIRECTIONAL_PLACE_DISTANCE", "0.35"))
GRASP_MIN_HEIGHT_FRACTION = float(os.environ.get("AURA_GRASP_MIN_HEIGHT_FRACTION", "0.10"))
MINIMUM_OBJECT_LIFT = float(os.environ.get("AURA_MINIMUM_OBJECT_LIFT", "0.08"))
PLACE_SUCCESS_TOLERANCE = float(os.environ.get("AURA_PLACE_SUCCESS_TOLERANCE", "0.18"))
# Horizontal error alone accepted "dropped on the table next to the basket".
# The object must also sit clearly above the tabletop to count as contained.
PLACE_MIN_HEIGHT_ABOVE_TABLE = float(
    os.environ.get("AURA_PLACE_MIN_HEIGHT_ABOVE_TABLE", "0.02")
)
BASKET_RESET_POSITION = json.loads(os.environ.get("AURA_BASKET_RESET_POSITION_JSON", "null"))
BASKET_RESET_ORIENTATION = json.loads(os.environ.get("AURA_BASKET_RESET_ORIENTATION_JSON", "null"))
BASKET_PLANNING_MARGIN = max(float(os.environ.get("AURA_BASKET_PLANNING_MARGIN", "0.03")), 0.0)
BASKET_PLACE_TABLE_CLEARANCE = max(
    float(os.environ.get("AURA_BASKET_PLACE_TABLE_CLEARANCE", "0.003")),
    0.0,
)

# ── 轨迹规划 ─────────────────────────────────────────────────────
DACH_PATH_CLEARANCE = float(os.environ.get("AURA_PATH_CLEARANCE", "0.10"))
TRAJECTORY_MAX_JOINT_STEP = float(os.environ.get("AURA_TRAJECTORY_MAX_JOINT_STEP", "0.050"))
TRAJECTORY_MIN_FRAMES = max(int(os.environ.get("AURA_TRAJECTORY_MIN_FRAMES", "6")), 2)
TRAJECTORY_SETTLE_FRAMES = max(int(os.environ.get("AURA_TRAJECTORY_SETTLE_FRAMES", "4")), 0)
GRASP_APPROACH_MAX_JOINT_STEP = max(float(os.environ.get("AURA_GRASP_APPROACH_MAX_JOINT_STEP", "0.018")), 0.001)
GRASP_APPROACH_MIN_FRAMES = max(int(os.environ.get("AURA_GRASP_APPROACH_MIN_FRAMES", "24")), 4)
GRASP_LIFT_MAX_JOINT_STEP = max(
    float(os.environ.get("AURA_GRASP_LIFT_MAX_JOINT_STEP", "0.03")), 0.001
)
GRASP_LIFT_MIN_FRAMES = max(
    int(os.environ.get("AURA_GRASP_LIFT_MIN_FRAMES", "20")), 20
)
ACTION_WAYPOINT_LIMIT = max(int(os.environ.get("AURA_ACTION_WAYPOINT_LIMIT", "3")), 1)
CARTESIAN_WAYPOINT_SPACING = max(float(os.environ.get("AURA_CARTESIAN_WAYPOINT_SPACING", "0.04")), 0.01)
CARRY_APEX_CLEARANCE = float(os.environ.get("AURA_CARRY_APEX_CLEARANCE", "0.15"))
TRANSPORT_LIFT_HEIGHT = max(
    float(os.environ.get("AURA_TRANSPORT_LIFT_HEIGHT", "0.16")),
    0.08,
)
CARRY_CARTESIAN_WAYPOINT_SPACING = max(
    float(os.environ.get("AURA_CARRY_CARTESIAN_WAYPOINT_SPACING", "0.12")),
    0.04,
)
CARRY_MAX_JOINT_STEP = max(
    float(os.environ.get("AURA_CARRY_MAX_JOINT_STEP", "0.016")), 0.001
)
CARRY_MIN_FRAMES = max(
    int(os.environ.get("AURA_CARRY_MIN_FRAMES", "20")), 20
)
CARRY_REPLAN_CHECK_WAYPOINTS = max(
    int(os.environ.get("AURA_CARRY_REPLAN_CHECK_WAYPOINTS", "4")), 1
)
CARRY_REPLAN_POSITION_TOLERANCE_M = max(
    float(os.environ.get("AURA_CARRY_REPLAN_POSITION_TOLERANCE_M", "0.02")),
    0.001,
)
CARRY_REPLAN_MAX_ATTEMPTS = max(
    int(os.environ.get("AURA_CARRY_REPLAN_MAX_ATTEMPTS", "3")), 0
)
CARRY_REPLAN_FRAME_COUNT = max(
    int(os.environ.get("AURA_CARRY_REPLAN_FRAME_COUNT", "2")), 1
)
MIN_GRIPPER_TABLE_CLEARANCE = float(os.environ.get("AURA_MIN_GRIPPER_TABLE_CLEARANCE", "0.012"))
TABLE_CLEARANCE_ABORT_MARGIN = float(os.environ.get("AURA_TABLE_CLEARANCE_ABORT_MARGIN", "0.006"))
DUAL_ARM_MIN_TCP_SEPARATION = 0.10

# ── 夹爪 ─────────────────────────────────────────────────────────
GRIPPER_CLOSE_FRAMES = max(int(os.environ.get("AURA_GRIPPER_CLOSE_FRAMES", "20")), 1)
GRIPPER_MAX_EFFORT = float(os.environ.get("AURA_GRIPPER_MAX_EFFORT", "3.0"))
GRIPPER_STIFFNESS = float(os.environ.get("AURA_GRIPPER_STIFFNESS", "250.0"))
GRIPPER_DAMPING = float(os.environ.get("AURA_GRIPPER_DAMPING", "8.0"))
GRIPPER_CONTACT_RESIDUAL = float(os.environ.get("AURA_GRIPPER_CONTACT_RESIDUAL", "0.0015"))
GRIPPER_CONTACT_FORCE_THRESHOLD = max(float(os.environ.get("AURA_GRIPPER_CONTACT_FORCE_THRESHOLD", "0.25")), 0.0)
GRIPPER_CONTACT_PRELOAD_RESIDUAL = max(float(os.environ.get("AURA_GRIPPER_CONTACT_PRELOAD_RESIDUAL", "0.0015")), GRIPPER_CONTACT_RESIDUAL)
GRIPPER_CONTACT_HOLD_PRELOAD = max(float(os.environ.get("AURA_GRIPPER_CONTACT_HOLD_PRELOAD", "0.003")), 0.0)
GRIPPER_CONTACT_SETTLE_FRAMES = max(int(os.environ.get("AURA_GRIPPER_CONTACT_SETTLE_FRAMES", "15")), 0)
# Exiting the preload loop on a single qualifying frame let a transient effort
# spike stand in for a real grip.
GRIPPER_PRELOAD_CONFIRM_FRAMES = max(
    int(os.environ.get("AURA_GRIPPER_PRELOAD_CONFIRM_FRAMES", "3")), 1
)
# ── 物理 / 摩擦 ──────────────────────────────────────────────────
BANANA_STATIC_FRICTION = float(os.environ.get("AURA_BANANA_STATIC_FRICTION", "1.2"))
BANANA_DYNAMIC_FRICTION = float(os.environ.get("AURA_BANANA_DYNAMIC_FRICTION", "1.0"))
GRIPPER_STATIC_FRICTION = float(os.environ.get("AURA_GRIPPER_STATIC_FRICTION", "6.0"))
GRIPPER_DYNAMIC_FRICTION = float(os.environ.get("AURA_GRIPPER_DYNAMIC_FRICTION", "5.0"))
PHYSX_CONTACT_OFFSET = float(os.environ.get("AURA_PHYSX_CONTACT_OFFSET", "0.003"))
PHYSX_REST_OFFSET = float(os.environ.get("AURA_PHYSX_REST_OFFSET", "0.001"))
PHYSX_SOLVER_POSITION_ITERATIONS = int(os.environ.get("AURA_PHYSX_SOLVER_POSITION_ITERATIONS", "32"))
PHYSX_SOLVER_VELOCITY_ITERATIONS = int(os.environ.get("AURA_PHYSX_SOLVER_VELOCITY_ITERATIONS", "8"))
PHYSX_MAX_DEPENETRATION_VELOCITY = float(os.environ.get("AURA_PHYSX_MAX_DEPENETRATION_VELOCITY", "0.2"))
# CCD is opt-in. Global CCD plus per-link CCD is unstable for articulated
# grippers in several Isaac Sim releases, so contact offset + TGS is default.
PHYSX_ENABLE_CCD = os.environ.get("AURA_PHYSX_ENABLE_CCD", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
PHYSX_CONVEX_HULL_VERTEX_LIMIT = int(os.environ.get("AURA_PHYSX_CONVEX_HULL_VERTEX_LIMIT", "64"))
PHYSX_CONVEX_MAX_HULLS = int(os.environ.get("AURA_PHYSX_CONVEX_MAX_HULLS", "16"))
PHYSX_CONVEX_MIN_THICKNESS = float(os.environ.get("AURA_PHYSX_CONVEX_MIN_THICKNESS", "0.001"))
PHYSX_CONVEX_SHRINK_WRAP = os.environ.get("AURA_PHYSX_CONVEX_SHRINK_WRAP", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
PHYSX_CONVEX_ERROR_PERCENTAGE = float(os.environ.get("AURA_PHYSX_CONVEX_ERROR_PERCENTAGE", "0.1"))
# /PhysicsScene previously inherited stage defaults for all of these, so the
# substep rate physics ran at was unrelated to the declared physics_dt.
PHYSX_TIME_STEPS_PER_SECOND = int(os.environ.get("AURA_PHYSX_TIME_STEPS_PER_SECOND", "60"))
PHYSX_SOLVER_TYPE = os.environ.get("AURA_PHYSX_SOLVER_TYPE", "TGS").strip().upper()
if PHYSX_SOLVER_TYPE not in {"TGS", "PGS"}:
    raise RuntimeError(f"AURA_PHYSX_SOLVER_TYPE 必须是 TGS 或 PGS: {PHYSX_SOLVER_TYPE}")
PHYSX_BOUNCE_THRESHOLD_VELOCITY = float(
    os.environ.get("AURA_PHYSX_BOUNCE_THRESHOLD_VELOCITY", "0.05")
)

# ── 全局可变状态容器 ──────────────────────────────────────────────
state = SimpleNamespace(
    sim_context=None,
    stage=None,
    dach_arm=None,
    dach_left=None,
    dach_right=None,
    controller=None,
    _left_ik=None,
    _right_ik=None,
    active_arm_side=None,
    arm_controllers=None,
    arm_views=None,
    arm_fingers=None,
    _diffuser=None,
    grasp_camera=None,
    _task_motion_started=False,
    TARGET_OBJECT_PRIM_PATH="/World/banana",
    planning_table_obstacle=None,
    planning_table_surface_z=None,
    planning_basket_obstacle=None,
    planning_basket_obstacle_enabled=False,
    _anygrasp_model=None,
    _anygrasp_imported=False,
    _graspnet_demo=None,
    _graspnet_net=None,
    _graspnet_imported=False,
    _sam_model=None,
    _camera_preview_window=None,
    _camera_preview_provider=None,
    right_gripper=None,
    left_finger=None,
    right_finger=None,
    _preserved_arm_state=None,
    _reuse_dach_articulations=False,
    _existing_dach_arm=None,
    _existing_dach_left=None,
    timeline=None,
    camera_bridge=None,
    GRIPPER_OPEN_POSITIONS=None,
    DACH_HOME_JOINT_POSITIONS=None,
    task_bridge=None,
    SCENE_NAME_RESOLVER=None,
)
