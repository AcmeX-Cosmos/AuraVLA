from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import sys
import types

import yaml


PROJECT_ROOT = Path(
    os.getenv(
        "AURA_VLA_ROOT",
        "/home/acmex/Code/learning/courses/AuraVLA",
    )
).expanduser().resolve()
AURA_DIRECTORY = PROJECT_ROOT
CONFIG_PATH = AURA_DIRECTORY / "aura_bringup" / "config" / "config.yaml"
STUDY_DIRECTORY = str(AURA_DIRECTORY)
if STUDY_DIRECTORY not in sys.path:
    sys.path.insert(0, STUDY_DIRECTORY)

# Stop existing bridges if any
for bridge_name in ("camera_bridge", "task_bridge"):
    existing_bridge = globals().get(bridge_name)
    if existing_bridge is not None and hasattr(existing_bridge, "stop"):
        existing_bridge.stop()

importlib.invalidate_caches()

# Clear all Aura module caches
for module_name in sorted(
    list(sys.modules.keys()),
    key=lambda name: name.count("."),
    reverse=True,
):
    if not (module_name.startswith("aura_") or module_name == "aura"):
        continue
    module = sys.modules.pop(module_name, None)
    if module is None or "." not in module_name:
        continue
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, child_name, None) is module:
        try:
            delattr(parent, child_name)
        except AttributeError:
            pass

if CONFIG_PATH.is_file():
    settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    camera_settings = settings.get("camera") or {}
    perception_settings = settings.get("perception") or {}
    graspnet_calibration = perception_settings.get("graspnet_calibration") or {}
    robot_settings = settings.get("robot") or {}

    # Set environment variables from config
    os.environ["AURA_CAMERA_PRIM_PATH"] = str(
        camera_settings.get(
            "prim_path",
            "/World/DACH_TRON2A/head_pitch_Link/camera",
        )
    )
    os.environ["AURA_DACH_ARM_SIDE"] = str(
        robot_settings.get("arm_side", "right")
    ).strip().lower()
    os.environ["AURA_DACH_BASE_XY_JSON"] = json.dumps(
        robot_settings.get("base_xy"), ensure_ascii=False
    )
    os.environ["AURA_GRASPNET_CALIBRATION_ENABLED"] = str(
        bool(graspnet_calibration.get("enabled", False))
    ).lower()
    os.environ["AURA_GRASPNET_CAMERA_OFFSET_JSON"] = json.dumps(
        graspnet_calibration.get("camera_offset_m", [0.0, 0.0, 0.0])
    )
    os.environ["AURA_GRASPNET_CALIBRATION_MAX_CORRECTION_M"] = str(
        float(graspnet_calibration.get("max_correction_m", 0.06))
    )

    basket_settings = ((settings.get("scene") or {}).get("objects") or {}).get(
        "basket"
    ) or {}
    os.environ["AURA_BASKET_RESET_POSITION_JSON"] = json.dumps(
        basket_settings.get("reset_position"), ensure_ascii=False
    )
    os.environ["AURA_BASKET_RESET_ORIENTATION_JSON"] = json.dumps(
        basket_settings.get("reset_orientation"), ensure_ascii=False
    )

    # Robot configuration
    os.environ["AURA_DACH_GRASP_HEIGHT_OFFSET"] = str(
        float(robot_settings.get("grasp_height_offset", 0.015))
    )
    os.environ["AURA_DACH_GRASP_YAW_OFFSET_DEG"] = str(
        float(robot_settings.get("grasp_yaw_offset_deg", 0.0))
    )
    os.environ["AURA_BANANA_GRASP_TILT_DEG"] = str(
        float(robot_settings.get("banana_grasp_tilt_deg", 7.0))
    )
    os.environ["AURA_USE_GRASPNET"] = str(
        bool(robot_settings.get("use_graspnet", True))
    ).lower()
    os.environ["AURA_USE_GRASPNET_ORIENTATION"] = str(
        bool(robot_settings.get("use_graspnet_orientation", True))
    ).lower()

    os.environ["AURA_SCENE_OBJECTS_JSON"] = json.dumps(
        (settings.get("scene") or {}).get("objects") or {},
        ensure_ascii=False,
    )

print("=" * 80)
print("AuraVLA Isaac Sim Integration Loaded")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Config Path: {CONFIG_PATH}")
print(f"Camera Prim: {os.environ.get('AURA_CAMERA_PRIM_PATH', 'Not set')}")
print(f"Arm Side: {os.environ.get('AURA_DACH_ARM_SIDE', 'Not set')}")
print("=" * 80)
print("\nTo start camera bridge:")
print("  Run scripts/start_camera_bridge.py in Isaac Sim Script Editor")
print("\nTo run agent:")
print("  python3 nvidia_agent.py")
print("=" * 80)
