from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import yaml


project_root = Path(
    os.getenv(
        "AURA_VLA_ROOT",
        "/home/acmex/Code/learning/courses/AuraVLA",
    )
).expanduser().resolve()
aura_directory = project_root
study_directory = str(aura_directory)
if study_directory not in sys.path:
    sys.path.insert(0, study_directory)

config_path = aura_directory / "aura_bringup" / "config" / "config.yaml"
settings = (
    yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config_path.is_file()
    else {}
)
camera_settings = settings.get("camera") or {}
camera_prim_path = str(
    camera_settings.get(
        "prim_path",
        "/World/DACH_TRON2A/head_pitch_Link/camera",
    )
)
os.environ["AURA_CAMERA_PRIM_PATH"] = camera_prim_path

# Import camera bridge from hardware package
sys.path.insert(0, str(aura_directory / "aura_hardware" / "aura_camera_bridge"))
from camera_bridge import start_camera_bridge


camera_bridge = start_camera_bridge(camera_prim_path=camera_prim_path)
