#!/usr/bin/env python3
"""Start the plan-only MoveIt 2 stack used by AuraVLA."""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[-1]


def _package_share_or_source(package_name: str, source_path: Path) -> Path:
    try:
        return Path(get_package_share_directory(package_name))
    except Exception:
        return source_path


def generate_launch_description():
    project_root = Path(os.environ.get("AURA_VLA_ROOT", os.getcwd())).expanduser()
    config_share = _package_share_or_source(
        "aura_moveit_config", project_root / "aura_moveit_config"
    )
    description_share = _package_share_or_source(
        "aura_robot_description", project_root / "aura_description"
    )
    urdf_path = _first_existing(
        Path(os.environ.get("AURA_TRON2_URDF_PATH", "")),
        project_root / "aura_description/urdf/tron2_v5_DACH_validing/robot.urdf",
        description_share / "urdf/tron2_v5_DACH_validing/robot.urdf",
    )
    srdf_path = config_share / "config/dach_tron2.srdf"

    robot_description = {"robot_description": urdf_path.read_text(encoding="utf-8")}
    robot_description_semantic = {
        "robot_description_semantic": srdf_path.read_text(encoding="utf-8")
    }
    common_parameters = [
        robot_description,
        robot_description_semantic,
        {"robot_description_kinematics": _load_yaml(config_share / "config/kinematics.yaml")},
        {"robot_description_planning": _load_yaml(config_share / "config/joint_limits.yaml")},
        {"ompl": _load_yaml(config_share / "config/ompl_planning.yaml")},
        _load_yaml(config_share / "config/planning_scene_monitor_parameters.yaml"),
        {"use_sim_time": LaunchConfiguration("use_sim_time")},
    ]

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=common_parameters + [
            {"allow_trajectory_execution": False},
            {"moveit_manage_controllers": False},
            {"trajectory_execution": {"allowed_execution_duration_scaling": 1.2}},
        ],
    )
    planner_adapter = Node(
        package="aura_isaac_bridge",
        executable="moveit_file_planner_node",
        output="screen",
        parameters=[
            {"request_directory": LaunchConfiguration("request_directory")},
            {"poll_period_sec": 0.02},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "request_directory",
            default_value="/tmp/aura-vla-control",
            description="AuraVLA Isaac/MoveIt JSON bridge directory",
        ),
        move_group,
        TimerAction(period=2.0, actions=[planner_adapter]),
    ])


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
