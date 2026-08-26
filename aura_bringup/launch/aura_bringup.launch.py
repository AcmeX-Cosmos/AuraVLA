#!/usr/bin/env python3
"""Unified AuraVLA startup, following the RCIA-vision bringup pattern.

Isaac Sim must already be open with the VS Code executor enabled. This launch
file injects AuraVLA into that process and starts the ROS graph and Foxglove
bridge. The interactive NVIDIA Agent remains an independent terminal process.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    project_root = os.environ.get("AURA_VLA_ROOT", os.getcwd())
    bringup_share = get_package_share_directory("aura_bringup")

    config_file = DeclareLaunchArgument(
        "config_file",
        default_value="",
        description="Optional AuraVLA YAML configuration file",
    )
    project_root_arg = DeclareLaunchArgument(
        "project_root",
        default_value=project_root,
        description="AuraVLA source workspace root for external launch scripts",
    )
    start_ros_system = DeclareLaunchArgument(
        "start_ros_system",
        default_value="true",
        description="Start AuraVLA perception/planning/execution ROS nodes",
    )
    start_isaac = DeclareLaunchArgument(
        "start_isaac",
        default_value="true",
        description="Inject the robot runtime into the existing Isaac Sim process",
    )
    start_foxglove = DeclareLaunchArgument(
        "start_foxglove",
        default_value="true",
        description="Start the Foxglove WebSocket bridge on port 8765",
    )
    start_moveit = DeclareLaunchArgument(
        "start_moveit",
        default_value="false",
        description="Start MoveIt 2 plan-only backend (requires MoveIt 2 installed)",
    )
    source_root = LaunchConfiguration("project_root")
    try:
        moveit_share = get_package_share_directory("aura_moveit_config")
    except Exception:
        moveit_share = os.path.join(project_root, "aura_moveit_config")
    isaac_script = PathJoinSubstitution([
        source_root, "aura_scripts", "start_isaac_robot.sh"
    ])
    system_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(bringup_share) / "launch" / "aura_system.launch.py")
        ),
        launch_arguments={
            "config_file": LaunchConfiguration("config_file"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_ros_system")),
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                moveit_share,
                "launch",
                "moveit.launch.py",
            )
        ),
        condition=IfCondition(LaunchConfiguration("start_moveit")),
    )

    isaac_runtime = ExecuteProcess(
        cmd=[isaac_script],
        name="aura_isaac_runtime_start",
        output="screen",
        additional_env={"AURA_VLA_ROOT": source_root},
        condition=IfCondition(LaunchConfiguration("start_isaac")),
    )

    foxglove_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "foxglove_bridge",
            "foxglove_bridge_launch.xml",
        ],
        name="aura_foxglove_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_foxglove")),
    )

    return LaunchDescription([
        config_file,
        project_root_arg,
        start_ros_system,
        start_isaac,
        start_foxglove,
        start_moveit,
        system_launch,
        moveit_launch,
        isaac_runtime,
        TimerAction(period=1.0, actions=[foxglove_bridge]),
    ])
