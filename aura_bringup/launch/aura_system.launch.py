#!/usr/bin/env python3
"""
AuraVLA System Launch File

Launches the complete AuraVLA system with all nodes.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for AuraVLA system"""

    # Declare arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to configuration file'
    )

    # Perception Node
    perception_node = Node(
        package='aura_perception',
        executable='perception_node',
        name='aura_perception_node',
        parameters=[{'config_file': LaunchConfiguration('config_file')}],
        output='screen'
    )

    # Planning Node
    planning_node = Node(
        package='aura_planning',
        executable='planning_node',
        name='aura_planning_node',
        parameters=[{'config_file': LaunchConfiguration('config_file')}],
        output='screen'
    )

    # Execution Node
    execution_node = Node(
        package='aura_execution',
        executable='execution_node',
        name='aura_execution_node',
        parameters=[{'config_file': LaunchConfiguration('config_file')}],
        output='screen'
    )

    # Verification Node
    verification_node = Node(
        package='aura_verification',
        executable='verification_node',
        name='aura_verification_node',
        output='screen'
    )

    # Orchestration Node
    orchestration_node = Node(
        package='aura_orchestration',
        executable='orchestration_node',
        name='aura_orchestration_node',
        parameters=[{'max_replans': 2}],
        output='screen'
    )

    return LaunchDescription([
        config_file_arg,
        perception_node,
        planning_node,
        execution_node,
        verification_node,
        orchestration_node,
    ])
