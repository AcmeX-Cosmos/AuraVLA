# AuraVLA MoveIt 2 backend

This package provides the plan-only MoveIt 2 configuration for the DACH
TRON2A robot. Isaac Sim remains the trajectory executor. AnyGrasp, physical
finger contact checks, object retention checks, and gripper actuation remain
in `aura_isaac_bridge`.

## Prerequisites

Install the ROS 2 Humble MoveIt 2 packages before starting this backend:

```bash
sudo apt install ros-humble-moveit ros-humble-moveit-ros-move-group
```

The current workspace does not contain these packages, so MoveIt startup has
not been runtime-tested here.

## Start

Build and source the workspace, then start the normal AuraVLA launch with
MoveIt enabled:

```bash
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch aura_bringup aura_bringup.launch.py start_moveit:=true
```

The Isaac runtime reads `planner.backend: moveit` from
`aura_bringup/config/config.yaml`. When the MoveIt adapter is not running, it
falls back to Lula because `fallback_to_lula` is enabled. Set it to `false`
when a hard MoveIt-only deployment is required.

`AURA_TRON2_URDF_PATH` can point to the asset copy that contains the robot
meshes. The launch file otherwise uses the workspace URDF.
