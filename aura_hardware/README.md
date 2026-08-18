# AuraVLA Hardware Drivers

This package contains hardware interface drivers for AuraVLA system.

## Subpackages

- `aura_camera_bridge`: RGBD camera interface with Isaac Sim
- `aura_isaac_bridge`: Scene observation and synchronization

## Usage

Launch camera bridge:
```bash
ros2 run aura_camera_bridge camera_bridge_node
```

Launch Isaac bridge:
```bash
ros2 run aura_isaac_bridge isaac_bridge_node
```
