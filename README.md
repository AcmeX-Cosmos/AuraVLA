<p align="center">
  <h1 align="center">AuraVLA</h1>
  <p align="center">Autonomous Unified Robotic Agent with Vision-Language-Action</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/ROS2-Humble-blue.svg" alt="ROS2">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build">
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

---

## Overview

AuraVLA is a embodied AI system implementing closed-loop perception-planning-execution control for robotic manipulation tasks. The system seamlessly integrates vision-language understanding with structured task planning, precise robot control, and automatic result verification.

**Core Technology Stack:**
- **ROS2 Humble** - Robotics middleware
- **Python 3.10+** - Primary implementation language
- **NVIDIA Nemotron VLM** - Vision-language model for scene understanding
- **Isaac Sim** - Physics simulation and robot control
- **Schema-based Validation** - Safe task planning with constraint checking

---

## Features

- **Vision-Language Scene Understanding**: Leverages state-of-the-art VLM to interpret natural language instructions and analyze visual scenes for intelligent task comprehension
- **Schema-Validated Task Planning**: Automatic task decomposition with JSON schema validation, forbidden motion command detection, and safety constraint enforcement
- **Robust Robot Execution**: File-based Isaac Sim communication protocol with progress monitoring, timeout handling, and error recovery mechanisms
- **Geometric Result Verification**: Automated task completion checking through geometric validation and spatial relationship analysis
- **Closed-Loop Orchestration**: Self-correcting control system with automatic replanning, state machine management, and configurable retry strategies

---

## Architecture

See detailed architecture documentation in [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Core Modules

| Module | File Path | Description |
|--------|-----------|-------------|
| **Interfaces** | `aura_interfaces/` | ROS2 message, service, and action type definitions for inter-module communication |
| **Perception** | `aura_perception/aura_perception/` | VLM client, doability evaluator, scene name resolver, and perception service node |
| **Planning** | `aura_planning/aura_planning/` | Task planner, schema validator, action decomposer, and planning service node |
| **Execution** | `aura_execution/aura_execution/` | Task bridge, action executor, and execution action server for Isaac Sim control |
| **Verification** | `aura_verification/aura_verification/` | Completion checker, geometric verifier, and verification service node |
| **Orchestration** | `aura_orchestration/aura_orchestration/` | Main orchestrator, state machine, and closed-loop coordination node |
| **Hardware** | `aura_hardware/` | Camera bridge and Isaac Sim bridge for hardware interface abstraction |
| **Bringup** | `aura_bringup/` | System launch files and configuration management for complete system startup |
| **Utils** | `aura_utils/aura_utils/` | Configuration loader, structured logger, and common utility functions |
| **Description** | `aura_description/` | Robot URDF files, mesh resources, and kinematic configurations |

---

## Project Structure

```
AuraVLA/
├── aura_interfaces/              # ROS2 communication interface definitions
│   ├── msg/                      # Message type definitions
│   │   ├── TaskRequest.msg       # Task request with instruction and scene
│   │   ├── TaskPlan.msg          # Structured task plan with actions
│   │   └── TaskAction.msg        # Individual action specification
│   ├── srv/                      # Service type definitions
│   │   ├── EvaluateDoable.srv    # Task feasibility evaluation
│   │   ├── GeneratePlan.srv      # Plan generation from request
│   │   └── CheckCompletion.srv   # Task completion verification
│   ├── action/                   # Action type definitions
│   │   └── ExecuteTask.action    # Task execution with feedback
│   ├── CMakeLists.txt
│   └── package.xml
├── aura_perception/              # Vision-language perception module
│   ├── aura_perception/
│   │   ├── vlm_client.py         # NVIDIA Nemotron VLM integration
│   │   ├── doable_evaluator.py   # Task feasibility assessment
│   │   ├── scene_names.py        # Object name canonicalization
│   │   └── perception_node.py    # ROS2 perception service node
│   ├── config/
│   │   └── perception.yaml       # VLM and perception parameters
│   ├── package.xml
│   └── setup.py
├── aura_planning/                # Task planning module
│   ├── aura_planning/
│   │   ├── task_planner.py       # High-level task decomposition
│   │   ├── schema_validator.py   # Safety and structure validation
│   │   └── planning_node.py      # ROS2 planning service node
│   ├── config/
│   │   └── planning.yaml         # Planning parameters and constraints
│   ├── package.xml
│   └── setup.py
├── aura_execution/               # Robot action execution module
│   ├── aura_execution/
│   │   ├── task_bridge.py        # File-based Isaac Sim communication
│   │   ├── action_executor.py    # Robot action control
│   │   └── execution_node.py     # ROS2 execution action server
│   ├── config/
│   │   └── execution.yaml        # Execution and bridge parameters
│   ├── package.xml
│   └── setup.py
├── aura_verification/            # Result verification module
│   ├── aura_verification/
│   │   ├── completion_checker.py # Geometric verification logic
│   │   └── verification_node.py  # ROS2 verification service node
│   ├── package.xml
│   └── setup.py
├── aura_orchestration/           # Closed-loop orchestration module
│   ├── aura_orchestration/
│   │   ├── orchestrator.py       # Main closed-loop coordinator
│   │   └── orchestration_node.py # ROS2 orchestration node
│   ├── package.xml
│   └── setup.py
├── aura_hardware/                # Hardware interface drivers
│   ├── aura_camera_bridge/       # RGBD camera interface
│   │   ├── camera_bridge_node.py
│   │   ├── package.xml
│   │   └── setup.py
│   └── aura_isaac_bridge/        # Isaac Sim scene interface
│       ├── isaac_bridge_node.py
│       ├── package.xml
│       └── setup.py
├── aura_bringup/                 # System launch and configuration
│   ├── launch/
│   │   └── aura_system.launch.py # Complete system startup
│   ├── config/
│   │   ├── system.yaml           # Base system configuration
│   │   ├── dev.yaml              # Development environment
│   │   └── prod.yaml             # Production environment
│   ├── CMakeLists.txt
│   └── package.xml
├── aura_utils/                   # Utility libraries
│   ├── aura_utils/
│   │   ├── config_loader.py      # YAML configuration management
│   │   └── logger.py             # Structured logging system
│   ├── package.xml
│   └── setup.py
├── aura_description/             # Robot URDF descriptions
│   ├── urdf/                     # Robot model files
│   │   └── aura_robot.urdf
│   ├── meshes/                   # Visual and collision meshes
│   ├── config/
│   │   └── robot_config.yaml     # Robot parameters
│   ├── CMakeLists.txt
│   └── package.xml
├── README.md                     # Project documentation
├── README_zh.md                  # Chinese documentation
└── LICENSE                       # Apache 2.0 license
```

---

## Installation

### Prerequisites

- **Operating System**: Ubuntu 22.04 LTS
- **ROS2**: Humble Hawksbill ([installation guide](https://docs.ros.org/en/humble/Installation.html))
- **Python**: 3.10 or higher
- **Isaac Sim**: 2023.1.0 or higher (optional, for simulation)

### Clone Repository

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Install Dependencies

```bash
# Install ROS2 dependencies
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-image-transport

# Install Python dependencies
pip install numpy opencv-python pyyaml pillow
```

### Build System

```bash
# Build all packages with colcon
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

---

## Usage

### Set Environment Variables

```bash
# Required: NVIDIA API key for VLM
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### Launch System

```bash
# Start complete AuraVLA system
ros2 launch aura_bringup aura_system.launch.py

# Launch with custom configuration
ros2 launch aura_bringup aura_system.launch.py config_file:=/path/to/config.yaml
```

### Send Task Request

```bash
# Example: Pick-and-place task via ROS2 topic
ros2 topic pub /aura/task_request aura_interfaces/msg/TaskRequest \
  "{instruction: 'Put the banana in the basket'}"
```

### System Components

The system launches the following ROS2 nodes:

- **Perception Node** (`aura_perception_node`): Provides `/aura/perception/doable` service for task feasibility evaluation
- **Planning Node** (`aura_planning_node`): Provides `/aura/planning/generate_plan` service for task plan generation
- **Execution Node** (`aura_execution_node`): Provides `/aura/execution/execute_task` action for robot control
- **Verification Node** (`aura_verification_node`): Provides `/aura/verification/check_completion` service for result checking
- **Orchestration Node** (`aura_orchestration_node`): Coordinates the complete task lifecycle with automatic replanning

All nodes output to screen for real-time monitoring.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or contributions:

- **GitHub Issues**: [https://github.com/AcmeX-Cosmos/AuraVLA/issues](https://github.com/AcmeX-Cosmos/AuraVLA/issues)
- **Email**: AcmeX@foxmail.com

---

<p align="center">
  <sub>Built with ROS2 for Embodied AI Research</sub>
</p>
