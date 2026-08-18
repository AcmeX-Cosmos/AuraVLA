<p align="center">
  <h1 align="center">AuraVLA</h1>
  <p align="center">自主统一机器人智能体 - 视觉-语言-动作系统</p>
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

## 项目概述

AuraVLA 是一个专业的具身AI系统，实现了机器人操作任务的闭环感知-规划-执行控制。该系统无缝集成了视觉-语言理解、结构化任务规划、精确机器人控制和自动结果验证。

**核心技术栈：**
- **ROS2 Humble** - 机器人中间件
- **Python 3.10+** - 主要实现语言
- **NVIDIA Nemotron VLM** - 场景理解的视觉-语言模型
- **Isaac Sim** - 物理仿真和机器人控制
- **Schema验证** - 带约束检查的安全任务规划

---

## 核心特性

- **视觉-语言场景理解**：利用先进的VLM解释自然语言指令并分析视觉场景，实现智能任务理解
- **Schema验证的任务规划**：自动任务分解，支持JSON schema验证、禁止危险运动指令检测和安全约束强制执行
- **鲁棒的机器人执行**：基于文件的Isaac Sim通信协议，具有进度监控、超时处理和错误恢复机制
- **几何结果验证**：通过几何验证和空间关系分析自动检查任务完成情况
- **闭环编排**：自纠错控制系统，支持自动重规划、状态机管理和可配置的重试策略

---

## 系统架构

详细架构文档见 [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 核心模块

| 模块 | 文件路径 | 描述 |
|------|---------|------|
| **接口** | `aura_interfaces/` | 模块间通信的ROS2消息、服务和动作类型定义 |
| **感知** | `aura_perception/aura_perception/` | VLM客户端、可行性评估器、场景名称解析器和感知服务节点 |
| **规划** | `aura_planning/aura_planning/` | 任务规划器、schema验证器、动作分解器和规划服务节点 |
| **执行** | `aura_execution/aura_execution/` | 任务桥接、动作执行器和Isaac Sim控制的执行动作服务器 |
| **验证** | `aura_verification/aura_verification/` | 完成检查器、几何验证器和验证服务节点 |
| **编排** | `aura_orchestration/aura_orchestration/` | 主编排器、状态机和闭环协调节点 |
| **硬件** | `aura_hardware/` | 相机桥接和Isaac Sim桥接用于硬件接口抽象 |
| **启动** | `aura_bringup/` | 系统启动文件和配置管理用于完整系统启动 |
| **工具** | `aura_utils/aura_utils/` | 配置加载器、结构化日志器和通用工具函数 |
| **描述** | `aura_description/` | 机器人URDF文件、网格资源和运动学配置 |

---

## 项目结构

```
AuraVLA/
├── aura_interfaces/              # ROS2通信接口定义
│   ├── msg/                      # 消息类型定义
│   │   ├── TaskRequest.msg       # 带指令和场景的任务请求
│   │   ├── TaskPlan.msg          # 带动作的结构化任务计划
│   │   └── TaskAction.msg        # 单个动作规范
│   ├── srv/                      # 服务类型定义
│   │   ├── EvaluateDoable.srv    # 任务可行性评估
│   │   ├── GeneratePlan.srv      # 从请求生成计划
│   │   └── CheckCompletion.srv   # 任务完成验证
│   ├── action/                   # 动作类型定义
│   │   └── ExecuteTask.action    # 带反馈的任务执行
│   ├── CMakeLists.txt
│   └── package.xml
├── aura_perception/              # 视觉-语言感知模块
│   ├── aura_perception/
│   │   ├── vlm_client.py         # NVIDIA Nemotron VLM集成
│   │   ├── doable_evaluator.py   # 任务可行性评估
│   │   ├── scene_names.py        # 对象名称规范化
│   │   └── perception_node.py    # ROS2感知服务节点
│   ├── config/
│   │   └── perception.yaml       # VLM和感知参数
│   ├── package.xml
│   └── setup.py
├── aura_planning/                # 任务规划模块
│   ├── aura_planning/
│   │   ├── task_planner.py       # 高层任务分解
│   │   ├── schema_validator.py   # 安全性和结构验证
│   │   └── planning_node.py      # ROS2规划服务节点
│   ├── config/
│   │   └── planning.yaml         # 规划参数和约束
│   ├── package.xml
│   └── setup.py
├── aura_execution/               # 机器人动作执行模块
│   ├── aura_execution/
│   │   ├── task_bridge.py        # 基于文件的Isaac Sim通信
│   │   ├── action_executor.py    # 机器人动作控制
│   │   └── execution_node.py     # ROS2执行动作服务器
│   ├── config/
│   │   └── execution.yaml        # 执行和桥接参数
│   ├── package.xml
│   └── setup.py
├── aura_verification/            # 结果验证模块
│   ├── aura_verification/
│   │   ├── completion_checker.py # 几何验证逻辑
│   │   └── verification_node.py  # ROS2验证服务节点
│   ├── package.xml
│   └── setup.py
├── aura_orchestration/           # 闭环编排模块
│   ├── aura_orchestration/
│   │   ├── orchestrator.py       # 主闭环协调器
│   │   └── orchestration_node.py # ROS2编排节点
│   ├── package.xml
│   └── setup.py
├── aura_hardware/                # 硬件接口驱动
│   ├── aura_camera_bridge/       # RGBD相机接口
│   │   ├── camera_bridge_node.py
│   │   ├── package.xml
│   │   └── setup.py
│   └── aura_isaac_bridge/        # Isaac Sim场景接口
│       ├── isaac_bridge_node.py
│       ├── package.xml
│       └── setup.py
├── aura_bringup/                 # 系统启动和配置
│   ├── launch/
│   │   └── aura_system.launch.py # 完整系统启动
│   ├── config/
│   │   ├── system.yaml           # 基础系统配置
│   │   ├── dev.yaml              # 开发环境
│   │   └── prod.yaml             # 生产环境
│   ├── CMakeLists.txt
│   └── package.xml
├── aura_utils/                   # 工具库
│   ├── aura_utils/
│   │   ├── config_loader.py      # YAML配置管理
│   │   └── logger.py             # 结构化日志系统
│   ├── package.xml
│   └── setup.py
├── aura_description/             # 机器人URDF描述
│   ├── urdf/                     # 机器人模型文件
│   │   └── aura_robot.urdf
│   ├── meshes/                   # 视觉和碰撞网格
│   ├── config/
│   │   └── robot_config.yaml     # 机器人参数
│   ├── CMakeLists.txt
│   └── package.xml
├── README.md                     # 项目文档（英文）
├── README_zh.md                  # 项目文档（中文）
└── LICENSE                       # Apache 2.0许可证
```

---

## 安装部署

### 系统要求

- **操作系统**: Ubuntu 22.04 LTS
- **ROS2**: Humble Hawksbill ([安装指南](https://docs.ros.org/en/humble/Installation.html))
- **Python**: 3.10或更高版本
- **Isaac Sim**: 2023.1.0或更高版本（可选，用于仿真）

### 克隆仓库

详细安装说明见 [INSTALL.md](INSTALL.md)

### 安装依赖

```bash
# 安装ROS2依赖
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-image-transport

# 安装Python依赖
pip install numpy opencv-python pyyaml pillow
```

### 构建系统

```bash
# 使用colcon构建所有包
colcon build --symlink-install

# 加载工作空间
source install/setup.bash
```

---

## 使用说明

### 设置环境变量

```bash
# 必需：VLM的NVIDIA API密钥
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### 启动系统

```bash
# 启动完整的AuraVLA系统
ros2 launch aura_bringup aura_system.launch.py

# 使用自定义配置启动
ros2 launch aura_bringup aura_system.launch.py config_file:=/path/to/config.yaml
```

### 发送任务请求

```bash
# 示例：通过ROS2 topic发送拾取-放置任务
ros2 topic pub /aura/task_request aura_interfaces/msg/TaskRequest \
  "{instruction: '把香蕉放进篮子里'}"
```

### 系统组件

系统启动以下ROS2节点：

- **感知节点** (`aura_perception_node`): 提供 `/aura/perception/doable` 服务用于任务可行性评估
- **规划节点** (`aura_planning_node`): 提供 `/aura/planning/generate_plan` 服务用于任务计划生成
- **执行节点** (`aura_execution_node`): 提供 `/aura/execution/execute_task` 动作用于机器人控制
- **验证节点** (`aura_verification_node`): 提供 `/aura/verification/check_completion` 服务用于结果检查
- **编排节点** (`aura_orchestration_node`): 协调完整任务生命周期，支持自动重规划

所有节点输出到屏幕以便实时监控。

---

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

如有问题、issue或贡献：

- **GitHub Issues**: [https://github.com/AcmeX-Cosmos/AuraVLA/issues](https://github.com/AcmeX-Cosmos/AuraVLA/issues)
- **Email**: AcmeX@foxmail.com

---

<p align="center">
  <sub>基于ROS2构建，用于具身AI研究</sub>
</p>
