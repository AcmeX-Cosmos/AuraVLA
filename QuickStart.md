# AuraVLA Quick Start

AuraVLA 使用已经打开的 Isaac Sim VS Code Edition 实例运行机械臂。启动脚本通过 VS Code executor（`127.0.0.1:8226`）注入运行时，不会启动第二个 Isaac Sim。

## 1. 一键启动完整系统

确认 Isaac Sim VS Code Edition 已打开，并加载包含 DACH TRON2A、相机、桌面、篮子和目标物体的场景。然后执行：

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch aura_bringup aura_bringup.launch.py
```

该 launch 会统一启动：

- AuraVLA ROS 2 感知、规划、执行、验证和编排节点
- Isaac Sim VS Code runtime 注入
- `aura_isaac_bridge_node`
- Foxglove WebSocket bridge
- 不接管标准输入的 NVIDIA Agent 独立终端

Foxglove Studio 连接 `ws://localhost:8765`。launch 启动完成后，另开一个终端启动交互式 NVIDIA Agent：

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
source /opt/ros/humble/setup.bash
source install/setup.bash
./aura_scripts/start_nvidia_agent.sh
```

调试时可关闭指定组件：

```bash
# 只启动 ROS 节点和 bridge，不重复注入 Isaac
ros2 launch aura_bringup aura_bringup.launch.py \
  start_isaac:=false

# Isaac、ROS 和 Foxglove 已经运行时，不重复启动任何组件；直接运行上面的 Agent 脚本
```

## 2. 手动启动 Isaac 运行时

先确认 Isaac Sim VS Code Edition 已打开，并加载包含 DACH TRON2A、相机、桌面、篮子和目标物体的场景。

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
./aura_scripts/start_isaac_robot.sh
```

看到以下信息后才继续下一步：

```text
=== AuraVLA 机器人运行时就绪 ===
Isaac runtime loaded in VS Code Edition
Isaac task bridge started: /tmp/aura-vla-control
```

## 3. 手动启动 NVIDIA Agent

打开新的终端执行：

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
source install/setup.bash
./aura_scripts/start_nvidia_agent.sh
```

Agent 启动后会显示：

```text
Isaac execution: enabled
Isaac camera: /tmp/aura-vla-camera
```

## 4. AnyGrasp 运行要求

AuraVLA 当前强制使用 AnyGrasp 生成抓取点。AnyGrasp SDK、Python 依赖和
官方 checkpoint 必须存在于 `aura_hardware/aura_isaac_bridge/thirdparty/anygrasp/`；模型不可用时任务会安全终止，
不会回退到场景几何抓取。

SDK 位于 `aura_hardware/aura_isaac_bridge/thirdparty/anygrasp/sdk`，其目录被 colcon 排除，仅由
Isaac Sim 运行时加载。首次配置时执行：

```bash
git clone https://github.com/graspnet/anygrasp_sdk.git \
  aura_hardware/aura_isaac_bridge/thirdparty/anygrasp/sdk
```

将官方 checkpoint 放入：

```text
aura_hardware/aura_isaac_bridge/thirdparty/anygrasp/checkpoint_detection.tar
```

模型路径可在 `aura_bringup/config/config.yaml` 的 `perception.anygrasp`
中覆盖。没有有效许可证或 checkpoint 时，任务会返回 `ANYGRASP_UNAVAILABLE`。

### 获取当前机器许可证 ID

使用 Isaac Sim Python 运行下面的脚本。脚本严格按 AnyGrasp 官方流程选择当前
Python ABI 对应的 `gsnet` 二进制，复制到 `license_registration/gsnet.so`，并
打印 feature ID；不会读取或修改任何许可证文件。必须在宿主机终端执行，不能
在沙箱、容器或受限网络环境中执行。

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
/home/acmex/Code/learning/isaacsim/python.sh \
  aura_scripts/get_anygrasp_feature_id.py
```

将输出的 `feature_id=N...` 值填入官方申请表。需要显式使用其他 Isaac Python
路径时可执行：

```bash
python3 aura_scripts/get_anygrasp_feature_id.py \
  --runtime-python /home/acmex/Code/learning/isaacsim/python.sh
```

不要使用 conda Python；注册和实际 AnyGrasp 推理必须使用同一 Isaac Python
运行时，并在同一宿主机网络环境执行。

AnyGrasp 还需要与 Isaac Python/PyTorch CUDA 版本匹配的 MinkowskiEngine。
如果运行时提示缺少该依赖，先安装 CUDA Toolkit（必须包含 `nvcc`），再执行：

```bash
bash aura_hardware/aura_isaac_bridge/thirdparty/anygrasp/install_minkowski_engine.sh
```

## 5. 快捷指令

在 NVIDIA Agent 的 `you>` 提示符输入：

| 输入 | 指令 |
| --- | --- |
| `1` | 把香蕉放进篮子里 |
| `2` | 把杯子放进篮子里 |
| `3` | 把蓝色罐头放进篮子里 |
| `4` | 把红白罐头放进篮子里 |
| `5` | `/reload`，重新注入当前 AuraVLA Isaac 运行时 |
| `6` | `/camera`，刷新并检查 Isaac 相机帧 |
| `7` | 现在画面中有什么物体 |

ROS 2 实时运输跟踪话题：

使用一键 launch 时，`aura_isaac_bridge_node` 已经自动启动，不要重复运行下面的
bridge 命令。仅在手动启动流程中执行：

```bash
# 新终端：启动 Isaac 状态与运输遥测 ROS 2 bridge
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run aura_isaac_bridge isaac_bridge_node --ros-args \
  -p check_rate:=5.0

# 另一个终端：订阅实时运输跟踪事件
ros2 topic echo /aura/transport_tracking std_msgs/msg/String
```

该话题会先发布 bridge 的 `waiting_for_event` 心跳；任务进入运输阶段后，
发布 AnyGrasp 运输校验、物体偏差、重规划请求以及重规划结果。

Foxglove Plot 可直接使用以下标准话题：

```text
/aura/anygrasp/position_error_m
/aura/anygrasp/confidence
/aura/anygrasp/observed_position
/aura/anygrasp/expected_position
/aura/anygrasp/position_error_vector_m
/aura/anygrasp/replan
/aura/anygrasp/state
```

启动 Foxglove ROS 2 bridge（默认 WebSocket `ws://localhost:8765`）：

```bash
./aura_scripts/start_foxglove.sh
```

该入口参考 RCIA-vision 的 `foxglove_bridge_launch.xml` 启动方式，但不调用
RCIA-vision 的文件或节点。若尚未安装 Foxglove bridge：

```bash
sudo apt install ros-humble-foxglove-bridge
```

然后在 Foxglove Studio 连接 `ws://localhost:8765`，打开 **Plot** 面板，
选择 `position_error_m`、`confidence` 或位置向量字段即可绘图。

遥测 bridge 每 5 秒发布一次心跳，并重发最近一次有效 AnyGrasp 样本；Foxglove
晚于任务启动连接时也能收到数据。新的任务执行后，Plot 会继续接收实时样本。

## 6. Agent 命令

```text
/isaac       恢复当前 Isaac Sim VS Code Edition 运行时
/reload      从当前 AuraVLA 源码重新加载运行时
/camera      设置并检查 Isaac 相机目录
/rgb PATH    使用指定 RGB 图片
/depth PATH  使用指定深度图片
/reset       清空当前会话状态
/history     查看当前对话历史
/quit        退出 NVIDIA Agent
```

## 7. 状态检查

检查 Task Bridge：

```bash
cat /tmp/aura-vla-control/status.json
```

正常状态应包含：

```json
{"ready": true, "state": "idle"}
```

检查相机帧：

```bash
ls -lh /tmp/aura-vla-camera/
```

目录应持续更新 RGB、深度和元数据文件。

检查 ROS 2 运输跟踪话题：

```bash
ros2 topic list | rg 'aura/transport_tracking'
ros2 topic echo /aura/transport_tracking std_msgs/msg/String
```

如果话题不存在，确认已通过 `aura_system.launch.py` 启动
`aura_isaac_bridge_node`，并在 Isaac Sim 中执行 `/reload`。

## 8. 常见问题

### AnyGrasp 不可用

检查模型目录和 checkpoint 配置，然后重新加载 Isaac 运行时：

```bash
./aura_scripts/start_isaac_robot.sh
```

任务响应会返回 `ANYGRASP_UNAVAILABLE`，在模型恢复前不会执行抓取。

### Isaac executor 不可用

确认 Isaac Sim VS Code Edition 已打开，并且 VS Code executor 正在监听 `127.0.0.1:8226`。然后重新执行：

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
./aura_scripts/start_isaac_robot.sh
```

### Camera bridge 没有新 RGBD 帧

先启动或重新加载 Isaac 运行时，再启动 NVIDIA Agent：

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
./aura_scripts/start_isaac_robot.sh
```

不要直接运行普通系统 Python 或 `/home/acmex/Code/learning/isaacsim/python.sh` 来代替 VS Code Edition 注入流程。

### Task Bridge 未就绪

检查状态文件：

```bash
cat /tmp/aura-vla-control/status.json
```

如果 `ready` 不是 `true`，重新执行 `./aura_scripts/start_isaac_robot.sh`，等待 `AuraVLA 机器人运行时就绪` 后再提交任务。

### 退出

在 NVIDIA Agent 中输入：

```text
/quit
```

Isaac Sim 本身保持打开，便于下一次使用 `/reload` 继续调试。
