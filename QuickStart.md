# AuraVLA Quick Start

AuraVLA 使用已经打开的 Isaac Sim VS Code Edition 实例运行机械臂。启动脚本通过 VS Code executor（`127.0.0.1:8226`）注入运行时，不会启动第二个 Isaac Sim。

## 1. 启动 Isaac 运行时

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

## 2. 启动 NVIDIA Agent

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

## 3.1 GraspNet 运行要求

AuraVLA 当前强制使用 GraspNet 生成抓取点。GraspNet baseline、Python 依赖和
`checkpoint-rs.tar` 必须存在于配置的模型目录中；模型不可用时任务会安全终止，
不会回退到场景几何抓取。

## 4. 快捷指令

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

## 5. Agent 命令

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

## 6. 状态检查

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

## 7. 常见问题

### GraspNet 不可用

检查模型目录和 checkpoint 配置，然后重新加载 Isaac 运行时：

```bash
./aura_scripts/start_isaac_robot.sh
```

任务响应会返回 `GRASPNET_UNAVAILABLE`，在模型恢复前不会执行抓取。

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
