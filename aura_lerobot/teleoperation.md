# SO101 遥操作 Isaac Sim

参考 [leisaac 项目](https://github.com/LightwheelAI/leisaac)和[官方遥操作文档](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation)，新增 `aura_lerobot.teleop`。可以使用键盘，也可以接收官方 SO101 Leader 发布器的数据；操作自己的香蕉/盒子 USD 场景，并按需要录制 ACT 示范。

该入口复用 AuraVLA 的 SO101 场景配置、物理控制和相机同步。它不导入 leisaac 的 IsaacLab task，也不自动修改你的 USD 文件。此入口启动独立的 Isaac Sim 实例。

## 1. 配置场景

先按 [采集说明](README.md)配置 SO101、动态香蕉、静态盒子和相机，保存 USD 文件，并创建本地配置：

```bash
cd /path/to/AuraVLA
cp aura_lerobot/config/so101_banana.yaml aura_lerobot/config/so101_banana.local.yaml
# 已有 local.yaml 时直接编辑，不要覆盖。
```

填好 `stage_path`、关节与相机路径、TCP、home、夹爪角度、盒内范围。遥操作也使用这个香蕉入盒场景，不只要求机械臂存在。程序初始化会缓慢回到配置的 home，不会运行自动抓取路线。

`teleoperation` 配置可选；已有本地配置缺少这一段时使用默认值：

```yaml
teleoperation:
  translation_speed_m_s: 0.04
  wrist_speed_rad_s: 0.4
  remote_timeout_seconds: 0.5
  arm_range_rad: null
```

不录制时不会创建原始数据目录。加 `--record` 时，`recording.root` 必须为空或不存在；请为遥操作示范设置独立目录，例如 `data/so101_banana_teleop_raw`。

## 2. 键盘操作

使用 Isaac Sim 自带 Python，从 AuraVLA 根目录运行：

```bash
/path/to/isaacsim/python.sh -m aura_lerobot.teleop \
  --config aura_lerobot/config/so101_banana.local.yaml \
  --teleop-device keyboard
```

点击 Isaac 视口让窗口获得键盘焦点，按 **B** 开始控制。按键布局沿用 leisaac 的常用按键，但本适配器的平移使用**世界坐标**；腕部旋转直接控制关节，具体语义如下：

- **W / S**：TCP 沿世界 X 正 / 负方向移动。
- **A / D**：TCP 沿世界 Y 正 / 负方向移动。
- **Q / E**：TCP 沿世界 Z 上 / 下移动。
- **J / L**：`wrist_roll` 正 / 负方向旋转。
- **K / I**：`wrist_flex` 正 / 负方向旋转。
- **U / O**：打开 / 闭合夹爪。

按住移动键连续控制，松开停止增加目标。平移通过现有五自由度 IK 求解，腕部旋转允许改变接近姿态。不可达平移保持上一个目标并输出原因；所有输入都受关节限位和关节速度限制。

通用操作键：

- **B**：开始或恢复输入；启用 `--record` 时开始一条新示范，已有示范则继续。
- **SPACE**：暂停输入，保持最后目标；物理和当前示范采样继续，B 恢复。
- **R**：本条标记失败并丢弃，然后回 home、复位香蕉；再次 B 开始下一条。
- **N**：验证香蕉已在盒内稳定落下，成功时保存示范并复位。录制模式还要求本条曾实际抬起香蕉；不满足时保持本条，B 可继续操作。
- **BACKSPACE**：丢弃本条并停止输入，不复位场景。
- **ESC**：退出，保留已经完成的示范，丢弃未完成的示范。

先检查 home 和运动区域的碰撞情况。本控制器提供限位、限速和 IK 可达性检查，没有通用碰撞避障。

## 3. SO101 Leader 操作

采用官方的远程发布器，即使 Leader 和 Isaac 在同一台机器也分成两个进程，避免将串口驱动及其依赖装入仿真进程。需要真实 SO101 Leader；无需连接实体 Follower。

在外部目录准备 leisaac 源码，硬件端按官方说明安装 remote 依赖：

```bash
# 在独立 Python 环境中；替换为你的 leisaac 源码路径。
python -m pip install '/path/to/leisaac/source/leisaac[remote]'
```

硬件端启动官方发布器：

```bash
python /path/to/leisaac/scripts/environments/teleoperation/so101_joint_state_server.py \
  --port /dev/ttyACM0 --id aura_so101_leader \
  --bind tcp://127.0.0.1:5556 --rate 50
```

第一次使用会按发布器提示进行标定。需要重新标定时追加 `--recalibrate`；标定文件由官方发布器管理。串口路径由你的设备确定。

仿真端仅需要额外的 `pyzmq`：

```bash
/path/to/isaacsim/python.sh -m pip install pyzmq
/path/to/isaacsim/python.sh -m aura_lerobot.teleop \
  --config aura_lerobot/config/so101_banana.local.yaml \
  --teleop-device so101leader --remote-endpoint tcp://127.0.0.1:5556
```

收到第一条有效状态后，点击视口并按 B，手动移动 Leader 来控制仿真 SO101。N / R / SPACE 等流程键仍在 Isaac 窗口使用。

接收端兼容官方单帧 `<6f` 消息：固定顺序为五个手臂关节和夹爪；手臂是 `[-100,100]` 的归一化位置，夹爪是 `[0,100]`，不是角度或弧度。程序默认把手臂映射到对应 USD 关节范围，夹爪映射到配置的 closed/open 弧度，再限速后下发。

实际导入模型的零位、方向和运动范围可能不同，首次应确认各关节一一对应。可以用 `arm_range_rad` 覆盖五个手臂关节的映射端点，例如：

```yaml
# 以下仅为格式示例，数值需要根据你的模型填写，并位于 USD 限位内。
arm_range_rad:
  - [-1.5, 1.5]  # normalized -100 / +100 对应的 shoulder_pan 弧度
  - [1.2, -1.2]  # 反向映射示例
  - [-1.5, 1.5]
  - [-1.2, 1.2]
  - [-1.5, 1.5]
```

无数据时无法开始；超过 `remote_timeout_seconds` 未收到有效数据时停止跟随、保持最后目标并丢弃当前录制。网络恢复后需要重新按 B，不会突然自动恢复运动。畸形、非有限值或超出归一化范围的消息不用于控制。

断流发生在抓取中途时，恢复网络后先 R 复位，再 B 开始完整示范，避免从半途状态开始训练数据。

跨机器时把 `--bind` 改为该机器可被访问的地址，将仿真端 `--remote-endpoint` 指向 Leader 主机。也可以按官方文档使用 SSH 转发，使仿真端继续连接 localhost。这里只传控制状态，Isaac 视口的视频显示由你的远程桌面负责。

## 4. 录制并用于 ACT

在上述任一仿真启动命令末尾追加 `--record`：

```bash
/path/to/isaacsim/python.sh -m aura_lerobot.teleop \
  --config aura_lerobot/config/so101_banana.local.yaml \
  --teleop-device keyboard --record
```

B 开始采样，完成香蕉入盒后按 N 保存并复位，再次 B 开始下一条。不能仅按 N 就跳过物理成功检查。复位不会进入训练示范。输出沿用已有 `session.json + frames.npz + RGB PNG` 格式，记录的六维状态和动作仍为仿真弧度。

```bash
conda activate lerobot
python -m aura_lerobot.export \
  --raw-root data/so101_banana_teleop_raw \
  --root data/so101_banana_teleop_lerobot \
  --repo-id local/so101_banana_teleop
```

训练使用 [README.md](README.md) 的 ACT 命令，并将 dataset 的 repo_id/root 替换为这一份遥操作数据。录制数据来自自己的场景，与官方 leisaac 的 HDF5 recorder 不互换。

## 5. 可选：直接运行官方 leisaac 示例

如果你需要官方 IsaacLab task 和其 recorder，在已按官方 Getting Started 配好完整 leisaac / IsaacLab 环境的源码根目录运行：

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 --teleop_device=keyboard \
  --num_envs=1 --enable_cameras --record --dataset_file=./datasets/teleop.hdf5

# 对接前面的官方 Leader 发布器：
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 --teleop_device=so101leader \
  --remote_endpoint=tcp://127.0.0.1:5556 --num_envs=1 --enable_cameras
```

这是官方的橙子任务示例，使用其注册的 IsaacLab 环境；AuraVLA 自己的香蕉/盒子 USD 场景使用前面的 `aura_lerobot.teleop` 入口。这里的命令作为参考，不会自动安装或启动官方示例。

## 验证范围

本机在 Isaac 环境中通过 26 项测试，覆盖消息格式、归一化到弧度的映射、反向映射、限速、录制状态切换、断流丢弃和 USD 包围盒回归。独立六关节测试场景还验证了原生键盘订阅及按键事件处理、17 帧同步遥操作录制、保存/复位，以及模拟官方消息格式的 ZMQ 跟随。

```bash
python3 -m unittest discover -s aura_lerobot/tests -v
```

普通 Python 缺少 `pyzmq` 或 Isaac 的 `pxr` 时，会跳过对应集成测试；USD 测试在已初始化的 Isaac 环境中执行。这些验证不包含真实 SO101 Leader 串口标定、你的最终香蕉场景物理抓取，或官方 IsaacLab 示例运行。
