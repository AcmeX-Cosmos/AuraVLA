# SO101 仿真采集 → LeRobot → ACT

这个可选模块为 AuraVLA 增加固定桌面任务的自动示范采集：**抓取香蕉 → 抬升 → 放入盒子 → 复位 → 下一轮**。独立运行，不依赖 NVIDIA API、ROS2、AnyGrasp 或现有 DACH 控制器，也不创建或修改你的场景文件。

需要手工示范时，参见 [SO101 遥操作 Isaac Sim](teleoperation.md)：提供键盘控制和兼容 leisaac 官方发布器的 SO101 Leader 输入，可继续录制、导出并训练 ACT。

ACT 是用示范数据训练的策略。这里的自动示范来自一个读取 USD 关节运动学的数值 IK 控制器；训练完成前并没有 ACT 策略控制机械臂。本模块提供采集、数据转换和训练命令，暂不提供 ACT 在线推理或实体 SO101 控制。

## 1. 准备自己的场景

在 Isaac Sim 5.x 中配置并保存 USD 场景，然后填写配置。仓库内的路径和数值只是示例，需要用你的场景参数替换。

- 场景单位为米，Z 轴向上；SO101 基座固定，五个机械臂关节加一个夹爪关节均有有限限位和有效 position drive。关节名称按基座到手腕顺序填写。
- `robot.prim_path` 包含 articulation root。五个机械臂关节的 USD `body0 → body1` 构成串联链，允许链中存在 fixed joint。
- `end_effector_path` 指向固定手指所在的刚体 link；`tcp_offset_m` 是该 link 坐标系中的两指抓取中心。不能填写随夹爪开合移动的手指。
- `tool_axis` 是该 link 的接近方向，`approach_direction` 是世界系接近方向。SO101 为五自由度机械臂，控制器约束抓取点和接近方向，不额外强制完整六维末端姿态。
- 香蕉路径指向其**动态刚体根节点**，香蕉和夹爪需要碰撞体、合理质量及摩擦；香蕉保持物理抓取，不通过附着或瞬移模拟成功。香蕉初始姿态应适合指定抓取方向。
- 盒子需要静态碰撞体；盒子的 local +Z 向上，原点和朝向由你确定。`box_inner_bounds_m` 填写盒子**内部**的局部 XYZ 范围，不能使用包含侧壁的外部包围盒。
- 场景中已有相机，能看到香蕉、盒子及机器人运动区域。模块只为该相机创建采集 render product。

```bash
cd /path/to/AuraVLA
cp aura_lerobot/config/so101_banana.yaml aura_lerobot/config/so101_banana.local.yaml
# 编辑 local.yaml：stage_path、prim 路径、关节名、TCP、home、夹爪角度和盒内范围。
python3 -m aura_lerobot.collect --check-config \
  --config aura_lerobot/config/so101_banana.local.yaml
```

`--check-config` 检查文件路径和数值；场景 prim、刚体、关节链、drive 和限位会在 Isaac 启动后检查。

## 2. 先采一条，再连续采集

使用 **Isaac Sim 自带 Python** 运行，不要在普通 LeRobot Python 中导入 Isaac。此入口启动独立的 Isaac 实例；在资源有限的机器上先关闭已有 Isaac 窗口。

```bash
# 在 AuraVLA 根目录执行，将路径替换为你的 Isaac Sim 安装目录。
/path/to/isaacsim/python.sh -m aura_lerobot.collect \
  --config aura_lerobot/config/so101_banana.local.yaml --episodes 1
```

先观察一次物理抓取，校准 `home_rad`、`tcp_offset_m`、`grasp_offset_m`、夹爪开合角度和盒子范围。`recording.root` 必须为空或不存在；每次采集使用新目录，避免把不同配置的数据混在一起。

```bash
# 为本次采集设置新的 recording.root，再运行。
# 默认收集配置中的 100 条成功 episode；0 表示持续重复直到 Ctrl+C。
/path/to/isaacsim/python.sh -m aura_lerobot.collect \
  --config aura_lerobot/config/so101_banana.local.yaml --episodes 0
# 无窗口采集可追加 --headless。
```

每轮首先回到 home，再恢复香蕉的初始物理姿态和速度。复位不进入示范；只有实际抬升、运输期间未掉落、释放后香蕉包围盒完整进入盒内并连续稳定的 episode 才保存。失败记录写入 `attempts.jsonl`，失败图像清理；连续失败达到上限时停止，防止长期收集无效数据。Ctrl+C 保留完成的 episode，丢弃当前半条 episode。

数据为 30 FPS 的模拟时间采样，每帧执行 4 个 120 Hz 物理步，然后冻结物理并等待渲染管线返回对应时间的图像，消除 RTX 图像交付延迟。保存 `RGB_t`、实测 `joint_position_t` 与随后下发的绝对关节目标 `action_t`，所有六个关节均为**弧度**。相机缺帧或时间不同步会使本条 episode 失败。采集不要求机器实时跑到 30 FPS。

原始数据位于配置的 `recording.root`：

```text
session.json                       # 配置、关节顺序和单位
attempts.jsonl                     # 成功/失败及原因
episode_000000/
  episode.json                    # 成功标志、帧数和复位姿态
  frames.npz                      # state、action、timestamp、phase
  rgb/000000.png ...               # 连续 RGB 图像
```

该控制器使用预检过的悬停/抬升/越过盒沿路线和限速关节插值，**没有通用碰撞规划器**。你需要保证桌面路径没有额外障碍，校准 home 和运输高度并检查首条示范。运动学可达不代表整条关节插值路径无碰撞。

默认复位位置不随机，适合固定位置重复抓取。需要训练视觉泛化时，在已验证的抓取范围内逐步增大 `reset_xy_jitter_m`，建议分别采集不同香蕉姿态/场景配置并保持相同关节顺序和相机尺寸。重放同一轨迹很多次不能保证 ACT 对新位置泛化。

## 3. 转为 LeRobot 数据集

切换到已有 LeRobot 环境。导出使用官方 `LeRobotDataset.create/add_frame/save_episode/finalize` API，输出视频和 parquet 数据，保存成功 episode，并在 `meta/aura_source.json` 记录来源配置。无需上传 Hugging Face。

```bash
conda activate lerobot
# 若尚未安装：按 LeRobot 官方安装说明准备独立环境，然后安装 LeRobot。
# pip install 'lerobot>=0.4,<0.7'

python -m aura_lerobot.export \
  --raw-root data/so101_banana_raw \
  --root data/so101_banana_lerobot \
  --repo-id local/so101_banana
```

导出目录必须不存在。导出前检查所有帧数量、时间间隔和图像；`.partial` 目录不导出。只检查原始数据、不生成数据集时追加 `--validate-only`。支持 LeRobot v3 的数据集 API，实际导出/重载验证以本机 LeRobot 0.6.2 为准。

## 4. 训练 ACT

```bash
lerobot-train \
  --dataset.repo_id=local/so101_banana \
  --dataset.root="$PWD/data/so101_banana_lerobot" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=8 \
  --steps=50000 \
  --output_dir=outputs/act_so101_banana \
  --wandb.enable=false
```

图像键为 `observation.images.overhead`，状态键为 `observation.state`，动作键为 `action`。六维顺序固定为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`，映射到配置中相同顺序的仿真关节名。

`robot_type=so101_sim` 表明这些数据使用仿真弧度；实体 LeRobot SO101 通常使用标定后的角度或归一化位置，夹爪使用 0–100，**不能直接将本模块输出当作实体控制命令**。后续如需 ACT 仿真推理，应保持这里的六维弧度约定、相机键和控制频率。

## 验证

```bash
python3 -m unittest discover -s aura_lerobot/tests -v
```

测试覆盖运动学、IK 可达/不可达、轨迹速度限制、旋转盒内检测、失败/中断数据丢弃和采样校验。场景物理抓取效果需要在你提供的 SO101 场景中实测。

本机还完成了 LeRobot 0.6.2 的两条合成 episode 导出/重载、ACT 训练前向计算，以及独立 Isaac 六关节测试场景中的适配器初始化、运动学实测比对和 50 帧相机/状态/动作同步采集。这些检查不代表真实 SO101 场景抓取已经验证。
