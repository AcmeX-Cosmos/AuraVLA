# AuraVLA - Isaac Sim 使用说明

## VS Code 中使用 Isaac Sim Extension

### 1. 打开工作区
在 VS Code 中打开工作区文件：
```bash
code AuraVLA.code-workspace
```

### 2. 在 Isaac Sim Script Editor 中运行 main.py

#### 方法一：使用 Isaac Sim VS Code Extension
1. 确保已安装 Isaac Sim 4.2.0 或更高版本
2. 在 VS Code 中打开 `main.py`
3. 使用 Isaac Sim Extension 将代码发送到 Script Editor

#### 方法二：直接在 Isaac Sim 中运行
1. 打开 Isaac Sim
2. 打开 Script Editor (Window -> Script Editor)
3. 复制 `main.py` 的内容并粘贴到 Script Editor
4. 点击 Run 按钮执行

### 3. 启动相机桥接
在 Isaac Sim Script Editor 中运行：
```python
exec(open('/home/acmex/Code/learning/courses/AuraVLA/scripts/start_camera_bridge.py').read())
```

### 4. 运行对话代理
在终端中运行：
```bash
python3 nvidia_agent.py
```

## 快捷命令

在 nvidia_agent.py 对话界面中可以使用：

- `1` - 把香蕉放进篮子里
- `2` - 把杯子放进篮子里
- `3` - 把蓝色罐头放进篮子里
- `4` - 把红白罐头放进篮子里
- `5` - `/reload` 重新加载
- `6` - `/camera` 刷新相机
- `7` - 现在画面中有什么物体

## 系统命令

- `/camera` - 刷新相机画面
- `/isaac` - 检查 Isaac 运行状态
- `/reload` - 重新加载 Isaac 运行时
- `/reset` - 重置对话历史
- `/history` - 显示对话历史
- `/quit` - 退出程序

## 项目结构

```
AuraVLA/
├── main.py                          # Isaac Sim 入口文件
├── nvidia_agent.py                  # 对话代理入口
├── scripts/
│   └── start_camera_bridge.py      # 相机桥接启动脚本
├── aura_bringup/
│   └── config/
│       └── config.yaml             # 系统配置文件
├── aura_hardware/
│   └── eva_camera_bridge/
│       └── camera_bridge.py        # 相机桥接实现
├── aura_perception/
│   └── aura_perception/
│       └── eva_perception/
│           └── nvidia_agent.py     # VLM 对话系统
└── .vscode/
    ├── settings.json               # VS Code 配置
    └── launch.json                 # 调试配置
```

## 环境变量

系统会自动从 `config.yaml` 加载以下环境变量：

- `AURA_CAMERA_PRIM_PATH` - 相机路径
- `AURA_DACH_ARM_SIDE` - 机械臂侧面 (left/right)
- `AURA_DACH_BASE_XY` - 机器人基座位置
- `AURA_USE_GRASPNET` - 是否使用 GraspNet
- `AURA_SCENE_OBJECTS_JSON` - 场景物体配置

## 配置文件

主配置文件位于 `aura_bringup/config/config.yaml`，包含：

- NVIDIA API 配置
- 相机设置
- 机器人参数
- 场景物体定义
- PhysX 物理参数

## 故障排除

### Isaac Sim 无法连接
确保 Isaac Sim 正在运行，并且已在 Script Editor 中执行 `main.py`

### 相机图像无法获取
1. 检查相机路径是否正确：`/World/DACH_TRON2A/head_pitch_Link/camera`
2. 确保已运行 `scripts/start_camera_bridge.py`
3. 检查 `/tmp/aura-vla-camera` 目录是否有图像文件

### VLM 对话无响应
1. 检查 NVIDIA API 密钥是否有效
2. 确认网络连接正常
3. 查看 `config.yaml` 中的 API 配置
