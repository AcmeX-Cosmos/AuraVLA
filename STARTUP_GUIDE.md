# AuraVLA 完整启动指南

## 🎯 系统架构

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  NVIDIA Agent   │────────▶│  Task Bridge     │────────▶│   Isaac Sim     │
│  (对话界面)      │         │  (文件通信)       │         │   (机械臂)       │
│  nvidia_agent   │         │ /tmp/aura-vla-   │         │ execute_pick_   │
│                 │         │     control/     │         │     place()     │
└─────────────────┘         └──────────────────┘         └─────────────────┘
      ↑                              ↑                           ↑
      │                              │                           │
   用户输入                      request.json                机械臂控制
   自然语言                      response.json               物体操作
```

---

## 📋 启动步骤

### 步骤 1：启动 Isaac Sim

1. 打开 NVIDIA Isaac Sim
2. 加载您的机器人场景（包含机械臂、相机、物体）
3. 确保场景中有以下物体（根据 config.yaml）：
   - `/World/banana` - 香蕉
   - `/World/small_KLT_visual` - 篮子/盒子
   - `/World/_37_scissors` - 剪刀
   - `/World/_02_master_chef_can` - 蓝色罐头
   - `/World/_05_tomato_soup_can` - 红白罐头
   - `/World/SM_Mug_A2` - 杯子

---

### 步骤 2：在 Isaac Sim 中启动任务桥接

**方法 A：在 Isaac Sim Python 控制台运行**

```python
exec(open('/home/acmex/Code/learning/courses/AuraVLA/scripts/start_isaac_bridge.py').read())
```

**方法 B：手动运行（如果您已有机械臂控制代码）**

```python
from aura_execution.task_bridge import start_task_bridge

def execute_pick_place(object_name: str, target_name: str):
    # 您的机械臂控制逻辑
    print(f"Picking {object_name}, placing on {target_name}")
    
    # TODO: 实现
    # 1. 获取物体位置
    # 2. 规划抓取路径
    # 3. 执行抓取
    # 4. 移动到目标
    # 5. 放置物体
    
    return True

# 启动桥接
bridge = start_task_bridge(execute_pick_place)
print("✓ Task bridge started")
```

您会看到：
```
✓ 任务桥接已启动

状态:
  - 请求文件: /tmp/aura-vla-control/request.json
  - 响应文件: /tmp/aura-vla-control/response.json
  - 状态文件: /tmp/aura-vla-control/status.json

现在可以从 NVIDIA Agent 发送任务了！
```

---

### 步骤 3：启动 NVIDIA Agent 对话界面

**在新终端运行：**

```bash
cd /home/acmex/Code/learning/courses/AuraVLA
source install/setup.bash
./scripts/start_nvidia_agent.sh
```

您会看到：

```
============================================================
AuraVLA - NVIDIA Agent Interactive Mode
============================================================

✓ NVIDIA API key loaded from config

🚀 Starting NVIDIA Agent...

Configuration:
  Config file: /home/acmex/Code/learning/courses/AuraVLA/aura_bringup/config/config.yaml
  Camera dir:  /tmp/aura-vla-camera
  Task dir:    /tmp/aura-vla-control

Features:
  ✓ Natural language understanding (Chinese/English)
  ✓ RGBD camera integration
  ✓ Scene analysis with VLM
  ✓ Direct Isaac Sim control

Quick commands:
  1 - 把香蕉放进篮子里
  2 - 把杯子放进篮子里
  3 - 把蓝色罐头放进篮子里
  7 - 现在画面中有什么物体

Special commands:
  /reload  - Reload VLM model
  /camera  - Refresh camera frame
  /help    - Show all commands

Type your instruction or number, Ctrl+C to quit
============================================================

🤖 >
```

---

## 💬 使用示例

### 快捷命令（输入数字）

```
🤖 > 1
[执行: 把香蕉放进篮子里]
[分析场景...]
[生成计划...]
[发送到 Isaac Sim...]
✓ 任务完成
```

### 自然语言（中文）

```
🤖 > 把剪刀放到左边
[分析场景...]
[识别物体: 剪刀, 左边区域]
[生成计划...]
[执行...]
✓ 任务完成
```

### 自然语言（英文）

```
🤖 > Pick up the blue can and place it in the basket
[Scene analysis...]
[Detected: blue can, basket]
[Planning...]
[Executing...]
✓ Task completed
```

### 场景查询

```
🤖 > 7
[或输入: 现在画面中有什么物体]

场景中的物体:
- 香蕉 (banana) - 黄色, 位于中间
- 篮子 (basket) - 紫色盒子, 位于右侧
- 剪刀 (scissors) - 银色, 位于左侧
- 蓝色罐头 (master_chef_can) - 蓝色, 位于桌面
- 杯子 (mug) - 绿色, 位于左侧
```

---

## 🔧 配置说明

配置文件位置：`aura_bringup/config/config.yaml`

### 关键配置项

```yaml
nvidia:
  api_key: "Bearer nvapi-xxx..."  # NVIDIA API 密钥
  model: "nvidia/nemotron-nano-12b-v2-vl"  # VLM 模型

camera:
  enabled: true
  directory: "/tmp/aura-vla-camera"  # 相机数据目录
  prim_path: "/World/DACH_TRON2A/head_pitch_Link/camera"  # Isaac 相机路径

execution:
  enabled: true
  task_directory: "/tmp/aura-vla-control"  # 任务桥接目录

scene:
  objects:  # 场景物体映射
    banana:
      prim_path: "/World/banana"
      aliases: ["香蕉", "黄色香蕉"]
```

---

## 🐛 故障排除

### 问题 1：Agent 提示 "Task bridge not ready"

**原因**：Isaac Sim 中的任务桥接未启动

**解决**：
```python
# 在 Isaac Sim Python 控制台运行
exec(open('/home/acmex/Code/learning/courses/AuraVLA/scripts/start_isaac_bridge.py').read())
```

### 问题 2：无法读取相机数据

**原因**：相机桥接未启动或路径配置错误

**解决**：
1. 检查 Isaac Sim 中相机 prim 路径是否正确
2. 启动相机桥接（如果独立使用）：
   ```python
   python3 scripts/start_camera_bridge.py
   ```

### 问题 3：VLM API 错误

**原因**：API key 无效或网络问题

**解决**：
1. 检查 config.yaml 中的 api_key
2. 测试网络连接：`curl -I https://integrate.api.nvidia.com`
3. 手动设置：`export NVIDIA_API_KEY="your-key"`

### 问题 4：物体名称无法识别

**原因**：VLM 识别的名称与 config.yaml 中的别名不匹配

**解决**：
在 config.yaml 的 `scene.objects` 中添加别名：
```yaml
scene:
  objects:
    banana:
      aliases: ["香蕉", "黄色香蕉", "banana", "yellow banana"]
```

---

## 📊 系统监控

### 检查任务桥接状态

```bash
cat /tmp/aura-vla-control/status.json
```

预期输出：
```json
{"ready":true,"state":"idle","updated_at_unix":1724012345.678}
```

### 监控任务请求

```bash
watch -n 1 "ls -lh /tmp/aura-vla-control/"
```

### 查看相机数据

```bash
ls -lh /tmp/aura-vla-camera/
# 应该看到: rgb.png, depth.npy, metadata.json
```

---

## 📚 下一步

1. **实现机械臂控制**：编辑 `scripts/start_isaac_bridge.py` 中的 `execute_pick_place()` 函数
2. **添加新物体**：在 `config.yaml` 的 `scene.objects` 中添加配置
3. **调整 VLM 参数**：修改 `config.yaml` 中的 `nvidia` 配置
4. **自定义快捷命令**：编辑 `config.yaml` 中的 `agent.quick_commands`

---

## 🆚 与 ROS 2 系统的对比

| 特性 | NVIDIA Agent (当前) | ROS 2 系统 |
|------|-------------------|-----------|
| 启动方式 | Isaac + Agent | `ros2 launch` |
| 通信方式 | 文件桥接 | ROS 2 topics/services |
| 对话界面 | ✓ 完整 VLM 对话 | ✗ 仅接收指令 |
| 场景理解 | ✓ RGBD + VLM | ○ 需要额外实现 |
| 闭环验证 | ✗ 无 | ✓ 自动验证 |
| 重规划 | ✗ 手动 | ✓ 自动重规划 |
| 适用场景 | 开发、演示、调试 | 生产、自动化 |

---

**维护者**: AcmeX <AcmeX@foxmail.com>
