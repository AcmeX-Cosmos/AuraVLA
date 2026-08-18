#!/usr/bin/env python3
"""
AuraVLA - NVIDIA VLM Agent 对话系统

在终端独立运行此文件，启动交互式VLM对话系统。

使用方式:
1. 确保 main.py 已在 Isaac Sim VS Code Extension 中运行
2. 在终端运行: python nvidia_agent.py
3. 在交互界面中输入自然语言指令

示例指令:
- 把香蕉放进篮子里
- 把杯子移到篮子
- 场景里有什么物体？

快捷命令:
- /camera  : 刷新相机画面
- /isaac   : 检查Isaac运行状态
- /reload  : 重新加载Isaac运行时
- /reset   : 重置对话历史
- /quit    : 退出程序
"""

import sys
from pathlib import Path

# 设置项目根目录
_AURA_ROOT = Path(__file__).resolve().parent

# 添加所有必要的模块路径
_paths_to_add = [
    _AURA_ROOT,
    _AURA_ROOT / "aura_perception" / "aura_perception",
    _AURA_ROOT / "aura_hardware" / "aura_camera_bridge",
    _AURA_ROOT / "aura_hardware" / "aura_isaac_bridge",
    _AURA_ROOT / "aura_execution" / "aura_execution" / "aura_execution",
    _AURA_ROOT / "aura_planning" / "aura_planning" / "aura_planning",
]

for _p in _paths_to_add:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 导入并运行主Agent
try:
    from aura_perception.nvidia_agent import main

    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"错误: 无法导入nvidia_agent模块")
    print(f"详细信息: {e}")
    print(f"\n请确保已创建 aura_perception/aura_perception/aura_perception/nvidia_agent.py 文件")
    print(f"项目根目录: {_AURA_ROOT}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
