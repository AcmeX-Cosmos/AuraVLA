"""
Isaac Sim Task Bridge Launcher

在 Isaac Sim 中运行此脚本以启动任务桥接，接收来自 NVIDIA Agent 的任务请求。

使用方法：
1. 在 Isaac Sim 中打开您的场景（包含机械臂和物体）
2. 在 Isaac Sim Python 控制台或扩展中运行：
   exec(open('/path/to/this/script.py').read())
3. 桥接将开始监听 /tmp/aura-vla-control/ 目录
"""

from __future__ import annotations
import sys
from pathlib import Path

# 添加 AuraVLA 路径
aura_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(aura_root / "aura_execution" / "aura_execution"))

from aura_execution.task_bridge import start_task_bridge


def execute_pick_place(object_name: str, target_name: str) -> bool:
    """
    机械臂拾取放置执行函数

    Args:
        object_name: 要拾取的物体名称（例如：banana, mug, scissors）
        target_name: 放置目标位置（例如：basket, left_zone, right_zone）

    Returns:
        True if successful, False otherwise

    TODO: 实现您的机械臂控制逻辑：
    1. 定位物体（使用 object_name 查找场景中的 prim）
    2. 规划抓取路径
    3. 执行抓取动作
    4. 移动到目标位置
    5. 执行放置动作
    6. 返回初始位置
    """

    print(f"\n{'='*60}")
    print(f"🤖 执行任务:")
    print(f"   物体: {object_name}")
    print(f"   目标: {target_name}")
    print(f"{'='*60}")

    try:
        # ============================================
        # TODO: 在这里实现您的机械臂控制代码
        # ============================================

        # 示例：获取物体的 prim
        # from omni.isaac.core.utils.prims import get_prim_at_path
        # object_prim = get_prim_at_path(f"/World/{object_name}")

        # 示例：使用运动规划
        # from your_motion_planner import plan_pick_and_place
        # success = plan_pick_and_place(object_name, target_name)

        # 临时：模拟执行
        print("⚠️  Warning: Using placeholder execution")
        print("   Please implement your robot control logic in execute_pick_place()")

        # 模拟执行延迟
        import time
        time.sleep(1)

        success = True

        # ============================================

        if success:
            print(f"✓ 任务完成: {object_name} → {target_name}")
        else:
            print(f"✗ 任务失败: {object_name} → {target_name}")

        print(f"{'='*60}\n")
        return success

    except Exception as e:
        print(f"✗ 执行错误: {e}")
        print(f"{'='*60}\n")
        return False


def main():
    """启动任务桥接"""
    print("\n" + "="*60)
    print("AuraVLA Isaac Sim Task Bridge")
    print("="*60)
    print("")
    print("正在启动任务桥接...")
    print("监听目录: /tmp/aura-vla-control/")
    print("")

    try:
        # 启动桥接
        bridge = start_task_bridge(execute_pick_place)

        print("✓ 任务桥接已启动")
        print("")
        print("状态:")
        print(f"  - 请求文件: {bridge.paths.request}")
        print(f"  - 响应文件: {bridge.paths.response}")
        print(f"  - 状态文件: {bridge.paths.status}")
        print("")
        print("现在可以从 NVIDIA Agent 发送任务了！")
        print("="*60 + "\n")

        return bridge

    except Exception as e:
        print(f"\n✗ 启动失败: {e}")
        print("="*60 + "\n")
        raise


# 自动启动
if __name__ == "__main__":
    bridge = main()
else:
    # 在 Isaac Sim 控制台中执行时
    print("\n执行: bridge = main() 来启动任务桥接")
