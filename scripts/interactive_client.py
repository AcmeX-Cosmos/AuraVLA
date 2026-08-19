#!/usr/bin/env python3
"""
Simple Interactive Task Client
Sends natural language instructions via ROS2 topic (using CLI, no Python bindings)
"""

import subprocess
import sys
import json
from pathlib import Path


def send_task_via_ros2_cli(instruction: str):
    """Send task using ros2 topic pub command"""
    # Escape quotes in instruction
    instruction_safe = instruction.replace('"', '\\"')

    cmd = [
        'ros2', 'topic', 'pub', '--once',
        '/aura/task_request',
        'aura_interfaces/msg/TaskRequest',
        f'{{"instruction": "{instruction_safe}", "scene": ""}}'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Task sent: {instruction}")
            return True
        else:
            print(f"✗ Failed to send task: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Timeout sending task")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main interactive loop"""
    print("\n" + "="*60)
    print("AuraVLA Interactive Task Client")
    print("="*60)
    print("Enter instructions in natural language (Chinese or English)")
    print("\nExamples:")
    print("  - Pick up the red cube")
    print("  - 把香蕉放进篮子里")
    print("  - Move the apple to the left zone")
    print("\nCommands:")
    print("  - 'quit' or 'exit' to quit")
    print("  - Ctrl+C to quit")
    print("="*60 + "\n")

    # Check if ROS2 system is running
    try:
        result = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True,
            text=True,
            timeout=3
        )
        if '/aura/task_request' not in result.stdout:
            print("⚠️  Warning: /aura/task_request topic not found.")
            print("   Make sure ROS2 system is running:")
            print("   ros2 launch aura_bringup aura_system.launch.py\n")
    except:
        print("⚠️  Warning: Cannot connect to ROS2.")
        print("   Make sure you've sourced the workspace:\n")
        print("   source install/setup.bash\n")

    try:
        while True:
            try:
                instruction = input("🤖 Your instruction: ").strip()

                if not instruction:
                    continue

                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 再见！")
                    break

                # Send task via ROS2 CLI
                send_task_via_ros2_cli(instruction)
                print()

            except EOFError:
                break

    except KeyboardInterrupt:
        print("\n\nShutting down... 正在关闭...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
