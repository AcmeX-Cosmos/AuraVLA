#!/usr/bin/env python3
"""
NVIDIA Agent ROS2 Wrapper
Allows interactive dialogue that sends tasks through ROS2 orchestration
"""

import sys
import rclpy
from rclpy.node import Node
from aura_interfaces.msg import TaskRequest

# Import the NVIDIA agent dialogue logic
from aura_perception.nvidia_agent import NvidiaVLAgent, NvidiaConfig
from aura_perception.scene_names import SceneNameResolver


class NvidiaAgentROSWrapper(Node):
    """ROS2 wrapper for NVIDIA agent dialogue interface"""

    def __init__(self, agent: NvidiaVLAgent):
        super().__init__('nvidia_agent_ros_wrapper')
        self.agent = agent

        # Publisher to send task requests
        self.task_publisher = self.create_publisher(
            TaskRequest,
            '/aura/task_request',
            10
        )

        self.get_logger().info('NVIDIA Agent ROS2 wrapper initialized')
        self.get_logger().info('Type your instructions (Ctrl+C to quit)')

    def send_task(self, instruction: str, scene_info: str = ""):
        """Send task request through ROS2"""
        msg = TaskRequest()
        msg.instruction = instruction
        msg.scene = scene_info
        msg.timestamp = self.get_clock().now().to_msg()

        self.task_publisher.publish(msg)
        self.get_logger().info(f'Published task: {instruction}')


def main(args=None):
    """Main interactive loop"""
    rclpy.init(args=args)

    # Create NVIDIA agent with default config
    config = NvidiaConfig.from_env()
    agent = NvidiaVLAgent(
        config=config,
        scene_name_resolver=SceneNameResolver.from_default()
    )

    # Create ROS2 wrapper
    node = NvidiaAgentROSWrapper(agent)

    try:
        print("\n" + "="*60)
        print("NVIDIA Agent - ROS2 Interactive Mode")
        print("="*60)
        print("Enter instructions in natural language (Chinese or English)")
        print("Examples:")
        print("  - Pick up the red cube")
        print("  - 把香蕉放进篮子里")
        print("  - What objects do you see?")
        print("Press Ctrl+C to quit")
        print("="*60 + "\n")

        while rclpy.ok():
            try:
                instruction = input("\n🤖 Your instruction: ").strip()

                if not instruction:
                    continue

                # Send through ROS2
                node.send_task(instruction)

                # Spin once to process callbacks
                rclpy.spin_once(node, timeout_sec=0.1)

            except EOFError:
                break

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
