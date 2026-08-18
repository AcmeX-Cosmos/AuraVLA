#!/usr/bin/env python3
"""
Isaac Bridge Node

Synchronizes scene state with Isaac Sim.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class IsaacBridgeNode(Node):
    """
    Bridge between Isaac Sim and ROS2

    Monitors Isaac Sim state and publishes scene observations.
    """

    def __init__(self):
        super().__init__('aura_isaac_bridge_node')

        self.declare_parameter('status_file', '/tmp/eva-agent-control/status.json')
        self.declare_parameter('check_rate', 1.0)

        status_file = self.get_parameter('status_file').value
        rate = self.get_parameter('check_rate').value

        self.status_file = status_file

        # Publisher
        self.status_pub = self.create_publisher(String, 'isaac/status', 10)

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.check_status)

        self.get_logger().info('Isaac Bridge Node initialized')

    def check_status(self):
        """Check Isaac Sim status"""
        try:
            from pathlib import Path
            status_path = Path(self.status_file)

            if status_path.exists():
                with open(status_path, 'r') as f:
                    status = json.load(f)

                    msg = String()
                    msg.data = json.dumps(status)
                    self.status_pub.publish(msg)

        except Exception as e:
            self.get_logger().debug(f'Status check: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = IsaacBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
