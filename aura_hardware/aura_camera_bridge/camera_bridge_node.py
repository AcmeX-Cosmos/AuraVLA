#!/usr/bin/env python3
"""
Camera Bridge Node

Captures RGBD images from Isaac Sim and publishes to ROS2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
import json


class CameraBridgeNode(Node):
    """
    Bridge between Isaac Sim camera and ROS2

    Reads images from file system and publishes as ROS2 messages.
    """

    def __init__(self):
        super().__init__('aura_camera_bridge_node')

        self.declare_parameter('camera_directory', '/tmp/aura-vla-camera')
        self.declare_parameter('publish_rate', 10.0)

        camera_dir = self.get_parameter('camera_directory').value
        rate = self.get_parameter('publish_rate').value

        self.camera_dir = Path(camera_dir)
        self.bridge = CvBridge()

        # Publishers
        self.rgb_pub = self.create_publisher(Image, 'camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.publish_callback)

        self.get_logger().info('Camera Bridge Node initialized')

    def publish_callback(self):
        """Publish camera images"""
        try:
            # Check for new images
            rgb_path = self.camera_dir / 'rgb.png'
            depth_path = self.camera_dir / 'depth.png'
            metadata_path = self.camera_dir / 'metadata.json'

            if not rgb_path.exists():
                return

            # Read and publish RGB
            # (Would use cv2.imread in real implementation)

            # Read and publish depth
            # (Would use cv2.imread in real implementation)

            # Read and publish camera info
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Construct CameraInfo message

        except Exception as e:
            self.get_logger().error(f'Camera capture failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
