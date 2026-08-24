#!/usr/bin/env python3
"""
Isaac Bridge Node

Synchronizes scene state with Isaac Sim.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from pathlib import Path
import time


class IsaacBridgeNode(Node):
    """
    Bridge between Isaac Sim and ROS2

    Monitors Isaac Sim state and publishes scene observations.
    """

    def __init__(self):
        super().__init__('aura_isaac_bridge_node')

        self.declare_parameter('status_file', '/tmp/aura-vla-control/status.json')
        self.declare_parameter(
            'status_topic',
            'isaac/status',
        )
        self.declare_parameter(
            'transport_tracking_file',
            '/tmp/aura-vla-control/transport_tracking.json',
        )
        self.declare_parameter(
            'transport_tracking_topic',
            'aura/transport_tracking',
        )
        self.declare_parameter('transport_tracking_heartbeat_sec', 5.0)
        self.declare_parameter('check_rate', 1.0)

        status_file = self.get_parameter('status_file').value
        status_topic = self.get_parameter('status_topic').value
        transport_tracking_file = self.get_parameter('transport_tracking_file').value
        transport_tracking_topic = self.get_parameter('transport_tracking_topic').value
        heartbeat_sec = self.get_parameter('transport_tracking_heartbeat_sec').value
        rate = self.get_parameter('check_rate').value

        self.status_file = Path(status_file)
        self.transport_tracking_file = Path(transport_tracking_file)
        self._last_tracking_signature = None
        self._tracking_file_missing_logged = False
        self._tracking_event_logged = False
        self._tracking_heartbeat_sec = max(float(heartbeat_sec), 0.0)
        self._last_tracking_heartbeat = 0.0

        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.transport_tracking_pub = self.create_publisher(
            String,
            transport_tracking_topic,
            10,
        )

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.check_status)

        self.get_logger().info(
            'Isaac Bridge Node initialized; '
            f'transport topic={transport_tracking_topic}, '
            f'file={self.transport_tracking_file}'
        )

    def check_status(self):
        """Check Isaac Sim status"""
        try:
            if self.status_file.exists():
                with self.status_file.open('r', encoding='utf-8') as stream:
                    status = json.load(stream)
                msg = String()
                msg.data = json.dumps(status, ensure_ascii=False)
                self.status_pub.publish(msg)
            self._publish_file_once(
                self.transport_tracking_file,
                self.transport_tracking_pub,
            )
            self._publish_tracking_heartbeat()

        except Exception as e:
            self.get_logger().debug(f'Status check: {e}')

    def _publish_file_once(self, path, publisher):
        """Publish an atomically-written JSON file only when it changes."""
        if not path.exists():
            if not self._tracking_file_missing_logged:
                self.get_logger().info(
                    f'Waiting for transport tracking events: {path}'
                )
                self._tracking_file_missing_logged = True
            return
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._last_tracking_signature:
            return

        with path.open('r', encoding='utf-8') as stream:
            payload = json.load(stream)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        publisher.publish(msg)
        self._last_tracking_signature = signature
        if not self._tracking_event_logged:
            self.get_logger().info('Transport tracking event published')
            self._tracking_event_logged = True

    def _publish_tracking_heartbeat(self):
        """Expose bridge readiness without touching the Isaac runtime."""
        if self._tracking_heartbeat_sec <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_tracking_heartbeat < self._tracking_heartbeat_sec:
            return
        msg = String()
        msg.data = json.dumps({
            'event': 'transport_tracking_bridge',
            'state': (
                'waiting_for_event'
                if not self.transport_tracking_file.exists()
                else 'monitoring'
            ),
            'timestamp_unix': time.time(),
            'transport_tracking_file': str(self.transport_tracking_file),
        }, ensure_ascii=False, separators=(',', ':'))
        self.transport_tracking_pub.publish(msg)
        self._last_tracking_heartbeat = now


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
