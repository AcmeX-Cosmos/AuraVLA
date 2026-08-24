#!/usr/bin/env python3
"""
Isaac Bridge Node

Synchronizes scene state with Isaac Sim.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Vector3Stamped
from std_msgs.msg import Bool, Float32, String
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
        self.declare_parameter('graspnet_topic_prefix', 'aura/graspnet')
        self.declare_parameter('graspnet_frame_id', 'world')
        self.declare_parameter('check_rate', 1.0)

        status_file = self.get_parameter('status_file').value
        status_topic = self.get_parameter('status_topic').value
        transport_tracking_file = self.get_parameter('transport_tracking_file').value
        transport_tracking_topic = self.get_parameter('transport_tracking_topic').value
        heartbeat_sec = self.get_parameter('transport_tracking_heartbeat_sec').value
        topic_prefix = str(self.get_parameter('graspnet_topic_prefix').value).rstrip('/')
        frame_id = str(self.get_parameter('graspnet_frame_id').value)
        rate = self.get_parameter('check_rate').value

        self.status_file = Path(status_file)
        self.transport_tracking_file = Path(transport_tracking_file)
        self._last_tracking_signature = None
        self._tracking_file_missing_logged = False
        self._tracking_event_logged = False
        self._tracking_heartbeat_sec = max(float(heartbeat_sec), 0.0)
        self._last_tracking_heartbeat = 0.0
        self.graspnet_frame_id = frame_id

        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.transport_tracking_pub = self.create_publisher(
            String,
            transport_tracking_topic,
            10,
        )
        # Scalar/vector topics use standard ROS messages so Foxglove Plot can
        # graph them directly without parsing the JSON event string.
        self.graspnet_state_pub = self.create_publisher(
            String, f'{topic_prefix}/state', 10
        )
        self.graspnet_error_pub = self.create_publisher(
            Float32, f'{topic_prefix}/position_error_m', 10
        )
        self.graspnet_confidence_pub = self.create_publisher(
            Float32, f'{topic_prefix}/confidence', 10
        )
        self.graspnet_replan_pub = self.create_publisher(
            Bool, f'{topic_prefix}/replan', 10
        )
        self.graspnet_observed_pub = self.create_publisher(
            PointStamped, f'{topic_prefix}/observed_position', 10
        )
        self.graspnet_expected_pub = self.create_publisher(
            PointStamped, f'{topic_prefix}/expected_position', 10
        )
        self.graspnet_error_vector_pub = self.create_publisher(
            Vector3Stamped, f'{topic_prefix}/position_error_vector_m', 10
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
        self._publish_graspnet_topics(payload)
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
        self._publish_graspnet_topics({
            'state': 'waiting_for_event' if not self.transport_tracking_file.exists() else 'monitoring',
            'replan': False,
        })
        self._last_tracking_heartbeat = now

    def _publish_graspnet_topics(self, payload):
        """Fan out one JSON event into Foxglove-friendly standard messages."""
        state_msg = String()
        state_msg.data = str(payload.get('state', payload.get('event', 'unknown')))
        self.graspnet_state_pub.publish(state_msg)

        replan_msg = Bool()
        replan_msg.data = bool(payload.get('replan', False))
        self.graspnet_replan_pub.publish(replan_msg)

        fusion = payload.get('fusion') or {}
        confidence = fusion.get('confidence', payload.get('confidence'))
        if confidence is not None:
            confidence_msg = Float32()
            confidence_msg.data = float(confidence)
            self.graspnet_confidence_pub.publish(confidence_msg)

        observed = self._finite_xyz(payload.get('observed_position'))
        expected = self._finite_xyz(payload.get('expected_position'))
        stamp = self.get_clock().now().to_msg()
        if observed is not None:
            self.graspnet_observed_pub.publish(
                self._point_message(observed, stamp)
            )
        if expected is not None:
            self.graspnet_expected_pub.publish(
                self._point_message(expected, stamp)
            )

        error = payload.get('error_m')
        if error is not None:
            error_msg = Float32()
            error_msg.data = float(error)
            self.graspnet_error_pub.publish(error_msg)
        if observed is not None and expected is not None:
            vector_msg = Vector3Stamped()
            vector_msg.header.stamp = stamp
            vector_msg.header.frame_id = self.graspnet_frame_id
            vector_msg.vector.x = observed[0] - expected[0]
            vector_msg.vector.y = observed[1] - expected[1]
            vector_msg.vector.z = observed[2] - expected[2]
            self.graspnet_error_vector_pub.publish(vector_msg)

    @staticmethod
    def _finite_xyz(value):
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if len(values) != 3:
            return None
        return values

    def _point_message(self, position, stamp):
        message = PointStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.graspnet_frame_id
        message.point.x, message.point.y, message.point.z = position
        return message


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
