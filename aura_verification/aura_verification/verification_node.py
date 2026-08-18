#!/usr/bin/env python3
"""
AuraVLA Verification Node

ROS2 service node for completion checking.
"""

import rclpy
from rclpy.node import Node
from aura_interfaces.srv import CheckCompletion
import json

from aura_verification.completion_checker import CompletionChecker


class VerificationNode(Node):
    """ROS2 node for task verification"""

    def __init__(self):
        super().__init__('aura_verification_node')

        self.checker = CompletionChecker()

        self.check_service = self.create_service(
            CheckCompletion,
            'aura/verification/check_completion',
            self.check_callback
        )

        self.get_logger().info('AuraVLA Verification Node initialized')

    def check_callback(self, request, response):
        """Service callback for completion checking"""
        self.get_logger().info('Checking task completion')

        try:
            # Convert to dicts
            plan_dict = self._msg_to_dict(request.plan)
            exec_result = json.loads(request.execution_result_json)

            # Check completion
            result = self.checker.check(plan_dict, exec_result)

            # Fill response
            response.success = result.get('success', False)
            response.need_replan = result.get('need_replan', False)
            response.confidence = float(result.get('confidence', 0.0))
            response.reason = result.get('reason', '')

            self.get_logger().info(f'Check result: {response.success}')

        except Exception as e:
            self.get_logger().error(f'Check failed: {e}')
            response.success = False
            response.need_replan = True
            response.confidence = 0.0
            response.reason = f'Check error: {str(e)}'

        return response

    def _msg_to_dict(self, plan_msg) -> dict:
        """Convert plan message to dict"""
        return {
            'instruction': plan_msg.instruction,
            'actions': []  # Would convert actions
        }


def main(args=None):
    rclpy.init(args=args)
    node = VerificationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
