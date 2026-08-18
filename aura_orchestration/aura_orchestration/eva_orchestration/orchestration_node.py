#!/usr/bin/env python3
"""
AuraVLA Orchestration Node

ROS2 node that orchestrates the complete closed-loop.
"""

import rclpy
from rclpy.node import Node
from aura_interfaces.srv import EvaluateDoable, GeneratePlan, CheckCompletion
from aura_interfaces.action import ExecuteTask
from rclpy.action import ActionClient

from aura_orchestration.orchestrator import Orchestrator


class OrchestrationNode(Node):
    """
    ROS2 node for task orchestration

    Coordinates all services and actions for closed-loop control
    """

    def __init__(self):
        super().__init__('aura_orchestration_node')

        self.declare_parameter('max_replans', 2)
        max_replans = self.get_parameter('max_replans').value

        self.orchestrator = Orchestrator(max_replans=max_replans)

        # Create service clients
        self.doable_client = self.create_client(
            EvaluateDoable,
            'eva/perception/doable'
        )
        self.planning_client = self.create_client(
            GeneratePlan,
            'eva/planning/generate_plan'
        )
        self.check_client = self.create_client(
            CheckCompletion,
            'eva/verification/check_completion'
        )

        # Create action client
        self.execution_client = ActionClient(
            self,
            ExecuteTask,
            'eva/execution/execute_task'
        )

        self.get_logger().info('AuraVLA Orchestration Node initialized')
        self.get_logger().info(f'Max replans: {max_replans}')

    # Full implementation would include:
    # - Task execution service
    # - Service client calls
    # - Action client calls
    # - State management
    # - Progress monitoring


def main(args=None):
    rclpy.init(args=args)
    node = OrchestrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
