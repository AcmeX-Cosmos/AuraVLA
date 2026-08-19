#!/usr/bin/env python3
"""
AuraVLA Execution Node

ROS2 action server for task execution.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from aura_interfaces.action import ExecuteTask
import json
import yaml

from aura_execution.task_bridge import FileTaskClient
from aura_execution.action_executor import ActionExecutor


class ExecutionNode(Node):
    """
    ROS2 action server for task execution
    """

    def __init__(self):
        super().__init__('aura_execution_node')

        # Load configuration
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').value

        if config_file:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()

        # Initialize task bridge
        bridge_cfg = config.get('execution', {}).get('bridge', {})
        bridge = FileTaskClient(
            directory=bridge_cfg.get('directory', '/tmp/aura-vla-control'),
            timeout_sec=float(bridge_cfg.get('timeout_sec', 300.0)),
            poll_interval_sec=float(bridge_cfg.get('check_interval_sec', 0.5))
        )

        self.action_executor = ActionExecutor(bridge)

        # Create action server
        self._action_server = ActionServer(
            self,
            ExecuteTask,
            'aura/execution/execute_task',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info('AuraVLA Execution Node initialized')

    def goal_callback(self, goal_request):
        """Accept or reject goal"""
        self.get_logger().info('Received execution goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancellation request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Execute task action

        Args:
            goal_handle: Action goal handle

        Returns:
            Result
        """
        self.get_logger().info('Executing task...')

        request = goal_handle.request
        feedback_msg = ExecuteTask.Feedback()

        try:
            # Convert plan to dictionary
            plan_dict = self._task_plan_to_dict(request.plan)

            # Send initial feedback
            feedback_msg.current_action = 0
            feedback_msg.state = 'executing'
            feedback_msg.progress = 0.0
            feedback_msg.message = 'Starting execution'
            goal_handle.publish_feedback(feedback_msg)

            # Execute via bridge
            result_dict = self.action_executor.execute_plan(plan_dict)

            # Send final feedback
            feedback_msg.current_action = len(request.plan.actions)
            feedback_msg.state = 'completed'
            feedback_msg.progress = 1.0
            feedback_msg.message = 'Execution completed'
            goal_handle.publish_feedback(feedback_msg)

            # Set success
            goal_handle.succeed()

            # Prepare result
            result = ExecuteTask.Result()
            result.success = result_dict.get('success', True)
            result.message = result_dict.get('message', 'Execution completed')
            result.result_json = json.dumps(result_dict)

            self.get_logger().info(f'Execution completed: {result.success}')
            return result

        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')

            # Abort goal
            goal_handle.abort()

            result = ExecuteTask.Result()
            result.success = False
            result.message = f'Execution error: {str(e)}'
            result.result_json = '{}'

            return result

    def _task_plan_to_dict(self, plan_msg) -> dict:
        """Convert ROS TaskPlan message to dictionary"""
        actions = []
        for action_msg in plan_msg.actions:
            action = {
                'action_id': action_msg.action_id,
                'task': action_msg.task,
                'object_name': action_msg.object_name,
                'target_name': action_msg.target_name,
                'attributes': json.loads(action_msg.attributes_json)
                if action_msg.attributes_json else {}
            }
            actions.append(action)

        return {
            'instruction': plan_msg.instruction,
            'actions': actions,
            'rationale': plan_msg.rationale,
            'confidence': plan_msg.confidence
        }

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'execution': {
                'bridge': {
                    'directory': '/tmp/aura-vla-control',
                    'timeout_sec': 300.0,
                    'check_interval_sec': 0.5
                }
            }
        }


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    node = ExecutionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
