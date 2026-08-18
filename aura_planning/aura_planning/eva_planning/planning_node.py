#!/usr/bin/env python3
"""
AuraVLA Planning Node

ROS2 node for task planning and validation.
"""

import rclpy
from rclpy.node import Node
from aura_interfaces.srv import GeneratePlan
from aura_interfaces.msg import TaskPlan, TaskAction
import json
import yaml

from aura_planning.task_planner import TaskPlanner
from aura_planning.schema_validator import SchemaError


class PlanningNode(Node):
    """
    ROS2 node for task planning
    """

    def __init__(self):
        super().__init__('aura_planning_node')

        # Load configuration
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').value

        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        # Initialize planner
        self.planner = TaskPlanner()

        # Create service
        self.planning_service = self.create_service(
            GeneratePlan,
            'eva/planning/generate_plan',
            self.generate_plan_callback
        )

        self.get_logger().info('AuraVLA Planning Node initialized')

    def generate_plan_callback(self, request, response):
        """
        Service callback for plan generation

        Args:
            request: GeneratePlan request
            response: GeneratePlan response

        Returns:
            response
        """
        instruction = request.request.instruction
        self.get_logger().info(f'Generating plan for: {instruction}')

        try:
            # Parse replanning context if provided
            context = None
            if request.replan_context_json:
                context = json.loads(request.replan_context_json)

            # In a full implementation, this would:
            # 1. Use VLM to understand instruction
            # 2. Generate action sequence
            # 3. Validate with schema validator

            # For now, use simplified planning
            plan_dict = self.planner.decompose_task(
                instruction=instruction,
                scene_objects=[]  # Would come from scene observation
            )

            # Convert to ROS message
            response.success = True
            response.plan = self._dict_to_task_plan(plan_dict)
            response.error_message = ''

            self.get_logger().info(
                f'Plan generated with {len(response.plan.actions)} actions'
            )

        except SchemaError as e:
            self.get_logger().error(f'Schema validation failed: {e}')
            response.success = False
            response.error_message = f'Schema validation failed: {str(e)}'

        except Exception as e:
            self.get_logger().error(f'Plan generation failed: {e}')
            response.success = False
            response.error_message = f'Planning error: {str(e)}'

        return response

    def _dict_to_task_plan(self, plan_dict: dict) -> TaskPlan:
        """Convert plan dictionary to ROS message"""
        plan_msg = TaskPlan()
        plan_msg.header.stamp = self.get_clock().now().to_msg()
        plan_msg.task_id = ''  # Would be set by orchestrator
        plan_msg.instruction = plan_dict.get('instruction', '')
        plan_msg.rationale = plan_dict.get('rationale', '')
        plan_msg.confidence = float(plan_dict.get('confidence', 0.0))
        plan_msg.constraints = plan_dict.get('constraints', [])

        # Convert actions
        for action_dict in plan_dict.get('actions', []):
            action_msg = TaskAction()
            action_msg.action_id = action_dict.get('action_id', '')
            action_msg.task = action_dict.get('task', '')
            action_msg.object_name = action_dict.get('object_name', '')
            action_msg.target_name = action_dict.get('target_name', '')
            action_msg.attributes_json = json.dumps(
                action_dict.get('attributes', {})
            )
            plan_msg.actions.append(action_msg)

        return plan_msg

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'planning': {
                'max_actions': 10,
                'supported_tasks': ['pick_and_place'],
                'schema_version': '1.0'
            }
        }


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    node = PlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
