"""
Action Executor

Executes task actions and monitors progress.
"""

import json
from typing import Dict, Any, Union
from aura_execution.task_bridge import IsaacTaskBridge, FileTaskClient


class ActionExecutor:
    """
    Executes robot actions via task bridge
    """

    def __init__(self, bridge: Union[IsaacTaskBridge, FileTaskClient]):
        self.bridge = bridge

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete task plan

        Args:
            plan: Task plan dictionary

        Returns:
            Execution result

        Raises:
            RuntimeError: If execution fails
        """
        # Check bridge status
        if not self.bridge.is_ready():
            raise RuntimeError("Task bridge not ready")

        # Convert plan to JSON
        plan_json = json.dumps(plan)

        # Execute via bridge
        try:
            result_json = self.bridge.execute(plan_json)
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
            return result

        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action

        Args:
            action: Action dictionary

        Returns:
            Execution result
        """
        # Wrap single action in a plan
        plan = {
            'actions': [action]
        }
        return self.execute_plan(plan)
