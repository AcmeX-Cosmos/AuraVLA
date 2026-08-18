"""
Task Planner

Generates structured task plans from high-level requests.
"""

from typing import Dict, Any, Optional
from aura_planning.schema_validator import SchemaValidator, SchemaError


class TaskPlanner:
    """
    Plans tasks and validates execution plans
    """

    def __init__(self):
        self.validator = SchemaValidator()

    def plan(
        self,
        agent_output: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate and validate a task plan

        Args:
            agent_output: Raw output from VLM agent (JSON string)
            context: Optional replanning context

        Returns:
            Validated task plan

        Raises:
            SchemaError: If plan validation fails
        """
        try:
            # Validate the plan structure
            plan = self.validator.validate_plan(agent_output)

            # Add metadata
            plan['schema_version'] = SchemaValidator.SCHEMA_VERSION
            plan['validated'] = True

            # Add replanning context if provided
            if context:
                plan['replan_context'] = context

            return plan

        except SchemaError as e:
            raise SchemaError(f"Plan validation failed: {e}")

    def decompose_task(
        self,
        instruction: str,
        scene_objects: list
    ) -> Dict[str, Any]:
        """
        Decompose high-level instruction into actions

        Args:
            instruction: Natural language instruction
            scene_objects: Available objects in scene

        Returns:
            Task plan dictionary
        """
        # Simple rule-based decomposition
        # In practice, this would use VLM or more sophisticated planning

        instruction_lower = instruction.lower()

        # Detect task type
        if '放' in instruction or 'put' in instruction_lower:
            task_type = 'pick_and_place'
        else:
            task_type = 'pick_and_place'  # Default

        # Extract object and target (simple heuristic)
        # This is simplified - real implementation would use NLP/VLM
        actions = [
            {
                'action_id': 'action_0',
                'task': task_type,
                'object_name': 'object',  # Would be extracted from instruction
                'target_name': 'target',   # Would be extracted from instruction
                'attributes': {}
            }
        ]

        return {
            'instruction': instruction,
            'actions': actions,
            'rationale': 'Generated from instruction',
            'confidence': 0.8
        }
