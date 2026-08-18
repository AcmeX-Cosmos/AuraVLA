"""
Schema Validator

Validates task plans against defined schemas.
"""

import json
from typing import Dict, Any, Set
from dataclasses import dataclass


class SchemaError(ValueError):
    """Schema validation error"""
    pass


@dataclass
class TaskAction:
    """Task action specification"""
    action_id: str
    task: str
    object_name: str
    target_name: str
    attributes: Dict[str, Any]


class SchemaValidator:
    """
    Validates task plans and ensures safety
    """

    SCHEMA_VERSION = "1.0"
    SUPPORTED_TASKS = {"pick_and_place"}
    FORBIDDEN_MOTION_KEYS = {
        "control",
        "grasp_pose",
        "ik",
        "joint_positions",
        "pose",
        "trajectory",
        "velocity",
        "waypoints",
    }

    def __init__(self):
        pass

    def validate_plan(self, plan_json: str) -> Dict[str, Any]:
        """
        Validate a task plan

        Args:
            plan_json: JSON string containing the plan

        Returns:
            Validated plan dict

        Raises:
            SchemaError: If validation fails
        """
        # Parse JSON
        try:
            plan = self._parse_json(plan_json)
        except Exception as e:
            raise SchemaError(f"Invalid JSON: {e}")

        # Validate structure
        if not isinstance(plan, dict):
            raise SchemaError("Plan must be a JSON object")

        # Validate actions
        actions = plan.get('actions', [])
        if not isinstance(actions, list):
            raise SchemaError("Actions must be a list")

        if len(actions) == 0:
            raise SchemaError("Plan must contain at least one action")

        # Validate each action
        validated_actions = []
        for i, action in enumerate(actions):
            validated_action = self._validate_action(action, i)
            validated_actions.append(validated_action)

        # Check for forbidden motion commands
        self._check_no_motion_commands(plan)

        plan['actions'] = validated_actions
        return plan

    def _validate_action(self, action: Dict[str, Any], index: int) -> TaskAction:
        """Validate a single action"""

        # Required fields
        action_id = action.get('action_id', f'action_{index}')
        task = str(action.get('task', 'pick_and_place')).strip().lower()

        if task not in self.SUPPORTED_TASKS:
            raise SchemaError(f"Unsupported task: {task}")

        object_name = self._required_field(action, 'object_name', 'object')
        target_name = self._required_field(action, 'target_name', 'target')

        # Optional attributes
        attributes = action.get('attributes', {})
        if not isinstance(attributes, dict):
            attributes = {}

        return TaskAction(
            action_id=action_id,
            task=task,
            object_name=object_name,
            target_name=target_name,
            attributes=attributes
        )

    def _check_no_motion_commands(self, obj: Any, path: str = "root"):
        """Recursively check for forbidden motion command keys"""
        if isinstance(obj, dict):
            for key in obj.keys():
                if key in self.FORBIDDEN_MOTION_KEYS:
                    raise SchemaError(
                        f"Forbidden motion command key '{key}' found at {path}"
                    )
                self._check_no_motion_commands(obj[key], f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._check_no_motion_commands(item, f"{path}[{i}]")

    def _required_field(self, obj: Dict, field: str, display_name: str) -> str:
        """Extract required string field"""
        value = str(obj.get(field, '')).strip()
        if not value:
            raise SchemaError(f"Missing required field: {display_name}")
        return value

    def _parse_json(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON, handling markdown code blocks"""
        text = json_str.strip()

        # Extract from markdown code block
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Find closing ```
                for i, line in enumerate(lines):
                    if line.strip() == "```":
                        lines = lines[:i]
                        break
                text = "\n".join(lines)

        return json.loads(text)
