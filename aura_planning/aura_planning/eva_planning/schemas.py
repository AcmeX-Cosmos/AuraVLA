from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
SUPPORTED_TASKS = {"pick_and_place"}
FORBIDDEN_MOTION_KEYS = {
    "control",
    "grasp_pose",
    "ik",
    "joint_positions",
    "joint_trajectory",
    "pose",
    "trajectory",
    "velocity",
    "waypoints",
}


class SchemaError(ValueError):
    pass


def dumps_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads_json(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("Expected a non-empty JSON object")

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = _extract_first_json_object(text)
    if not isinstance(decoded, dict):
        raise SchemaError("JSON message must be an object")
    return decoded


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise SchemaError("No valid JSON object found in model response")


def assert_no_motion_commands(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in FORBIDDEN_MOTION_KEYS:
                raise SchemaError(f"Motion command field is forbidden: {path}.{key}")
            assert_no_motion_commands(nested_value, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_motion_commands(item, f"{path}[{index}]")


def _required_name(value: Any, field_name: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise SchemaError(f"Missing required field: {field_name}")
    return name


@dataclass(frozen=True)
class TaskAction:
    action_id: str
    task: str
    object_name: str
    target_name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "TaskAction":
        task = str(value.get("task", "pick_and_place")).strip().lower()
        if task not in SUPPORTED_TASKS:
            raise SchemaError(f"Unsupported task: {task}")
        return cls(
            action_id=str(value.get("action_id") or f"action-{index + 1}"),
            task=task,
            object_name=_required_name(
                value.get("object_name", value.get("target_object")), "object_name"
            ),
            target_name=_required_name(
                value.get("target_name", value.get("target_container")), "target_name"
            ),
            attributes=dict(value.get("attributes") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "task": self.task,
            "object_name": self.object_name,
            "target_name": self.target_name,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class TaskPlan:
    doable: bool
    task: str
    actions: tuple[TaskAction, ...]
    reason: str = ""
    constraints: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    source: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_agent_response(cls, value: Mapping[str, Any]) -> "TaskPlan":
        data = dict(value)
        assert_no_motion_commands(data)
        doable = data.get("doable")
        if not isinstance(doable, bool):
            raise SchemaError("Field 'doable' must be true or false")

        task = str(data.get("task", "pick_and_place")).strip().lower()
        if doable and task not in SUPPORTED_TASKS:
            raise SchemaError(f"Unsupported task: {task}")

        actions: list[TaskAction] = []
        raw_actions = data.get("actions")
        if raw_actions is not None:
            if not isinstance(raw_actions, list):
                raise SchemaError("Field 'actions' must be a list")
            for index, item in enumerate(raw_actions):
                if not isinstance(item, Mapping):
                    raise SchemaError(f"actions[{index}] must be an object")
                actions.append(TaskAction.from_mapping(item, index))
        elif doable:
            target_name = data.get("target_container", data.get("target_name"))
            raw_objects = data.get("target_objects")
            if raw_objects is None:
                raw_objects = [data.get("target_object", data.get("object_name"))]
            if not isinstance(raw_objects, list):
                raise SchemaError("Field 'target_objects' must be a list")
            for index, object_value in enumerate(raw_objects):
                if isinstance(object_value, Mapping):
                    action_value = dict(object_value)
                    action_value.setdefault("target_name", target_name)
                    action_value.setdefault("task", task)
                else:
                    action_value = {
                        "task": task,
                        "object_name": object_value,
                        "target_name": target_name,
                    }
                actions.append(TaskAction.from_mapping(action_value, index))

        unique_actions = []
        seen_actions = set()
        for action in actions:
            action_key = (
                action.task.casefold(),
                action.object_name.strip().casefold(),
                action.target_name.strip().casefold(),
            )
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)
            unique_actions.append(action)
        actions = unique_actions

        if doable and not actions:
            raise SchemaError("A doable task must contain at least one concrete action")

        constraints_value = data.get("constraints") or []
        if not isinstance(constraints_value, list):
            raise SchemaError("Field 'constraints' must be a list")
        return cls(
            doable=doable,
            task=task,
            actions=tuple(actions),
            reason=str(data.get("reason", "")),
            constraints=tuple(str(item) for item in constraints_value),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            source=data,
        )

    @classmethod
    def from_json(cls, value: str | Mapping[str, Any]) -> "TaskPlan":
        return cls.from_agent_response(loads_json(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "doable": self.doable,
            "task": self.task,
            "reason": self.reason,
            "constraints": list(self.constraints),
            "actions": [action.to_dict() for action in self.actions],
        }
