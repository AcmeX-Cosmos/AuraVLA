from __future__ import annotations

from collections.abc import Callable, Mapping
import traceback
from typing import Any

from schemas import TaskPlan, dumps_json, loads_json


class PickPlaceExecutor:
    """Adapter around the existing execute_pick_place(object_name, target_name)."""

    def __init__(
        self,
        execute_pick_place: Callable[[str, str], Any],
        *,
        none_is_success: bool = True,
    ) -> None:
        if not callable(execute_pick_place):
            raise TypeError("execute_pick_place must be callable")
        self._execute_pick_place = execute_pick_place
        self._none_is_success = none_is_success

    def execute(self, plan_json: str) -> str:
        plan = TaskPlan.from_json(plan_json)
        if not plan.doable:
            return dumps_json(
                {
                    "success": False,
                    "state": "rejected",
                    "reason": plan.reason or "Task is not doable",
                    "results": [],
                }
            )

        results: list[dict[str, Any]] = []
        for action in plan.actions:
            try:
                raw_result = self._execute_pick_place(
                    action.object_name, action.target_name
                )
                success, message, details = self._normalize_result(raw_result)
            except Exception as exc:
                success, message, details = False, str(exc), {
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc().splitlines(),
                }

            results.append(
                {
                    "action_id": action.action_id,
                    "task": action.task,
                    "object_name": action.object_name,
                    "target_name": action.target_name,
                    "success": success,
                    "message": message,
                    "details": details,
                }
            )
            if not success:
                break

        all_succeeded = len(results) == len(plan.actions) and all(
            result["success"] for result in results
        )
        return dumps_json(
            {
                "success": all_succeeded,
                "state": "completed" if all_succeeded else "failed",
                "reason": "All actions completed" if all_succeeded else results[-1]["message"],
                "results": results,
            }
        )

    def _normalize_result(self, value: Any) -> tuple[bool, str, dict[str, Any]]:
        if value is None:
            return self._none_is_success, "execute_pick_place returned", {}
        if isinstance(value, bool):
            return value, "execute_pick_place returned success" if value else "execute_pick_place failed", {}
        if isinstance(value, str):
            try:
                value = loads_json(value)
            except ValueError:
                return True, value, {}
        if isinstance(value, tuple) and len(value) >= 2:
            return bool(value[0]), str(value[1]), {}
        if isinstance(value, Mapping):
            result = dict(value)
            success = bool(result.pop("success", False))
            message = str(result.pop("message", result.pop("reason", "")))
            return success, message, result
        raise TypeError(
            "execute_pick_place must return None, bool, (success, message), dict, or JSON"
        )
