from __future__ import annotations

from schemas import SchemaError, TaskPlan, dumps_json, loads_json


class PlannerError(ValueError):
    pass


class TaskPlanner:
    """Converts the NVIDIA VLM semantic JSON into validated executor JSON."""

    def plan(self, agent_response_json: str) -> str:
        try:
            response = loads_json(agent_response_json)
            plan = TaskPlan.from_agent_response(response)
        except (SchemaError, TypeError, ValueError) as exc:
            raise PlannerError(f"Invalid NVIDIA VLM task JSON: {exc}") from exc
        return dumps_json(plan.to_dict())
