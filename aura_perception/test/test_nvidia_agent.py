import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from aura_perception.nvidia_agent import (
    NvidiaBuildBackend,
    NvidiaConfig,
    NvidiaTransientError,
    NvidiaVLAgent,
    _summarize_execution_result,
)


class RecordingBackend:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response or {
            "schema_version": "1.0",
            "doable": False,
            "task": "conversation",
        }

    def load(self):
        return None

    def generate(self, *args, **kwargs):
        self.calls += 1
        return json.dumps(self.response)


class NvidiaVLAgentRoutingTest(unittest.TestCase):
    def test_scalar_target_object_is_normalized_to_list(self):
        backend = RecordingBackend(
            {
                "schema_version": "1.0",
                "doable": True,
                "task": "pick_and_place",
                "target_objects": "banana",
                "target_container": "basket",
            }
        )
        agent = NvidiaVLAgent(backend=backend)

        response = json.loads(agent.infer("把香蕉放进篮子里", None))

        self.assertEqual(response["target_objects"], ["banana"])

    def test_exact_configured_task_still_uses_backend(self):
        backend = RecordingBackend(
            {
                "schema_version": "1.0",
                "doable": True,
                "task": "pick_and_place",
                "target_objects": ["banana"],
                "target_container": "basket",
                "inference_source": "nvidia_test_backend",
            }
        )
        agent = NvidiaVLAgent(
            backend=backend,
            fallback_tasks={
                "把香蕉放进篮子里": {
                    "object_name": "banana",
                    "target_name": "basket",
                }
            },
        )

        response = json.loads(agent.infer(" 把香蕉放进篮子里。 ", None))

        self.assertEqual(backend.calls, 1)
        self.assertEqual(response["target_objects"], ["banana"])
        self.assertEqual(response["target_container"], "basket")
        self.assertEqual(response["inference_source"], "nvidia_test_backend")

    def test_unknown_instruction_uses_backend(self):
        backend = RecordingBackend()
        agent = NvidiaVLAgent(backend=backend)

        agent.infer("请描述桌面", None)

        self.assertEqual(backend.calls, 1)


class NvidiaBuildBackendRetryTest(unittest.TestCase):
    def test_retry_override_disables_configured_retries(self):
        backend = NvidiaBuildBackend(
            NvidiaConfig(api_key="test-key", max_retries=2)
        )

        with patch(
            "aura_perception.nvidia_agent.urlopen",
            side_effect=URLError("offline"),
        ) as urlopen_mock:
            with self.assertRaises(NvidiaTransientError):
                backend._post_json(
                    "/chat/completions",
                    {},
                    max_retries=0,
                )

        self.assertEqual(urlopen_mock.call_count, 1)


class ExecutionSummaryTest(unittest.TestCase):
    def test_failure_summary_hides_internal_planning_diagnostics(self):
        response = {
            "request_id": "658dd698d9654c3cb59a3aafe7376272",
            "success": False,
            "execution": {
                "success": False,
                "state": "failed",
                "reason": "no strictly reachable banana short-axis grasp pose",
                "results": [
                    {
                        "object_name": "banana",
                        "target_name": "basket",
                        "success": False,
                        "message": "no strictly reachable banana short-axis grasp pose",
                        "details": {
                            "banana_pose_attempts": [{"ik_diagnostics": {"waypoint_count": 11}}],
                            "grasp_fusion": {"backend": "graspnet"},
                        },
                    }
                ],
            },
        }

        summary = _summarize_execution_result(
            json.dumps(response), "/tmp/aura-vla-control/response.json"
        )

        self.assertEqual(
            summary,
            "executor> FAILED [graspnet] banana -> basket: "
            "no strictly reachable banana short-axis grasp pose, request=658dd698; "
            "details=/tmp/aura-vla-control/response.json",
        )
        self.assertNotIn("banana_pose_attempts", summary)
        self.assertNotIn("waypoint_count", summary)

    def test_success_summary_is_compact(self):
        response = {
            "request_id": "sandbox-success-123456",
            "success": True,
            "execution": {
                "success": True,
                "state": "completed",
                "results": [
                    {
                        "object_name": "banana",
                        "target_name": "basket",
                        "success": True,
                        "message": "placed successfully",
                        "details": {
                            "grasp_fusion": {"backend": "graspnet"},
                            "banana_pose_attempts": [{"waypoint_count": 11}],
                        },
                    }
                ],
            },
        }

        summary = _summarize_execution_result(json.dumps(response), "/tmp/response.json")

        self.assertEqual(
            summary,
            "executor> SUCCESS [graspnet] banana -> basket: "
            "placed successfully, request=sandbox-; details=/tmp/response.json",
        )
        self.assertNotIn("banana_pose_attempts", summary)
        self.assertNotIn("waypoint_count", summary)


if __name__ == "__main__":
    unittest.main()
