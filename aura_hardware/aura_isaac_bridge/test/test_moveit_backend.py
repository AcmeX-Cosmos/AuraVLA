import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from aura_isaac_bridge.robot.moveit_backend import MoveItFilePlanner


class MoveItFilePlannerTest(unittest.TestCase):
    def test_missing_node_is_detected_without_waiting_for_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            planner = MoveItFilePlanner(request_directory=directory, timeout_sec=0.1)
            self.assertFalse(planner.is_ready())

    def test_response_is_matched_by_request_id(self):
        with tempfile.TemporaryDirectory() as directory:
            planner = MoveItFilePlanner(request_directory=directory, timeout_sec=1.0)
            ready = os.path.join(directory, "moveit_planner_ready.json")
            with open(ready, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid()}, stream)

            def respond():
                request_path = os.path.join(directory, "moveit_plan_request.json")
                while not os.path.exists(request_path):
                    time.sleep(0.005)
                with open(request_path, encoding="utf-8") as stream:
                    request = json.load(stream)
                response = {
                    "request_id": request["request_id"],
                    "success": True,
                    "trajectory_positions": [[0.0] * 7, [0.1] * 7],
                }
                with open(os.path.join(directory, "moveit_plan_response.json"), "w", encoding="utf-8") as stream:
                    json.dump(response, stream)

            thread = threading.Thread(target=respond, daemon=True)
            thread.start()
            path = planner.plan_to_pose(
                group_name="right_arm",
                end_effector_link="tcp_R_Link",
                target_position=[0.1, -0.2, 0.3],
                target_orientation=[1.0, 0.0, 0.0, 0.0],
                start_joint_positions=np.zeros(7),
                joint_names=[f"joint_{index}" for index in range(7)],
            )
            self.assertEqual(len(path), 2)
            np.testing.assert_allclose(path[-1], np.full(7, 0.1))


if __name__ == "__main__":
    unittest.main()
