import unittest

import numpy as np

from aura_isaac_bridge.robot.motion_planner import (
    DiffusionConfig,
    SparseKeyposeDiffuser,
    compute_payload_tracking_state,
    container_place_candidates,
    select_continuation_start,
)


class SparseKeyposeDiffuserTest(unittest.TestCase):
    def test_joint_diffusion_preserves_internal_rrt_corner(self):
        diffuser = SparseKeyposeDiffuser(
            DiffusionConfig(max_joint_step_rad=0.05, min_frames=20)
        )
        start = np.array([0.0, 0.0])
        corner = np.array([1.0, 0.0])
        end = np.array([1.0, 1.0])

        trajectory = diffuser.diffuse_joint_keyposes(start, [corner, end])

        self.assertTrue(np.any(np.all(np.isclose(trajectory, corner), axis=1)))
        np.testing.assert_allclose(trajectory[-1], end)

    def test_joint_diffusion_eases_each_segment_endpoint(self):
        diffuser = SparseKeyposeDiffuser(
            DiffusionConfig(max_joint_step_rad=0.05, min_frames=40)
        )
        trajectory = diffuser.diffuse_joint_keyposes(
            np.array([0.0]),
            [np.array([1.0])],
        )
        steps = np.abs(np.diff(np.concatenate(([0.0], trajectory[:, 0]))))

        self.assertLess(steps[0], np.max(steps) * 0.2)
        self.assertLess(steps[-1], np.max(steps) * 0.2)

    def test_continuation_start_preserves_previous_command_when_tracking_is_close(self):
        start = select_continuation_start(
            measured_position=np.array([0.19, -0.11]),
            previous_command=np.array([0.20, -0.10]),
            maximum_tracking_error_rad=0.02,
        )

        np.testing.assert_allclose(start, [0.20, -0.10])

    def test_continuation_start_rejects_stale_command(self):
        start = select_continuation_start(
            measured_position=np.array([0.10, -0.10]),
            previous_command=np.array([0.20, -0.10]),
            maximum_tracking_error_rad=0.02,
        )

        np.testing.assert_allclose(start, [0.10, -0.10])

    def test_container_candidates_keep_payload_inside_wall_margin(self):
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([0.40, 0.30, 0.20])
        payload_half_extents = np.array([0.08, 0.03])

        candidates = container_place_candidates(
            lower,
            upper,
            payload_half_extents,
            base_xy=np.array([0.20, -1.0]),
            wall_margin_m=0.02,
        )

        usable_lower = lower[:2] + payload_half_extents + 0.02
        usable_upper = upper[:2] - payload_half_extents - 0.02
        self.assertGreater(len(candidates), 1)
        self.assertTrue(np.all(candidates >= usable_lower))
        self.assertTrue(np.all(candidates <= usable_upper))
        np.testing.assert_allclose(candidates[0], [0.20, 0.15])

    def test_payload_tracking_uses_live_geometry_for_replan_offset(self):
        expected, error, offset = compute_payload_tracking_state(
            reference_object_position=np.array([0.10, -0.40, 0.12]),
            reference_gripper_center=np.array([0.10, -0.40, 0.35]),
            current_gripper_center=np.array([0.22, -0.56, 0.37]),
            current_object_position=np.array([0.215, -0.568, 0.14]),
        )

        np.testing.assert_allclose(expected, [0.22, -0.56, 0.14])
        self.assertAlmostEqual(error, np.hypot(0.005, 0.008))
        np.testing.assert_allclose(offset, [-0.005, -0.008, -0.23])


if __name__ == "__main__":
    unittest.main()
