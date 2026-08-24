import unittest

import numpy as np

from aura_isaac_bridge.robot.motion_planner import (
    DiffusionConfig,
    SparseKeyposeDiffuser,
    container_place_candidates,
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


if __name__ == "__main__":
    unittest.main()
