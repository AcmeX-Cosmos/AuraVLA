import unittest

import numpy as np

from aura_isaac_bridge.core.grasp_fusion import (
    GraspFusionError,
    GraspObservation,
    fuse_grasp_observations,
)


def observation(position, orientation=(1.0, 0.0, 0.0, 0.0), **kwargs):
    return GraspObservation(
        position=np.asarray(position, dtype=float),
        orientation=np.asarray(orientation, dtype=float),
        **kwargs,
    )


class GraspFusionTest(unittest.TestCase):
    def test_weighted_position_and_quaternion_hemisphere_fusion(self):
        result = fuse_grasp_observations(
            [
                observation([0.0, 0.0, 0.0], score=1.0),
                observation([0.01, 0.0, 0.0], orientation=(-1.0, 0.0, 0.0, 0.0), score=0.5),
            ],
            max_position_dispersion_m=0.02,
        )

        self.assertEqual(result["accepted_frame_count"], 2)
        self.assertGreater(result["confidence"], 0.0)
        self.assertAlmostEqual(result["orientation"][0], 1.0, places=6)
        self.assertGreater(result["position"][0], 0.0)
        self.assertLess(result["position"][0], 0.01)

    def test_temporal_position_outlier_is_rejected(self):
        result = fuse_grasp_observations(
            [
                observation([0.000, 0.0, 0.0]),
                observation([0.001, 0.0, 0.0]),
                observation([0.100, 0.0, 0.0]),
            ],
            max_position_dispersion_m=0.02,
        )

        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["accepted_frame_count"], 2)
        self.assertEqual(result["rejected_frame_count"], 1)
        self.assertLess(result["position"][0], 0.01)

    def test_dispersion_gate_rejects_inconsistent_frames(self):
        with self.assertRaises(GraspFusionError):
            fuse_grasp_observations(
                [observation([0.0, 0.0, 0.0]), observation([0.04, 0.0, 0.0])],
                max_position_dispersion_m=0.01,
                position_outlier_floor_m=0.05,
            )

    def test_invalid_observations_are_not_used(self):
        result = fuse_grasp_observations(
            [
                observation([0.0, 0.0, 0.0]),
                observation([np.nan, 0.0, 0.0]),
                observation([0.0, 0.0, 0.0], orientation=(0.0, 0.0, 0.0, 0.0)),
            ]
        )
        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(result["accepted_frame_count"], 1)

    def test_no_valid_observations_fail_closed(self):
        with self.assertRaises(GraspFusionError):
            fuse_grasp_observations(
                [observation([np.inf, 0.0, 0.0])]
            )


if __name__ == "__main__":
    unittest.main()
