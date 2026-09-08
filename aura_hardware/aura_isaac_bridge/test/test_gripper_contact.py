import unittest

import numpy as np

from aura_isaac_bridge.core.gripper_contact import (
    classify_finger_contacts,
    evaluate_grasp_stability,
)


class GripperContactClassificationTests(unittest.TestCase):
    def test_position_blocking_confirms_contact_when_force_is_below_threshold(self):
        contacts = classify_finger_contacts(
            feedback=[0.0188, 0.0128],
            command_targets=[0.0170, 0.0],
            efforts=[0.90, 0.84],
            residual_threshold=0.0015,
            force_threshold=2.0,
        )

        np.testing.assert_array_equal(contacts, [True, True])

    def test_each_finger_must_have_independent_physical_evidence(self):
        contacts = classify_finger_contacts(
            feedback=[0.0188, 0.0002],
            command_targets=[0.0170, 0.0],
            efforts=[0.90, 0.84],
            residual_threshold=0.0015,
            force_threshold=2.0,
        )

        np.testing.assert_array_equal(contacts, [True, False])

    def test_force_can_confirm_contact_without_position_residual(self):
        contacts = classify_finger_contacts(
            feedback=[0.0172, 0.0002],
            command_targets=[0.0170, 0.0],
            efforts=[2.1, 2.2],
            residual_threshold=0.0015,
            force_threshold=2.0,
        )

        np.testing.assert_array_equal(contacts, [True, True])

    def test_single_sided_force_is_not_a_stable_grasp(self):
        stability = evaluate_grasp_stability(
            finger_contacts=[True, True],
            efforts=[0.05, 2.5],
            residuals=[0.002, 0.002],
            force_threshold=0.25,
            residual_threshold=0.0015,
        )

        self.assertFalse(stability["stable"])
        self.assertFalse(stability["force_balanced"])

    def test_balanced_force_is_a_stable_grasp(self):
        stability = evaluate_grasp_stability(
            finger_contacts=[True, True],
            efforts=[1.2, 1.0],
            residuals=[0.002, 0.002],
            force_threshold=0.25,
            residual_threshold=0.0015,
        )

        self.assertTrue(stability["stable"])
        self.assertTrue(stability["force_balanced"])

    def test_balanced_position_residual_can_confirm_without_force(self):
        stability = evaluate_grasp_stability(
            finger_contacts=[True, True],
            efforts=[0.05, 0.10],
            residuals=[0.003, 0.0025],
            force_threshold=0.25,
            residual_threshold=0.0015,
        )

        self.assertTrue(stability["stable"])
        self.assertTrue(stability["residual_balanced"])

    def test_extreme_position_residual_imbalance_is_rejected(self):
        stability = evaluate_grasp_stability(
            finger_contacts=[True, True],
            efforts=[0.05, 0.10],
            residuals=[0.0005, 0.010],
            force_threshold=0.25,
            residual_threshold=0.0015,
        )

        self.assertFalse(stability["stable"])
        self.assertFalse(stability["residual_balanced"])


if __name__ == "__main__":
    unittest.main()
