import unittest

import numpy as np

from aura_isaac_bridge.core.gripper_contact import classify_finger_contacts


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


if __name__ == "__main__":
    unittest.main()
