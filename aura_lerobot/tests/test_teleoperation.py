from collections import deque
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from aura_lerobot.dataset import EpisodeWriter
from aura_lerobot.teleop import DEFAULT_SETTINGS, Teleoperator
from aura_lerobot.teleop_inputs import (LeaderReceiver, decode_leader_packet, keyboard_target,
                                      leader_target, limited_target)
from aura_lerobot.tests import test_collection


class InputTests(unittest.TestCase):
    def setUp(self):
        self.limits = np.tile([-2.0, 2.0], (6, 1))
        self.robot = {"gripper_closed_rad": 0.0, "gripper_open_rad": .8, "tool_axis": [0, 0, 1]}

    def test_official_binary_format_and_invalid_packets(self):
        values = [-100, -50, 0, 50, 100, 25]
        np.testing.assert_allclose(decode_leader_packet(struct.pack("<6f", *values)), values)
        for packet in (b"", b"x" * 25, struct.pack("<6f", 0, 0, 0, 0, np.nan, 0),
                       struct.pack("<6f", 101, 0, 0, 0, 0, 0), struct.pack("<6f", 0, 0, 0, 0, 0, -1)):
            with self.assertRaises(ValueError):
                decode_leader_packet(packet)

    def test_normalized_units_are_mapped_to_radians(self):
        np.testing.assert_allclose(leader_target([-100, -50, 0, 50, 100, 25], self.limits, self.robot),
                                   [-2, -1, 0, 1, 2, .2])

    def test_custom_ranges_can_reverse_direction(self):
        ranges = np.tile([-1.0, 1.0], (5, 1))
        ranges[0] = [1, -1]
        np.testing.assert_allclose(leader_target([100] * 6, self.limits, self.robot, ranges),
                                   [-1, 1, 1, 1, 1, .8])
        ranges[0, 0] = 3
        with self.assertRaises(ValueError):
            leader_target(np.zeros(6), self.limits, self.robot, ranges)

    def test_limiter_bounds_first_leader_command(self):
        current = np.zeros(6)
        desired = np.array([100, -100, .005, 0, 10, -10])
        limited = limited_target(current, desired, self.limits, .6, 30)
        np.testing.assert_allclose(limited, [.02, -.02, .005, 0, .02, -.02])

    def test_keyboard_gripper_wrist_and_opposing_keys(self):
        command = np.zeros(6)
        target = keyboard_target({"U", "J", "K", "W", "S"}, command, None, self.robot, DEFAULT_SETTINGS, 30)
        self.assertAlmostEqual(target[5], .8)
        self.assertAlmostEqual(target[4], .4 / 30)
        self.assertAlmostEqual(target[3], .4 / 30)
        np.testing.assert_allclose(keyboard_target({"U", "O"}, command, None, self.robot, DEFAULT_SETTINGS, 30), command)

    def test_keyboard_translation_uses_world_frame_ik(self):
        fixture = test_collection.TeacherTests()
        fixture.setUp()
        command = np.array([.3, -.4, .8, -.4, .3, .8])
        old = fixture.model.forward(command[:5])
        target = keyboard_target({"A"}, command, fixture.model, self.robot, DEFAULT_SETTINGS, 30)
        actual = fixture.model.forward(target[:5])
        np.testing.assert_allclose(actual[:3, 3], old[:3, 3] + [0, .04 / 30, 0], atol=.0002)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "raw"
        writer = EpisodeWriter(self.root, {"camera": {"resolution": [16, 16]}, "recording": {"fps": 30}})
        self.collector = SimpleNamespace(
            writer=writer, command=np.zeros(6), limits=np.tile([-2, 2], (6, 1)), fps=30, model=None,
            robot_config={"gripper_closed_rad": 0., "gripper_open_rad": .8, "max_joint_velocity_rad_s": .6},
            initial_banana_pose=(np.zeros(3), np.array([1, 0, 0, 0])),
            task={"minimum_object_lift_m": .035})
        self.center = np.zeros(3)
        self.collector.object_points = lambda: self.center[None, :]
        self.collector.reset_episode = self.reset_scene
        self.collector.step_frame = self.step_frame
        self.collector.verify_placement = lambda: None
        self.keyboard = SimpleNamespace(keys=set(), actions=deque())
        self.keyboard.clear = lambda: (self.keyboard.keys.clear(), self.keyboard.actions.clear())
        self.operator = Teleoperator(self.collector, self.keyboard, "keyboard", DEFAULT_SETTINGS)

    def tearDown(self):
        self.temp.cleanup()

    def reset_scene(self):
        self.center[:] = 0
        return [[0, 0, 0], [1, 0, 0, 0]]

    def step_frame(self, target, phase):
        c = self.collector
        if c.writer.pending is not None:
            c.writer.add(c.command, target, np.zeros((16, 16, 3), dtype=np.uint8), c.frame_index / 30, phase)
            c.frame_index += 1
        c.command = np.asarray(target).copy()

    def test_only_explicit_success_commits_demonstration(self):
        self.operator.begin()
        self.center[2] = .05
        self.operator.tick()
        self.operator.complete()
        self.assertTrue((self.root / "episode_000000" / "frames.npz").exists())
        self.assertFalse(self.operator.armed)
        self.assertEqual(self.operator.episode_index, 1)
        self.assertIsNone(self.collector.writer.pending)

    def test_success_without_lift_is_not_committed(self):
        self.operator.begin()
        self.operator.tick()
        self.operator.complete()
        self.assertFalse((self.root / "episode_000000").exists())
        self.assertIsNotNone(self.collector.writer.pending)

    def test_placement_failure_can_continue_same_episode(self):
        self.operator.begin()
        self.center[2] = .05
        self.operator.tick()
        def fail():
            raise RuntimeError("Banana outside box")
        self.collector.verify_placement = fail
        self.operator.complete()
        pending = self.collector.writer.pending
        self.operator.begin()
        self.assertEqual(self.collector.writer.pending, pending)
        self.assertTrue(self.operator.armed)

    def test_failure_reset_discards_partial_data(self):
        self.operator.begin()
        self.operator.tick()
        self.operator.process_action("R")
        self.assertFalse(list(self.root.glob("episode_*")))
        self.assertFalse(json.loads((self.root / "attempts.jsonl").read_text())["success"])
        self.assertFalse(self.operator.armed)

    def test_disconnected_leader_holds_and_discards(self):
        self.operator.begin()
        self.operator.tick()
        self.operator.device = "so101leader"
        old = self.collector.command.copy()
        np.testing.assert_allclose(self.operator.target(None), old)
        self.assertFalse(self.operator.armed)
        self.assertIsNone(self.collector.writer.pending)

    def test_control_without_recording_creates_no_episode(self):
        self.collector.writer = None
        self.operator.begin()
        self.assertTrue(self.operator.armed)
        self.assertFalse(list(self.root.glob("episode_*")))


@unittest.skipUnless(importlib.util.find_spec("zmq"), "Optional pyzmq not installed")
class ReceiverTests(unittest.TestCase):
    def test_live_publisher_packet_validation_and_timeout(self):
        import zmq
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.LINGER, 0)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        receiver = LeaderReceiver(f"tcp://127.0.0.1:{port}", .03)
        try:
            self.assertIsNone(receiver.poll())
            deadline = time.monotonic() + 3
            while receiver.poll() is None and time.monotonic() < deadline:
                publisher.send(struct.pack("<6f", 1, 2, 3, 4, 5, 6))
                time.sleep(.01)
            np.testing.assert_allclose(receiver.latest(), [1, 2, 3, 4, 5, 6])
            publisher.send(b"bad")
            deadline = time.monotonic() + 1
            while receiver.invalid_packets == 0 and time.monotonic() < deadline:
                receiver.poll()
                time.sleep(.005)
            self.assertGreater(receiver.invalid_packets, 0)
            time.sleep(.04)
            self.assertIsNone(receiver.latest())
        finally:
            receiver.close()
            publisher.close()
            context.term()


if __name__ == "__main__":
    unittest.main()
