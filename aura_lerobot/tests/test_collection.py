import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aura_lerobot.collect import matrix_from_pose, rigid_body_corners
from aura_lerobot.dataset import EpisodeWriter
from aura_lerobot.export import export_dataset, load_episode
from aura_lerobot.teacher import Kinematics, inside_box, trajectory


class TeacherTests(unittest.TestCase):
    def setUp(self):
        axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]
        origins = [[0, 0, 0], [0, 0, .08], [.18, 0, .08], [.36, 0, .08], [.40, 0, .08]]
        reference = np.eye(4)
        reference[:3, 3] = [.44, 0, .08]
        self.model = Kinematics(axes, origins, np.zeros(5), reference, np.tile([-2, 2], (5, 1)))

    def test_forward_reference_and_rotated_base(self):
        np.testing.assert_allclose(self.model.forward(np.zeros(5))[:3, 3], [.44, 0, .08])
        np.testing.assert_allclose(self.model.forward([np.pi / 2, 0, 0, 0, 0])[:3, 3],
                                   [0, .44, .08], atol=1e-8)

    def test_ik_recovers_reachable_pose_direction(self):
        target = self.model.forward([.3, -.4, .8, -.4, .3])
        direction = target[:3, :3] @ [0, 0, 1]
        q = self.model.solve(target[:3, 3], [0, 0, 1], direction, np.zeros(5),
                             position_tolerance=.001, angle_tolerance_deg=2)
        actual = self.model.forward(q)
        self.assertLess(np.linalg.norm(actual[:3, 3] - target[:3, 3]), .001)
        self.assertGreater(np.dot(actual[:3, :3] @ [0, 0, 1], direction), np.cos(np.radians(2)))
        self.assertTrue(np.all(np.abs(q) <= 2))

    def test_ik_rejects_unreachable_target(self):
        with self.assertRaisesRegex(RuntimeError, "unreachable"):
            self.model.solve([4, 4, 4], [0, 0, 1], [0, 0, -1], np.zeros(5), attempts=2)

    def test_reference_can_be_nonzero_measured_configuration(self):
        q0 = np.array([.3, -.4, .8, -.4, .3])
        # Transform each screw into its position at q0 to emulate the live USD adapter.
        prefix = np.eye(4)
        axes, origins = [], []
        from aura_lerobot.teacher import revolute_transform
        for axis, origin, delta in zip(self.model.axes, self.model.origins, q0):
            axes.append(prefix[:3, :3] @ axis)
            origins.append((prefix @ np.r_[origin, 1])[:3])
            prefix = prefix @ revolute_transform(axis, origin, delta)
        calibrated = Kinematics(axes, origins, q0, self.model.forward(q0), self.model.limits)
        q1 = [.4, -.3, .9, -.2, .2]
        np.testing.assert_allclose(calibrated.forward(q1), self.model.forward(q1), atol=1e-8)

    def test_minimum_jerk_respects_speed_and_endpoint(self):
        start = np.zeros(6)
        goal = np.array([1, -2, .5, .8, 0, .4])
        commands = np.array([start, *trajectory(start, goal, 30, .1, .6)])
        np.testing.assert_allclose(commands[-1], goal, atol=1e-8)
        self.assertLessEqual(np.max(np.abs(np.diff(commands, axis=0))) * 30, .6 + 1e-8)

    def test_box_containment_uses_local_coordinates_and_full_extent(self):
        pose = matrix_from_pose([1, 2, .5], [np.sqrt(.5), 0, 0, np.sqrt(.5)])
        local = np.array([[-.1, -.2, .05], [.1, .2, .15]])
        world = local @ pose[:3, :3].T + pose[:3, 3]
        bounds = [[-.15, -.25, 0], [.15, .25, .2]]
        self.assertTrue(inside_box(world, pose, bounds))
        world[1, 2] += .1
        self.assertFalse(inside_box(world, pose, bounds))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "raw"
        self.config = {"camera": {"resolution": [16, 16]},
                       "recording": {"fps": 30, "task": "Put the banana in the box"}}
        self.writer = EpisodeWriter(self.root, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def write_episode(self, success=True):
        self.writer.begin(0)
        for i in range(3):
            self.writer.add(np.full(6, i * .01), np.full(6, (i + 1) * .01),
                            np.full((16, 16, 3), i, dtype=np.uint8), i / 30, "lift")
        self.writer.finish(success=success)
        return self.root / "episode_000000"

    def test_atomic_commit_preserves_command_state_alignment(self):
        episode = self.write_episode()
        self.assertFalse(list(self.root.glob("*.partial")))
        state, action, images = load_episode(episode, 30, [16, 16])
        np.testing.assert_allclose(action - state, .01, atol=1e-8)
        self.assertEqual(len(images), 3)
        export_dataset(self.root, Path(self.temp.name) / "unused", "local/test", validate_only=True)

    def test_failed_episode_is_excluded_and_logged(self):
        episode = self.write_episode(success=False)
        self.assertFalse(episode.exists())
        self.assertFalse(list(self.root.glob("*.partial")))
        log = json.loads((self.root / "attempts.jsonl").read_text())
        self.assertFalse(log["success"])
        with self.assertRaisesRegex(ValueError, "No committed"):
            export_dataset(self.root, Path(self.temp.name) / "unused", "local/test", validate_only=True)

    def test_missing_image_and_irregular_timestamps_rejected(self):
        episode = self.write_episode()
        image = episode / "rgb" / "000001.png"
        original = image.read_bytes()
        image.unlink()
        with self.assertRaisesRegex(ValueError, "Missing/misaligned"):
            load_episode(episode, 30, [16, 16])
        image.write_bytes(original)
        with np.load(episode / "frames.npz") as data:
            fields = {name: data[name] for name in data.files}
        fields["timestamp"][1] += .01
        np.savez_compressed(episode / "frames.npz", **fields)
        with self.assertRaisesRegex(ValueError, "Irregular"):
            load_episode(episode, 30, [16, 16])

    def test_partial_episode_not_exported(self):
        self.writer.begin(0)
        with self.assertRaisesRegex(ValueError, "No committed"):
            export_dataset(self.root, Path(self.temp.name) / "unused", "local/test", validate_only=True)
        self.writer.finish(success=False, reason="Interrupted")
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_existing_raw_directory_is_not_overwritten(self):
        self.write_episode()
        with self.assertRaises(FileExistsError):
            EpisodeWriter(self.root, self.config)


@unittest.skipUnless(importlib.util.find_spec("pxr"), "Run inside initialized Isaac Sim for USD geometry tests")
class USDGeometryTests(unittest.TestCase):
    def test_rigid_geometry_does_not_include_root_translation_twice(self):
        from pxr import Usd, UsdGeom, Gf
        stage = Usd.Stage.CreateInMemory()
        cube = UsdGeom.Cube.Define(stage, "/banana")
        cube.CreateSizeAttr(.2)
        cube.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        cube.AddRotateZOp().Set(90)
        corners = rigid_body_corners(cube.GetPrim())
        np.testing.assert_allclose(corners.min(axis=0), [-.1, -.1, -.1])
        np.testing.assert_allclose(corners.max(axis=0), [.1, .1, .1])
        # Apply the root pose once, exactly as the simulator adapter does.
        pose = matrix_from_pose([1, 2, 3], [np.sqrt(.5), 0, 0, np.sqrt(.5)])
        points = corners @ pose[:3, :3].T + pose[:3, 3]
        np.testing.assert_allclose(points.mean(axis=0), [1, 2, 3])

    def test_empty_rigid_geometry_is_rejected(self):
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/banana").GetPrim()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            rigid_body_corners(prim)


if __name__ == "__main__":
    unittest.main()
