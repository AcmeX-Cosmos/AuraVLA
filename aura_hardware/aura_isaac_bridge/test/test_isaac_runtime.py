import unittest

from aura_isaac_bridge.isaac_runtime import IsaacRuntimeLauncher


class IsaacRuntimeSourcePreludeTests(unittest.TestCase):
    def setUp(self):
        self.launcher = IsaacRuntimeLauncher(None)

    def test_camera_restore_preserves_runtime_modules(self):
        source = self.launcher._with_camera_environment(
            "camera_started = True\n",
            reload_modules=False,
        )

        self.assertIn("_previous_camera_bridge.stop()", source)
        self.assertNotIn("sys.modules.pop(_module_name", source)
        self.assertNotIn("_graspnet_net", source)

    def test_runtime_reload_clears_inference_modules(self):
        source = self.launcher._with_camera_environment("runtime_started = True\n")

        self.assertIn("sys.modules.pop(_module_name", source)
        self.assertIn("_graspnet_net", source)


if __name__ == "__main__":
    unittest.main()
