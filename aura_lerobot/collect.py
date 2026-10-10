"""Run with Isaac Sim's python.sh; scene construction remains the user's job."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from .dataset import EpisodeWriter
from .teacher import Kinematics, inside_box, trajectory


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Referenced paths are resolved from the workspace/launch directory.
    stage = Path(config["stage_path"]).expanduser()
    if not config["stage_path"] or not stage.is_file():
        raise ValueError("Set stage_path to your existing SO101 USD scene")
    config["stage_path"] = str(stage.resolve())
    robot, task, recording = config["robot"], config["task"], config["recording"]
    if len(robot["joint_names"]) != 6 or len(set(robot["joint_names"])) != 6:
        raise ValueError("joint_names must list five arm joints then one gripper joint")
    for name, value, size in (
            ("home_rad", robot["home_rad"], 5),
            ("tcp_offset_m", robot["tcp_offset_m"], 3),
            ("tool_axis", robot["tool_axis"], 3),
            ("approach_direction", task["approach_direction"], 3),
            ("grasp_offset_m", task["grasp_offset_m"], 3),
            ("reset_xy_jitter_m", task["reset_xy_jitter_m"], 2)):
        if np.asarray(value).shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}")
    for value in (robot["tool_axis"], task["approach_direction"]):
        if np.linalg.norm(value) < 1e-8:
            raise ValueError("Tool axis and approach direction cannot be zero")
    bounds = np.asarray(task["box_inner_bounds_m"])
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.all(bounds[1] > bounds[0]):
        raise ValueError("box_inner_bounds_m must contain increasing local min/max XYZ")
    if np.any(np.asarray(task["reset_xy_jitter_m"]) < 0):
        raise ValueError("Reset jitter cannot be negative")
    fps, hz = recording["fps"], recording["physics_hz"]
    if not isinstance(fps, int) or not isinstance(hz, int) or fps <= 0 or hz < fps or hz % fps:
        raise ValueError("physics_hz must be an integer multiple of positive integer fps")
    if recording["episodes"] < 0 or recording["max_consecutive_failures"] < 1:
        raise ValueError("Invalid episode/failure limits")
    for key in ("hover_height_m", "lift_height_m", "minimum_object_lift_m", "maximum_carry_slip_m", "segment_seconds",
                "gripper_seconds", "settle_seconds", "success_hold_seconds", "success_max_speed_m_s"):
        if not np.isfinite(task[key]) or task[key] <= 0:
            raise ValueError(f"task.{key} must be positive")
    if not np.isfinite(task["drop_clearance_m"]) or task["drop_clearance_m"] < 0:
        raise ValueError("drop_clearance_m must be nonnegative")
    if not np.isfinite(robot["max_joint_velocity_rad_s"]) or robot["max_joint_velocity_rad_s"] <= 0:
        raise ValueError("max_joint_velocity_rad_s must be positive")
    resolution = config["camera"]["resolution"]
    if len(resolution) != 2 or any(not isinstance(v, int) or v < 16 for v in resolution):
        raise ValueError("camera.resolution must contain positive integer width/height >= 16")
    return config


def matrix_from_pose(position, orientation):
    # Isaac's world quaternions use [w, x, y, z].
    w, x, y, z = np.asarray(orientation) / np.linalg.norm(orientation)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]]
    result[:3, 3] = position
    return result


def rigid_body_corners(prim):
    from pxr import Usd, UsdGeom
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    # ComputeLocalBound includes the root's transform; world pose is applied
    # separately in object_points, so use geometry in the rigid body's frame.
    local_range = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
    if local_range.IsEmpty():
        raise ValueError("Banana must contain nonempty three-dimensional geometry")
    corners = np.array(list(itertools.product(*zip(local_range.GetMin(), local_range.GetMax()))))
    if not np.isfinite(corners).all() or np.any(np.ptp(corners, axis=0) <= 0):
        raise ValueError("Banana must contain nonempty three-dimensional geometry")
    return corners


class Collector:
    def __init__(self, app, config, *, record=True):
        # Isaac imports must follow SimulationApp creation.
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import get_current_stage, open_stage, is_stage_loading
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from pxr import Usd, UsdGeom, UsdPhysics
        import carb

        self.app, self.config = app, config
        self.robot_config, self.task = config["robot"], config["task"]
        self.fps = config["recording"]["fps"]
        self.substeps = config["recording"]["physics_hz"] // self.fps
        self.Action = ArticulationAction
        if not open_stage(config["stage_path"]):
            raise RuntimeError("Unable to open stage")
        while is_stage_loading():
            app.update()
        self.stage = get_current_stage()
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage) - 1.0) > 1e-6:
            raise ValueError("Collector requires a meter-unit scene")
        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z:
            raise ValueError("Collector requires a Z-up scene")
        for key in ("prim_path", "end_effector_path"):
            self.require_prim(self.robot_config[key])
        self.require_prim(config["camera"]["prim_path"], "Camera")
        banana_prim = self.require_prim(self.task["banana_path"])
        self.require_prim(self.task["box_path"])
        for prim in (banana_prim, self.require_prim(self.robot_config["end_effector_path"])):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Expected rigid-body ROOT at {prim.GetPath()}")
            self.check_unit_scale(prim)
        rigid_api = UsdPhysics.RigidBodyAPI(banana_prim)
        if rigid_api.GetKinematicEnabledAttr().Get() or rigid_api.GetRigidBodyEnabledAttr().Get() is False:
            raise ValueError("Banana must be a dynamic, enabled rigid body")
        if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in
                   Usd.PrimRange(self.require_prim(self.robot_config["prim_path"]))):
            raise ValueError("Robot prim must contain an articulation root")
        # USD geometry queries must track the live physical scene in this dedicated instance.
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        self.sim = SimulationContext(physics_dt=1 / config["recording"]["physics_hz"],
                                     rendering_dt=1 / self.fps, stage_units_in_meters=1,
                                     backend="numpy", device="cpu")
        self.robot = SingleArticulation(self.robot_config["prim_path"], reset_xform_properties=False)
        self.banana = SingleRigidPrim(self.task["banana_path"], reset_xform_properties=False)
        self.eef = SingleRigidPrim(self.robot_config["end_effector_path"], reset_xform_properties=False)
        self.sim.reset()
        for prim in (self.robot, self.banana, self.eef):
            prim.initialize()
        self.indices = np.array([self.robot.get_dof_index(name) for name in self.robot_config["joint_names"]])
        props = self.robot.dof_properties
        self.limits = np.column_stack((props["lower"][self.indices], props["upper"][self.indices]))
        if not np.isfinite(self.limits).all() or not np.all(self.limits[:, 1] > self.limits[:, 0]):
            raise ValueError("All six configured revolute joints require finite joint limits")
        if np.any(props["stiffness"][self.indices] <= 0):
            raise ValueError("Configure positive position-drive stiffness for all six SO101 joints in your scene")
        self.home = np.r_[self.robot_config["home_rad"], self.robot_config["gripper_open_rad"]]
        for target in (self.home, np.r_[self.home[:5], self.robot_config["gripper_closed_rad"]]):
            self.check_limits(target)
        self.camera = Camera(prim_path=config["camera"]["prim_path"], frequency=-1,
                             resolution=tuple(config["camera"]["resolution"]))
        self.camera.initialize()
        self.command = self.state()
        self.writer = None
        self.frame_index = 0
        self.last_camera_time = None
        self.rng = np.random.default_rng(config["recording"]["seed"])
        self.hold(0.5, "warmup")
        self.move(self.home, "home")
        self.hold(self.task["settle_seconds"], "home_settle")
        self.check_tracking()
        self.initial_banana_pose = tuple(np.array(x).copy() for x in self.banana.get_world_pose())
        self.banana_corners = rigid_body_corners(banana_prim)
        self.model = self.build_kinematics()
        if record:
            self.writer = EpisodeWriter(config["recording"]["root"], config)

    def require_prim(self, path, type_name=None):
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid() or (type_name and prim.GetTypeName() != type_name):
            raise ValueError(f"Missing/invalid scene prim: {path}")
        return prim

    @staticmethod
    def check_unit_scale(prim):
        from pxr import UsdGeom
        linear = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))[:3, :3]
        if not np.allclose(linear.T @ linear, np.eye(3), atol=1e-5) or np.linalg.det(linear) < 0:
            raise ValueError(f"Bake scale into geometry; rigid-body world transforms must have unit scale: {prim.GetPath()}")

    def state(self):
        return np.asarray(self.robot.get_joint_positions(joint_indices=self.indices), dtype=float)

    def check_limits(self, target):
        if not np.isfinite(target).all() or np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1]):
            raise ValueError(f"Joint target outside SO101 limits: {np.asarray(target).tolist()}")

    def tcp_pose(self):
        pose = matrix_from_pose(*self.eef.get_world_pose())
        pose[:3, 3] += pose[:3, :3] @ np.asarray(self.robot_config["tcp_offset_m"])
        return pose

    def build_kinematics(self):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Usd, UsdPhysics
        # Match USD joints by name and validate a single serial, forward-directed chain.
        prims = list(Usd.PrimRange(self.require_prim(self.robot_config["prim_path"])))
        fixed_parents = {}
        for prim in prims:
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if parent and child:
                    fixed_parents[str(child[0])] = str(parent[0])
        gripper_matches = [p for p in prims if p.GetName() == self.robot_config["joint_names"][5]
                           and p.IsA(UsdPhysics.RevoluteJoint)]
        if len(gripper_matches) != 1:
            raise ValueError("Gripper must be one named USD RevoluteJoint; actions use radians")

        def connected(parent, child):
            visited = set()
            while child != parent and child in fixed_parents and child not in visited:
                visited.add(child)
                child = fixed_parents[child]
            return child == parent

        axes, origins, previous_child = [], [], None
        for name in self.robot_config["joint_names"][:5]:
            matches = [p for p in prims if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one USD RevoluteJoint named {name}")
            joint = UsdPhysics.RevoluteJoint(matches[0])
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parent or not child:
                raise ValueError(f"Arm joint {name} must have body0 and body1")
            if previous_child is not None and not connected(previous_child, str(parent[0])):
                raise ValueError("Arm joints must form a proximal-to-distal serial body0->body1 chain")
            previous_child = str(child[0])
            parent_prim = self.require_prim(str(parent[0]))
            if not parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Joint body0 must be a rigid body: {parent[0]}")
            self.check_unit_scale(parent_prim)
            body = SingleRigidPrim(str(parent[0]), reset_xform_properties=False)
            body.initialize()
            parent_pose = matrix_from_pose(*body.get_world_pose())
            quat = joint.GetLocalRot0Attr().Get()
            local_pose = matrix_from_pose(joint.GetLocalPos0Attr().Get(), [quat.GetReal(), *quat.GetImaginary()])
            joint_pose = parent_pose @ local_pose
            local_axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[joint.GetAxisAttr().Get()]
            axes.append(joint_pose[:3, :3] @ local_axis)
            origins.append(joint_pose[:3, 3])
        if not connected(previous_child, self.robot_config["end_effector_path"]):
            raise ValueError("end_effector_path must be the wrist link or its fixed rigid-body descendant")
        return Kinematics(axes, origins, self.state()[:5], self.tcp_pose(), self.limits[:5])

    def step_frame(self, target, phase):
        if not self.app.is_running() or not self.sim.is_playing():
            raise KeyboardInterrupt("Simulator closed/paused")
        self.check_limits(target)
        # RGB and measured q_t precede commanded q_target_t. Never label measured q as action.
        if self.writer is not None and self.writer.pending is not None:
            frame = self.camera.get_current_frame()
            camera_time = frame.get("rendering_time")
            if camera_time is None or camera_time <= (self.last_camera_time if self.last_camera_time is not None else -1):
                raise RuntimeError("Camera did not deliver a fresh frame; episode discarded")
            if abs(camera_time - self.sim.current_time) > 0.5 / self.fps:
                raise RuntimeError("Camera/state simulation timestamps are not synchronized")
            self.writer.add(self.state(), target, np.asarray(self.camera.get_rgba())[..., :3].copy(),
                            self.frame_index / self.fps, phase)
            self.last_camera_time = camera_time
            self.frame_index += 1
        self.robot.apply_action(self.Action(joint_positions=np.asarray(target), joint_indices=self.indices))
        self.command = np.asarray(target).copy()
        for _ in range(self.substeps):
            self.sim.step(render=False)
        self.render_synchronized()

    def render_synchronized(self):
        # RTX delivery can lag by multiple render frames. Keep physics frozen
        # while draining that pipeline; one app update is not sufficient.
        sample_time = self.sim.current_time
        width, height = self.config["camera"]["resolution"]
        for _ in range(16):
            self.sim.render()
            if abs(self.sim.current_time - sample_time) > 1e-6:
                raise RuntimeError("Physics advanced during camera synchronization")
            frame_time = self.camera.get_current_frame().get("rendering_time", -1)
            rgba = np.asarray(self.camera.get_rgba())
            if abs(frame_time - sample_time) <= 1e-6 and rgba.shape == (height, width, 4):
                return
        raise RuntimeError("Camera could not synchronize to the physical sampling point")

    def move(self, target, phase, seconds=None):
        for command in trajectory(self.command, target, self.fps,
                                  seconds or self.task["segment_seconds"],
                                  self.robot_config["max_joint_velocity_rad_s"]):
            self.step_frame(command, phase)

    def hold(self, seconds, phase):
        for _ in range(max(1, int(np.ceil(seconds * self.fps)))):
            self.step_frame(self.command, phase)

    def check_tracking(self):
        if np.max(np.abs(self.state() - self.command)) > 0.12:
            raise RuntimeError("Position drives did not track target; check collisions, gains and home pose")

    def object_points(self):
        pose = matrix_from_pose(*self.banana.get_world_pose())
        return self.banana_corners @ pose[:3, :3].T + pose[:3, 3]

    def box_pose(self):
        from pxr import UsdGeom
        # Gf matrices use row vectors; our numpy transforms use column vectors.
        prim = self.require_prim(self.task["box_path"])
        self.check_unit_scale(prim)
        return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T

    def reset_episode(self):
        # Reset is excluded from demonstrations; teleport only happens between episodes.
        self.move(self.home, "reset_home")
        self.hold(self.task["settle_seconds"], "reset_home_settle")
        self.check_tracking()
        position, orientation = (x.copy() for x in self.initial_banana_pose)
        jitter = np.asarray(self.task["reset_xy_jitter_m"])
        position[:2] += self.rng.uniform(-jitter, jitter)
        self.banana.set_world_pose(position=position, orientation=orientation)
        self.banana.set_linear_velocity(np.zeros(3))
        self.banana.set_angular_velocity(np.zeros(3))
        self.hold(self.task["settle_seconds"], "reset_object_settle")
        return [position.tolist(), orientation.tolist()]

    def move_tcp(self, q, gripper, phase):
        self.move(np.r_[q, gripper], phase)
        self.hold(0.2, phase + "_settle")
        # Check only the arm: a physical grasp can prevent the jaw reaching its closed target.
        if np.max(np.abs(self.state()[:5] - q)) > 0.12:
            raise RuntimeError(f"Arm failed to track {phase}; check collisions/position drives")
        expected, actual = self.model.forward(self.state()[:5]), self.tcp_pose()
        if (np.linalg.norm(expected[:3, 3] - actual[:3, 3]) > 0.008 or
                np.linalg.norm(expected[:3, :3] - actual[:3, :3]) > 0.15):
            raise RuntimeError("Live TCP disagrees with kinematics; check fixed base and USD joint frames")

    def pick_place(self):
        points = self.object_points()
        initial_center = np.mean(points, axis=0)
        grasp = initial_center + np.asarray(self.task["grasp_offset_m"])
        hover = grasp + [0, 0, self.task["hover_height_m"]]
        lifted = grasp + [0, 0, self.task["lift_height_m"]]
        box_pose = self.box_pose()
        # Task geometry uses world Z; require a level box for vertical approach/drop.
        if np.dot(box_pose[:3, 2], [0, 0, 1]) < 0.999:
            raise ValueError("Box's local +Z must point upward; calibrate its frame/interior bounds")
        bounds = np.asarray(self.task["box_inner_bounds_m"])
        drop_local = (bounds[0] + bounds[1]) / 2
        drop_local[2] = bounds[1, 2]
        drop = (box_pose @ np.r_[drop_local, 1])[:3]
        # Preserve object-to-TCP offset; object's lowest point clears the rim before release.
        drop[2] += self.task["drop_clearance_m"] + grasp[2] - points[:, 2].min()
        transport_z = max(lifted[2], drop[2]) + self.task["hover_height_m"]
        route = [hover, grasp, lifted, np.r_[lifted[:2], transport_z],
                 np.r_[drop[:2], transport_z], drop]
        seed = self.state()[:5]
        solutions = []
        # Resolve the complete route before closing the gripper, so unreachable boxes fail early.
        for target in route:
            seed = self.model.solve(target, self.robot_config["tool_axis"],
                                    self.task["approach_direction"], seed)
            solutions.append(seed)
        opened, closed = self.robot_config["gripper_open_rad"], self.robot_config["gripper_closed_rad"]
        self.move_tcp(solutions[0], opened, "approach")
        self.move_tcp(solutions[1], opened, "descend")
        self.hold(0.25, "grasp_settle")
        self.check_tracking()
        self.move(np.r_[solutions[1], closed], "close", self.task["gripper_seconds"])
        self.hold(0.3, "contact_settle")
        self.move_tcp(solutions[2], closed, "lift")
        self.hold(0.3, "lift_settle")
        if np.mean(self.object_points(), axis=0)[2] - initial_center[2] < self.task["minimum_object_lift_m"]:
            raise RuntimeError("Banana did not physically lift; failed grasp")
        tcp = self.tcp_pose()
        carry_offset = tcp[:3, :3].T @ (np.mean(self.object_points(), axis=0) - tcp[:3, 3])
        for q, phase in zip(solutions[3:], ("raise_transport", "transport", "place")):
            self.move_tcp(q, closed, phase)
            if np.mean(self.object_points(), axis=0)[2] - initial_center[2] < self.task["minimum_object_lift_m"]:
                raise RuntimeError("Banana slipped during transport")
            tcp = self.tcp_pose()
            live_offset = tcp[:3, :3].T @ (np.mean(self.object_points(), axis=0) - tcp[:3, 3])
            if np.linalg.norm(live_offset - carry_offset) > self.task["maximum_carry_slip_m"]:
                raise RuntimeError("Banana moved relative to the gripper during transport")
        self.move(np.r_[solutions[-1], opened], "release", self.task["gripper_seconds"])
        self.hold(0.3, "release_settle")
        self.move_tcp(solutions[-2], opened, "retreat")
        self.hold(self.task["settle_seconds"], "settle")
        self.verify_placement()

    def verify_placement(self):
        bounds = np.asarray(self.task["box_inner_bounds_m"])
        for _ in range(int(np.ceil(self.task["success_hold_seconds"] * self.fps))):
            if not inside_box(self.object_points(), self.box_pose(), bounds):
                raise RuntimeError("Banana is not fully inside configured box interior")
            if np.linalg.norm(self.banana.get_linear_velocity()) > self.task["success_max_speed_m_s"]:
                raise RuntimeError("Banana has not settled inside box")
            self.step_frame(self.command, "success_hold")

    def run(self):
        successes = failures = 0
        limit = self.config["recording"]["episodes"]
        try:
            while limit == 0 or successes < limit:
                reset_pose = self.reset_episode()
                self.writer.begin(successes)
                self.frame_index, self.last_camera_time = 0, None
                try:
                    self.pick_place()
                except (RuntimeError, ValueError) as exc:
                    self.writer.finish(success=False, reason=str(exc))
                    failures += 1
                    print(f"[SO101] failed attempt ({failures} consecutive): {exc}", flush=True)
                    if failures >= self.config["recording"]["max_consecutive_failures"]:
                        raise RuntimeError("Repeated failures; calibrate scene/config before collecting further") from exc
                    continue
                self.writer.finish(success=True, reset_pose=reset_pose)
                successes += 1
                failures = 0
                print(f"[SO101] saved successful episode {successes}; output={self.writer.root}", flush=True)
        finally:
            self.writer.finish(success=False, reason="Interrupted or aborted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="aura_lerobot/config/so101_banana.local.yaml")
    parser.add_argument("--episodes", type=int, help="Number of successes; 0 repeats until Ctrl+C")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--check-config", action="store_true", help="Validate paths/numeric config without starting Isaac")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episodes is not None:
        if args.episodes < 0:
            parser.error("--episodes must be nonnegative")
        config["recording"]["episodes"] = args.episodes
    if args.check_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    output = Path(config["recording"]["root"]).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Raw output must be empty: {output}")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless})
    try:
        Collector(app, config).run()
    except KeyboardInterrupt:
        print("Collection stopped; completed episodes preserved.")
    finally:
        app.close()


if __name__ == "__main__":
    main()
