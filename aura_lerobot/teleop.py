"""SO101 keyboard / leisaac Leader teleoperation in a user-authored Isaac scene."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from .collect import Collector, load_config
from .teleop_inputs import LeaderReceiver, keyboard_target, leader_target, limited_target


DEFAULT_SETTINGS = {"translation_speed_m_s": 0.04, "wrist_speed_rad_s": 0.4,
                    "remote_timeout_seconds": 0.5, "arm_range_rad": None}
CONTROLS = "B start/resume | SPACE pause | R discard+reset | N verify+save+reset | BACKSPACE discard | ESC quit"


def teleop_settings(config):
    settings = {**DEFAULT_SETTINGS, **config.get("teleoperation", {})}
    for key in ("translation_speed_m_s", "wrist_speed_rad_s", "remote_timeout_seconds"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"teleoperation.{key} must be positive")
    ranges = settings["arm_range_rad"]
    if ranges is not None and (np.asarray(ranges).shape != (5, 2) or not np.isfinite(ranges).all()):
        raise ValueError("teleoperation.arm_range_rad must contain five finite pairs")
    return settings


class Keyboard:
    def __init__(self):
        import carb.input
        import omni.appwindow
        self.events = carb.input.KeyboardEventType
        self.interface = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.keys = set()
        self.actions = deque()
        self.subscription = self.interface.subscribe_to_keyboard_events(self.keyboard, self.on_event)

    def on_event(self, event, *args):
        key = event.input.name
        if event.type == self.events.KEY_PRESS:
            if key not in self.keys and key in {"B", "R", "N", "SPACE", "BACKSPACE", "ESCAPE"}:
                self.actions.append(key)
            self.keys.add(key)
        elif event.type == self.events.KEY_RELEASE:
            self.keys.discard(key)
        return True

    def clear(self):
        self.keys.clear()
        self.actions.clear()

    def close(self):
        self.interface.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)


class Teleoperator:
    def __init__(self, collector, keyboard, device, settings, receiver=None):
        self.collector, self.keyboard, self.receiver = collector, keyboard, receiver
        self.device, self.settings = device, settings
        self.armed = False
        self.episode_index = 0
        self.lifted = False
        self.reset_pose = [x.tolist() for x in collector.initial_banana_pose]
        self.initial_center = None
        self.last_rejection = 0
        if receiver is not None:
            # Validate configured mapping before enabling any leader commands.
            leader_target(np.zeros(6), collector.limits, collector.robot_config, settings["arm_range_rad"])

    def begin(self):
        if self.receiver is not None and self.receiver.latest() is None:
            print("[teleop] Waiting for fresh leader data; press B again after publisher connects.", flush=True)
            return
        c = self.collector
        self.armed = True
        if c.writer is not None and c.writer.pending is None:
            c.writer.begin(self.episode_index)
            c.frame_index, c.last_camera_time = 0, None
            self.initial_center = np.mean(c.object_points(), axis=0)
            self.lifted = False
        print("[teleop] Active" + ("; recording" if c.writer is not None else ""), flush=True)

    def discard(self, reason):
        if self.collector.writer is not None:
            self.collector.writer.finish(success=False, reason=reason)

    def reset(self):
        self.armed = False
        self.keyboard.clear()
        self.reset_pose = self.collector.reset_episode()
        print("[teleop] Reset complete; press B to start next demonstration.", flush=True)

    def complete(self):
        c = self.collector
        if c.writer is not None and c.writer.pending is None:
            print("[teleop] Press B to begin recording first.", flush=True)
            return
        if c.writer is not None and not self.lifted:
            print("[teleop] Not saved: banana has not physically lifted during this episode.", flush=True)
            return
        self.armed = False
        self.keyboard.keys.clear()
        try:
            c.verify_placement()
        except (RuntimeError, ValueError) as exc:
            print(f"[teleop] Not saved: {exc}; press B to continue or R to discard.", flush=True)
            return
        if c.writer is not None:
            c.writer.finish(success=True, reset_pose=self.reset_pose)
            self.episode_index += 1
            print(f"[teleop] Saved successful demonstration {self.episode_index}", flush=True)
        self.reset()

    def process_action(self, key):
        if key == "ESCAPE":
            return False
        if key == "B":
            self.begin()
        elif key == "SPACE":
            self.armed = False
            self.keyboard.keys.clear()
            print("[teleop] Paused input; physics and current recording continue. B resumes.", flush=True)
        elif key == "R":
            self.discard("Operator marked attempt failed")
            self.reset()
        elif key == "BACKSPACE":
            self.discard("Operator discarded attempt")
            self.armed = False
            self.keyboard.keys.clear()
        elif key == "N":
            self.complete()
        return True

    def target(self, values):
        c = self.collector
        if not self.armed:
            return c.command
        if self.device == "so101leader":
            if values is None:
                self.armed = False
                self.keyboard.keys.clear()
                self.discard("Leader timed out during demonstration")
                print("[teleop] Leader timeout: holding last target. Fresh data + B resumes.", flush=True)
                return c.command
            desired = leader_target(values, c.limits, c.robot_config, self.settings["arm_range_rad"])
        else:
            try:
                desired = keyboard_target(self.keyboard.keys, c.command, c.model, c.robot_config,
                                          self.settings, c.fps)
            except RuntimeError as exc:
                if time.monotonic() - self.last_rejection > 1:
                    print(f"[teleop] Motion rejected: {exc}", flush=True)
                    self.last_rejection = time.monotonic()
                return c.command
        return limited_target(c.command, desired, c.limits,
                              c.robot_config["max_joint_velocity_rad_s"], c.fps)

    def tick(self):
        values = self.receiver.poll() if self.receiver is not None else None
        while self.keyboard.actions:
            if not self.process_action(self.keyboard.actions.popleft()):
                return False
        c = self.collector
        c.step_frame(self.target(values), "teleop_" + (self.device if self.armed else "hold"))
        if c.writer is not None and c.writer.pending is not None:
            self.lifted |= bool(np.mean(c.object_points(), axis=0)[2] - self.initial_center[2] >=
                                c.task["minimum_object_lift_m"])
        return True

    def run(self):
        print("[teleop] Click the Isaac viewport to focus keyboard input. " + CONTROLS, flush=True)
        print("[teleop] Keyboard: W/S X, A/D Y, Q/E Z; J/L wrist roll, K/I wrist flex; U/O gripper", flush=True)
        deadline = time.monotonic()
        try:
            while self.collector.app.is_running() and self.tick():
                # Pace fast machines; timestamps still follow fixed simulation steps.
                deadline = max(deadline + 1 / self.collector.fps, time.monotonic())
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, 0.05))
        finally:
            self.discard("Teleoperation stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="aura_lerobot/config/so101_banana.local.yaml")
    parser.add_argument("--teleop-device", choices=("keyboard", "so101leader"), default="keyboard")
    parser.add_argument("--remote-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--record", action="store_true", help="Record successful demonstrations for existing LeRobot export")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = teleop_settings(config)
    config["teleoperation"] = {**settings, "device": args.teleop_device}
    if args.teleop_device == "so101leader":
        config["teleoperation"]["remote_endpoint"] = args.remote_endpoint
    if args.check_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    if args.record:
        output = Path(config["recording"]["root"]).expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Recording output must be empty: {output}")
    if args.teleop_device == "so101leader":
        import importlib.util
        if importlib.util.find_spec("zmq") is None:
            raise RuntimeError("Install pyzmq in Isaac Python for SO101 Leader remote control")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False})
    keyboard = receiver = collector = None
    try:
        collector = Collector(app, config, record=args.record)
        keyboard = Keyboard()
        if args.teleop_device == "so101leader":
            receiver = LeaderReceiver(args.remote_endpoint, settings["remote_timeout_seconds"])
        Teleoperator(collector, keyboard, args.teleop_device, settings, receiver).run()
    except KeyboardInterrupt:
        print("Teleoperation stopped; completed demonstrations preserved.")
    finally:
        if collector is not None and collector.writer is not None:
            collector.writer.finish(success=False, reason="Teleoperation interrupted")
        if keyboard is not None:
            keyboard.close()
        if receiver is not None:
            receiver.close()
        app.close()


if __name__ == "__main__":
    main()
