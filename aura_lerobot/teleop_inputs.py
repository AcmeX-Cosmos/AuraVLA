"""Keyboard targets and the official leisaac SO101 ZMQ wire format."""
from __future__ import annotations

import struct
import time

import numpy as np


def decode_leader_packet(packet):
    """leisaac publishes exactly six little-endian normalized float32 values."""
    if len(packet) != 24:
        raise ValueError("Expected one 24-byte leisaac SO101 packet")
    values = np.array(struct.unpack("<6f", packet), dtype=float)
    lower = np.array([-100] * 5 + [0])
    if not np.isfinite(values).all() or np.any(values < lower) or np.any(values > 100):
        raise ValueError("Invalid normalized leader positions")
    return values


def leader_target(values, joint_limits, robot_config, arm_range=None):
    limits = np.asarray(joint_limits, dtype=float)
    ranges = limits[:5] if arm_range is None else np.asarray(arm_range, dtype=float)
    if (ranges.shape != (5, 2) or not np.isfinite(ranges).all() or
            np.any(ranges < limits[:5, 0, None]) or np.any(ranges > limits[:5, 1, None]) or
            np.any(ranges[:, 0] == ranges[:, 1])):
        raise ValueError("arm_range_rad must map five nonzero ranges within USD joint limits")
    target = np.empty(6)
    target[:5] = ranges[:, 0] + (np.asarray(values)[:5] + 100) / 200 * (ranges[:, 1] - ranges[:, 0])
    target[5] = robot_config["gripper_closed_rad"] + values[5] / 100 * (
        robot_config["gripper_open_rad"] - robot_config["gripper_closed_rad"])
    return target


def limited_target(current, desired, limits, speed, fps):
    return np.clip(np.asarray(current) + np.clip(np.asarray(desired) - current, -speed / fps, speed / fps),
                   np.asarray(limits)[:, 0], np.asarray(limits)[:, 1])


def keyboard_target(keys, command, model, robot_config, settings, fps):
    """World-space translation via five-DOF IK, plus direct wrist/gripper control."""
    target = np.array(command, dtype=float)
    translation = np.array([int("W" in keys) - int("S" in keys),
                            int("A" in keys) - int("D" in keys),
                            int("Q" in keys) - int("E" in keys)], dtype=float)
    if np.linalg.norm(translation) > 0:
        translation /= max(1.0, np.linalg.norm(translation))
        pose = model.forward(target[:5])
        tool_axis = np.asarray(robot_config["tool_axis"], dtype=float)
        target[:5] = model.solve(pose[:3, 3] + translation * settings["translation_speed_m_s"] / fps,
                                tool_axis, pose[:3, :3] @ tool_axis, target[:5],
                                position_tolerance=0.0002, attempts=1)
    step = settings["wrist_speed_rad_s"] / fps
    target[4] += step * (int("J" in keys) - int("L" in keys))
    target[3] += step * (int("K" in keys) - int("I" in keys))
    if "U" in keys and "O" not in keys:
        target[5] = robot_config["gripper_open_rad"]
    elif "O" in keys and "U" not in keys:
        target[5] = robot_config["gripper_closed_rad"]
    return target


class LeaderReceiver:
    """Nonblocking latest-state subscriber; never wait on networking in Isaac's UI."""

    def __init__(self, endpoint, timeout):
        import zmq
        self.zmq = zmq
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(endpoint)
        self.values = None
        self.received_at = None
        self.invalid_packets = 0

    def poll(self):
        # Bound work even if a publisher misbehaves; CONFLATE drops older packets.
        for _ in range(64):
            try:
                packet = self.socket.recv(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            try:
                values = decode_leader_packet(packet)
            except ValueError:
                self.invalid_packets += 1
                continue
            self.values = values
            self.received_at = time.monotonic()
        return self.latest()

    def latest(self):
        if self.received_at is None or time.monotonic() - self.received_at > self.timeout:
            return None
        return self.values.copy()

    def close(self):
        self.socket.close()
        self.context.term()
