"""Small, simulator-independent joint-space teacher for a five-axis arm."""
from __future__ import annotations

import numpy as np


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def revolute_transform(axis, origin, angle):
    cross = skew(axis)
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = origin - rotation @ origin
    return result


class Kinematics:
    """Product of exponentials around a measured reference configuration.

    Axes, origins and TCP reference transform share world coordinates. Ordered
    axes must belong to one serial arm; the simulator adapter validates this.
    """

    def __init__(self, axes, origins, reference_q, reference_tcp, limits):
        self.axes = np.asarray(axes, dtype=float)
        self.origins = np.asarray(origins, dtype=float)
        self.reference_q = np.asarray(reference_q, dtype=float)
        self.reference_tcp = np.asarray(reference_tcp, dtype=float)
        self.limits = np.asarray(limits, dtype=float)

    def forward(self, q):
        pose = np.eye(4)
        for axis, origin, delta in zip(self.axes, self.origins, np.asarray(q) - self.reference_q):
            pose = pose @ revolute_transform(axis, origin, delta)
        return pose @ self.reference_tcp

    def solve(self, position, tool_axis, direction, seed, *, position_tolerance=0.003,
              angle_tolerance_deg=10, attempts=8):
        position = np.asarray(position, dtype=float)
        tool_axis = np.asarray(tool_axis, dtype=float)
        tool_axis /= np.linalg.norm(tool_axis)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        rng = np.random.default_rng(0)
        orientation_weight = 0.08

        def residual(q):
            pose = self.forward(q)
            return np.r_[position - pose[:3, 3],
                         orientation_weight * (direction - pose[:3, :3] @ tool_axis)]

        for attempt in range(attempts):
            q = np.clip(np.array(seed, dtype=float) if attempt == 0 else
                        np.asarray(seed) + rng.normal(0, 0.6, len(seed)),
                        self.limits[:, 0], self.limits[:, 1])
            for _ in range(160):
                error = residual(q)
                pose = self.forward(q)
                cosine = np.clip(np.dot(pose[:3, :3] @ tool_axis, direction), -1, 1)
                if (np.linalg.norm(error[:3]) <= position_tolerance and
                        np.degrees(np.arccos(cosine)) <= angle_tolerance_deg):
                    return q
                jacobian = np.column_stack([
                    (residual(q + np.eye(len(q))[j] * 1e-5) - error) / 1e-5
                    for j in range(len(q))])
                delta = -jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + np.eye(6) * 1e-4, error)
                # Backtracking prevents a limit or singularity from increasing error.
                for scale in (1.0, 0.5, 0.25, 0.1):
                    trial = np.clip(q + scale * np.clip(delta, -0.15, 0.15),
                                    self.limits[:, 0], self.limits[:, 1])
                    if np.linalg.norm(residual(trial)) < np.linalg.norm(error):
                        q = trial
                        break
                else:
                    break
        raise RuntimeError(f"IK unreachable: world target {position.tolist()}; check TCP, tool axis and workspace")


def trajectory(start, goal, fps, seconds, max_velocity):
    """Minimum-jerk absolute joint targets; length obeys the peak speed bound."""
    start, goal = np.asarray(start), np.asarray(goal)
    duration = max(seconds, float(np.max(np.abs(goal - start))) * 1.875 / max_velocity)
    count = max(2, int(np.ceil(duration * fps)))
    for t in np.linspace(0, 1, count + 1)[1:]:
        yield start + (goal - start) * (10 * t**3 - 15 * t**4 + 6 * t**5)


def inside_box(points_world, box_pose, bounds, margin=0.0):
    points = np.asarray(points_world)
    local = (points - box_pose[:3, 3]) @ box_pose[:3, :3]
    bounds = np.asarray(bounds)
    return bool(np.all(local >= bounds[0] + margin) and np.all(local <= bounds[1] - margin))
