from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DiffusionConfig:
    cartesian_spacing_m: float = 0.04
    max_joint_step_rad: float = 0.008
    min_frames: int = 8
    dual_arm_min_tcp_separation_m: float = 0.18


@dataclass(frozen=True)
class DualArmTrajectory:
    left_positions: np.ndarray
    right_positions: np.ndarray
    frame_count: int


def minimum_jerk(progress):
    progress = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


class SparseKeyposeDiffuser:
    """Diffuses sparse Cartesian or joint keyposes into smooth dense paths."""

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        self.config = config or DiffusionConfig()

    def diffuse_cartesian_keyposes(
        self,
        keyposes: Iterable[Iterable[float]],
        *,
        max_spacing: float | None = None,
    ) -> np.ndarray:
        poses = self._as_path(keyposes)
        if poses.shape[0] <= 1:
            return poses.copy()
        spacing = float(max_spacing or self.config.cartesian_spacing_m)
        if spacing <= 0.0:
            raise ValueError("Cartesian keypose spacing must be positive")

        dense = [poses[0].copy()]
        for start, end in zip(poses[:-1], poses[1:]):
            distance = float(np.linalg.norm(end - start))
            segment_count = max(1, int(np.ceil(distance / spacing)))
            dense.extend(
                start + (index / segment_count) * (end - start)
                for index in range(1, segment_count + 1)
            )
        return np.asarray(dense, dtype=float)

    def required_joint_frames(
        self,
        start_position,
        keyposes: Iterable[Iterable[float]],
    ) -> int:
        path = self._joint_path(start_position, keyposes)
        total_length = self._polyline_length(path)
        if total_length <= 1e-9:
            return 1
        return max(
            int(self.config.min_frames),
            int(
                np.ceil(
                    1.875
                    * total_length
                    / float(self.config.max_joint_step_rad)
                )
            ),
        )

    def diffuse_joint_keyposes(
        self,
        start_position,
        keyposes: Iterable[Iterable[float]],
        *,
        frame_count: int | None = None,
    ) -> np.ndarray:
        path = self._joint_path(start_position, keyposes)
        frames = (
            self.required_joint_frames(start_position, path[1:])
            if frame_count is None
            else max(int(frame_count), 1)
        )
        return self._sample_polyline(path, frames)

    def synchronize_dual_arm_paths(
        self,
        left_start,
        left_keyposes: Iterable[Iterable[float]],
        right_start,
        right_keyposes: Iterable[Iterable[float]],
    ) -> DualArmTrajectory:
        left_keyposes = list(left_keyposes)
        right_keyposes = list(right_keyposes)
        frame_count = max(
            self.required_joint_frames(left_start, left_keyposes),
            self.required_joint_frames(right_start, right_keyposes),
        )
        return DualArmTrajectory(
            left_positions=self.diffuse_joint_keyposes(
                left_start,
                left_keyposes,
                frame_count=frame_count,
            ),
            right_positions=self.diffuse_joint_keyposes(
                right_start,
                right_keyposes,
                frame_count=frame_count,
            ),
            frame_count=frame_count,
        )

    def validate_tcp_clearance(
        self,
        left_tcp_positions,
        right_tcp_positions,
        *,
        minimum_separation: float | None = None,
    ) -> float:
        left = self._as_path(left_tcp_positions)
        right = self._as_path(right_tcp_positions)
        if left.shape != right.shape or left.shape[1] != 3:
            raise ValueError(
                "Left and right TCP paths must have matching (N, 3) shapes"
            )
        distances = np.linalg.norm(left - right, axis=1)
        minimum_distance = float(np.min(distances))
        threshold = float(
            minimum_separation
            if minimum_separation is not None
            else self.config.dual_arm_min_tcp_separation_m
        )
        if minimum_distance < threshold:
            raise RuntimeError(
                "dual-arm TCP clearance violated: "
                f"minimum={minimum_distance:.3f} m, required={threshold:.3f} m"
            )
        return minimum_distance

    @staticmethod
    def _as_path(values) -> np.ndarray:
        path = np.asarray(list(values), dtype=float)
        if path.size == 0:
            return np.empty((0, 0), dtype=float)
        if path.ndim != 2:
            raise ValueError(f"Keyposes must be a two-dimensional array, got {path.shape}")
        if not np.all(np.isfinite(path)):
            raise ValueError("Keyposes contain non-finite values")
        return path

    def _joint_path(self, start_position, keyposes) -> np.ndarray:
        start = np.asarray(start_position, dtype=float).reshape(-1)
        targets = self._as_path(keyposes)
        if targets.size == 0:
            return start.reshape(1, -1)
        if targets.shape[1] != start.size:
            raise ValueError(
                f"Joint keyposes have {targets.shape[1]} columns; expected {start.size}"
            )
        return np.vstack((start, targets))

    @staticmethod
    def _polyline_length(path: np.ndarray) -> float:
        if path.shape[0] <= 1:
            return 0.0
        return float(np.sum(np.max(np.abs(np.diff(path, axis=0)), axis=1)))

    @staticmethod
    def _sample_polyline(path: np.ndarray, frame_count: int) -> np.ndarray:
        if path.shape[0] == 1:
            return np.repeat(path, frame_count, axis=0)
        segment_deltas = np.diff(path, axis=0)
        segment_lengths = np.max(np.abs(segment_deltas), axis=1)
        total_length = float(np.sum(segment_lengths))
        if total_length <= 1e-9:
            return np.repeat(path[-1:], frame_count, axis=0)

        cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        progress = minimum_jerk(np.arange(1, frame_count + 1) / frame_count)
        distances = progress * total_length
        samples = []
        for distance in distances:
            segment_index = min(
                int(
                    np.searchsorted(
                        cumulative_lengths[1:],
                        distance,
                        side="right",
                    )
                ),
                len(segment_lengths) - 1,
            )
            segment_length = float(segment_lengths[segment_index])
            local_progress = (
                1.0
                if segment_length <= 1e-9
                else np.clip(
                    (distance - cumulative_lengths[segment_index]) / segment_length,
                    0.0,
                    1.0,
                )
            )
            samples.append(
                path[segment_index]
                + local_progress * segment_deltas[segment_index]
            )
        samples[-1] = path[-1].copy()
        return np.asarray(samples, dtype=float)


def shift_grasp_toward_base(
    position,
    base_xy,
    offset_m: float,
    bbox_min,
    bbox_max,
    *,
    edge_margin_m: float = 0.003,
) -> np.ndarray:
    shifted = np.asarray(position, dtype=float).copy()
    base = np.asarray(base_xy, dtype=float).reshape(2)
    direction = base - shifted[:2]
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9 or offset_m <= 0.0:
        return shifted

    shifted[:2] += direction / distance * float(offset_m)
    lower = np.asarray(bbox_min, dtype=float)[:2] + float(edge_margin_m)
    upper = np.asarray(bbox_max, dtype=float)[:2] - float(edge_margin_m)
    shifted[:2] = np.clip(shifted[:2], lower, upper)
    return shifted
