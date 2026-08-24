"""Pure GraspNet temporal fusion utilities.

This module deliberately has no Isaac Sim, ROS 2, Torch, or VLM dependency.
It keeps perception fusion deterministic and unit-testable outside the Isaac
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class GraspFusionError(RuntimeError):
    """Raised when RGB-D observations cannot produce a stable grasp pose."""


@dataclass(frozen=True)
class GraspObservation:
    """One calibrated GraspNet result in the robot/world coordinate frame."""

    position: np.ndarray
    orientation: np.ndarray
    score: float = 1.0
    depth_quality: float = 1.0
    geometric_validity: float = 1.0
    timestamp_sec: float | None = None


def _finite_vector(value, size: int) -> np.ndarray | None:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _unit_quaternion(value) -> np.ndarray | None:
    quaternion = _finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        return None
    return quaternion / norm


def _quality_weight(observation: GraspObservation) -> float:
    score = float(np.clip(observation.score, 0.0, 1.0))
    depth = float(np.clip(observation.depth_quality, 0.0, 1.0))
    geometry = float(np.clip(observation.geometric_validity, 0.0, 1.0))
    return max(score * depth * geometry, 1e-6)


def _weighted_quaternion_mean(quaternions: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Markley quaternion mean after putting all quaternions in one hemisphere."""
    reference = np.asarray(quaternions[0], dtype=float)
    aligned = []
    for quaternion in quaternions:
        value = np.asarray(quaternion, dtype=float)
        aligned.append(value if np.dot(reference, value) >= 0.0 else -value)
    matrix = np.zeros((4, 4), dtype=float)
    for quaternion, weight in zip(aligned, weights):
        matrix += float(weight) * np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(matrix)
    result = vectors[:, -1]
    if np.dot(result, reference) < 0.0:
        result = -result
    return result / max(float(np.linalg.norm(result)), 1e-9)


def _angular_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _normalise_observations(observations: Iterable[GraspObservation]) -> list[GraspObservation]:
    normalised = []
    for observation in observations:
        position = _finite_vector(observation.position, 3)
        orientation = _unit_quaternion(observation.orientation)
        if position is None or orientation is None:
            continue
        normalised.append(
            GraspObservation(
                position=position,
                orientation=orientation,
                score=float(observation.score),
                depth_quality=float(observation.depth_quality),
                geometric_validity=float(observation.geometric_validity),
                timestamp_sec=observation.timestamp_sec,
            )
        )
    return normalised


def fuse_grasp_observations(
    observations: Iterable[GraspObservation],
    *,
    max_position_dispersion_m: float = 0.025,
    max_orientation_dispersion_deg: float = 25.0,
    position_outlier_floor_m: float = 0.012,
    min_confidence: float = 0.10,
) -> dict:
    """Fuse stable GraspNet observations and reject temporal outliers.

    Position outliers are rejected using a coordinate-wise median/MAD gate.
    The remaining observations are weighted by model score, valid depth, and
    geometric validity. Quaternion signs are aligned before the Markley mean.
    """
    values = _normalise_observations(observations)
    if not values:
        raise GraspFusionError("no valid GraspNet observations")

    positions = np.asarray([item.position for item in values], dtype=float)
    median = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - median, axis=1)
    mad = float(np.median(distances))
    gate = max(float(position_outlier_floor_m), 3.0 * max(mad, 1e-6))
    inliers = distances <= gate
    if not np.any(inliers):
        raise GraspFusionError("all GraspNet observations rejected as temporal outliers")

    candidates = [item for item, keep in zip(values, inliers) if keep]
    candidate_distances = distances[inliers]
    weights = np.asarray([_quality_weight(item) for item in candidates], dtype=float)
    consistency_scale = max(gate, 1e-6)
    weights *= np.exp(-np.square(candidate_distances / consistency_scale))
    if not np.any(weights > 0.0):
        raise GraspFusionError("GraspNet fusion weights are invalid")
    weights /= np.sum(weights)

    fused_position = np.sum(
        np.asarray([item.position for item in candidates]) * weights[:, None],
        axis=0,
    )
    fused_orientation = _weighted_quaternion_mean(
        [item.orientation for item in candidates], weights
    )
    position_std = float(
        math.sqrt(
            np.sum(
                weights
                * np.square(
                    np.linalg.norm(
                        np.asarray([item.position for item in candidates])
                        - fused_position,
                        axis=1,
                    )
                )
            )
        )
    )
    orientation_dispersion = max(
        _angular_distance_deg(item.orientation, fused_orientation)
        for item in candidates
    )
    base_confidence = float(
        sum(weight * _quality_weight(item) for item, weight in zip(candidates, weights))
    )
    acceptance_ratio = len(candidates) / len(values)
    dispersion_confidence = math.exp(
        -position_std / max(float(max_position_dispersion_m), 1e-6)
    )
    confidence = float(np.clip(base_confidence * acceptance_ratio * dispersion_confidence, 0.0, 1.0))

    if position_std > float(max_position_dispersion_m):
        raise GraspFusionError(
            f"GraspNet position dispersion is too high: {position_std:.4f} m"
        )
    if orientation_dispersion > float(max_orientation_dispersion_deg):
        raise GraspFusionError(
            "GraspNet orientation dispersion is too high: "
            f"{orientation_dispersion:.2f} deg"
        )
    if confidence < float(min_confidence):
        raise GraspFusionError(
            f"GraspNet fused confidence is too low: {confidence:.3f}"
        )

    return {
        "position": fused_position.tolist(),
        "orientation": fused_orientation.tolist(),
        "frame_count": len(values),
        "accepted_frame_count": len(candidates),
        "rejected_frame_count": len(values) - len(candidates),
        "position_std_m": position_std,
        "orientation_dispersion_deg": float(orientation_dispersion),
        "confidence": confidence,
        "source": "graspnet_temporal_fusion",
    }
