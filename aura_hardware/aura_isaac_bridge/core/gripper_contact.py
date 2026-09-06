"""Pure gripper-contact classification shared by grasp and carry checks."""

from __future__ import annotations

import numpy as np


def classify_finger_contacts(
    feedback,
    command_targets,
    efforts,
    *,
    residual_threshold,
    force_threshold,
):
    """Return per-finger contact using position blocking or measured force."""
    feedback = np.asarray(feedback, dtype=float)
    command_targets = np.asarray(command_targets, dtype=float)
    efforts = np.asarray(efforts, dtype=float)
    if feedback.shape != command_targets.shape or feedback.shape != efforts.shape:
        raise ValueError("feedback, command targets, and efforts must have matching shapes")

    residuals = np.maximum(feedback - command_targets, 0.0)
    return (residuals >= float(residual_threshold)) | (
        np.isfinite(efforts) & (efforts >= float(force_threshold))
    )
