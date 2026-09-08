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


def evaluate_grasp_stability(
    finger_contacts,
    efforts,
    residuals,
    *,
    force_threshold,
    residual_threshold,
    minimum_balance_ratio=0.25,
):
    """Validate bilateral grasp evidence before lift or transport.

    Per-finger contact classification is intentionally permissive because a
    position residual and a force reading are complementary sensors.  A grasp
    is only stable when both fingers have evidence and the active evidence is
    not concentrated on one finger.  Residual-only contact is valid when both
    residuals are present and reasonably balanced.
    """
    finger_contacts = np.asarray(finger_contacts, dtype=bool)
    efforts = np.asarray(efforts, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if (
        finger_contacts.ndim != 1
        or efforts.shape != finger_contacts.shape
        or residuals.shape != finger_contacts.shape
    ):
        raise ValueError(
            "finger contacts, efforts, and residuals must have matching 1-D shapes"
        )

    bilateral_contact = bool(finger_contacts.size >= 2 and np.all(finger_contacts))
    finite_efforts = np.isfinite(efforts)
    force_evidence = finite_efforts & (efforts >= float(force_threshold))
    residual_evidence = residuals >= float(residual_threshold)

    def balance_ratio(values, valid):
        active_values = np.asarray(values, dtype=float)[valid]
        active_values = active_values[np.isfinite(active_values)]
        if active_values.size < 2:
            return None
        maximum = float(np.max(np.abs(active_values)))
        minimum = float(np.min(np.abs(active_values)))
        if maximum <= 1e-9:
            return 1.0
        return minimum / maximum

    force_ratio = balance_ratio(efforts, finite_efforts)
    residual_ratio = balance_ratio(residuals, np.ones_like(residuals, dtype=bool))
    force_balanced = True
    if np.any(force_evidence):
        force_balanced = bool(
            np.all(force_evidence)
            and force_ratio is not None
            and force_ratio >= float(minimum_balance_ratio)
        )
    residual_balanced = bool(
        np.all(residual_evidence)
        and residual_ratio is not None
        and residual_ratio >= float(minimum_balance_ratio)
    )

    # If force is available on only one side, do not treat that spike as a
    # grasp.  Residual-only confirmation remains valid for position-controlled
    # grippers, but it must still be bilateral and balanced.
    force_available = bool(np.any(force_evidence))
    stable = bool(
        bilateral_contact
        and (
            (force_available and force_balanced)
            or (not force_available and residual_balanced)
        )
    )
    return {
        "stable": stable,
        "bilateral_contact": bilateral_contact,
        "force_balanced": force_balanced,
        "residual_balanced": residual_balanced,
        "force_evidence": force_evidence,
        "residual_evidence": residual_evidence,
        "force_ratio": force_ratio,
        "residual_ratio": residual_ratio,
        "minimum_balance_ratio": float(minimum_balance_ratio),
    }
