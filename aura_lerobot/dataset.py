"""Atomic raw episodes, deliberately independent of Isaac and LeRobot imports."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .teacher import JOINT_NAMES


class EpisodeWriter:
    def __init__(self, root, config):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Refuse accidental mixing of sessions/configurations. Resume via a new raw root.
        manifest = self.root / "session.json"
        if manifest.exists() or any(self.root.iterdir()):
            raise FileExistsError(f"Raw output must be empty: {self.root}")
        self.config = config
        manifest.write_text(json.dumps({"format_version": 1, "joint_names": JOINT_NAMES,
                                       "units": "all six joints in radians", "config": config},
                                      indent=2), encoding="utf-8")
        self.pending = None

    def begin(self, index):
        self.pending = self.root / f"episode_{index:06d}.partial"
        self.pending.mkdir()
        (self.pending / "rgb").mkdir()
        self.frames = []

    def add(self, state, action, rgb, timestamp, phase):
        state, action, rgb = np.asarray(state), np.asarray(action), np.asarray(rgb)
        if (state.shape != (6,) or action.shape != (6,) or
                not np.isfinite(state).all() or not np.isfinite(action).all()):
            raise ValueError("Expected finite six-joint measured state and commanded action")
        width, height = self.config["camera"]["resolution"]
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Invalid RGB camera frame")
        Image.fromarray(rgb).save(self.pending / "rgb" / f"{len(self.frames):06d}.png")
        self.frames.append((state.copy(), action.copy(), timestamp, phase))

    def finish(self, *, success, reason="", reset_pose=None):
        if self.pending is None:
            return
        with (self.root / "attempts.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"episode": self.pending.stem, "success": success,
                                  "frames": len(self.frames), "reason": reason}) + "\n")
        if success and self.frames:
            state, action, timestamps, phases = zip(*self.frames)
            np.savez_compressed(self.pending / "frames.npz",
                                state=np.asarray(state, dtype=np.float32),
                                action=np.asarray(action, dtype=np.float32),
                                timestamp=np.asarray(timestamps, dtype=np.float64),
                                phase=np.asarray(phases))
            (self.pending / "episode.json").write_text(json.dumps(
                {"success": True, "frames": len(self.frames), "reset_pose": reset_pose}, indent=2),
                encoding="utf-8")
            self.pending.rename(self.pending.with_suffix(""))
        else:
            shutil.rmtree(self.pending)
        self.pending = None
