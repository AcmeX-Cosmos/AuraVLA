"""
Start Isaac Sim Camera Bridge

This script is executed in Isaac Sim to start the RGBD camera bridge.
It publishes camera frames to /tmp/aura-vla-camera/ for the NVIDIA agent to read.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any


# ============================================================================
# Inline camera bridge implementation (to avoid import issues in Isaac Sim)
# ============================================================================

DEFAULT_CAMERA_PRIM_PATH = os.getenv(
    "AURA_CAMERA_PRIM_PATH",
    "/World/DACH_TRON2A/head_pitch_Link/camera",
)
# The NVIDIA agent downsamples frames before inference. Keep Isaac rendering
# at the same working resolution to avoid unnecessary render and PNG cost.
DEFAULT_CAMERA_RESOLUTION = (640, 360)
DEFAULT_CAMERA_BRIDGE_DIR = Path(
    os.getenv("AURA_CAMERA_DIR", "/tmp/aura-vla-camera")
).expanduser()


@dataclass(frozen=True)
class CameraBridgePaths:
    directory: Path
    rgb: Path
    depth: Path
    metadata: Path


def camera_bridge_paths(directory: str | Path | None = None) -> CameraBridgePaths:
    bridge_dir = Path(directory or DEFAULT_CAMERA_BRIDGE_DIR).expanduser().resolve()
    return CameraBridgePaths(
        directory=bridge_dir,
        rgb=bridge_dir / "rgb.png",
        depth=bridge_dir / "depth.png",
        metadata=bridge_dir / "metadata.json",
    )


class IsaacCameraBridge:
    """Publishes an Isaac camera to fixed image files for external agents."""

    def __init__(
        self,
        camera: Any | None = None,
        *,
        camera_prim_path: str = DEFAULT_CAMERA_PRIM_PATH,
        resolution: tuple[int, int] = DEFAULT_CAMERA_RESOLUTION,
        output_directory: str | Path | None = None,
        update_interval_sec: float = 12.0,
    ) -> None:
        if update_interval_sec <= 0.0:
            raise ValueError("update_interval_sec must be positive")
        self._camera = camera
        self.camera_prim_path = camera_prim_path
        self.resolution = resolution
        self.paths = camera_bridge_paths(output_directory)
        self.update_interval_sec = float(update_interval_sec)
        self._task: asyncio.Task[None] | None = None
        self._last_error: str | None = None
        self._sequence = 0

    def start(self) -> "IsaacCameraBridge":
        if self._task is not None and not self._task.done():
            return self
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        if self._camera is None:
            self._camera = self._initialize_camera()
        self._task = asyncio.ensure_future(self._publish_loop())
        print(
            f"Isaac RGBD bridge started: {self.camera_prim_path} -> "
            f"{self.paths.directory}"
        )
        return self

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def _initialize_camera(self) -> Any:
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            try:
                from omni.isaac.sensor import Camera
            except ImportError as exc:
                raise RuntimeError(
                    "Isaac Sim Camera sensor is not available. "
                    "Run this script inside Isaac Sim."
                ) from exc

        camera = Camera(
            prim_path=self.camera_prim_path,
            resolution=self.resolution,
        )
        camera.initialize()
        # Isaac only exposes RGB/depth through the frame annotators after they
        # have been explicitly registered. Without these calls the publish
        # loop can run indefinitely while get_rgb()/get_depth() return None.
        if hasattr(camera, "add_rgb_to_frame"):
            camera.add_rgb_to_frame()
        if hasattr(camera, "add_distance_to_image_plane_to_frame"):
            camera.add_distance_to_image_plane_to_frame()
        return camera

    async def _publish_loop(self) -> None:
        from omni.kit.app import get_app
        import cv2
        import numpy as np

        app = get_app()
        last_capture = 0.0
        while True:
            try:
                now = time.monotonic()
                if now - last_capture >= self.update_interval_sec:
                    rgb = self._read_frame("rgb", "rgba")
                    depth = self._read_frame(
                        "depth", "distance_to_image_plane", "distance"
                    )

                    if rgb is not None and depth is not None:
                        # Process RGB
                        rgb_array = np.asarray(rgb)[..., :3]
                        if rgb_array.size > 0:
                            if np.issubdtype(rgb_array.dtype, np.floating):
                                if np.nanmax(rgb_array) <= 1.0:
                                    rgb_array = rgb_array * 255.0
                            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
                            rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

                            # Process depth
                            depth_array = np.asarray(depth)
                            depth_vis = self._depth_to_colormap(depth_array)

                            self._write_png(self.paths.rgb, rgb_bgr)
                            self._write_png(self.paths.depth, depth_vis)

                            # Save metadata
                            # Keep this key aligned with the consumer in
                            # aura_perception.nvidia_agent. The old launcher
                            # used ``timestamp``, which made valid frames look
                            # permanently missing/stale to the agent.
                            self._sequence += 1
                            metadata = {
                                "captured_at_unix": time.time(),
                                "camera_prim_path": self.camera_prim_path,
                                "resolution": self.resolution,
                                "sequence": self._sequence,
                                "update_interval_sec": self.update_interval_sec,
                            }
                            self._write_json(self.paths.metadata, metadata)
                            last_capture = now
                            self._last_error = None

                await app.next_update_async()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if str(exc) != self._last_error:
                    print(f"Camera bridge error: {exc}")
                    self._last_error = str(exc)
                await app.next_update_async()

    @staticmethod
    def _write_png(path: Path, image: Any) -> None:
        import cv2

        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise RuntimeError(f"Failed to encode camera image: {path.name}")
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_bytes(buffer.tobytes())
        os.replace(temporary_path, path)

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    def _read_frame(self, *names: str) -> Any | None:
        for name in names:
            getter = getattr(self._camera, f"get_{name}", None)
            if callable(getter):
                return getter()
        current_frame = getattr(self._camera, "get_current_frame", None)
        if callable(current_frame):
            frame = current_frame() or {}
            for name in names:
                if name in frame:
                    return frame[name]
        return None

    @staticmethod
    def _depth_to_colormap(depth_array: Any) -> Any:
        import cv2
        import numpy as np

        depth_normalized = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
        if depth_normalized.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        positive_depth = depth_normalized[depth_normalized > 0]
        if positive_depth.size == 0:
            raise RuntimeError("Isaac camera depth frame contains no valid values")
        depth_min = np.percentile(positive_depth, 1)
        depth_max = np.percentile(depth_normalized, 99)
        if depth_max > depth_min:
            depth_scaled = (depth_normalized - depth_min) / (depth_max - depth_min)
        else:
            depth_scaled = depth_normalized
        depth_scaled = np.clip(depth_scaled, 0, 1)
        depth_uint8 = (depth_scaled * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)


def start_camera_bridge(
    camera_prim_path: str | None = None,
    output_directory: str | Path | None = None,
) -> IsaacCameraBridge:
    """Start the Isaac Sim camera bridge."""
    global _active_bridge
    if _active_bridge is not None:
        _active_bridge.stop()
    bridge = IsaacCameraBridge(
        camera_prim_path=camera_prim_path or DEFAULT_CAMERA_PRIM_PATH,
        output_directory=output_directory,
    )
    _active_bridge = bridge.start()
    return _active_bridge


_active_bridge: IsaacCameraBridge | None = None


# ============================================================================
# Main execution
# ============================================================================

def main() -> IsaacCameraBridge:
    camera_prim_path = os.getenv(
        "AURA_CAMERA_PRIM_PATH",
        "/World/DACH_TRON2A/head_pitch_Link/camera"
    )

    camera_bridge = start_camera_bridge(camera_prim_path=camera_prim_path)
    print(f"✓ Camera bridge started: {camera_prim_path}")
    print(f"  Output: {camera_bridge.paths.directory}")
    return camera_bridge


# This file is also sent as source to the Isaac VS Code executor. That
# executor runs it with exec(), where __name__ is not necessarily __main__, so
# a main guard would silently skip bridge startup.
camera_bridge = main()
