from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any


DEFAULT_CAMERA_PRIM_PATH = os.getenv(
    "AURA_CAMERA_PRIM_PATH",
    "/World/DACH_TRON2A/head_pitch_Link/camera",
)
DEFAULT_CAMERA_RESOLUTION = (1280, 720)
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
        update_every_frames: int = 15,
    ) -> None:
        if update_every_frames < 1:
            raise ValueError("update_every_frames must be positive")
        self._camera = camera
        self.camera_prim_path = camera_prim_path
        self.resolution = resolution
        self.paths = camera_bridge_paths(output_directory)
        self.update_every_frames = update_every_frames
        self._task: asyncio.Task[None] | None = None
        self._last_error: str | None = None

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

    def capture_once(self) -> CameraBridgePaths:
        if self._camera is None:
            self._camera = self._initialize_camera()

        import cv2
        import numpy as np

        rgb = self._camera.get_rgb()
        depth = self._camera.get_depth()
        if rgb is None or depth is None:
            raise RuntimeError("Isaac camera has not produced RGBD data yet")

        rgb_array = np.asarray(rgb)[..., :3]
        if rgb_array.size == 0:
            raise RuntimeError("Isaac camera returned an empty RGB frame")
        if np.issubdtype(rgb_array.dtype, np.floating):
            if np.nanmax(rgb_array) <= 1.0:
                rgb_array = rgb_array * 255.0
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        depth_array = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_array.ndim != 2:
            raise RuntimeError(
                f"Isaac camera returned invalid depth shape: {depth_array.shape}"
            )
        finite = np.isfinite(depth_array) & (depth_array > 0)
        if not finite.any():
            raise RuntimeError("Isaac camera depth frame contains no valid values")
        depth_visualization = np.zeros(depth_array.shape, dtype=np.uint8)
        near, far = np.percentile(depth_array[finite], [2.0, 98.0])
        if far <= near:
            far = near + 1.0
        clipped = np.clip(depth_array, near, far)
        depth_visualization[finite] = (
            (1.0 - (clipped[finite] - near) / (far - near)) * 255.0
        ).astype(np.uint8)

        self._write_png(self.paths.rgb, rgb_bgr)
        self._write_png(self.paths.depth, depth_visualization)
        self._write_metadata(
            {
                "camera_prim_path": self.camera_prim_path,
                "captured_at_unix": time.time(),
                "rgb_path": str(self.paths.rgb),
                "depth_path": str(self.paths.depth),
                "rgb_shape": list(rgb_array.shape),
                "depth_shape": list(depth_array.shape),
                "depth_near": float(near),
                "depth_far": float(far),
            }
        )
        return self.paths

    def _initialize_camera(self) -> Any:
        try:
            from isaacsim.core.utils.stage import get_current_stage
            from isaacsim.sensors.camera import Camera
        except ImportError:
            try:
                from omni.isaac.core.utils.stage import get_current_stage
                from omni.isaac.sensor import Camera
            except ImportError as exc:
                raise RuntimeError(
                    "IsaacCameraBridge must be started inside Isaac Sim"
                ) from exc

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.camera_prim_path)
        if not prim.IsValid():
            print(f"⚠️ 相机 {self.camera_prim_path} 不存在，自动创建...")
            from pxr import UsdGeom
            camera_path = str(self.camera_prim_path)
            UsdGeom.Camera.Define(stage, camera_path)
            from omni.kit.app import get_app as _get_app; _get_app().update()
            prim = stage.GetPrimAtPath(camera_path)
            if not prim.IsValid():
                raise RuntimeError(
                    f"Failed to auto-create camera: {self.camera_prim_path}"
                )
            print(f"✅ 相机已创建: {camera_path}")
        camera = Camera(
            prim_path=self.camera_prim_path,
            resolution=self.resolution,
        )
        camera.initialize()
        camera.add_rgb_to_frame()
        camera.add_distance_to_image_plane_to_frame()
        return camera

    async def _publish_loop(self) -> None:
        from omni.kit.app import get_app

        app = get_app()
        while True:
            try:
                for _ in range(self.update_every_frames):
                    await app.next_update_async()
                self.capture_once()
                if self._last_error is not None:
                    print("Isaac RGBD bridge recovered")
                    self._last_error = None
            except asyncio.CancelledError:
                return
            except Exception as exc:
                message = str(exc)
                if message != self._last_error:
                    print(f"Isaac RGBD bridge waiting for a valid frame: {message}")
                    self._last_error = message
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

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        temporary_path = self.paths.metadata.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, self.paths.metadata)


_active_bridge: IsaacCameraBridge | None = None


def start_camera_bridge(
    camera: Any | None = None,
    **kwargs: Any,
) -> IsaacCameraBridge:
    global _active_bridge
    if _active_bridge is not None:
        _active_bridge.stop()
    _active_bridge = IsaacCameraBridge(camera, **kwargs).start()
    return _active_bridge


def stop_camera_bridge() -> None:
    global _active_bridge
    if _active_bridge is not None:
        _active_bridge.stop()
        _active_bridge = None
