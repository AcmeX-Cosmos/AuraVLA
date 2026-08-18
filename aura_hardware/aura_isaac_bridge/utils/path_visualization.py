"""USD debug rendering for planned 3D paths.

This module only draws supplied path data.  It does not control the robot,
change planner output, or modify any physics state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom


def _load_path_visualization_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    try:
        import yaml

        settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return ((settings.get("visualization") or {}).get("path") or {})
    except (ImportError, OSError, TypeError, ValueError):
        return {}


_CONFIG = _load_path_visualization_config()
DEFAULT_ENABLED = bool(_CONFIG.get("enabled", True))
DEFAULT_ROOT_PATH = str(_CONFIG.get("root_prim_path", "/World/S5Debug/PlannedPath"))
DEFAULT_REFRESH_INTERVAL_SEC = max(
    float(_CONFIG.get("refresh_interval_sec", 0.5)),
    0.0,
)
_VISUALIZERS: dict[tuple[str, float], "PathVisualizer"] = {}


@dataclass(frozen=True)
class PathStyle:
    point_diameter_m: float = 0.006
    line_width_m: float = 0.002
    waypoint_radius_m: float = 0.014
    point_color: tuple[float, float, float] = (0.1, 0.8, 1.0)
    line_color: tuple[float, float, float] = (0.0, 0.45, 0.9)
    waypoint_color: tuple[float, float, float] = (1.0, 0.75, 0.05)


class PathVisualizer:
    """Draw dense path samples and sparse waypoint spheres at a fixed rate."""

    def __init__(
        self,
        *,
        root_path: str = DEFAULT_ROOT_PATH,
        refresh_interval_sec: float = DEFAULT_REFRESH_INTERVAL_SEC,
        style: PathStyle | None = None,
    ) -> None:
        self.root_path = str(root_path)
        self.refresh_interval_sec = max(float(refresh_interval_sec), 0.0)
        self.style = style or PathStyle()
        self._last_render_time = float("-inf")

    @staticmethod
    def _points(points) -> np.ndarray:
        array = np.asarray(points, dtype=float)
        if array.size == 0:
            return np.empty((0, 3), dtype=float)
        array = array.reshape(-1, 3)
        if not np.all(np.isfinite(array)):
            raise ValueError("Path points must be finite XYZ values")
        return array

    @staticmethod
    def _vec3f(points: np.ndarray) -> list[Gf.Vec3f]:
        return [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points]

    def clear(self, stage=None) -> None:
        stage = stage or get_current_stage()
        if stage.GetPrimAtPath(self.root_path).IsValid():
            stage.RemovePrim(self.root_path)

    def render(self, continuous_points, waypoint_points, *, force=False, stage=None) -> dict:
        """Render blue points/polyline and yellow waypoint spheres.

        Calls faster than ``refresh_interval_sec`` keep the previous drawing.
        """
        now = time.monotonic()
        if not force and now - self._last_render_time < self.refresh_interval_sec:
            return {"rendered": False, "reason": "throttled"}

        stage = stage or get_current_stage()
        samples = self._points(continuous_points)
        waypoints = self._points(waypoint_points)
        self.clear(stage)
        UsdGeom.Xform.Define(stage, self.root_path)

        if len(samples):
            dots = UsdGeom.Points.Define(stage, f"{self.root_path}/ContinuousPoints")
            dots.CreatePointsAttr().Set(self._vec3f(samples))
            dots.CreateWidthsAttr().Set([self.style.point_diameter_m] * len(samples))
            dots.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            dots.CreateDisplayColorAttr().Set([Gf.Vec3f(*self.style.point_color)])

        if len(samples) >= 2:
            line = UsdGeom.BasisCurves.Define(stage, f"{self.root_path}/ContinuousLine")
            line.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            line.CreateCurveVertexCountsAttr().Set([len(samples)])
            line.CreatePointsAttr().Set(self._vec3f(samples))
            line.CreateWidthsAttr().Set([self.style.line_width_m])
            line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            line.CreateDisplayColorAttr().Set([Gf.Vec3f(*self.style.line_color)])

        for index, point in enumerate(waypoints):
            sphere = UsdGeom.Sphere.Define(
                stage, f"{self.root_path}/Waypoint_{index:02d}"
            )
            sphere.CreateRadiusAttr().Set(self.style.waypoint_radius_m)
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*self.style.waypoint_color)])
            UsdGeom.XformCommonAPI(sphere).SetTranslate(
                (float(point[0]), float(point[1]), float(point[2]))
            )

        self._last_render_time = now
        return {
            "rendered": True,
            "root_path": self.root_path,
            "continuous_point_count": int(len(samples)),
            "waypoint_count": int(len(waypoints)),
        }


def render_planned_path(
    continuous_points,
    waypoint_points,
    *,
    refresh_interval_sec: float = DEFAULT_REFRESH_INTERVAL_SEC,
    root_path: str = DEFAULT_ROOT_PATH,
    force: bool = False,
    stage=None,
) -> dict:
    """Convenience renderer for one-off planned-path visualization."""
    if not DEFAULT_ENABLED:
        return {"rendered": False, "reason": "disabled"}
    key = (str(root_path), max(float(refresh_interval_sec), 0.0))
    visualizer = _VISUALIZERS.get(key)
    if visualizer is None:
        visualizer = PathVisualizer(
            root_path=key[0],
            refresh_interval_sec=key[1],
        )
        _VISUALIZERS[key] = visualizer
    return visualizer.render(
        continuous_points,
        waypoint_points,
        force=force,
        stage=stage,
    )


def joint_path_to_tcp_points(controller, joint_targets) -> np.ndarray:
    """Convert planned joint samples to TCP world positions without commanding them."""
    targets = [np.asarray(target, dtype=float).reshape(-1) for target in joint_targets]
    if not targets:
        return np.empty((0, 3), dtype=float)
    frame_name = controller.kinematics.get_end_effector_frame()
    return np.asarray(
        [
            controller.lula.compute_forward_kinematics(frame_name, target)[0]
            for target in targets
        ],
        dtype=float,
    )


def render_joint_path(
    controller,
    continuous_joint_targets,
    waypoint_joint_targets,
    *,
    refresh_interval_sec: float = DEFAULT_REFRESH_INTERVAL_SEC,
    root_path: str = DEFAULT_ROOT_PATH,
    force: bool = False,
    stage=None,
) -> dict:
    """Render a planner's joint path as TCP dots/line and waypoint spheres."""
    return render_planned_path(
        joint_path_to_tcp_points(controller, continuous_joint_targets),
        joint_path_to_tcp_points(controller, waypoint_joint_targets),
        refresh_interval_sec=refresh_interval_sec,
        root_path=root_path,
        force=force,
        stage=stage,
    )
