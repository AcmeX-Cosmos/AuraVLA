"""AuraVLA 感知模块：BBox、PCA、网格质心、SAM、AnyGrasp 与相机。"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import math
import os
import site
import struct
import sys
import time
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.rotations import (
    euler_angles_to_quat,
    quat_to_euler_angles,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from aura_isaac_bridge.core.state import state
from aura_isaac_bridge.core.state import (
    PROJECT_ROOT, ISAAC_SIM_ROOT, ISAAC_SITE_PACKAGES, STUDY_DIR, AURA_DIR, SECTION3_DIR,
    DEFAULT_PROJECT_ROOT, DEFAULT_ISAAC_SIM_ROOT, DEFAULT_ISAAC_SITE_PACKAGES,
    ANYGRASP_DIR, ANYGRASP_CHECKPOINT_PATH, SAM_MODEL_PATH,
    DEFAULT_ANYGRASP_DIR, DEFAULT_SAM_MODEL_PATH,
    ANYGRASP_CALIBRATION_ENABLED, ANYGRASP_CAMERA_OFFSET,
    ANYGRASP_CALIBRATION_MAX_CORRECTION,
    CAMERA_PRIM_PATH, CAMERA_RESOLUTION, CAMERA_PREVIEW_RESOLUTION,
    SHOW_GRASP_DEBUG, USE_ANYGRASP, USE_ANYGRASP_ORIENTATION,
    GRASP_POSITION_OFFSET, GRASP_INSERT_DEPTH,
    ANYGRASP_FUSION_FRAME_COUNT, ANYGRASP_FUSION_FRAME_INTERVAL_SEC,
    ANYGRASP_FUSION_MAX_POSITION_DISPERSION_M,
    ANYGRASP_FUSION_MAX_ORIENTATION_DISPERSION_DEG,
    ANYGRASP_FUSION_POSITION_OUTLIER_FLOOR_M, ANYGRASP_FUSION_MIN_CONFIDENCE,
)
from aura_isaac_bridge.core.physics import step_app
from aura_isaac_bridge.core.grasp_fusion import (
    GraspObservation,
    fuse_grasp_observations,
)

def get_bbox_center(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bbox_min = np.array(aligned_box.GetMin(), dtype=float)
    bbox_max = np.array(aligned_box.GetMax(), dtype=float)
    center = (bbox_min + bbox_max) * 0.5
    return center, bbox_min, bbox_max


def get_mesh_horizontal_principal_axes(stage, prim_path):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"无法计算物体主轴，Prim 不存在: {prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_point_clouds = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        local_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not local_points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        world_point_clouds.append(
            np.asarray(
                [
                    local_to_world.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                    for point in local_points
                ],
                dtype=float,
            )
        )
    if not world_point_clouds:
        raise RuntimeError(f"无法计算物体主轴，Prim 下没有 Mesh: {prim_path}")

    world_points = np.concatenate(world_point_clouds, axis=0)
    centered_xy = world_points[:, :2] - np.mean(world_points[:, :2], axis=0)
    covariance = np.cov(centered_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    long_axis /= np.linalg.norm(long_axis)
    if long_axis[int(np.argmax(np.abs(long_axis)))] < 0.0:
        long_axis = -long_axis
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=float)
    return long_axis, short_axis


def get_mesh_horizontal_min_width_axis(stage, prim_path, samples=720):
    """Return the horizontal axis with the smallest mesh projection."""
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"无法计算物体最窄方向，Prim 不存在: {prim_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_point_clouds = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        local_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not local_points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        world_point_clouds.append(
            np.asarray(
                [
                    local_to_world.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                    for point in local_points
                ],
                dtype=float,
            )[:, :2]
        )
    if not world_point_clouds:
        raise RuntimeError(f"无法计算物体最窄方向，Prim 下没有 Mesh: {prim_path}")
    points = np.concatenate(world_point_clouds, axis=0)
    sample_count = max(int(samples), 36)
    angles = np.arange(sample_count, dtype=float) * (np.pi / sample_count)
    axes = np.column_stack((np.cos(angles), np.sin(angles)))
    widths = np.ptp(points @ axes.T, axis=0)
    axis = np.asarray(axes[int(np.argmin(widths))], dtype=float)
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return axis


def get_mesh_extent_along_axis(stage, prim_path, axis):
    """Return the visible mesh extent projected onto a world-space axis."""
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"无法计算物体投影宽度，Prim 不存在: {prim_path}")
    normalized_axis = np.asarray(axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(normalized_axis))
    if axis_norm < 1e-9:
        raise RuntimeError("无法计算物体投影宽度，投影轴长度为零")
    normalized_axis /= axis_norm
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    projections = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        local_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not local_points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        world_points = np.asarray(
            [
                local_to_world.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                for point in local_points
            ],
            dtype=float,
        )
        projections.append(world_points @ normalized_axis)
    if not projections:
        raise RuntimeError(f"无法计算物体投影宽度，Prim 下没有 Mesh: {prim_path}")
    projected = np.concatenate(projections)
    minimum = float(np.min(projected))
    maximum = float(np.max(projected))
    return minimum, maximum, maximum - minimum


def get_mesh_horizontal_cross_section_center(
    stage,
    prim_path,
    source_xy,
    half_width=0.018,
):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"无法计算局部截面，Prim 不存在: {prim_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_point_clouds = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        local_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not local_points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        world_point_clouds.append(
            np.asarray(
                [
                    local_to_world.Transform(
                        Gf.Vec3d(
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                        )
                    )
                    for point in local_points
                ],
                dtype=float,
            )
        )
    if not world_point_clouds:
        raise RuntimeError(f"无法计算局部截面，Prim 下没有 Mesh: {prim_path}")

    world_xy = np.concatenate(world_point_clouds, axis=0)[:, :2]
    long_axis, short_axis = get_mesh_horizontal_principal_axes(stage, prim_path)
    source_xy = np.asarray(source_xy, dtype=float)
    source_long = float(np.dot(source_xy, long_axis))
    long_coordinates = world_xy @ long_axis
    section_mask = np.abs(long_coordinates - source_long) <= float(half_width)
    if int(np.count_nonzero(section_mask)) < 8:
        nearest_indices = np.argsort(np.abs(long_coordinates - source_long))[:8]
        section_xy = world_xy[nearest_indices]
    else:
        section_xy = world_xy[section_mask]
    short_coordinates = section_xy @ short_axis
    section_short_center = float(
        (np.min(short_coordinates) + np.max(short_coordinates)) * 0.5
    )
    return (
        long_axis * source_long
        + short_axis * section_short_center
    )


def get_mesh_center(stage, prim_path):
    """计算物体网格质心（所有 mesh 顶点世界坐标的均值）。"""
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"无法计算网格质心，Prim 不存在: {prim_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_point_clouds = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        local_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not local_points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        world_point_clouds.append(
            np.asarray(
                [
                    local_to_world.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                    for point in local_points
                ],
                dtype=float,
            )
        )
    if not world_point_clouds:
        raise RuntimeError(f"无法计算网格质心，Prim 下没有 Mesh: {prim_path}")
    world_points = np.concatenate(world_point_clouds, axis=0)
    return np.asarray(np.mean(world_points, axis=0), dtype=float)


def quat_rotate(quat_wxyz, vector):
    q_vec = np.array(quat_wxyz[1:4], dtype=float)
    vector = np.array(vector, dtype=float)
    uv = np.cross(q_vec, vector)
    uuv = np.cross(q_vec, uv)
    return vector + 2.0 * (quat_wxyz[0] * uv + uuv)

def get_sim_pose(single_prim):
    try:
        positions, orientations = single_prim._prim_view.get_world_poses(usd=False)
        return np.array(positions[0], dtype=float), np.array(orientations[0], dtype=float), "fabric"
    except Exception:
        position, orientation = single_prim.get_world_pose()
        return np.array(position, dtype=float), np.array(orientation, dtype=float), "usd"


def _get_usd_to_sim_geometry_transform(target_prim):
    """Return the rigid transform from authored USD geometry to live Fabric."""
    sim_positions, sim_orientations = target_prim._prim_view.get_world_poses(
        usd=False
    )
    usd_positions, usd_orientations = target_prim._prim_view.get_world_poses(
        usd=True
    )
    sim_position = np.asarray(sim_positions[0], dtype=float)
    usd_position = np.asarray(usd_positions[0], dtype=float)
    sim_rotation = quat_to_rot_matrix(
        np.asarray(sim_orientations[0], dtype=float)
    )
    usd_rotation = quat_to_rot_matrix(
        np.asarray(usd_orientations[0], dtype=float)
    )
    return sim_position, usd_position, sim_rotation, usd_rotation


def transform_usd_world_points_to_sim(target_prim, points):
    points = np.asarray(points, dtype=float)
    original_shape = points.shape
    points = points.reshape(-1, 3)
    sim_position, usd_position, sim_rotation, usd_rotation = (
        _get_usd_to_sim_geometry_transform(target_prim)
    )
    local_points = (points - usd_position) @ usd_rotation
    transformed = local_points @ sim_rotation.T + sim_position
    return transformed.reshape(original_shape)


def transform_sim_world_points_to_usd(target_prim, points):
    points = np.asarray(points, dtype=float)
    original_shape = points.shape
    points = points.reshape(-1, 3)
    sim_position, usd_position, sim_rotation, usd_rotation = (
        _get_usd_to_sim_geometry_transform(target_prim)
    )
    local_points = (points - sim_position) @ sim_rotation
    transformed = local_points @ usd_rotation.T + usd_position
    return transformed.reshape(original_shape)


def get_current_bbox_center(stage, target_prim, prim_path=None):
    prim_path = str(prim_path or target_prim.prim_path)
    cache = getattr(state, "_local_bbox_corner_cache", None)
    if cache is None:
        cache = {}
        state._local_bbox_corner_cache = cache
    local_corners = cache.get(prim_path)
    if local_corners is None:
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            raise RuntimeError(f"无法计算局部包围盒，Prim 不存在: {prim_path}")
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        root_to_world = xform_cache.GetLocalToWorldTransform(root)
        world_to_root = root_to_world.GetInverse()
        local_point_clouds = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            if not points:
                continue
            mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
            local_point_clouds.append(
                np.asarray(
                    [
                        world_to_root.Transform(
                            mesh_to_world.Transform(
                                Gf.Vec3d(
                                    float(point[0]),
                                    float(point[1]),
                                    float(point[2]),
                                )
                            )
                        )
                        for point in points
                    ],
                    dtype=float,
                )
            )
        if not local_point_clouds:
            raise RuntimeError(
                f"无法计算局部包围盒，Prim 下没有 Mesh: {prim_path}"
            )
        local_points = np.concatenate(local_point_clouds, axis=0)
        local_min = np.min(local_points, axis=0)
        local_max = np.max(local_points, axis=0)
        local_corners = np.asarray(
            [
                [x, y, z]
                for x in (local_min[0], local_max[0])
                for y in (local_min[1], local_max[1])
                for z in (local_min[2], local_max[2])
            ],
            dtype=float,
        )
        cache[prim_path] = local_corners

    sim_position, sim_orientation, _ = get_sim_pose(target_prim)
    sim_rotation = quat_to_rot_matrix(sim_orientation)
    sim_corners = local_corners @ sim_rotation.T + sim_position
    sim_min = np.min(sim_corners, axis=0)
    sim_max = np.max(sim_corners, axis=0)
    return (sim_min + sim_max) * 0.5, sim_min, sim_max

def get_current_mesh_horizontal_cross_section_center(
    stage,
    target_prim,
    source_xy,
    prim_path=None,
    half_width=0.018,
):
    prim_path = str(prim_path or target_prim.prim_path)
    sim_position, _, _, _ = _get_usd_to_sim_geometry_transform(target_prim)
    sim_source = np.asarray(
        [source_xy[0], source_xy[1], sim_position[2]], dtype=float
    )
    usd_source = transform_sim_world_points_to_usd(target_prim, sim_source)
    usd_center_xy = get_mesh_horizontal_cross_section_center(
        stage,
        prim_path,
        usd_source[:2],
        half_width=half_width,
    )
    usd_center = np.asarray(
        [usd_center_xy[0], usd_center_xy[1], usd_source[2]], dtype=float
    )
    return transform_usd_world_points_to_sim(target_prim, usd_center)[:2]

def show_red_grasp_point(stage, position, radius=0.015):
    marker_path = "/World/debug_grasp_point"
    if stage.GetPrimAtPath(marker_path).IsValid():
        delete_prim(marker_path)
    if not SHOW_GRASP_DEBUG:
        return
    marker = UsdGeom.Sphere.Define(stage, marker_path)
    marker.GetRadiusAttr().Set(radius)
    marker.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
    UsdGeom.XformCommonAPI(marker).SetTranslate(tuple(position.tolist()))

def get_transform(translation, rotation_matrix):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform

def initialize_grasp_camera(stage):
    if not stage.GetPrimAtPath(CAMERA_PRIM_PATH).IsValid():
        raise RuntimeError(
            f"未找到相机 {CAMERA_PRIM_PATH}。请先在 USD 场景中创建并调整为正对目标物体。"
        )

    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera

    camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=CAMERA_RESOLUTION)
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera.add_rgb_to_frame()
    print(f"✅ AnyGrasp 相机初始化完成: {CAMERA_PRIM_PATH}, resolution={CAMERA_RESOLUTION}")
    print("📷 相机内参:\n", camera.get_intrinsics_matrix())
    return camera

def capture_camera_data(camera, max_attempts=60):
    for _ in range(max_attempts):
        step_app()
        rgb_data = camera.get_rgb()
        depth_data = camera.get_depth()
        if rgb_data is None or depth_data is None:
            continue
        rgb_data = np.asarray(rgb_data)
        depth_data = np.squeeze(np.asarray(depth_data, dtype=np.float32))
        if rgb_data.size > 0 and depth_data.size > 0 and np.any(np.isfinite(depth_data) & (depth_data > 0)):
            rgb_data = rgb_data[..., :3]
            if np.issubdtype(rgb_data.dtype, np.floating) and np.nanmax(rgb_data) <= 1.0:
                rgb_data = rgb_data * 255.0
            rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
            depth_mm = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(depth_mm * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            return rgb_data, depth_mm
    raise RuntimeError("相机未返回有效 RGB/Depth 数据，请确认相机可见、渲染已启用且场景中存在目标物体。")

def show_camera_preview(rgb_data, prompt_points=None, prompt_labels=None):
    import omni.ui as ui

    preview_image = np.asarray(rgb_data, dtype=np.uint8).copy()
    if prompt_points is not None:
        labels = prompt_labels if prompt_labels is not None else np.ones(len(prompt_points), dtype=np.int32)
        for point, label in zip(prompt_points, labels):
            color = (0, 255, 0) if int(label) == 1 else (255, 0, 0)
            cv2.circle(
                preview_image,
                (int(round(point[0])), int(round(point[1]))),
                10,
                color,
                3,
            )

    preview_width, preview_height = CAMERA_PREVIEW_RESOLUTION
    preview_image = cv2.resize(
        preview_image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    alpha_channel = np.full((preview_height, preview_width, 1), 255, dtype=np.uint8)
    preview_rgba = np.concatenate([preview_image, alpha_channel], axis=2)

    state._camera_preview_provider = ui.ByteImageProvider(
        preview_rgba.reshape(-1).tolist(),
        [preview_width, preview_height],
    )
    state._camera_preview_window = ui.Window(
        "AnyGrasp Camera RGB",
        width=preview_width + 30,
        height=preview_height + 55,
    )

    with state._camera_preview_window.frame:
        with ui.VStack(spacing=6):
            ui.ImageWithProvider(
                state._camera_preview_provider,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
            )
            ui.Label("绿色圆圈为自动 SAM 目标提示点", height=20)

    state._camera_preview_window.visible = True
    step_app(2)
    print("📷 相机预览已打开，继续执行 SAM、AnyGrasp 和机械臂夹取。")

def get_current_object_center(stage, target_prim, prim_path=None):
    return get_current_bbox_center(stage, target_prim, prim_path)[0]

def create_sam_prompt_points(stage, camera, target_prim, rgb_data, prim_path=None):
    image_height, image_width = rgb_data.shape[:2]
    # A calibration run can infer several objects without executing a task, so
    # the global last-task path may refer to a different object.  The Prim
    # passed to this inference call is the only valid source for the SAM hint.
    object_center = get_current_object_center(
        stage,
        target_prim,
        prim_path or target_prim.prim_path,
    )
    try:
        image_coordinates = camera.get_image_coords_from_world_points(
            np.asarray([object_center], dtype=np.float32)
        )
        if hasattr(image_coordinates, "cpu"):
            image_coordinates = image_coordinates.cpu().numpy()
        image_coordinates = np.asarray(image_coordinates, dtype=np.float32).reshape(-1, 2)
        prompt_point = image_coordinates[0]
        if not np.all(np.isfinite(prompt_point)):
            raise ValueError("投影结果包含 NaN/Inf")
        if not (0 <= prompt_point[0] < image_width and 0 <= prompt_point[1] < image_height):
            raise ValueError(f"投影点位于图像外: {prompt_point}")
        print(f"🎯 目标物体世界中心: {object_center}")
        print(f"🎯 自动 SAM 像素提示点: {prompt_point}")
    except Exception as exc:
        prompt_point = np.array([image_width * 0.5, image_height * 0.5], dtype=np.float32)
        print(f"⚠️ 目标中心投影失败，退回图像中心 {prompt_point}: {exc}")
    return prompt_point.reshape(1, 2), np.ones(1, dtype=np.int32)

def segment_target_with_sam(stage, camera, target_prim, rgb_data, prim_path=None):
    from ultralytics import SAM

    prompt_points, prompt_labels = create_sam_prompt_points(
        stage,
        camera,
        target_prim,
        rgb_data,
        prim_path=prim_path or target_prim.prim_path,
    )
    if SHOW_GRASP_DEBUG:
        show_camera_preview(rgb_data, prompt_points, prompt_labels)
    if state._sam_model is None:
        print(f"🧠 加载 SAM 模型: {SAM_MODEL_PATH}")
        state._sam_model = SAM(SAM_MODEL_PATH)
    sam_model = state._sam_model
    results = sam_model(
        rgb_data,
        points=prompt_points.copy(),
        labels=prompt_labels.copy(),
        verbose=False,
    )
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        raise RuntimeError("SAM 未生成有效目标掩码，请重新选择更靠近物体中心的提示点。")

    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
    if mask.shape != rgb_data.shape[:2]:
        mask = cv2.resize(mask, (rgb_data.shape[1], rgb_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not np.any(mask):
        raise RuntimeError("SAM 掩码为空，无法执行 AnyGrasp 推理。")
    return mask

def ensure_anygrasp_python_dependencies():
    """Load the licensed AnyGrasp SDK from the isolated project directory."""
    site_packages_path = str(ISAAC_SITE_PACKAGES)
    if not ISAAC_SITE_PACKAGES.is_dir():
        raise RuntimeError(
            f"未找到 Isaac Python site-packages: {ISAAC_SITE_PACKAGES}。"
            "请通过 ISAAC_PYTHON_SITE_PACKAGES 指定实际目录。"
        )
    grasp_detection_dir = ANYGRASP_DIR / "grasp_detection"
    anygrasp_root = ANYGRASP_DIR.parent
    if not grasp_detection_dir.is_dir():
        raise RuntimeError(
            f"未找到 AnyGrasp SDK: {grasp_detection_dir}。"
            "请将官方 AnyGrasp SDK 放入项目 AnyGrasp/sdk，或设置 ANYGRASP_DIR。"
        )
    sdk_paths = (
        ANYGRASP_DIR,
        grasp_detection_dir,
        grasp_detection_dir / "gsnet_versions",
        ANYGRASP_DIR / "pointnet2",
        anygrasp_root / "dependencies" / "python",
    )
    private_python_dir = anygrasp_root / "dependencies" / "python"
    sdk_paths += tuple(private_python_dir.glob("pointnet2-*.egg"))
    for sdk_path in sdk_paths:
        path = str(sdk_path)
        if sdk_path.is_dir() and path not in sys.path:
            sys.path.insert(0, path)
            sys.path_importer_cache.pop(path, None)
    sys.path[:] = [path for path in sys.path if path != site_packages_path]
    site.addsitedir(site_packages_path)
    sys.path.remove(site_packages_path)
    sys.path.insert(0, site_packages_path)
    sys.path_importer_cache.pop(site_packages_path, None)
    importlib.invalidate_caches()
    print(f"✅ 已刷新 Isaac Python 依赖目录: {site_packages_path}")

    # The licensed SDK binary bundles legacy Python code that still references
    # NumPy aliases removed in NumPy 1.24. Keep this compatibility shim local
    # to the AnyGrasp adapter instead of changing Isaac's global environment.
    if "float" not in np.__dict__:
        np.float = float
    if "int" not in np.__dict__:
        np.int = int
    if "bool" not in np.__dict__:
        np.bool = bool

    # AnyGrasp's embedded legacy module imports IPython even in headless
    # inference. Isaac Sim does not ship the notebook package, and inference
    # does not require its interactive APIs.
    if importlib.util.find_spec("IPython") is None:
        ipython_stub = ModuleType("IPython")
        ipython_stub.__spec__ = importlib.machinery.ModuleSpec(
            "IPython", loader=None
        )
        sys.modules.setdefault("IPython", ipython_stub)

    importlib.invalidate_caches()
    try:
        importlib.import_module("MinkowskiEngine")
        importlib.import_module("pointnet2._ext")
        importlib.import_module("gsnet")
    except Exception as exc:
        raise RuntimeError(
            "AnyGrasp 依赖不可用：需要 Isaac Python 中的 MinkowskiEngine，"
            "PointNet++ CUDA 扩展，并确认 SDK 的 gsnet 二进制、license 目录和 Python 版本匹配。"
        ) from exc
    print(f"✅ AnyGrasp 依赖路径已就绪: {ANYGRASP_DIR}")


def load_anygrasp_model():
    """Create one resident AnyGrasp model for the entire Isaac session."""
    if state._anygrasp_model is not None:
        return state._anygrasp_model
    ensure_anygrasp_python_dependencies()
    if not ANYGRASP_CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            f"未找到 AnyGrasp 权重: {ANYGRASP_CHECKPOINT_PATH}。"
            "请将官方 checkpoint_detection.tar 放入项目 AnyGrasp，"
            "或设置 ANYGRASP_CHECKPOINT_PATH。"
        )
    config = SimpleNamespace(
        checkpoint_path=str(ANYGRASP_CHECKPOINT_PATH),
        max_gripper_width=0.10,
        gripper_height=0.03,
    )
    try:
        from gsnet import create_detector
    except Exception as exc:
        raise RuntimeError("AnyGrasp SDK 已配置，但 gsnet.create_detector 导入失败。") from exc
    try:
        model = create_detector(config)
    except Exception as exc:
        raise RuntimeError(
            f"AnyGrasp 模型加载失败: {ANYGRASP_CHECKPOINT_PATH}; {type(exc).__name__}: {exc}"
        ) from exc
    if model is None:
        raise RuntimeError("AnyGrasp create_detector 返回 None，可能是许可证校验失败。")
    state._anygrasp_model = model
    state._anygrasp_imported = True
    print(f"✅ AnyGrasp 模型已加载: {ANYGRASP_CHECKPOINT_PATH}")
    return model


def release_cuda_inference_cache():
    import gc

    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is None or not torch_module.cuda.is_available():
        return
    torch_module.cuda.empty_cache()
    if hasattr(torch_module.cuda, "ipc_collect"):
        torch_module.cuda.ipc_collect()
    print("✅ 已释放 SAM/AnyGrasp CUDA 缓存")

def _apply_anygrasp_camera_calibration(
    grasp_position,
    camera_position,
    camera_orientation,
):
    """Apply the configured AnyGrasp camera-frame calibration bias."""
    if not ANYGRASP_CALIBRATION_ENABLED:
        return np.asarray(grasp_position, dtype=float), np.zeros(3, dtype=float)

    world_from_camera = quat_to_rot_matrix(camera_orientation)
    correction_world = np.asarray(world_from_camera, dtype=float) @ ANYGRASP_CAMERA_OFFSET
    correction_norm = float(np.linalg.norm(correction_world))
    if (
        ANYGRASP_CALIBRATION_MAX_CORRECTION > 0.0
        and correction_norm > ANYGRASP_CALIBRATION_MAX_CORRECTION
    ):
        correction_world *= ANYGRASP_CALIBRATION_MAX_CORRECTION / correction_norm
    return np.asarray(grasp_position, dtype=float) + correction_world, correction_world


def _masked_rgbd_point_cloud(rgb_data, depth_mm, mask, intrinsic):
    """Convert the segmented RGB-D crop to AnyGrasp's camera-frame cloud."""
    valid = mask.astype(bool) & np.isfinite(depth_mm) & (depth_mm > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size < 32:
        raise RuntimeError("AnyGrasp 点云有效点不足 32 个")
    max_points = 20000
    if rows.size > max_points:
        indices = np.linspace(0, rows.size - 1, max_points, dtype=int)
        rows, cols = rows[indices], cols[indices]
    depth_m = np.asarray(depth_mm[rows, cols], dtype=np.float32) * 0.001
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if min(fx, fy) <= 1e-6:
        raise RuntimeError("相机内参焦距无效，无法生成 AnyGrasp 点云")
    points = np.column_stack(
        ((cols.astype(np.float32) - cx) * depth_m / fx,
         (rows.astype(np.float32) - cy) * depth_m / fy,
         depth_m)
    ).astype(np.float32)
    colors = np.asarray(rgb_data[rows, cols], dtype=np.float32) / 255.0
    return points, colors


def _select_anygrasp_candidate(grasp_group):
    if isinstance(grasp_group, tuple):
        grasp_group = grasp_group[0]
    if hasattr(grasp_group, "nms"):
        grasp_group = grasp_group.nms()
    if hasattr(grasp_group, "sort_by_score"):
        grasp_group = grasp_group.sort_by_score()
    try:
        if len(grasp_group) == 0:
            return None
        return grasp_group[0]
    except TypeError:
        return grasp_group


def _anygrasp_observation_from_frame(
    anygrasp_model,
    rgb_data,
    depth_mm,
    mask,
    intrinsic,
    camera_position,
    camera_orientation,
):
    valid_mask = mask.astype(bool)
    valid_depth = valid_mask & (depth_mm > 0)
    valid_ratio = float(np.count_nonzero(valid_depth)) / max(float(np.count_nonzero(valid_mask)), 1.0)
    if valid_ratio <= 0.05:
        raise RuntimeError("SAM 掩码区域有效深度不足")
    points, colors = _masked_rgbd_point_cloud(
        rgb_data, depth_mm, mask, intrinsic
    )
    try:
        result = anygrasp_model.get_grasp(
            points,
            {
                "dense_grasp": False,
                "collision_detection": True,
                "region_steering": None,
                "approach_steering": None,
                "approach_thresh": np.pi,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"AnyGrasp 推理失败: {exc}") from exc
    detected_grasp = _select_anygrasp_candidate(result)
    if detected_grasp is None:
        raise RuntimeError("AnyGrasp 未返回抓取候选。")
    world_from_camera = get_transform(camera_position, quat_to_rot_matrix(camera_orientation))
    grasp_rotation = np.asarray(detected_grasp.rotation_matrix, dtype=float)
    grasp_translation = np.asarray(detected_grasp.translation, dtype=float)
    # AnyGrasp reports the grasp center. Its documented TCP position is the
    # center plus depth along the first rotation axis.
    grasp_depth = float(getattr(detected_grasp, "depth", 0.0))
    grasp_tip = grasp_translation + grasp_depth * grasp_rotation[:, 0]
    camera_from_grasp = get_transform(grasp_tip, grasp_rotation)
    camera_axis_correction = get_transform([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    gripper_axis_correction = get_transform([0, 0, 0], [[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    world_from_grasp = (
        world_from_camera
        @ camera_axis_correction
        @ camera_from_grasp
        @ gripper_axis_correction
    )
    grasp_approach_direction = world_from_grasp[:3, 2]
    grasp_approach_direction /= max(float(np.linalg.norm(grasp_approach_direction)), 1e-9)
    grasp_position = (
        world_from_grasp[:3, 3]
        + GRASP_POSITION_OFFSET
        # AnyGrasp already accounts for gripper insertion depth above.
    )
    grasp_position, calibration_world_offset = _apply_anygrasp_camera_calibration(
        grasp_position,
        np.asarray(camera_position, dtype=float),
        np.asarray(camera_orientation, dtype=float),
    )
    return GraspObservation(
        position=grasp_position,
        orientation=rot_matrix_to_quat(world_from_grasp[:3, :3]),
        score=float(np.asarray(getattr(detected_grasp, "score", 1.0)).reshape(-1)[0]),
        depth_quality=valid_ratio,
        geometric_validity=1.0 if np.all(np.isfinite(grasp_position)) else 0.0,
    ), calibration_world_offset


def infer_anygrasp_fused_world_pose(
    stage,
    camera,
    target_prim,
    *,
    frame_count=ANYGRASP_FUSION_FRAME_COUNT,
    frame_interval_sec=ANYGRASP_FUSION_FRAME_INTERVAL_SEC,
    apply_calibration=True,
):
    """Infer a stable world grasp pose from a short fresh RGB-D burst."""
    anygrasp_model = load_anygrasp_model()
    frame_count = max(int(frame_count), 1)
    rgb_data, depth_mm = capture_camera_data(camera)
    mask = segment_target_with_sam(
        stage,
        camera,
        target_prim,
        rgb_data,
        prim_path=target_prim.prim_path,
    )
    if not np.any(mask.astype(bool) & (depth_mm > 0)):
        raise RuntimeError("SAM 掩码区域没有有效深度，请检查相机视角、裁剪范围和目标提示点。")
    intrinsic = np.asarray(camera.get_intrinsics_matrix(), dtype=float)
    _, target_bbox_min, target_bbox_max = get_current_bbox_center(
        stage, target_prim, target_prim.prim_path
    )
    expanded_bbox_min = np.asarray(target_bbox_min, dtype=float) - 0.01
    expanded_bbox_max = np.asarray(target_bbox_max, dtype=float) + 0.01
    observations = []
    started = time.perf_counter()
    calibration_offsets = []
    for frame_index in range(frame_count):
        if frame_index:
            frame_steps = max(1, int(math.ceil(float(frame_interval_sec) * 60.0)))
            step_app(frame_steps)
            rgb_data, depth_mm = capture_camera_data(camera)
        camera_position, camera_orientation = SingleXFormPrim(
            name="grasp_camera_pose", prim_path=CAMERA_PRIM_PATH
        ).get_world_pose()
        try:
            observation, calibration_offset = _anygrasp_observation_from_frame(
                anygrasp_model,
                rgb_data,
                depth_mm,
                mask,
                intrinsic,
                camera_position,
                camera_orientation,
            )
        except Exception as exc:
            print(f"⚠️ AnyGrasp 第 {frame_index + 1}/{frame_count} 帧无效: {exc}")
            continue
        outside_distance = np.maximum(expanded_bbox_min - observation.position, 0.0)
        outside_distance += np.maximum(observation.position - expanded_bbox_max, 0.0)
        geometry_quality = float(
            math.exp(-float(np.linalg.norm(outside_distance)) / 0.02)
        )
        observation = GraspObservation(
            position=observation.position,
            orientation=observation.orientation,
            score=observation.score,
            depth_quality=observation.depth_quality,
            geometric_validity=geometry_quality,
        )
        if not apply_calibration:
            observation = GraspObservation(
                position=observation.position - calibration_offset,
                orientation=observation.orientation,
                score=observation.score,
                depth_quality=observation.depth_quality,
                geometric_validity=observation.geometric_validity,
            )
        observations.append(observation)
        calibration_offsets.append(
            calibration_offset if apply_calibration else np.zeros(3, dtype=float)
        )

    fused = fuse_grasp_observations(
        observations,
        max_position_dispersion_m=ANYGRASP_FUSION_MAX_POSITION_DISPERSION_M,
        max_orientation_dispersion_deg=ANYGRASP_FUSION_MAX_ORIENTATION_DISPERSION_DEG,
        position_outlier_floor_m=ANYGRASP_FUSION_POSITION_OUTLIER_FLOOR_M,
        min_confidence=ANYGRASP_FUSION_MIN_CONFIDENCE,
    )
    grasp_position = np.asarray(fused["position"], dtype=float)
    grasp_orientation = np.asarray(fused["orientation"], dtype=float)
    if SHOW_GRASP_DEBUG:
        visualization_path = "/World/GraspVisualization"
        if not stage.GetPrimAtPath(visualization_path).IsValid():
            UsdGeom.Xform.Define(stage, visualization_path)
        grasp_visualization = SingleXFormPrim(
            name="grasp_visualization", prim_path=visualization_path
        )
        grasp_visualization.set_world_pose(
            position=grasp_position, orientation=grasp_orientation
        )
    fused["calibration_offset_world"] = (
        np.mean(np.asarray(calibration_offsets, dtype=float), axis=0).tolist()
        if calibration_offsets else [0.0, 0.0, 0.0]
    )
    fused["inference_duration_sec"] = round(time.perf_counter() - started, 3)
    print(
        "✅ AnyGrasp 多帧融合: "
        f"accepted={fused['accepted_frame_count']}/{fused['frame_count']}, "
        f"position_std={fused['position_std_m']:.4f} m, "
        f"orientation_dispersion={fused['orientation_dispersion_deg']:.2f} deg, "
        f"confidence={fused['confidence']:.3f}"
    )
    return fused


def infer_anygrasp_world_pose(
    stage,
    camera,
    target_prim,
    *,
    apply_calibration=True,
):
    """Backward-compatible single-frame API for diagnostics."""
    result = infer_anygrasp_fused_world_pose(
        stage,
        camera,
        target_prim,
        frame_count=1,
        frame_interval_sec=0.0,
        apply_calibration=apply_calibration,
    )
    return np.asarray(result["position"], dtype=float), np.asarray(result["orientation"], dtype=float)


def resolve_scene_prim_path(name):
    normalized_name = str(name).strip()
    if not normalized_name:
        raise ValueError("目标名称不能为空")
    for prim_path in state.SCENE_NAME_RESOLVER.prim_candidates(normalized_name):
        if state.stage.GetPrimAtPath(prim_path).IsValid():
            return prim_path
    canonical_name = state.SCENE_NAME_RESOLVER.canonicalize(normalized_name)
    lowered_names = (
        {normalized_name.lower(), canonical_name.lower()}
        | state.SCENE_NAME_RESOLVER.resolve_traversal_names(canonical_name)
    )
    for prim in state.stage.Traverse():
        if prim.GetName().lower() in lowered_names:
            return str(prim.GetPath())
    raise RuntimeError(
        f"场景中未找到目标 Prim: {name} (canonical={canonical_name})"
    )
