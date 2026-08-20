"""AuraVLA 感知模块：BBox、PCA、网格质心、SAM、GraspNet 与相机。"""

from __future__ import annotations

import importlib
import json
import os
import site
import struct
import sys
import time
import zlib
from pathlib import Path

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
    GRASPNET_DIR, GRASPNET_CHECKPOINT_PATH, SAM_MODEL_PATH,
    DEFAULT_GRASPNET_DIR, DEFAULT_SAM_MODEL_PATH,
    CAMERA_PRIM_PATH, CAMERA_RESOLUTION, CAMERA_PREVIEW_RESOLUTION,
    SHOW_GRASP_DEBUG, USE_GRASPNET, USE_GRASPNET_ORIENTATION,
    GRASP_POSITION_OFFSET, GRASP_INSERT_DEPTH,
    GRASPNET_CALIBRATION_ENABLED, GRASPNET_CAMERA_OFFSET,
    GRASPNET_CALIBRATION_MAX_CORRECTION,
)
from aura_isaac_bridge.core.physics import step_app

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
    print(f"✅ GraspNet 相机初始化完成: {CAMERA_PRIM_PATH}, resolution={CAMERA_RESOLUTION}")
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
        "GraspNet Camera RGB",
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
    print("📷 相机预览已打开，继续执行 SAM、GraspNet 和机械臂夹取。")

def get_current_object_center(stage, target_prim, prim_path=None):
    prim_path = str(prim_path or target_prim.prim_path)
    usd_origin, usd_orientation = target_prim.get_world_pose()
    sim_origin, sim_orientation, _ = get_sim_pose(target_prim)
    bbox_center, _, _ = get_bbox_center(stage, prim_path)
    local_bbox_offset = quat_rotate(
        np.asarray(usd_orientation, dtype=float) * np.array([1.0, -1.0, -1.0, -1.0]),
        bbox_center - np.asarray(usd_origin, dtype=float),
    )
    return sim_origin + quat_rotate(sim_orientation, local_bbox_offset)

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
    show_camera_preview(rgb_data, prompt_points, prompt_labels)
    print(f"🧠 加载 SAM 模型: {SAM_MODEL_PATH}")
    sam_model = SAM(SAM_MODEL_PATH)
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
        raise RuntimeError("SAM 掩码为空，无法执行 GraspNet 推理。")
    return mask

def ensure_graspnet_python_dependencies():
    site_packages_path = str(ISAAC_SITE_PACKAGES)
    if not ISAAC_SITE_PACKAGES.is_dir():
        raise RuntimeError(
            f"未找到 Isaac Python site-packages: {ISAAC_SITE_PACKAGES}。"
            "请通过 ISAAC_PYTHON_SITE_PACKAGES 指定实际目录。"
        )
    sys.path[:] = [path for path in sys.path if path != site_packages_path]
    site.addsitedir(site_packages_path)
    sys.path.remove(site_packages_path)
    sys.path.insert(0, site_packages_path)
    sys.path_importer_cache.pop(site_packages_path, None)
    importlib.invalidate_caches()
    print(f"✅ 已刷新 Isaac Python 依赖目录: {site_packages_path}")

    required_modules = ("filelock", "open3d", "ultralytics", "grasp_nms")
    for module_name in required_modules:
        if sys.modules.get(module_name) is None:
            for loaded_module_name in list(sys.modules):
                if loaded_module_name == module_name or loaded_module_name.startswith(module_name + "."):
                    sys.modules.pop(loaded_module_name, None)
    missing_modules = [
        name
        for name in required_modules
        if not (ISAAC_SITE_PACKAGES / name).is_dir()
        and not (ISAAC_SITE_PACKAGES / f"{name}.py").is_file()
        and not any(ISAAC_SITE_PACKAGES.glob(f"{name}.*"))
    ]
    if missing_modules:
        raise RuntimeError(
            "Isaac Python 缺少 GraspNet 依赖: "
            f"{', '.join(missing_modules)}。当前依赖目录: {site_packages_path}"
        )

def load_graspnet_demo():
    ensure_graspnet_python_dependencies()
    demo_path = GRASPNET_DIR / "demo.py"
    if not demo_path.is_file():
        replacement_demo = SECTION3_DIR / "thirdparty" / "graspnet-baseline_change" / "demo.py"
        raise RuntimeError(
            "未找到完整的 graspnet-baseline。请将其放到 "
            f"{GRASPNET_DIR}，并用 {replacement_demo} 替换 baseline 中的 demo.py；"
            "也可以通过 GRASPNET_BASELINE_DIR 指定实际目录。"
        )
    if not GRASPNET_CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            f"未找到 GraspNet 权重: {GRASPNET_CHECKPOINT_PATH}。"
            "请放入 checkpoint-rs.tar，或设置 GRASPNET_CHECKPOINT_PATH。"
        )

    module_name = "section3_graspnet_demo"
    graspnet_dir_path = str(GRASPNET_DIR)
    if graspnet_dir_path not in sys.path:
        sys.path.insert(0, graspnet_dir_path)
        sys.path_importer_cache.pop(graspnet_dir_path, None)
        importlib.invalidate_caches()
    if module_name in sys.modules:
        demo_module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, demo_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载 GraspNet demo: {demo_path}")
        demo_module = importlib.util.module_from_spec(spec)
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            sys.modules[module_name] = demo_module
            spec.loader.exec_module(demo_module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            sys.argv = original_argv

    if not hasattr(demo_module, "demo_variable"):
        raise RuntimeError(f"{demo_path} 不包含 demo_variable，请使用 section3 提供的改版 demo.py。")
    if hasattr(demo_module, "cfgs"):
        demo_module.cfgs.checkpoint_path = str(GRASPNET_CHECKPOINT_PATH)
    return demo_module

def release_cuda_inference_cache():
    import gc

    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is None or not torch_module.cuda.is_available():
        return
    torch_module.cuda.empty_cache()
    if hasattr(torch_module.cuda, "ipc_collect"):
        torch_module.cuda.ipc_collect()
    print("✅ 已释放 SAM/GraspNet CUDA 缓存")

def _apply_graspnet_camera_calibration(
    grasp_position,
    camera_position,
    camera_orientation,
):
    """Apply one weighted multi-object GraspNet bias in the camera frame."""
    if not GRASPNET_CALIBRATION_ENABLED:
        return np.asarray(grasp_position, dtype=float), np.zeros(3, dtype=float)

    world_from_camera = quat_to_rot_matrix(camera_orientation)
    correction_world = np.asarray(world_from_camera, dtype=float) @ GRASPNET_CAMERA_OFFSET
    correction_norm = float(np.linalg.norm(correction_world))
    if (
        GRASPNET_CALIBRATION_MAX_CORRECTION > 0.0
        and correction_norm > GRASPNET_CALIBRATION_MAX_CORRECTION
    ):
        correction_world *= GRASPNET_CALIBRATION_MAX_CORRECTION / correction_norm
    return np.asarray(grasp_position, dtype=float) + correction_world, correction_world


def infer_graspnet_world_pose(
    stage,
    camera,
    target_prim,
    *,
    apply_calibration=True,
):
    demo_module = load_graspnet_demo()
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

    print("🧠 开始 GraspNet 推理...")
    detected_grasp = demo_module.demo_variable(rgb_data, depth_mm, mask, intrinsic)
    if detected_grasp is None:
        raise RuntimeError("GraspNet 未返回抓取候选。")

    camera_position, camera_orientation = SingleXFormPrim(
        name="grasp_camera_pose", prim_path=CAMERA_PRIM_PATH
    ).get_world_pose()
    world_from_camera = get_transform(camera_position, quat_to_rot_matrix(camera_orientation))
    camera_from_grasp = get_transform(detected_grasp.translation, detected_grasp.rotation_matrix)
    camera_axis_correction = get_transform([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    gripper_axis_correction = get_transform([0, 0, 0], [[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    world_from_grasp = world_from_camera @ camera_axis_correction @ camera_from_grasp @ gripper_axis_correction

    grasp_approach_direction = world_from_grasp[:3, 2]
    grasp_approach_direction /= np.linalg.norm(grasp_approach_direction)
    grasp_position = (
        world_from_grasp[:3, 3]
        + GRASP_POSITION_OFFSET
        + grasp_approach_direction * GRASP_INSERT_DEPTH
    )
    if apply_calibration:
        grasp_position, calibration_world_offset = _apply_graspnet_camera_calibration(
            grasp_position,
            np.asarray(camera_position, dtype=float),
            np.asarray(camera_orientation, dtype=float),
        )
        if GRASPNET_CALIBRATION_ENABLED:
            print(
                "📐 GraspNet 相机标定修正: "
                f"camera_offset={GRASPNET_CAMERA_OFFSET}, "
                f"world_offset={calibration_world_offset}"
            )
    grasp_orientation = rot_matrix_to_quat(world_from_grasp[:3, :3])
    visualization_path = "/World/GraspVisualization"
    if not stage.GetPrimAtPath(visualization_path).IsValid():
        UsdGeom.Xform.Define(stage, visualization_path)
    grasp_visualization = SingleXFormPrim(name="grasp_visualization", prim_path=visualization_path)
    grasp_visualization.set_world_pose(position=grasp_position, orientation=grasp_orientation)
    print(f"✅ GraspNet 抓取位置: {grasp_position}")
    print(f"✅ GraspNet 抓取姿态(wxyz): {grasp_orientation}")
    print(f"✅ 抓取点沿接近方向深入: {GRASP_INSERT_DEPTH:.3f} m")
    return np.asarray(grasp_position, dtype=float), np.asarray(grasp_orientation, dtype=float)


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
