"""AuraVLA 物理仿真模块：SimulationContext、碰撞体与接触物理。"""

from __future__ import annotations

import os
import time

import numpy as np
from omni.kit.app import get_app
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from aura_isaac_bridge.core.state import state
from aura_isaac_bridge.core.state import (
    BANANA_STATIC_FRICTION, BANANA_DYNAMIC_FRICTION,
    GRIPPER_STATIC_FRICTION, GRIPPER_DYNAMIC_FRICTION,
    PHYSX_CONTACT_OFFSET, PHYSX_REST_OFFSET,
    PHYSX_SOLVER_POSITION_ITERATIONS, PHYSX_SOLVER_VELOCITY_ITERATIONS,
    PHYSX_MAX_DEPENETRATION_VELOCITY,
    PHYSX_ENABLE_CCD,
    PHYSX_CONVEX_HULL_VERTEX_LIMIT, PHYSX_CONVEX_MAX_HULLS,
    PHYSX_CONVEX_MIN_THICKNESS, PHYSX_CONVEX_SHRINK_WRAP,
    PHYSX_CONVEX_ERROR_PERCENTAGE,
    PHYSX_TIME_STEPS_PER_SECOND, PHYSX_SOLVER_TYPE,
    PHYSX_BOUNCE_THRESHOLD_VELOCITY,
    GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_MAX_EFFORT,
)


STEP_TIMING_WINDOW = max(int(os.environ.get("AURA_STEP_TIMING_WINDOW", "120")), 0)
_step_timing = {"count": 0, "wall": 0.0, "worst": 0.0}


def _physics_steppable():
    """True only when sim_context.step() can safely dereference physics.

    step() calls get_physics_dt() with no guard, which raises on a None
    _physics_context -- the state before play() and after stop().
    """
    sim_context = state.sim_context
    if sim_context is None or not sim_context.is_playing():
        return False
    return getattr(sim_context, "_physics_context", None) is not None


def step_app(frames=1, render=True):
    """Advance physics by a fixed physics_dt, decoupled from rendering.

    `get_app().update()` runs the Kit main loop, so the physics step it drives
    is as long as the frame took to render.  Under GraspNet/SAM GPU contention
    that stretched to ~67 ms (the 15 Hz the trajectory constants were tuned
    against) versus the declared 16.7 ms, which is what makes the arm stutter
    and teleport: every frame integrates a different, unknown dt.

    `sim_context.step()` uses the declared physics_dt regardless of how long
    rendering takes, so motion advances at a constant rate.

    Before play() the physics context does not exist yet, and during teardown
    it is torn down again; sim_context.step() dereferences it unconditionally.
    Those phases only need the app to pump, so fall back to update() there.
    """
    if not _physics_steppable():
        for _ in range(frames):
            get_app().update()
        return

    sim_context = state.sim_context
    for _ in range(frames):
        started = time.perf_counter()
        sim_context.step(render=render)
        elapsed = time.perf_counter() - started
        if STEP_TIMING_WINDOW:
            _step_timing["count"] += 1
            _step_timing["wall"] += elapsed
            _step_timing["worst"] = max(_step_timing["worst"], elapsed)
            if _step_timing["count"] >= STEP_TIMING_WINDOW:
                mean_ms = 1000.0 * _step_timing["wall"] / _step_timing["count"]
                print(
                    f"⏱️ 步进耗时: physics_dt="
                    f"{1000.0 * float(sim_context.get_physics_dt()):.2f} ms, "
                    f"墙钟均值={mean_ms:.2f} ms, "
                    f"最差={1000.0 * _step_timing['worst']:.2f} ms, "
                    f"样本={_step_timing['count']}"
                )
                _step_timing.update(count=0, wall=0.0, worst=0.0)

def configure_physics_scene(scene_prim):
    """Pin the scene's solver settings instead of inheriting Kit defaults.

    `UsdPhysics.Scene.Define` alone leaves timeStepsPerSecond, solver type and
    stabilization at whatever the stage defaults are, so the substep count that
    physics actually runs at is decoupled from the declared physics_dt.
    """
    if not scene_prim.IsValid():
        print("⚠️ /PhysicsScene 无效，跳过求解器配置")
        return

    physx_scene = (
        PhysxSchema.PhysxSceneAPI(scene_prim)
        if scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI)
        else PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    )
    physx_scene.CreateTimeStepsPerSecondAttr().Set(PHYSX_TIME_STEPS_PER_SECOND)
    physx_scene.CreateSolverTypeAttr().Set(PHYSX_SOLVER_TYPE)
    physx_scene.CreateEnableStabilizationAttr().Set(True)
    physx_scene.CreateEnableCCDAttr().Set(PHYSX_ENABLE_CCD)
    physx_scene.CreateEnableGPUDynamicsAttr().Set(False)
    # Default lets the jaw and the tabletop bounce on first touch.  The schema
    # attribute is physxScene:bounceThreshold -- there is no ...VelocityAttr.
    physx_scene.CreateBounceThresholdAttr().Set(PHYSX_BOUNCE_THRESHOLD_VELOCITY)
    print(
        "✅ /PhysicsScene 求解器已配置: "
        f"{PHYSX_TIME_STEPS_PER_SECOND} Hz, "
        f"solver={PHYSX_SOLVER_TYPE}, stabilization=on, "
        f"CCD={'on' if PHYSX_ENABLE_CCD else 'off'}, GPU=off, "
        f"bounce_threshold={PHYSX_BOUNCE_THRESHOLD_VELOCITY} m/s"
    )


def cleanup_debug_markers(stage):
    marker_paths = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path == "/World/debug_grasp_point" or prim_path.startswith("/World/debug_target_"):
            marker_paths.append(prim_path)
    for prim_path in marker_paths:
        delete_prim(prim_path)


def create_grasp_physics_material(
    stage,
    material_path,
    static_friction,
    dynamic_friction,
):
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = (
        UsdPhysics.MaterialAPI(material.GetPrim())
        if material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
        else UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    )
    physics_material.CreateStaticFrictionAttr(float(static_friction))
    physics_material.CreateDynamicFrictionAttr(float(dynamic_friction))
    physics_material.CreateRestitutionAttr(0.0)
    physx_material = (
        PhysxSchema.PhysxMaterialAPI(material.GetPrim())
        if material.GetPrim().HasAPI(PhysxSchema.PhysxMaterialAPI)
        else PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    )
    physx_material.CreateFrictionCombineModeAttr().Set("max")
    return material


def bind_grasp_physics_material(prim, material):
    binding_api = (
        UsdShade.MaterialBindingAPI(prim)
        if prim.HasAPI(UsdShade.MaterialBindingAPI)
        else UsdShade.MaterialBindingAPI.Apply(prim)
    )
    binding_api.Bind(
        material,
        bindingStrength="strongerThanDescendants",
        materialPurpose="physics",
    )

def ensure_pickable_object(stage, prim_path):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"{prim_path} 不存在，无法配置物理属性")

    material_path = f"{prim_path.rstrip('/')}_grasp_material"
    material = create_grasp_physics_material(
        stage,
        material_path,
        BANANA_STATIC_FRICTION,
        BANANA_DYNAMIC_FRICTION,
    )
    bind_grasp_physics_material(root, material)

    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root)
        print(f"✅ 已给 {prim_path} root 添加 RigidBodyAPI")
    else:
        print(f"✅ {prim_path} root 已有 RigidBodyAPI")
    # Every grasp starts from a dynamic object. This also clears the kinematic
    # state left by an older runtime that froze the object before jaw closure.
    UsdPhysics.RigidBodyAPI(root).CreateKinematicEnabledAttr().Set(False)

    # 适当增加质量和阻尼，减少搬运时因接触/惯性产生的晃动。
    mass_api = UsdPhysics.MassAPI(root) if root.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(root)
    mass_api.GetMassAttr().Set(0.05)

    physx_body_api = (
        PhysxSchema.PhysxRigidBodyAPI(root)
        if root.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        else PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    )
    physx_body_api.CreateLinearDampingAttr().Set(12.0)
    physx_body_api.CreateAngularDampingAttr().Set(12.0)
    physx_body_api.CreateEnableCCDAttr().Set(PHYSX_ENABLE_CCD)
    physx_body_api.CreateSolverPositionIterationCountAttr().Set(
        PHYSX_SOLVER_POSITION_ITERATIONS
    )
    physx_body_api.CreateSolverVelocityIterationCountAttr().Set(
        PHYSX_SOLVER_VELOCITY_ITERATIONS
    )
    physx_body_api.CreateMaxDepenetrationVelocityAttr().Set(
        PHYSX_MAX_DEPENETRATION_VELOCITY
    )

    collider_count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = (
                UsdPhysics.MeshCollisionAPI(prim)
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                else UsdPhysics.MeshCollisionAPI.Apply(prim)
            )
            mesh_collision.GetApproximationAttr().Set("convexDecomposition")
            # Replace coarse source-asset hull settings with a close-fitting
            # decomposition. ShrinkWrap is important for narrow jaw contact.
            for schema in (
                PhysxSchema.PhysxConvexHullCollisionAPI,
                PhysxSchema.PhysxConvexDecompositionCollisionAPI,
            ):
                if prim.HasAPI(schema):
                    try:
                        prim.RemoveAPI(schema)
                    except Exception:
                        pass
            decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            decomposition.CreateHullVertexLimitAttr().Set(PHYSX_CONVEX_HULL_VERTEX_LIMIT)
            decomposition.CreateMaxConvexHullsAttr().Set(PHYSX_CONVEX_MAX_HULLS)
            decomposition.CreateMinThicknessAttr().Set(PHYSX_CONVEX_MIN_THICKNESS)
            decomposition.CreateShrinkWrapAttr().Set(PHYSX_CONVEX_SHRINK_WRAP)
            decomposition.CreateErrorPercentageAttr().Set(PHYSX_CONVEX_ERROR_PERCENTAGE)
            physx_collision = (
                PhysxSchema.PhysxCollisionAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                else PhysxSchema.PhysxCollisionAPI.Apply(prim)
            )
            physx_collision.CreateContactOffsetAttr().Set(PHYSX_CONTACT_OFFSET)
            physx_collision.CreateRestOffsetAttr().Set(PHYSX_REST_OFFSET)
            bind_grasp_physics_material(prim, material)
            collider_count += 1

    if collider_count == 0 and not root.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(root)
        collider_count = 1

    print(
        f"✅ {prim_path} 碰撞体数量: {collider_count}, "
        f"mass=0.05kg, CCD={'on' if PHYSX_ENABLE_CCD else 'off'}, contact/rest="
        f"{PHYSX_CONTACT_OFFSET:.4f}/{PHYSX_REST_OFFSET:.4f}m"
    )


def configure_static_contact_offsets(stage, prim_path):
    """Keep static support surfaces on the same contact shell as the gripper."""
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"{prim_path} 不存在，无法配置静态碰撞接触距离")

    configured = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_enabled = UsdPhysics.CollisionAPI(
            prim
        ).GetCollisionEnabledAttr().Get()
        if collision_enabled is False:
            continue
        collision_api = (
            PhysxSchema.PhysxCollisionAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            else PhysxSchema.PhysxCollisionAPI.Apply(prim)
        )
        collision_api.CreateContactOffsetAttr().Set(PHYSX_CONTACT_OFFSET)
        collision_api.CreateRestOffsetAttr().Set(PHYSX_REST_OFFSET)
        configured += 1

    print(
        f"✅ {prim_path} 静态碰撞接触距离已同步: colliders={configured}, "
        f"contact/rest={PHYSX_CONTACT_OFFSET:.4f}/{PHYSX_REST_OFFSET:.4f}m"
    )
    return configured


def get_dach_finger_paths():
    finger_paths = []
    for suffix in ("L", "R"):
        for side in ("left", "right"):
            finger_paths.append(
                f"/World/DACH_TRON2A/grasper_{suffix}_jaw_{side}_Link"
            )
    return finger_paths


def prepare_dach_finger_collision_instances(stage):
    deinstanced = 0
    for finger_path in get_dach_finger_paths():
        collision_root = stage.GetPrimAtPath(f"{finger_path}/collisions")
        if collision_root.IsValid() and collision_root.IsInstance():
            collision_root.SetInstanceable(False)
            deinstanced += 1
    print(f"✅ DACH 夹指碰撞实例已展开: {deinstanced}")


def configure_dach_contact_physics(stage, *, active_arm=None, left_arm=None, right_arm=None):
    SimulationManager.enable_ccd(PHYSX_ENABLE_CCD)
    articulation_prim = stage.GetPrimAtPath("/World/DACH_TRON2A")
    articulation_api = (
        PhysxSchema.PhysxArticulationAPI(articulation_prim)
        if articulation_prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
        else PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
    )
    articulation_api.CreateSolverPositionIterationCountAttr().Set(
        PHYSX_SOLVER_POSITION_ITERATIONS
    )
    articulation_api.CreateSolverVelocityIterationCountAttr().Set(
        PHYSX_SOLVER_VELOCITY_ITERATIONS
    )

    finger_material = create_grasp_physics_material(
        stage,
        "/World/AuraFingerGripPhysicsMaterial",
        GRIPPER_STATIC_FRICTION,
        GRIPPER_DYNAMIC_FRICTION,
    )
    configured_colliders = 0
    for finger_path in get_dach_finger_paths():
        finger_prim = stage.GetPrimAtPath(finger_path)
        if not finger_prim.IsValid():
            continue
        if finger_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_api = (
                PhysxSchema.PhysxRigidBodyAPI(finger_prim)
                if finger_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                else PhysxSchema.PhysxRigidBodyAPI.Apply(finger_prim)
            )
            body_api.CreateEnableCCDAttr().Set(PHYSX_ENABLE_CCD)
            body_api.CreateSolverPositionIterationCountAttr().Set(
                PHYSX_SOLVER_POSITION_ITERATIONS
            )
            body_api.CreateSolverVelocityIterationCountAttr().Set(
                PHYSX_SOLVER_VELOCITY_ITERATIONS
            )
            body_api.CreateMaxDepenetrationVelocityAttr().Set(
                PHYSX_MAX_DEPENETRATION_VELOCITY
            )
        for prim in Usd.PrimRange(finger_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_api = (
                PhysxSchema.PhysxCollisionAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                else PhysxSchema.PhysxCollisionAPI.Apply(prim)
            )
            collision_api.CreateContactOffsetAttr().Set(PHYSX_CONTACT_OFFSET)
            collision_api.CreateRestOffsetAttr().Set(PHYSX_REST_OFFSET)
            bind_grasp_physics_material(prim, finger_material)
            configured_colliders += 1

    active_arm = active_arm or state.dach_arm
    left_arm = left_arm or state.dach_left
    right_arm = right_arm or state.dach_right
    if active_arm is None or left_arm is None or right_arm is None:
        raise RuntimeError("DACH arm views must be initialized before contact physics")
    articulation_controller = active_arm.articulation.get_articulation_controller()
    stiffnesses, dampings = articulation_controller.get_gains()
    max_efforts = articulation_controller.get_max_efforts()
    gripper_indices = np.unique(
        np.concatenate(
            (
                left_arm._gripper_indices,
                right_arm._gripper_indices,
            )
        )
    )
    stiffnesses[gripper_indices] = GRIPPER_STIFFNESS
    dampings[gripper_indices] = GRIPPER_DAMPING
    max_efforts[gripper_indices] = GRIPPER_MAX_EFFORT
    articulation_controller.set_gains(kps=stiffnesses, kds=dampings)
    articulation_controller.set_max_efforts(max_efforts)
    print(
        "✅ DACH 夹爪接触物理已配置: "
        f"CCD={SimulationManager.is_ccd_enabled()}, "
        f"solver={PHYSX_SOLVER_POSITION_ITERATIONS}/"
        f"{PHYSX_SOLVER_VELOCITY_ITERATIONS}, "
        f"jaw_kp/kd={GRIPPER_STIFFNESS:.1f}/{GRIPPER_DAMPING:.1f}, "
        f"max_effort={GRIPPER_MAX_EFFORT:.1f}N, "
        f"colliders={configured_colliders}"
    )
