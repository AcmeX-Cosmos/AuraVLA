"""
AuraVLA Isaac Sim Complete Runtime
一键启动任务桥接和相机桥接
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Isaac's VS Code executor evaluates this source with its own __file__. Use an
# explicit project root instead of deriving paths from the executor module.
_aura_root = Path(
    os.environ.get(
        "AURA_VLA_ROOT",
        "/home/acmex/Code/learning/courses/AuraVLA",
    )
).expanduser().resolve()
_execution_path = _aura_root / "aura_execution" / "aura_execution"
_python_paths = (
    _execution_path,
    _aura_root / "aura_orchestration",
    _aura_root / "aura_planning",
)
for _path in _python_paths:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task_bridge import start_task_bridge


_execute_pick_place = globals().get("execute_pick_place_with_reset")
if not callable(_execute_pick_place):
    _execute_pick_place = globals().get("execute_pick_place")
if not callable(_execute_pick_place):
    raise RuntimeError(
        "A real Isaac robot runtime is not loaded: expected callable "
        "execute_pick_place_with_reset or execute_pick_place"
    )


# 启动任务桥接
print("启动任务桥接...")
task_bridge = start_task_bridge(_execute_pick_place)

# 启动相机桥接
print("启动相机桥接...")
_camera_script = _aura_root / "aura_hardware" / "aura_isaac_bridge" / "start_camera_bridge.py"
exec(compile(_camera_script.read_text(encoding="utf-8"), str(_camera_script), "exec"))

print("\n" + "="*60)
print("✓ AuraVLA Isaac Runtime 已就绪")
print("="*60)
print(f"任务桥接: {task_bridge.paths.directory}")
print(f"相机输出: {camera_bridge.paths.directory}")
print("\n现在可以启动 NVIDIA Agent 了")
print("="*60 + "\n")
