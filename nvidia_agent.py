#!/usr/bin/env python3
"""AuraVLA conversational NVIDIA agent entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
for package_root in (
    PROJECT_ROOT / "aura_perception",
    PROJECT_ROOT / "aura_hardware" / "aura_isaac_bridge",
    PROJECT_ROOT / "aura_execution" / "aura_execution",
    PROJECT_ROOT / "aura_planning" / "aura_planning",
):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from aura_perception.nvidia_agent import main


if __name__ == "__main__":
    raise SystemExit(main())
