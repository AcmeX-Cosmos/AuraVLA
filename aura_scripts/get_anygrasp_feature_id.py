#!/usr/bin/env python3
"""Print the official AnyGrasp feature ID for this machine.

This follows the SDK license-registration procedure: copy the gsnet binary
matching the active Python ABI into ``license_registration/gsnet.so``, then
call ``gsnet.get_feature_id()``. It does not inspect or modify license files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDK_DIR = (
    PROJECT_ROOT
    / "aura_hardware"
    / "aura_isaac_bridge"
    / "thirdparty"
    / "anygrasp"
    / "sdk"
)
DEFAULT_ISAAC_PYTHON = Path(
    os.environ.get("ISAAC_PYTHON", "/home/acmex/Code/learning/isaacsim/python.sh")
)


def _binary_name() -> str:
    return (
        "gsnet.cpython-"
        f"{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
    )


def _run_with_interpreter(interpreter: Path, script: Path, argv: list[str]) -> int:
    """Re-run this script with the requested runtime interpreter."""
    command = [str(interpreter), str(script), *argv, "--_runtime"]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the official AnyGrasp machine feature ID"
    )
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=DEFAULT_SDK_DIR,
        help="Path to the private AnyGrasp SDK directory",
    )
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=None,
        help=(
            "Re-run with this Python interpreter. Use Isaac Sim python.sh "
            "when registering the ID used by the Isaac runtime."
        ),
    )
    parser.add_argument(
        "--_runtime",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.runtime_python is not None and not args._runtime:
        interpreter = args.runtime_python.expanduser().resolve()
        if not interpreter.is_file():
            print(f"ERROR: Python interpreter not found: {interpreter}", file=sys.stderr)
            return 2
        forwarded = ["--sdk-dir", str(args.sdk_dir)]
        return _run_with_interpreter(interpreter, Path(__file__).resolve(), forwarded)

    sdk_dir = args.sdk_dir.expanduser().resolve()
    registration_dir = sdk_dir / "license_registration"
    source = sdk_dir / "grasp_detection" / "gsnet_versions" / _binary_name()
    destination = registration_dir / "gsnet.so"
    if not source.is_file():
        print(
            "ERROR: no compatible AnyGrasp gsnet binary for "
            f"Python {sys.version_info.major}.{sys.version_info.minor}: {source}",
            file=sys.stderr,
        )
        return 2
    if not registration_dir.is_dir():
        print(f"ERROR: AnyGrasp license-registration directory not found: {registration_dir}", file=sys.stderr)
        return 2

    # This is the SDK's documented `cp ... gsnet.so` preparation step.
    shutil.copy2(source, destination)
    sys.path.insert(0, str(registration_dir))
    try:
        from gsnet import get_feature_id

        feature_id = str(get_feature_id()).rstrip("%")
    except Exception as exc:
        print(f"ERROR: AnyGrasp feature-ID query failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"python={sys.executable}", file=sys.stderr)
    print(f"gsnet={source}", file=sys.stderr)
    print(f"feature_id={feature_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
