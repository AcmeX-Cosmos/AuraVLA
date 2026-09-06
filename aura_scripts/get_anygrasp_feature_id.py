#!/usr/bin/env python3
"""Print the official AnyGrasp feature ID for this machine.

This follows the SDK license-registration procedure: copy the gsnet binary
matching the active Python ABI into ``license_registration/gsnet.so``, then
call ``gsnet.get_feature_id()``. It does not inspect or modify license files.
"""

from __future__ import annotations

import argparse
import os
import re
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


def _running_inside_isaac_python() -> bool:
    """Identify the Isaac Sim interpreter without relying on conda metadata."""
    executable = str(Path(sys.executable).resolve())
    prefix = str(Path(sys.prefix).resolve())
    isaac_root = _isaac_root()
    return executable.startswith(isaac_root + os.sep) or prefix.startswith(isaac_root + os.sep)


def _isaac_root() -> str:
    return str(Path(os.environ.get(
        "ISAAC_SIM_ROOT", "/home/acmex/Code/learning/isaacsim"
    )).expanduser().resolve())


def _has_host_network_interface() -> bool:
    """Reject sandboxes that expose only loopback/container interfaces."""
    if Path("/.dockerenv").exists():
        return False
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8").lower()
    except OSError:
        cgroup = ""
    if any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "podman")):
        return False
    network_root = Path("/sys/class/net")
    if not network_root.is_dir():
        return False
    virtual_prefixes = ("docker", "veth", "br-", "virbr", "cni", "tun", "tap")
    for interface in network_root.iterdir():
        name = interface.name
        if name == "lo" or name.startswith(virtual_prefixes):
            continue
        try:
            mac = (interface / "address").read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if re.fullmatch(r"[0-9a-f]{2}(:[0-9a-f]{2}){5}", mac) and mac != "00:00:00:00:00:00":
            return True
    return False


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

    # Running this file with `python3` is supported, but it must still use the
    # same Isaac Python ABI as AnyGrasp inference. The child is marked as a
    # runtime invocation to prevent a second re-exec.
    if (
        not args._runtime
        and args.runtime_python is None
        and not _running_inside_isaac_python()
    ):
        if not DEFAULT_ISAAC_PYTHON.is_file():
            print(
                "ERROR: Isaac Sim Python was not found. Refusing to generate a "
                "feature ID with system or conda Python: "
                f"{DEFAULT_ISAAC_PYTHON}",
                file=sys.stderr,
            )
            return 2
        return _run_with_interpreter(
            DEFAULT_ISAAC_PYTHON,
            Path(__file__).resolve(),
            ["--sdk-dir", str(args.sdk_dir)],
        )

    if args.runtime_python is not None and not args._runtime:
        interpreter = args.runtime_python.expanduser().resolve()
        if not interpreter.is_file():
            print(f"ERROR: Python interpreter not found: {interpreter}", file=sys.stderr)
            return 2
        if not str(interpreter).startswith(_isaac_root() + os.sep):
            print(
                "ERROR: --runtime-python must be an interpreter under "
                f"ISAAC_SIM_ROOT ({_isaac_root()}); refusing a mismatched runtime.",
                file=sys.stderr,
            )
            return 2
        forwarded = ["--sdk-dir", str(args.sdk_dir)]
        return _run_with_interpreter(interpreter, Path(__file__).resolve(), forwarded)

    if not _has_host_network_interface():
        print(
            "ERROR: no host network interface is visible. Refusing to print an "
            "AnyGrasp feature ID from a sandbox, container, or restricted network namespace.",
            file=sys.stderr,
        )
        return 3

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
    if not re.fullmatch(r"N\d{20}", feature_id):
        print(
            "ERROR: AnyGrasp returned an unexpected feature-ID format; "
            "no ID was printed for registration.",
            file=sys.stderr,
        )
        return 1
    print(f"python={sys.executable}", file=sys.stderr)
    print(f"gsnet={source}", file=sys.stderr)
    print(f"feature_id={feature_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
