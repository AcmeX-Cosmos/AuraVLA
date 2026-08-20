#!/usr/bin/env bash
# Start the complete AuraVLA Isaac robot runtime.
#
# This injects the AuraVLA robot runtime into the already-running Isaac Sim
# VS Code Edition executor. It never starts a second Isaac Sim process.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CAMERA_DIR="${AURA_CAMERA_DIR:-/tmp/aura-vla-camera}"
TASK_DIR="${AURA_VLA_TASK_DIR:-/tmp/aura-vla-control}"
URDF_PATH="${AURA_TRON2_URDF_PATH:-$PROJECT_ROOT/aura_description/urdf/tron2_v5_DACH_validing/robot.urdf}"
ROBOT_ENTRY="$PROJECT_ROOT/aura_hardware/aura_isaac_bridge/robot/robot.py"
EXECUTOR_CLIENT="$PROJECT_ROOT/aura_scripts/isaac_vscode_run.py"

if [[ ! -f "$ROBOT_ENTRY" ]]; then
    echo "ERROR: AuraVLA robot runtime not found: $ROBOT_ENTRY" >&2
    exit 1
fi
if [[ ! -f "$EXECUTOR_CLIENT" ]]; then
    echo "ERROR: VS Code executor client not found: $EXECUTOR_CLIENT" >&2
    exit 1
fi
if [[ ! -f "$URDF_PATH" ]]; then
    echo "ERROR: DACH TRON2A URDF not found: $URDF_PATH" >&2
    echo "Set AURA_TRON2_URDF_PATH to the robot.urdf path." >&2
    exit 1
fi

export AURA_VLA_ROOT="$PROJECT_ROOT"
export AURA_TRON2_URDF_PATH="$URDF_PATH"
export AURA_CAMERA_DIR="$CAMERA_DIR"
export AURA_VLA_TASK_DIR="$TASK_DIR"

mkdir -p "$CAMERA_DIR" "$TASK_DIR"

echo "Starting AuraVLA Isaac robot runtime"
echo "  Project root: $PROJECT_ROOT"
echo "  Camera directory: $CAMERA_DIR"
echo "  Task directory: $TASK_DIR"
echo "  Entry point: $ROBOT_ENTRY"
echo
echo "Isaac Sim process: existing VS Code Edition instance"
echo "  Executor: ${AURA_ISAAC_HOST:-127.0.0.1}:${AURA_ISAAC_PORT:-8226}"
echo "  Runtime source: $ROBOT_ENTRY"
echo "  DACH URDF: $URDF_PATH"

exec /usr/bin/python3 "$EXECUTOR_CLIENT" "$ROBOT_ENTRY" "$@"
