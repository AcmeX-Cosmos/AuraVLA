#!/bin/bash
# AuraVLA Complete System Launcher
# Starts NVIDIA Agent with full VLM, camera, and Isaac Sim integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/aura_bringup/config/config.yaml"
NVIDIA_ENV_FILE="${NVIDIA_ENV_FILE:-$PROJECT_ROOT/aura_bringup/config/nvidia.local.env}"
export AURA_VLA_ROOT="$PROJECT_ROOT"

echo "============================================================"
echo "AuraVLA - NVIDIA Agent Interactive Mode"
echo "============================================================"
echo ""

# Check if workspace is sourced
if [ -z "$AMENT_PREFIX_PATH" ]; then
    echo "❌ ROS 2 workspace not sourced!"
    echo ""
    echo "Please run first:"
    echo "  cd $PROJECT_ROOT"
    echo "  source install/setup.bash"
    echo ""
    exit 1
fi

# Deactivate conda if active
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "ℹ️  Conda environment detected: $CONDA_DEFAULT_ENV"
    echo "   Deactivating to use system Python for ROS 2..."
    eval "$(conda shell.bash hook)"
    conda deactivate
    echo ""
fi

# Load credentials only from the optional, git-ignored local environment file.
# The tracked config.yaml is intentionally never used for secrets.
if [ -f "$NVIDIA_ENV_FILE" ]; then
    # Parse only the two supported variable names. Do not source this file:
    # malformed or pasted continuation lines must never be executed as shell.
    while IFS= read -r ENV_LINE || [ -n "$ENV_LINE" ]; do
        ENV_LINE="${ENV_LINE%$'\r'}"
        if [[ "$ENV_LINE" =~ ^[[:space:]]*(#.*)?$ ]]; then
            continue
        fi
        case "$ENV_LINE" in
            NVIDIA_API_KEY=*)
                NVIDIA_API_KEY="${NVIDIA_API_KEY}${ENV_LINE#NVIDIA_API_KEY=}"
                ;;
            NVIDIA_API_KEYS=*)
                NVIDIA_API_KEYS="${ENV_LINE#NVIDIA_API_KEYS=}"
                ;;
            Bearer\ *|nvapi-*)
                # Accept wrapped multi-key values pasted across lines.
                NVIDIA_API_KEYS="${NVIDIA_API_KEYS}${ENV_LINE}"
                ;;
            *)
                echo "ERROR: unsupported entry in $NVIDIA_ENV_FILE" >&2
                exit 1
                ;;
        esac
    done < "$NVIDIA_ENV_FILE"
    NVIDIA_API_KEYS="${NVIDIA_API_KEYS//\\/}"
    NVIDIA_API_KEYS="${NVIDIA_API_KEYS#\"}"
    NVIDIA_API_KEYS="${NVIDIA_API_KEYS%\"}"
    NVIDIA_API_KEYS="${NVIDIA_API_KEYS#\'}"
    NVIDIA_API_KEYS="${NVIDIA_API_KEYS%\'}"
    NVIDIA_API_KEY="${NVIDIA_API_KEY#\"}"
    NVIDIA_API_KEY="${NVIDIA_API_KEY%\"}"
    NVIDIA_API_KEY="${NVIDIA_API_KEY#\'}"
    NVIDIA_API_KEY="${NVIDIA_API_KEY%\'}"
fi

if [ -n "$NVIDIA_API_KEY" ]; then
    export NVIDIA_API_KEY
fi
if [ -n "$NVIDIA_API_KEYS" ]; then
    export NVIDIA_API_KEYS
fi

# API credentials must be injected through the environment, never read from
# a tracked configuration file.
if [ -z "$NVIDIA_API_KEY" ] && [ -z "$NVIDIA_API_KEYS" ]; then
    echo "ERROR: NVIDIA credentials are missing in $NVIDIA_ENV_FILE" >&2
    exit 1
fi

echo "🚀 Starting NVIDIA Agent..."
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Camera dir:  /tmp/aura-vla-camera"
echo "  Task dir:    /tmp/aura-vla-control"
echo ""
echo "Features:"
echo "  ✓ Natural language understanding (Chinese/English)"
echo "  ✓ RGBD camera integration"
echo "  ✓ Scene analysis with VLM"
echo "  ✓ Direct Isaac Sim control"
echo ""
echo "Quick commands:"
echo "  1 - 把香蕉放进篮子里"
echo "  2 - 把杯子放进篮子里"
echo "  3 - 把蓝色罐头放进篮子里"
echo "  7 - 现在画面中有什么物体"
echo ""
echo "Special commands:"
echo "  /reload  - Reload VLM model"
echo "  /camera  - Refresh camera frame"
echo "  /help    - Show all commands"
echo ""
echo "Type your instruction or number, Ctrl+C to quit"
echo "============================================================"
echo ""

# Run nvidia_agent in chat mode with execution enabled
exec /usr/bin/python3 -m aura_perception.nvidia_agent \
    --config "$CONFIG_FILE" \
    --chat \
    --execute \
    "$@"
