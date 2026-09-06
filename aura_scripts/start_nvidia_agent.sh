#!/bin/bash
# AuraVLA Complete System Launcher
# Starts NVIDIA Agent with full VLM, camera, and Isaac Sim integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/aura_bringup/config/config.yaml"
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

# API credentials must be injected through the environment, never read from
# a tracked configuration file.
if [ -z "$NVIDIA_API_KEY" ] && [ -z "$NVIDIA_API_KEYS" ]; then
    echo "⚠️  Warning: NVIDIA_API_KEY not set"
    echo "   Set NVIDIA_API_KEY or NVIDIA_API_KEYS before launching"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
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
