#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "${PROJECT_ROOT}/install/setup.bash"

if ! ros2 pkg prefix foxglove_bridge >/dev/null 2>&1; then
  cat >&2 <<'EOF'
foxglove_bridge is not installed in the sourced ROS 2 environment.
Install it with:
  sudo apt install ros-humble-foxglove-bridge
EOF
  exit 1
fi

echo "Starting AuraVLA Foxglove bridge"
echo "  WebSocket: ws://localhost:8765"
echo "  Topics:    /aura/anygrasp/* and /aura/transport_tracking"

# Match the RCIA-vision operational entry point while keeping AuraVLA's
# workspace and topic namespace independent.
exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml
