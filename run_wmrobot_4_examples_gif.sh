#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runs MPPI, Log-MPPI, Cluster-MPPI, and BiC-MPPI on WMRobot map 285,
# then writes the PNG summaries and path GIF.
exec bash "$SCRIPT_DIR/run_wmrobot_map_285_visualization.sh" "$@"
