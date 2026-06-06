#!/usr/bin/env bash
# Run all WMRobot GPU ablation variants with the smoke configuration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_wmrobot_ablation_gpu.sh" \
  --smoke \
  --output-prefix wmrobot_ablation_gpu_smoke \
  "$@"
