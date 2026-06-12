#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD=1
VISUALIZE=1
OVERWRITE=0
OUT_DIR="results/wmrobot_waypoint_stitched"
ROLLOUT_SCENARIO="0"
ROLLOUT_STEP="first"
MAX_ROLLOUTS="24"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)
            BUILD=0
            shift
            ;;
        --no-viz|--no-visualize)
            VISUALIZE=0
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --out)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --out requires a directory." >&2
                exit 1
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        --rollout-scenario)
            ROLLOUT_SCENARIO="$2"
            shift 2
            ;;
        --rollout-step)
            ROLLOUT_STEP="$2"
            shift 2
            ;;
        --max-rollouts)
            MAX_ROLLOUTS="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
Usage: ./run_wmrobot_waypoint_stitched.sh [options] [solver options]

Common options:
  --overwrite              Remove existing output before running.
  --no-build               Use existing build/gpu/wmrobot_waypoint_* binaries.
  --no-viz                 Skip PNG visualization generation.
  --out DIR                Output directory. Default: results/wmrobot_waypoint_stitched
  --rollout-scenario N     Scenario used for rollout panel. Default: 0
  --rollout-step STEP      first, middle, last, or step_0000. Default: first
  --max-rollouts N         Max sampled rollout trajectories per group. Default: 24

Forwarded solver options:
  Default parameters live in src/wmrobot/gpu_waypoint/stitched_barn_params.h
  --smoke                  One short scenario for validation.
  --scenarios N            Default: 10
  --maps-per-scenario N    Default: 5
  --seed N                 Random map-selection seed.
  --global-T N             MPPI/Log/Cluster global-goal horizon. Default: 500
  --waypoint-count N       BiC waypoint count.
  --bic-Tf N --bic-Tb N    BiC forward/backward rollout horizon. Default: 50/50
  --N N --Nf N --Nb N --Nr N --maxiter N
  --vis-every N            Save rollout data every N closed-loop iterations.
  --no-rollouts            Save CSV paths only.
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

if [[ "$BUILD" -eq 1 ]]; then
    bash build_gpu.sh wmrobot_waypoint_stitched
fi

EXES=(
    wmrobot_waypoint_mppi
    wmrobot_waypoint_log_mppi
    wmrobot_waypoint_cluster_mppi
    wmrobot_waypoint_bi_mppi
)

for exe in "${EXES[@]}"; do
    if [[ ! -x "build/gpu/$exe" ]]; then
        echo "ERROR: build/gpu/$exe not found. Run without --no-build first." >&2
        exit 1
    fi
done

if [[ "$OUT_DIR" = /* ]]; then
    BUILD_OUT="$OUT_DIR"
else
    BUILD_OUT="../$OUT_DIR"
fi

rm -rf build/vis_data/stitched_*_scenario_*

for exe in "${EXES[@]}"; do
    echo "[run] $exe"
    (cd build && "./gpu/$exe" --out "$BUILD_OUT" "${EXTRA_ARGS[@]}")
done

rm -rf "$OUT_DIR/vis_data"
mkdir -p "$OUT_DIR/vis_data"
if compgen -G "build/vis_data/stitched_*_scenario_*" >/dev/null; then
    find build/vis_data -maxdepth 1 -type d -name 'stitched_*_scenario_*' \
        -exec cp -R {} "$OUT_DIR/vis_data/" \;
fi

if [[ "$VISUALIZE" -eq 1 ]]; then
    python3 visualize_wmrobot_waypoint_stitched.py \
        --dir "$OUT_DIR" \
        --rollout-scenario "$ROLLOUT_SCENARIO" \
        --rollout-step "$ROLLOUT_STEP" \
        --max-rollouts "$MAX_ROLLOUTS"
fi

echo ""
echo "Outputs:"
echo "  $OUT_DIR"
if [[ "$VISUALIZE" -eq 1 ]]; then
    echo "  $OUT_DIR/wmrobot_stitched_aggregate.csv"
    echo "  $OUT_DIR/wmrobot_stitched_timing.png"
    echo "  $OUT_DIR/wmrobot_stitched_paths_scenario_00.png"
    echo "  $OUT_DIR/wmrobot_stitched_rollouts_scenario_$(printf '%02d' "$ROLLOUT_SCENARIO").png"
else
    echo "  visualization skipped (--no-viz)"
fi
