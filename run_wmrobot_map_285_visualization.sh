#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="results/wmrobot_map_285"
NO_BUILD=0
OVERWRITE=0
ROLLOUT_STEP="first"
MAX_ROLLOUTS=24
GIF_FRAMES=100
GIF_DURATION_MS=80
NO_GIF=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)
            NO_BUILD=1
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --out)
            OUT_DIR="$2"
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
        --gif-frames)
            GIF_FRAMES="$2"
            shift 2
            ;;
        --gif-duration-ms)
            GIF_DURATION_MS="$2"
            shift 2
            ;;
        --no-gif)
            NO_GIF=1
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: ./run_wmrobot_map_285_visualization.sh [options]

Runs the four WMRobot map 285 examples:
  MPPI, Log-MPPI, Cluster-MPPI, BiC-MPPI

Options:
  --no-build           Run existing build/gpu/vis_* binaries.
  --overwrite          Remove the output directory before writing results.
  --out DIR            Output directory. Default: results/wmrobot_map_285
  --rollout-step STEP  first, middle, last, or step_0000. Default: first
  --max-rollouts N     Max sampled trajectories per rollout group. Default: 24
  --gif-frames N       Max frames for path GIF. Default: 100
  --gif-duration-ms N  GIF frame duration in milliseconds. Default: 80
  --no-gif             Skip GIF generation.
EOF
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

if [[ "$NO_BUILD" -eq 0 ]]; then
    bash build_gpu.sh wmrobot_map_285
fi

for exe in vis_mppi vis_log_mppi vis_cluster_mppi vis_bi_mppi; do
    if [[ ! -x "build/gpu/$exe" ]]; then
        echo "ERROR: build/gpu/$exe not found. Run without --no-build first." >&2
        exit 1
    fi
done

rm -rf build/vis_data/mppi \
       build/vis_data/log_mppi \
       build/vis_data/cluster_mppi \
       build/vis_data/bi_mppi
rm -f build/result_mppi.csv \
      build/result_log_mppi.csv \
      build/result_cluster_mppi.csv \
      build/result_bi_mppi.csv

echo "[run] MPPI"
(cd build && ./gpu/vis_mppi)

echo "[run] Log-MPPI"
(cd build && ./gpu/vis_log_mppi)

echo "[run] Cluster-MPPI"
(cd build && ./gpu/vis_cluster_mppi)

echo "[run] BiC-MPPI"
(cd build && ./gpu/vis_bi_mppi)

rm -rf "$OUT_DIR/vis_data"
mkdir -p "$OUT_DIR/vis_data"
for solver in mppi log_mppi cluster_mppi bi_mppi; do
    if [[ ! -d "build/vis_data/$solver" ]]; then
        echo "ERROR: build/vis_data/$solver was not generated." >&2
        exit 1
    fi
    cp -R "build/vis_data/$solver" "$OUT_DIR/vis_data/"
done

for csv in result_mppi.csv result_log_mppi.csv result_cluster_mppi.csv result_bi_mppi.csv; do
    if [[ ! -f "build/$csv" ]]; then
        echo "ERROR: build/$csv was not generated." >&2
        exit 1
    fi
    cp "build/$csv" "$OUT_DIR/$csv"
done

VIS_ARGS=(
    --vis-root "$OUT_DIR/vis_data" \
    --result-dir "$OUT_DIR" \
    --out-dir "$OUT_DIR" \
    --rollout-step "$ROLLOUT_STEP" \
    --max-rollouts "$MAX_ROLLOUTS" \
    --gif-frames "$GIF_FRAMES" \
    --gif-duration-ms "$GIF_DURATION_MS"
)

if [[ "$NO_GIF" -eq 1 ]]; then
    VIS_ARGS+=(--no-gif)
fi

python3 visualize_wmrobot_map_285.py "${VIS_ARGS[@]}"

echo ""
echo "Outputs:"
echo "  $OUT_DIR/wmrobot_map_285_summary.csv"
echo "  $OUT_DIR/wmrobot_map_285_paths.png"
echo "  $OUT_DIR/wmrobot_map_285_timing.png"
echo "  $OUT_DIR/wmrobot_map_285_rollouts.png"
if [[ "$NO_GIF" -eq 0 ]]; then
    echo "  $OUT_DIR/wmrobot_map_285_paths.gif"
    echo "  $OUT_DIR/solver_gifs/wmrobot_map_285_{mppi,log_mppi,cluster_mppi,bi_mppi}_paths.gif"
fi
