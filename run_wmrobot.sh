#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/experiment_common.sh"
ORIGINAL_ARGS=("$@")

SCENARIO="basic"
SOLVER="all"
OUT_DIR="results/wmrobot"
DO_BUILD=1
OVERWRITE=0
NO_VIZ=0
ROLLOUT_STEP="first"
MAX_ROLLOUTS=24
GIF_FRAMES=100
GIF_DURATION_MS=80
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_wmrobot.sh [options] [-- solver args]

WMRobot GPU examples.  Outputs include raw CSVs, aggregate timing/success CSV,
and rollout/cluster `vis_data` when the selected scenario emits it.

Options:
  --scenario NAME       basic or map_285. Default: basic
  --solver NAME         all, mppi, log_mppi, cluster_mppi, bi_mppi, svgd_mppi.
                        `map_285` supports all except svgd_mppi. Default: all
  --out DIR             Output directory. Default: results/wmrobot
  --overwrite           Replace output directory.
  --no-build            Use existing build/gpu binaries.
  --no-viz              Skip PNG/GIF generation for map_285.
  --rollout-step STEP   first, middle, last, or step_0000. Default: first
  --max-rollouts N      Max sampled rollout trajectories. Default: 24
  --gif-frames N        map_285 path GIF frame cap. Default: 100
  --gif-duration-ms N   map_285 GIF frame duration. Default: 80
  -h, --help            Show this help.

Examples:
  ./run_wmrobot.sh --scenario basic --solver all
  ./run_wmrobot.sh --scenario map_285 --overwrite

Arguments after `--` are forwarded to basic-scenario executables.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="$2"; shift 2 ;;
        --solver)
            SOLVER="$2"; shift 2 ;;
        --out)
            OUT_DIR="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
        --no-build)
            DO_BUILD=0; shift ;;
        --no-viz|--no-visualize)
            NO_VIZ=1; shift ;;
        --rollout-step)
            ROLLOUT_STEP="$2"; shift 2 ;;
        --max-rollouts)
            MAX_ROLLOUTS="$2"; shift 2 ;;
        --gif-frames)
            GIF_FRAMES="$2"; shift 2 ;;
        --gif-duration-ms)
            GIF_DURATION_MS="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; RUN_ARGS+=("$@"); break ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

solver_enabled() {
    [[ "$SOLVER" == "all" || "$SOLVER" == "$1" ]]
}

run_basic() {
    local out="$1"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot
    reset_build_outputs

    local specs=(
        "mppi:wmrobot_mppi"
        "log_mppi:wmrobot_log_mppi"
        "cluster_mppi:wmrobot_cluster_mppi"
        "bi_mppi:wmrobot_bi_mppi"
        # "svgd_mppi:wmrobot_svgd_mppi"
    )
    local ran=0
    for spec in "${specs[@]}"; do
        local name="${spec%%:*}"
        local exe="${spec#*:}"
        solver_enabled "$name" || continue
        [[ -x "$GPU_DIR/$exe" ]] || { echo "ERROR: missing executable: $GPU_DIR/$exe" >&2; exit 1; }
        echo "[run] $exe"
        (cd "$BUILD_DIR" && "./gpu/$exe" "${RUN_ARGS[@]}")
        ran=1
    done
    [[ "$ran" -eq 1 ]] || { echo "ERROR: no solver matched --solver $SOLVER" >&2; exit 1; }

    copy_build_csv_outputs "$out"
    copy_build_vis_data "$out"
}

run_map_285() {
    local out="$1"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot_map_285
    reset_build_outputs

    local specs=(
        "mppi:vis_mppi"
        "log_mppi:vis_log_mppi"
        "cluster_mppi:vis_cluster_mppi"
        "bi_mppi:vis_bi_mppi"
    )
    local ran=0
    for spec in "${specs[@]}"; do
        local name="${spec%%:*}"
        local exe="${spec#*:}"
        solver_enabled "$name" || continue
        [[ -x "$GPU_DIR/$exe" ]] || { echo "ERROR: missing executable: $GPU_DIR/$exe" >&2; exit 1; }
        echo "[run] $exe"
        (cd "$BUILD_DIR" && "./gpu/$exe")
        ran=1
    done
    [[ "$ran" -eq 1 ]] || { echo "ERROR: no solver matched --solver $SOLVER" >&2; exit 1; }

    copy_build_csv_outputs "$out"
    copy_build_vis_data "$out"

    if [[ "$NO_VIZ" -eq 0 ]]; then
        local vis_args=(
            --vis-root "$out/vis_data"
            --result-dir "$out/raw_csv"
            --out-dir "$out"
            --rollout-step "$ROLLOUT_STEP"
            --max-rollouts "$MAX_ROLLOUTS"
            --gif-frames "$GIF_FRAMES"
            --gif-duration-ms "$GIF_DURATION_MS"
        )
        python3 "$REPO_ROOT/visualize_wmrobot_map_285.py" "${vis_args[@]}"
    fi
}

OUT_ABS="$(repo_out_path "$OUT_DIR")"
prepare_output_dir "$OUT_ABS" "$OVERWRITE"
write_run_manifest "$OUT_ABS" "$0" "${ORIGINAL_ARGS[@]}"

case "$SCENARIO" in
    basic)
        run_basic "$OUT_ABS" ;;
    map_285)
        run_map_285 "$OUT_ABS" ;;
    *)
        echo "ERROR: unsupported WMRobot scenario: $SCENARIO" >&2
        usage >&2
        exit 1 ;;
esac

aggregate_timing_success_csv "$OUT_ABS" "$OUT_ABS/aggregate_timing_success.csv"
write_outputs_manifest "$OUT_ABS"

echo ""
echo "Outputs:"
echo "  $OUT_ABS"
echo "  $OUT_ABS/raw_csv/"
echo "  $OUT_ABS/aggregate_timing_success.csv"
if [[ -d "$OUT_ABS/vis_data" ]]; then
    echo "  $OUT_ABS/vis_data/"
fi
