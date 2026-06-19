#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/experiment_common.sh"
ORIGINAL_ARGS=("$@")

SCENARIO="basic"
SOLVER="all"
OUT_DIR="results/quadrotor"
DO_BUILD=1
OVERWRITE=0
NO_VIZ=0
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_quadrotor.sh [options] [-- solver args]

Quadrotor GPU examples.  The basic scenario stores raw result/path CSVs and
can generate per-map path plots.  The precision_landing scenario runs the
`src/quadrotor/gpu new` examples.

Options:
  --scenario NAME    basic or precision_landing. Default: basic
  --solver NAME      all, mppi, log_mppi, cluster_mppi, bi_mppi, svgd_mppi.
                     Default: all
  --out DIR          Output directory. Default: results/quadrotor
  --overwrite        Replace output directory.
  --no-build         Use existing build/gpu binaries.
  --no-viz           Skip basic-scenario path plots.
  -h, --help         Show this help.

Examples:
  ./run_quadrotor.sh --scenario basic --solver all
  ./run_quadrotor.sh --scenario precision_landing -- --smoke
  ./run_quadrotor.sh --scenario basic --solver cluster_mppi -- --vis-every 5
  ./run_quadrotor.sh --scenario basic --solver all -- --no-rollouts

Arguments after `--` are forwarded to executables.
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
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" quadrotor
    reset_build_outputs

    local specs=(
        "mppi:quadrotor_mppi"
        "log_mppi:quadrotor_log_mppi"
        "cluster_mppi:quadrotor_cluster_mppi"
        "bi_mppi:quadrotor_bi_mppi"
        "svgd_mppi:quadrotor_svgd_mppi"
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

    if [[ "$NO_VIZ" -eq 0 ]]; then
        mkdir -p "$out/plots"
        python3 "$REPO_ROOT/plot_quadrotor_gpu_paths.py" \
            --build-dir "$BUILD_DIR" \
            --map-dir "$REPO_ROOT/BARN_dataset/txt_files" \
            --output-dir "$out/plots"
    fi
}

run_precision_landing() {
    local out="$1"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" quadrotor_gpu_new
    reset_build_outputs

    local specs=(
        "mppi:quadrotor_gpu_new_mppi"
        "log_mppi:quadrotor_gpu_new_log_mppi"
        "cluster_mppi:quadrotor_gpu_new_cluster_mppi"
        "bi_mppi:quadrotor_gpu_new_bi_mppi"
        "svgd_mppi:quadrotor_gpu_new_svgd_mppi"
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

OUT_ABS="$(repo_out_path "$OUT_DIR")"
prepare_output_dir "$OUT_ABS" "$OVERWRITE"
write_run_manifest "$OUT_ABS" "$0" "${ORIGINAL_ARGS[@]}"

case "$SCENARIO" in
    basic)
        run_basic "$OUT_ABS" ;;
    precision_landing)
        run_precision_landing "$OUT_ABS" ;;
    *)
        echo "ERROR: unsupported quadrotor scenario: $SCENARIO" >&2
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
