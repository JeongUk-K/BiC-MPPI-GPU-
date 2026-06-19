#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/experiment_common.sh"
ORIGINAL_ARGS=("$@")

SOLVER="all"
OUT_DIR="results/manipulator"
OBSTACLE_FILE="obstacles/cylinder_test.txt"
DO_BUILD=1
OVERWRITE=0
NO_VIZ=0

usage() {
    cat <<'EOF'
Usage:
  ./run_manipulator.sh [options]

Manipulator GPU examples for the 3D cylinder avoidance task.  Outputs include
raw CSVs, aggregate timing/success CSV, MPPIVisLogger rollout/cluster data,
and optional rollout GIF / Plotly HTML visualizations when helper scripts exist.

Options:
  --solver NAME       all, mppi, cluster_mppi, bi_mppi, svgd_mppi, mppi_cylinder.
                      Default: all
  --obstacle FILE     Cylinder obstacle file. Default: obstacles/cylinder_test.txt
  --out DIR           Output directory. Default: results/manipulator
  --overwrite         Replace output directory.
  --no-build          Use existing build/gpu binaries.
  --no-viz            Skip GIF/HTML post-processing.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)
            SOLVER="$2"; shift 2 ;;
        --obstacle)
            OBSTACLE_FILE="$2"; shift 2 ;;
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
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

solver_enabled() {
    [[ "$SOLVER" == "all" || "$SOLVER" == "$1" ]]
}

OUT_ABS="$(repo_out_path "$OUT_DIR")"
prepare_output_dir "$OUT_ABS" "$OVERWRITE"
write_run_manifest "$OUT_ABS" "$0" "${ORIGINAL_ARGS[@]}"

[[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" manipulator
reset_build_outputs

RUN_OBSTACLE_FILE="../$OBSTACLE_FILE"
[[ "$OBSTACLE_FILE" = /* ]] && RUN_OBSTACLE_FILE="$OBSTACLE_FILE"

specs=(
    "mppi:mppi:manipulator_mppi"
    "cluster_mppi:cluster_mppi:manipulator_cluster_mppi"
    "bi_mppi:bi_mppi:manipulator_bi_mppi"
    "svgd_mppi:svgd_mppi:manipulator_svgd_mppi"
    "mppi_cylinder:mppi_cylinder:"
)

ran=0
mkdir -p "$OUT_ABS/visualizations"
for spec in "${specs[@]}"; do
    IFS=: read -r name exe vis_name <<<"$spec"
    solver_enabled "$name" || continue
    [[ -x "$GPU_DIR/$exe" ]] || { echo "ERROR: missing executable: $GPU_DIR/$exe" >&2; exit 1; }
    echo "[run] $exe"
    (cd "$BUILD_DIR" && "./gpu/$exe" "$RUN_OBSTACLE_FILE")
    ran=1

    if [[ "$NO_VIZ" -eq 0 && -n "$vis_name" ]]; then
        if [[ -f "$BUILD_DIR/visualize_manipulator_rollouts.py" && -d "$BUILD_DIR/vis_data/$vis_name" ]]; then
            (cd "$BUILD_DIR" && python3 visualize_manipulator_rollouts.py \
                "$vis_name" "$RUN_OBSTACLE_FILE" \
                --output "$OUT_ABS/visualizations/${vis_name}_rollouts.gif")
        fi
        local_csv="result_manipulator_gpu_${name}.csv"
        if [[ -f "$BUILD_DIR/visualize_manipulator_plotly.py" && -f "$BUILD_DIR/$local_csv" ]]; then
            (cd "$BUILD_DIR" && python3 visualize_manipulator_plotly.py \
                "$local_csv" "$RUN_OBSTACLE_FILE" \
                "$OUT_ABS/visualizations/${vis_name}_3d.html")
        fi
    fi
done
[[ "$ran" -eq 1 ]] || { echo "ERROR: no solver matched --solver $SOLVER" >&2; exit 1; }

copy_build_csv_outputs "$OUT_ABS"
copy_build_vis_data "$OUT_ABS"
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
