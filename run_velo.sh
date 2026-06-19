#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/experiment_common.sh"
ORIGINAL_ARGS=("$@")

SOLVER="all"
OUT_DIR="results/velo"
DO_BUILD=1
OVERWRITE=0
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_velo.sh [options] [-- solver args]

Velo GPU examples. Outputs include raw CSVs and aggregate timing/success CSV.

Options:
  --solver NAME    all, mppi, log_mppi, cluster_mppi, bi_mppi, svgd_mppi.
                   Default: all
  --out DIR        Output directory. Default: results/velo
  --overwrite      Replace output directory.
  --no-build       Use existing build/gpu binaries.
  -h, --help       Show this help.

Arguments after `--` are forwarded to executables.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)
            SOLVER="$2"; shift 2 ;;
        --out)
            OUT_DIR="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
        --no-build)
            DO_BUILD=0; shift ;;
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

OUT_ABS="$(repo_out_path "$OUT_DIR")"
prepare_output_dir "$OUT_ABS" "$OVERWRITE"
write_run_manifest "$OUT_ABS" "$0" "${ORIGINAL_ARGS[@]}"

[[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" velo
reset_build_outputs

specs=(
    "mppi:velo_mppi"
    "log_mppi:velo_log_mppi"
    "cluster_mppi:velo_cluster_mppi"
    "bi_mppi:velo_bi_mppi"
    "svgd_mppi:velo_svgd_mppi"
)

ran=0
for spec in "${specs[@]}"; do
    name="${spec%%:*}"
    exe="${spec#*:}"
    solver_enabled "$name" || continue
    [[ -x "$GPU_DIR/$exe" ]] || { echo "ERROR: missing executable: $GPU_DIR/$exe" >&2; exit 1; }
    echo "[run] $exe"
    (cd "$BUILD_DIR" && "./gpu/$exe" "${RUN_ARGS[@]}")
    ran=1
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
