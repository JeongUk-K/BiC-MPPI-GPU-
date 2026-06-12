#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD=1
OVERWRITE=0
OUT_DIR="results/quadrotor_gpu_log_mppi"
RUN_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-build)
            BUILD=0
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --out)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --out requires a directory" >&2
                exit 1
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
Usage: ./run_quadrotor_gpu_log_mppi.sh [script options] [Log-MPPI options]

Script options:
  --no-build       Skip build step and run existing build/gpu/quadrotor_log_mppi
  --out DIR        Directory to receive quadrotor Log-MPPI CSV files
  --overwrite      Replace existing CSV files in OUT DIR

Common Log-MPPI options:
  --smoke          Tiny validation run
  --num-maps N     Number of BARN maps from map_begin downward
  --maxiter N      Max closed-loop iterations per run
USAGE
            exit 0
            ;;
        *)
            RUN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$BUILD" -eq 1 ]; then
    bash build_gpu.sh quadrotor_log_mppi
fi

if [ ! -x build/gpu/quadrotor_log_mppi ]; then
    echo "ERROR: build/gpu/quadrotor_log_mppi not found. Run without --no-build first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

RESULT_OUT="$OUT_DIR/result_quadrotor_log_mppi.csv"
SUMMARY_OUT="$OUT_DIR/result_quadrotor_log_mppi_summary.csv"
PATH_OUT="$OUT_DIR/path_quadrotor_log_mppi.csv"
if [ "$OVERWRITE" -ne 1 ] && {
    [ -e "$RESULT_OUT" ] || [ -e "$SUMMARY_OUT" ] || [ -e "$PATH_OUT" ]
}; then
    echo "ERROR: output CSV already exists in $OUT_DIR. Pass --overwrite to replace." >&2
    exit 1
fi

rm -f build/result_quadrotor_log_mppi.csv \
      build/result_quadrotor_log_mppi_summary.csv \
      build/path_quadrotor_log_mppi.csv

(
    cd build
    ./gpu/quadrotor_log_mppi "${RUN_ARGS[@]}"
)

if [ ! -f build/result_quadrotor_log_mppi.csv ] || \
   [ ! -f build/result_quadrotor_log_mppi_summary.csv ] || \
   [ ! -f build/path_quadrotor_log_mppi.csv ]; then
    echo "ERROR: quadrotor_log_mppi did not produce expected CSV files." >&2
    exit 1
fi

cp build/result_quadrotor_log_mppi.csv "$RESULT_OUT"
cp build/result_quadrotor_log_mppi_summary.csv "$SUMMARY_OUT"
cp build/path_quadrotor_log_mppi.csv "$PATH_OUT"

echo "CSV results written:"
echo "  $RESULT_OUT"
echo "  $SUMMARY_OUT"
echo "  $PATH_OUT"
