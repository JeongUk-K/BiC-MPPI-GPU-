#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD=1
OVERWRITE=0
OUT_DIR="results/quadrotor_gpu_new"
ONLY="all"
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
        --solver)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --solver requires one of all,mppi,log_mppi,cluster_mppi,bi_mppi,svgd_mppi" >&2
                exit 1
            fi
            ONLY="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
Usage: ./run_quadrotor_gpu_new.sh [script options] [experiment options]

Script options:
  --no-build       Skip build step
  --out DIR        Directory to receive result/path/summary/all-summary CSV files
  --overwrite      Replace existing CSV files in OUT DIR
  --solver NAME    all, mppi, log_mppi, cluster_mppi, bi_mppi, or svgd_mppi

Common experiment options:
  --smoke          Tiny validation run
  --num-maps N     Number of BARN maps from map_begin downward
  --maxiter N      Max closed-loop iterations per map
  --T N            One-direction horizon
  --Tf N --Tb N    Bidirectional horizons
  --N N            One-direction rollout samples
  --Nf N --Nb N    Bidirectional rollout samples
  --Nr N           Guide rollout samples
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
    bash build_gpu.sh quadrotor_gpu_new
fi

ALL_SOLVERS=(mppi log_mppi cluster_mppi bi_mppi svgd_mppi)
if [ "$ONLY" = "all" ]; then
    SOLVERS=("${ALL_SOLVERS[@]}")
else
    case "$ONLY" in
        mppi|log_mppi|cluster_mppi|bi_mppi|svgd_mppi)
            SOLVERS=("$ONLY")
            ;;
        *)
            echo "ERROR: unknown solver: $ONLY" >&2
            exit 1
            ;;
    esac
fi

mkdir -p "$OUT_DIR"
aggregate_summary_out="$OUT_DIR/result_quadrotor_all_summary.csv"

if [ "$OVERWRITE" -ne 1 ] && [ -e "$aggregate_summary_out" ]; then
    echo "ERROR: aggregate summary CSV already exists in $OUT_DIR. Pass --overwrite." >&2
    exit 1
fi

for solver in "${SOLVERS[@]}"; do
    bin="build/gpu/quadrotor_gpu_new_${solver}"
    if [ ! -x "$bin" ]; then
        echo "ERROR: $bin not found. Run without --no-build first." >&2
        exit 1
    fi

    result_out="$OUT_DIR/result_quadrotor_${solver}.csv"
    path_out="$OUT_DIR/path_quadrotor_${solver}.csv"
    summary_out="$OUT_DIR/result_quadrotor_${solver}_summary.csv"

    if [ "$OVERWRITE" -ne 1 ] && {
        [ -e "$result_out" ] || [ -e "$path_out" ] || [ -e "$summary_out" ]
    }; then
        echo "ERROR: output CSV for $solver already exists in $OUT_DIR. Pass --overwrite." >&2
        exit 1
    fi
done

for solver in "${SOLVERS[@]}"; do
    echo "=== Running quadrotor gpu new: $solver ==="
    rm -f "build/result_quadrotor_${solver}.csv" \
          "build/path_quadrotor_${solver}.csv" \
          "build/result_quadrotor_${solver}_summary.csv"

    (
        cd build
        "./gpu/quadrotor_gpu_new_${solver}" "${RUN_ARGS[@]}"
    )

    for kind in result path; do
        src="build/${kind}_quadrotor_${solver}.csv"
        dst="$OUT_DIR/${kind}_quadrotor_${solver}.csv"
        if [ ! -f "$src" ]; then
            echo "ERROR: missing expected CSV: $src" >&2
            exit 1
        fi
        cp "$src" "$dst"
    done

    src="build/result_quadrotor_${solver}_summary.csv"
    dst="$OUT_DIR/result_quadrotor_${solver}_summary.csv"
    if [ ! -f "$src" ]; then
        echo "ERROR: missing expected CSV: $src" >&2
        exit 1
    fi
    cp "$src" "$dst"
done

first_summary=1
rm -f "$aggregate_summary_out"
for solver in "${SOLVERS[@]}"; do
    summary="$OUT_DIR/result_quadrotor_${solver}_summary.csv"
    if [ "$first_summary" -eq 1 ]; then
        head -n 1 "$summary" > "$aggregate_summary_out"
        first_summary=0
    fi
    tail -n +2 "$summary" >> "$aggregate_summary_out"
done

echo "CSV results written to $OUT_DIR"
echo "Aggregate summary written to $aggregate_summary_out"
