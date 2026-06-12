#!/usr/bin/env bash
# Build and run the WMRobot GPU same-budget sampling scaling experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
GPU_BIN="$BUILD_DIR/gpu/wmrobot_same_budget_all"
DO_BUILD=1
FORWARD_ARGS=()
OUT_DIR=""
HOST_OUT_DIR=""

usage() {
    cat <<'EOF'
Usage:
  ./run_wmrobot_gpu_same_budget.sh [options]

Common options:
  --smoke                         Run a tiny validation experiment.
  --no-build                      Skip build and run the existing binary.
  --overwrite                     Replace existing output CSV files.
  --out DIR                       Output directory. Default: results/sampling_budget_scaling
  --budgets 1000 3000 6000        Budget levels to run.
  --include-B12000                Also run B12000.
  --num-maps N                    Number of maps from map_begin downward.
  --start-cases N                 Number of start states per map.
  --maxiter N                     Max closed-loop iterations.

Outputs:
  results/sampling_budget_scaling/raw_trials.csv
  results/sampling_budget_scaling/summary.csv
  results/sampling_budget_scaling/metadata.json
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --no-build)
            DO_BUILD=0
            shift
            ;;
        --out)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --out requires a value" >&2
                exit 1
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    FORWARD_ARGS+=("--out" "../results/sampling_budget_scaling")
    HOST_OUT_DIR="$SCRIPT_DIR/results/sampling_budget_scaling"
elif [[ "$OUT_DIR" = /* ]]; then
    FORWARD_ARGS+=("--out" "$OUT_DIR")
    HOST_OUT_DIR="$OUT_DIR"
else
    FORWARD_ARGS+=("--out" "../$OUT_DIR")
    HOST_OUT_DIR="$SCRIPT_DIR/$OUT_DIR"
fi

GIT_COMMIT="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
FORWARD_ARGS+=("--git-commit" "$GIT_COMMIT")

if [[ "$DO_BUILD" -eq 1 ]]; then
    bash "$SCRIPT_DIR/build_gpu.sh" wmrobot_same_budget
fi

if [[ ! -x "$GPU_BIN" ]]; then
    echo "ERROR: executable not found: $GPU_BIN" >&2
    echo "Run: bash build_gpu.sh wmrobot_same_budget" >&2
    exit 1
fi

cd "$BUILD_DIR"
"$GPU_BIN" "${FORWARD_ARGS[@]}"

echo ""
echo "=== WMRobot GPU same-budget experiment completed ==="
if [[ -f "$HOST_OUT_DIR/summary.csv" ]]; then
    echo "Summary:"
    echo "  $HOST_OUT_DIR/summary.csv"
fi
