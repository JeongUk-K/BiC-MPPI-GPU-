#!/usr/bin/env bash
# Build and run all WMRobot GPU ablation examples, then collect one summary CSV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
GPU_BIN="$BUILD_DIR/gpu/wmrobot_ablation_gpu_all"
DEFAULT_OUTPUT_PREFIX="wmrobot_ablation_gpu"
LATEST_SUMMARY="$BUILD_DIR/wmrobot_ablation_gpu_all_results.csv"

DO_BUILD=1
OUTPUT_PREFIX=""
FORWARD_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash run_wmrobot_ablation_gpu.sh [options passed to wmrobot_ablation_gpu_all]

Common options:
  --smoke                 Run a small quick check.
  --output-prefix PREFIX  Prefix for generated CSV files. Default: wmrobot_ablation_gpu
  --no-build              Skip build and only run the existing binary.

Examples:
  bash run_wmrobot_ablation_gpu.sh --smoke
  bash run_wmrobot_ablation_gpu.sh
  bash run_wmrobot_ablation_gpu_smoke_all.sh
  bash run_wmrobot_ablation_gpu_full_all.sh

Outputs:
  build/<PREFIX>_summary.csv
  build/<PREFIX>_<variant>_runs.csv
  build/<PREFIX>_<variant>_summary.csv
  build/wmrobot_ablation_gpu_all_results.csv
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
        --output-prefix)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --output-prefix requires a value" >&2
                exit 1
            fi
            OUTPUT_PREFIX="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$OUTPUT_PREFIX" ]]; then
    OUTPUT_PREFIX="$DEFAULT_OUTPUT_PREFIX"
    FORWARD_ARGS+=("--output-prefix" "$OUTPUT_PREFIX")
fi

if [[ "$OUTPUT_PREFIX" = /* ]]; then
    OUTPUT_PREFIX_PATH="$OUTPUT_PREFIX"
else
    OUTPUT_PREFIX_PATH="$BUILD_DIR/$OUTPUT_PREFIX"
fi
mkdir -p "$(dirname "$OUTPUT_PREFIX_PATH")"

if [[ "$DO_BUILD" -eq 1 ]]; then
    bash "$SCRIPT_DIR/build_gpu.sh" wmrobot_ablation
fi

if [[ ! -x "$GPU_BIN" ]]; then
    echo "ERROR: executable not found: $GPU_BIN" >&2
    echo "Run: bash build_gpu.sh wmrobot_ablation" >&2
    exit 1
fi

cd "$BUILD_DIR"
"$GPU_BIN" "${FORWARD_ARGS[@]}"

SUMMARY_CSV="${OUTPUT_PREFIX_PATH}_summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "ERROR: expected summary CSV was not created: $SUMMARY_CSV" >&2
    exit 1
fi

cp "$SUMMARY_CSV" "$LATEST_SUMMARY"

echo ""
echo "=== WMRobot GPU ablation completed ==="
echo "Aggregate summary:"
echo "  $SUMMARY_CSV"
echo "Latest combined results copy:"
echo "  $LATEST_SUMMARY"
