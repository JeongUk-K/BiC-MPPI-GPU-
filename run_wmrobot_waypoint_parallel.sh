#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD=1
VISUALIZE=1
SAMPLE_GIF=0
ARGS=()
OUT_DIR="../results/wmrobot_waypoint_parallel"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-build)
            BUILD=0
            shift
            ;;
        --no-viz|--no-visualize)
            VISUALIZE=0
            shift
            ;;
        --sample-gif|--sampling-gif|--animate-samples)
            SAMPLE_GIF=1
            ARGS+=("--vis-samples")
            shift
            ;;
        --vis-samples)
            SAMPLE_GIF=1
            ARGS+=("$1")
            shift
            ;;
        --out)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --out requires a directory." >&2
                exit 1
            fi
            OUT_DIR="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
Usage: ./run_wmrobot_waypoint_parallel.sh [--no-build] [--no-viz] [--sample-gif] [comparison options]

Common options:
  --smoke                 Small validation run
  --overwrite             Replace existing outputs
  --out DIR               Output directory
  --serial-segments       Disable segment-level std::async execution
  --no-viz                Skip PNG visualization generation
  --sample-gif            Export sampled paths and generate frame PNGs + GIF

The script builds, runs, and then generates:
  wmrobot_waypoint_paths.png
  wmrobot_waypoint_timing.png
  wmrobot_waypoint_runtime_breakdown.png

With --sample-gif it also generates:
  sampling_frames/frame_*.png
  wmrobot_waypoint_sampling.gif

All other options are forwarded to build/gpu/wmrobot_waypoint_comparison.
USAGE
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$BUILD" -eq 1 ]; then
    bash build_gpu.sh wmrobot_waypoint
fi

if [ ! -x build/gpu/wmrobot_waypoint_comparison ]; then
    echo "ERROR: build/gpu/wmrobot_waypoint_comparison not found. Run without --no-build first." >&2
    exit 1
fi

if [ "$SAMPLE_GIF" -eq 1 ]; then
    rm -rf build/vis_data/wmrobot_waypoint_*
fi

cd build
./gpu/wmrobot_waypoint_comparison "${ARGS[@]}"

if [ "$VISUALIZE" -eq 1 ]; then
    if [[ "$OUT_DIR" = /* ]]; then
        VIZ_DIR="$OUT_DIR"
    else
        VIZ_DIR="$SCRIPT_DIR/build/$OUT_DIR"
    fi

    cd "$SCRIPT_DIR"
    python3 visualize_wmrobot_waypoint_parallel.py --dir "$VIZ_DIR"
fi

if [ "$SAMPLE_GIF" -eq 1 ]; then
    VIS_ROOT="$SCRIPT_DIR/build/vis_data"
    SAMPLE_VIS_DIR="$VIZ_DIR/sample_vis_data"
    rm -rf "$SAMPLE_VIS_DIR"
    mkdir -p "$SAMPLE_VIS_DIR"
    find "$VIS_ROOT" -maxdepth 1 -type d -name 'wmrobot_waypoint_*' -exec cp -a {} "$SAMPLE_VIS_DIR/" \;

    cd "$SCRIPT_DIR"
    python3 animate_wmrobot_waypoint_sampling.py --dir "$VIZ_DIR" --vis-root "$SAMPLE_VIS_DIR"
fi
