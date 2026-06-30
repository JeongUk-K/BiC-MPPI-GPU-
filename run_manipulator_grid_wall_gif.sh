#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD=1
FPS=12
MAX_FRAMES=120
OUT_DIR=""
EXAMPLE_ARGS=()

usage() {
    cat <<'EOF'
usage: ./run_manipulator_grid_wall_gif.sh [script options] [example options]

Script options:
  --skip-build       Reuse the existing build/gpu/manipulator_grid_approach_workspace_example
  --fps N            GIF frame rate. Default: 12
  --max-frames N     Maximum sampled animation frames. Default: 120
  --out-dir DIR      GIF output directory. Default: build/manipulator_grid_approach_gifs
  -h, --help         Show this help

Example options are passed to manipulator_grid_approach_workspace_example:
  --seed N
  --goal CELL_ID

Examples:
  ./run_manipulator_grid_wall_gif.sh
  ./run_manipulator_grid_wall_gif.sh --seed 7
  ./run_manipulator_grid_wall_gif.sh --goal 8
  ./run_manipulator_grid_wall_gif.sh --skip-build --fps 16 --goal 6
EOF
}

while (($#)); do
    case "$1" in
        --skip-build)
            BUILD=0
            shift
            ;;
        --fps)
            FPS="${2:?missing value for --fps}"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="${2:?missing value for --max-frames}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:?missing value for --out-dir}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXAMPLE_ARGS+=("$@")
            break
            ;;
        *)
            EXAMPLE_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$SCRIPT_DIR"
if [ -n "$OUT_DIR" ] && [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$SCRIPT_DIR/$OUT_DIR"
fi

if [ "$BUILD" -eq 1 ]; then
    echo "[build] manipulator_grid_approach_workspace_example"
    bash "$SCRIPT_DIR/build_gpu.sh" manipulator_grid_approach_workspace_example
fi

EXE="$SCRIPT_DIR/build/gpu/manipulator_grid_approach_workspace_example"
if [ ! -x "$EXE" ]; then
    echo "ERROR: executable not found: $EXE" >&2
    echo "Run without --skip-build first, or build it with:" >&2
    echo "  bash build_gpu.sh manipulator_grid_approach_workspace_example" >&2
    exit 1
fi

mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

echo ""
echo "[run] build/gpu/manipulator_grid_approach_workspace_example ${EXAMPLE_ARGS[*]}"
set +e
"$EXE" "${EXAMPLE_ARGS[@]}"
RUN_STATUS=$?
set -e

if [ "$RUN_STATUS" -ne 0 ]; then
    echo ""
    echo "[warn] example exited with status $RUN_STATUS; continuing if CSV outputs were written"
fi

if [ ! -f "$SCRIPT_DIR/build/manipulator_grid_approach_bicmppi_executed_x.csv" ]; then
    echo "ERROR: trajectory CSV was not generated" >&2
    exit "$RUN_STATUS"
fi

GIF_ARGS=(
    --build-dir "$SCRIPT_DIR/build"
    --fps "$FPS"
    --max-frames "$MAX_FRAMES"
)
if [ -n "$OUT_DIR" ]; then
    GIF_ARGS+=(--out-dir "$OUT_DIR")
fi

echo ""
echo "[gif] rendering grid-approach manipulator path"
python3 "$SCRIPT_DIR/visualize_manipulator_grid_approach_gif.py" "${GIF_ARGS[@]}"

echo ""
echo "CSV outputs:"
echo "  $SCRIPT_DIR/build/manipulator_grid_approach_bicmppi_*.csv"
if [ -n "$OUT_DIR" ]; then
    echo "GIF output:"
    echo "  $OUT_DIR/manipulator_grid_approach_bicmppi.gif"
else
    echo "GIF output:"
    echo "  $SCRIPT_DIR/build/manipulator_grid_approach_gifs/manipulator_grid_approach_bicmppi.gif"
fi
