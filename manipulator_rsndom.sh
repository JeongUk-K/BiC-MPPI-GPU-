#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DEFAULT_BENCHMARK_DIR="$SCRIPT_DIR/random_pose_benchmark/results"
BENCHMARK_DIR="$DEFAULT_BENCHMARK_DIR"
BENCHMARK_DIR_CUSTOM=0
OUT_DIR=""
FPS=12
MAX_FRAMES=72
SKIP_BUILD=0
SKIP_RUN=0
SKIP_GIFS=0
SUMMARY_ALL=0
SELECTED_SOLVERS=()
ALL_SOLVERS=(mppi logmppi clustermppi bicmppi)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build and run the manipulator random_pose_benchmark examples, then create GIFs.
The full benchmark evaluates all 240 ordered start-goal pairs per solver.
Rendering all solvers without --skip-gifs creates 960 GIFs.

Options:
  --skip-build          Reuse existing build/gpu random-pose benchmark binaries
  --skip-run            Reuse existing benchmark CSV outputs and only render stats/GIFs
  --skip-gifs           Rebuild only final aggregate stats CSV
  --summary-all         Rebuild final stats using all solver CSVs after running
                        only the selected solver(s). Useful for rerunning one solver.
  --solver NAME         Run/render one solver: mppi, logmppi, clustermppi, bicmppi
                        Can be passed multiple times. Default: all solvers
  --benchmark-dir DIR   Existing benchmark CSV directory for --skip-run
                        Default: random_pose_benchmark/results
  --out-dir DIR         GIF output directory
                        Default: <benchmark-dir>/gifs
  --fps N              GIF frame rate. Default: 12
  --max-frames N       Maximum sampled frames per GIF. Default: 72
  -h, --help           Show this help

Examples:
  ./$(basename "$0")
  ./$(basename "$0") --skip-build --solver bicmppi
  ./$(basename "$0") --skip-build --solver logmppi --skip-gifs --summary-all
  ./$(basename "$0") --skip-build --skip-run --skip-gifs
EOF
}

is_solver() {
  case "$1" in
    mppi|logmppi|clustermppi|bicmppi) return 0 ;;
    *) return 1 ;;
  esac
}

solver_exe() {
  case "$1" in
    mppi) echo "manipulator_mppi_random_pose_benchmark" ;;
    logmppi) echo "manipulator_logmppi_random_pose_benchmark" ;;
    clustermppi) echo "manipulator_clustermppi_random_pose_benchmark" ;;
    bicmppi) echo "manipulator_bicmppi_random_pose_benchmark" ;;
    *) return 1 ;;
  esac
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$SCRIPT_DIR/$path"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --skip-gifs)
      SKIP_GIFS=1
      shift
      ;;
    --summary-all)
      SUMMARY_ALL=1
      shift
      ;;
    --solver)
      if [[ $# -lt 2 ]]; then
        echo "error: --solver requires a value" >&2
        exit 2
      fi
      if ! is_solver "$2"; then
        echo "error: unknown solver '$2'" >&2
        exit 2
      fi
      SELECTED_SOLVERS+=("$2")
      shift 2
      ;;
    --benchmark-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --benchmark-dir requires a value" >&2
        exit 2
      fi
      BENCHMARK_DIR="$(abs_path "$2")"
      BENCHMARK_DIR_CUSTOM=1
      shift 2
      ;;
    --out-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --out-dir requires a value" >&2
        exit 2
      fi
      OUT_DIR="$(abs_path "$2")"
      shift 2
      ;;
    --fps)
      if [[ $# -lt 2 ]]; then
        echo "error: --fps requires a value" >&2
        exit 2
      fi
      FPS="$2"
      shift 2
      ;;
    --max-frames)
      if [[ $# -lt 2 ]]; then
        echo "error: --max-frames requires a value" >&2
        exit 2
      fi
      MAX_FRAMES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SELECTED_SOLVERS[@]} -eq 0 ]]; then
  SELECTED_SOLVERS=("${ALL_SOLVERS[@]}")
fi

if ! [[ "$FPS" =~ ^[0-9]+$ ]] || [[ "$FPS" -lt 1 ]]; then
  echo "error: --fps must be a positive integer" >&2
  exit 2
fi

if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ ]] || [[ "$MAX_FRAMES" -lt 1 ]]; then
  echo "error: --max-frames must be a positive integer" >&2
  exit 2
fi

if [[ "$BENCHMARK_DIR_CUSTOM" -eq 1 && "$SKIP_RUN" -eq 0 && "$BENCHMARK_DIR" != "$DEFAULT_BENCHMARK_DIR" ]]; then
  echo "error: --benchmark-dir can only be used with --skip-run" >&2
  echo "hint: the C++ random_pose_benchmark examples write to random_pose_benchmark/results" >&2
  exit 2
fi

cd "$SCRIPT_DIR"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[build] manipulator workspace-collision random_pose_benchmark examples"
  bash "$SCRIPT_DIR/build_gpu.sh" manipulator_workspace_collision
else
  echo "[build] skipped"
fi

for solver in "${SELECTED_SOLVERS[@]}"; do
  exe="$(solver_exe "$solver")"
  exe_path="$BUILD_DIR/gpu/$exe"
  if [[ ! -x "$exe_path" ]]; then
    echo "error: executable not found: $exe_path" >&2
    echo "hint: rerun without --skip-build" >&2
    exit 1
  fi
done

if [[ "$SKIP_RUN" -eq 0 ]]; then
  mkdir -p "$BENCHMARK_DIR"
  cd "$BUILD_DIR"
  for solver in "${SELECTED_SOLVERS[@]}"; do
    exe="$(solver_exe "$solver")"
    echo ""
    echo "[run] build/gpu/$exe"
    "./gpu/$exe"
  done
  cd "$SCRIPT_DIR"
else
  echo "[run] skipped"
fi

for solver in "${SELECTED_SOLVERS[@]}"; do
  stats_path="$BENCHMARK_DIR/manipulator_random_pose_benchmark_${solver}_stats.csv"
  if [[ ! -s "$stats_path" ]]; then
    echo "error: missing benchmark stats CSV: $stats_path" >&2
    exit 1
  fi
done

if [[ "$SUMMARY_ALL" -eq 1 ]]; then
  for solver in "${ALL_SOLVERS[@]}"; do
    stats_path="$BENCHMARK_DIR/manipulator_random_pose_benchmark_${solver}_stats.csv"
    if [[ ! -s "$stats_path" ]]; then
      echo "error: --summary-all requires existing stats CSV: $stats_path" >&2
      echo "hint: run the missing solver first or omit --summary-all" >&2
      exit 1
    fi
  done
fi

VIS_ARGS=(
  --build-dir "$BUILD_DIR"
  --benchmark-dir "$BENCHMARK_DIR"
  --fps "$FPS"
  --max-frames "$MAX_FRAMES"
)

if [[ -n "$OUT_DIR" ]]; then
  VIS_ARGS+=(--out-dir "$OUT_DIR")
fi

if [[ "$SKIP_GIFS" -eq 1 ]]; then
  VIS_ARGS+=(--skip-gifs)
fi

VIS_SOLVERS=("${SELECTED_SOLVERS[@]}")
if [[ "$SUMMARY_ALL" -eq 1 && "$SKIP_GIFS" -eq 1 ]]; then
  VIS_SOLVERS=("${ALL_SOLVERS[@]}")
fi

for solver in "${VIS_SOLVERS[@]}"; do
  VIS_ARGS+=(--solver "$solver")
done

echo ""
if [[ "$SKIP_GIFS" -eq 1 ]]; then
  echo "[stats] rebuilding random pose benchmark final stats"
else
  echo "[gif] rendering random pose benchmark GIFs and final stats"
fi
python3 "$SCRIPT_DIR/visualize_manipulator_random_pose_benchmark_gifs.py" "${VIS_ARGS[@]}"

if [[ "$SUMMARY_ALL" -eq 1 && "$SKIP_GIFS" -eq 0 ]]; then
  SUMMARY_ARGS=(
    --build-dir "$BUILD_DIR"
    --benchmark-dir "$BENCHMARK_DIR"
    --skip-gifs
  )
  for solver in "${ALL_SOLVERS[@]}"; do
    SUMMARY_ARGS+=(--solver "$solver")
  done
  echo ""
  echo "[stats] rebuilding final stats from all solver CSVs"
  python3 "$SCRIPT_DIR/visualize_manipulator_random_pose_benchmark_gifs.py" "${SUMMARY_ARGS[@]}"
fi

GIF_DIR="${OUT_DIR:-$BENCHMARK_DIR/gifs}"
echo ""
echo "Random pose benchmark outputs:"
for solver in "${VIS_SOLVERS[@]}"; do
  echo "  $BENCHMARK_DIR/manipulator_random_pose_benchmark_${solver}_stats.csv"
done
echo "  $BENCHMARK_DIR/manipulator_random_pose_benchmark_final_stats.csv"
if [[ "$SKIP_GIFS" -eq 0 ]]; then
  echo "  $GIF_DIR/"
fi
