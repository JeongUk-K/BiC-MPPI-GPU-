#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-random_pose}"
if [ "$MODE" = "random" ]; then
  MODE="random_pose"
fi
if [ "$MODE" != "build" ] && [ "$MODE" != "workspace" ] && [ "$MODE" != "pinkNplace" ] && [ "$MODE" != "random_pose" ] && [ "$MODE" != "all" ]; then
  echo "usage: $0 [build|workspace|pinkNplace|random_pose|all]" >&2
  exit 2
fi

echo "[build] manipulator workspace-collision MPPI examples"
bash "$SCRIPT_DIR/build_gpu.sh" manipulator_workspace_collision

if [ "$MODE" = "build" ]; then
  echo ""
  echo "Build complete. No examples were run."
  exit 0
fi

echo ""
cd "$SCRIPT_DIR/build"
if [ "$MODE" = "workspace" ] || [ "$MODE" = "all" ]; then
  for exe in \
    manipulator_mppi_workspace_example \
    manipulator_logmppi_workspace_example \
    manipulator_clustermppi_workspace_example \
    manipulator_bicmppi_workspace_example; do
    echo "[run] build/gpu/$exe"
    "./gpu/$exe"
    echo ""
  done

  echo ""
  echo "Workspace outputs:"
  echo "  $SCRIPT_DIR/build/manipulator_workspace_mppi_*.csv"
  echo "  $SCRIPT_DIR/build/manipulator_workspace_logmppi_*.csv"
  echo "  $SCRIPT_DIR/build/manipulator_workspace_clustermppi_*.csv"
  echo "  $SCRIPT_DIR/build/manipulator_workspace_bicmppi_*.csv"

  echo ""
  echo "[gif] rendering executed manipulator motions"
  python3 "$SCRIPT_DIR/visualize_manipulator_workspace_gifs.py" \
    --build-dir "$SCRIPT_DIR/build"
fi

if [ "$MODE" = "pinkNplace" ] || [ "$MODE" = "all" ]; then
  for exe in \
    manipulator_mppi_pinkNplace_workspace_example \
    manipulator_logmppi_pinkNplace_workspace_example \
    manipulator_clustermppi_pinkNplace_workspace_example \
    manipulator_bicmppi_pinkNplace_workspace_example; do
    echo "[run] build/gpu/$exe"
    "./gpu/$exe"
    echo ""
  done

  echo ""
  echo "pinkNplace outputs:"
  echo "  $SCRIPT_DIR/build/manipulator_pinkNplace_mppi_*.csv"
  echo "  $SCRIPT_DIR/build/manipulator_pinkNplace_logmppi_*.csv"
  echo "  $SCRIPT_DIR/build/manipulator_pinkNplace_clustermppi_*.csv"
  echo "  $SCRIPT_DIR/build/manipulator_pinkNplace_bicmppi_*.csv"

  echo ""
  echo "[gif] rendering pinkNplace manipulator motions"
  python3 "$SCRIPT_DIR/visualize_manipulator_pinkNplace_gifs.py" \
    --build-dir "$SCRIPT_DIR/build"
fi

if [ "$MODE" = "random_pose" ] || [ "$MODE" = "all" ]; then
  for exe in \
    manipulator_mppi_random_pose_benchmark \
    manipulator_logmppi_random_pose_benchmark \
    manipulator_clustermppi_random_pose_benchmark \
    manipulator_bicmppi_random_pose_benchmark; do
    echo "[run] build/gpu/$exe"
    "./gpu/$exe"
    echo ""
  done

  echo ""
  echo "Random pose benchmark outputs:"
  echo "  $SCRIPT_DIR/random_pose_benchmark/results/manipulator_random_pose_benchmark_mppi_stats.csv"
  echo "  $SCRIPT_DIR/random_pose_benchmark/results/manipulator_random_pose_benchmark_logmppi_stats.csv"
  echo "  $SCRIPT_DIR/random_pose_benchmark/results/manipulator_random_pose_benchmark_clustermppi_stats.csv"
  echo "  $SCRIPT_DIR/random_pose_benchmark/results/manipulator_random_pose_benchmark_bicmppi_stats.csv"
  echo "  $SCRIPT_DIR/random_pose_benchmark/results/manipulator_random_pose_benchmark_final_stats.csv"
  echo "  $SCRIPT_DIR/random_pose_benchmark/results/gifs/"

  echo ""
  echo "[gif] rendering random pose benchmark motions and final stats"
  python3 "$SCRIPT_DIR/visualize_manipulator_random_pose_benchmark_gifs.py" \
    --build-dir "$SCRIPT_DIR/build" \
    --benchmark-dir "$SCRIPT_DIR/random_pose_benchmark/results"
fi
