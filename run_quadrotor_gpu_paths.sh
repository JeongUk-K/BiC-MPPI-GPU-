#!/usr/bin/env bash
# ============================================================
# run_quadrotor_gpu_paths.sh
#
# Builds quadrotor GPU examples, runs all MPPI variants, then
# saves one combined path plot per BARN map.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"

cd "$SCRIPT_DIR"

echo "=== Building quadrotor GPU examples ==="
bash build_gpu.sh quadrotor

SOLVERS=("mppi" "log_mppi" "cluster_mppi" "bi_mppi" "svgd_mppi")

echo ""
echo "=== Running quadrotor GPU examples ==="
for solver in "${SOLVERS[@]}"; do
    binary="$GPU_BUILD_DIR/quadrotor_$solver"
    if [ ! -x "$binary" ]; then
        echo "[SKIP] missing executable: $binary"
        continue
    fi

    echo ""
    echo "--------------------------------------------------"
    echo "[RUN] quadrotor_$solver"
    echo "--------------------------------------------------"
    (cd "$BUILD_ROOT" && ./gpu/"quadrotor_$solver")
done

echo ""
echo "=== Plotting per-map quadrotor paths ==="
python3 plot_quadrotor_gpu_paths.py \
    --build-dir "$BUILD_ROOT" \
    --map-dir "$SCRIPT_DIR/BARN_dataset/txt_files" \
    --output-dir "$BUILD_ROOT/plots/quadrotor_gpu"
