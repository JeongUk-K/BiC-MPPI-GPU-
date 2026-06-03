#!/usr/bin/env bash
# ==============================================================================
# run_manipulator_comparison.sh — Manipulator 3D Obstacle Avoidance Comparison
# ==============================================================================
# Builds and runs all 4 GPU solvers on the 3D cylinder avoidance task,
# then generates 3D trajectory and rollout animations (GIFs) for each.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"

# 1. Compile the GPU solvers
echo "🔨 Compiling GPU solvers..."
bash build_gpu.sh

# 2. Select obstacle file
OBSTACLE_FILE="obstacles/cylinder_test.txt"
if [ -n "$1" ]; then
    OBSTACLE_FILE="$1"
fi
echo "📍 Using obstacle file: $OBSTACLE_FILE"

# 3. Create unified GPU build directory if it doesn't exist
mkdir -p "$GPU_BUILD_DIR"

# 4. Solvers list
SOLVERS=("mppi" "cluster_mppi" "bi_mppi" "svgd_mppi")

for SV in "${SOLVERS[@]}"; do
    BINARY="$GPU_BUILD_DIR/$SV"
    if [ -x "$BINARY" ]; then
        echo ""
        echo "========================================================================"
        echo "🚀 Running Solver: $SV with cylinder avoidance"
        echo "========================================================================"

        RUN_OBSTACLE_FILE="../$OBSTACLE_FILE"
        if [[ "$OBSTACLE_FILE" = /* ]]; then
            RUN_OBSTACLE_FILE="$OBSTACLE_FILE"
        fi

        # Run solver with obstacle file argument
        (cd "$BUILD_ROOT" && ./gpu/"$SV" "$RUN_OBSTACLE_FILE")
        
        # Determine step folder name in vis_data
        VIS_FOLDER="$BUILD_ROOT/vis_data/manipulator_$SV"
        if [ ! -d "$VIS_FOLDER" ]; then
            VIS_FOLDER="vis_data/manipulator_$SV"
        fi
        
        if [ -d "$VIS_FOLDER" ]; then
            echo "🎨 Generating 3D rollouts GIF for $SV..."
            (cd "$BUILD_ROOT" && python3 visualize_manipulator_rollouts.py "manipulator_$SV" "$RUN_OBSTACLE_FILE" --output "gpu/manipulator_${SV}_rollouts.gif")
        else
            echo "⚠️  No vis data found at $VIS_FOLDER for solver $SV"
        fi

        RESULT_CSV="$BUILD_ROOT/result_manipulator_gpu_${SV}.csv"
        if [ -f "$RESULT_CSV" ]; then
            echo "🌐 Generating Plotly Interactive 3D HTML for $SV..."
            (cd "$BUILD_ROOT" && python3 visualize_manipulator_plotly.py "result_manipulator_gpu_${SV}.csv" "$RUN_OBSTACLE_FILE" "gpu/manipulator_${SV}_3d.html")
        fi
    else
        echo "⚠️  Binary $BINARY not found or not executable. Please verify build_gpu.sh."
    fi
done

echo ""
echo "========================================================================"
echo "🎉 Manipulator Comparison Completed!"
echo "Generated Visualizations:"
for SV in "${SOLVERS[@]}"; do
    if [ -f "$GPU_BUILD_DIR/manipulator_${SV}_rollouts.gif" ]; then
        echo "  - GIF:  build/gpu/manipulator_${SV}_rollouts.gif"
    fi
    if [ -f "$GPU_BUILD_DIR/manipulator_${SV}_3d.html" ]; then
        echo "  - HTML: build/gpu/manipulator_${SV}_3d.html (Open in browser for interactive 3D)"
    fi
done
echo "========================================================================"
