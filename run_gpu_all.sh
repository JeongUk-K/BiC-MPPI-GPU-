#!/usr/bin/env bash
# ============================================================
# run_gpu_all.sh — 모든 GPU 예제 순차 실행 스크립트
#
# build/gpu 폴더 내의 실행 가능한 모든 GPU 예제를
# 자동으로 실행합니다.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"
TARGET_GROUP="${1:-manipulator}"

if [ ! -d "$GPU_BUILD_DIR" ]; then
    echo "build/gpu 폴더가 없습니다. 먼저 ./build_gpu.sh 를 실행하세요."
    exit 1
fi

cd "$BUILD_ROOT" || exit 1

echo "=== Running all GPU examples ==="
echo "target group: $TARGET_GROUP"

case "$TARGET_GROUP" in
    manipulator)
        PREFIX=""
        EXAMPLES=("mppi" "cluster_mppi" "bi_mppi" "svgd_mppi")
        ;;
    quadrotor|wmrobot|velo)
        PREFIX="${TARGET_GROUP}_"
        EXAMPLES=("mppi" "log_mppi" "cluster_mppi" "bi_mppi" "svgd_mppi")
        ;;
    wmrobot_ablation)
        PREFIX="wmrobot_ablation_gpu_"
        EXAMPLES=("mppi" "cluster_mppi" "bic_without_guide" "bic_without_backward" "bic_without_clustering" "full_bic_mppi" "all")
        ;;
    *)
        echo "unknown target group: $TARGET_GROUP"
        echo "Usage: bash run_gpu_all.sh [manipulator|quadrotor|wmrobot|wmrobot_ablation|velo]"
        exit 1
        ;;
esac

for EX in "${EXAMPLES[@]}"; do
    BIN="gpu/${PREFIX}${EX}"
    if [ -x "$BIN" ]; then
        echo ""
        echo "--------------------------------------------------"
        echo " 🚀 실행 중: ./$BIN"
        echo "--------------------------------------------------"
        ./"$BIN"
    else
        echo ""
        echo "⚠️  실행 파일이 없습니다: $BIN (빌드 시 주석 처리됨)"
    fi
done

echo ""
echo "=== 모든 GPU 예제 실행 완료 ==="
