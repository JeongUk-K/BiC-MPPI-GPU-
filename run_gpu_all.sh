#!/usr/bin/env bash
# ============================================================
# run_gpu_all.sh — 모든 GPU 예제 순차 실행 스크립트
#
# build/gpu 폴더 내의 WMRobot GPU 예제를 기본으로 실행합니다.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"
TARGET_GROUP="${1:-wmrobot}"

if [ ! -d "$GPU_BUILD_DIR" ]; then
    echo "build/gpu 폴더가 없습니다. 먼저 ./build_gpu.sh $TARGET_GROUP 를 실행하세요."
    exit 1
fi

cd "$BUILD_ROOT" || exit 1

echo "=== Running all GPU examples ==="
echo "target group: $TARGET_GROUP"

AGG_SUMMARY=""
if [ "$TARGET_GROUP" = "wmrobot" ]; then
    AGG_SUMMARY="wmrobot_gpu_summary.csv"
    rm -f "$AGG_SUMMARY"
fi

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
        if [ "$TARGET_GROUP" = "wmrobot" ]; then
            rm -f "result_${EX}.csv" "result_${EX}_summary.csv"
        fi
        echo ""
        echo "--------------------------------------------------"
        echo " 🚀 실행 중: ./$BIN"
        echo "--------------------------------------------------"
        ./"$BIN"
        STATUS=$?
        if [ "$STATUS" -ne 0 ]; then
            echo "ERROR: ./$BIN failed with exit code $STATUS"
            exit "$STATUS"
        fi
        if [ "$TARGET_GROUP" = "wmrobot" ]; then
            SUMMARY_CSV="result_${EX}_summary.csv"
            if [ ! -f "$SUMMARY_CSV" ]; then
                echo "ERROR: expected summary CSV was not created: build/$SUMMARY_CSV"
                exit 1
            fi
            if [ -z "$AGG_SUMMARY" ] || [ ! -f "$AGG_SUMMARY" ]; then
                cp "$SUMMARY_CSV" "$AGG_SUMMARY"
            else
                tail -n +2 "$SUMMARY_CSV" >> "$AGG_SUMMARY"
            fi
        fi
    else
        echo ""
        echo "⚠️  실행 파일이 없습니다: $BIN (빌드 시 주석 처리됨)"
    fi
done

echo ""
if [ "$TARGET_GROUP" = "wmrobot" ] && [ -f "$AGG_SUMMARY" ]; then
    echo "Aggregate summary: build/$AGG_SUMMARY"
    cat "$AGG_SUMMARY"
    echo ""
fi
echo "=== 모든 GPU 예제 실행 완료 ==="
