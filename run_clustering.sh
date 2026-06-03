#!/usr/bin/env bash
# ============================================================
# run_clustering.sh — Clustering Parameter 예제들 순차 실행 스크립트
#
# build/gpu 폴더 내에 빌드된 clustering parameter 실행 파일들을
# 순차적으로 실행하고, 각 실행 파일이 덮어쓰는 result_bi_mppi.csv를
# 고유한 이름(예: result_bi_mppi_0.001.csv)으로 변경하여 저장합니다.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"

if [ ! -d "$GPU_BUILD_DIR" ]; then
    echo "[ERROR] build/gpu/ 디렉토리가 없습니다. 먼저 ./build_clustering.sh 를 실행하여 빌드하세요."
    exit 1
fi

# 실행할 솔버 리스트 (src/wmrobot/clustering parameter/ 에 맞춰 등록)
SOLVERS=(
    "bi_mppi_0.001"
    "bi_mppi_0.005"
    "bi_mppi_0.01"
    "bi_mppi_0.05"
    "bi_mppi_0.1"
)

echo "=================================================="
echo "  WMRobot Clustering Parameter Benchmark Runner"
echo "  build dir : $BUILD_ROOT"
echo "  binary dir: $GPU_BUILD_DIR"
echo "=================================================="

run_solver() {
    local name=$1
    local exe="$GPU_BUILD_DIR/$name"

    if [ ! -f "$exe" ]; then
        echo "[SKIP] $name 이 빌드되지 않았습니다 (파일 없음: $exe)"
        return
    fi

    local log_file="$BUILD_ROOT/log_${name}.txt"
    echo "[START] $name 시작..."

    local t_start=$(date +%s%3N)

    # build/에서 실행해야 소스의 ../BARN_dataset 상대경로가 맞습니다.
    cd "$BUILD_ROOT"
    
    local epsilon="${name#bi_mppi_}"
    local csv_file="result_bi_mppi_clustering_${epsilon}.csv"

    # 기존 결과 파일이 있다면 백업 또는 삭제 (깨끗한 측정을 위해 삭제)
    rm -f "$csv_file"

    ./gpu/"$name" 2>&1 | tee "$log_file"

    local t_end=$(date +%s%3N)
    
    if [ -f "$csv_file" ]; then
        echo "[DONE ] $name  →  build/$csv_file 생성 완료 ($(( t_end - t_start )) ms)"
    else
        echo "[WARN ] $name 실행은 완료되었으나 $csv_file 결과 파일이 생성되지 않았습니다."
    fi
    echo "--------------------------------------------------"
    cd "$SCRIPT_DIR"
}

# 순차적으로 솔버 실행
for s in "${SOLVERS[@]}"; do
    run_solver "$s"
done

echo ""
echo "=================================================="
echo "  Results Summary in build/:"
for s in "${SOLVERS[@]}"; do
    epsilon="${s#bi_mppi_}"
    f="$BUILD_ROOT/result_bi_mppi_clustering_${epsilon}.csv"
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  result_bi_mppi_clustering_${epsilon}.csv  ($((lines-1)) rows)"
    fi
done
echo "=================================================="
