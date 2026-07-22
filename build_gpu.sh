#!/usr/bin/env bash
# ============================================================
# build_gpu.sh — BiC-MPPI GPU 버전 빌드 스크립트
#
# 시스템 CUDA toolkit을 우선 사용합니다. 없으면 MATLAB 번들 nvcc를
# fallback으로 사용합니다.
#
# 사용법:
#   bash build_gpu.sh             # manipulator GPU 예제 빌드
#   bash build_gpu.sh manipulator_workspace_collision  # workspace link-collision manipulator BiC-MPPI 예제 빌드
#   bash build_gpu.sh manipulator_grid_wall_workspace_example  # grid-wall manipulator BiC-MPPI 예제 빌드
#   bash build_gpu.sh manipulator_grid_approach_workspace_example  # grid-approach manipulator BiC-MPPI 예제 빌드
#   bash build_gpu.sh quadrotor   # quadrotor GPU 예제 빌드
#   bash build_gpu.sh quadrotor_log_mppi  # quadrotor Log-MPPI GPU 예제만 빌드
#   bash build_gpu.sh quadrotor_gpu_new  # quadrotor precision-landing GPU 예제 빌드
#   bash build_gpu.sh wmrobot     # wmrobot GPU 예제 빌드
#   bash build_gpu.sh wmrobot_accel  # 통합 CUDA 솔버의 호환용 gpu-accel 출력 빌드
#   bash build_gpu.sh wmrobot_log_mppi  # wmrobot Log-MPPI GPU 예제만 빌드
#   bash build_gpu.sh wmrobot_ablation  # wmrobot ablation GPU 예제 빌드
#   bash build_gpu.sh wmrobot_same_budget  # wmrobot same-budget scaling 예제 빌드
#   bash build_gpu.sh wmrobot_waypoint  # wmrobot waypoint-parallel BiC 예제 빌드
#   bash build_gpu.sh wmrobot_waypoint_stitched  # wmrobot 5-map stitched waypoint 예제 빌드
#   bash build_gpu.sh wmrobot_waypoint_stitched_sequential  # wmrobot 5-map stitched 순차추종 예제 빌드
#   bash build_gpu.sh wmrobot_map_285  # wmrobot map 285 4개 시각화 예제 빌드
#   bash build_gpu.sh velo        # velo GPU 예제 빌드
# ============================================================
set -e

# ── 경로 설정 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"
TARGET_GROUP="${1:-manipulator}"
GPU_SOLVER_DIR="mppi/cuda"
BUILD_PREFIX=""
BUILD_COMMON_GPU_LIB=1

if [ "$TARGET_GROUP" = "wmrobot_accel" ]; then
    GPU_BUILD_DIR="$BUILD_ROOT/gpu-accel"
    # Compatibility target: the selectable implementation now lives in cuda/.
    GPU_SOLVER_DIR="mppi/cuda"
fi

if [ "$TARGET_GROUP" = "manipulator_workspace_collision" ] || \
   [ "$TARGET_GROUP" = "manipulator_bicmppi_workspace_example" ] || \
   [ "$TARGET_GROUP" = "manipulator_grid_wall_workspace_example" ] || \
   [ "$TARGET_GROUP" = "manipulator_grid_approach_workspace_example" ]; then
    BUILD_COMMON_GPU_LIB=0
fi

NVCC=""
CUDA_INCLUDE=""
CUDA_LIBDIR=""
CURAND_LIB=""
CUDART_LIB=""

# 1) 시스템 CUDA toolkit (/usr/local/cuda*)
for d in /usr/local/cuda /usr/local/cuda-12.5 /usr/local/cuda-12.6 /usr/local/cuda-13; do
    if [ -f "$d/bin/nvcc" ]; then
        NVCC="$d/bin/nvcc"
        CUDA_INCLUDE="$d/include"
        CUDA_LIBDIR="$d/lib64"
        break
    fi
done

# 2) apt 등으로 설치된 전역 nvcc 확인
if [ -z "$NVCC" ] && command -v nvcc >/dev/null 2>&1; then
    NVCC=$(command -v nvcc)
    NVCC_ROOT="$(cd "$(dirname "$NVCC")/.." && pwd)"
    if [ -f "$NVCC_ROOT/include/cuda_runtime.h" ]; then
        CUDA_INCLUDE="$NVCC_ROOT/include"
        if [ -d "$NVCC_ROOT/lib64" ]; then
            CUDA_LIBDIR="$NVCC_ROOT/lib64"
        else
            CUDA_LIBDIR="/usr/lib/x86_64-linux-gnu"
        fi
    else
        CUDA_INCLUDE="/usr/include"
        CUDA_LIBDIR="/usr/lib/x86_64-linux-gnu"
    fi
fi

# 3) MATLAB 번들 CUDA 확인
if [ -z "$NVCC" ]; then
    for d in /usr/local/MATLAB/R*/sys/cuda/glnxa64/cuda; do
        if [ -f "$d/bin/nvcc" ]; then
            NVCC="$d/bin/nvcc"
            CUDA_INCLUDE="$d/include"
            CUDA_LIBDIR="$d/lib64"
            break
        fi
    done
fi

if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc를 찾을 수 없습니다. CUDA toolkit을 설치해 주세요."
    echo "  sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

NVCC_BIN_DIR="$(cd "$(dirname "$NVCC")" && pwd)"
NVCC_ROOT="$(cd "$NVCC_BIN_DIR/.." && pwd)"
export PATH="$NVCC_BIN_DIR:$NVCC_ROOT/nvvm/bin:$PATH"

MATLAB_LIBDIR=""
case "$NVCC" in
    /usr/local/MATLAB/*/sys/cuda/glnxa64/cuda/bin/nvcc)
        MATLAB_ROOT="${NVCC%/sys/cuda/glnxa64/cuda/bin/nvcc}"
        MATLAB_LIBDIR="$MATLAB_ROOT/bin/glnxa64"
        ;;
esac

# curand 탐색
CURAND_LIB=""
for lib in "$CUDA_LIBDIR"/libcurand.so* "$MATLAB_LIBDIR"/libcurand.so* "/usr/lib/x86_64-linux-gnu"/libcurand.so*; do
    if [ -f "$lib" ]; then CURAND_LIB="$lib"; break; fi
done

# cudart 탐색
CUDART_LIB=""
for lib in "$CUDA_LIBDIR"/libcudart.so* "$MATLAB_LIBDIR"/libcudart.so* "/usr/lib/x86_64-linux-gnu"/libcudart.so*; do
    if [ -f "$lib" ]; then CUDART_LIB="$lib"; break; fi
done

echo "=== GPU Build Configuration ==="
echo "  nvcc        : $NVCC"
echo "  CUDA include: $CUDA_INCLUDE"
echo "  cudart      : ${CUDART_LIB:-NOT FOUND}"
echo "  curand      : ${CURAND_LIB:-NOT FOUND}"
echo "  target group: $TARGET_GROUP"
echo "================================"

# ── 컴파일 옵션 ───────────────────────────────────────────────────
ARCH="-arch=sm_86"          # RTX 5060 (sm_120)은 PTX JIT 사용
CXX=g++
CXXFLAGS="-O3 -std=c++17 -fopenmp"
# GCC 11 버그 우회 (nvcc 11.5 호환성을 위해 g++-10 사용 권장)
CCBIN_FLAG=""
if command -v g++-10 >/dev/null 2>&1; then
    CCBIN_FLAG="-ccbin g++-10"
fi

NVCCFLAGS="-O3 --std=c++14 $ARCH $CCBIN_FLAG --expt-relaxed-constexpr -Xcompiler -fopenmp"

INCLUDES="-I./mppi -I./$GPU_SOLVER_DIR -I./mppi/cpu-legacy -I./model \
          -I./include/fastsc \
          -I./include/EigenRand -I./include/matplotlibcpp \
          -I$CUDA_INCLUDE \
          $(python3-config --includes) \
          $(pkg-config --cflags eigen3)"

EIGEN_INC=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

INCLUDES="-I./mppi -I./$GPU_SOLVER_DIR -I./mppi/cpu-legacy -I./model \
          -I./include/fastsc \
          -I./include/EigenRand -I./include/matplotlibcpp \
          -I$CUDA_INCLUDE \
          $(python3-config --includes) \
          $EIGEN_INC"

LDFLAGS="-fopenmp"
[ -n "$CUDART_LIB" ] && LDFLAGS="$LDFLAGS $CUDART_LIB"
[ -n "$CURAND_LIB" ] && LDFLAGS="$LDFLAGS $CURAND_LIB"
LDFLAGS="$LDFLAGS $(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)"
LDFLAGS="$LDFLAGS -lcublas"

# ── 빌드 디렉토리 ─────────────────────────────────────────────────
mkdir -p "$GPU_BUILD_DIR"

BUILD_SVGD_OBJ=1
if [ "$TARGET_GROUP" = "wmrobot_waypoint_stitched" ] || \
   [ "$TARGET_GROUP" = "wmrobot_waypoint_stitched_sequential" ] || \
   [ "$TARGET_GROUP" = "wmrobot_map_285" ]; then
    BUILD_SVGD_OBJ=0
fi

# ── GPU solver 오브젝트 파일 컴파일 ──────────────────────────────
if [ "$BUILD_COMMON_GPU_LIB" -eq 1 ]; then
    echo "[1/?] Compiling mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c "$GPU_SOLVER_DIR/mppi_gpu.cu" -o "$GPU_BUILD_DIR/mppi_gpu.o"

    echo "[2/?] Compiling cluster_mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c "$GPU_SOLVER_DIR/cluster_mppi_gpu.cu" -o "$GPU_BUILD_DIR/cluster_mppi_gpu.o"

    echo "[3/?] Compiling bi_mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c "$GPU_SOLVER_DIR/bi_mppi_gpu.cu" -o "$GPU_BUILD_DIR/bi_mppi_gpu.o"

    GPU_OBJS=("$GPU_BUILD_DIR/mppi_gpu.o" "$GPU_BUILD_DIR/cluster_mppi_gpu.o" "$GPU_BUILD_DIR/bi_mppi_gpu.o")

    if [ "$BUILD_SVGD_OBJ" -eq 1 ]; then
        echo "[4/?] Compiling svgd_mppi_gpu.cu ..."
        $NVCC $NVCCFLAGS $INCLUDES -c "$GPU_SOLVER_DIR/svgd_mppi_gpu.cu" -o "$GPU_BUILD_DIR/svgd_mppi_gpu.o"
        GPU_OBJS+=("$GPU_BUILD_DIR/svgd_mppi_gpu.o")
    else
        echo "[4/?] Skipping svgd_mppi_gpu.cu for $TARGET_GROUP"
    fi

    echo "[5/?] Compiling log_mppi_gpu.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c "$GPU_SOLVER_DIR/log_mppi_gpu.cu" -o "$GPU_BUILD_DIR/log_mppi_gpu.o"
    GPU_OBJS+=("$GPU_BUILD_DIR/log_mppi_gpu.o")

    echo "[fastsc] Compiling labels.cu ..."
    $NVCC $NVCCFLAGS $INCLUDES -c include/fastsc/labels.cu -o "$GPU_BUILD_DIR/fastsc_labels.o"
    GPU_OBJS+=("$GPU_BUILD_DIR/fastsc_labels.o")

    # GPU 오브젝트들을 ar로 정적 라이브러리로 묶기
    rm -f "$GPU_BUILD_DIR/libmppi_gpu.a"
    ar rcs "$GPU_BUILD_DIR/libmppi_gpu.a" "${GPU_OBJS[@]}"
    echo "  → ${GPU_BUILD_DIR#$SCRIPT_DIR/}/libmppi_gpu.a 생성 완료"
else
    echo "[1/?] Skipping shared GPU solver library for $TARGET_GROUP"
fi

# ── 실행 파일 링크 ────────────────────────────────────────────────
build_target() {
    local SRC="$1"
    local NAME="${BUILD_PREFIX}$(basename "${SRC%.cpp}")"
    echo "[4/?] Building $NAME ..."
    $CXX $CXXFLAGS $INCLUDES "$SRC" \
        -L"$GPU_BUILD_DIR" -lmppi_gpu \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/$NAME"
    echo "  → ${GPU_BUILD_DIR#$SCRIPT_DIR/}/$NAME 완료"
}

build_target_named() {
    local SRC="$1"
    local NAME="$2"
    echo "[vis] Building $NAME ..."
    $CXX $CXXFLAGS $INCLUDES "$SRC" \
        -L"$GPU_BUILD_DIR" -lmppi_gpu \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/$NAME"
    echo "  → ${GPU_BUILD_DIR#$SCRIPT_DIR/}/$NAME 완료"
}

build_manipulator_workspace_collision_target() {
    local PKG_DIR="src/manipulator/manipulator_workspace_collision_package"
    local MPPI_OBJ="$GPU_BUILD_DIR/manipulator_workspace_mppi_gpu.o"
    local LOG_OBJ="$GPU_BUILD_DIR/manipulator_workspace_log_mppi_gpu.o"
    local CLUSTER_OBJ="$GPU_BUILD_DIR/manipulator_workspace_cluster_mppi_gpu.o"
    local OBJ="$GPU_BUILD_DIR/manipulator_workspace_bi_mppi_gpu.o"
    local FASTSC_OBJ="$GPU_BUILD_DIR/manipulator_workspace_fastsc_labels.o"
    local WS_INCLUDES="-I./$PKG_DIR/include -I./$PKG_DIR/examples -I./$PKG_DIR/examples/random_pose_benchmark -I./$PKG_DIR/examples/grid_wall $INCLUDES"

    echo "[workspace] Compiling manipulator workspace MPPI CUDA source ..."
    $NVCC $NVCCFLAGS $WS_INCLUDES -c "$PKG_DIR/src/mppi_gpu.cu" -o "$MPPI_OBJ"

    echo "[workspace] Compiling manipulator workspace Log-MPPI CUDA source ..."
    $NVCC $NVCCFLAGS $WS_INCLUDES -c "$PKG_DIR/src/log_mppi_gpu.cu" -o "$LOG_OBJ"

    echo "[workspace] Compiling manipulator workspace Cluster-MPPI CUDA source ..."
    $NVCC $NVCCFLAGS $WS_INCLUDES -c "$PKG_DIR/src/cluster_mppi_gpu.cu" -o "$CLUSTER_OBJ"

    echo "[workspace] Compiling manipulator workspace BiC-MPPI CUDA source ..."
    $NVCC $NVCCFLAGS $WS_INCLUDES -c "$PKG_DIR/src/bi_mppi_gpu.cu" -o "$OBJ"

    echo "[workspace] Compiling fastsc K-means support ..."
    $NVCC $NVCCFLAGS $WS_INCLUDES -c "include/fastsc/labels.cu" -o "$FASTSC_OBJ"

    echo "[workspace] Building manipulator_mppi_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/manipulator_mppi_workspace_example.cpp" "$MPPI_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_mppi_workspace_example"
    echo "  → build/gpu/manipulator_mppi_workspace_example 완료"

    echo "[workspace] Building manipulator_logmppi_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/manipulator_logmppi_workspace_example.cpp" "$MPPI_OBJ" "$LOG_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_logmppi_workspace_example"
    echo "  → build/gpu/manipulator_logmppi_workspace_example 완료"

    echo "[workspace] Building manipulator_clustermppi_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/manipulator_clustermppi_workspace_example.cpp" "$MPPI_OBJ" "$CLUSTER_OBJ" "$FASTSC_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_clustermppi_workspace_example"
    echo "  → build/gpu/manipulator_clustermppi_workspace_example 완료"

    echo "[workspace] Building manipulator_bicmppi_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/manipulator_bicmppi_workspace_example.cpp" "$OBJ" "$FASTSC_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_bicmppi_workspace_example"
    echo "  → build/gpu/manipulator_bicmppi_workspace_example 완료"

    echo "[workspace] Building manipulator_mppi_pinkNplace_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/pinkNplace/manipulator_mppi_pinkNplace_workspace_example.cpp" "$MPPI_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_mppi_pinkNplace_workspace_example"
    echo "  → build/gpu/manipulator_mppi_pinkNplace_workspace_example 완료"

    echo "[workspace] Building manipulator_logmppi_pinkNplace_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/pinkNplace/manipulator_logmppi_pinkNplace_workspace_example.cpp" "$MPPI_OBJ" "$LOG_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_logmppi_pinkNplace_workspace_example"
    echo "  → build/gpu/manipulator_logmppi_pinkNplace_workspace_example 완료"

    echo "[workspace] Building manipulator_clustermppi_pinkNplace_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/pinkNplace/manipulator_clustermppi_pinkNplace_workspace_example.cpp" "$MPPI_OBJ" "$CLUSTER_OBJ" "$FASTSC_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_clustermppi_pinkNplace_workspace_example"
    echo "  → build/gpu/manipulator_clustermppi_pinkNplace_workspace_example 완료"

    echo "[workspace] Building manipulator_bicmppi_pinkNplace_workspace_example ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/pinkNplace/manipulator_bicmppi_pinkNplace_workspace_example.cpp" "$OBJ" "$FASTSC_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS \
        -o "$GPU_BUILD_DIR/manipulator_bicmppi_pinkNplace_workspace_example"
    echo "  → build/gpu/manipulator_bicmppi_pinkNplace_workspace_example 완료"

    echo "[workspace] Building manipulator_mppi_random_pose_benchmark ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/random_pose_benchmark/manipulator_mppi_random_pose_benchmark.cpp" "$MPPI_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS -lzstd \
        -o "$GPU_BUILD_DIR/manipulator_mppi_random_pose_benchmark"
    echo "  → build/gpu/manipulator_mppi_random_pose_benchmark 완료"

    echo "[workspace] Building manipulator_logmppi_random_pose_benchmark ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/random_pose_benchmark/manipulator_logmppi_random_pose_benchmark.cpp" "$MPPI_OBJ" "$LOG_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS -lzstd \
        -o "$GPU_BUILD_DIR/manipulator_logmppi_random_pose_benchmark"
    echo "  → build/gpu/manipulator_logmppi_random_pose_benchmark 완료"

    echo "[workspace] Building manipulator_clustermppi_random_pose_benchmark ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/random_pose_benchmark/manipulator_clustermppi_random_pose_benchmark.cpp" "$MPPI_OBJ" "$CLUSTER_OBJ" "$FASTSC_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS -lzstd \
        -o "$GPU_BUILD_DIR/manipulator_clustermppi_random_pose_benchmark"
    echo "  → build/gpu/manipulator_clustermppi_random_pose_benchmark 완료"

    echo "[workspace] Building manipulator_bicmppi_random_pose_benchmark ..."
    $CXX $CXXFLAGS $WS_INCLUDES "$PKG_DIR/examples/random_pose_benchmark/manipulator_bicmppi_random_pose_benchmark.cpp" "$OBJ" "$FASTSC_OBJ" \
        -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
        $LDFLAGS -lzstd \
        -o "$GPU_BUILD_DIR/manipulator_bicmppi_random_pose_benchmark"
    echo "  → build/gpu/manipulator_bicmppi_random_pose_benchmark 완료"

    local GRID_WALL_SRC="$PKG_DIR/examples/grid_wall/manipulator_grid_wall_workspace_example.cpp"
    if [ -f "$GRID_WALL_SRC" ]; then
        echo "[workspace] Building manipulator_grid_wall_workspace_example ..."
        $CXX $CXXFLAGS $WS_INCLUDES "$GRID_WALL_SRC" "$OBJ" "$FASTSC_OBJ" \
            -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
            $LDFLAGS \
            -o "$GPU_BUILD_DIR/manipulator_grid_wall_workspace_example"
        echo "  → build/gpu/manipulator_grid_wall_workspace_example 완료"
    elif [ "$TARGET_GROUP" = "manipulator_grid_wall_workspace_example" ]; then
        echo "ERROR: missing source: $GRID_WALL_SRC" >&2
        exit 1
    else
        echo "[workspace] Skipping manipulator_grid_wall_workspace_example; source not found"
    fi

    local GRID_APPROACH_SRC="$PKG_DIR/examples/grid_wall/manipulator_grid_approach_workspace_example.cpp"
    if [ -f "$GRID_APPROACH_SRC" ]; then
        echo "[workspace] Building manipulator_grid_approach_workspace_example ..."
        $CXX $CXXFLAGS $WS_INCLUDES "$GRID_APPROACH_SRC" "$OBJ" "$FASTSC_OBJ" \
            -Wl,-rpath,$(dirname ${CUDART_LIB:-/dev/null}) \
            $LDFLAGS \
            -o "$GPU_BUILD_DIR/manipulator_grid_approach_workspace_example"
        echo "  → build/gpu/manipulator_grid_approach_workspace_example 완료"
    elif [ "$TARGET_GROUP" = "manipulator_grid_approach_workspace_example" ]; then
        echo "ERROR: missing source: $GRID_APPROACH_SRC" >&2
        exit 1
    fi
}

# ===========================================================================
# 빌드할 소스 파일 선택 (사용할 예제의 주석을 해제하세요)
# ===========================================================================
case "$TARGET_GROUP" in
    manipulator)
        # ---- Manipulator GPU 버전 ----
        build_target src/manipulator/gpu/mppi.cpp
        build_target src/manipulator/gpu/cluster_mppi.cpp
        build_target src/manipulator/gpu/bi_mppi.cpp
        build_target src/manipulator/gpu/svgd_mppi.cpp
        build_target src/manipulator/gpu/mppi_cylinder.cpp
        ;;
    manipulator_workspace_collision|manipulator_bicmppi_workspace_example|manipulator_grid_wall_workspace_example|manipulator_grid_approach_workspace_example)
        # ---- Manipulator workspace link-collision BiC-MPPI 예제 ----
        build_manipulator_workspace_collision_target
        ;;
    quadrotor)
        # ---- Quadrotor GPU 버전 ----
        BUILD_PREFIX="quadrotor_"
        build_target src/quadrotor/gpu/mppi.cpp
        build_target src/quadrotor/gpu/log_mppi.cpp
        build_target src/quadrotor/gpu/cluster_mppi.cpp
        build_target src/quadrotor/gpu/bi_mppi.cpp
        build_target src/quadrotor/gpu/svgd_mppi.cpp
        ;;
    quadrotor_log_mppi)
        # ---- Quadrotor Log-MPPI GPU 단일 예제 ----
        BUILD_PREFIX="quadrotor_"
        build_target src/quadrotor/gpu/log_mppi.cpp
        ;;
    quadrotor_gpu_new)
        # ---- Quadrotor Precision-Landing GPU New 예제 ----
        BUILD_PREFIX="quadrotor_gpu_new_"
        build_target "src/quadrotor/gpu new/mppi.cpp"
        build_target "src/quadrotor/gpu new/log_mppi.cpp"
        build_target "src/quadrotor/gpu new/cluster_mppi.cpp"
        build_target "src/quadrotor/gpu new/bi_mppi.cpp"
        build_target "src/quadrotor/gpu new/svgd_mppi.cpp"
        ;;
    wmrobot)
        # ---- 모든 src/wmrobot/gpu/*.cpp 예제 ----
        BUILD_PREFIX="wmrobot_"
        for SRC in src/wmrobot/gpu/*.cpp; do
            STEM="$(basename "${SRC%.cpp}")"
            if [ "$STEM" = "bi_mppi_SE(2)" ]; then
                build_target_named "$SRC" "wmrobot_bi_mppi_se2"
            else
                build_target "$SRC"
            fi
        done
        ;;
    wmrobot_accel)
        # ---- Compatibility alias for the unified selectable CUDA solver ----
        BUILD_PREFIX="wmrobot_"
        for SRC in src/wmrobot/gpu/*.cpp; do
            STEM="$(basename "${SRC%.cpp}")"
            if [ "$STEM" = "bi_mppi_SE(2)" ]; then
                build_target_named "$SRC" "wmrobot_bi_mppi_se2"
            else
                build_target "$SRC"
            fi
        done
        ;;
    wmrobot_log_mppi)
        # ---- WMRobot Log-MPPI GPU 단일 예제 ----
        BUILD_PREFIX="wmrobot_"
        build_target src/wmrobot/gpu/log_mppi.cpp
        ;;
    wmrobot_ablation)
        # ---- WMRobot Ablation Study GPU 버전 ----
        BUILD_PREFIX=""
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_mppi.cpp"
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_cluster_mppi.cpp"
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_bic_without_guide.cpp"
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_bic_without_backward.cpp"
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_bic_without_clustering.cpp"
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_full_bic_mppi.cpp"
        build_target "src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_all.cpp"
        ;;
    wmrobot_same_budget)
        # ---- WMRobot Sampling-Budget Scaling GPU 버전 ----
        BUILD_PREFIX="wmrobot_same_budget_"
        build_target "src/wmrobot/gpu same budget/mppi.cpp"
        build_target "src/wmrobot/gpu same budget/log_mppi.cpp"
        build_target "src/wmrobot/gpu same budget/cluster_mppi.cpp"
        build_target "src/wmrobot/gpu same budget/bic_raw_connect.cpp"
        build_target "src/wmrobot/gpu same budget/full_bic_mppi.cpp"
        build_target "src/wmrobot/gpu same budget/all.cpp"
        ;;
    wmrobot_waypoint)
        # ---- WMRobot Waypoint-Parallel BiC GPU 예제 ----
        BUILD_PREFIX="wmrobot_waypoint_"
        build_target "src/wmrobot/gpu_waypoint/comparison.cpp"
        ;;
    wmrobot_waypoint_stitched)
        # ---- WMRobot 5-map stitched BARN waypoint GPU 예제 ----
        BUILD_PREFIX="wmrobot_waypoint_"
        rm -f "$GPU_BUILD_DIR/wmrobot_waypoint_svgd_mppi"
        build_target "src/wmrobot/gpu_waypoint/mppi.cpp"
        build_target "src/wmrobot/gpu_waypoint/log_mppi.cpp"
        build_target "src/wmrobot/gpu_waypoint/cluster_mppi.cpp"
        build_target "src/wmrobot/gpu_waypoint/bi_mppi.cpp"
        ;;
    wmrobot_waypoint_stitched_sequential)
        # ---- WMRobot 5-map stitched BARN sequential-waypoint GPU 예제 ----
        BUILD_PREFIX="wmrobot_waypoint_seq_"
        rm -f "$GPU_BUILD_DIR/wmrobot_waypoint_seq_svgd_mppi"
        build_target "src/wmrobot/gpu_waypoint_순차추종/mppi.cpp"
        build_target "src/wmrobot/gpu_waypoint_순차추종/log_mppi.cpp"
        build_target "src/wmrobot/gpu_waypoint_순차추종/cluster_mppi.cpp"
        build_target "src/wmrobot/gpu_waypoint_순차추종/bi_mppi.cpp"
        ;;
    wmrobot_map_285)
        # ---- WMRobot map 285 4개 시각화 예제 ----
        build_target_named src/wmrobot/map_285/mppi.cpp vis_mppi
        build_target_named src/wmrobot/map_285/log_mppi.cpp vis_log_mppi
        build_target_named src/wmrobot/map_285/cluster_mppi.cpp vis_cluster_mppi
        build_target_named src/wmrobot/map_285/bi_mppi.cpp vis_bi_mppi
        ;;
    velo)
        # ---- Velo GPU 버전 ----
        BUILD_PREFIX="velo_"
        build_target src/velo/gpu/mppi.cpp
        build_target src/velo/gpu/log_mppi.cpp
        build_target src/velo/gpu/cluster_mppi.cpp
        build_target src/velo/gpu/bi_mppi.cpp
        build_target src/velo/gpu/svgd_mppi.cpp
        ;;
    *)
        echo "ERROR: unknown GPU target group: $TARGET_GROUP"
        echo "Usage: bash build_gpu.sh [manipulator|manipulator_workspace_collision|manipulator_grid_wall_workspace_example|manipulator_grid_approach_workspace_example|quadrotor|quadrotor_log_mppi|quadrotor_gpu_new|wmrobot|wmrobot_accel|wmrobot_log_mppi|wmrobot_ablation|wmrobot_same_budget|wmrobot_waypoint|wmrobot_waypoint_stitched|wmrobot_waypoint_stitched_sequential|wmrobot_map_285|velo]"
        exit 1
        ;;
esac

echo ""
echo "=== 빌드 완료 ==="
if [ "$TARGET_GROUP" = "manipulator" ]; then
    echo "실행: cd build && ./gpu/mppi"
    echo "      cd build && ./gpu/bi_mppi"
    echo "      cd build && ./gpu/cluster_mppi"
elif [ "$TARGET_GROUP" = "manipulator_workspace_collision" ] || \
     [ "$TARGET_GROUP" = "manipulator_bicmppi_workspace_example" ] || \
     [ "$TARGET_GROUP" = "manipulator_grid_wall_workspace_example" ] || \
     [ "$TARGET_GROUP" = "manipulator_grid_approach_workspace_example" ]; then
    echo "실행: cd build && ./gpu/manipulator_mppi_workspace_example"
    echo "      cd build && ./gpu/manipulator_logmppi_workspace_example"
    echo "      cd build && ./gpu/manipulator_clustermppi_workspace_example"
    echo "      cd build && ./gpu/manipulator_bicmppi_workspace_example"
    echo "      cd build && ./gpu/manipulator_mppi_pinkNplace_workspace_example"
    echo "      cd build && ./gpu/manipulator_logmppi_pinkNplace_workspace_example"
    echo "      cd build && ./gpu/manipulator_clustermppi_pinkNplace_workspace_example"
    echo "      cd build && ./gpu/manipulator_bicmppi_pinkNplace_workspace_example"
    echo "      cd build && ./gpu/manipulator_mppi_random_pose_benchmark"
    echo "      cd build && ./gpu/manipulator_logmppi_random_pose_benchmark"
    echo "      cd build && ./gpu/manipulator_clustermppi_random_pose_benchmark"
    echo "      cd build && ./gpu/manipulator_bicmppi_random_pose_benchmark"
    echo "      cd build && ./gpu/manipulator_grid_wall_workspace_example --init 0 --goal 8"
    echo "      cd build && ./gpu/manipulator_grid_approach_workspace_example --goal 8"
    echo "출력: build/manipulator_workspace_{mppi,logmppi,clustermppi,bicmppi}_*.csv"
    echo "      build/manipulator_pinkNplace_{mppi,logmppi,clustermppi,bicmppi}_*.csv"
    echo "      result/manipulator/"
    echo "      build/manipulator_grid_wall_bicmppi_*.csv"
    echo "      build/manipulator_grid_approach_bicmppi_*.csv"
elif [ "$TARGET_GROUP" = "quadrotor_log_mppi" ]; then
    echo "실행: cd build && ./gpu/quadrotor_log_mppi"
elif [ "$TARGET_GROUP" = "quadrotor_gpu_new" ]; then
    echo "실행: cd build && ./gpu/quadrotor_gpu_new_mppi --smoke"
    echo "      cd build && ./gpu/quadrotor_gpu_new_bi_mppi --smoke"
elif [ "$TARGET_GROUP" = "wmrobot" ]; then
    echo "실행: cd build && ./gpu/wmrobot_cluster_mppi --smoke --clustering dbscan"
    echo "      cd build && ./gpu/wmrobot_bi_mppi --smoke --clustering dbscan"
    echo "      cd build && ./gpu/wmrobot_cluster_mppi_kmeans --smoke"
    echo "      cd build && ./gpu/wmrobot_bi_mppi_kmeans --smoke"
elif [ "$TARGET_GROUP" = "wmrobot_ablation" ]; then
    echo "실행: cd build && ./gpu/wmrobot_ablation_gpu_all --smoke"
    echo "      cd build && ./gpu/wmrobot_ablation_gpu_mppi"
    echo "      cd build && ./gpu/wmrobot_ablation_gpu_full_bic_mppi"
elif [ "$TARGET_GROUP" = "wmrobot_log_mppi" ]; then
    echo "실행: cd build && ./gpu/wmrobot_log_mppi"
elif [ "$TARGET_GROUP" = "wmrobot_accel" ]; then
    echo "실행: cd build && ./gpu-accel/wmrobot_mppi --smoke"
    echo "      cd build && ./gpu-accel/wmrobot_cluster_mppi --smoke --clustering kmeans"
    echo "      cd build && ./gpu-accel/wmrobot_bi_mppi --smoke --clustering dbscan"
elif [ "$TARGET_GROUP" = "wmrobot_same_budget" ]; then
    echo "실행: cd build && ./gpu/wmrobot_same_budget_all --smoke"
    echo "      cd build && ./gpu/wmrobot_same_budget_mppi --smoke"
    echo "      cd build && ./gpu/wmrobot_same_budget_log_mppi --smoke"
    echo "      cd build && ./gpu/wmrobot_same_budget_full_bic_mppi --smoke"
elif [ "$TARGET_GROUP" = "wmrobot_waypoint" ]; then
    echo "실행: cd build && ./gpu/wmrobot_waypoint_comparison --smoke --overwrite"
elif [ "$TARGET_GROUP" = "wmrobot_waypoint_stitched" ]; then
    echo "실행: cd build && ./gpu/wmrobot_waypoint_mppi --smoke --overwrite"
    echo "      cd build && ./gpu/wmrobot_waypoint_bi_mppi --smoke --overwrite"
elif [ "$TARGET_GROUP" = "wmrobot_waypoint_stitched_sequential" ]; then
    echo "실행: cd build && ./gpu/wmrobot_waypoint_seq_mppi --smoke --overwrite"
    echo "      cd build && ./gpu/wmrobot_waypoint_seq_bi_mppi --smoke --overwrite"
elif [ "$TARGET_GROUP" = "wmrobot_map_285" ]; then
    echo "실행: cd build && ./gpu/vis_mppi && ./gpu/vis_log_mppi"
    echo "      cd build && ./gpu/vis_cluster_mppi && ./gpu/vis_bi_mppi"
else
    echo "실행: cd build && ./gpu/${TARGET_GROUP}_mppi"
    echo "      cd build && ./gpu/${TARGET_GROUP}_bi_mppi"
    echo "      cd build && ./gpu/${TARGET_GROUP}_cluster_mppi"
fi
echo ""
if [ "$TARGET_GROUP" != "manipulator_workspace_collision" ] && \
   [ "$TARGET_GROUP" != "manipulator_bicmppi_workspace_example" ] && \
   [ "$TARGET_GROUP" != "manipulator_grid_wall_workspace_example" ] && \
   [ "$TARGET_GROUP" != "manipulator_grid_approach_workspace_example" ]; then
    echo "시각화: cd build && ./gpu/vis_mppi && python3 visualize_mppi.py vis_data/mppi/"
    echo "        cd build && ./gpu/vis_bi_mppi && python3 visualize_mppi.py vis_data/bi_mppi/"
fi
