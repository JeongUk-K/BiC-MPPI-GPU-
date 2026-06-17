#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_ROOT="$SCRIPT_DIR/build"
GPU_BUILD_DIR="$BUILD_ROOT/gpu"

BUILD=1
VISUALIZE=1
OVERWRITE=0
OUT_DIR="results/wmrobot_waypoint_stitched_sequential_variants"
ROLLOUT_SCENARIO="0"
ROLLOUT_STEP="first"
MAX_ROLLOUTS="24"
GIF_FPS="20"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)
            BUILD=0
            shift
            ;;
        --no-viz|--no-visualize)
            VISUALIZE=0
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --out)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --out requires a directory." >&2
                exit 1
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        --rollout-scenario)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --rollout-scenario requires a value." >&2
                exit 1
            fi
            ROLLOUT_SCENARIO="$2"
            shift 2
            ;;
        --rollout-step)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --rollout-step requires a value." >&2
                exit 1
            fi
            ROLLOUT_STEP="$2"
            shift 2
            ;;
        --max-rollouts)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --max-rollouts requires a value." >&2
                exit 1
            fi
            MAX_ROLLOUTS="$2"
            shift 2
            ;;
        --gif-fps)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --gif-fps requires a value." >&2
                exit 1
            fi
            GIF_FPS="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
Usage: ./run_wmrobot_waypoint_stitched_sequential_variants.sh [options] [solver options]

This single script builds the gpu_waypoint_순차추종_variants binaries, runs all
four solvers, and optionally generates visualizations.

Common options:
  --overwrite              Remove existing output before running.
  --no-build               Skip compilation and use existing build/gpu/wmrobot_waypoint_seq_var_* binaries.
  --no-viz                 Skip PNG/GIF visualization generation.
  --out DIR                Output directory. Default: results/wmrobot_waypoint_stitched_sequential_variants
  --rollout-scenario N     Kept for compatibility; rollout PNG/GIF outputs cover all scenarios.
  --rollout-step STEP      first, middle, last, or step_0000. Default: first
  --max-rollouts N         Max sampled rollout trajectories per group. Default: 24
  --gif-fps N              GIF frames per second. Default: 20

Forwarded solver options:
  Default parameters live in src/wmrobot/gpu_waypoint_순차추종_variants/stitched_barn_common.h
  --smoke                  One short scenario for validation.
  --scenarios N            Default: 10
  --maps-per-scenario N    Default: 5
  --seed N                 Random map-selection seed.
  --global-T N             MPPI/Log/Cluster per-waypoint horizon, capped at 100.
  --waypoint-count N       MPPI sequential waypoints / BiC waypoint rollout targets.
  --bic-Tf N --bic-Tb N    BiC forward/backward rollout horizon. Default: 50/50
  --N N --Nf N --Nb N --Nr N --maxiter N
  --vis-every N            Save rollout data every N closed-loop iterations.
                           Default is 1 for this sequential visualization.
  --no-rollouts            Save CSV paths only.
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

find_cuda() {
    NVCC=""
    CUDA_INCLUDE=""
    CUDA_LIBDIR=""
    MATLAB_LIBDIR=""

    for d in /usr/local/cuda /usr/local/cuda-12.5 /usr/local/cuda-12.6 /usr/local/cuda-13; do
        if [[ -f "$d/bin/nvcc" ]]; then
            NVCC="$d/bin/nvcc"
            CUDA_INCLUDE="$d/include"
            CUDA_LIBDIR="$d/lib64"
            break
        fi
    done

    if [[ -z "$NVCC" ]] && command -v nvcc >/dev/null 2>&1; then
        NVCC="$(command -v nvcc)"
        NVCC_ROOT="$(cd "$(dirname "$NVCC")/.." && pwd)"
        if [[ -f "$NVCC_ROOT/include/cuda_runtime.h" ]]; then
            CUDA_INCLUDE="$NVCC_ROOT/include"
            if [[ -d "$NVCC_ROOT/lib64" ]]; then
                CUDA_LIBDIR="$NVCC_ROOT/lib64"
            else
                CUDA_LIBDIR="/usr/lib/x86_64-linux-gnu"
            fi
        else
            CUDA_INCLUDE="/usr/include"
            CUDA_LIBDIR="/usr/lib/x86_64-linux-gnu"
        fi
    fi

    if [[ -z "$NVCC" ]]; then
        for d in /usr/local/MATLAB/R*/sys/cuda/glnxa64/cuda; do
            if [[ -f "$d/bin/nvcc" ]]; then
                NVCC="$d/bin/nvcc"
                CUDA_INCLUDE="$d/include"
                CUDA_LIBDIR="$d/lib64"
                break
            fi
        done
    fi

    if [[ -z "$NVCC" ]]; then
        echo "ERROR: nvcc를 찾을 수 없습니다. CUDA toolkit을 설치해 주세요." >&2
        exit 1
    fi

    NVCC_BIN_DIR="$(cd "$(dirname "$NVCC")" && pwd)"
    NVCC_ROOT="$(cd "$NVCC_BIN_DIR/.." && pwd)"
    export PATH="$NVCC_BIN_DIR:$NVCC_ROOT/nvvm/bin:$PATH"

    case "$NVCC" in
        /usr/local/MATLAB/*/sys/cuda/glnxa64/cuda/bin/nvcc)
            MATLAB_ROOT="${NVCC%/sys/cuda/glnxa64/cuda/bin/nvcc}"
            MATLAB_LIBDIR="$MATLAB_ROOT/bin/glnxa64"
            ;;
    esac

    CURAND_LIB=""
    for lib in "$CUDA_LIBDIR"/libcurand.so* "$MATLAB_LIBDIR"/libcurand.so* "/usr/lib/x86_64-linux-gnu"/libcurand.so*; do
        if [[ -f "$lib" ]]; then
            CURAND_LIB="$lib"
            break
        fi
    done

    CUDART_LIB=""
    for lib in "$CUDA_LIBDIR"/libcudart.so* "$MATLAB_LIBDIR"/libcudart.so* "/usr/lib/x86_64-linux-gnu"/libcudart.so*; do
        if [[ -f "$lib" ]]; then
            CUDART_LIB="$lib"
            break
        fi
    done
}

build_variants() {
    find_cuda

    mkdir -p "$GPU_BUILD_DIR"

    local eigen_inc
    eigen_inc="$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")"

    local py_includes
    py_includes="$(python3-config --includes)"

    local py_ldflags
    py_ldflags="$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)"

    local includes
    includes="-I./mppi -I./mppi/cuda -I./mppi/cpu-legacy -I./model \
              -I./include/EigenRand -I./include/matplotlibcpp \
              -I$CUDA_INCLUDE \
              $py_includes \
              $eigen_inc"

    local cxx
    cxx="g++"
    local cxxflags
    cxxflags="-O3 -std=c++17 -fopenmp"
    local arch
    arch="-arch=sm_86"
    local ccbin_flag
    ccbin_flag=""
    if command -v g++-10 >/dev/null 2>&1; then
        ccbin_flag="-ccbin g++-10"
    fi

    local nvccflags
    nvccflags="-O3 --std=c++14 $arch $ccbin_flag --expt-relaxed-constexpr -Xcompiler -fopenmp"

    local ldflags
    ldflags="-fopenmp"
    [[ -n "$CUDART_LIB" ]] && ldflags="$ldflags $CUDART_LIB"
    [[ -n "$CURAND_LIB" ]] && ldflags="$ldflags $CURAND_LIB"
    ldflags="$ldflags $py_ldflags"

    local rpath_dir
    rpath_dir="$(dirname "${CUDART_LIB:-/dev/null}")"

    echo "=== Variants GPU Build ==="
    echo "  nvcc        : $NVCC"
    echo "  CUDA include: $CUDA_INCLUDE"
    echo "  cudart      : ${CUDART_LIB:-NOT FOUND}"
    echo "  curand      : ${CURAND_LIB:-NOT FOUND}"
    echo "  output      : build/gpu/wmrobot_waypoint_seq_var_*"
    echo "=========================="

    echo "[build] Compiling mppi_gpu.cu"
    "$NVCC" $nvccflags $includes -c mppi/cuda/mppi_gpu.cu -o "$GPU_BUILD_DIR/mppi_gpu.o"

    echo "[build] Compiling cluster_mppi_gpu.cu"
    "$NVCC" $nvccflags $includes -c mppi/cuda/cluster_mppi_gpu.cu -o "$GPU_BUILD_DIR/cluster_mppi_gpu.o"

    echo "[build] Compiling bi_mppi_gpu.cu"
    "$NVCC" $nvccflags $includes -c mppi/cuda/bi_mppi_gpu.cu -o "$GPU_BUILD_DIR/bi_mppi_gpu.o"

    echo "[build] Compiling log_mppi_gpu.cu"
    "$NVCC" $nvccflags $includes -c mppi/cuda/log_mppi_gpu.cu -o "$GPU_BUILD_DIR/log_mppi_gpu.o"

    rm -f "$GPU_BUILD_DIR/libmppi_gpu.a"
    ar rcs "$GPU_BUILD_DIR/libmppi_gpu.a" \
        "$GPU_BUILD_DIR/mppi_gpu.o" \
        "$GPU_BUILD_DIR/cluster_mppi_gpu.o" \
        "$GPU_BUILD_DIR/bi_mppi_gpu.o" \
        "$GPU_BUILD_DIR/log_mppi_gpu.o"

    build_variant_target() {
        local src="$1"
        local name="wmrobot_waypoint_seq_var_$(basename "${src%.cpp}")"
        echo "[build] Linking $name"
        "$cxx" $cxxflags $includes "$src" \
            -L"$GPU_BUILD_DIR" -lmppi_gpu \
            -Wl,-rpath,"$rpath_dir" \
            $ldflags \
            -o "$GPU_BUILD_DIR/$name"
    }

    rm -f "$GPU_BUILD_DIR"/wmrobot_waypoint_seq_var_*
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/mppi.cpp"
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/log_mppi.cpp"
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/cluster_mppi.cpp"
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/bi_mppi.cpp"

    echo "=== Variants GPU build complete ==="
}

if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

if [[ "$BUILD" -eq 1 ]]; then
    build_variants
fi

EXES=(
    wmrobot_waypoint_seq_var_mppi
    wmrobot_waypoint_seq_var_log_mppi
    wmrobot_waypoint_seq_var_cluster_mppi
    wmrobot_waypoint_seq_var_bi_mppi
)

for exe in "${EXES[@]}"; do
    if [[ ! -x "build/gpu/$exe" ]]; then
        echo "ERROR: build/gpu/$exe not found. Run without --no-build first." >&2
        exit 1
    fi
done

if [[ "$OUT_DIR" = /* ]]; then
    BUILD_OUT="$OUT_DIR"
else
    BUILD_OUT="../$OUT_DIR"
fi

rm -rf build/vis_data/stitched_*_scenario_*

for exe in "${EXES[@]}"; do
    echo "[run] $exe"
    (cd build && "./gpu/$exe" --out "$BUILD_OUT" "${EXTRA_ARGS[@]}")
done

rm -rf "$OUT_DIR/vis_data"
mkdir -p "$OUT_DIR/vis_data"
if compgen -G "build/vis_data/stitched_*_scenario_*" >/dev/null; then
    find build/vis_data -maxdepth 1 -type d -name 'stitched_*_scenario_*' \
        -exec cp -R {} "$OUT_DIR/vis_data/" \;
fi

if [[ "$VISUALIZE" -eq 1 ]]; then
    python3 visualize_wmrobot_waypoint_stitched.py \
        --dir "$OUT_DIR" \
        --rollout-scenario "$ROLLOUT_SCENARIO" \
        --rollout-step "$ROLLOUT_STEP" \
        --max-rollouts "$MAX_ROLLOUTS" \
        --gif-fps "$GIF_FPS"
fi

echo ""
echo "Outputs:"
echo "  $OUT_DIR"
echo "  $OUT_DIR/{mppi,log_mppi,cluster_mppi,bi_mppi}/timings.csv"
if [[ "$VISUALIZE" -eq 1 ]]; then
    echo "  $OUT_DIR/wmrobot_stitched_aggregate.csv"
    echo "  $OUT_DIR/wmrobot_stitched_timing.png"
    echo "  $OUT_DIR/wmrobot_stitched_paths_scenario_XX.png"
    echo "  $OUT_DIR/wmrobot_stitched_rollouts_scenario_XX.png"
    echo "  $OUT_DIR/rollout_gifs/wmrobot_stitched_{mppi,log_mppi,cluster_mppi,bi_mppi}_scenario_XX.gif"
else
    echo "  visualization skipped (--no-viz)"
fi
