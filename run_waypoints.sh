#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/experiment_common.sh"
ORIGINAL_ARGS=("$@")

SCENARIO="sequential"
OUT_DIR=""
DO_BUILD=1
OVERWRITE=0
NO_VIZ=0
SMOKE=0
NO_SAMPLE_DATA=0
ROLLOUT_SCENARIO="0"
ROLLOUT_STEP="first"
MAX_ROLLOUTS="24"
GIF_FPS="20"
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_waypoints.sh [options] [solver options]

WMRobot waypoint runners.  The stitched/sequential scenarios save solver CSVs
and rollout/cluster logger data under vis_data/.  The parallel scenario saves
sampled rollout data under sample_vis_data/ by default.

Options:
  --scenario NAME       parallel, stitched, sequential, sequential_variants,
                        or all. Default: sequential
  --out DIR             Output directory. Default: results/waypoints/<scenario>
  --overwrite           Replace output directory and solver outputs.
  --no-build            Use existing build/gpu binaries.
  --no-viz              Skip PNG/GIF post-processing, but still save raw data.
  --smoke               Forward a short validation run to supporting binaries.
  --no-sample-data      For parallel only, skip sampled rollout export.
  --rollout-scenario N  Kept for compatibility. Default: 0
  --rollout-step STEP   first, middle, last, or step_0000. Default: first
  --max-rollouts N      Max sampled rollouts per group. Default: 24
  --gif-fps N           Stitched/sequential GIF FPS. Default: 20
  -h, --help            Show this help.

Forwarded solver options include:
  --scenarios N --maps-per-scenario N --seed N --global-T N --waypoint-count N
  --bic-Tf N --bic-Tb N --N N --Nf N --Nb N --Nr N --maxiter N --vis-every N
  --no-rollouts

Examples:
  ./run_waypoints.sh --scenario sequential --smoke --overwrite
  ./run_waypoints.sh --scenario sequential_variants --overwrite --gif-fps 20
  ./run_waypoints.sh --scenario all --smoke --overwrite --no-viz
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="$2"; shift 2 ;;
        --out)
            OUT_DIR="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
        --no-build)
            DO_BUILD=0; shift ;;
        --no-viz|--no-visualize)
            NO_VIZ=1; shift ;;
        --smoke)
            SMOKE=1; shift ;;
        --no-sample-data)
            NO_SAMPLE_DATA=1; shift ;;
        --rollout-scenario)
            ROLLOUT_SCENARIO="$2"; shift 2 ;;
        --rollout-step)
            ROLLOUT_STEP="$2"; shift 2 ;;
        --max-rollouts)
            MAX_ROLLOUTS="$2"; shift 2 ;;
        --gif-fps)
            GIF_FPS="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; RUN_ARGS+=("$@"); break ;;
        *)
            RUN_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    if [[ "$SCENARIO" == "all" ]]; then
        OUT_DIR="results/waypoints"
    else
        OUT_DIR="results/waypoints/$SCENARIO"
    fi
fi

has_arg() {
    local needle="$1"
    shift
    local arg
    for arg in "$@"; do
        [[ "$arg" == "$needle" ]] && return 0
    done
    return 1
}

common_solver_args() {
    local args=()
    [[ "$OVERWRITE" -eq 1 ]] && args+=("--overwrite")
    [[ "$SMOKE" -eq 1 ]] && args+=("--smoke")
    args+=("${RUN_ARGS[@]}")
    if [[ "${#args[@]}" -gt 0 ]]; then
        printf '%s\n' "${args[@]}"
    fi
}

ensure_executable() {
    local exe="$1"
    [[ -x "$GPU_DIR/$exe" ]] || {
        echo "ERROR: missing executable: $GPU_DIR/$exe" >&2
        exit 1
    }
}

find_cuda_for_variants() {
    NVCC=""
    CUDA_INCLUDE=""
    CUDA_LIBDIR=""
    MATLAB_LIBDIR=""

    local d
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
        local nvcc_root
        nvcc_root="$(cd "$(dirname "$NVCC")/.." && pwd)"
        if [[ -f "$nvcc_root/include/cuda_runtime.h" ]]; then
            CUDA_INCLUDE="$nvcc_root/include"
            if [[ -d "$nvcc_root/lib64" ]]; then
                CUDA_LIBDIR="$nvcc_root/lib64"
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
        echo "ERROR: nvcc not found. Install CUDA toolkit first." >&2
        exit 1
    fi

    local nvcc_bin_dir nvcc_root
    nvcc_bin_dir="$(cd "$(dirname "$NVCC")" && pwd)"
    nvcc_root="$(cd "$nvcc_bin_dir/.." && pwd)"
    export PATH="$nvcc_bin_dir:$nvcc_root/nvvm/bin:$PATH"

    case "$NVCC" in
        /usr/local/MATLAB/*/sys/cuda/glnxa64/cuda/bin/nvcc)
            MATLAB_ROOT="${NVCC%/sys/cuda/glnxa64/cuda/bin/nvcc}"
            MATLAB_LIBDIR="$MATLAB_ROOT/bin/glnxa64"
            ;;
    esac

    CURAND_LIB=""
    local lib
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

build_sequential_variants() {
    find_cuda_for_variants

    mkdir -p "$GPU_DIR"

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
    local cxx="g++"
    local cxxflags="-O3 -std=c++17 -fopenmp"
    local arch="-arch=sm_86"
    local ccbin_flag=""
    if command -v g++-10 >/dev/null 2>&1; then
        ccbin_flag="-ccbin g++-10"
    fi
    local nvccflags="-O3 --std=c++14 $arch $ccbin_flag --expt-relaxed-constexpr -Xcompiler -fopenmp"
    local ldflags="-fopenmp"
    [[ -n "$CUDART_LIB" ]] && ldflags="$ldflags $CUDART_LIB"
    [[ -n "$CURAND_LIB" ]] && ldflags="$ldflags $CURAND_LIB"
    ldflags="$ldflags $py_ldflags"
    local rpath_dir
    rpath_dir="$(dirname "${CUDART_LIB:-/dev/null}")"

    echo "=== Sequential variants GPU build ==="
    echo "  nvcc  : $NVCC"
    echo "  output: build/gpu/wmrobot_waypoint_seq_var_*"

    (cd "$REPO_ROOT" && "$NVCC" $nvccflags $includes -c mppi/cuda/mppi_gpu.cu -o "$GPU_DIR/mppi_gpu.o")
    (cd "$REPO_ROOT" && "$NVCC" $nvccflags $includes -c mppi/cuda/cluster_mppi_gpu.cu -o "$GPU_DIR/cluster_mppi_gpu.o")
    (cd "$REPO_ROOT" && "$NVCC" $nvccflags $includes -c mppi/cuda/bi_mppi_gpu.cu -o "$GPU_DIR/bi_mppi_gpu.o")
    (cd "$REPO_ROOT" && "$NVCC" $nvccflags $includes -c mppi/cuda/log_mppi_gpu.cu -o "$GPU_DIR/log_mppi_gpu.o")

    rm -f "$GPU_DIR/libmppi_gpu.a"
    ar rcs "$GPU_DIR/libmppi_gpu.a" \
        "$GPU_DIR/mppi_gpu.o" \
        "$GPU_DIR/cluster_mppi_gpu.o" \
        "$GPU_DIR/bi_mppi_gpu.o" \
        "$GPU_DIR/log_mppi_gpu.o"

    build_variant_target() {
        local src="$1"
        local name="wmrobot_waypoint_seq_var_$(basename "${src%.cpp}")"
        echo "[build] $name"
        (cd "$REPO_ROOT" && "$cxx" $cxxflags $includes "$src" \
            -L"$GPU_DIR" -lmppi_gpu \
            -Wl,-rpath,"$rpath_dir" \
            $ldflags \
            -o "$GPU_DIR/$name")
    }

    rm -f "$GPU_DIR"/wmrobot_waypoint_seq_var_*
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/mppi.cpp"
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/log_mppi.cpp"
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/cluster_mppi.cpp"
    build_variant_target "src/wmrobot/gpu_waypoint_순차추종_variants/bi_mppi.cpp"
}

copy_stitched_vis_data() {
    local out="$1"
    rm -rf "$out/vis_data"
    mkdir -p "$out/vis_data"
    if compgen -G "$BUILD_DIR/vis_data/stitched_*_scenario_*" >/dev/null; then
        find "$BUILD_DIR/vis_data" -maxdepth 1 -type d -name 'stitched_*_scenario_*' \
            -exec cp -R {} "$out/vis_data/" \;
    else
        rmdir "$out/vis_data"
        copy_build_vis_data "$out"
    fi
}

visualize_stitched() {
    local out="$1"
    python3 "$REPO_ROOT/visualize_wmrobot_waypoint_stitched.py" \
        --dir "$out" \
        --rollout-scenario "$ROLLOUT_SCENARIO" \
        --rollout-step "$ROLLOUT_STEP" \
        --max-rollouts "$MAX_ROLLOUTS" \
        --gif-fps "$GIF_FPS"
}

run_parallel() {
    local out="$1"
    mkdir -p "$out"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot_waypoint
    reset_build_outputs
    ensure_executable wmrobot_waypoint_comparison

    local args=()
    mapfile -t args < <(common_solver_args)
    if [[ "$NO_SAMPLE_DATA" -eq 0 ]] && ! has_arg "--vis-samples" "${args[@]}"; then
        args+=("--vis-samples")
    fi

    echo "[run] wmrobot_waypoint_comparison"
    (cd "$BUILD_DIR" && ./gpu/wmrobot_waypoint_comparison --out "$out" "${args[@]}")

    if compgen -G "$BUILD_DIR/vis_data/wmrobot_waypoint_*" >/dev/null; then
        rm -rf "$out/sample_vis_data"
        mkdir -p "$out/sample_vis_data"
        find "$BUILD_DIR/vis_data" -maxdepth 1 -type d -name 'wmrobot_waypoint_*' \
            -exec cp -a {} "$out/sample_vis_data/" \;
    fi

    if [[ "$NO_VIZ" -eq 0 ]]; then
        python3 "$REPO_ROOT/visualize_wmrobot_waypoint_parallel.py" --dir "$out"
        if [[ -d "$out/sample_vis_data" ]]; then
            python3 "$REPO_ROOT/animate_wmrobot_waypoint_sampling.py" \
                --dir "$out" --vis-root "$out/sample_vis_data"
        fi
    fi
}

run_stitched() {
    local out="$1"
    mkdir -p "$out"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot_waypoint_stitched
    reset_build_outputs

    local exes=(
        wmrobot_waypoint_mppi
        wmrobot_waypoint_log_mppi
        wmrobot_waypoint_cluster_mppi
        wmrobot_waypoint_bi_mppi
    )
    run_stitched_exes "$out" "${exes[@]}"
}

run_sequential() {
    local out="$1"
    mkdir -p "$out"
    [[ "$DO_BUILD" -eq 1 ]] && bash "$REPO_ROOT/build_gpu.sh" wmrobot_waypoint_stitched_sequential
    reset_build_outputs

    local exes=(
        wmrobot_waypoint_seq_mppi
        wmrobot_waypoint_seq_log_mppi
        wmrobot_waypoint_seq_cluster_mppi
        wmrobot_waypoint_seq_bi_mppi
    )
    run_stitched_exes "$out" "${exes[@]}"
}

run_sequential_variants() {
    local out="$1"
    mkdir -p "$out"
    [[ "$DO_BUILD" -eq 1 ]] && build_sequential_variants
    reset_build_outputs

    local exes=(
        wmrobot_waypoint_seq_var_mppi
        wmrobot_waypoint_seq_var_log_mppi
        wmrobot_waypoint_seq_var_cluster_mppi
        wmrobot_waypoint_seq_var_bi_mppi
    )
    run_stitched_exes "$out" "${exes[@]}"
}

run_stitched_exes() {
    local out="$1"
    shift
    local exes=("$@")
    local exe
    for exe in "${exes[@]}"; do
        ensure_executable "$exe"
    done

    if [[ -d "$BUILD_DIR/vis_data" ]]; then
        find "$BUILD_DIR/vis_data" -maxdepth 1 -type d -name 'stitched_*_scenario_*' \
            -exec rm -rf {} +
    fi

    local args=()
    mapfile -t args < <(common_solver_args)
    for exe in "${exes[@]}"; do
        echo "[run] $exe"
        (cd "$BUILD_DIR" && "./gpu/$exe" --out "$out" "${args[@]}")
    done

    copy_stitched_vis_data "$out"
    if [[ "$NO_VIZ" -eq 0 ]]; then
        visualize_stitched "$out"
    fi
}

OUT_ABS="$(repo_out_path "$OUT_DIR")"
prepare_output_dir "$OUT_ABS" "$OVERWRITE"
write_run_manifest "$OUT_ABS" "$0" "${ORIGINAL_ARGS[@]}"

scenario_out() {
    local name="$1"
    if [[ "$SCENARIO" == "all" ]]; then
        printf '%s/%s\n' "$OUT_ABS" "$name"
    else
        printf '%s\n' "$OUT_ABS"
    fi
}

case "$SCENARIO" in
    parallel)
        run_parallel "$(scenario_out parallel)" ;;
    stitched)
        run_stitched "$(scenario_out stitched)" ;;
    sequential)
        run_sequential "$(scenario_out sequential)" ;;
    sequential_variants)
        run_sequential_variants "$(scenario_out sequential_variants)" ;;
    all)
        run_parallel "$(scenario_out parallel)"
        run_stitched "$(scenario_out stitched)"
        run_sequential "$(scenario_out sequential)"
        run_sequential_variants "$(scenario_out sequential_variants)" ;;
    *)
        echo "ERROR: unsupported waypoint scenario: $SCENARIO" >&2
        usage >&2
        exit 1 ;;
esac

aggregate_timing_success_csv "$OUT_ABS" "$OUT_ABS/aggregate_timing_success.csv"
write_outputs_manifest "$OUT_ABS"

echo ""
echo "Outputs:"
echo "  $OUT_ABS"
echo "  $OUT_ABS/aggregate_timing_success.csv"
