#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
GPU_DIR="$BUILD_DIR/gpu"
OUTPUT_ROOT="${ITER_ABLATION_OUTPUT:-$ROOT/result/iter_ablatin}"
DATASET_DIR="${WMROBOT_DATASET_DIR:-$ROOT/BARN_dataset/txt_files}"
CXX="${CXX:-g++}"
NO_BUILD=0
ITERATIONS=()

usage() {
  echo "Usage: $0 [--no-build] [200 400 800]"
  echo "  Runs all WMRobot and manipulator solvers at each max iteration."
  echo "  Output: result/iter_ablatin/<iteration>/{wmrobot,manipulator}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) NO_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -* ) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    * ) ITERATIONS+=("$1"); shift ;;
  esac
done
[[ ${#ITERATIONS[@]} -gt 0 ]] || ITERATIONS=(200 400 800)
for iteration in "${ITERATIONS[@]}"; do
  [[ "$iteration" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid iteration: $iteration" >&2
    exit 2
  }
done
[[ -d "$DATASET_DIR" ]] || {
  echo "WMRobot dataset not found: $DATASET_DIR" >&2
  exit 1
}

find_cuda() {
  NVCC="${NVCC:-$(command -v nvcc || true)}"
  [[ -n "$NVCC" ]] || { echo "nvcc not found" >&2; exit 1; }
  CUDA_ROOT="$(CDPATH= cd -- "$(dirname -- "$NVCC")/.." && pwd)"
  CUDA_INCLUDE="${CUDA_INCLUDE:-$CUDA_ROOT/include}"
  CUDA_LIBDIR="${CUDA_LIBDIR:-$CUDA_ROOT/lib64}"
  [[ -d "$CUDA_INCLUDE" ]] || CUDA_INCLUDE=/usr/include
  [[ -d "$CUDA_LIBDIR" ]] || CUDA_LIBDIR=/usr/lib/x86_64-linux-gnu
  for name in libcudart.so libcurand.so libcublas.so; do
    local_path=""
    for candidate in "$CUDA_LIBDIR/$name" "/usr/lib/x86_64-linux-gnu/$name"; do
      if [[ -e "$candidate" ]]; then local_path="$candidate"; break; fi
    done
    [[ -n "$local_path" ]] || { echo "missing $name" >&2; exit 1; }
    case "$name" in
      libcudart.so) CUDART_LIB="$local_path" ;;
      libcurand.so) CURAND_LIB="$local_path" ;;
      libcublas.so) CUBLAS_LIB="$local_path" ;;
    esac
  done
}

build_wmrobot() {
  echo "[build] WMRobot GPU solvers"
  find_cuda
  mkdir -p "$GPU_DIR"
  local -a host_flags=()
  command -v g++-10 >/dev/null 2>&1 && host_flags=(-ccbin g++-10)
  local -a cuda_flags=(-O3 --std=c++14 -arch=sm_86 "${host_flags[@]}"
                       --expt-relaxed-constexpr -Xcompiler -fopenmp)
  local -a includes=(-I"$ROOT/mppi" -I"$ROOT/mppi/cuda-accel"
                     -I"$ROOT/mppi/cpu-legacy" -I"$ROOT/model"
                     -I"$ROOT/include"
                     -I"$ROOT/include/fastsc" -I"$ROOT/include/EigenRand"
                     -I"$ROOT/include/matplotlibcpp" -I"$CUDA_INCLUDE"
                     $(python3-config --includes)
                     $(pkg-config --cflags eigen3 2>/dev/null || echo -I/usr/include/eigen3))
  local -a objects=()
  local source stem object
  for source in \
      "$ROOT/mppi/cuda-accel/mppi_gpu.cu" \
      "$ROOT/mppi/cuda-accel/cluster_mppi_gpu.cu" \
      "$ROOT/mppi/cuda-accel/bi_mppi_gpu.cu" \
      "$ROOT/mppi/cuda-accel/log_mppi_gpu.cu" \
      "$ROOT/include/fastsc/labels.cu"; do
    stem="$(basename "${source%.cu}")"
    object="$GPU_DIR/wmrobot_${stem}.o"
    nvcc "${cuda_flags[@]}" "${includes[@]}" -c "$source" -o "$object"
    objects+=("$object")
  done
  ar rcs "$GPU_DIR/libmppi_gpu.a" "${objects[@]}"

  local -a cxx_flags=(-O3 -std=c++17 -fopenmp)
  local -a link_flags=(-fopenmp -Wl,-rpath,"$CUDA_LIBDIR"
                       "$CUDART_LIB" "$CURAND_LIB" "$CUBLAS_LIB"
                       $(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags))
  for source in "$ROOT/src/wmrobot/gpu"/*.cpp; do
    stem="$(basename "${source%.cpp}")"
    "$CXX" "${cxx_flags[@]}" "${includes[@]}" "$source" \
      -L"$GPU_DIR" -lmppi_gpu "${link_flags[@]}" \
      -o "$GPU_DIR/wmrobot_$stem"
  done
}

run_wmrobot() {
  local max_iter="$1"
  local output_dir="$2"
  local -a solvers=(
    mppi log_mppi cluster_mppi_dbscan cluster_mppi_kmeans
    bi_mppi_dbscan bi_mppi_kmeans
  )
  local solver solver_dir exe
  local -a summary_files=()
  mkdir -p "$output_dir"
  for solver in "${solvers[@]}"; do
    exe="$GPU_DIR/wmrobot_$solver"
    [[ -x "$exe" ]] || { echo "missing executable: $exe" >&2; exit 1; }
    solver_dir="$output_dir/$solver"
    mkdir -p "$solver_dir/vis_data"
    find "$solver_dir" -maxdepth 1 -type f \
      \( -name 'result*.csv' -o -name 'run.log' \) -delete
    find "$solver_dir/vis_data" -mindepth 1 -delete
    echo "[run][$max_iter] WMRobot $solver"
    (
      export WMROBOT_RESULT_DIR="$solver_dir"
      export MPPI_VIS_ROOT="$solver_dir/vis_data"
      cd "$BUILD_DIR"
      "$exe" --maxiter "$max_iter" --dataset-dir "$DATASET_DIR" --no-rollouts
    ) 2>&1 | tee "$solver_dir/run.log"
    summary_files+=("$solver_dir"/result_*_summary.csv)
  done

  local aggregate="$output_dir/aggregate_timing_success.csv"
  local first=1 summary
  : > "$aggregate"
  for summary in "${summary_files[@]}"; do
    [[ -s "$summary" ]] || { echo "missing summary: $summary" >&2; exit 1; }
    if [[ "$first" -eq 1 ]]; then
      command cat "$summary" >> "$aggregate"
      first=0
    else
      tail -n +2 "$summary" >> "$aggregate"
    fi
  done
}

if [[ "$NO_BUILD" -eq 0 ]]; then
  build_wmrobot
fi

first_iteration=1
for iteration in "${ITERATIONS[@]}"; do
  iteration_dir="$OUTPUT_ROOT/$iteration"
  run_wmrobot "$iteration" "$iteration_dir/wmrobot"

  echo "[run][$iteration] Manipulator all solvers"
  manipulator_args=(all)
  if [[ "$NO_BUILD" -eq 1 || "$first_iteration" -eq 0 ]]; then
    manipulator_args+=(--no-build)
  fi
  MANIPULATOR_BENCHMARK_OUTPUT="$iteration_dir/manipulator" \
  MANIPULATOR_BENCHMARK_MAX_ITER="$iteration" \
    "$ROOT/manipulator.sh" "${manipulator_args[@]}"
  first_iteration=0
done

echo
echo "Iteration-ablation results: $OUTPUT_ROOT/{${ITERATIONS[*]}}"
