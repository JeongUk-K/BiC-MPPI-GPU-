#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
GPU_DIR="$BUILD_DIR/gpu"
OUTPUT_DIR="${MANIPULATOR_BENCHMARK_OUTPUT:-$ROOT/result/manipulator}"
CXX="${CXX:-g++}"
NO_BUILD=0

usage() {
  echo "Usage: $0 [--no-build]"
  echo "  Builds and runs every random_pose_benchmark MPPI variant."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) NO_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi

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

build() {
  echo "[build] recompiling manipulator GPU solvers and benchmarks..."
  rm -rf "$GPU_DIR"/*
  find_cuda
  mkdir -p "$GPU_DIR"
  local -a host_flags=()
  command -v g++-10 >/dev/null 2>&1 && host_flags=(-ccbin g++-10)
  local -a cuda_flags=(-O3 --std=c++14 -arch=sm_86 "${host_flags[@]}"
                       --expt-relaxed-constexpr -Xcompiler -fopenmp)
  local -a includes=(-I"$ROOT/src/manipulator/manipulator_workspace_collision_package/include"
                    -I"$ROOT/src/manipulator/random_pose_benchmark"
                    -I"$ROOT/src/manipulator/gpu"
                    -I"$ROOT/mppi" -I"$ROOT/mppi/cuda-accel"
                    -I"$ROOT/mppi/cpu-legacy" -I"$ROOT/model"
                    -I"$ROOT/include/fastsc" -I"$ROOT/include/EigenRand"
                    -I"$ROOT/include/matplotlibcpp" -I"$CUDA_INCLUDE"
                    $(python3-config --includes)
                    $(pkg-config --cflags eigen3 2>/dev/null || echo -I/usr/include/eigen3))
  local source stem object
  declare -A objects
  for source in \
      "$ROOT/src/manipulator/manipulator_workspace_collision_package/src/mppi_gpu.cu" \
      "$ROOT/src/manipulator/manipulator_workspace_collision_package/src/log_mppi_gpu.cu" \
      "$ROOT/src/manipulator/manipulator_workspace_collision_package/src/cluster_mppi_gpu.cu" \
      "$ROOT/src/manipulator/manipulator_workspace_collision_package/src/bi_mppi_gpu.cu" \
      "$ROOT/src/manipulator/manipulator_workspace_collision_package/src/bi_mppi_gpu_legacy.cu" \
      "$ROOT/include/fastsc/labels.cu"; do
    stem="$(basename "${source%.cu}")"
    object="$GPU_DIR/manipulator_${stem}.o"
    echo "[build] $source"
    nvcc "${cuda_flags[@]}" "${includes[@]}" -c "$source" -o "$object"
    objects["$stem"]="$object"
  done

  local -a cxx_flags=(-O3 -std=c++17 -fopenmp)
  local -a link_flags=(-fopenmp -Wl,-rpath,"$CUDA_LIBDIR"
                       "$CUDART_LIB" "$CURAND_LIB" "$CUBLAS_LIB" -lzstd
                       $(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags))
  local package="$ROOT/src/manipulator/manipulator_workspace_collision_package"
  local benchmark="$ROOT/src/manipulator/random_pose_benchmark"
  local gpu="$ROOT/src/manipulator/gpu"
  local -a ws_includes=("${includes[@]}" -I"$package/include" -I"$benchmark")

  # Standard Manipulator GPU Solvers
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$gpu/mppi.cpp" "${objects[mppi_gpu]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_mppi"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$gpu/log_mppi.cpp" "${objects[mppi_gpu]}" "${objects[log_mppi_gpu]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_log_mppi"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$gpu/cluster_mppi.cpp" "${objects[mppi_gpu]}" "${objects[cluster_mppi_gpu]}" "${objects[labels]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_cluster_mppi"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$gpu/bi_mppi.cpp" "${objects[mppi_gpu]}" "${objects[bi_mppi_gpu]}" "${objects[bi_mppi_gpu_legacy]}" "${objects[labels]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_bi_mppi"

  # Random Pose Benchmark Examples
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$benchmark/manipulator_mppi_random_pose_benchmark.cpp" \
    "${objects[mppi_gpu]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_mppi_random_pose_benchmark"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$benchmark/manipulator_logmppi_random_pose_benchmark.cpp" \
    "${objects[mppi_gpu]}" "${objects[log_mppi_gpu]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_logmppi_random_pose_benchmark"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$benchmark/manipulator_clustermppi_random_pose_benchmark.cpp" \
    "${objects[mppi_gpu]}" "${objects[cluster_mppi_gpu]}" "${objects[labels]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_clustermppi_random_pose_benchmark"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$benchmark/manipulator_bicmppi_random_pose_benchmark.cpp" \
    "${objects[bi_mppi_gpu]}" "${objects[bi_mppi_gpu_legacy]}" "${objects[labels]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_bicmppi_random_pose_benchmark"
  "$CXX" "${cxx_flags[@]}" "${ws_includes[@]}" \
    "$benchmark/manipulator_bicmppi_legacy_random_pose_benchmark.cpp" \
    "${objects[bi_mppi_gpu_legacy]}" "${objects[labels]}" "${link_flags[@]}" \
    -o "$GPU_DIR/manipulator_bicmppi_legacy_random_pose_benchmark"
}

if [[ "$NO_BUILD" -eq 0 ]]; then
  build
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/manipulator_random_pose_benchmark_all_stats.csv"

solver_keys=(mppi logmppi clustermppi bicmppi bicmppi_legacy)
solver_executables=(
  manipulator_mppi_random_pose_benchmark
  manipulator_logmppi_random_pose_benchmark
  manipulator_clustermppi_random_pose_benchmark
  manipulator_bicmppi_random_pose_benchmark
  manipulator_bicmppi_legacy_random_pose_benchmark
)

for index in "${!solver_keys[@]}"; do
  solver="${solver_keys[$index]}"
  exe="${solver_executables[$index]}"
  solver_dir="$OUTPUT_DIR/$solver"
  mkdir -p "$solver_dir"
  find "$solver_dir" -maxdepth 1 -type f \
    \( -name 'manipulator_random_pose_benchmark_*.csv' -o -name 'run.log' \) -delete
  find "$solver_dir/trajectories" "$solver_dir/rollout_ee" \
    "$solver_dir/rollout_state" -type f -delete 2>/dev/null || true

  [[ -x "$GPU_DIR/$exe" ]] || { echo "missing executable: $GPU_DIR/$exe" >&2; exit 1; }
  echo "[run] $exe -> $solver_dir"
  (cd "$ROOT" && \
    MANIPULATOR_BENCHMARK_OUTPUT="$solver_dir" "$GPU_DIR/$exe") \
    2>&1 | tee "$solver_dir/run.log"
done

first=1
for solver in "${solver_keys[@]}"; do
  stats="$OUTPUT_DIR/$solver/manipulator_random_pose_benchmark_${solver}_stats.csv"
  [[ -s "$stats" ]] || { echo "missing statistics: $stats" >&2; exit 1; }
  if [[ "$first" -eq 1 ]]; then
    cat "$stats"
    first=0
  else
    tail -n +2 "$stats"
  fi
done > "$OUTPUT_DIR/manipulator_random_pose_benchmark_all_stats.csv"
cp "$OUTPUT_DIR/manipulator_random_pose_benchmark_all_stats.csv" "$OUTPUT_DIR/aggregate_timing_success.csv"

echo ""
echo "Manipulator statistics: $OUTPUT_DIR/aggregate_timing_success.csv"
echo "Manipulator all stats CSV: $OUTPUT_DIR/manipulator_random_pose_benchmark_all_stats.csv"
