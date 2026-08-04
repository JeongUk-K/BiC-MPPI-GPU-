#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
GPU_DIR="$BUILD_DIR/gpu"
OUTPUT_DIR="${WMROBOT_OUTPUT:-$ROOT/result}"
CXX="${CXX:-g++}"
NO_BUILD=0

usage() {
  echo "Usage: $0 [--no-build]"
  echo "  Builds and runs every src/wmrobot/gpu/*.cpp example."
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
  find_cuda
  mkdir -p "$GPU_DIR"
  local -a host_flags=()
  command -v g++-10 >/dev/null 2>&1 && host_flags=(-ccbin g++-10)
  local -a cuda_flags=(-O3 --std=c++14 -arch=sm_86 "${host_flags[@]}"
                       --expt-relaxed-constexpr -Xcompiler -fopenmp)
  local -a includes=(-I"$ROOT/mppi" -I"$ROOT/mppi/cuda-accel"
                    -I"$ROOT/mppi/cpu-legacy" -I"$ROOT/model"
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
    echo "[build] $source"
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
    echo "[build] $source"
    "$CXX" "${cxx_flags[@]}" "${includes[@]}" "$source" \
      -L"$GPU_DIR" -lmppi_gpu "${link_flags[@]}" \
      -o "$GPU_DIR/wmrobot_$stem"
  done
}

if [[ "$NO_BUILD" -eq 0 ]]; then
  build
fi

mkdir -p "$OUTPUT_DIR"
find "$BUILD_DIR" -maxdepth 1 -type f -name 'result*.csv' -delete
if [[ -d "$BUILD_DIR/vis_data" ]]; then
  find "$BUILD_DIR/vis_data" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  rmdir "$BUILD_DIR/vis_data" 2>/dev/null || true
fi
find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'aggregate_timing_success.csv' -delete

shopt -s nullglob
ran=0
solver_dirs=()
for source in "$ROOT/src/wmrobot/gpu"/*.cpp; do
  stem="$(basename "${source%.cpp}")"
  exe="$GPU_DIR/wmrobot_$stem"
  [[ -x "$exe" ]] || { echo "missing executable: $exe" >&2; exit 1; }
  solver_dir="$OUTPUT_DIR/$stem"
  mkdir -p "$solver_dir/vis_data"
  find "$solver_dir" -maxdepth 1 -type f -name 'result*.csv' -delete
  find "$solver_dir" -maxdepth 1 -type f -name 'run.log' -delete
  find "$solver_dir/vis_data" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  solver_dirs+=("$solver_dir")
  echo "[run] $exe"
  args=()
  [[ "${WMROBOT_SMOKE:-0}" == 1 ]] && args+=(--smoke)
  (
    export WMROBOT_RESULT_DIR="$solver_dir"
    export MPPI_VIS_ROOT="$solver_dir/vis_data"
    cd "$BUILD_DIR"
    "$exe" "${args[@]}"
  ) 2>&1 | tee "$solver_dir/run.log"
  ran=1
done
[[ "$ran" -eq 1 ]] || { echo "no WMRobot examples found" >&2; exit 1; }

summary_files=()
for solver_dir in "${solver_dirs[@]}"; do
  solver_summary_files=("$solver_dir"/result_*_summary.csv)
  summary_files+=("${solver_summary_files[@]}")
done
[[ -e "${summary_files[0]}" ]] || { echo "no summary CSV generated" >&2; exit 1; }
{
  first=1
  for summary in "${summary_files[@]}"; do
    if [[ "$first" -eq 1 ]]; then
      cat "$summary"
      first=0
    else
      tail -n +2 "$summary"
    fi
  done
} > "$OUTPUT_DIR/aggregate_timing_success.csv"

echo ""
echo "WMRobot statistics: $OUTPUT_DIR/aggregate_timing_success.csv"
echo "WMRobot cost/trajectory data: $OUTPUT_DIR/<solver>/vis_data"
