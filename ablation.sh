#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build/ablation"
OUTPUT_DIR="${ABLATION_OUTPUT:-$ROOT/result/ablation}"
DATASET_DIR="${BARN_DATASET_DIR:-$ROOT/BARN_dataset/txt_files}"
CXX="${CXX:-g++}"
NO_BUILD=0
RUN_ARGS=()

usage() {
  echo "Usage: $0 [--no-build] [ablation options]"
  echo "  Runs w/o Guide with K-Means/DBSCAN and BiC-MPPI w/o Cluster."
  echo "  Example: $0 --smoke --num-maps 1 --maxiter 2"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) NO_BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) RUN_ARGS+=("$1") ;;
  esac
  shift
done

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi
if [[ "$DATASET_DIR" != /* ]]; then
  DATASET_DIR="$ROOT/$DATASET_DIR"
fi
[[ -f "$DATASET_DIR/output_299.txt" ]] || {
  echo "BARN maps not found in: $DATASET_DIR" >&2
  echo "Set BARN_DATASET_DIR to the directory containing output_*.txt." >&2
  exit 1
}

find_cuda() {
  NVCC="${NVCC:-$(command -v nvcc || true)}"
  [[ -n "$NVCC" ]] || { echo "nvcc not found" >&2; exit 1; }
  local cuda_root
  cuda_root="$(CDPATH= cd -- "$(dirname -- "$NVCC")/.." && pwd)"
  CUDA_INCLUDE="${CUDA_INCLUDE:-$cuda_root/include}"
  CUDA_LIBDIR="${CUDA_LIBDIR:-$cuda_root/lib64}"
  [[ -d "$CUDA_INCLUDE" ]] || CUDA_INCLUDE=/usr/include
  [[ -d "$CUDA_LIBDIR" ]] || CUDA_LIBDIR=/usr/lib/x86_64-linux-gnu

  local name candidate
  for name in libcudart.so libcurand.so libcublas.so; do
    for candidate in "$CUDA_LIBDIR/$name" "/usr/lib/x86_64-linux-gnu/$name"; do
      [[ -e "$candidate" ]] || continue
      case "$name" in
        libcudart.so) CUDART_LIB="$candidate" ;;
        libcurand.so) CURAND_LIB="$candidate" ;;
        libcublas.so) CUBLAS_LIB="$candidate" ;;
      esac
      break
    done
  done
  [[ -n "${CUDART_LIB:-}" && -n "${CURAND_LIB:-}" &&
     -n "${CUBLAS_LIB:-}" ]] || { echo "CUDA libraries not found" >&2; exit 1; }
}

build() {
  find_cuda
  mkdir -p "$BUILD_DIR"
  local -a host_flags=()
  command -v g++-10 >/dev/null 2>&1 && host_flags=(-ccbin g++-10)
  local -a includes=(
    -I"$ROOT/mppi" -I"$ROOT/mppi/cuda-accel"
    -I"$ROOT/mppi/cpu-legacy" -I"$ROOT/model" -I"$ROOT/include"
    -I"$ROOT/include/fastsc" -I"$ROOT/include/EigenRand"
    -I"$ROOT/include/matplotlibcpp" -I"$CUDA_INCLUDE"
    $(python3-config --includes)
    $(pkg-config --cflags eigen3 2>/dev/null || echo -I/usr/include/eigen3)
  )
  local -a cuda_flags=(-O3 --std=c++14 -arch="${CUDA_ARCH:-sm_86}"
                       "${host_flags[@]}" --expt-relaxed-constexpr
                       -Xcompiler -fopenmp)
  local -a objects=()
  local source object stem
  for source in "$ROOT/mppi/cuda-accel/mppi_gpu.cu" \
                "$ROOT/mppi/cuda-accel/cluster_mppi_gpu.cu" \
                "$ROOT/mppi/cuda-accel/bi_mppi_gpu.cu" \
                "$ROOT/mppi/cuda-accel/log_mppi_gpu.cu" \
                "$ROOT/include/fastsc/labels.cu"; do
    stem="$(basename "${source%.cu}")"
    object="$BUILD_DIR/$stem.o"
    echo "[build] $source"
    "$NVCC" "${cuda_flags[@]}" "${includes[@]}" -c "$source" -o "$object"
    objects+=("$object")
  done
  ar rcs "$BUILD_DIR/libablation_gpu.a" "${objects[@]}"

  local -a link_flags=(-fopenmp -Wl,-rpath,"$CUDA_LIBDIR"
                       "$CUDART_LIB" "$CURAND_LIB" "$CUBLAS_LIB"
                       $(python3-config --ldflags --embed 2>/dev/null ||
                         python3-config --ldflags))
  for stem in bic_without_guide bic_without_guide_dbscan \
              bic_without_clustering; do
    source="$ROOT/src/wmrobot/Ablation study/GPU/wmrobot_ablation_gpu_$stem.cpp"
    echo "[build] $source"
    "$CXX" -O3 -std=c++17 -fopenmp "${includes[@]}" "$source" \
      -L"$BUILD_DIR" -lablation_gpu "${link_flags[@]}" \
      -o "$BUILD_DIR/wmrobot_ablation_gpu_$stem"
  done
}

if [[ "$NO_BUILD" -eq 0 ]]; then
  build
else
  find_cuda
fi

mkdir -p "$OUTPUT_DIR"
PREFIX="$OUTPUT_DIR/wmrobot_ablation"
rm -f "$PREFIX"_bic_without_{guide_kmeans,guide_dbscan,clustering}_{runs,summary}.csv \
      "$PREFIX"_bic_without_guide_{runs,summary}.csv \
      "$OUTPUT_DIR/wmrobot_ablation_{summary,success_summary}.csv" \
      "$OUTPUT_DIR"/bic_without_{guide,guide_dbscan,clustering}.log

for stem in bic_without_guide bic_without_guide_dbscan \
            bic_without_clustering; do
  executable="$BUILD_DIR/wmrobot_ablation_gpu_$stem"
  [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 1; }
  echo "[run] $stem"
  (
    cd "$BUILD_DIR"
    "$executable" --dataset-dir "$DATASET_DIR" --output-prefix "$PREFIX" \
      "${RUN_ARGS[@]}"
  ) 2>&1 | tee "$OUTPUT_DIR/$stem.log"
done

GUIDE_KM_SUMMARY="${PREFIX}_bic_without_guide_kmeans_summary.csv"
GUIDE_DB_SUMMARY="${PREFIX}_bic_without_guide_dbscan_summary.csv"
CLUSTER_SUMMARY="${PREFIX}_bic_without_clustering_summary.csv"
{
  head -n 1 "$CLUSTER_SUMMARY"
  tail -n +2 "$CLUSTER_SUMMARY"
  tail -n +2 "$GUIDE_KM_SUMMARY"
  tail -n +2 "$GUIDE_DB_SUMMARY"
} > "$OUTPUT_DIR/wmrobot_ablation_summary.csv"

cut -d, -f10,12,15-36 "$OUTPUT_DIR/wmrobot_ablation_summary.csv" \
  > "$OUTPUT_DIR/wmrobot_ablation_success_summary.csv"

echo "Ablation statistics: $OUTPUT_DIR/wmrobot_ablation_summary.csv"
echo "Success-only statistics: $OUTPUT_DIR/wmrobot_ablation_success_summary.csv"
