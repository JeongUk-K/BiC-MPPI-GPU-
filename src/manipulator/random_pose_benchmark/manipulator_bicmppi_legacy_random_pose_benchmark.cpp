#include "bi_mppi_gpu_legacy.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  const auto params =
      manipulator_random_pose_benchmark::randomPoseSolverParams();
  return manipulator_random_pose_benchmark::runBidirectionalBenchmark<
      BiMPPI_GPU_Legacy>("bicmppi_legacy", "BiC-MPPI Legacy (K-means)",
                         params.bicmppi);
}
