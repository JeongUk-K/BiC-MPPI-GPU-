#include "bi_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  const auto params = manipulator_random_pose_benchmark::randomPoseSolverParams();
  return manipulator_random_pose_benchmark::runBidirectionalBenchmark<BiMPPI_GPU>(
      "bicmppi", "BiC-MPPI", params.bicmppi);
}
