#include "mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  const auto params = manipulator_random_pose_benchmark::randomPoseSolverParams();
  return manipulator_random_pose_benchmark::runForwardBenchmark<MPPI_GPU>(
      "mppi", "MPPI", params.mppi,
      [](MPPI_GPU &, const manipulator_random_pose_benchmark::SolverParams &) {
      });
}
