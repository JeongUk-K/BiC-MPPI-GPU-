#include "log_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  const auto params = manipulator_random_pose_benchmark::randomPoseSolverParams();
  return manipulator_random_pose_benchmark::runForwardBenchmark<LogMPPI_GPU>(
      "logmppi", "Log-MPPI", params.logmppi,
      [](LogMPPI_GPU &,
         const manipulator_random_pose_benchmark::SolverParams &) {});
}
