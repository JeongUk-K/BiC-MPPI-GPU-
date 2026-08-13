#include "mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  using namespace manipulator_random_pose_benchmark;

  SolverParams params;
  params.dt = 0.02f;
  params.forward_horizon = 90;
  params.forward_samples = 1024;
  params.gamma_u = 0.0015;
  params.sigma = {4.5, 4.5, 3.8, 2.2, 1.8, 1.2};

  BenchmarkRunner benchmark("mppi", "MPPI");
  for (const auto &scenario : benchmark.scenarios()) {
    benchmark.runForwardScenario<MPPI_GPU>(
        scenario, params,
        [](MPPI_GPU &, const SolverParams &) {});
  }
  return benchmark.finish();
}
