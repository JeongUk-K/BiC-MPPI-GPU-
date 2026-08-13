#include "bi_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  using namespace manipulator_random_pose_benchmark;

  SolverParams params;
  params.dt = 0.02f;
  params.forward_horizon = 45;
  params.backward_horizon = 45;
  params.forward_samples = 1024;
  params.backward_samples = 1024;
  params.reverse_samples = 1024;
  params.gamma_u = 0.0015;
  params.sigma = {4.5, 4.5, 3.8, 2.2, 1.8, 1.2};
  params.bic_deviation_mu = 1.0;
  params.bic_cost_mu = 1.0;
  params.bic_epsilon = 3.8;
  params.bic_minpts = 8;
  params.bic_psi = 0.0;
  params.clustering_method = ClusteringMethod::KMeans;

  BenchmarkRunner benchmark("bicmppi", "BiC-MPPI (K-means)");
  for (const auto &scenario : benchmark.scenarios()) {
    benchmark.runBidirectionalScenario<BiMPPI_GPU>(scenario, params);
  }
  return benchmark.finish();
}
