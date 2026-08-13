#include "cluster_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  using namespace manipulator_random_pose_benchmark;

  SolverParams params;
  params.dt = 0.02f;
  params.forward_horizon = 90;
  params.forward_samples = 1024;
  params.gamma_u = 0.0015;
  params.sigma = {4.5, 4.5, 3.8, 2.2, 1.8, 1.2};
  params.cluster_deviation_mu = 1.0;
  params.cluster_epsilon = 3.8;
  params.cluster_minpts = 8;
  params.clustering_method = ClusteringMethod::KMeans;

  BenchmarkRunner benchmark("clustermppi", "Cluster-MPPI (K-means)");
  for (const auto &scenario : benchmark.scenarios()) {
    benchmark.runForwardScenario<ClusterMPPI_GPU>(
        scenario, params,
        [](ClusterMPPI_GPU &solver, const SolverParams &solver_params) {
          solver.setClusterParams(solver_params.cluster_deviation_mu,
                                  solver_params.cluster_epsilon,
                                  solver_params.cluster_minpts);
        });
  }
  return benchmark.finish();
}
