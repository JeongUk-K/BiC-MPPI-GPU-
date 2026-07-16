#include "cluster_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  const auto params = manipulator_random_pose_benchmark::randomPoseSolverParams();
  return manipulator_random_pose_benchmark::runForwardBenchmark<ClusterMPPI_GPU>(
      "clustermppi", "Cluster-MPPI", params.clustermppi,
      [](ClusterMPPI_GPU &solver,
         const manipulator_random_pose_benchmark::SolverParams &solver_params) {
        solver.setClusterParams(solver_params.cluster_deviation_mu,
                                solver_params.cluster_epsilon,
                                solver_params.cluster_minpts);
      });
}
