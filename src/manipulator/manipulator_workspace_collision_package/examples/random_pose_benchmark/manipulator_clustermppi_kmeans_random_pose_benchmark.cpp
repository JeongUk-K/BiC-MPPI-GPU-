#include "cluster_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  auto params = manipulator_random_pose_benchmark::randomPoseSolverParams();
  params.clustermppi.clustering_method = ClusteringMethod::KMeans;
  return manipulator_random_pose_benchmark::runForwardBenchmark<ClusterMPPI_GPU>(
      "clustermppi_kmeans", "Cluster-MPPI (K-means)", params.clustermppi,
      [](ClusterMPPI_GPU &,
         const manipulator_random_pose_benchmark::SolverParams &) {});
}
