#include "bi_mppi_gpu.cuh"
#include "manipulator_random_pose_benchmark_common.h"

int main() {
  auto params = manipulator_random_pose_benchmark::randomPoseSolverParams();
  params.bicmppi.clustering_method = ClusteringMethod::DBSCAN;
  return manipulator_random_pose_benchmark::runBidirectionalBenchmark<BiMPPI_GPU>(
      "bicmppi_dbscan", "BiC-MPPI (DBSCAN)", params.bicmppi);
}
