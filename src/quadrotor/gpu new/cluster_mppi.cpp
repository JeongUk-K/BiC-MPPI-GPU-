#include <cluster_mppi_gpu.cuh>

#include "quadrotor_gpu_new_common.h"

int main(int argc, char **argv) {
  return runQuadrotorGpuNewOneDirectional<ClusterMPPI_GPU>(
      argc, argv, "cluster_mppi", "Cluster-MPPI");
}
