#include "stitched_barn_common.h"

int main(int argc, char **argv) {
  return wmrobot_stitched::runMppiLikeExecutable<ClusterMPPI_GPU>(
      argc, argv, "cluster_mppi", "Cluster-MPPI", 2500,
      wmrobot_stitched::makeClusterMppiConfig());
}
