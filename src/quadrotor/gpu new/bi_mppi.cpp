#include <bi_mppi_gpu.cuh>

#include "quadrotor_gpu_new_common.h"

int main(int argc, char **argv) {
  return runQuadrotorGpuNewBidirectional<BiMPPI_GPU>(
      argc, argv, "bi_mppi", "BiC-MPPI");
}
