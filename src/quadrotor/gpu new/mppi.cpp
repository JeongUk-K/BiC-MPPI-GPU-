#include <mppi_gpu.cuh>

#include "quadrotor_gpu_new_common.h"

int main(int argc, char **argv) {
  return runQuadrotorGpuNewOneDirectional<MPPI_GPU>(
      argc, argv, "mppi", "MPPI");
}
