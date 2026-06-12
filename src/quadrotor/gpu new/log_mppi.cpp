#include <log_mppi_gpu.cuh>

#include "quadrotor_gpu_new_common.h"

int main(int argc, char **argv) {
  return runQuadrotorGpuNewOneDirectional<LogMPPI_GPU>(
      argc, argv, "log_mppi", "Log-MPPI");
}
