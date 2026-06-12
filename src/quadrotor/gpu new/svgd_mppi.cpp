#include <svgd_mppi_gpu.cuh>

#include "quadrotor_gpu_new_common.h"

int main(int argc, char **argv) {
  return runQuadrotorGpuNewSvgd<SVGDMPPI_GPU>(
      argc, argv, "svgd_mppi", "SVGD-MPPI");
}
