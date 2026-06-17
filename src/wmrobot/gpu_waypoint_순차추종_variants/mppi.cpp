#include "stitched_barn_common.h"

int main(int argc, char **argv) {
  return wmrobot_stitched::runMppiLikeExecutable<MPPI_GPU>(
      argc, argv, "mppi", "MPPI", 1000, wmrobot_stitched::makeMppiConfig());
}
