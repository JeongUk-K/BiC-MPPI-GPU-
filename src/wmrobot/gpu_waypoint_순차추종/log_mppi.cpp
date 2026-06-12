#include "stitched_barn_params.h"

int main(int argc, char **argv) {
  return wmrobot_stitched::runMppiLikeExecutable<LogMPPI_GPU>(
      argc, argv, "log_mppi", "Log-MPPI", 2000,
      wmrobot_stitched::makeLogMppiConfig());
}
