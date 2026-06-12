#include "stitched_barn_params.h"

int main(int argc, char **argv) {
  return wmrobot_stitched::runBiExecutable(
      argc, argv, wmrobot_stitched::makeBiMppiConfig());
}
