#include "stitched_barn_common.h"

int main(int argc, char **argv) {
  return wmrobot_stitched::runBiExecutable(
      argc, argv, wmrobot_stitched::makeBiMppiConfig());
}
