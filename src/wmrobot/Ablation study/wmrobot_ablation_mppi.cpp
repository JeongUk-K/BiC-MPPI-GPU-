#include "wmrobot_ablation_common.h"

int main(int argc, char **argv) {
  AblationConfig config;
  parseAblationArgs(argc, argv, config);
  runAndWriteAblationVariant(AblationVariant::MPPI, config);
  return 0;
}
