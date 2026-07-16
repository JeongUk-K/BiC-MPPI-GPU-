#include "wmrobot_ablation_gpu_common.h"

int main(int argc, char **argv) {
  GpuAblationConfig config;
  parseGpuAblationArgs(argc, argv, config);
  config.raw_connection_candidates = 8;
  runAndWriteGpuAblationVariant(GpuAblationVariant::BiCNoClustering, config);
  return 0;
}
