#include "wmrobot_ablation_gpu_common.h"

int main(int argc, char **argv) {
  GpuAblationConfig config;
  parseGpuAblationArgs(argc, argv, config);
  runAndWriteGpuAblationVariant(GpuAblationVariant::BiCNoGuideDBSCAN, config);
  return 0;
}
