#include "wmrobot_ablation_gpu_common.h"

int main(int argc, char **argv) {
  GpuAblationConfig config;
  parseGpuAblationArgs(argc, argv, config);

  const std::vector<GpuAblationVariant> variants = {
      GpuAblationVariant::MPPI,
      GpuAblationVariant::ClusterMPPI,
      GpuAblationVariant::BiCNoGuide,
      GpuAblationVariant::BiCNoBackward,
      GpuAblationVariant::BiCNoClustering,
      GpuAblationVariant::FullBiC,
  };

  std::vector<GpuAblationSummary> summaries;
  for (GpuAblationVariant variant : variants) {
    summaries.push_back(runAndWriteGpuAblationVariant(variant, config));
  }

  std::ofstream summary_csv(config.output_prefix + "_summary.csv");
  writeGpuSummaryHeader(summary_csv);
  for (const auto &summary : summaries) {
    writeGpuSummaryRow(summary_csv, summary);
  }
  return 0;
}
