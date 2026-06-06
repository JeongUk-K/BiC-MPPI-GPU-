#include "wmrobot_ablation_common.h"

int main(int argc, char **argv) {
  AblationConfig config;
  parseAblationArgs(argc, argv, config);

  const std::vector<AblationVariant> variants = {
      AblationVariant::MPPI,
      AblationVariant::ClusterMPPI,
      AblationVariant::BiCNoGuide,
      AblationVariant::BiCNoBackward,
      AblationVariant::BiCNoClustering,
      AblationVariant::FullBiC,
  };

  std::vector<AblationSummary> summaries;
  for (AblationVariant variant : variants) {
    summaries.push_back(runAndWriteAblationVariant(variant, config));
  }

  std::ofstream summary_csv(config.output_prefix + "_summary.csv");
  writeSummaryHeader(summary_csv);
  for (const auto &summary : summaries) {
    writeSummaryRow(summary_csv, summary);
  }

  return 0;
}
