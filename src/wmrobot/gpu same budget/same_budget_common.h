#pragma once

#include "../Ablation study/GPU/wmrobot_ablation_gpu_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

enum class SameBudgetMethod {
  MPPI,
  ClusterMPPI,
  BiCRawConnect,
  FullBiC,
};

struct SameBudgetMethodSpec {
  SameBudgetMethod method;
  GpuAblationVariant variant;
  std::string label;
  bool bidirectional = false;
  bool uses_guide = false;
};

struct SameBudgetConfig {
  std::vector<int> budgets{1000, 3000, 6000};
  int map_begin = 299;
  int num_maps = 300;
  int start_cases = 2;
  int maxiter = 200;
  int T = 100;
  int Tf = 50;
  int Tb = 50;
  int raw_connection_candidates = 64;
  std::uint_fast64_t seed = 1;
  std::string dataset_dir = "../BARN_dataset/txt_files";
  std::string out_dir = "../results/sampling_budget_scaling";
  std::string guide_budget_policy = "proportional";
  int fixed_guide_budget = 6000;
  std::string git_commit = "unknown";
  bool overwrite = false;
};

struct SameBudgetSummaryRow {
  std::string method;
  std::string budget_label;
  int num_trials = 0;
  int num_success = 0;
  int num_failures = 0;
  double success_rate = 0.0;
  double success_ci_low = 0.0;
  double success_ci_high = 0.0;
  double mean_iterations_success = std::numeric_limits<double>::quiet_NaN();
  double std_iterations_success = std::numeric_limits<double>::quiet_NaN();
  double mean_elapsed_success_s = std::numeric_limits<double>::quiet_NaN();
  double std_elapsed_success_s = std::numeric_limits<double>::quiet_NaN();
  double median_elapsed_success_s = std::numeric_limits<double>::quiet_NaN();
  double iqr_elapsed_success_s = std::numeric_limits<double>::quiet_NaN();
  double mean_rollout_success_s = std::numeric_limits<double>::quiet_NaN();
  double mean_clustering_success_s = std::numeric_limits<double>::quiet_NaN();
  double mean_connection_success_s = std::numeric_limits<double>::quiet_NaN();
  double mean_guide_success_s = std::numeric_limits<double>::quiet_NaN();
};

inline std::string sameBudgetCsvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

inline std::string sameBudgetLabel(int budget) {
  return "B" + std::to_string(budget);
}

inline double sameBudgetMean(const std::vector<double> &values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

inline double sameBudgetStd(const std::vector<double> &values) {
  if (values.size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double mean = sameBudgetMean(values);
  double accum = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    accum += delta * delta;
  }
  return std::sqrt(accum / static_cast<double>(values.size() - 1));
}

inline double sameBudgetQuantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  if (values.size() == 1) {
    return values.front();
  }
  const double pos = q * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double alpha = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

inline std::pair<double, double> sameBudgetWilsonCi(int success, int total) {
  if (total <= 0) {
    return {0.0, 0.0};
  }
  constexpr double z = 1.96;
  const double n = static_cast<double>(total);
  const double p = static_cast<double>(success) / n;
  const double denom = 1.0 + z * z / n;
  const double center = p + z * z / (2.0 * n);
  const double margin =
      z * std::sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)));
  return {(center - margin) / denom, (center + margin) / denom};
}

inline std::vector<SameBudgetMethodSpec> sameBudgetAllMethods() {
  return {
      {SameBudgetMethod::MPPI, GpuAblationVariant::MPPI, "MPPI", false, false},
      {SameBudgetMethod::ClusterMPPI, GpuAblationVariant::ClusterMPPI,
       "Cluster-MPPI", false, false},
      {SameBudgetMethod::BiCRawConnect, GpuAblationVariant::BiCNoClustering,
       "BiC-RawConnect", true, true},
      {SameBudgetMethod::FullBiC, GpuAblationVariant::FullBiC,
       "Full BiC-MPPI", true, true},
  };
}

inline SameBudgetMethodSpec sameBudgetMethodSpec(SameBudgetMethod method) {
  for (const auto &spec : sameBudgetAllMethods()) {
    if (spec.method == method) {
      return spec;
    }
  }
  throw std::runtime_error("unknown same-budget method");
}

inline std::vector<SameBudgetMethodSpec>
sameBudgetSelectedMethods(SameBudgetMethod method) {
  return {sameBudgetMethodSpec(method)};
}

inline bool sameBudgetLooksLikeOption(const std::string &value) {
  return value.rfind("--", 0) == 0;
}

inline void sameBudgetPrintUsage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "Options:\n"
      << "  --budgets 1000 3000 6000   Budget levels to run\n"
      << "  --include-B12000           Add B12000\n"
      << "  --num-maps N               Number of maps from map_begin downward\n"
      << "  --start-cases N            Number of start states per map\n"
      << "  --maxiter N                Max closed-loop iterations\n"
      << "  --out DIR                  Output directory\n"
      << "  --guide-budget-policy proportional|fixed_6000\n"
      << "  --fixed-guide-budget N     Guide samples for fixed policy\n"
      << "  --smoke                    Tiny validation run\n"
      << "  --overwrite                Replace existing CSV outputs\n";
}

inline SameBudgetConfig parseSameBudgetArgs(int argc, char **argv) {
  SameBudgetConfig config;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--help" || key == "-h") {
      sameBudgetPrintUsage(argv[0]);
      std::exit(0);
    } else if (key == "--budgets") {
      config.budgets.clear();
      while (i + 1 < argc && !sameBudgetLooksLikeOption(argv[i + 1])) {
        config.budgets.push_back(std::stoi(argv[++i]));
      }
      if (config.budgets.empty()) {
        throw std::runtime_error("--budgets requires at least one value");
      }
    } else if (key == "--include-B12000") {
      config.budgets.push_back(12000);
    } else if (key == "--map-begin") {
      config.map_begin = std::stoi(require_value(key));
    } else if (key == "--num-maps") {
      config.num_maps = std::stoi(require_value(key));
    } else if (key == "--start-cases") {
      config.start_cases = std::stoi(require_value(key));
    } else if (key == "--maxiter") {
      config.maxiter = std::stoi(require_value(key));
    } else if (key == "--T") {
      config.T = std::stoi(require_value(key));
    } else if (key == "--Tf") {
      config.Tf = std::stoi(require_value(key));
    } else if (key == "--Tb") {
      config.Tb = std::stoi(require_value(key));
    } else if (key == "--raw-candidates") {
      config.raw_connection_candidates = std::stoi(require_value(key));
    } else if (key == "--seed") {
      config.seed =
          static_cast<std::uint_fast64_t>(std::stoull(require_value(key)));
    } else if (key == "--dataset-dir") {
      config.dataset_dir = require_value(key);
    } else if (key == "--out") {
      config.out_dir = require_value(key);
    } else if (key == "--guide-budget-policy") {
      config.guide_budget_policy = require_value(key);
    } else if (key == "--fixed-guide-budget") {
      config.fixed_guide_budget = std::stoi(require_value(key));
    } else if (key == "--git-commit") {
      config.git_commit = require_value(key);
    } else if (key == "--overwrite") {
      config.overwrite = true;
    } else if (key == "--smoke") {
      config.budgets = {128};
      config.num_maps = 1;
      config.start_cases = 1;
      config.maxiter = 2;
      config.raw_connection_candidates = 8;
      config.overwrite = true;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  std::sort(config.budgets.begin(), config.budgets.end());
  config.budgets.erase(std::unique(config.budgets.begin(), config.budgets.end()),
                       config.budgets.end());

  if (config.guide_budget_policy != "proportional" &&
      config.guide_budget_policy != "fixed_6000") {
    throw std::runtime_error(
        "--guide-budget-policy must be proportional or fixed_6000");
  }
  return config;
}

inline int sameBudgetGuideSamples(const SameBudgetConfig &config, int budget) {
  if (config.guide_budget_policy == "fixed_6000") {
    return config.fixed_guide_budget;
  }
  return budget;
}

inline GpuAblationConfig sameBudgetGpuConfig(const SameBudgetConfig &config,
                                             int budget) {
  GpuAblationConfig gpu;
  gpu.map_begin = config.map_begin;
  gpu.num_maps = config.num_maps;
  gpu.start_cases = config.start_cases;
  gpu.maxiter = config.maxiter;
  gpu.N = budget;
  gpu.Nf = budget;
  gpu.Nb = budget;
  gpu.Nr = sameBudgetGuideSamples(config, budget);
  gpu.T = config.T;
  gpu.Tf = config.Tf;
  gpu.Tb = config.Tb;
  gpu.raw_connection_candidates = config.raw_connection_candidates;
  gpu.seed = config.seed;
  gpu.dataset_dir = config.dataset_dir;
  return gpu;
}

inline void sameBudgetWriteRawHeader(std::ostream &csv) {
  csv << "method,budget_label,Ns,Nf,Nb,Ng,T,Tf,Tb,map_id,start_id,seed,"
         "success,failure_reason,iterations,elapsed_total_s,elapsed_rollout_s,"
         "elapsed_clustering_s,elapsed_connection_s,elapsed_guide_s,"
         "final_cost,path_length,notes\n";
}

inline std::string sameBudgetFailureReason(const GpuAblationRunResult &row,
                                           int maxiter,
                                           const SameBudgetMethodSpec &method) {
  if (row.is_success) {
    return "success";
  }
  if (row.is_collision) {
    return "collision";
  }
  if (method.bidirectional && !std::isfinite(row.d_conn)) {
    return "no_connection";
  }
  if (row.iter >= maxiter) {
    return "timeout";
  }
  return "unknown";
}

inline std::uint_fast64_t sameBudgetTrialSeed(const SameBudgetConfig &config,
                                              int map_id, int start_id) {
  return config.seed + static_cast<std::uint_fast64_t>(1009 * map_id +
                                                       9176 * start_id);
}

inline void sameBudgetWriteRawRow(std::ostream &csv,
                                  const SameBudgetMethodSpec &method,
                                  const SameBudgetConfig &config, int budget,
                                  const GpuAblationRunResult &row) {
  const bool one_directional = !method.bidirectional;
  const int Ns = one_directional ? budget : 0;
  const int Nf = method.bidirectional ? budget : 0;
  const int Nb = method.bidirectional ? budget : 0;
  const int Ng = method.uses_guide ? sameBudgetGuideSamples(config, budget) : 0;
  csv << method.label << ',' << sameBudgetLabel(budget) << ',' << Ns << ','
      << Nf << ',' << Nb << ',' << Ng << ',' << config.T << ','
      << config.Tf << ',' << config.Tb << ',' << row.map_id << ','
      << row.start_case << ','
      << sameBudgetTrialSeed(config, row.map_id, row.start_case) << ','
      << (row.is_success ? 1 : 0) << ','
      << sameBudgetFailureReason(row, config.maxiter, method) << ','
      << row.iter << ',' << sameBudgetCsvDouble(row.elapsed) << ','
      << sameBudgetCsvDouble(row.elapsed_rollout) << ','
      << sameBudgetCsvDouble(row.elapsed_clustering) << ','
      << sameBudgetCsvDouble(row.elapsed_connection) << ','
      << sameBudgetCsvDouble(row.elapsed_guide) << ','
      << sameBudgetCsvDouble(row.d_goal) << ",,\n";
}

inline SameBudgetSummaryRow
sameBudgetSummarize(const SameBudgetMethodSpec &method,
                    const std::string &budget_label,
                    const std::vector<GpuAblationRunResult> &rows) {
  SameBudgetSummaryRow summary;
  summary.method = method.label;
  summary.budget_label = budget_label;
  summary.num_trials = static_cast<int>(rows.size());
  for (const auto &row : rows) {
    summary.num_success += row.is_success ? 1 : 0;
  }
  summary.num_failures = summary.num_trials - summary.num_success;
  if (summary.num_trials > 0) {
    summary.success_rate =
        static_cast<double>(summary.num_success) / summary.num_trials;
  }
  const auto ci = sameBudgetWilsonCi(summary.num_success, summary.num_trials);
  summary.success_ci_low = ci.first;
  summary.success_ci_high = ci.second;

  std::vector<double> iterations;
  std::vector<double> elapsed;
  std::vector<double> rollout;
  std::vector<double> clustering;
  std::vector<double> connection;
  std::vector<double> guide;
  for (const auto &row : rows) {
    if (!row.is_success) {
      continue;
    }
    iterations.push_back(static_cast<double>(row.iter));
    elapsed.push_back(row.elapsed);
    rollout.push_back(row.elapsed_rollout);
    clustering.push_back(row.elapsed_clustering);
    connection.push_back(row.elapsed_connection);
    guide.push_back(row.elapsed_guide);
  }

  summary.mean_iterations_success = sameBudgetMean(iterations);
  summary.std_iterations_success = sameBudgetStd(iterations);
  summary.mean_elapsed_success_s = sameBudgetMean(elapsed);
  summary.std_elapsed_success_s = sameBudgetStd(elapsed);
  summary.median_elapsed_success_s = sameBudgetQuantile(elapsed, 0.50);
  summary.iqr_elapsed_success_s =
      sameBudgetQuantile(elapsed, 0.75) - sameBudgetQuantile(elapsed, 0.25);
  summary.mean_rollout_success_s = sameBudgetMean(rollout);
  summary.mean_clustering_success_s = sameBudgetMean(clustering);
  summary.mean_connection_success_s = sameBudgetMean(connection);
  summary.mean_guide_success_s = sameBudgetMean(guide);
  return summary;
}

inline void sameBudgetWriteSummaryHeader(std::ostream &csv) {
  csv << "method,budget_label,num_trials,num_success,num_failures,"
         "success_rate,success_ci_low,success_ci_high,"
         "mean_iterations_success,std_iterations_success,"
         "mean_elapsed_success_s,std_elapsed_success_s,"
         "median_elapsed_success_s,iqr_elapsed_success_s,"
         "mean_rollout_success_s,mean_clustering_success_s,"
         "mean_connection_success_s,mean_guide_success_s,"
         "guide_budget_policy,git_commit\n";
}

inline void sameBudgetWriteSummaryRow(std::ostream &csv,
                                      const SameBudgetSummaryRow &row,
                                      const SameBudgetConfig &config) {
  csv << row.method << ',' << row.budget_label << ',' << row.num_trials << ','
      << row.num_success << ',' << row.num_failures << ','
      << sameBudgetCsvDouble(row.success_rate) << ','
      << sameBudgetCsvDouble(row.success_ci_low) << ','
      << sameBudgetCsvDouble(row.success_ci_high) << ','
      << sameBudgetCsvDouble(row.mean_iterations_success) << ','
      << sameBudgetCsvDouble(row.std_iterations_success) << ','
      << sameBudgetCsvDouble(row.mean_elapsed_success_s) << ','
      << sameBudgetCsvDouble(row.std_elapsed_success_s) << ','
      << sameBudgetCsvDouble(row.median_elapsed_success_s) << ','
      << sameBudgetCsvDouble(row.iqr_elapsed_success_s) << ','
      << sameBudgetCsvDouble(row.mean_rollout_success_s) << ','
      << sameBudgetCsvDouble(row.mean_clustering_success_s) << ','
      << sameBudgetCsvDouble(row.mean_connection_success_s) << ','
      << sameBudgetCsvDouble(row.mean_guide_success_s) << ','
      << config.guide_budget_policy << ',' << config.git_commit << '\n';
}

inline std::string sameBudgetCurrentDateTime() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S%z");
  return out.str();
}

inline void sameBudgetWriteMetadata(const SameBudgetConfig &config,
                                    const std::filesystem::path &path,
                                    const std::vector<SameBudgetMethodSpec>
                                        &methods) {
  std::ofstream json(path);
  json << "{\n";
  json << "  \"experiment\": \"sampling_budget_scaling\",\n";
  json << "  \"date\": \"" << sameBudgetCurrentDateTime() << "\",\n";
  json << "  \"git_commit\": \"" << config.git_commit << "\",\n";
  json << "  \"num_maps\": " << config.num_maps << ",\n";
  json << "  \"starts_per_map\": " << config.start_cases << ",\n";
  json << "  \"num_trials\": " << config.num_maps * config.start_cases
       << ",\n";
  json << "  \"budgets\": [";
  for (std::size_t i = 0; i < config.budgets.size(); ++i) {
    if (i) {
      json << ", ";
    }
    json << config.budgets[i];
  }
  json << "],\n";
  json << "  \"horizon_policy\": {\"T\": " << config.T
       << ", \"Tf\": " << config.Tf << ", \"Tb\": " << config.Tb
       << "},\n";
  json << "  \"guide_budget_policy\": \"" << config.guide_budget_policy
       << "\",\n";
  json << "  \"methods\": [";
  for (std::size_t i = 0; i < methods.size(); ++i) {
    if (i) {
      json << ", ";
    }
    json << "\"" << methods[i].label << "\"";
  }
  json << "]\n";
  json << "}\n";
}

inline void sameBudgetValidate(const SameBudgetConfig &config,
                               const std::vector<SameBudgetMethodSpec>
                                   &methods,
                               const std::vector<SameBudgetSummaryRow>
                                   &summaries,
                               int raw_rows_written) {
  const int expected_trials = config.num_maps * config.start_cases;
  const int expected_raw =
      expected_trials * static_cast<int>(config.budgets.size()) *
      static_cast<int>(methods.size());
  const int expected_summary =
      static_cast<int>(config.budgets.size()) * static_cast<int>(methods.size());
  if (raw_rows_written != expected_raw) {
    throw std::runtime_error("raw row count mismatch: expected " +
                             std::to_string(expected_raw) + ", got " +
                             std::to_string(raw_rows_written));
  }
  if (static_cast<int>(summaries.size()) != expected_summary) {
    throw std::runtime_error("summary row count mismatch");
  }
  for (const auto &summary : summaries) {
    if (summary.num_trials != expected_trials) {
      throw std::runtime_error("unexpected trial count in summary");
    }
    if (summary.success_rate < 0.0 || summary.success_rate > 1.0 ||
        summary.success_ci_low < 0.0 || summary.success_ci_low > 1.0 ||
        summary.success_ci_high < 0.0 || summary.success_ci_high > 1.0 ||
        summary.success_ci_low > summary.success_rate ||
        summary.success_ci_high < summary.success_rate) {
      throw std::runtime_error("invalid success rate or Wilson CI");
    }
    if ((std::isfinite(summary.mean_elapsed_success_s) &&
         summary.mean_elapsed_success_s < 0.0) ||
        (std::isfinite(summary.mean_rollout_success_s) &&
         summary.mean_rollout_success_s < 0.0) ||
        (std::isfinite(summary.mean_clustering_success_s) &&
         summary.mean_clustering_success_s < 0.0) ||
        (std::isfinite(summary.mean_connection_success_s) &&
         summary.mean_connection_success_s < 0.0) ||
        (std::isfinite(summary.mean_guide_success_s) &&
         summary.mean_guide_success_s < 0.0)) {
      throw std::runtime_error("negative timing value in summary");
    }
  }
}

inline int runSameBudgetExperiment(int argc, char **argv,
                                   std::vector<SameBudgetMethodSpec> methods) {
  const SameBudgetConfig config = parseSameBudgetArgs(argc, argv);
  const std::filesystem::path out_dir(config.out_dir);
  const auto raw_path = out_dir / "raw_trials.csv";
  const auto summary_path = out_dir / "summary.csv";
  const auto metadata_path = out_dir / "metadata.json";

  if (!config.overwrite &&
      (std::filesystem::exists(raw_path) ||
       std::filesystem::exists(summary_path) ||
       std::filesystem::exists(metadata_path))) {
    throw std::runtime_error("output files already exist; pass --overwrite");
  }

  std::filesystem::create_directories(out_dir);

  std::ofstream raw_csv(raw_path);
  sameBudgetWriteRawHeader(raw_csv);

  std::vector<SameBudgetSummaryRow> summaries;
  int raw_rows_written = 0;

  for (int budget : config.budgets) {
    GpuAblationConfig gpu_config = sameBudgetGpuConfig(config, budget);
    for (const auto &method : methods) {
      std::cout << "\n=== " << method.label << " " << sameBudgetLabel(budget)
                << " ===" << std::endl;
      auto rows = runGpuAblationVariant(method.variant, gpu_config);
      for (auto &row : rows) {
        row.variant = method.label;
        sameBudgetWriteRawRow(raw_csv, method, config, budget, row);
        ++raw_rows_written;
      }
      summaries.push_back(
          sameBudgetSummarize(method, sameBudgetLabel(budget), rows));
    }
  }
  raw_csv.close();

  sameBudgetValidate(config, methods, summaries, raw_rows_written);

  std::ofstream summary_csv(summary_path);
  sameBudgetWriteSummaryHeader(summary_csv);
  for (const auto &summary : summaries) {
    sameBudgetWriteSummaryRow(summary_csv, summary, config);
  }
  summary_csv.close();

  sameBudgetWriteMetadata(config, metadata_path, methods);

  std::cout << "\n=== Sampling budget scaling completed ===\n";
  std::cout << "Raw trials: " << raw_path << "\n";
  std::cout << "Summary   : " << summary_path << "\n";
  std::cout << "Metadata  : " << metadata_path << "\n";
  return 0;
}
