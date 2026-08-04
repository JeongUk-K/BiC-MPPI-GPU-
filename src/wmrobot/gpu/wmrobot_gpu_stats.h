#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

struct WmrobotGpuRunResult {
  std::string variant;
  int start_case = 0;
  int map_id = 0;
  bool is_success = false;
  bool is_collision = false;
  int iter = 0;
  double cost = std::numeric_limits<double>::quiet_NaN();
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;
  double d_goal = 0.0;
  double d_conn = std::numeric_limits<double>::quiet_NaN();
};

struct WmrobotGpuSummary {
  std::string variant;
  int num_runs = 0;
  int num_success = 0;
  int nfail = 0;
  double success_rate = 0.0;
  double nbar_iter = 0.0;
  double tbar_elapsed = 0.0;
  double avg_cost = std::numeric_limits<double>::quiet_NaN();
  double dbar_goal = 0.0;
  double dbar_conn = std::numeric_limits<double>::quiet_NaN();
  double avg_iter_success = std::numeric_limits<double>::quiet_NaN();
  double avg_total_elapsed_success = std::numeric_limits<double>::quiet_NaN();
  double avg_rollout_time_success = std::numeric_limits<double>::quiet_NaN();
  double avg_clustering_time_success = std::numeric_limits<double>::quiet_NaN();
  double avg_connection_time_success = std::numeric_limits<double>::quiet_NaN();
  double avg_guide_time_success = std::numeric_limits<double>::quiet_NaN();
  double total_elapsed_success_q1 = std::numeric_limits<double>::quiet_NaN();
  double total_elapsed_success_q2 = std::numeric_limits<double>::quiet_NaN();
  double total_elapsed_success_q3 = std::numeric_limits<double>::quiet_NaN();
  double rollout_time_success_q1 = std::numeric_limits<double>::quiet_NaN();
  double rollout_time_success_q2 = std::numeric_limits<double>::quiet_NaN();
  double rollout_time_success_q3 = std::numeric_limits<double>::quiet_NaN();
  double clustering_time_success_q1 = std::numeric_limits<double>::quiet_NaN();
  double clustering_time_success_q2 = std::numeric_limits<double>::quiet_NaN();
  double clustering_time_success_q3 = std::numeric_limits<double>::quiet_NaN();
  double connection_time_success_q1 = std::numeric_limits<double>::quiet_NaN();
  double connection_time_success_q2 = std::numeric_limits<double>::quiet_NaN();
  double connection_time_success_q3 = std::numeric_limits<double>::quiet_NaN();
  double guide_time_success_q1 = std::numeric_limits<double>::quiet_NaN();
  double guide_time_success_q2 = std::numeric_limits<double>::quiet_NaN();
  double guide_time_success_q3 = std::numeric_limits<double>::quiet_NaN();
};

inline std::string wmrobotGpuCsvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

inline std::string wmrobotGpuResultPath(const std::string &filename) {
  const char *result_dir = std::getenv("WMROBOT_RESULT_DIR");
  if (!result_dir || !*result_dir) {
    return filename;
  }
  return std::string(result_dir) + "/" + filename;
}

inline double wmrobotGpuQuantile(std::vector<double> values, double q) {
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

inline void writeWmrobotGpuRunHeader(std::ostream &csv) {
  csv << "variant,start_case,map,is_success,is_collision,iter,cost,elapsed,"
         "elapsed_rollout,elapsed_clustering,elapsed_connection,elapsed_guide,"
         "d_goal,d_conn\n";
}

inline void writeWmrobotGpuRunRow(std::ostream &csv,
                                  const WmrobotGpuRunResult &row) {
  csv << row.variant << ',' << row.start_case << ',' << row.map_id << ','
      << row.is_success << ',' << row.is_collision << ',' << row.iter << ','
      << wmrobotGpuCsvDouble(row.cost) << ','
      << wmrobotGpuCsvDouble(row.elapsed) << ','
      << wmrobotGpuCsvDouble(row.elapsed_rollout) << ','
      << wmrobotGpuCsvDouble(row.elapsed_clustering) << ','
      << wmrobotGpuCsvDouble(row.elapsed_connection) << ','
      << wmrobotGpuCsvDouble(row.elapsed_guide) << ','
      << wmrobotGpuCsvDouble(row.d_goal) << ','
      << wmrobotGpuCsvDouble(row.d_conn) << '\n';
}

inline void printWmrobotGpuRunRow(std::ostream &out,
                                  const WmrobotGpuRunResult &row) {
  out << row.variant << '\t' << row.start_case << '\t' << row.map_id << '\t'
      << row.is_success << '\t' << row.iter << '\t' << row.cost << '\t'
      << row.elapsed << '\t'
      << row.d_goal << '\t' << wmrobotGpuCsvDouble(row.d_conn) << std::endl;
}

inline void writeWmrobotGpuSummaryHeader(std::ostream &csv) {
  csv << "Variant,Success Rate,Nfail,Nbar_iter,Tbar_elapsed,dbar_goal,"
         "dbar_conn,avg_cost,num_runs,num_success,"
         "variant,num_sim,num_success,num_failure,success_rate,"
         "avg_iter_success,avg_total_elapsed_success,"
         "avg_rollout_time_success,avg_clustering_time_success,"
         "avg_connection_time_success,avg_guide_time_success,"
         "total_elapsed_success_q1,total_elapsed_success_q2,"
         "total_elapsed_success_q3,rollout_time_success_q1,"
         "rollout_time_success_q2,rollout_time_success_q3,"
         "clustering_time_success_q1,clustering_time_success_q2,"
         "clustering_time_success_q3,connection_time_success_q1,"
         "connection_time_success_q2,connection_time_success_q3,"
         "guide_time_success_q1,guide_time_success_q2,"
         "guide_time_success_q3\n";
}

inline void writeWmrobotGpuSummaryRow(std::ostream &csv,
                                      const WmrobotGpuSummary &summary) {
  csv << summary.variant << ',' << wmrobotGpuCsvDouble(summary.success_rate)
      << ',' << summary.nfail << ',' << wmrobotGpuCsvDouble(summary.nbar_iter)
      << ',' << wmrobotGpuCsvDouble(summary.tbar_elapsed) << ','
      << wmrobotGpuCsvDouble(summary.dbar_goal) << ','
      << wmrobotGpuCsvDouble(summary.dbar_conn) << ','
      << wmrobotGpuCsvDouble(summary.avg_cost) << ',' << summary.num_runs
      << ',' << summary.num_success << ',' << summary.variant << ','
      << summary.num_runs << ',' << summary.num_success << ','
      << summary.nfail << ',' << wmrobotGpuCsvDouble(summary.success_rate)
      << ',' << wmrobotGpuCsvDouble(summary.avg_iter_success) << ','
      << wmrobotGpuCsvDouble(summary.avg_total_elapsed_success) << ','
      << wmrobotGpuCsvDouble(summary.avg_rollout_time_success) << ','
      << wmrobotGpuCsvDouble(summary.avg_clustering_time_success) << ','
      << wmrobotGpuCsvDouble(summary.avg_connection_time_success) << ','
      << wmrobotGpuCsvDouble(summary.avg_guide_time_success) << ','
      << wmrobotGpuCsvDouble(summary.total_elapsed_success_q1) << ','
      << wmrobotGpuCsvDouble(summary.total_elapsed_success_q2) << ','
      << wmrobotGpuCsvDouble(summary.total_elapsed_success_q3) << ','
      << wmrobotGpuCsvDouble(summary.rollout_time_success_q1) << ','
      << wmrobotGpuCsvDouble(summary.rollout_time_success_q2) << ','
      << wmrobotGpuCsvDouble(summary.rollout_time_success_q3) << ','
      << wmrobotGpuCsvDouble(summary.clustering_time_success_q1) << ','
      << wmrobotGpuCsvDouble(summary.clustering_time_success_q2) << ','
      << wmrobotGpuCsvDouble(summary.clustering_time_success_q3) << ','
      << wmrobotGpuCsvDouble(summary.connection_time_success_q1) << ','
      << wmrobotGpuCsvDouble(summary.connection_time_success_q2) << ','
      << wmrobotGpuCsvDouble(summary.connection_time_success_q3) << ','
      << wmrobotGpuCsvDouble(summary.guide_time_success_q1) << ','
      << wmrobotGpuCsvDouble(summary.guide_time_success_q2) << ','
      << wmrobotGpuCsvDouble(summary.guide_time_success_q3) << '\n';
}

inline WmrobotGpuSummary
summarizeWmrobotGpuRuns(const std::string &variant,
                        const std::vector<WmrobotGpuRunResult> &runs) {
  WmrobotGpuSummary summary;
  summary.variant = variant;
  summary.num_runs = static_cast<int>(runs.size());
  if (runs.empty()) {
    return summary;
  }

  double iter_sum = 0.0;
  double elapsed_sum = 0.0;
  double goal_sum = 0.0;
  double conn_sum = 0.0;
  int conn_count = 0;
  double cost_sum = 0.0;
  int cost_count = 0;
  double success_iter_sum = 0.0;
  double success_elapsed_sum = 0.0;
  double success_rollout_sum = 0.0;
  double success_clustering_sum = 0.0;
  double success_connection_sum = 0.0;
  double success_guide_sum = 0.0;
  std::vector<double> success_elapsed_values;
  std::vector<double> success_rollout_values;
  std::vector<double> success_clustering_values;
  std::vector<double> success_connection_values;
  std::vector<double> success_guide_values;

  for (const auto &run : runs) {
    summary.num_success += run.is_success ? 1 : 0;
    iter_sum += run.iter;
    elapsed_sum += run.elapsed;
    goal_sum += run.d_goal;
    if (std::isfinite(run.cost)) {
      cost_sum += run.cost;
      ++cost_count;
    }
    if (std::isfinite(run.d_conn)) {
      conn_sum += run.d_conn;
      ++conn_count;
    }
    if (run.is_success) {
      success_iter_sum += run.iter;
      success_elapsed_sum += run.elapsed;
      success_rollout_sum += run.elapsed_rollout;
      success_clustering_sum += run.elapsed_clustering;
      success_connection_sum += run.elapsed_connection;
      success_guide_sum += run.elapsed_guide;
      success_elapsed_values.push_back(run.elapsed);
      success_rollout_values.push_back(run.elapsed_rollout);
      success_clustering_values.push_back(run.elapsed_clustering);
      success_connection_values.push_back(run.elapsed_connection);
      success_guide_values.push_back(run.elapsed_guide);
    }
  }

  summary.nfail = summary.num_runs - summary.num_success;
  summary.success_rate =
      static_cast<double>(summary.num_success) / summary.num_runs;
  summary.nbar_iter = iter_sum / summary.num_runs;
  summary.tbar_elapsed = elapsed_sum / summary.num_runs;
  summary.dbar_goal = goal_sum / summary.num_runs;
  if (cost_count > 0) {
    summary.avg_cost = cost_sum / cost_count;
  }
  if (conn_count > 0) {
    summary.dbar_conn = conn_sum / conn_count;
  }
  if (summary.num_success > 0) {
    const double denom = static_cast<double>(summary.num_success);
    summary.avg_iter_success = success_iter_sum / denom;
    summary.avg_total_elapsed_success = success_elapsed_sum / denom;
    summary.avg_rollout_time_success = success_rollout_sum / denom;
    summary.avg_clustering_time_success = success_clustering_sum / denom;
    summary.avg_connection_time_success = success_connection_sum / denom;
    summary.avg_guide_time_success = success_guide_sum / denom;
    summary.total_elapsed_success_q1 =
        wmrobotGpuQuantile(success_elapsed_values, 0.25);
    summary.total_elapsed_success_q2 =
        wmrobotGpuQuantile(success_elapsed_values, 0.50);
    summary.total_elapsed_success_q3 =
        wmrobotGpuQuantile(success_elapsed_values, 0.75);
    summary.rollout_time_success_q1 =
        wmrobotGpuQuantile(success_rollout_values, 0.25);
    summary.rollout_time_success_q2 =
        wmrobotGpuQuantile(success_rollout_values, 0.50);
    summary.rollout_time_success_q3 =
        wmrobotGpuQuantile(success_rollout_values, 0.75);
    summary.clustering_time_success_q1 =
        wmrobotGpuQuantile(success_clustering_values, 0.25);
    summary.clustering_time_success_q2 =
        wmrobotGpuQuantile(success_clustering_values, 0.50);
    summary.clustering_time_success_q3 =
        wmrobotGpuQuantile(success_clustering_values, 0.75);
    summary.connection_time_success_q1 =
        wmrobotGpuQuantile(success_connection_values, 0.25);
    summary.connection_time_success_q2 =
        wmrobotGpuQuantile(success_connection_values, 0.50);
    summary.connection_time_success_q3 =
        wmrobotGpuQuantile(success_connection_values, 0.75);
    summary.guide_time_success_q1 =
        wmrobotGpuQuantile(success_guide_values, 0.25);
    summary.guide_time_success_q2 =
        wmrobotGpuQuantile(success_guide_values, 0.50);
    summary.guide_time_success_q3 =
        wmrobotGpuQuantile(success_guide_values, 0.75);
  }
  return summary;
}
