#pragma once

#include <bi_mppi_gpu.cuh>
#include <cluster_mppi_gpu.cuh>
#include <mppi_gpu.cuh>
#include <wmrobot_map.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class GpuAblationVariant {
  MPPI,
  ClusterMPPI,
  BiCNoGuide,
  BiCNoBackward,
  BiCNoClustering,
  FullBiC
};

struct GpuAblationConfig {
  int map_begin = 299;
  int num_maps = 300;
  int start_cases = 2;
  int maxiter = 200;
  int N = 10000;
  int Nf = 10000;
  int Nb = 10000;
  int Nr = 5000;
  int T = 100;
  int Tf = 50;
  int Tb = 50;
  int raw_connection_candidates = 64;
  double dt = 0.1;
  double gamma_u = 10.0;
  double sigma_v = 0.6;
  double sigma_w = 0.6;
  double goal_tolerance = 0.1;
  double deviation_mu = 1.0;
  double epsilon = 0.01;
  int minpts = 5;
  std::uint_fast64_t seed = 1;
  std::string dataset_dir = "../BARN_dataset/txt_files";
  std::string output_prefix = "wmrobot_ablation_gpu";
};

struct GpuAblationRunResult {
  std::string variant;
  int start_case = 0;
  int map_id = 0;
  bool is_success = false;
  bool is_collision = false;
  int iter = 0;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;
  double d_goal = 0.0;
  double d_conn = std::numeric_limits<double>::quiet_NaN();
};

struct GpuAblationSummary {
  std::string variant;
  int num_runs = 0;
  int num_success = 0;
  int nfail = 0;
  double success_rate = 0.0;
  double nbar_iter = 0.0;
  double tbar_elapsed = 0.0;
  double dbar_goal = 0.0;
  double dbar_conn = std::numeric_limits<double>::quiet_NaN();
};

inline std::string gpuCsvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

inline std::string gpuVariantFileToken(GpuAblationVariant variant) {
  switch (variant) {
  case GpuAblationVariant::MPPI:
    return "mppi";
  case GpuAblationVariant::ClusterMPPI:
    return "cluster_mppi";
  case GpuAblationVariant::BiCNoGuide:
    return "bic_without_guide";
  case GpuAblationVariant::BiCNoBackward:
    return "bic_without_backward";
  case GpuAblationVariant::BiCNoClustering:
    return "bic_without_clustering";
  case GpuAblationVariant::FullBiC:
    return "full_bic_mppi";
  }
  return "unknown";
}

inline std::string gpuVariantLabel(GpuAblationVariant variant) {
  switch (variant) {
  case GpuAblationVariant::MPPI:
    return "MPPI";
  case GpuAblationVariant::ClusterMPPI:
    return "Cluster-MPPI";
  case GpuAblationVariant::BiCNoGuide:
    return "BiC without Guide";
  case GpuAblationVariant::BiCNoBackward:
    return "BiC without Backward";
  case GpuAblationVariant::BiCNoClustering:
    return "BiC without Clustering";
  case GpuAblationVariant::FullBiC:
    return "Full BiC-MPPI";
  }
  return "Unknown";
}

inline Eigen::VectorXd gpuWmrobotStartState(int start_case) {
  Eigen::VectorXd x(3);
  x << 2.5, 0.0, M_PI_2;
  switch (start_case) {
  case 0:
    x(0) = 0.5;
    break;
  case 1:
    x(0) = 2.5;
    break;
  case 2:
    x(0) = 1.5;
    break;
  default:
    x(0) = 0.5 + static_cast<double>(start_case % 3);
    break;
  }
  return x;
}

inline Eigen::VectorXd gpuWmrobotTargetState() {
  Eigen::VectorXd x(3);
  x << 1.5, 5.0, M_PI_2;
  return x;
}

inline Eigen::MatrixXd gpuSigmaMatrix(const GpuAblationConfig &config) {
  Eigen::VectorXd sigma(2);
  sigma << config.sigma_v, config.sigma_w;
  return sigma.asDiagonal();
}

inline void parseGpuAblationArgs(int argc, char **argv,
                                 GpuAblationConfig &config) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--map-begin") {
      config.map_begin = std::stoi(require_value(key));
    } else if (key == "--num-maps") {
      config.num_maps = std::stoi(require_value(key));
    } else if (key == "--start-cases") {
      config.start_cases = std::stoi(require_value(key));
    } else if (key == "--maxiter") {
      config.maxiter = std::stoi(require_value(key));
    } else if (key == "--N") {
      config.N = std::stoi(require_value(key));
    } else if (key == "--Nf") {
      config.Nf = std::stoi(require_value(key));
    } else if (key == "--Nb") {
      config.Nb = std::stoi(require_value(key));
    } else if (key == "--Nr") {
      config.Nr = std::stoi(require_value(key));
    } else if (key == "--T") {
      config.T = std::stoi(require_value(key));
    } else if (key == "--Tf") {
      config.Tf = std::stoi(require_value(key));
    } else if (key == "--Tb") {
      config.Tb = std::stoi(require_value(key));
    } else if (key == "--raw-candidates") {
      config.raw_connection_candidates = std::stoi(require_value(key));
    } else if (key == "--sigma-v") {
      config.sigma_v = std::stod(require_value(key));
    } else if (key == "--sigma-w") {
      config.sigma_w = std::stod(require_value(key));
    } else if (key == "--seed") {
      config.seed = static_cast<std::uint_fast64_t>(
          std::stoull(require_value(key)));
    } else if (key == "--dataset-dir") {
      config.dataset_dir = require_value(key);
    } else if (key == "--output-prefix") {
      config.output_prefix = require_value(key);
    } else if (key == "--smoke") {
      config.num_maps = 2;
      config.start_cases = 1;
      config.maxiter = 20;
      config.N = 256;
      config.Nf = 256;
      config.Nb = 256;
      config.Nr = 128;
      config.raw_connection_candidates = 24;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
}

struct GpuRawBranch {
  Eigen::MatrixXd U;
  Eigen::MatrixXd X;
  double cost = std::numeric_limits<double>::infinity();
};

class GpuBiMPPIAblationSolver : public BiMPPI_GPU {
public:
  template <typename ModelClass>
  explicit GpuBiMPPIAblationSolver(ModelClass model) : BiMPPI_GPU(model) {}

  void solveVariant(GpuAblationVariant variant,
                    const GpuAblationConfig &config);
  double connectionDistance() const { return last_connection_distance; }

private:
  double last_connection_distance = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd rolloutForwardCpu(const Eigen::VectorXd &x0,
                                    const Eigen::MatrixXd &U) const;
  Eigen::MatrixXd rolloutBackwardCpu(const Eigen::VectorXd &goal,
                                     const Eigen::MatrixXd &U) const;
  bool trajectoryFeasible(const Eigen::MatrixXd &X) const;
  double trajectoryCost(const Eigen::MatrixXd &X) const;
  double connectionDistanceFromJoints() const;
  void chooseBestConnectedCandidate();
  void chooseBestForwardCandidate();
  std::vector<int> topFeasibleIndices(const Eigen::VectorXd &costs,
                                      int max_count) const;
  void solveNoGuide();
  void solveNoBackward();
  void solveNoClustering(const GpuAblationConfig &config);
  void solveFullWithConnectionMetric();
};

inline Eigen::MatrixXd GpuBiMPPIAblationSolver::rolloutForwardCpu(
    const Eigen::VectorXd &x0, const Eigen::MatrixXd &U) const {
  Eigen::MatrixXd X(dim_x, U.cols() + 1);
  X.col(0) = x0;
  for (int t = 0; t < U.cols(); ++t) {
    X.col(t + 1) = X.col(t) + static_cast<double>(dt) * f(X.col(t), U.col(t));
  }
  return X;
}

inline Eigen::MatrixXd GpuBiMPPIAblationSolver::rolloutBackwardCpu(
    const Eigen::VectorXd &goal, const Eigen::MatrixXd &U) const {
  const int Tlocal = U.cols();
  Eigen::MatrixXd X(dim_x, Tlocal + 1);
  X.col(Tlocal) = goal;
  for (int t = Tlocal - 1; t >= 0; --t) {
    const int u_col = (t == Tlocal - 1) ? t : t + 1;
    X.col(t) =
        X.col(t + 1) - static_cast<double>(dt) * f(X.col(t + 1), U.col(u_col));
  }
  return X;
}

inline bool
GpuBiMPPIAblationSolver::trajectoryFeasible(const Eigen::MatrixXd &X) const {
  for (int t = 0; t < X.cols(); ++t) {
    if (collision_checker->getCollisionGrid(X.col(t))) {
      return false;
    }
  }
  return true;
}

inline double
GpuBiMPPIAblationSolver::trajectoryCost(const Eigen::MatrixXd &X) const {
  double cost = 0.0;
  for (int t = 0; t < X.cols(); ++t) {
    cost += p(X.col(t), x_target);
  }
  return cost;
}

inline double GpuBiMPPIAblationSolver::connectionDistanceFromJoints() const {
  double total = 0.0;
  int count = 0;
  for (const auto &joint : joints) {
    const int cf = joint[0];
    const int cb = joint[1];
    const int df = joint[2];
    const int db = joint[3];
    total +=
        (Xf.block(cf * dim_x, df, dim_x, 1) -
         Xb.block(cb * dim_x, db, dim_x, 1))
            .norm();
    ++count;
  }
  if (count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return total / count;
}

inline void GpuBiMPPIAblationSolver::chooseBestConnectedCandidate() {
  int best = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(Xc.size()); ++i) {
    if (!trajectoryFeasible(Xc[i])) {
      continue;
    }
    const double cost = trajectoryCost(Xc[i]);
    if (cost < best_cost) {
      best_cost = cost;
      best = i;
    }
  }
  if (Uc.empty()) {
    Uo = Eigen::MatrixXd::Zero(dim_u, Tf);
    Xo = rolloutForwardCpu(x_init, Uo);
  } else {
    Uo = Uc[best];
    Xo = Xc[best];
  }
  if (Uo.cols() == 0) {
    Uo = Eigen::MatrixXd::Zero(dim_u, 1);
  }
  u0 = Uo.col(0);
}

inline void GpuBiMPPIAblationSolver::chooseBestForwardCandidate() {
  int best = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int ci = 0; ci < static_cast<int>(clusters_f.size()); ++ci) {
    const Eigen::MatrixXd Xi = Xf.block(ci * dim_x, 0, dim_x, Tf + 1);
    if (!trajectoryFeasible(Xi)) {
      continue;
    }
    const double terminal = p(Xi.col(Tf), x_target);
    const double cost = trajectoryCost(Xi) + terminal;
    if (cost < best_cost) {
      best_cost = cost;
      best = ci;
    }
  }
  Uo = Uf.middleRows(best * dim_u, dim_u);
  Xo = Xf.block(best * dim_x, 0, dim_x, Tf + 1);
  u0 = Uo.col(0);
  last_connection_distance = (Xo.col(Xo.cols() - 1) - x_target).norm();
}

inline std::vector<int> GpuBiMPPIAblationSolver::topFeasibleIndices(
    const Eigen::VectorXd &costs, int max_count) const {
  std::vector<int> indices;
  for (int i = 0; i < costs.size(); ++i) {
    if (costs(i) < 1e7) {
      indices.push_back(i);
    }
  }
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return costs(a) < costs(b); });
  if (static_cast<int>(indices.size()) > max_count) {
    indices.resize(max_count);
  }
  return indices;
}

inline void GpuBiMPPIAblationSolver::solveNoGuide() {
  elapsed_rollout = elapsed_clustering = 0.0;
  const auto t0 = std::chrono::high_resolution_clock::now();
  backwardRollout();
  forwardRollout();
  const auto t1 = std::chrono::high_resolution_clock::now();
  selectConnection();
  last_connection_distance = connectionDistanceFromJoints();
  concatenate();
  chooseBestConnectedCandidate();
  const auto t2 = std::chrono::high_resolution_clock::now();
  elapsed_connection = std::chrono::duration<double>(t2 - t1).count();
  elapsed_guide = 0.0;
  partitioningControl();
  elapsed = elapsed_rollout + elapsed_clustering + elapsed_connection;
  elapsed_1 = t2 - t0;
  visual_traj.push_back(x_init);
}

inline void GpuBiMPPIAblationSolver::solveNoBackward() {
  elapsed_rollout = elapsed_clustering = 0.0;
  const auto t0 = std::chrono::high_resolution_clock::now();
  forwardRollout();
  const auto t1 = std::chrono::high_resolution_clock::now();
  chooseBestForwardCandidate();
  const auto t2 = std::chrono::high_resolution_clock::now();
  elapsed_connection = std::chrono::duration<double>(t2 - t1).count();
  elapsed_guide = 0.0;
  partitioningControl();
  elapsed = elapsed_rollout + elapsed_clustering + elapsed_connection;
  elapsed_1 = t2 - t0;
  visual_traj.push_back(x_init);
}

inline void GpuBiMPPIAblationSolver::solveNoClustering(
    const GpuAblationConfig &config) {
  elapsed_rollout = elapsed_clustering = 0.0;
  Eigen::VectorXd costs_f;
  Eigen::VectorXd costs_b;
  Eigen::MatrixXd Ui_f;
  Eigen::MatrixXd Ui_b;

  const auto t0 = std::chrono::high_resolution_clock::now();
  backwardRawRollout(costs_b, Ui_b);
  forwardRawRollout(costs_f, Ui_f);
  const auto t1 = std::chrono::high_resolution_clock::now();

  auto f_indices =
      topFeasibleIndices(costs_f, config.raw_connection_candidates);
  auto b_indices =
      topFeasibleIndices(costs_b, config.raw_connection_candidates);

  std::vector<GpuRawBranch> f_branches;
  std::vector<GpuRawBranch> b_branches;
  for (int idx : f_indices) {
    GpuRawBranch branch;
    branch.U = Ui_f.middleRows(idx * dim_u, dim_u);
    branch.X = rolloutForwardCpu(x_init, branch.U);
    branch.cost = costs_f(idx);
    f_branches.push_back(branch);
  }
  for (int idx : b_indices) {
    GpuRawBranch branch;
    branch.U = Ui_b.middleRows(idx * dim_u, dim_u);
    branch.X = rolloutBackwardCpu(x_target, branch.U);
    branch.cost = costs_b(idx);
    b_branches.push_back(branch);
  }

  int best_f = 0;
  int best_b = 0;
  int best_tf = 0;
  int best_tb = Tb;
  double best_connection = std::numeric_limits<double>::infinity();
  for (int fi = 0; fi < static_cast<int>(f_branches.size()); ++fi) {
    for (int bi = 0; bi < static_cast<int>(b_branches.size()); ++bi) {
      for (int tf = 0; tf < f_branches[fi].X.cols(); ++tf) {
        for (int tb = 0; tb < b_branches[bi].X.cols(); ++tb) {
          const double distance =
              (f_branches[fi].X.col(tf) - b_branches[bi].X.col(tb)).norm();
          if (distance < best_connection) {
            best_connection = distance;
            best_f = fi;
            best_b = bi;
            best_tf = tf;
            best_tb = tb;
          }
        }
      }
    }
  }

  if (f_branches.empty()) {
    Uo = Eigen::MatrixXd::Zero(dim_u, Tf);
  } else if (b_branches.empty()) {
    Uo = f_branches[best_f].U;
    best_connection =
        (f_branches[best_f].X.col(f_branches[best_f].X.cols() - 1) - x_target)
            .norm();
  } else {
    const int len = std::max(Tf, best_tf + (Tb - best_tb));
    Uo = Eigen::MatrixXd::Zero(dim_u, len);
    if (best_tf > 0) {
      Uo.leftCols(best_tf) = f_branches[best_f].U.leftCols(best_tf);
    }
    if (best_tb < Tb) {
      Uo.middleCols(best_tf, Tb - best_tb) =
          b_branches[best_b].U.middleCols(best_tb, Tb - best_tb);
    }
  }
  h(Uo);
  Xo = rolloutForwardCpu(x_init, Uo);
  u0 = Uo.col(0);
  last_connection_distance = best_connection;

  Uc.clear();
  Xc.clear();
  joints.clear();
  Uc.push_back(Uo);
  Xc.push_back(Xo);
  joints.push_back({0, 0, best_tf, best_tb});

  const auto t2 = std::chrono::high_resolution_clock::now();
  elapsed_clustering = 0.0;
  elapsed_connection = std::chrono::duration<double>(t2 - t1).count();
  guideMPPI();
  const auto t3 = std::chrono::high_resolution_clock::now();
  elapsed_guide = std::chrono::duration<double>(t3 - t2).count();
  last_connection_distance = best_connection;
  partitioningControl();
  elapsed = elapsed_rollout + elapsed_connection + elapsed_guide;
  elapsed_1 = t3 - t0;
  visual_traj.push_back(x_init);
}

inline void GpuBiMPPIAblationSolver::solveFullWithConnectionMetric() {
  elapsed_rollout = elapsed_clustering = 0.0;
  const auto t0 = std::chrono::high_resolution_clock::now();
  backwardRollout();
  forwardRollout();
  const auto t1 = std::chrono::high_resolution_clock::now();
  selectConnection();
  last_connection_distance = connectionDistanceFromJoints();
  concatenate();
  const auto t2 = std::chrono::high_resolution_clock::now();
  elapsed_connection = std::chrono::duration<double>(t2 - t1).count();
  guideMPPI();
  const auto t3 = std::chrono::high_resolution_clock::now();
  elapsed_guide = std::chrono::duration<double>(t3 - t2).count();
  partitioningControl();
  elapsed = elapsed_rollout + elapsed_clustering + elapsed_connection +
            elapsed_guide;
  elapsed_1 = t3 - t0;
  visual_traj.push_back(x_init);
}

inline void GpuBiMPPIAblationSolver::solveVariant(
    GpuAblationVariant variant, const GpuAblationConfig &config) {
  switch (variant) {
  case GpuAblationVariant::BiCNoGuide:
    solveNoGuide();
    break;
  case GpuAblationVariant::BiCNoBackward:
    solveNoBackward();
    break;
  case GpuAblationVariant::BiCNoClustering:
    solveNoClustering(config);
    break;
  case GpuAblationVariant::FullBiC:
    solveFullWithConnectionMetric();
    break;
  default:
    solveFullWithConnectionMetric();
    break;
  }
}

inline MPPIParam makeGpuMppiParam(const GpuAblationConfig &config,
                                  const Eigen::VectorXd &x_init,
                                  const Eigen::VectorXd &x_target) {
  MPPIParam param;
  param.dt = config.dt;
  param.T = config.T;
  param.x_init = x_init;
  param.x_target = x_target;
  param.N = config.N;
  param.gamma_u = config.gamma_u;
  param.sigma_u = gpuSigmaMatrix(config);
  return param;
}

inline BiMPPIParam makeGpuBiParam(const GpuAblationConfig &config,
                                  const Eigen::VectorXd &x_init,
                                  const Eigen::VectorXd &x_target) {
  BiMPPIParam param;
  param.dt = config.dt;
  param.Tf = config.Tf;
  param.Tb = config.Tb;
  param.x_init = x_init;
  param.x_target = x_target;
  param.Nf = config.Nf;
  param.Nb = config.Nb;
  param.Nr = config.Nr;
  param.gamma_u = config.gamma_u;
  param.sigma_u = gpuSigmaMatrix(config);
  param.deviation_mu = config.deviation_mu;
  param.cost_mu = 1.0;
  param.minpts = config.minpts;
  param.epsilon = config.epsilon;
  param.psi = 0.6;
  return param;
}

inline void writeGpuRunHeader(std::ofstream &csv) {
  csv << "variant,start_case,map,is_success,is_collision,iter,elapsed,"
         "elapsed_rollout,elapsed_clustering,elapsed_connection,elapsed_guide,"
         "d_goal,d_conn\n";
}

inline void writeGpuRunRow(std::ofstream &csv,
                           const GpuAblationRunResult &row) {
  csv << row.variant << ',' << row.start_case << ',' << row.map_id << ','
      << row.is_success << ',' << row.is_collision << ',' << row.iter << ','
      << gpuCsvDouble(row.elapsed) << ',' << gpuCsvDouble(row.elapsed_rollout)
      << ',' << gpuCsvDouble(row.elapsed_clustering) << ','
      << gpuCsvDouble(row.elapsed_connection) << ','
      << gpuCsvDouble(row.elapsed_guide) << ',' << gpuCsvDouble(row.d_goal)
      << ',' << gpuCsvDouble(row.d_conn) << '\n';
}

inline void writeGpuSummaryHeader(std::ofstream &csv) {
  csv << "Variant,Success Rate,Nfail,Nbar_iter,Tbar_elapsed,dbar_goal,dbar_conn,"
         "num_runs,num_success\n";
}

inline void writeGpuSummaryRow(std::ofstream &csv,
                               const GpuAblationSummary &summary) {
  csv << summary.variant << ',' << gpuCsvDouble(summary.success_rate) << ','
      << summary.nfail << ',' << gpuCsvDouble(summary.nbar_iter) << ','
      << gpuCsvDouble(summary.tbar_elapsed) << ','
      << gpuCsvDouble(summary.dbar_goal) << ','
      << gpuCsvDouble(summary.dbar_conn) << ',' << summary.num_runs << ','
      << summary.num_success << '\n';
}

inline GpuAblationSummary
summarizeGpuRuns(const std::string &variant,
                 const std::vector<GpuAblationRunResult> &runs) {
  GpuAblationSummary summary;
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
  for (const auto &run : runs) {
    summary.num_success += run.is_success ? 1 : 0;
    iter_sum += run.iter;
    elapsed_sum += run.elapsed;
    goal_sum += run.d_goal;
    if (std::isfinite(run.d_conn)) {
      conn_sum += run.d_conn;
      ++conn_count;
    }
  }
  summary.nfail = summary.num_runs - summary.num_success;
  summary.success_rate =
      static_cast<double>(summary.num_success) / summary.num_runs;
  summary.nbar_iter = iter_sum / summary.num_runs;
  summary.tbar_elapsed = elapsed_sum / summary.num_runs;
  summary.dbar_goal = goal_sum / summary.num_runs;
  if (conn_count > 0) {
    summary.dbar_conn = conn_sum / conn_count;
  }
  return summary;
}

inline std::vector<int> gpuMapIds(const GpuAblationConfig &config) {
  std::vector<int> ids;
  for (int i = 0; i < config.num_maps; ++i) {
    ids.push_back(config.map_begin - i);
  }
  return ids;
}

template <typename Solver>
inline GpuAblationRunResult runGpuOneDirectionalSolver(
    GpuAblationVariant variant, Solver &solver, CollisionChecker &collision_checker,
    const Eigen::VectorXd &target, const GpuAblationConfig &config,
    int start_case, int map_id) {
  GpuAblationRunResult row;
  row.variant = gpuVariantLabel(variant);
  row.start_case = start_case;
  row.map_id = map_id;

  for (row.iter = 0; row.iter < config.maxiter; ++row.iter) {
    solver.solve();
    solver.move();

    row.elapsed += solver.elapsed;
    row.elapsed_rollout += solver.elapsed_rollout;
    row.elapsed_clustering += solver.elapsed_clustering;
    row.elapsed_connection += solver.elapsed_connection;
    row.elapsed_guide += solver.elapsed_guide;

    if (collision_checker.getCollisionGrid(solver.x_init)) {
      row.is_collision = true;
      break;
    }
    row.d_goal = (solver.x_init - target).norm();
    if (row.d_goal < config.goal_tolerance) {
      row.is_success = true;
      break;
    }
  }
  if (row.iter >= config.maxiter) {
    row.iter = config.maxiter;
  }
  return row;
}

inline GpuAblationRunResult runGpuBiSolver(
    GpuAblationVariant variant, GpuBiMPPIAblationSolver &solver,
    CollisionChecker &collision_checker, const Eigen::VectorXd &target,
    const GpuAblationConfig &config, int start_case, int map_id) {
  GpuAblationRunResult row;
  row.variant = gpuVariantLabel(variant);
  row.start_case = start_case;
  row.map_id = map_id;

  for (row.iter = 0; row.iter < config.maxiter; ++row.iter) {
    solver.solveVariant(variant, config);
    solver.move();

    row.elapsed += solver.elapsed;
    row.elapsed_rollout += solver.elapsed_rollout;
    row.elapsed_clustering += solver.elapsed_clustering;
    row.elapsed_connection += solver.elapsed_connection;
    row.elapsed_guide += solver.elapsed_guide;
    row.d_conn = solver.connectionDistance();

    if (collision_checker.getCollisionGrid(solver.x_init)) {
      row.is_collision = true;
      break;
    }
    row.d_goal = (solver.x_init - target).norm();
    if (row.d_goal < config.goal_tolerance) {
      row.is_success = true;
      break;
    }
  }
  if (row.iter >= config.maxiter) {
    row.iter = config.maxiter;
  }
  return row;
}

inline std::vector<GpuAblationRunResult>
runGpuAblationVariant(GpuAblationVariant variant,
                      const GpuAblationConfig &config) {
  std::vector<GpuAblationRunResult> rows;
  WMRobotMap model;
  const Eigen::VectorXd target = gpuWmrobotTargetState();

  for (int start_case = 0; start_case < config.start_cases; ++start_case) {
    for (int map_id : gpuMapIds(config)) {
      CollisionChecker collision_checker;
      collision_checker.loadMap(config.dataset_dir + "/output_" +
                                    std::to_string(map_id) + ".txt",
                                0.1);
      const Eigen::VectorXd start = gpuWmrobotStartState(start_case);
      const auto run_seed =
          config.seed + static_cast<std::uint_fast64_t>(1009 * map_id +
                                                        9176 * start_case);

      GpuAblationRunResult row;
      if (variant == GpuAblationVariant::MPPI) {
        MPPI_GPU solver(model);
        auto param = makeGpuMppiParam(config, start, target);
        solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
        solver.init(param);
        solver.setSeed(run_seed);
        solver.setCollisionChecker(&collision_checker);
        row = runGpuOneDirectionalSolver(variant, solver, collision_checker,
                                         target, config, start_case, map_id);
      } else if (variant == GpuAblationVariant::ClusterMPPI) {
        ClusterMPPI_GPU solver(model);
        auto param = makeGpuMppiParam(config, start, target);
        solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
        solver.init(param);
        solver.setSeed(run_seed);
        solver.setCollisionChecker(&collision_checker);
        row = runGpuOneDirectionalSolver(variant, solver, collision_checker,
                                         target, config, start_case, map_id);
      } else {
        GpuBiMPPIAblationSolver solver(model);
        auto param = makeGpuBiParam(config, start, target);
        solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
        solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
        solver.init(param);
        solver.setSeed(run_seed);
        solver.setCollisionChecker(&collision_checker);
        row = runGpuBiSolver(variant, solver, collision_checker, target, config,
                             start_case, map_id);
      }

      rows.push_back(row);
      std::cout << row.variant << '\t' << row.start_case << '\t' << row.map_id
                << '\t' << row.is_success << '\t' << row.iter << '\t'
                << row.elapsed << '\t' << row.d_goal << '\t'
                << gpuCsvDouble(row.d_conn) << std::endl;
    }
  }
  return rows;
}

inline GpuAblationSummary
runAndWriteGpuAblationVariant(GpuAblationVariant variant,
                              const GpuAblationConfig &config) {
  const std::string token = gpuVariantFileToken(variant);
  const std::string runs_path =
      config.output_prefix + "_" + token + "_runs.csv";
  const std::string summary_path =
      config.output_prefix + "_" + token + "_summary.csv";

  const auto rows = runGpuAblationVariant(variant, config);
  const auto summary = summarizeGpuRuns(gpuVariantLabel(variant), rows);

  std::ofstream runs_csv(runs_path);
  writeGpuRunHeader(runs_csv);
  for (const auto &row : rows) {
    writeGpuRunRow(runs_csv, row);
  }

  std::ofstream summary_csv(summary_path);
  writeGpuSummaryHeader(summary_csv);
  writeGpuSummaryRow(summary_csv, summary);
  return summary;
}
