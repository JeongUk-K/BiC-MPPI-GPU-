#pragma once

#include <collision_checker.h>
#include <mppi_param.h>
#include <wmrobot_map.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class AblationVariant {
  MPPI,
  ClusterMPPI,
  BiCNoGuide,
  BiCNoBackward,
  BiCNoClustering,
  FullBiC
};

struct AblationConfig {
  int map_begin = 299;
  int num_maps = 10;
  int start_cases = 2;
  int maxiter = 120;
  int N = 800;
  int Nf = 800;
  int Nb = 800;
  int Nr = 500;
  int T = 100;
  int Tf = 50;
  int Tb = 50;
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
  std::string output_prefix = "wmrobot_ablation";
};

struct AblationRunResult {
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

struct AblationSummary {
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

inline bool isFiniteMetric(double value) { return std::isfinite(value); }

inline std::string csvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

inline std::string variantFileToken(AblationVariant variant) {
  switch (variant) {
  case AblationVariant::MPPI:
    return "mppi";
  case AblationVariant::ClusterMPPI:
    return "cluster_mppi";
  case AblationVariant::BiCNoGuide:
    return "bic_without_guide";
  case AblationVariant::BiCNoBackward:
    return "bic_without_backward";
  case AblationVariant::BiCNoClustering:
    return "bic_without_clustering";
  case AblationVariant::FullBiC:
    return "full_bic_mppi";
  }
  return "unknown";
}

inline std::string variantLabel(AblationVariant variant) {
  switch (variant) {
  case AblationVariant::MPPI:
    return "MPPI";
  case AblationVariant::ClusterMPPI:
    return "Cluster-MPPI";
  case AblationVariant::BiCNoGuide:
    return "BiC without Guide";
  case AblationVariant::BiCNoBackward:
    return "BiC without Backward";
  case AblationVariant::BiCNoClustering:
    return "BiC without Clustering";
  case AblationVariant::FullBiC:
    return "Full BiC-MPPI";
  }
  return "Unknown";
}

inline Eigen::VectorXd wmrobotStartState(int start_case) {
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

inline Eigen::VectorXd wmrobotTargetState() {
  Eigen::VectorXd x(3);
  x << 1.5, 5.0, M_PI_2;
  return x;
}

inline void parseAblationArgs(int argc, char **argv, AblationConfig &config) {
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
      config.N = 120;
      config.Nf = 120;
      config.Nb = 120;
      config.Nr = 80;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
}

struct RolloutSample {
  Eigen::MatrixXd U;
  Eigen::MatrixXd X;
  Eigen::VectorXd deviation;
  double cost = std::numeric_limits<double>::infinity();
  bool feasible = false;
};

struct RepresentativeBranch {
  Eigen::MatrixXd U;
  Eigen::MatrixXd X;
  double cost = std::numeric_limits<double>::infinity();
};

class WmrobotAblationSolver {
public:
  WmrobotAblationSolver(const WMRobotMap &model,
                        CollisionChecker *collision_checker,
                        const AblationConfig &config,
                        AblationVariant variant);

  void reset(const Eigen::VectorXd &x_init, const Eigen::VectorXd &x_target);
  void solve();
  void move();

  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd Uo;
  Eigen::MatrixXd Xo;
  Eigen::VectorXd u0;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;
  double last_connection_distance = std::numeric_limits<double>::quiet_NaN();

private:
  int dim_x;
  int dim_u;
  float dt;
  AblationConfig config;
  AblationVariant variant;
  CollisionChecker *collision_checker;
  std::uint_fast64_t solve_count = 0;

  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
  std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

  Eigen::MatrixXd U0;
  Eigen::MatrixXd Uf0;
  Eigen::MatrixXd Ub0;

  Eigen::MatrixXd makeNoise(int T, int sample_index, int phase_index,
                            std::uint_fast64_t solve_index) const;
  double trajectoryCost(const Eigen::MatrixXd &X,
                        const Eigen::VectorXd &target) const;
  bool trajectoryFeasible(const Eigen::MatrixXd &X) const;
  Eigen::MatrixXd rolloutForward(const Eigen::VectorXd &x0,
                                 const Eigen::MatrixXd &U) const;
  Eigen::MatrixXd rolloutBackward(const Eigen::VectorXd &goal,
                                  const Eigen::MatrixXd &U) const;
  std::vector<RolloutSample> sampleForward(int N, int T,
                                           const Eigen::MatrixXd &nominal,
                                           const Eigen::VectorXd &start,
                                           const Eigen::VectorXd &target,
                                           int phase,
                                           std::uint_fast64_t solve_index);
  std::vector<RolloutSample> sampleBackward(int N, int T,
                                            const Eigen::MatrixXd &nominal,
                                            const Eigen::VectorXd &target,
                                            const Eigen::VectorXd &start,
                                            int phase,
                                            std::uint_fast64_t solve_index);
  std::vector<std::vector<int>> dbscan(const std::vector<RolloutSample> &samples,
                                       int minpts, double epsilon) const;
  std::vector<RepresentativeBranch>
  representatives(const std::vector<RolloutSample> &samples,
                  const std::vector<std::vector<int>> &clusters, int T,
                  bool backward) const;
  RepresentativeBranch weightedBranch(const std::vector<RolloutSample> &samples,
                                      const std::vector<int> &indices, int T,
                                      bool backward) const;
  Eigen::MatrixXd weightedControls(const std::vector<RolloutSample> &samples,
                                   int T) const;
  int bestForwardRepresentative(const std::vector<RepresentativeBranch> &branches,
                                const Eigen::VectorXd &target) const;
  void solveMPPI();
  void solveClusterMPPI();
  void solveBiC(bool use_guide, bool use_backward, bool use_clustering);
  void guide(const Eigen::MatrixXd &reference_u);
  void setResult(const Eigen::MatrixXd &U, double connection_distance);
};

inline WmrobotAblationSolver::WmrobotAblationSolver(
    const WMRobotMap &model, CollisionChecker *collision_checker,
    const AblationConfig &config, AblationVariant variant)
    : dim_x(model.dim_x), dim_u(model.dim_u), dt(config.dt), config(config),
      variant(variant), collision_checker(collision_checker), f(model.f),
      p(model.p), h(model.h) {
  u0 = Eigen::VectorXd::Zero(dim_u);
}

inline void WmrobotAblationSolver::reset(const Eigen::VectorXd &x0,
                                         const Eigen::VectorXd &target) {
  x_init = x0;
  x_target = target;
  U0 = Eigen::MatrixXd::Zero(dim_u, config.T);
  Uf0 = Eigen::MatrixXd::Zero(dim_u, config.Tf);
  Ub0 = Eigen::MatrixXd::Zero(dim_u, config.Tb);
  Uo = Eigen::MatrixXd::Zero(dim_u, config.T);
  Xo = Eigen::MatrixXd::Zero(dim_x, config.T + 1);
  u0 = Eigen::VectorXd::Zero(dim_u);
  solve_count = 0;
  last_connection_distance = std::numeric_limits<double>::quiet_NaN();
}

inline Eigen::MatrixXd WmrobotAblationSolver::makeNoise(
    int T, int sample_index, int phase_index,
    std::uint_fast64_t solve_index) const {
  std::uint_fast64_t mixed = config.seed;
  mixed ^= 0x9e3779b97f4a7c15ULL + static_cast<std::uint_fast64_t>(sample_index) +
           (mixed << 6) + (mixed >> 2);
  mixed ^= 0xbf58476d1ce4e5b9ULL + solve_index + (mixed << 6) +
           (mixed >> 2);
  mixed ^= 0x94d049bb133111ebULL + static_cast<std::uint_fast64_t>(phase_index) +
           (mixed << 6) + (mixed >> 2);

  std::mt19937_64 rng(mixed);
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd noise(dim_u, T);
  for (int r = 0; r < dim_u; ++r) {
    const double sigma = r == 0 ? config.sigma_v : config.sigma_w;
    for (int c = 0; c < T; ++c) {
      noise(r, c) = sigma * normal(rng);
    }
  }
  return noise;
}

inline Eigen::MatrixXd
WmrobotAblationSolver::rolloutForward(const Eigen::VectorXd &x0,
                                      const Eigen::MatrixXd &U) const {
  Eigen::MatrixXd X(dim_x, U.cols() + 1);
  X.col(0) = x0;
  for (int t = 0; t < U.cols(); ++t) {
    X.col(t + 1) = X.col(t) + dt * f(X.col(t), U.col(t));
  }
  return X;
}

inline Eigen::MatrixXd
WmrobotAblationSolver::rolloutBackward(const Eigen::VectorXd &goal,
                                       const Eigen::MatrixXd &U) const {
  Eigen::MatrixXd X(dim_x, U.cols() + 1);
  const int T = U.cols();
  X.col(T) = goal;
  for (int t = T - 1; t >= 0; --t) {
    const int u_col = (t == T - 1) ? t : t + 1;
    X.col(t) = X.col(t + 1) - dt * f(X.col(t + 1), U.col(u_col));
  }
  return X;
}

inline bool WmrobotAblationSolver::trajectoryFeasible(
    const Eigen::MatrixXd &X) const {
  for (int i = 0; i < X.cols(); ++i) {
    if (collision_checker->getCollisionGrid(X.col(i))) {
      return false;
    }
  }
  return true;
}

inline double
WmrobotAblationSolver::trajectoryCost(const Eigen::MatrixXd &X,
                                      const Eigen::VectorXd &target) const {
  double cost = 0.0;
  for (int i = 0; i < X.cols(); ++i) {
    cost += p(X.col(i), target);
  }
  return cost;
}

inline std::vector<RolloutSample> WmrobotAblationSolver::sampleForward(
    int N, int T, const Eigen::MatrixXd &nominal, const Eigen::VectorXd &start,
    const Eigen::VectorXd &target, int phase, std::uint_fast64_t solve_index) {
  std::vector<RolloutSample> samples(N);
  for (int i = 0; i < N; ++i) {
    samples[i].U = nominal + makeNoise(T, i, phase, solve_index);
    h(samples[i].U);
    samples[i].X = rolloutForward(start, samples[i].U);
    samples[i].feasible = trajectoryFeasible(samples[i].X);
    samples[i].cost = samples[i].feasible
                          ? trajectoryCost(samples[i].X, target)
                          : std::numeric_limits<double>::infinity();
    samples[i].deviation = (samples[i].U - nominal).rowwise().mean();
  }
  return samples;
}

inline std::vector<RolloutSample> WmrobotAblationSolver::sampleBackward(
    int N, int T, const Eigen::MatrixXd &nominal, const Eigen::VectorXd &target,
    const Eigen::VectorXd &start, int phase, std::uint_fast64_t solve_index) {
  std::vector<RolloutSample> samples(N);
  for (int i = 0; i < N; ++i) {
    samples[i].U = nominal + makeNoise(T, i, phase, solve_index);
    h(samples[i].U);
    samples[i].X = rolloutBackward(target, samples[i].U);
    samples[i].feasible = trajectoryFeasible(samples[i].X);
    samples[i].cost = samples[i].feasible ? trajectoryCost(samples[i].X, start)
                                          : std::numeric_limits<double>::infinity();
    samples[i].deviation = (samples[i].U - nominal).rowwise().mean();
  }
  return samples;
}

inline std::vector<std::vector<int>> WmrobotAblationSolver::dbscan(
    const std::vector<RolloutSample> &samples, int minpts,
    double epsilon) const {
  const int N = static_cast<int>(samples.size());
  std::vector<bool> core_points(N, false);
  std::map<int, std::vector<int>> neighbors;

  for (int i = 0; i < N; ++i) {
    if (!samples[i].feasible) {
      continue;
    }
    for (int j = i + 1; j < N; ++j) {
      if (!samples[j].feasible) {
        continue;
      }
      if (config.deviation_mu *
              (samples[i].deviation - samples[j].deviation).norm() <
          epsilon) {
        neighbors[i].push_back(j);
        neighbors[j].push_back(i);
      }
    }
  }

  for (int i = 0; i < N; ++i) {
    if (static_cast<int>(neighbors[i].size()) > minpts) {
      core_points[i] = true;
    }
  }

  std::vector<std::vector<int>> clusters;
  std::vector<bool> visited(N, false);
  for (int i = 0; i < N; ++i) {
    if (!core_points[i] || visited[i]) {
      continue;
    }
    std::deque<int> queue;
    std::vector<int> cluster;
    queue.push_back(i);
    visited[i] = true;
    while (!queue.empty()) {
      int current = queue.front();
      queue.pop_front();
      cluster.push_back(current);
      for (int next : neighbors[current]) {
        if (visited[next]) {
          continue;
        }
        visited[next] = true;
        if (core_points[next]) {
          queue.push_back(next);
        } else if (samples[next].feasible) {
          cluster.push_back(next);
        }
      }
    }
    clusters.push_back(cluster);
  }

  if (clusters.empty()) {
    std::vector<int> feasible;
    for (int i = 0; i < N; ++i) {
      if (samples[i].feasible) {
        feasible.push_back(i);
      }
    }
    if (!feasible.empty()) {
      clusters.push_back(feasible);
    }
  }
  return clusters;
}

inline RepresentativeBranch WmrobotAblationSolver::weightedBranch(
    const std::vector<RolloutSample> &samples, const std::vector<int> &indices,
    int T, bool backward) const {
  RepresentativeBranch branch;
  branch.U = Eigen::MatrixXd::Zero(dim_u, T);
  if (indices.empty()) {
    branch.X = Eigen::MatrixXd::Zero(dim_x, T + 1);
    return branch;
  }

  double min_cost = std::numeric_limits<double>::infinity();
  for (int idx : indices) {
    min_cost = std::min(min_cost, samples[idx].cost);
  }

  double total_weight = 0.0;
  for (int idx : indices) {
    const double weight = std::exp(-config.gamma_u * (samples[idx].cost - min_cost));
    branch.U += weight * samples[idx].U;
    total_weight += weight;
  }
  if (total_weight > 0.0) {
    branch.U /= total_weight;
  }
  h(branch.U);

  branch.X = backward ? rolloutBackward(x_target, branch.U)
                      : rolloutForward(x_init, branch.U);
  branch.cost = trajectoryFeasible(branch.X)
                    ? trajectoryCost(branch.X, backward ? x_init : x_target)
                    : std::numeric_limits<double>::infinity();
  return branch;
}

inline std::vector<RepresentativeBranch> WmrobotAblationSolver::representatives(
    const std::vector<RolloutSample> &samples,
    const std::vector<std::vector<int>> &clusters, int T, bool backward) const {
  std::vector<RepresentativeBranch> branches;
  for (const auto &cluster : clusters) {
    branches.push_back(weightedBranch(samples, cluster, T, backward));
  }
  return branches;
}

inline Eigen::MatrixXd WmrobotAblationSolver::weightedControls(
    const std::vector<RolloutSample> &samples, int T) const {
  std::vector<int> feasible;
  for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
    if (samples[i].feasible) {
      feasible.push_back(i);
    }
  }
  return weightedBranch(samples, feasible, T, false).U;
}

inline int WmrobotAblationSolver::bestForwardRepresentative(
    const std::vector<RepresentativeBranch> &branches,
    const Eigen::VectorXd &target) const {
  int best_index = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(branches.size()); ++i) {
    if (!std::isfinite(branches[i].cost)) {
      continue;
    }
    const double terminal_cost =
        p(branches[i].X.col(branches[i].X.cols() - 1), target);
    const double total_cost = branches[i].cost + terminal_cost;
    if (total_cost < best_cost) {
      best_cost = total_cost;
      best_index = i;
    }
  }
  return best_index;
}

inline void WmrobotAblationSolver::setResult(const Eigen::MatrixXd &U,
                                             double connection_distance) {
  Uo = U;
  if (Uo.cols() == 0) {
    Uo = Eigen::MatrixXd::Zero(dim_u, 1);
  }
  h(Uo);
  Xo = rolloutForward(x_init, Uo);
  u0 = Uo.col(0);
  last_connection_distance = connection_distance;
}

inline void WmrobotAblationSolver::solveMPPI() {
  const auto start_time = std::chrono::high_resolution_clock::now();
  const auto samples = sampleForward(config.N, config.T, U0, x_init, x_target,
                                     10, solve_count);
  Uo = weightedControls(samples, config.T);
  h(Uo);
  Xo = rolloutForward(x_init, Uo);
  u0 = Uo.col(0);
  last_connection_distance = std::numeric_limits<double>::quiet_NaN();
  const auto finish = std::chrono::high_resolution_clock::now();
  elapsed_rollout = std::chrono::duration<double>(finish - start_time).count();
  elapsed_clustering = 0.0;
  elapsed_connection = 0.0;
  elapsed_guide = 0.0;
  elapsed = elapsed_rollout;
}

inline void WmrobotAblationSolver::solveClusterMPPI() {
  auto start_time = std::chrono::high_resolution_clock::now();
  const auto samples = sampleForward(config.N, config.T, U0, x_init, x_target,
                                     20, solve_count);
  auto after_rollout = std::chrono::high_resolution_clock::now();
  auto clusters = dbscan(samples, config.minpts, config.epsilon);
  auto branches = representatives(samples, clusters, config.T, false);
  int best = bestForwardRepresentative(branches, x_target);
  setResult(branches.empty() ? weightedControls(samples, config.T)
                             : branches[best].U,
            std::numeric_limits<double>::quiet_NaN());
  auto finish = std::chrono::high_resolution_clock::now();
  elapsed_rollout =
      std::chrono::duration<double>(after_rollout - start_time).count();
  elapsed_clustering =
      std::chrono::duration<double>(finish - after_rollout).count();
  elapsed_connection = 0.0;
  elapsed_guide = 0.0;
  elapsed = elapsed_rollout + elapsed_clustering;
}

inline void WmrobotAblationSolver::guide(const Eigen::MatrixXd &reference_u) {
  const int Tr = reference_u.cols();
  const auto samples =
      sampleForward(config.Nr, Tr, reference_u, x_init, x_target, 40, solve_count);
  Uo = weightedControls(samples, Tr);
  h(Uo);
  Xo = rolloutForward(x_init, Uo);
  u0 = Uo.col(0);
}

inline void WmrobotAblationSolver::solveBiC(bool use_guide, bool use_backward,
                                            bool use_clustering) {
  const auto rollout_start = std::chrono::high_resolution_clock::now();
  const auto forward_samples = sampleForward(config.Nf, config.Tf, Uf0, x_init,
                                             x_target, 30, solve_count);
  std::vector<RolloutSample> backward_samples;
  if (use_backward) {
    backward_samples = sampleBackward(config.Nb, config.Tb, Ub0, x_target,
                                      x_init, 31, solve_count);
  }
  const auto rollout_finish = std::chrono::high_resolution_clock::now();

  const auto cluster_start = std::chrono::high_resolution_clock::now();
  std::vector<RepresentativeBranch> forward_branches;
  std::vector<RepresentativeBranch> backward_branches;
  if (use_clustering) {
    forward_branches =
        representatives(forward_samples,
                        dbscan(forward_samples, config.minpts, config.epsilon),
                        config.Tf, false);
    if (use_backward) {
      backward_branches = representatives(
          backward_samples,
          dbscan(backward_samples, config.minpts, config.epsilon), config.Tb,
          true);
    }
  } else {
    for (const auto &sample : forward_samples) {
      if (sample.feasible) {
        forward_branches.push_back({sample.U, sample.X, sample.cost});
      }
    }
    if (use_backward) {
      for (const auto &sample : backward_samples) {
        if (sample.feasible) {
          backward_branches.push_back({sample.U, sample.X, sample.cost});
        }
      }
    }
  }
  const auto cluster_finish = std::chrono::high_resolution_clock::now();

  const auto connection_start = std::chrono::high_resolution_clock::now();
  Eigen::MatrixXd selected_u = Eigen::MatrixXd::Zero(dim_u, config.Tf);
  double selected_connection = std::numeric_limits<double>::infinity();

  if (forward_branches.empty()) {
    selected_u = weightedControls(forward_samples, config.Tf);
  } else if (!use_backward || backward_branches.empty()) {
    const int best = bestForwardRepresentative(forward_branches, x_target);
    selected_u = forward_branches[best].U;
    selected_connection =
        p(forward_branches[best].X.col(forward_branches[best].X.cols() - 1),
          x_target);
  } else {
    int best_f = 0;
    int best_b = 0;
    int best_tf = 0;
    int best_tb = config.Tb;
    for (int fi = 0; fi < static_cast<int>(forward_branches.size()); ++fi) {
      for (int bi = 0; bi < static_cast<int>(backward_branches.size()); ++bi) {
        for (int tf = 0; tf < forward_branches[fi].X.cols(); ++tf) {
          for (int tb = 0; tb < backward_branches[bi].X.cols(); ++tb) {
            const double distance =
                (forward_branches[fi].X.col(tf) -
                 backward_branches[bi].X.col(tb))
                    .norm();
            if (distance < selected_connection) {
              selected_connection = distance;
              best_f = fi;
              best_b = bi;
              best_tf = tf;
              best_tb = tb;
            }
          }
        }
      }
    }

    const int total_u_cols =
        std::max(config.Tf, best_tf + (config.Tb - best_tb));
    selected_u = Eigen::MatrixXd::Zero(dim_u, total_u_cols);
    if (best_tf > 0) {
      selected_u.leftCols(best_tf) = forward_branches[best_f].U.leftCols(best_tf);
    }
    if (best_tb < config.Tb) {
      selected_u.middleCols(best_tf, config.Tb - best_tb) =
          backward_branches[best_b].U.middleCols(best_tb, config.Tb - best_tb);
    }
  }

  if (!std::isfinite(selected_connection)) {
    selected_connection = std::numeric_limits<double>::quiet_NaN();
  }
  setResult(selected_u, selected_connection);
  const auto connection_finish = std::chrono::high_resolution_clock::now();

  const auto guide_start = std::chrono::high_resolution_clock::now();
  if (use_guide) {
    guide(selected_u);
    last_connection_distance = selected_connection;
  }
  const auto guide_finish = std::chrono::high_resolution_clock::now();

  elapsed_rollout =
      std::chrono::duration<double>(rollout_finish - rollout_start).count();
  elapsed_clustering =
      std::chrono::duration<double>(cluster_finish - cluster_start).count();
  elapsed_connection =
      std::chrono::duration<double>(connection_finish - connection_start).count();
  elapsed_guide = std::chrono::duration<double>(guide_finish - guide_start).count();
  elapsed = elapsed_rollout + elapsed_clustering + elapsed_connection +
            elapsed_guide;
}

inline void WmrobotAblationSolver::solve() {
  ++solve_count;
  switch (variant) {
  case AblationVariant::MPPI:
    solveMPPI();
    break;
  case AblationVariant::ClusterMPPI:
    solveClusterMPPI();
    break;
  case AblationVariant::BiCNoGuide:
    solveBiC(false, true, true);
    break;
  case AblationVariant::BiCNoBackward:
    solveBiC(false, false, true);
    break;
  case AblationVariant::BiCNoClustering:
    solveBiC(true, true, false);
    break;
  case AblationVariant::FullBiC:
    solveBiC(true, true, true);
    break;
  }
}

inline void WmrobotAblationSolver::move() {
  x_init = x_init + dt * f(x_init, u0);
  if (U0.cols() > 1 && Uo.cols() >= config.T) {
    U0.leftCols(config.T - 1) = Uo.rightCols(config.T - 1);
  }
  if (Uf0.cols() > 1 && Uo.cols() >= config.Tf) {
    Uf0.leftCols(config.Tf - 1) = Uo.rightCols(config.Tf - 1);
  }
  Ub0.setZero();
}

inline void writeRunHeader(std::ofstream &csv) {
  csv << "variant,start_case,map,is_success,is_collision,iter,elapsed,"
         "elapsed_rollout,elapsed_clustering,elapsed_connection,elapsed_guide,"
         "d_goal,d_conn\n";
}

inline void writeRunRow(std::ofstream &csv, const AblationRunResult &row) {
  csv << row.variant << ',' << row.start_case << ',' << row.map_id << ','
      << row.is_success << ',' << row.is_collision << ',' << row.iter << ','
      << csvDouble(row.elapsed) << ',' << csvDouble(row.elapsed_rollout) << ','
      << csvDouble(row.elapsed_clustering) << ','
      << csvDouble(row.elapsed_connection) << ','
      << csvDouble(row.elapsed_guide) << ',' << csvDouble(row.d_goal) << ','
      << csvDouble(row.d_conn) << '\n';
}

inline void writeSummaryHeader(std::ofstream &csv) {
  csv << "Variant,Success Rate,Nfail,Nbar_iter,Tbar_elapsed,dbar_goal,dbar_conn,"
         "num_runs,num_success\n";
}

inline void writeSummaryRow(std::ofstream &csv, const AblationSummary &summary) {
  csv << summary.variant << ',' << csvDouble(summary.success_rate) << ','
      << summary.nfail << ',' << csvDouble(summary.nbar_iter) << ','
      << csvDouble(summary.tbar_elapsed) << ',' << csvDouble(summary.dbar_goal)
      << ',' << csvDouble(summary.dbar_conn) << ',' << summary.num_runs << ','
      << summary.num_success << '\n';
}

inline AblationSummary summarizeRuns(const std::string &variant,
                                     const std::vector<AblationRunResult> &runs) {
  AblationSummary summary;
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
    if (isFiniteMetric(run.d_conn)) {
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

inline std::vector<int> mapIds(const AblationConfig &config) {
  std::vector<int> ids;
  for (int i = 0; i < config.num_maps; ++i) {
    ids.push_back(config.map_begin - i);
  }
  return ids;
}

inline std::vector<AblationRunResult>
runAblationVariant(AblationVariant variant, const AblationConfig &config) {
  std::vector<AblationRunResult> rows;
  WMRobotMap model;
  const Eigen::VectorXd target = wmrobotTargetState();

  for (int start_case = 0; start_case < config.start_cases; ++start_case) {
    for (int map_id : mapIds(config)) {
      CollisionChecker collision_checker;
      collision_checker.loadMap(config.dataset_dir + "/output_" +
                                    std::to_string(map_id) + ".txt",
                                0.1);

      WmrobotAblationSolver solver(model, &collision_checker, config, variant);
      solver.reset(wmrobotStartState(start_case), target);

      AblationRunResult row;
      row.variant = variantLabel(variant);
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
        row.d_conn = solver.last_connection_distance;

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
      if (!row.is_success && row.d_goal == 0.0) {
        row.d_goal = (solver.x_init - target).norm();
      }
      rows.push_back(row);

      std::cout << row.variant << '\t' << row.start_case << '\t' << row.map_id
                << '\t' << row.is_success << '\t' << row.iter << '\t'
                << row.elapsed << '\t' << row.d_goal << '\t'
                << csvDouble(row.d_conn) << std::endl;
    }
  }

  return rows;
}

inline AblationSummary runAndWriteAblationVariant(AblationVariant variant,
                                                  AblationConfig config) {
  const std::string token = variantFileToken(variant);
  const std::string runs_path = config.output_prefix + "_" + token + "_runs.csv";
  const std::string summary_path =
      config.output_prefix + "_" + token + "_summary.csv";

  auto rows = runAblationVariant(variant, config);
  auto summary = summarizeRuns(variantLabel(variant), rows);

  std::ofstream runs_csv(runs_path);
  writeRunHeader(runs_csv);
  for (const auto &row : rows) {
    writeRunRow(runs_csv, row);
  }

  std::ofstream summary_csv(summary_path);
  writeSummaryHeader(summary_csv);
  writeSummaryRow(summary_csv, summary);

  return summary;
}
