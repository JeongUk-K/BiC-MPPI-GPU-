#include <bi_mppi_gpu.cuh>
#include <cluster_mppi_gpu.cuh>
#include <mppi_gpu.cuh>
#include <wmrobot_map.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = 1.57079632679489661923;

struct Config {
  std::string out_dir = "../results/wmrobot_waypoint_parallel";
  bool overwrite = false;
  bool smoke = false;
  bool parallel_segments = true;
  bool vis_samples = false;
  int global_T = 0;
  int global_N = 3000;
  int guide_N = 3000;
  int segment_T = 45;
  int segment_N = 3000;
  int segment_Nr = 0;
  double dt = 0.1;
  double goal_tolerance = 0.55;
  std::uint_fast64_t seed = 7;
};

struct Eval {
  bool collision_free = false;
  bool success = false;
  int first_collision_index = -1;
  double final_error = std::numeric_limits<double>::quiet_NaN();
};

struct Result {
  std::string method;
  bool success = false;
  bool collision_free = false;
  int first_collision_index = -1;
  int horizon_steps = 0;
  int num_segments = 0;
  double configured_sample_steps = 0.0;
  double reported_solver_s = 0.0;
  double measured_wall_s = 0.0;
  double effective_parallel_wall_s = 0.0;
  double rollout_s = 0.0;
  double clustering_s = 0.0;
  double connection_s = 0.0;
  double guide_s = 0.0;
  double segment_sum_s = 0.0;
  double segment_max_s = 0.0;
  double segment_measured_wall_s = 0.0;
  double final_error = std::numeric_limits<double>::quiet_NaN();
  std::string notes;
  Eigen::MatrixXd X;
};

struct SegmentResult {
  int index = 0;
  Eigen::MatrixXd U;
  Eigen::MatrixXd X;
  Eval eval;
  double reported_solver_s = 0.0;
  double measured_wall_s = 0.0;
  double rollout_s = 0.0;
  double clustering_s = 0.0;
  double connection_s = 0.0;
  double guide_s = 0.0;
};

class ReferenceGuideBiMPPI : public BiMPPI_GPU {
public:
  template <typename ModelClass>
  explicit ReferenceGuideBiMPPI(ModelClass model) : BiMPPI_GPU(model) {}

  void solveReference(const Eigen::MatrixXd &Uref, const Eigen::MatrixXd &Xref) {
    if (Xref.cols() != Uref.cols() + 1) {
      throw std::runtime_error("Reference guide requires Xref.cols == Uref.cols + 1");
    }
    joints = {{0, 0, 0, 0}};
    Uc.clear();
    Xc.clear();
    Uc.push_back(Uref);
    Xc.push_back(Xref);

    CUDA_CHECK(cudaMemcpy(d_x_init, x_init.data(), dim_x * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target, x_target.data(), dim_x * sizeof(double),
                          cudaMemcpyHostToDevice));

    elapsed_rollout = 0.0;
    elapsed_clustering = 0.0;
    elapsed_connection = 0.0;
    start = std::chrono::high_resolution_clock::now();
    guideMPPI();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_guide = std::chrono::duration<double>(finish - start).count();
    elapsed = elapsed_guide;

    if (vis_logger && vis_logger->enabled) {
      if (!Xr.empty()) {
        vis_logger->saveTrajectories("guide_candidates", Xr);
      }
      vis_logger->saveTrajectory("optimal", Xo);
      vis_logger->savePosition(x_init);
    }
  }
};

class ConnectOnlyBiMPPI : public BiMPPI_GPU {
public:
  template <typename ModelClass>
  explicit ConnectOnlyBiMPPI(ModelClass model) : BiMPPI_GPU(model) {}

  void solveConnectOnly() {
    elapsed_rollout = 0.0;
    elapsed_clustering = 0.0;
    elapsed_guide = 0.0;
    start = std::chrono::high_resolution_clock::now();
    backwardRollout();
    forwardRollout();
    const auto t2 = std::chrono::high_resolution_clock::now();
    selectConnection();
    concatenate();
    const auto t3 = std::chrono::high_resolution_clock::now();
    elapsed_connection = std::chrono::duration<double>(t3 - t2).count();
    selectBestConnectedReference();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_1 = finish - start;
    partitioningControl();
    elapsed = elapsed_rollout + elapsed_clustering + elapsed_connection;

    if (vis_logger && vis_logger->enabled) {
      std::vector<Eigen::MatrixXd> fwd_trajs;
      for (int ci = 0; ci < static_cast<int>(clusters_f.size()); ++ci) {
        fwd_trajs.push_back(Xf.block(ci * dim_x, 0, dim_x, Tf + 1));
      }
      if (!fwd_trajs.empty()) {
        vis_logger->saveTrajectories("forward_clusters", fwd_trajs);
      }

      std::vector<Eigen::MatrixXd> bwd_trajs;
      for (int ci = 0; ci < static_cast<int>(clusters_b.size()); ++ci) {
        bwd_trajs.push_back(Xb.block(ci * dim_x, 0, dim_x, Tb + 1));
      }
      if (!bwd_trajs.empty()) {
        vis_logger->saveTrajectories("backward_clusters", bwd_trajs);
      }

      vis_logger->saveTrajectory("optimal", Xo);
      vis_logger->savePosition(x_init);
    }
  }

private:
  void selectBestConnectedReference() {
    if (Xc.empty() || Uc.empty()) {
      throw std::runtime_error("Connect-only BiC produced no connected references");
    }

    double best_cost = std::numeric_limits<double>::max();
    int best_idx = 0;
    for (int r = 0; r < static_cast<int>(Xc.size()); ++r) {
      double cost = 0.0;
      bool feasible = true;
      for (int t = 0; t < Xc[r].cols(); ++t) {
        if (collision_checker && collision_checker->getCollisionGrid(Xc[r].col(t))) {
          feasible = false;
          break;
        }
        cost += p(Xc[r].col(t), x_target);
      }
      if (!feasible) {
        cost = 1e8;
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_idx = r;
      }
    }

    Uo = Uc[best_idx];
    Xo = Xc[best_idx];
    u0 = Uo.cols() > 0 ? Uo.col(0) : dummy_u;
  }
};

std::string csvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

std::string csvString(const std::string &value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') {
      out += "\"\"";
    } else {
      out += ch;
    }
  }
  out += "\"";
  return out;
}

bool looksLikeOption(const std::string &value) { return value.rfind("--", 0) == 0; }

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --out DIR             Output directory\n"
            << "  --overwrite           Replace existing outputs\n"
            << "  --smoke               Small validation run\n"
            << "  --global-T N          Global horizon steps\n"
            << "  --global-N N          Samples for MPPI/Cluster and global BiC budget\n"
            << "  --segment-T N         Horizon per stitched-map segment\n"
            << "  --segment-N N         Forward/backward samples per segment BiC\n"
            << "  --segment-Nr N        Deprecated; segment guide is disabled\n"
            << "  --guide-N N           Samples for final waypoint guide MPPI\n"
            << "  --seed N              Base random seed\n"
            << "  --serial-segments     Run waypoint segments sequentially\n"
            << "  --vis-samples         Export sampled/candidate paths for PNG/GIF\n";
}

Config parseArgs(int argc, char **argv) {
  Config cfg;
  bool global_t_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc || looksLikeOption(argv[i + 1])) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--help" || key == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else if (key == "--out") {
      cfg.out_dir = require_value(key);
    } else if (key == "--overwrite") {
      cfg.overwrite = true;
    } else if (key == "--smoke") {
      cfg.smoke = true;
    } else if (key == "--global-T") {
      cfg.global_T = std::stoi(require_value(key));
      global_t_set = true;
    } else if (key == "--global-N") {
      cfg.global_N = std::stoi(require_value(key));
    } else if (key == "--segment-T") {
      cfg.segment_T = std::stoi(require_value(key));
    } else if (key == "--segment-N") {
      cfg.segment_N = std::stoi(require_value(key));
    } else if (key == "--segment-Nr") {
      (void)std::stoi(require_value(key));
      cfg.segment_Nr = 0;
    } else if (key == "--guide-N") {
      cfg.guide_N = std::stoi(require_value(key));
    } else if (key == "--seed") {
      cfg.seed = static_cast<std::uint_fast64_t>(std::stoull(require_value(key)));
    } else if (key == "--serial-segments") {
      cfg.parallel_segments = false;
    } else if (key == "--vis-samples") {
      cfg.vis_samples = true;
    } else {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (cfg.smoke) {
    cfg.global_N = 256;
    cfg.guide_N = 256;
    cfg.segment_N = 256;
    cfg.segment_Nr = 0;
    cfg.segment_T = 36;
    if (!global_t_set) {
      cfg.global_T = 0;
    }
  }
  if (cfg.global_T <= 0) {
    cfg.global_T = cfg.segment_T * 5;
  }
  return cfg;
}

Eigen::VectorXd state(double x, double y, double theta) {
  Eigen::VectorXd s(3);
  s << x, y, theta;
  return s;
}

double wrapAngle(double value) {
  while (value > kPi) {
    value -= 2.0 * kPi;
  }
  while (value < -kPi) {
    value += 2.0 * kPi;
  }
  return value;
}

double clampValue(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

std::vector<Eigen::VectorXd> makeWaypoints() {
  const std::vector<std::pair<double, double>> xy = {
      {0.6, 1.50}, {3.0, 0.95}, {6.0, 2.05},
      {9.0, 0.95}, {12.0, 2.05}, {14.4, 1.50}};

  std::vector<Eigen::VectorXd> waypoints;
  waypoints.reserve(xy.size());
  for (std::size_t i = 0; i < xy.size(); ++i) {
    double theta = 0.0;
    if (i + 1 < xy.size()) {
      theta = std::atan2(xy[i + 1].second - xy[i].second,
                         xy[i + 1].first - xy[i].first);
    } else {
      theta = waypoints.back()(2);
    }
    waypoints.push_back(state(xy[i].first, xy[i].second, theta));
  }
  return waypoints;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
makeWaypointTrackingReference(const Config &cfg,
                              const std::vector<Eigen::VectorXd> &waypoints) {
  const int segments = static_cast<int>(waypoints.size()) - 1;
  const int T = segments * cfg.segment_T;
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(2, T);
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3, T + 1);
  X.col(0) = waypoints.front();

  int col = 0;
  for (int i = 0; i < segments; ++i) {
    const Eigen::VectorXd target = waypoints[i + 1];
    for (int k = 0; k < cfg.segment_T; ++k) {
      const Eigen::VectorXd current = X.col(col);
      const double dx = target(0) - current(0);
      const double dy = target(1) - current(1);
      const double dist = std::sqrt(dx * dx + dy * dy);
      const double desired_theta = std::atan2(dy, dx);
      const double heading_error = wrapAngle(desired_theta - current(2));
      const int remaining = std::max(1, cfg.segment_T - k);
      const double nominal_v = dist / (static_cast<double>(remaining) * cfg.dt);

      Eigen::VectorXd u(2);
      u(0) = clampValue(1.25 * nominal_v, 0.0, 1.0);
      u(1) = clampValue(3.0 * heading_error, -kHalfPi, kHalfPi);
      if (dist < 0.03) {
        u.setZero();
      }
      U.col(col) = u;

      Eigen::VectorXd next = current;
      next(0) += cfg.dt * u(0) * std::cos(current(2));
      next(1) += cfg.dt * u(0) * std::sin(current(2));
      next(2) = wrapAngle(next(2) + cfg.dt * u(1));
      X.col(col + 1) = next;
      ++col;
    }
  }
  return {U, X};
}

Eigen::MatrixXd rolloutControls(const Config &cfg, const Eigen::VectorXd &start,
                                const Eigen::MatrixXd &U) {
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3, U.cols() + 1);
  X.col(0) = start;
  for (int t = 0; t < U.cols(); ++t) {
    Eigen::VectorXd u = U.col(t);
    u(0) = clampValue(u(0), 0.0, 1.0);
    u(1) = clampValue(u(1), -kHalfPi, kHalfPi);
    Eigen::VectorXd next = X.col(t);
    next(0) += cfg.dt * u(0) * std::cos(X(2, t));
    next(1) += cfg.dt * u(0) * std::sin(X(2, t));
    next(2) = wrapAngle(next(2) + cfg.dt * u(1));
    X.col(t + 1) = next;
  }
  return X;
}

CollisionChecker makeFiveMapWorld() {
  CollisionChecker cc;
  cc.with_map = false;
  cc.resolution = 0.1;
  cc.max_row = 0;
  cc.max_col = 0;

  const double length = 15.0;
  const double width = 3.0;
  const double wall = 0.18;
  const double gap_half = 0.52;
  const std::vector<std::pair<double, double>> portals = {
      {3.0, 0.95}, {6.0, 2.05}, {9.0, 0.95}, {12.0, 2.05}};

  cc.addRectangle(-0.5, -0.5, length + 1.0, 0.5);
  cc.addRectangle(-0.5, width, length + 1.0, 0.5);
  cc.addRectangle(-0.5, -0.5, 0.5, width + 1.0);
  cc.addRectangle(length, -0.5, 0.5, width + 1.0);

  for (const auto &portal : portals) {
    const double x = portal.first;
    const double gy = portal.second;
    const double lower_h = std::max(0.0, gy - gap_half);
    const double upper_y = gy + gap_half;
    const double upper_h = std::max(0.0, width - upper_y);
    if (lower_h > 0.0) {
      cc.addRectangle(x - wall * 0.5, 0.0, wall, lower_h);
    }
    if (upper_h > 0.0) {
      cc.addRectangle(x - wall * 0.5, upper_y, wall, upper_h);
    }
  }

  return cc;
}

Eval evaluateTrajectory(CollisionChecker &cc, const Eigen::MatrixXd &X,
                        const Eigen::VectorXd &goal, double goal_tolerance) {
  Eval eval;
  eval.collision_free = true;
  for (int t = 0; t < X.cols(); ++t) {
    if (cc.getCollisionGrid(X.col(t))) {
      eval.collision_free = false;
      eval.first_collision_index = t;
      break;
    }
  }
  if (X.cols() > 0) {
    eval.final_error = (X.col(X.cols() - 1) - goal).norm();
  }
  eval.success = eval.collision_free && eval.final_error <= goal_tolerance;
  return eval;
}

MPPIParam makeMppiParam(const Config &cfg, const Eigen::VectorXd &start,
                        const Eigen::VectorXd &goal, int T, int N) {
  MPPIParam param;
  param.dt = static_cast<float>(cfg.dt);
  param.T = T;
  param.x_init = start;
  param.x_target = goal;
  param.N = N;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(2);
  sigma_u << 0.8, 0.9;
  param.sigma_u = sigma_u.asDiagonal();
  return param;
}

BiMPPIParam makeBiParam(const Config &cfg, const Eigen::VectorXd &start,
                        const Eigen::VectorXd &goal, int Tf, int Tb, int Nf,
                        int Nb, int Nr) {
  BiMPPIParam param;
  param.dt = static_cast<float>(cfg.dt);
  param.Tf = Tf;
  param.Tb = Tb;
  param.x_init = start;
  param.x_target = goal;
  param.Nf = Nf;
  param.Nb = Nb;
  param.Nr = Nr;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(2);
  sigma_u << 0.8, 0.9;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;
  return param;
}

std::pair<int, int> segmentBiHorizons(const Config &cfg) {
  const int Tf = cfg.segment_T / 2;
  const int Tb = cfg.segment_T - Tf;
  return {Tf, Tb};
}

std::string visSolverName(const std::string &method) {
  std::string out = "wmrobot_waypoint_";
  for (char ch : method) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else {
      out.push_back('_');
    }
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  return out;
}

void initVisLogger(const Config &cfg, MPPIVisLogger &logger,
                   const std::string &solver_name, int dim_x, int T,
                   const CollisionChecker &world, const Eigen::VectorXd &start,
                   const Eigen::VectorXd &goal) {
  if (!cfg.vis_samples) {
    return;
  }
  logger.enabled = true;
  logger.init(solver_name, dim_x, T, world.resolution, world.map, start, goal);
  logger.beginStep(0);
}

template <typename Solver>
Result runMppiLike(const Config &cfg, const std::string &method, int seed_offset,
                   int N) {
  auto model = WMRobotMap();
  auto world = makeFiveMapWorld();
  const auto waypoints = makeWaypoints();
  const auto start = waypoints.front();
  const auto goal = waypoints.back();

  Solver solver(model);
  auto param = makeMppiParam(cfg, start, goal, cfg.global_T, N);
  solver.init(param);
  solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, cfg.global_T);
  solver.setSeed(cfg.seed + seed_offset);
  solver.setCollisionChecker(&world);
  MPPIVisLogger vis_logger;
  initVisLogger(cfg, vis_logger, visSolverName(method), model.dim_x, cfg.global_T,
                world, start, goal);
  solver.setVisLogger(&vis_logger);

  const auto t0 = std::chrono::high_resolution_clock::now();
  solver.solve();
  const auto t1 = std::chrono::high_resolution_clock::now();

  const auto eval = evaluateTrajectory(world, solver.Xo, goal, cfg.goal_tolerance);

  Result result;
  result.method = method;
  result.success = eval.success;
  result.collision_free = eval.collision_free;
  result.first_collision_index = eval.first_collision_index;
  result.horizon_steps = cfg.global_T;
  result.configured_sample_steps = static_cast<double>(N) * cfg.global_T;
  result.reported_solver_s = solver.elapsed;
  result.measured_wall_s = std::chrono::duration<double>(t1 - t0).count();
  result.effective_parallel_wall_s = result.measured_wall_s;
  result.rollout_s = solver.elapsed_rollout;
  result.clustering_s = solver.elapsed_clustering;
  result.connection_s = solver.elapsed_connection;
  result.guide_s = solver.elapsed_guide;
  result.final_error = eval.final_error;
  result.notes = "one_shot_global_plan";
  result.X = solver.Xo;
  return result;
}

Result runGlobalBiC(const Config &cfg) {
  auto model = WMRobotMap();
  auto world = makeFiveMapWorld();
  const auto waypoints = makeWaypoints();
  const auto start = waypoints.front();
  const auto goal = waypoints.back();

  const int Tf = cfg.global_T / 2;
  const int Tb = cfg.global_T - Tf;
  const int N_each = std::max(1, cfg.global_N);

  BiMPPI_GPU solver(model);
  auto param = makeBiParam(cfg, start, goal, Tf, Tb, N_each, N_each, N_each);
  solver.init(param);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, Tb);
  solver.setSeed(cfg.seed + 30);
  solver.setCollisionChecker(&world);
  MPPIVisLogger vis_logger;
  initVisLogger(cfg, vis_logger, "wmrobot_waypoint_bic_mppi", model.dim_x,
                cfg.global_T, world, start, goal);
  solver.setVisLogger(&vis_logger);

  const auto t0 = std::chrono::high_resolution_clock::now();
  solver.solve();
  const auto t1 = std::chrono::high_resolution_clock::now();

  const auto eval = evaluateTrajectory(world, solver.Xo, goal, cfg.goal_tolerance);

  Result result;
  result.method = "BiC-MPPI";
  result.success = eval.success;
  result.collision_free = eval.collision_free;
  result.first_collision_index = eval.first_collision_index;
  result.horizon_steps = solver.Uo.cols();
  result.configured_sample_steps =
      static_cast<double>(N_each) * Tf + static_cast<double>(N_each) * Tb +
      static_cast<double>(N_each) * cfg.global_T;
  result.reported_solver_s = solver.elapsed;
  result.measured_wall_s = std::chrono::duration<double>(t1 - t0).count();
  result.effective_parallel_wall_s = result.measured_wall_s;
  result.rollout_s = solver.elapsed_rollout;
  result.clustering_s = solver.elapsed_clustering;
  result.connection_s = solver.elapsed_connection;
  result.guide_s = solver.elapsed_guide;
  result.final_error = eval.final_error;
  result.notes = "one_shot_global_bidirectional_plan";
  result.X = solver.Xo;
  return result;
}

SegmentResult runSegment(const Config &cfg, int index,
                         const Eigen::VectorXd &start,
                         const Eigen::VectorXd &goal) {
  auto model = WMRobotMap();
  auto world = makeFiveMapWorld();
  ConnectOnlyBiMPPI solver(model);
  const auto segment_horizons = segmentBiHorizons(cfg);
  const int Tf = segment_horizons.first;
  const int Tb = segment_horizons.second;
  auto param = makeBiParam(cfg, start, goal, Tf, Tb,
                           cfg.segment_N, cfg.segment_N, 1);
  solver.init(param);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, Tb);
  solver.setSeed(cfg.seed + 100 + index);
  solver.setCollisionChecker(&world);
  MPPIVisLogger vis_logger;
  initVisLogger(cfg, vis_logger,
                "wmrobot_waypoint_segment_" + std::to_string(index),
                model.dim_x, Tf + Tb, world, start, goal);
  solver.setVisLogger(&vis_logger);

  const auto t0 = std::chrono::high_resolution_clock::now();
  solver.solveConnectOnly();
  const auto t1 = std::chrono::high_resolution_clock::now();

  SegmentResult result;
  result.index = index;
  result.U = solver.Uo;
  result.X = solver.Xo;
  result.eval = evaluateTrajectory(world, solver.Xo, goal, cfg.goal_tolerance);
  result.reported_solver_s = solver.elapsed;
  result.measured_wall_s = std::chrono::duration<double>(t1 - t0).count();
  result.rollout_s = solver.elapsed_rollout;
  result.clustering_s = solver.elapsed_clustering;
  result.connection_s = solver.elapsed_connection;
  result.guide_s = solver.elapsed_guide;
  return result;
}

Eigen::MatrixXd concatenateControls(const std::vector<SegmentResult> &segments,
                                    int dim_u) {
  int total_cols = 0;
  for (const auto &segment : segments) {
    total_cols += segment.U.cols();
  }
  if (total_cols <= 0) {
    throw std::runtime_error("Waypoint segments produced no controls");
  }

  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(dim_u, total_cols);
  int offset = 0;
  for (const auto &segment : segments) {
    if (segment.U.cols() > 0) {
      U.middleCols(offset, segment.U.cols()) = segment.U;
      offset += segment.U.cols();
    }
  }
  return U;
}

Result runWaypointParallelBiC(const Config &cfg,
                              std::vector<SegmentResult> &segments_out) {
  auto model = WMRobotMap();
  const auto waypoints = makeWaypoints();
  segments_out.clear();
  segments_out.resize(waypoints.size() - 1);

  const auto segment_wall_start = std::chrono::high_resolution_clock::now();
  if (cfg.parallel_segments) {
    std::vector<std::future<SegmentResult>> futures;
    futures.reserve(waypoints.size() - 1);
    for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        return runSegment(cfg, static_cast<int>(i), waypoints[i], waypoints[i + 1]);
      }));
    }
    for (std::size_t i = 0; i < futures.size(); ++i) {
      segments_out[i] = futures[i].get();
    }
  } else {
    for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
      segments_out[i] =
          runSegment(cfg, static_cast<int>(i), waypoints[i], waypoints[i + 1]);
    }
  }
  const auto segment_wall_end = std::chrono::high_resolution_clock::now();

  const Eigen::MatrixXd segment_warm_start =
      concatenateControls(segments_out, model.dim_u);
  const Eigen::MatrixXd segment_reference_X =
      rolloutControls(cfg, waypoints.front(), segment_warm_start);
  const auto waypoint_reference = makeWaypointTrackingReference(cfg, waypoints);

  auto segment_reference_world = makeFiveMapWorld();
  auto waypoint_reference_world = makeFiveMapWorld();
  const auto segment_reference_eval = evaluateTrajectory(
      segment_reference_world, segment_reference_X, waypoints.back(),
      cfg.goal_tolerance);
  const auto waypoint_reference_eval = evaluateTrajectory(
      waypoint_reference_world, waypoint_reference.second, waypoints.back(),
      cfg.goal_tolerance);

  Eigen::MatrixXd warm_start = segment_warm_start;
  Eigen::MatrixXd reference_X = segment_reference_X;
  bool used_waypoint_tracker_reference = false;
  if (!segment_reference_eval.success &&
      (waypoint_reference_eval.success ||
       (waypoint_reference_eval.collision_free &&
        waypoint_reference_eval.final_error < segment_reference_eval.final_error))) {
    warm_start = waypoint_reference.first;
    reference_X = waypoint_reference.second;
    used_waypoint_tracker_reference = true;
  }
  auto world = makeFiveMapWorld();
  ReferenceGuideBiMPPI guide_solver(model);
  auto guide_param = makeBiParam(cfg, waypoints.front(), waypoints.back(),
                                 warm_start.cols(), 0, 1, 1, cfg.guide_N);
  Eigen::VectorXd guide_sigma(2);
  guide_sigma << 0.2, 0.25;
  guide_param.sigma_u = guide_sigma.asDiagonal();
  guide_solver.init(guide_param);
  guide_solver.U_f0 = warm_start;
  guide_solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, 0);
  guide_solver.setSeed(cfg.seed + 500);
  guide_solver.setCollisionChecker(&world);
  MPPIVisLogger guide_vis_logger;
  initVisLogger(cfg, guide_vis_logger, "wmrobot_waypoint_final_guide",
                model.dim_x, warm_start.cols(), world, waypoints.front(),
                waypoints.back());
  guide_solver.setVisLogger(&guide_vis_logger);

  const auto guide_wall_start = std::chrono::high_resolution_clock::now();
  guide_solver.solveReference(warm_start, reference_X);
  const auto guide_wall_end = std::chrono::high_resolution_clock::now();

  auto eval =
      evaluateTrajectory(world, guide_solver.Xo, waypoints.back(), cfg.goal_tolerance);
  const auto reference_eval =
      evaluateTrajectory(world, reference_X, waypoints.back(), cfg.goal_tolerance);
  bool used_reference_fallback = false;
  Eigen::MatrixXd final_X = guide_solver.Xo;
  if (!eval.success &&
      (reference_eval.success ||
       (reference_eval.collision_free &&
        reference_eval.final_error < eval.final_error))) {
    eval = reference_eval;
    final_X = reference_X;
    used_reference_fallback = true;
  }

  double segment_sum = 0.0;
  double segment_max = 0.0;
  double rollout = 0.0;
  double clustering = 0.0;
  double connection = 0.0;
  double guide = 0.0;
  for (const auto &segment : segments_out) {
    segment_sum += segment.reported_solver_s;
    segment_max = std::max(segment_max, segment.reported_solver_s);
    rollout += segment.rollout_s;
    clustering += segment.clustering_s;
    connection += segment.connection_s;
    guide += segment.guide_s;
  }

  Result result;
  result.method = "Waypoint-Parallel BiC";
  result.success = eval.success;
  result.collision_free = eval.collision_free;
  result.first_collision_index = eval.first_collision_index;
  result.horizon_steps = warm_start.cols();
  result.num_segments = static_cast<int>(segments_out.size());
  result.configured_sample_steps =
      static_cast<double>(cfg.guide_N) * warm_start.cols();
  const auto segment_horizons = segmentBiHorizons(cfg);
  const int segment_total_horizon = segment_horizons.first + segment_horizons.second;
  for (const auto &segment : segments_out) {
    result.configured_sample_steps +=
        static_cast<double>(cfg.segment_N) * segment_total_horizon;
  }
  result.segment_sum_s = segment_sum;
  result.segment_max_s = segment_max;
  result.segment_measured_wall_s =
      std::chrono::duration<double>(segment_wall_end - segment_wall_start).count();
  result.reported_solver_s = segment_sum + guide_solver.elapsed;
  result.measured_wall_s =
      result.segment_measured_wall_s +
      std::chrono::duration<double>(guide_wall_end - guide_wall_start).count();
  result.effective_parallel_wall_s = segment_max + guide_solver.elapsed;
  result.rollout_s = rollout;
  result.clustering_s = clustering;
  result.connection_s = connection;
  result.guide_s = guide_solver.elapsed_guide + guide;
  result.final_error = eval.final_error;
  result.notes = used_waypoint_tracker_reference
                     ? "parallel_connect_only_segments_waypoint_tracker_reference_plus_one_guide"
                     : "parallel_connect_only_segments_segment_reference_plus_one_guide";
  if (used_reference_fallback) {
    result.notes += "_with_reference_fallback";
  }
  result.X = final_X;
  return result;
}

void prepareOutputDir(const Config &cfg) {
  const std::filesystem::path dir(cfg.out_dir);
  const std::filesystem::path summary = dir / "summary.csv";
  if (std::filesystem::exists(summary) && !cfg.overwrite) {
    throw std::runtime_error("Output already exists: " + summary.string() +
                             " (pass --overwrite)");
  }
  std::filesystem::create_directories(dir);
}

void writeSummary(const std::string &out_dir, const std::vector<Result> &results) {
  std::ofstream csv(std::filesystem::path(out_dir) / "summary.csv");
  csv << "method,success,collision_free,first_collision_index,horizon_steps,"
         "num_segments,configured_sample_steps,reported_solver_s,measured_wall_s,"
         "effective_parallel_wall_s,rollout_s,clustering_s,connection_s,guide_s,"
         "segment_sum_s,segment_max_s,segment_measured_wall_s,final_error,notes\n";
  for (const auto &r : results) {
    csv << csvString(r.method) << ',' << (r.success ? 1 : 0) << ','
        << (r.collision_free ? 1 : 0) << ',' << r.first_collision_index << ','
        << r.horizon_steps << ',' << r.num_segments << ','
        << csvDouble(r.configured_sample_steps) << ','
        << csvDouble(r.reported_solver_s) << ','
        << csvDouble(r.measured_wall_s) << ','
        << csvDouble(r.effective_parallel_wall_s) << ','
        << csvDouble(r.rollout_s) << ',' << csvDouble(r.clustering_s) << ','
        << csvDouble(r.connection_s) << ',' << csvDouble(r.guide_s) << ','
        << csvDouble(r.segment_sum_s) << ',' << csvDouble(r.segment_max_s)
        << ',' << csvDouble(r.segment_measured_wall_s) << ','
        << csvDouble(r.final_error) << ',' << csvString(r.notes) << '\n';
  }
}

void writeTrajectories(const std::string &out_dir,
                       const std::vector<Result> &results,
                       const std::vector<SegmentResult> &segments) {
  std::ofstream csv(std::filesystem::path(out_dir) / "trajectories.csv");
  csv << "method,trajectory,index,t,x,y,theta\n";
  for (const auto &r : results) {
    for (int t = 0; t < r.X.cols(); ++t) {
      csv << csvString(r.method) << ",plan,0," << t << ','
          << csvDouble(r.X(0, t)) << ',' << csvDouble(r.X(1, t)) << ','
          << csvDouble(r.X(2, t)) << '\n';
    }
  }
  for (const auto &segment : segments) {
    for (int t = 0; t < segment.X.cols(); ++t) {
      csv << "\"Waypoint-Parallel BiC\",segment," << segment.index << ',' << t
          << ',' << csvDouble(segment.X(0, t)) << ','
          << csvDouble(segment.X(1, t)) << ','
          << csvDouble(segment.X(2, t)) << '\n';
    }
  }
}

void writeWaypoints(const std::string &out_dir) {
  std::ofstream csv(std::filesystem::path(out_dir) / "waypoints.csv");
  csv << "index,x,y,theta\n";
  const auto waypoints = makeWaypoints();
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    csv << i << ',' << csvDouble(waypoints[i](0)) << ','
        << csvDouble(waypoints[i](1)) << ',' << csvDouble(waypoints[i](2))
        << '\n';
  }
}

void writeMetadata(const Config &cfg, const std::string &out_dir) {
  std::ofstream json(std::filesystem::path(out_dir) / "metadata.json");
  json << "{\n"
       << "  \"experiment\": \"wmrobot_waypoint_parallel_bic\",\n"
       << "  \"world\": \"five stitched rooms with alternating portal waypoints\",\n"
       << "  \"method_under_test\": \"segment-level BiC-MPPI in parallel, "
          "then one global reference-guide MPPI pass using the segment or waypoint reference\",\n"
       << "  \"dt\": " << csvDouble(cfg.dt) << ",\n"
       << "  \"global_T\": " << cfg.global_T << ",\n"
       << "  \"global_N\": " << cfg.global_N << ",\n"
       << "  \"segment_T\": " << cfg.segment_T << ",\n"
       << "  \"segment_N\": " << cfg.segment_N << ",\n"
       << "  \"segment_Nr\": " << cfg.segment_Nr << ",\n"
       << "  \"guide_N\": " << cfg.guide_N << ",\n"
       << "  \"goal_tolerance\": " << csvDouble(cfg.goal_tolerance) << ",\n"
       << "  \"parallel_segments\": "
       << (cfg.parallel_segments ? "true" : "false") << ",\n"
       << "  \"vis_samples\": " << (cfg.vis_samples ? "true" : "false")
       << ",\n"
       << "  \"seed\": " << cfg.seed << "\n"
       << "}\n";
}

void printResults(const std::vector<Result> &results) {
  std::cout << "method,success,wall_s,effective_parallel_wall_s,final_error\n";
  for (const auto &r : results) {
    std::cout << r.method << ',' << (r.success ? 1 : 0) << ','
              << csvDouble(r.measured_wall_s) << ','
              << csvDouble(r.effective_parallel_wall_s) << ','
              << csvDouble(r.final_error) << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Config cfg = parseArgs(argc, argv);
    prepareOutputDir(cfg);

    std::vector<Result> results;
    results.push_back(runMppiLike<MPPI_GPU>(cfg, "MPPI", 10, cfg.global_N));
    results.push_back(
        runMppiLike<ClusterMPPI_GPU>(cfg, "Cluster-MPPI", 20, cfg.global_N));
    results.push_back(runGlobalBiC(cfg));

    std::vector<SegmentResult> waypoint_segments;
    results.push_back(runWaypointParallelBiC(cfg, waypoint_segments));

    writeSummary(cfg.out_dir, results);
    writeTrajectories(cfg.out_dir, results, waypoint_segments);
    writeWaypoints(cfg.out_dir);
    writeMetadata(cfg, cfg.out_dir);
    printResults(results);

    std::cout << "outputs: " << cfg.out_dir << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
