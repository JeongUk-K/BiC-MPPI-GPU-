#pragma once

#include <bi_mppi_gpu.cuh>
#include <cluster_mppi_gpu.cuh>
#include <log_mppi_gpu.cuh>
#include <mppi_gpu.cuh>
#include <mppi_vis_logger.h>
#include <wmrobot_map.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wmrobot_stitched {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = 1.57079632679489661923;

struct Config {
  std::string out_dir = "../results/wmrobot_waypoint_stitched";
  std::string dataset_dir = "../BARN_dataset/txt_files";
  bool overwrite = false;
  bool smoke = false;
  bool save_rollouts = true;
  int scenarios = 10;
  int maps_per_scenario = 5;
  int dataset_maps = 300;
  int maxiter = 500;
  int global_T = 500;
  int waypoint_count = 55;
  int bic_Tf = 50;
  int bic_Tb = 50;
  int N = 3000;
  int Nf = 3000;
  int Nb = 3000;
  int Nr = 1500;
  int vis_every = 10;
  double dt = 0.1;
  double resolution = 0.1;
  double waypoint_tolerance = 0.45;
  double start_x = 1.5;
  double map_length = 5.0;
  std::uint_fast64_t seed = 7;
};

struct Scenario {
  int index = 0;
  std::vector<int> map_ids;
  std::vector<std::vector<double>> map;
  int rows = 0;
  int cols = 0;
};

struct RunStats {
  bool success = false;
  bool is_collision = false;
  int iter = 0;
  int reached_waypoints = 1;
  int total_waypoints = 0;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;
  double final_error = std::numeric_limits<double>::quiet_NaN();
  std::vector<Eigen::VectorXd> path;
};

inline bool looksLikeOption(const std::string &value) {
  return value.rfind("--", 0) == 0;
}

inline void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --out DIR                 Output root directory\n"
            << "  --dataset-dir DIR         BARN txt map directory\n"
            << "  --overwrite               Replace this solver's existing outputs\n"
            << "  --smoke                   Small validation run\n"
            << "  --scenarios N             Number of random stitched scenarios\n"
            << "  --maps-per-scenario N     Maps stitched along the y axis\n"
            << "  --dataset-maps N          Number of BARN maps to sample from\n"
            << "  --maxiter N               Closed-loop iterations per scenario\n"
            << "  --global-T N              Horizon for MPPI/Log/Cluster\n"
            << "  --T N                     Alias for --global-T\n"
            << "  --waypoint-count N        Waypoints for BiC waypoint mode\n"
            << "  --bic-Tf N --bic-Tb N     BiC forward/backward rollout horizon (default 50/50)\n"
            << "  --Tf N --Tb N             Alias for --bic-Tf/--bic-Tb\n"
            << "  --N N                     Samples for MPPI/Log/Cluster\n"
            << "  --Nf N --Nb N --Nr N      Samples for BiC-MPPI\n"
            << "  --seed N                  Random seed for map selection\n"
            << "  --vis-every N             Save rollout data every N iterations\n"
            << "  --no-rollouts             Save CSV paths only\n";
}

inline Config parseArgs(int argc, char **argv, Config cfg = Config()) {
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
    } else if (key == "--dataset-dir") {
      cfg.dataset_dir = require_value(key);
    } else if (key == "--overwrite") {
      cfg.overwrite = true;
    } else if (key == "--smoke") {
      cfg.smoke = true;
    } else if (key == "--scenarios") {
      cfg.scenarios = std::stoi(require_value(key));
    } else if (key == "--maps-per-scenario") {
      cfg.maps_per_scenario = std::stoi(require_value(key));
    } else if (key == "--dataset-maps") {
      cfg.dataset_maps = std::stoi(require_value(key));
    } else if (key == "--maxiter") {
      cfg.maxiter = std::stoi(require_value(key));
    } else if (key == "--global-T" || key == "--T") {
      cfg.global_T = std::stoi(require_value(key));
    } else if (key == "--waypoint-count") {
      cfg.waypoint_count = std::stoi(require_value(key));
    } else if (key == "--bic-Tf" || key == "--Tf") {
      cfg.bic_Tf = std::stoi(require_value(key));
    } else if (key == "--bic-Tb" || key == "--Tb") {
      cfg.bic_Tb = std::stoi(require_value(key));
    } else if (key == "--N") {
      cfg.N = std::stoi(require_value(key));
    } else if (key == "--Nf") {
      cfg.Nf = std::stoi(require_value(key));
    } else if (key == "--Nb") {
      cfg.Nb = std::stoi(require_value(key));
    } else if (key == "--Nr") {
      cfg.Nr = std::stoi(require_value(key));
    } else if (key == "--seed") {
      cfg.seed = static_cast<std::uint_fast64_t>(std::stoull(require_value(key)));
    } else if (key == "--vis-every") {
      cfg.vis_every = std::stoi(require_value(key));
    } else if (key == "--no-rollouts") {
      cfg.save_rollouts = false;
    } else {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (cfg.smoke) {
    cfg.scenarios = 1;
    cfg.maxiter = 60;
    cfg.global_T = 120;
    cfg.waypoint_count = 12;
    cfg.bic_Tf = 24;
    cfg.bic_Tb = 24;
    cfg.N = 256;
    cfg.Nf = 256;
    cfg.Nb = 256;
    cfg.Nr = 128;
    cfg.vis_every = 20;
  }

  if (cfg.scenarios <= 0 || cfg.maps_per_scenario <= 0 ||
      cfg.dataset_maps <= 0 || cfg.maxiter <= 0 || cfg.global_T <= 0 ||
      cfg.waypoint_count < 2 || cfg.bic_Tf <= 0 || cfg.bic_Tb <= 0) {
    throw std::runtime_error(
        "Scenario count, map count, dataset size, maxiter, horizons, and at least two waypoints are required");
  }
  if (cfg.maps_per_scenario > cfg.dataset_maps) {
    throw std::runtime_error("maps-per-scenario cannot exceed dataset-maps");
  }
  return cfg;
}

inline std::string csvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

inline std::string csvString(const std::string &value) {
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

inline std::string jsonString(const std::string &value) {
  std::string out = "\"";
  for (char ch : value) {
    switch (ch) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out.push_back(ch);
      break;
    }
  }
  out += "\"";
  return out;
}

inline std::string joinIds(const std::vector<int> &ids, const std::string &sep) {
  std::ostringstream out;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) {
      out << sep;
    }
    out << ids[i];
  }
  return out.str();
}

inline Eigen::VectorXd state(double x, double y, double theta) {
  Eigen::VectorXd s(3);
  s << x, y, theta;
  return s;
}

inline double positionDistance(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  const double dx = a(0) - b(0);
  const double dy = a(1) - b(1);
  return std::sqrt(dx * dx + dy * dy);
}

inline int advanceReachedWaypointCount(
    const std::vector<Eigen::VectorXd> &waypoints, int reached_count,
    const Eigen::VectorXd &x, double tolerance) {
  int count = std::max(1, reached_count);
  while (count < static_cast<int>(waypoints.size()) &&
         positionDistance(x, waypoints[count]) <= tolerance) {
    ++count;
  }
  return count;
}

inline std::vector<int> selectMapIds(const Config &cfg, int scenario_index) {
  std::vector<int> ids(cfg.dataset_maps);
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937 rng(static_cast<unsigned int>(cfg.seed + scenario_index));
  std::shuffle(ids.begin(), ids.end(), rng);
  ids.resize(cfg.maps_per_scenario);
  return ids;
}

inline std::vector<std::vector<double>> loadTxtMap(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening map file: " + path.string());
  }

  std::vector<std::vector<double>> map;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream in(line);
    std::vector<double> row;
    double value = 0.0;
    while (in >> value) {
      row.push_back(value);
    }
    if (!row.empty()) {
      map.push_back(row);
    }
  }
  if (map.empty() || map[0].empty()) {
    throw std::runtime_error("Empty map file: " + path.string());
  }
  return map;
}

inline Scenario makeScenario(const Config &cfg, int scenario_index) {
  Scenario scenario;
  scenario.index = scenario_index;
  scenario.map_ids = selectMapIds(cfg, scenario_index);

  std::vector<std::vector<std::vector<double>>> maps;
  maps.reserve(scenario.map_ids.size());
  for (int id : scenario.map_ids) {
    maps.push_back(loadTxtMap(std::filesystem::path(cfg.dataset_dir) /
                              ("output_" + std::to_string(id) + ".txt")));
  }

  const int rows = static_cast<int>(maps.front().size());
  const int cols = static_cast<int>(maps.front()[0].size());
  for (const auto &map : maps) {
    if (static_cast<int>(map.size()) != rows ||
        static_cast<int>(map[0].size()) != cols) {
      throw std::runtime_error("All stitched maps must have identical dimensions");
    }
  }

  scenario.rows = rows;
  scenario.cols = cols * static_cast<int>(maps.size());
  scenario.map.assign(rows, std::vector<double>(scenario.cols, 0.0));
  for (int m = 0; m < static_cast<int>(maps.size()); ++m) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        scenario.map[r][m * cols + c] = maps[m][r][c];
      }
    }
  }
  return scenario;
}

inline CollisionChecker makeCollisionChecker(const Config &cfg,
                                             const Scenario &scenario) {
  CollisionChecker cc;
  cc.map = scenario.map;
  cc.with_map = true;
  cc.resolution = cfg.resolution;
  cc.max_row = scenario.rows;
  cc.max_col = scenario.cols;
  cc.is_3d = false;
  return cc;
}

inline std::vector<Eigen::VectorXd> makeWaypoints(const Config &cfg) {
  std::vector<Eigen::VectorXd> waypoints;
  waypoints.reserve(static_cast<std::size_t>(cfg.waypoint_count));
  const double total_length = cfg.map_length * cfg.maps_per_scenario;
  for (int i = 0; i < cfg.waypoint_count; ++i) {
    const double alpha =
        static_cast<double>(i) / static_cast<double>(cfg.waypoint_count - 1);
    waypoints.push_back(state(cfg.start_x, total_length * alpha, kHalfPi));
  }
  return waypoints;
}

inline MPPIParam makeMppiParam(const Config &cfg, const Eigen::VectorXd &start,
                               const Eigen::VectorXd &target, int T, int N) {
  MPPIParam param;
  param.dt = static_cast<float>(cfg.dt);
  param.T = T;
  param.x_init = start;
  param.x_target = target;
  param.N = N;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(2);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();
  return param;
}

inline BiMPPIParam makeBiParam(const Config &cfg, const Eigen::VectorXd &start,
                               const Eigen::VectorXd &target) {
  BiMPPIParam param;
  param.dt = static_cast<float>(cfg.dt);
  param.Tf = cfg.bic_Tf;
  param.Tb = cfg.bic_Tb;
  param.x_init = start;
  param.x_target = target;
  param.Nf = cfg.Nf;
  param.Nb = cfg.Nb;
  param.Nr = cfg.Nr;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(2);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;
  return param;
}

inline std::string solverVisName(const std::string &solver_key,
                                 int scenario_index) {
  std::ostringstream out;
  out << "stitched_" << solver_key << "_scenario_" << std::setw(2)
      << std::setfill('0') << scenario_index;
  return out.str();
}

inline void initLogger(const Config &cfg, MPPIVisLogger &logger,
                       const std::string &solver_key, int scenario_index,
                       int dim_x, int T, const Scenario &scenario,
                       const Eigen::VectorXd &start,
                       const Eigen::VectorXd &goal) {
  if (!cfg.save_rollouts) {
    return;
  }
  logger.enabled = true;
  logger.init(solverVisName(solver_key, scenario_index), dim_x, T,
              cfg.resolution, scenario.map, start, goal);
  logger.enabled = false;
}

inline bool shouldLogIteration(const Config &cfg, int iter) {
  return cfg.save_rollouts && cfg.vis_every > 0 && (iter % cfg.vis_every == 0);
}

inline void writeMapCsv(const std::filesystem::path &path,
                        const Scenario &scenario) {
  std::ofstream out(path);
  for (const auto &row : scenario.map) {
    for (std::size_t c = 0; c < row.size(); ++c) {
      if (c) {
        out << ',';
      }
      out << csvDouble(row[c]);
    }
    out << '\n';
  }
}

inline void writeScenarioMaps(std::ofstream &csv, const Scenario &scenario,
                              int original_cols) {
  for (int slot = 0; slot < static_cast<int>(scenario.map_ids.size()); ++slot) {
    csv << scenario.index << ',' << slot << ',' << scenario.map_ids[slot]
        << ',' << slot * original_cols << ','
        << ((slot + 1) * original_cols - 1) << '\n';
  }
}

inline void writeWaypoints(std::ofstream &csv, int scenario_index,
                           const std::vector<Eigen::VectorXd> &waypoints) {
  for (int i = 0; i < static_cast<int>(waypoints.size()); ++i) {
    csv << scenario_index << ',' << i << ',' << csvDouble(waypoints[i](0))
        << ',' << csvDouble(waypoints[i](1)) << ','
        << csvDouble(waypoints[i](2)) << '\n';
  }
}

inline void writePathRows(std::ofstream &csv, const std::string &solver_key,
                          int scenario_index, const RunStats &stats) {
  for (int i = 0; i < static_cast<int>(stats.path.size()); ++i) {
    csv << scenario_index << ',' << csvString(solver_key) << ',' << i << ','
        << csvDouble(stats.path[i](0)) << ',' << csvDouble(stats.path[i](1))
        << ',' << csvDouble(stats.path[i](2)) << '\n';
  }
}

inline void writeSummaryRow(std::ofstream &csv, const std::string &solver_key,
                            const std::string &solver_label,
                            const Scenario &scenario,
                            const RunStats &stats) {
  csv << csvString(solver_key) << ',' << csvString(solver_label) << ','
      << scenario.index << ',' << csvString(joinIds(scenario.map_ids, ";"))
      << ',' << (stats.success ? 1 : 0) << ','
      << (stats.is_collision ? 1 : 0) << ',' << stats.iter << ','
      << stats.reached_waypoints << ',' << stats.total_waypoints << ','
      << csvDouble(stats.elapsed) << ',' << csvDouble(stats.elapsed_rollout)
      << ',' << csvDouble(stats.elapsed_clustering) << ','
      << csvDouble(stats.elapsed_connection) << ','
      << csvDouble(stats.elapsed_guide) << ','
      << csvDouble(stats.final_error) << ',' << stats.path.size() << '\n';
}

inline void resetMppiWarmStart(MPPI_GPU &solver, int dim_u, int T) {
  solver.U_0 = Eigen::MatrixXd::Zero(dim_u, T);
}

inline void resetMppiWarmStart(LogMPPI_GPU &solver, int dim_u, int T) {
  solver.U_0 = Eigen::MatrixXd::Zero(dim_u, T);
}

inline void resetMppiWarmStart(ClusterMPPI_GPU &solver, int dim_u, int T) {
  solver.U_0 = Eigen::MatrixXd::Zero(dim_u, T);
}

inline void resetBiWarmStart(BiMPPI_GPU &solver, int dim_u, int Tf, int Tb) {
  solver.U_f0 = Eigen::MatrixXd::Zero(dim_u, Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(dim_u, Tb);
}

template <typename Solver>
RunStats runMppiLikeScenario(const Config &cfg, const std::string &solver_key,
                             int solver_seed_offset, const Scenario &scenario,
                             int N) {
  auto model = WMRobotMap();
  auto collision_checker = makeCollisionChecker(cfg, scenario);
  const auto waypoints = makeWaypoints(cfg);
  const auto start = waypoints.front();
  const auto final_goal = waypoints.back();

  Solver solver(model);
  auto param = makeMppiParam(cfg, start, final_goal, cfg.global_T, N);
  solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, cfg.global_T);
  solver.init(param);
  solver.setSeed(cfg.seed + solver_seed_offset + scenario.index * 101);
  solver.setCollisionChecker(&collision_checker);

  MPPIVisLogger logger;
  initLogger(cfg, logger, solver_key, scenario.index, model.dim_x, cfg.global_T,
             scenario, start, final_goal);
  solver.setVisLogger(&logger);

  RunStats stats;
  stats.total_waypoints = static_cast<int>(waypoints.size());
  stats.path.push_back(start);

  for (int iter = 0; iter < cfg.maxiter; ++iter) {
    stats.iter = iter + 1;
    if (collision_checker.getCollisionGrid(solver.x_init)) {
      stats.is_collision = true;
      break;
    }

    logger.enabled = shouldLogIteration(cfg, iter);
    if (logger.enabled) {
      logger.beginStep(iter);
    }
    solver.x_target = final_goal;
    solver.solve();
    logger.enabled = false;
    solver.move();

    stats.elapsed += solver.elapsed;
    stats.elapsed_rollout += solver.elapsed_rollout;
    stats.elapsed_clustering += solver.elapsed_clustering;
    stats.elapsed_connection += solver.elapsed_connection;
    stats.elapsed_guide += solver.elapsed_guide;
    stats.path.push_back(solver.x_init);
    stats.reached_waypoints = advanceReachedWaypointCount(
        waypoints, stats.reached_waypoints, solver.x_init,
        cfg.waypoint_tolerance);

    if (collision_checker.getCollisionGrid(solver.x_init)) {
      stats.is_collision = true;
      break;
    }

    if (positionDistance(solver.x_init, final_goal) <= cfg.waypoint_tolerance) {
      stats.success = true;
      break;
    }
  }

  stats.final_error = positionDistance(stats.path.back(), final_goal);
  stats.success = stats.success ||
                  (!stats.is_collision && stats.final_error <= cfg.waypoint_tolerance);
  return stats;
}

inline RunStats runBiScenario(const Config &cfg, const std::string &solver_key,
                              const Scenario &scenario) {
  auto model = WMRobotMap();
  auto collision_checker = makeCollisionChecker(cfg, scenario);
  const auto waypoints = makeWaypoints(cfg);
  const auto start = waypoints.front();
  const auto final_goal = waypoints.back();

  BiMPPI_GPU solver(model);
  auto param = makeBiParam(cfg, start, waypoints[1]);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
  solver.init(param);
  solver.setSeed(cfg.seed + 3000 + scenario.index * 101);
  solver.setCollisionChecker(&collision_checker);

  MPPIVisLogger logger;
  initLogger(cfg, logger, solver_key, scenario.index, model.dim_x,
             param.Tf + param.Tb, scenario, start, final_goal);
  solver.setVisLogger(&logger);

  RunStats stats;
  stats.total_waypoints = static_cast<int>(waypoints.size());
  stats.path.push_back(start);
  int target_index = 1;

  for (int iter = 0; iter < cfg.maxiter; ++iter) {
    stats.iter = iter + 1;
    if (collision_checker.getCollisionGrid(solver.x_init)) {
      stats.is_collision = true;
      break;
    }

    logger.enabled = shouldLogIteration(cfg, iter);
    if (logger.enabled) {
      logger.beginStep(iter);
    }
    solver.x_target = waypoints[target_index];
    solver.solve();
    logger.enabled = false;
    solver.move();

    stats.elapsed += solver.elapsed;
    stats.elapsed_rollout += solver.elapsed_rollout;
    stats.elapsed_clustering += solver.elapsed_clustering;
    stats.elapsed_connection += solver.elapsed_connection;
    stats.elapsed_guide += solver.elapsed_guide;
    stats.path.push_back(solver.x_init);

    if (collision_checker.getCollisionGrid(solver.x_init)) {
      stats.is_collision = true;
      break;
    }

    while (target_index < static_cast<int>(waypoints.size()) &&
           positionDistance(solver.x_init, waypoints[target_index]) <=
               cfg.waypoint_tolerance) {
      ++target_index;
      resetBiWarmStart(solver, model.dim_u, param.Tf, param.Tb);
      if (target_index < static_cast<int>(waypoints.size())) {
        solver.x_target = waypoints[target_index];
      }
    }

    stats.reached_waypoints = target_index;
    if (target_index >= static_cast<int>(waypoints.size())) {
      stats.success = true;
      break;
    }
  }

  stats.reached_waypoints = std::max(stats.reached_waypoints, target_index);
  stats.final_error = positionDistance(stats.path.back(), final_goal);
  stats.success = stats.success ||
                  (!stats.is_collision && stats.final_error <= cfg.waypoint_tolerance);
  return stats;
}

inline std::filesystem::path prepareSolverDir(const Config &cfg,
                                              const std::string &solver_key) {
  std::filesystem::path solver_dir =
      std::filesystem::path(cfg.out_dir) / solver_key;
  if (std::filesystem::exists(solver_dir) && cfg.overwrite) {
    std::filesystem::remove_all(solver_dir);
  }
  if (std::filesystem::exists(solver_dir / "summary.csv") && !cfg.overwrite) {
    throw std::runtime_error("Output already exists: " +
                             (solver_dir / "summary.csv").string() +
                             " (pass --overwrite)");
  }
  std::filesystem::create_directories(solver_dir / "stitched_maps");
  return solver_dir;
}

inline void writeMetadata(const Config &cfg, const std::string &solver_key,
                          const std::string &solver_label,
                          const std::filesystem::path &solver_dir,
                          const std::string &target_mode,
                          const std::string &rollout_mode) {
  std::ofstream json(solver_dir / "metadata.json");
  json << "{\n"
       << "  \"experiment\": \"wmrobot_gpu_waypoint_stitched_barn\",\n"
       << "  \"solver_key\": " << jsonString(solver_key) << ",\n"
       << "  \"solver_label\": " << jsonString(solver_label) << ",\n"
       << "  \"target_mode\": " << jsonString(target_mode) << ",\n"
       << "  \"rollout_mode\": " << jsonString(rollout_mode) << ",\n"
       << "  \"dataset_dir\": " << jsonString(cfg.dataset_dir) << ",\n"
       << "  \"scenarios\": " << cfg.scenarios << ",\n"
       << "  \"maps_per_scenario\": " << cfg.maps_per_scenario << ",\n"
       << "  \"dataset_maps\": " << cfg.dataset_maps << ",\n"
       << "  \"resolution\": " << csvDouble(cfg.resolution) << ",\n"
       << "  \"dt\": " << csvDouble(cfg.dt) << ",\n"
       << "  \"global_T\": " << cfg.global_T << ",\n"
       << "  \"waypoint_count\": " << cfg.waypoint_count << ",\n"
       << "  \"bic_Tf\": " << cfg.bic_Tf << ",\n"
       << "  \"bic_Tb\": " << cfg.bic_Tb << ",\n"
       << "  \"N\": " << cfg.N << ",\n"
       << "  \"Nf\": " << cfg.Nf << ",\n"
       << "  \"Nb\": " << cfg.Nb << ",\n"
       << "  \"Nr\": " << cfg.Nr << ",\n"
       << "  \"waypoint_tolerance\": " << csvDouble(cfg.waypoint_tolerance)
       << ",\n"
       << "  \"seed\": " << cfg.seed << ",\n"
       << "  \"save_rollouts\": " << (cfg.save_rollouts ? "true" : "false")
       << ",\n"
       << "  \"vis_every\": " << cfg.vis_every << "\n"
       << "}\n";
}

using ScenarioRunner = RunStats (*)(const Config &, const std::string &,
                                    const Scenario &);

template <typename Solver>
int runMppiLikeExecutable(int argc, char **argv, const std::string &solver_key,
                          const std::string &solver_label,
                          int solver_seed_offset,
                          Config cpp_defaults = Config()) {
  try {
    const Config cfg = parseArgs(argc, argv, cpp_defaults);
    const auto solver_dir = prepareSolverDir(cfg, solver_key);
    writeMetadata(cfg, solver_key, solver_label, solver_dir, "global_goal",
                  "single_direction_parallel_gpu_rollouts");

    std::ofstream summary(solver_dir / "summary.csv");
    summary << "solver,solver_label,scenario,map_ids,success,is_collision,iter,"
               "reached_waypoints,total_waypoints,elapsed,elapsed_rollout,"
               "elapsed_clustering,elapsed_connection,elapsed_guide,final_error,"
               "path_points\n";
    std::ofstream paths(solver_dir / "paths.csv");
    paths << "scenario,solver,step,x,y,theta\n";
    std::ofstream maps(solver_dir / "maps.csv");
    maps << "scenario,slot,map_id,col_begin,col_end\n";
    std::ofstream waypoints_csv(solver_dir / "waypoints.csv");
    waypoints_csv << "scenario,index,x,y,theta\n";

    for (int s = 0; s < cfg.scenarios; ++s) {
      const auto scenario = makeScenario(cfg, s);
      const int original_cols = scenario.cols / cfg.maps_per_scenario;
      writeScenarioMaps(maps, scenario, original_cols);
      writeWaypoints(waypoints_csv, s, makeWaypoints(cfg));
      std::ostringstream map_name;
      map_name << "scenario_" << std::setw(2) << std::setfill('0') << s
               << ".csv";
      writeMapCsv(solver_dir / "stitched_maps" / map_name.str(), scenario);

      const auto stats =
          runMppiLikeScenario<Solver>(cfg, solver_key, solver_seed_offset,
                                      scenario, cfg.N);
      writeSummaryRow(summary, solver_key, solver_label, scenario, stats);
      writePathRows(paths, solver_key, s, stats);
      std::cout << solver_label << " scenario " << s << " maps=["
                << joinIds(scenario.map_ids, ",") << "] success="
                << (stats.success ? 1 : 0) << " iter=" << stats.iter
                << " final_error=" << csvDouble(stats.final_error) << '\n';
    }

    std::cout << "outputs: " << solver_dir << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}

inline int runBiExecutable(int argc, char **argv,
                           Config cpp_defaults = Config()) {
  try {
    const Config cfg = parseArgs(argc, argv, cpp_defaults);
    const std::string solver_key = "bi_mppi";
    const std::string solver_label = "BiC-MPPI (parallel rollout)";
    const auto solver_dir = prepareSolverDir(cfg, solver_key);
    writeMetadata(cfg, solver_key, solver_label, solver_dir,
                  "waypoint_bidirectional_connections",
                  "forward_backward_parallel_gpu_rollouts");

    std::ofstream summary(solver_dir / "summary.csv");
    summary << "solver,solver_label,scenario,map_ids,success,is_collision,iter,"
               "reached_waypoints,total_waypoints,elapsed,elapsed_rollout,"
               "elapsed_clustering,elapsed_connection,elapsed_guide,final_error,"
               "path_points\n";
    std::ofstream paths(solver_dir / "paths.csv");
    paths << "scenario,solver,step,x,y,theta\n";
    std::ofstream maps(solver_dir / "maps.csv");
    maps << "scenario,slot,map_id,col_begin,col_end\n";
    std::ofstream waypoints_csv(solver_dir / "waypoints.csv");
    waypoints_csv << "scenario,index,x,y,theta\n";

    for (int s = 0; s < cfg.scenarios; ++s) {
      const auto scenario = makeScenario(cfg, s);
      const int original_cols = scenario.cols / cfg.maps_per_scenario;
      writeScenarioMaps(maps, scenario, original_cols);
      writeWaypoints(waypoints_csv, s, makeWaypoints(cfg));
      std::ostringstream map_name;
      map_name << "scenario_" << std::setw(2) << std::setfill('0') << s
               << ".csv";
      writeMapCsv(solver_dir / "stitched_maps" / map_name.str(), scenario);

      const auto stats = runBiScenario(cfg, solver_key, scenario);
      writeSummaryRow(summary, solver_key, solver_label, scenario, stats);
      writePathRows(paths, solver_key, s, stats);
      std::cout << solver_label << " scenario " << s << " maps=["
                << joinIds(scenario.map_ids, ",") << "] success="
                << (stats.success ? 1 : 0) << " iter=" << stats.iter
                << " final_error=" << csvDouble(stats.final_error) << '\n';
    }
    std::cout << "outputs: " << solver_dir << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}

} // namespace wmrobot_stitched
