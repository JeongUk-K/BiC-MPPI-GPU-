#pragma once

#include <quadrotor_precision_landing.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../gpu/quadrotor_path_logger.h"

struct QuadrotorGpuNewConfig {
  int T = 100;
  int Tf = 50;
  int Tb = 50;
  int N = 10000;
  int Nf = 10000;
  int Nb = 10000;
  int Nr = 5000;
  int Ns = 10;
  int istep = 5;
  int maxiter = 200;
  int map_begin = 299;
  int num_maps = 300;
  double dt = 0.1;
  double gamma_u = 10.0;
  double sigma_ax = 1.5;
  double sigma_ay = 1.5;
  double sigma_thrust = 1.5;
  double eps_xy = 0.30;
  double eps_vxy = 0.50;
  double eps_vz = 1.00;
  std::string dataset_dir = "../BARN_dataset/txt_files";
};

struct QuadrotorGpuNewRun {
  int map = 0;
  bool is_failed = true;
  bool is_landed = false;
  bool is_collision = false;
  bool is_arrived_plane = false;
  bool is_precision_landed = false;
  int iter = 0;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;
  double f_err = std::numeric_limits<double>::quiet_NaN();
  double per_step_elapsed = std::numeric_limits<double>::quiet_NaN();
  double per_step_rollout = std::numeric_limits<double>::quiet_NaN();
  double per_step_clustering = std::numeric_limits<double>::quiet_NaN();
  double per_step_connection = std::numeric_limits<double>::quiet_NaN();
  double per_step_guide = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd touchdown;
  double touchdown_xy_error = std::numeric_limits<double>::quiet_NaN();
  double touchdown_vxy = std::numeric_limits<double>::quiet_NaN();
  double touchdown_vz_abs = std::numeric_limits<double>::quiet_NaN();
  double touchdown_speed = std::numeric_limits<double>::quiet_NaN();
};

inline double quadrotorGpuNewQuietNaN() {
  return std::numeric_limits<double>::quiet_NaN();
}

inline double quadrotorGpuNewClamp01(double value) {
  if (value < 0.0) {
    return 0.0;
  }
  if (value > 1.0) {
    return 1.0;
  }
  return value;
}

inline bool quadrotorGpuNewLooksLikeOption(const std::string &value) {
  return value.rfind("--", 0) == 0;
}

inline std::string quadrotorGpuNewCsvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << value;
  return out.str();
}

inline double quadrotorGpuNewMean(const std::vector<double> &values) {
  if (values.empty()) {
    return quadrotorGpuNewQuietNaN();
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

inline void printQuadrotorGpuNewUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --smoke              Tiny validation run\n"
            << "  --num-maps N         Number of maps from map_begin downward\n"
            << "  --map-begin N        First map id\n"
            << "  --maxiter N          Max closed-loop iterations per map\n"
            << "  --T N                One-direction horizon\n"
            << "  --Tf N               Forward horizon\n"
            << "  --Tb N               Backward horizon\n"
            << "  --N N                One-direction rollout samples\n"
            << "  --Nf N               Forward rollout samples\n"
            << "  --Nb N               Backward rollout samples\n"
            << "  --Nr N               Guide rollout samples\n"
            << "  --Ns N               SVGD surrogate samples\n"
            << "  --istep N            SVGD inner iterations\n"
            << "  --dataset-dir DIR    BARN txt file directory\n";
}

inline QuadrotorGpuNewConfig parseQuadrotorGpuNewArgs(int argc, char **argv) {
  QuadrotorGpuNewConfig config;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc || quadrotorGpuNewLooksLikeOption(argv[i + 1])) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--help" || key == "-h") {
      printQuadrotorGpuNewUsage(argv[0]);
      std::exit(0);
    } else if (key == "--smoke") {
      config.T = 50;
      config.Tf = 25;
      config.Tb = 25;
      config.N = 256;
      config.Nf = 128;
      config.Nb = 128;
      config.Nr = 128;
      config.Ns = 4;
      config.istep = 2;
      config.maxiter = 10;
      config.num_maps = 3;
    } else if (key == "--num-maps") {
      config.num_maps = std::stoi(require_value(key));
    } else if (key == "--map-begin") {
      config.map_begin = std::stoi(require_value(key));
    } else if (key == "--maxiter") {
      config.maxiter = std::stoi(require_value(key));
    } else if (key == "--T") {
      config.T = std::stoi(require_value(key));
    } else if (key == "--Tf") {
      config.Tf = std::stoi(require_value(key));
    } else if (key == "--Tb") {
      config.Tb = std::stoi(require_value(key));
    } else if (key == "--N") {
      config.N = std::stoi(require_value(key));
    } else if (key == "--Nf") {
      config.Nf = std::stoi(require_value(key));
    } else if (key == "--Nb") {
      config.Nb = std::stoi(require_value(key));
    } else if (key == "--Nr") {
      config.Nr = std::stoi(require_value(key));
    } else if (key == "--Ns") {
      config.Ns = std::stoi(require_value(key));
    } else if (key == "--istep") {
      config.istep = std::stoi(require_value(key));
    } else if (key == "--dataset-dir") {
      config.dataset_dir = require_value(key);
    } else {
      throw std::runtime_error("Unknown option: " + key);
    }
  }
  return config;
}

inline Eigen::VectorXd quadrotorGpuNewStart(const Quadrotor &model) {
  Eigen::VectorXd x(model.dim_x);
  x << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
  return x;
}

inline Eigen::VectorXd quadrotorGpuNewTarget(const Quadrotor &model) {
  Eigen::VectorXd x(model.dim_x);
  x << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;
  return x;
}

inline Eigen::MatrixXd
quadrotorGpuNewSigma(const QuadrotorGpuNewConfig &config) {
  Eigen::VectorXd sigma(3);
  sigma << config.sigma_ax, config.sigma_ay, config.sigma_thrust;
  return sigma.asDiagonal();
}

inline MPPIParam makeQuadrotorGpuNewMppiParam(
    const QuadrotorGpuNewConfig &config, const Quadrotor &model) {
  MPPIParam param;
  param.dt = static_cast<float>(config.dt);
  param.T = config.T;
  param.x_init = quadrotorGpuNewStart(model);
  param.x_target = quadrotorGpuNewTarget(model);
  param.N = config.N;
  param.gamma_u = config.gamma_u;
  param.sigma_u = quadrotorGpuNewSigma(config);
  return param;
}

inline BiMPPIParam makeQuadrotorGpuNewBiParam(
    const QuadrotorGpuNewConfig &config, const Quadrotor &model) {
  BiMPPIParam param;
  param.dt = static_cast<float>(config.dt);
  param.Tf = config.Tf;
  param.Tb = config.Tb;
  param.x_init = quadrotorGpuNewStart(model);
  param.x_target = quadrotorGpuNewTarget(model);
  param.Nf = config.Nf;
  param.Nb = config.Nb;
  param.Nr = config.Nr;
  param.gamma_u = config.gamma_u;
  param.sigma_u = quadrotorGpuNewSigma(config);
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;
  return param;
}

inline SVGDMPPIParam makeQuadrotorGpuNewSvgdParam(
    const QuadrotorGpuNewConfig &config, const Quadrotor &model) {
  SVGDMPPIParam param;
  param.dt = static_cast<float>(config.dt);
  param.Tf = config.Tf;
  param.Tb = config.Tb;
  param.x_init = quadrotorGpuNewStart(model);
  param.x_target = quadrotorGpuNewTarget(model);
  param.Nf = config.Nf;
  param.Nb = config.Nb;
  param.Ns = config.Ns;
  param.istep = config.istep;
  param.Nr = config.Nr;
  param.gamma_u = config.gamma_u;
  param.sigma_u = quadrotorGpuNewSigma(config);
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;
  return param;
}

inline void writeQuadrotorGpuNewResultHeader(std::ofstream &csv) {
  csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,"
         "elapsed_clustering,elapsed_connection,elapsed_guide,f_err,"
         "is_collision,is_arrived_plane,is_precision_landed,"
         "per_step_elapsed,per_step_rollout,per_step_clustering,"
         "per_step_connection,per_step_guide,"
         "touchdown_x,touchdown_y,touchdown_z,"
         "touchdown_vx,touchdown_vy,touchdown_vz,"
         "touchdown_xy_error,touchdown_vxy,touchdown_vz_abs,"
         "touchdown_speed\n";
}

inline void writeQuadrotorGpuNewResultRow(std::ofstream &csv,
                                          const QuadrotorGpuNewRun &run) {
  csv << run.map << ',' << run.is_failed << ',' << run.is_landed << ','
      << run.iter << ',' << run.elapsed << ',' << run.elapsed_rollout << ','
      << run.elapsed_clustering << ',' << run.elapsed_connection << ','
      << run.elapsed_guide << ',' << run.f_err << ',' << run.is_collision
      << ',' << run.is_arrived_plane << ',' << run.is_precision_landed << ','
      << run.per_step_elapsed << ',' << run.per_step_rollout << ','
      << run.per_step_clustering << ',' << run.per_step_connection << ','
      << run.per_step_guide << ',' << run.touchdown(0) << ','
      << run.touchdown(1) << ',' << run.touchdown(2) << ','
      << run.touchdown(3) << ',' << run.touchdown(4) << ','
      << run.touchdown(5) << ',' << run.touchdown_xy_error << ','
      << run.touchdown_vxy << ',' << run.touchdown_vz_abs << ','
      << run.touchdown_speed << '\n';
}

inline void writeQuadrotorGpuNewSummary(const std::string &path,
                                        const char *variant,
                                        const std::vector<QuadrotorGpuNewRun> &runs) {
  int num_landed = 0;
  int num_precision = 0;
  int num_collision = 0;
  std::vector<double> precision_iters;
  std::vector<double> precision_elapsed;
  std::vector<double> precision_rollout;
  std::vector<double> precision_clustering;
  std::vector<double> precision_connection;
  std::vector<double> precision_guide;
  std::vector<double> precision_xy_error;
  std::vector<double> precision_vxy;
  std::vector<double> precision_vz_abs;
  std::vector<double> precision_speed;

  for (const auto &run : runs) {
    if (run.is_landed) {
      ++num_landed;
    }
    if (run.is_collision) {
      ++num_collision;
    }
    if (run.is_precision_landed) {
      ++num_precision;
      precision_iters.push_back(static_cast<double>(run.iter));
      precision_elapsed.push_back(run.elapsed);
      precision_rollout.push_back(run.elapsed_rollout);
      precision_clustering.push_back(run.elapsed_clustering);
      precision_connection.push_back(run.elapsed_connection);
      precision_guide.push_back(run.elapsed_guide);
      precision_xy_error.push_back(run.touchdown_xy_error);
      precision_vxy.push_back(run.touchdown_vxy);
      precision_vz_abs.push_back(run.touchdown_vz_abs);
      precision_speed.push_back(run.touchdown_speed);
    }
  }

  const int num_runs = static_cast<int>(runs.size());
  std::ofstream csv(path);
  csv << "variant,num_runs,num_precision_landed,num_landed,num_collision,"
         "num_failure,precision_success_rate,landed_rate,collision_rate,"
         "avg_iter_precision,avg_total_elapsed_precision,"
         "avg_rollout_time_precision,avg_clustering_time_precision,"
         "avg_connection_time_precision,avg_guide_time_precision,"
         "avg_touchdown_xy_error_precision,avg_touchdown_vxy_precision,"
         "avg_touchdown_vz_abs_precision,avg_touchdown_speed_precision\n";
  csv << variant << ',' << num_runs << ',' << num_precision << ','
      << num_landed << ',' << num_collision << ','
      << (num_runs - num_precision) << ','
      << (num_runs > 0 ? static_cast<double>(num_precision) / num_runs : 0.0)
      << ','
      << (num_runs > 0 ? static_cast<double>(num_landed) / num_runs : 0.0)
      << ','
      << (num_runs > 0 ? static_cast<double>(num_collision) / num_runs : 0.0)
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_iters))
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_elapsed))
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_rollout))
      << ','
      << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_clustering))
      << ','
      << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_connection))
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_guide))
      << ','
      << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_xy_error))
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_vxy))
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_vz_abs))
      << ',' << quadrotorGpuNewCsvDouble(quadrotorGpuNewMean(precision_speed))
      << '\n';
}

template <typename Solver>
QuadrotorGpuNewRun runQuadrotorGpuNewLoop(
    Solver &solver, CollisionChecker &collision_checker,
    const QuadrotorGpuNewConfig &config, const Eigen::VectorXd &target,
    const char *path_variant, int map, std::ofstream &path_csv) {
  QuadrotorGpuNewRun run;
  run.map = map;
  run.touchdown = Eigen::VectorXd::Constant(6, quadrotorGpuNewQuietNaN());

  writeQuadrotorPathRow(path_csv, path_variant, map, 0, solver.x_init);

  for (int i = 0; i < config.maxiter; ++i) {
    const Eigen::VectorXd x_prev = solver.x_init;

    solver.solve();
    solver.move();

    const Eigen::VectorXd x_next = solver.x_init;
    run.iter = i + 1;
    writeQuadrotorPathRow(path_csv, path_variant, map, run.iter, x_next);

    run.elapsed += solver.elapsed;
    run.elapsed_rollout += solver.elapsed_rollout;
    run.elapsed_clustering += solver.elapsed_clustering;
    run.elapsed_connection += solver.elapsed_connection;
    run.elapsed_guide += solver.elapsed_guide;

    if (collision_checker.getCollisionGrid(x_next)) {
      run.is_collision = true;
      run.is_failed = true;
      break;
    }

    const bool crossed_ground = (x_prev(2) > 0.0 && x_next(2) <= 0.0);
    if (!crossed_ground) {
      continue;
    }

    double alpha = x_prev(2) / (x_prev(2) - x_next(2));
    alpha = quadrotorGpuNewClamp01(alpha);
    run.touchdown = x_prev + alpha * (x_next - x_prev);
    run.touchdown(2) = 0.0;

    if (collision_checker.getCollisionGrid(run.touchdown)) {
      run.is_collision = true;
      run.is_failed = true;
      break;
    }

    run.touchdown_xy_error = (run.touchdown.head(2) - target.head(2)).norm();
    run.touchdown_vxy = run.touchdown.segment(3, 2).norm();
    run.touchdown_vz_abs = std::abs(run.touchdown(5));
    run.touchdown_speed = run.touchdown.segment(3, 3).norm();
    run.f_err = run.touchdown_xy_error;
    run.is_arrived_plane = true;
    run.is_landed = true;
    run.is_precision_landed =
        (run.touchdown_xy_error < config.eps_xy) &&
        (run.touchdown_vxy < config.eps_vxy) &&
        (run.touchdown_vz_abs < config.eps_vz);
    run.is_failed = !run.is_precision_landed;
    break;
  }

  const double denom = static_cast<double>(std::max(1, run.iter));
  run.per_step_elapsed = run.elapsed / denom;
  run.per_step_rollout = run.elapsed_rollout / denom;
  run.per_step_clustering = run.elapsed_clustering / denom;
  run.per_step_connection = run.elapsed_connection / denom;
  run.per_step_guide = run.elapsed_guide / denom;

  return run;
}

template <typename Solver>
inline int runQuadrotorGpuNewOneDirectional(
    int argc, char **argv, const char *variant_token, const char *variant_label) {
  const auto config = parseQuadrotorGpuNewArgs(argc, argv);
  auto model = QuadrotorPrecisionLanding();
  auto param = makeQuadrotorGpuNewMppiParam(config, model);
  const auto target = param.x_target;

  const std::string result_path =
      std::string("result_quadrotor_") + variant_token + ".csv";
  const std::string path_path =
      std::string("path_quadrotor_") + variant_token + ".csv";
  const std::string summary_path =
      std::string("result_quadrotor_") + variant_token + "_summary.csv";

  std::ofstream csv(result_path);
  writeQuadrotorGpuNewResultHeader(csv);
  std::ofstream path_csv(path_path);
  writeQuadrotorPathHeader(path_csv);

  std::vector<QuadrotorGpuNewRun> runs;
  for (int map = config.map_begin;
       map >= 0 && map > config.map_begin - config.num_maps; --map) {
    CollisionChecker collision_checker;
    collision_checker.loadMap(config.dataset_dir + "/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    Solver solver(model);
    solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
    solver.U_0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);

    auto run = runQuadrotorGpuNewLoop(
        solver, collision_checker, config, target, variant_token, map, path_csv);
    std::cout << map << '\t' << run.is_failed << '\t' << run.is_landed
              << '\t' << run.is_precision_landed << '\t' << run.iter << '\t'
              << run.elapsed << std::endl;
    writeQuadrotorGpuNewResultRow(csv, run);
    runs.push_back(run);
  }

  csv.close();
  path_csv.close();
  writeQuadrotorGpuNewSummary(summary_path, variant_label, runs);
  return 0;
}

template <typename Solver>
inline int runQuadrotorGpuNewBidirectional(
    int argc, char **argv, const char *variant_token, const char *variant_label) {
  const auto config = parseQuadrotorGpuNewArgs(argc, argv);
  auto model = QuadrotorPrecisionLanding();
  auto param = makeQuadrotorGpuNewBiParam(config, model);
  const auto target = param.x_target;

  const std::string result_path =
      std::string("result_quadrotor_") + variant_token + ".csv";
  const std::string path_path =
      std::string("path_quadrotor_") + variant_token + ".csv";
  const std::string summary_path =
      std::string("result_quadrotor_") + variant_token + "_summary.csv";

  std::ofstream csv(result_path);
  writeQuadrotorGpuNewResultHeader(csv);
  std::ofstream path_csv(path_path);
  writeQuadrotorPathHeader(path_csv);

  std::vector<QuadrotorGpuNewRun> runs;
  for (int map = config.map_begin;
       map >= 0 && map > config.map_begin - config.num_maps; --map) {
    CollisionChecker collision_checker;
    collision_checker.loadMap(config.dataset_dir + "/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    Solver solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
    solver.U_f0.row(2).array() += model.g;
    solver.U_b0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);
    solver.dummy_u(2) += model.g;

    auto run = runQuadrotorGpuNewLoop(
        solver, collision_checker, config, target, variant_token, map, path_csv);
    std::cout << map << '\t' << run.is_failed << '\t' << run.is_landed
              << '\t' << run.is_precision_landed << '\t' << run.iter << '\t'
              << run.elapsed << std::endl;
    writeQuadrotorGpuNewResultRow(csv, run);
    runs.push_back(run);
  }

  csv.close();
  path_csv.close();
  writeQuadrotorGpuNewSummary(summary_path, variant_label, runs);
  return 0;
}

template <typename Solver>
inline int runQuadrotorGpuNewSvgd(
    int argc, char **argv, const char *variant_token, const char *variant_label) {
  const auto config = parseQuadrotorGpuNewArgs(argc, argv);
  auto model = QuadrotorPrecisionLanding();
  auto param = makeQuadrotorGpuNewSvgdParam(config, model);
  const auto target = param.x_target;

  const std::string result_path =
      std::string("result_quadrotor_") + variant_token + ".csv";
  const std::string path_path =
      std::string("path_quadrotor_") + variant_token + ".csv";
  const std::string summary_path =
      std::string("result_quadrotor_") + variant_token + "_summary.csv";

  std::ofstream csv(result_path);
  writeQuadrotorGpuNewResultHeader(csv);
  std::ofstream path_csv(path_path);
  writeQuadrotorPathHeader(path_csv);

  std::vector<QuadrotorGpuNewRun> runs;
  for (int map = config.map_begin;
       map >= 0 && map > config.map_begin - config.num_maps; --map) {
    CollisionChecker collision_checker;
    collision_checker.loadMap(config.dataset_dir + "/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    Solver solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
    solver.U_f0.row(2).array() += model.g;
    solver.U_b0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);
    solver.dummy_u(2) += model.g;

    auto run = runQuadrotorGpuNewLoop(
        solver, collision_checker, config, target, variant_token, map, path_csv);
    std::cout << map << '\t' << run.is_failed << '\t' << run.is_landed
              << '\t' << run.is_precision_landed << '\t' << run.iter << '\t'
              << run.elapsed << std::endl;
    writeQuadrotorGpuNewResultRow(csv, run);
    runs.push_back(run);
  }

  csv.close();
  path_csv.close();
  writeQuadrotorGpuNewSummary(summary_path, variant_label, runs);
  return 0;
}
