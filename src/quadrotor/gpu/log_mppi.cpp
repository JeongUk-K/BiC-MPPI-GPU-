#include <log_mppi_gpu.cuh>
#include <quadrotor.h>

#include <Eigen/Dense>
#include <chrono>
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

#include "quadrotor_path_logger.h"
#include "quadrotor_gpu_vis.h"

namespace {

struct QuadrotorLogMppiConfig {
  int T = 100;
  int N = 10000;
  int maxiter = 200;
  int map_begin = 299;
  int num_maps = 300;
  int vis_every = 1;
  bool save_rollouts = true;
  std::string dataset_dir = "../BARN_dataset/txt_files";
};

struct QuadrotorLogMppiRun {
  int map = 0;
  bool is_failed = true;
  bool is_landed = false;
  int iter = 0;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;
  double f_err = 0.0;
};

bool looksLikeOption(const std::string &value) {
  return value.rfind("--", 0) == 0;
}

std::string csvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream out;
  out << value;
  return out.str();
}

double mean(const std::vector<double> &values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --smoke              Tiny validation run\n"
            << "  --num-maps N         Number of maps from map_begin downward\n"
            << "  --map-begin N        First map id\n"
            << "  --maxiter N          Max closed-loop iterations per run\n"
            << "  --T N                Horizon length\n"
            << "  --N N                Number of rollout samples\n"
            << "  --dataset-dir DIR    BARN txt file directory\n"
            << "  --vis-every N        Save rollout data every N iterations\n"
            << "  --no-rollouts        Save CSV files only\n";
}

QuadrotorLogMppiConfig parseArgs(int argc, char **argv) {
  QuadrotorLogMppiConfig config;
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
    } else if (key == "--smoke") {
      config.T = 50;
      config.N = 256;
      config.maxiter = 10;
      config.num_maps = 3;
      config.vis_every = 1;
    } else if (key == "--num-maps") {
      config.num_maps = std::stoi(require_value(key));
    } else if (key == "--map-begin") {
      config.map_begin = std::stoi(require_value(key));
    } else if (key == "--maxiter") {
      config.maxiter = std::stoi(require_value(key));
    } else if (key == "--T") {
      config.T = std::stoi(require_value(key));
    } else if (key == "--N") {
      config.N = std::stoi(require_value(key));
    } else if (key == "--dataset-dir") {
      config.dataset_dir = require_value(key);
    } else if (key == "--vis-every") {
      config.vis_every = std::stoi(require_value(key));
    } else if (key == "--no-rollouts") {
      config.save_rollouts = false;
    } else {
      throw std::runtime_error("Unknown option: " + key);
    }
  }
  return config;
}

void writeSummaryCsv(const std::vector<QuadrotorLogMppiRun> &runs) {
  int num_success = 0;
  int num_landed = 0;
  std::vector<double> success_iters;
  std::vector<double> success_elapsed;
  std::vector<double> success_rollout;
  std::vector<double> success_clustering;
  std::vector<double> success_connection;
  std::vector<double> success_guide;
  std::vector<double> success_f_err;

  for (const auto &run : runs) {
    if (run.is_landed) {
      ++num_landed;
    }
    if (!run.is_failed) {
      ++num_success;
      success_iters.push_back(static_cast<double>(run.iter));
      success_elapsed.push_back(run.elapsed);
      success_rollout.push_back(run.elapsed_rollout);
      success_clustering.push_back(run.elapsed_clustering);
      success_connection.push_back(run.elapsed_connection);
      success_guide.push_back(run.elapsed_guide);
      success_f_err.push_back(run.f_err);
    }
  }

  const int num_runs = static_cast<int>(runs.size());
  const int num_failure = num_runs - num_success;
  std::ofstream summary_csv("result_quadrotor_log_mppi_summary.csv");
  summary_csv
      << "variant,num_runs,num_success,num_failure,num_landed,success_rate,"
         "landed_rate,avg_iter_success,avg_total_elapsed_success,"
         "avg_rollout_time_success,avg_clustering_time_success,"
         "avg_connection_time_success,avg_guide_time_success,avg_f_err_success\n";
  summary_csv << "Log-MPPI," << num_runs << ',' << num_success << ','
              << num_failure << ',' << num_landed << ','
              << (num_runs > 0 ? static_cast<double>(num_success) / num_runs
                               : 0.0)
              << ','
              << (num_runs > 0 ? static_cast<double>(num_landed) / num_runs
                               : 0.0)
              << ',' << csvDouble(mean(success_iters)) << ','
              << csvDouble(mean(success_elapsed)) << ','
              << csvDouble(mean(success_rollout)) << ','
              << csvDouble(mean(success_clustering)) << ','
              << csvDouble(mean(success_connection)) << ','
              << csvDouble(mean(success_guide)) << ','
              << csvDouble(mean(success_f_err)) << '\n';
}

} // namespace

int main(int argc, char **argv) {
  const auto config = parseArgs(argc, argv);
  auto model = Quadrotor();

  using Solver = LogMPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = config.T;
  param.x_init.resize(model.dim_x);
  param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;
  param.N = config.N;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 1.5, 1.5, 1.5;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter = config.maxiter;

  std::ofstream csv("result_quadrotor_log_mppi.csv");
  csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,elapsed_"
         "clustering,"
         "elapsed_connection,elapsed_guide,f_err\n";
  std::ofstream path_csv("path_quadrotor_log_mppi.csv");
  writeQuadrotorPathHeader(path_csv);
  std::ofstream progress_csv("result_quadrotor_log_mppi_progress.csv");
  csv << std::flush;
  progress_csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,elapsed_"
                  "clustering,"
                  "elapsed_connection,elapsed_guide,f_err\n";
  progress_csv << std::flush;
  std::vector<QuadrotorLogMppiRun> runs;

  for (int map = config.map_begin;
       map >= 0 && map > config.map_begin - config.num_maps; --map) {
    CollisionChecker collision_checker = CollisionChecker();
    collision_checker.loadMap(config.dataset_dir + "/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);
    Solver solver(model);
    solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
    solver.U_0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);

    MPPIVisLogger vis_logger;
    initQuadrotorGpuVisLogger(vis_logger, config.save_rollouts, "log_mppi",
                              map, model.dim_x, param.T, collision_checker,
                              param.x_init, param.x_target);
    solver.setVisLogger(&vis_logger);

    writeQuadrotorPathRow(path_csv, "log_mppi", map, 0, solver.x_init);

    bool is_landed = false;
    bool is_failed = true;
    int i = 0;
    double total_elapsed = 0.0;
    double total_rollout = 0.0;
    double total_clustering = 0.0;
    double total_connection = 0.0;
    double total_guide = 0.0;
    double f_err = 0.0;
    for (i = 0; i < maxiter; ++i) {
      beginQuadrotorGpuVisStep(vis_logger, config.save_rollouts,
                               config.vis_every, i);
      solver.solve();
      endQuadrotorGpuVisStep(vis_logger);
      solver.move();
      writeQuadrotorPathRow(path_csv, "log_mppi", map, i + 1,
                            solver.x_init);

      total_elapsed += solver.elapsed;
      total_rollout += solver.elapsed_rollout;
      total_clustering += solver.elapsed_clustering;
      total_connection += solver.elapsed_connection;
      total_guide += solver.elapsed_guide;

      f_err = (solver.x_init.head(2) - param.x_target.head(2)).norm();
      if (collision_checker.getCollisionGrid(solver.x_init)) {
        progress_csv << map << ',' << is_failed << ',' << is_landed << ','
                     << i << ',' << total_elapsed << ',' << total_rollout
                     << ',' << total_clustering << ',' << total_connection
                     << ',' << total_guide << ',' << f_err << '\n'
                     << std::flush;
        break;
      } else {
        if (solver.x_init(2) < 0) {
          is_landed = true;
          if (f_err < 0.3) {
            is_failed = false;
          }
          progress_csv << map << ',' << is_failed << ',' << is_landed << ','
                       << i << ',' << total_elapsed << ',' << total_rollout
                       << ',' << total_clustering << ',' << total_connection
                       << ',' << total_guide << ',' << f_err << '\n'
                       << std::flush;
          break;
        }
      }
      progress_csv << map << ',' << is_failed << ',' << is_landed << ',' << i
                   << ',' << total_elapsed << ',' << total_rollout << ','
                   << total_clustering << ',' << total_connection << ','
                   << total_guide << ',' << f_err << '\n'
                   << std::flush;
    }
    std::cout << map << '\t' << is_failed << '\t' << is_landed << '\t' << i
              << '\t' << total_elapsed << std::endl;
    csv << map << ',' << is_failed << ',' << is_landed << ',' << i << ','
        << total_elapsed << ',' << total_rollout << ',' << total_clustering
        << ',' << total_connection << ',' << total_guide << ',' << f_err
        << '\n'
        << std::flush;
    runs.push_back({map, is_failed, is_landed, i, total_elapsed, total_rollout,
                    total_clustering, total_connection, total_guide, f_err});
  }

  csv.close();
  path_csv.close();
  progress_csv.close();
  writeSummaryCsv(runs);
  return 0;
}
