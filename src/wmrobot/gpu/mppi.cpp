#include <mppi_gpu.cuh>
#include <wmrobot_map.h>

#include "wmrobot_gpu_stats.h"
#include "wmrobot_gpu_vis.h"

#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct MppiConfig {
  int T = 100;
  int N = 6000;
  int maxiter = 200;
  int map_begin = 299;
  int num_maps = 300;
  int start_cases = 2;
  int vis_every = 1;
  bool save_rollouts = true;
  std::string dataset_dir = "../BARN_dataset/txt_files";
};

bool looksLikeOption(const std::string &value) {
  return value.rfind("--", 0) == 0;
}

void printUsage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "Options:\n"
      << "  --smoke              Run one tiny validation case\n"
      << "  --map-begin N        First BARN map id. Default: 299\n"
      << "  --num-maps N         Number of maps descending from map-begin. "
         "Default: 300\n"
      << "  --start-cases N      Number of start cases. Default: 2\n"
      << "  --maxiter N          Max closed-loop iterations. Default: 200\n"
      << "  --T N                Horizon. Default: 100\n"
      << "  --N N                Rollout count. Default: 10000\n"
      << "  --dataset-dir DIR    BARN txt file directory\n"
      << "  --vis-every N        Save rollout data every N iterations. "
         "Default: 1\n"
      << "  --no-rollouts        Save CSV files only\n";
}

MppiConfig parseArgs(int argc, char **argv) {
  MppiConfig config;
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
      config.T = 20;
      config.N = 256;
      config.maxiter = 2;
      config.num_maps = 1;
      config.start_cases = 1;
      config.vis_every = 1;
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
    } else if (key == "--N") {
      config.N = std::stoi(require_value(key));
    } else if (key == "--dataset-dir") {
      config.dataset_dir = require_value(key);
    } else if (key == "--vis-every") {
      config.vis_every = std::stoi(require_value(key));
    } else if (key == "--no-rollouts") {
      config.save_rollouts = false;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  return config;
}

} // namespace

int main(int argc, char **argv) {
  const auto config = parseArgs(argc, argv);
  auto model = WMRobotMap();

  using Solver = MPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = config.T;
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;
  param.N = config.N;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();

  int maxiter = config.maxiter;

  const std::string variant = "MPPI";
  std::vector<WmrobotGpuRunResult> runs;

  std::ofstream csv(wmrobotGpuResultPath("result_mppi.csv"));
  writeWmrobotGpuRunHeader(csv);
  csv.flush();

  std::ofstream progress_csv(wmrobotGpuResultPath("result_mppi_progress.csv"));
  writeWmrobotGpuRunHeader(progress_csv);
  progress_csv.flush();
  // for (int map = 299; map >= 0 ; --map) {
  // for (int map = 276; map >= 0; --map) {
  // for (int s = 1; s < 2; ++s) {
  for (int s = 0; s < config.start_cases; ++s) {
    switch (s) {
    case 0:
      param.x_init(0) = 0.5;
      break;
    case 1:
      param.x_init(0) = 2.5;
      break;
    case 2:
      param.x_init(0) = 1.5;
      break;
    default:
      break;
    }
    for (int map = config.map_begin;
         map >= 0 && map > config.map_begin - config.num_maps; --map) {
      // for (int map = 0; map < 300; ++map) {
      CollisionChecker collision_checker = CollisionChecker();
      collision_checker.loadMap(
          config.dataset_dir + "/output_" + std::to_string(map) + ".txt", 0.1);
      Solver solver(model);
      solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
      solver.init(param);
      solver.setCollisionChecker(&collision_checker);

      MPPIVisLogger vis_logger;
      initWmrobotGpuVisLogger(vis_logger, config.save_rollouts, "mppi", s, map,
                              model.dim_x, param.T, collision_checker,
                              param.x_init, param.x_target);
      solver.setVisLogger(&vis_logger);

      WmrobotGpuRunResult row;
      row.variant = variant;
      row.start_case = s;
      row.map_id = map;

      for (row.iter = 0; row.iter < maxiter; ++row.iter) {

        beginWmrobotGpuVisStep(vis_logger, config.save_rollouts,
                               config.vis_every, row.iter);
        solver.solve();
        endWmrobotGpuVisStep(vis_logger);
        row.cost = solver.trajectoryCost();
        solver.move();

        row.elapsed += solver.elapsed;
        row.elapsed_rollout += solver.elapsed_rollout;
        row.elapsed_clustering += solver.elapsed_clustering;
        row.elapsed_connection += solver.elapsed_connection;
        row.elapsed_guide += solver.elapsed_guide;

        // std::cout<<"1 solved in "<<solver.elapsed_1.count()<<std::endl;

        row.d_goal = (solver.x_init - param.x_target).norm();
        if (collision_checker.getCollisionGrid(solver.x_init)) {
          row.is_collision = true;
          writeWmrobotGpuRunRow(progress_csv, row);
          progress_csv.flush();
          break;
        } else {
          if (row.d_goal < 0.1) {
            row.is_success = true;
            writeWmrobotGpuRunRow(progress_csv, row);
            progress_csv.flush();
            break;
          }
        }
        writeWmrobotGpuRunRow(progress_csv, row);
        progress_csv.flush();
        // solver.show();
      }
      if (row.iter >= maxiter) {
        row.iter = maxiter;
      }
      printWmrobotGpuRunRow(std::cout, row);
      writeWmrobotGpuRunRow(csv, row);
      csv.flush();
      runs.push_back(row);
      // solver.showTraj();
    }
  }

  csv.close();
  progress_csv.close();

  const auto summary = summarizeWmrobotGpuRuns(variant, runs);
  std::ofstream summary_csv(wmrobotGpuResultPath("result_mppi_summary.csv"));
  writeWmrobotGpuSummaryHeader(summary_csv);
  writeWmrobotGpuSummaryRow(summary_csv, summary);
  return 0;
}
