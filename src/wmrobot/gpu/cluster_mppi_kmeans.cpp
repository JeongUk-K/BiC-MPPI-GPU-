#include <cluster_mppi_gpu.cuh>
#include <wmrobot_map.h>

#include "wmrobot_gpu_stats.h"
#include "wmrobot_gpu_vis.h"

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  auto model = WMRobotMap();

  using Solver = ClusterMPPI_GPU;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.T = 100;
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;
  param.N = 6000;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();
  // [fastsc GPU K-means]
  param.clustering_method = ClusteringMethod::KMeans;
  param.kmeans_clusters = 5;
  param.kmeans_max_iterations = 100;
  param.kmeans_threshold = 1e-6;

  bool has_seed = false;
  std::uint_fast64_t seed = 0;

  int map_begin = 299;
  int num_maps = 300;
  int start_cases = 2;
  int maxiter = 200;
  int vis_every = 1;
  bool save_rollouts = true;
  std::string dataset_dir = "../BARN_dataset/txt_files";

  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--smoke") {
      map_begin = 299;
      num_maps = 1;
      start_cases = 1;
      maxiter = 2;
      param.T = 20;
      param.N = 256;
      vis_every = 1;
    } else if (key == "--map-begin") {
      map_begin = std::stoi(require_value(key));
    } else if (key == "--num-maps") {
      num_maps = std::stoi(require_value(key));
    } else if (key == "--start-cases") {
      start_cases = std::stoi(require_value(key));
    } else if (key == "--maxiter") {
      maxiter = std::stoi(require_value(key));
    } else if (key == "--T") {
      param.T = std::stoi(require_value(key));
    } else if (key == "--N") {
      param.N = std::stoi(require_value(key));
    } else if (key == "--kmeans-clusters") {
      param.kmeans_clusters = std::stoi(require_value(key));
    } else if (key == "--kmeans-iters") {
      param.kmeans_max_iterations = std::stoi(require_value(key));
    } else if (key == "--kmeans-threshold") {
      param.kmeans_threshold = std::stod(require_value(key));
    } else if (key == "--seed") {
      seed = static_cast<std::uint_fast64_t>(
          std::stoull(require_value(key)));
      has_seed = true;
    } else if (key == "--dataset-dir") {
      dataset_dir = require_value(key);
    } else if (key == "--vis-every") {
      vis_every = std::stoi(require_value(key));
    } else if (key == "--no-rollouts") {
      save_rollouts = false;
    } else if (key == "--help" || key == "-h") {
      std::cout
          << "Usage: wmrobot_cluster_mppi_kmeans [options]\n"
          << "  --smoke              Run one tiny validation case\n"
          << "  --map-begin N        First BARN map id. Default: 299\n"
          << "  --num-maps N         Number of maps descending from map-begin. Default: 300\n"
          << "  --start-cases N      Number of start cases. Default: 2\n"
          << "  --maxiter N          Max closed-loop iterations. Default: 200\n"
          << "  --T N                Horizon. Default: 100\n"
          << "  --N N                Rollout count. Default: 10000\n"
          << "  --kmeans-clusters N  Number of K-means clusters. Default: 5\n"
          << "  --kmeans-iters N     Maximum K-means iterations. Default: 100\n"
          << "  --kmeans-threshold X Relative convergence threshold. Default: 1e-6\n"
          << "  --seed N             Fixed CUDA random seed\n"
          << "  --dataset-dir DIR    BARN txt file directory\n"
          << "  --vis-every N        Save rollout data every N iterations. Default: 1\n"
          << "  --no-rollouts        Save CSV files only\n";
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  if (param.kmeans_clusters <= 0 || param.kmeans_max_iterations <= 0 ||
      param.kmeans_threshold < 0.0)
    throw std::runtime_error(
        "K-means parameters must be positive (threshold may be zero)");

  const std::string variant = "Cluster-MPPI-KMeans";
  std::vector<WmrobotGpuRunResult> runs;

  std::ofstream csv("result_cluster_mppi_kmeans.csv");
  writeWmrobotGpuRunHeader(csv);
  csv.flush();

  std::ofstream progress_csv("result_cluster_mppi_kmeans_progress.csv");
  writeWmrobotGpuRunHeader(progress_csv);
  progress_csv.flush();
  // for (int map = 299; map >= 0 ; --map) {
  // for (int map = 276; map >= 0; --map) {
  // for (int s = 1; s < 2; ++s) {
  for (int s = 0; s < start_cases; ++s) {
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
    for (int map = map_begin; map >= 0 && map > map_begin - num_maps; --map) {
      // for (int map = 0; map < 300; ++map) {
      CollisionChecker collision_checker = CollisionChecker();
      collision_checker.loadMap(dataset_dir + "/output_" +
                                    std::to_string(map) + ".txt",
                                0.1);
      Solver solver(model);
      if (has_seed) solver.setSeed(seed);
      solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
      solver.init(param);
      solver.setCollisionChecker(&collision_checker);

      MPPIVisLogger vis_logger;
      initWmrobotGpuVisLogger(vis_logger, save_rollouts,
                              "cluster_mppi_kmeans", s,
                              map, model.dim_x, param.T, collision_checker,
                              param.x_init, param.x_target);
      solver.setVisLogger(&vis_logger);

      WmrobotGpuRunResult row;
      row.variant = variant;
      row.start_case = s;
      row.map_id = map;

      for (row.iter = 0; row.iter < maxiter; ++row.iter) {

        beginWmrobotGpuVisStep(vis_logger, save_rollouts, vis_every, row.iter);
        solver.solve();
        endWmrobotGpuVisStep(vis_logger);
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
  std::ofstream summary_csv("result_cluster_mppi_kmeans_summary.csv");
  writeWmrobotGpuSummaryHeader(summary_csv);
  writeWmrobotGpuSummaryRow(summary_csv, summary);
  return 0;
}
