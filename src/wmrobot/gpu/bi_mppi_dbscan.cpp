#include <bi_mppi_gpu.cuh>
#include <wmrobot_map.h>

#include "wmrobot_gpu_stats.h"
#include "wmrobot_gpu_vis.h"

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct BiMppiConfig {
  int Tf = 50;
  int Tb = 50;
  int Nf = 6000;
  int Nb = 6000;
  int Nr = 3000;
  int maxiter = 200;
  int map_begin = 299;
  int num_maps = 300;
  int start_cases = 2;
  int vis_every = 1;
  bool has_seed = false;
  std::uint_fast64_t seed = 0;
  ClusteringMethod clustering_method = ClusteringMethod::DBSCAN;
  int kmeans_clusters = 5;
  int kmeans_max_iterations = 100;
  double kmeans_threshold = 1e-6;
  std::string connection_metric = "se2";
  double se2_weight_xy = 1.0;
  double se2_weight_theta = 1.0;
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
      << "  --Tf N               Forward horizon. Default: 50\n"
      << "  --Tb N               Backward horizon. Default: 50\n"
      << "  --Nf N               Forward rollout count. Default: 10000\n"
      << "  --Nb N               Backward rollout count. Default: 10000\n"
      << "  --Nr N               Guide rollout count. Default: 5000\n"
      << "  --seed N             Fixed CUDA random seed\n"
      << "  --clustering NAME    dbscan or kmeans. Default: dbscan\n"
      << "  --kmeans-clusters N  K-means cluster count. Default: 5\n"
      << "  --kmeans-iters N     K-means max iterations. Default: 100\n"
      << "  --kmeans-threshold X K-means convergence threshold. Default: 1e-6\n"
      << "  --connection-metric NAME  euclidean or se2. Default: se2\n"
      << "  --se2-weight-xy W    SE(2) position weight. Default: 0.2\n"
      << "  --se2-weight-theta W SE(2) heading weight. Default: 1.0\n"
      << "  --dataset-dir DIR    BARN txt file directory\n"
      << "  --vis-every N        Save rollout data every N iterations. "
         "Default: 1\n"
      << "  --no-rollouts        Save CSV files only\n";
}

BiMppiConfig parseArgs(int argc, char **argv) {
  BiMppiConfig config;
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
      config.Tf = 20;
      config.Tb = 20;
      config.Nf = 256;
      config.Nb = 256;
      config.Nr = 128;
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
    } else if (key == "--Tf") {
      config.Tf = std::stoi(require_value(key));
    } else if (key == "--Tb") {
      config.Tb = std::stoi(require_value(key));
    } else if (key == "--Nf") {
      config.Nf = std::stoi(require_value(key));
    } else if (key == "--Nb") {
      config.Nb = std::stoi(require_value(key));
    } else if (key == "--Nr") {
      config.Nr = std::stoi(require_value(key));
    } else if (key == "--seed") {
      config.seed =
          static_cast<std::uint_fast64_t>(std::stoull(require_value(key)));
      config.has_seed = true;
    } else if (key == "--clustering") {
      config.clustering_method = parseClusteringMethod(require_value(key));
    } else if (key == "--kmeans-clusters") {
      config.kmeans_clusters = std::stoi(require_value(key));
    } else if (key == "--kmeans-iters") {
      config.kmeans_max_iterations = std::stoi(require_value(key));
    } else if (key == "--kmeans-threshold") {
      config.kmeans_threshold = std::stod(require_value(key));
    } else if (key == "--connection-metric") {
      config.connection_metric = require_value(key);
      if (config.connection_metric != "euclidean" &&
          config.connection_metric != "se2") {
        throw std::runtime_error(
            "--connection-metric must be either 'euclidean' or 'se2'");
      }
    } else if (key == "--se2-weight-xy") {
      config.se2_weight_xy = std::stod(require_value(key));
    } else if (key == "--se2-weight-theta") {
      config.se2_weight_theta = std::stod(require_value(key));
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
  if (config.kmeans_clusters <= 0 || config.kmeans_max_iterations <= 0 ||
      config.kmeans_threshold < 0.0)
    throw std::runtime_error(
        "K-means parameters must be positive (threshold may be zero)");
  if (config.se2_weight_xy < 0.0 || config.se2_weight_theta < 0.0 ||
      (config.se2_weight_xy == 0.0 && config.se2_weight_theta == 0.0))
    throw std::runtime_error(
        "SE(2) weights must be non-negative and not both zero");
  return config;
}

} // namespace

int main(int argc, char **argv) {
  const auto config = parseArgs(argc, argv);
  auto model = WMRobotMap();

  using Solver = BiMPPI_GPU;
  using SolverParam = BiMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = config.Tf;
  param.Tb = config.Tb;

  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;

  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;

  param.Nf = config.Nf;
  param.Nb = config.Nb;
  param.Nr = config.Nr;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;
  param.clustering_method = config.clustering_method;
  param.kmeans_clusters = config.kmeans_clusters;
  param.kmeans_max_iterations = config.kmeans_max_iterations;
  param.kmeans_threshold = config.kmeans_threshold;

  int maxiter = config.maxiter;

  const bool use_se2 = config.connection_metric == "se2";
  const std::string variant = use_se2 ? "BiC-MPPI-SE2" : "BiC-MPPI";
  const std::string output_prefix =
      use_se2 ? "result_bi_mppi_se2" : "result_bi_mppi";
  const std::string vis_prefix = use_se2 ? "bi_mppi_se2" : "bi_mppi";
  std::vector<WmrobotGpuRunResult> runs;

  std::ofstream csv(wmrobotGpuResultPath(output_prefix + ".csv"));
  writeWmrobotGpuRunHeader(csv);
  csv.flush();

  std::ofstream progress_csv(
      wmrobotGpuResultPath(output_prefix + "_progress.csv"));
  writeWmrobotGpuRunHeader(progress_csv);
  progress_csv.flush();

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
    // for (int map = 0; map < 300; ++map) {
    for (int map = config.map_begin;
         map >= 0 && map > config.map_begin - config.num_maps; --map) {
      CollisionChecker collision_checker = CollisionChecker();
      collision_checker.loadMap(
          config.dataset_dir + "/output_" + std::to_string(map) + ".txt", 0.1);

      Solver solver(model);
      if (config.has_seed)
        solver.setSeed(config.seed);
      solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
      solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
      solver.init(param);
      if (use_se2) {
        solver.setConnectionMetric(Solver::ConnectionMetric::SE2);
        solver.setSE2ConnectionWeights(config.se2_weight_xy,
                                       config.se2_weight_theta);
      }
      solver.setCollisionChecker(&collision_checker);

      MPPIVisLogger vis_logger;
      initWmrobotGpuVisLogger(vis_logger, config.save_rollouts, vis_prefix, s,
                              map, model.dim_x, param.Tf + param.Tb,
                              collision_checker, param.x_init, param.x_target);
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
        row.d_conn = solver.connectionDistance();
        solver.move();
        row.elapsed += solver.elapsed;
        row.elapsed_rollout += solver.elapsed_rollout;
        row.elapsed_clustering += solver.elapsed_clustering;
        row.elapsed_connection += solver.elapsed_connection;
        row.elapsed_guide += solver.elapsed_guide;

        // std::cout<<"1 solved in "<<solver.elapsed_1.count()<<std::endl;
        // std::cout<<"2 solved in "<<solver.elapsed_2.count()<<std::endl;
        // std::cout<<"3 solved in "<<solver.elapsed_3.count()<<std::endl;

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
      }
      if (row.iter >= maxiter) {
        row.iter = maxiter;
      }
      printWmrobotGpuRunRow(std::cout, row);
      writeWmrobotGpuRunRow(csv, row);
      csv.flush();
      runs.push_back(row);
    }
  }

  csv.close();
  progress_csv.close();

  const auto summary = summarizeWmrobotGpuRuns(variant, runs);
  std::ofstream summary_csv(
      wmrobotGpuResultPath(output_prefix + "_summary.csv"));
  writeWmrobotGpuSummaryHeader(summary_csv);
  writeWmrobotGpuSummaryRow(summary_csv, summary);
  return 0;
}
