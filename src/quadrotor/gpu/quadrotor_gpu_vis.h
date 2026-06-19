#pragma once

#include <collision_checker.h>
#include <mppi_vis_logger.h>

#include <Eigen/Dense>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

struct QuadrotorGpuConfig {
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
  int vis_every = 1;
  bool save_rollouts = true;
  std::string dataset_dir = "../BARN_dataset/txt_files";
};

inline bool quadrotorGpuLooksLikeOption(const std::string &value) {
  return value.rfind("--", 0) == 0;
}

inline void printQuadrotorGpuUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [options]\n"
            << "Options:\n"
            << "  --smoke              Run one tiny validation case\n"
            << "  --map-begin N        First BARN map id. Default: 299\n"
            << "  --num-maps N         Number of maps descending from map-begin. Default: 300\n"
            << "  --maxiter N          Max closed-loop iterations. Default: 200\n"
            << "  --T N                One-direction horizon. Default: 100\n"
            << "  --Tf N               Forward horizon. Default: 50\n"
            << "  --Tb N               Backward horizon. Default: 50\n"
            << "  --N N                One-direction rollout count. Default: 10000\n"
            << "  --Nf N               Forward rollout count. Default: 10000\n"
            << "  --Nb N               Backward rollout count. Default: 10000\n"
            << "  --Nr N               Guide rollout count. Default: 5000\n"
            << "  --Ns N               SVGD surrogate samples. Default: 10\n"
            << "  --istep N            SVGD inner iterations. Default: 5\n"
            << "  --dataset-dir DIR    BARN txt file directory\n"
            << "  --vis-every N        Save rollout data every N iterations. Default: 1\n"
            << "  --no-rollouts        Save CSV files only\n";
}

inline QuadrotorGpuConfig parseQuadrotorGpuArgs(int argc, char **argv,
                                                bool svgd_defaults = false) {
  QuadrotorGpuConfig config;
  if (svgd_defaults) {
    config.Nf = 200;
    config.Nb = 200;
  }
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc || quadrotorGpuLooksLikeOption(argv[i + 1])) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--help" || key == "-h") {
      printQuadrotorGpuUsage(argv[0]);
      std::exit(0);
    } else if (key == "--smoke") {
      config.T = 20;
      config.Tf = 20;
      config.Tb = 20;
      config.N = 256;
      config.Nf = 128;
      config.Nb = 128;
      config.Nr = 128;
      config.Ns = 4;
      config.istep = 2;
      config.maxiter = 2;
      config.num_maps = 1;
      config.vis_every = 1;
    } else if (key == "--map-begin") {
      config.map_begin = std::stoi(require_value(key));
    } else if (key == "--num-maps") {
      config.num_maps = std::stoi(require_value(key));
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

inline std::string quadrotorGpuVisName(const std::string &solver_key,
                                       int map_id,
                                       const std::string &prefix =
                                           "quadrotor") {
  std::ostringstream out;
  out << prefix << "_" << solver_key << "_map_" << std::setw(3)
      << std::setfill('0') << map_id;
  return out.str();
}

inline void initQuadrotorGpuVisLogger(
    MPPIVisLogger &logger, bool save_rollouts, const std::string &solver_key,
    int map_id, int dim_x, int T, const CollisionChecker &collision_checker,
    const Eigen::VectorXd &start, const Eigen::VectorXd &goal,
    const std::string &prefix = "quadrotor") {
  if (!save_rollouts) {
    return;
  }
  logger.enabled = true;
  logger.init(quadrotorGpuVisName(solver_key, map_id, prefix), dim_x, T,
              collision_checker.resolution, collision_checker.map, start,
              goal);
  logger.enabled = false;
}

inline bool shouldLogQuadrotorGpuIteration(bool save_rollouts, int vis_every,
                                           int iter) {
  return save_rollouts && vis_every > 0 && (iter % vis_every == 0);
}

inline void beginQuadrotorGpuVisStep(MPPIVisLogger &logger,
                                     bool save_rollouts, int vis_every,
                                     int iter) {
  logger.enabled = shouldLogQuadrotorGpuIteration(save_rollouts, vis_every,
                                                  iter);
  if (logger.enabled) {
    logger.beginStep(iter);
  }
}

inline void endQuadrotorGpuVisStep(MPPIVisLogger &logger) {
  logger.enabled = false;
}
