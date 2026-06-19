#pragma once

#include <collision_checker.h>
#include <mppi_vis_logger.h>

#include <Eigen/Dense>

#include <iomanip>
#include <sstream>
#include <string>

inline std::string wmrobotGpuVisName(const std::string &solver_key,
                                     int start_case, int map_id) {
  std::ostringstream out;
  out << "wmrobot_" << solver_key << "_start_" << std::setw(2)
      << std::setfill('0') << start_case << "_map_" << std::setw(3)
      << std::setfill('0') << map_id;
  return out.str();
}

inline void initWmrobotGpuVisLogger(MPPIVisLogger &logger, bool save_rollouts,
                                    const std::string &solver_key,
                                    int start_case, int map_id, int dim_x,
                                    int T,
                                    const CollisionChecker &collision_checker,
                                    const Eigen::VectorXd &start,
                                    const Eigen::VectorXd &goal) {
  if (!save_rollouts) {
    return;
  }
  logger.enabled = true;
  logger.init(wmrobotGpuVisName(solver_key, start_case, map_id), dim_x, T,
              collision_checker.resolution, collision_checker.map, start,
              goal);
  logger.enabled = false;
}

inline bool shouldLogWmrobotGpuIteration(bool save_rollouts, int vis_every,
                                         int iter) {
  return save_rollouts && vis_every > 0 && (iter % vis_every == 0);
}

inline void beginWmrobotGpuVisStep(MPPIVisLogger &logger, bool save_rollouts,
                                   int vis_every, int iter) {
  logger.enabled = shouldLogWmrobotGpuIteration(save_rollouts, vis_every, iter);
  if (logger.enabled) {
    logger.beginStep(iter);
  }
}

inline void endWmrobotGpuVisStep(MPPIVisLogger &logger) {
  logger.enabled = false;
}
