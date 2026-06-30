#pragma once

#include "manipulator_grid_wall_task.h"

#include <Eigen/Dense>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// manipulator_grid_approach_task.h
// -----------------------------------------------------------------------------
// Grid-wall approach scenario.
//
// Difference from the previous random cell-to-cell scenario:
//   - The initial state is a fixed/free-space posture with the wrist/tool side
//     intentionally flipped away from the grid ("opposite-facing free pose").
//   - The goal is one randomly selected grid insertion cell.
//   - The nominal route is
//
//       free_opposite -> free_aligned -> goal_retract -> goal_insert
//
// Current limitation:
//   The existing ManipulatorDynamicsModel benchmark primarily controls
//   end-effector position through terminal/task costs. Tool orientation is
//   represented only implicitly by the selected joint posture. For a strict
//   SO(3) orientation benchmark, add FK rotation extraction and orientation cost.
// =============================================================================

namespace manipulator_grid_approach {

struct GridApproachConfig {
  manipulator_grid_wall::GridWallConfig grid_cfg;

  // A free-space staging point outside the grid wall, closer to the robot/base.
  // This point is used to generate an aligned pre-approach posture.
  double x_free_align = -0.06;

  // If true, print selected free/goal states.
  bool verbose = true;

  int max_goal_resample_attempts = 200;
};

struct GridApproachTask {
  int goal_id = -1;
  manipulator_grid_wall::GridCellPose goal_cell;

  Eigen::VectorXd q_free_opposite;
  Eigen::VectorXd q_free_aligned;
  Eigen::VectorXd q_goal_retract;
  Eigen::VectorXd q_goal_insert;

  Eigen::Vector3d p_free_opposite = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_free_aligned = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_goal_retract = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_goal_insert = Eigen::Vector3d::Zero();

  bool free_opposite_collision = false;
  bool free_aligned_collision = false;
  bool goal_retract_collision = false;
  bool goal_insert_collision = false;
};

inline std::vector<Eigen::VectorXd> makeFreeOppositePoseCandidates() {
  // These are free-space joint postures. The wrist joints are deliberately
  // flipped relative to the grid insertion posture family, so the arm must first
  // reorient in free space before approaching the grid.
  std::vector<Eigen::VectorXd> qs;
  qs.reserve(8);

  qs.push_back(manipulator_grid_wall::makeQ({ 2.70, -1.05, 1.10,  0.20, -0.95,  3.10}));
  qs.push_back(manipulator_grid_wall::makeQ({ 2.35, -1.20, 1.25,  0.35, -0.80,  2.80}));
  qs.push_back(manipulator_grid_wall::makeQ({-2.60, -1.05, 1.10, -0.20, -0.95, -3.10}));
  qs.push_back(manipulator_grid_wall::makeQ({-2.25, -1.20, 1.25, -0.35, -0.80, -2.80}));
  qs.push_back(manipulator_grid_wall::makeQ({ 1.95, -1.35, 1.35,  0.20, -0.70,  3.00}));
  qs.push_back(manipulator_grid_wall::makeQ({-1.95, -1.35, 1.35, -0.20, -0.70, -3.00}));
  qs.push_back(manipulator_grid_wall::makeQ({ 2.85, -0.90, 1.45,  0.45, -0.45,  3.14}));
  qs.push_back(manipulator_grid_wall::makeQ({-2.85, -0.90, 1.45, -0.45, -0.45, -3.14}));

  return qs;
}

inline Eigen::VectorXd selectFreeOppositePose(
    const ManipulatorDynamicsModel &model,
    double *best_collision_score = nullptr) {
  const auto candidates = makeFreeOppositePoseCandidates();

  Eigen::VectorXd best_q = candidates.front();
  double best_score = std::numeric_limits<double>::infinity();

  for (const auto &q_raw : candidates) {
    const Eigen::VectorXd q =
        manipulator_grid_wall::clampJointLimits(model, q_raw);
    const Eigen::VectorXd x =
        manipulator_grid_wall::makeZeroVelocityState(model, q);
    const bool coll = model.inWorkspaceCollision(x);

    // Score favors non-collision. If every candidate collides, use the first
    // least-bad proxy. A precise clearance score can be added in the model later.
    const double score = coll ? 1.0 : 0.0;
    if (score < best_score) {
      best_q = q;
      best_score = score;
    }
  }

  if (best_collision_score != nullptr) {
    *best_collision_score = best_score;
  }

  if (best_score > 0.0) {
    std::cerr << "[WARN] All free-opposite pose candidates collide under the "
              << "current workspace collision model. Consider distal-link-only "
              << "grid collision, smaller link radius, or moving the free pose.\n";
  }

  return best_q;
}

inline Eigen::Vector3d makeFreeAlignedPosition(
    const GridApproachConfig &cfg,
    const manipulator_grid_wall::GridCellPose &goal_cell) {
  return Eigen::Vector3d(cfg.x_free_align,
                         goal_cell.p_retract.y(),
                         goal_cell.p_retract.z());
}

inline GridApproachTask makeGridApproachTaskByGoal(
    const ManipulatorDynamicsModel &model,
    const std::vector<manipulator_grid_wall::GridCellPose> &cells,
    int goal_id,
    const GridApproachConfig &cfg = GridApproachConfig()) {
  manipulator_grid_wall::validateGridCells(cells);

  if (goal_id < 0 || goal_id >= static_cast<int>(cells.size())) {
    throw std::out_of_range("Grid approach goal cell index out of range.");
  }

  GridApproachTask task;
  task.goal_id = goal_id;
  task.goal_cell = cells[static_cast<std::size_t>(goal_id)];

  double free_score = 0.0;
  task.q_free_opposite = selectFreeOppositePose(model, &free_score);

  task.p_free_opposite = model.endEffectorPosition(
      manipulator_grid_wall::makeZeroVelocityState(model, task.q_free_opposite));

  task.p_free_aligned = makeFreeAlignedPosition(cfg, task.goal_cell);

  double ik_err_align = 0.0;
  bool align_free = false;
  task.q_free_aligned =
      manipulator_grid_wall::solveCollisionFreePositionIK(
          model, task.p_free_aligned, cfg.grid_cfg, &ik_err_align, &align_free);

  task.q_goal_retract = task.goal_cell.q_retract;
  task.q_goal_insert = task.goal_cell.q_insert;
  task.p_goal_retract = task.goal_cell.p_retract;
  task.p_goal_insert = task.goal_cell.p_insert;

  task.free_opposite_collision =
      model.inWorkspaceCollision(manipulator_grid_wall::makeZeroVelocityState(
          model, task.q_free_opposite));
  task.free_aligned_collision =
      model.inWorkspaceCollision(manipulator_grid_wall::makeZeroVelocityState(
          model, task.q_free_aligned));
  task.goal_retract_collision =
      model.inWorkspaceCollision(manipulator_grid_wall::makeZeroVelocityState(
          model, task.q_goal_retract));
  task.goal_insert_collision =
      model.inWorkspaceCollision(manipulator_grid_wall::makeZeroVelocityState(
          model, task.q_goal_insert));

  if (cfg.verbose) {
    std::cout << "\n=== Grid-approach task ===\n"
              << "goal_cell: " << task.goal_id
              << " (row=" << task.goal_cell.row
              << ", col=" << task.goal_cell.col << ")\n"
              << "p_free_opposite: [" << task.p_free_opposite.transpose() << "]\n"
              << "p_free_aligned : [" << task.p_free_aligned.transpose()
              << "], ik_err=" << ik_err_align
              << ", ik_free=" << static_cast<int>(align_free) << "\n"
              << "p_goal_retract : [" << task.p_goal_retract.transpose() << "]\n"
              << "p_goal_insert  : [" << task.p_goal_insert.transpose() << "]\n"
              << "free_opposite_coll: "
              << static_cast<int>(task.free_opposite_collision) << "\n"
              << "free_aligned_coll : "
              << static_cast<int>(task.free_aligned_collision) << "\n"
              << "goal_retract_coll : "
              << static_cast<int>(task.goal_retract_collision) << "\n"
              << "goal_insert_coll  : "
              << static_cast<int>(task.goal_insert_collision) << "\n";
  }

  return task;
}

inline bool gridApproachTaskKeyStatesCollisionFree(
    const GridApproachTask &task) {
  return !task.free_opposite_collision &&
         !task.free_aligned_collision &&
         !task.goal_retract_collision &&
         !task.goal_insert_collision;
}

inline GridApproachTask sampleValidGridApproachTask(
    const ManipulatorDynamicsModel &model,
    const std::vector<manipulator_grid_wall::GridCellPose> &cells,
    std::mt19937 &rng,
    const GridApproachConfig &cfg = GridApproachConfig()) {
  manipulator_grid_wall::validateGridCells(cells);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(cells.size()) - 1);

  for (int attempt = 0; attempt < cfg.max_goal_resample_attempts; ++attempt) {
    const int goal_id = dist(rng);
    GridApproachTask task = makeGridApproachTaskByGoal(model, cells, goal_id, cfg);
    if (gridApproachTaskKeyStatesCollisionFree(task)) {
      return task;
    }
  }

  std::cerr << "[WARN] Failed to sample a fully collision-free grid approach "
            << "task. Returning the last sampled goal for debugging.\n";
  return makeGridApproachTaskByGoal(model, cells, dist(rng), cfg);
}

inline Eigen::VectorXd makeFreeOppositeState(const ManipulatorDynamicsModel &model,
                                             const GridApproachTask &task) {
  return manipulator_grid_wall::makeZeroVelocityState(model,
                                                      task.q_free_opposite);
}

inline Eigen::VectorXd makeFreeAlignedState(const ManipulatorDynamicsModel &model,
                                            const GridApproachTask &task) {
  return manipulator_grid_wall::makeZeroVelocityState(model,
                                                      task.q_free_aligned);
}

inline Eigen::VectorXd makeGoalRetractState(const ManipulatorDynamicsModel &model,
                                            const GridApproachTask &task) {
  return manipulator_grid_wall::makeZeroVelocityState(model,
                                                      task.q_goal_retract);
}

inline Eigen::VectorXd makeGoalInsertState(const ManipulatorDynamicsModel &model,
                                           const GridApproachTask &task) {
  return manipulator_grid_wall::makeZeroVelocityState(model,
                                                      task.q_goal_insert);
}

inline Eigen::MatrixXd makeGridApproachWarmStart(
    const ManipulatorDynamicsModel &model,
    const GridApproachTask &task,
    int horizon,
    double dt,
    double kp,
    double kd) {
  const Eigen::VectorXd x0 = makeFreeOppositeState(model, task);
  const Eigen::VectorXd x1 = makeFreeAlignedState(model, task);
  const Eigen::VectorXd x2 = makeGoalRetractState(model, task);
  const Eigen::VectorXd x3 = makeGoalInsertState(model, task);

  const int n1 = std::max(1, horizon / 3);
  const int n2 = std::max(1, horizon / 3);
  const int n3 = std::max(1, horizon - n1 - n2);

  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(model.dim_u, horizon);
  int offset = 0;

  const Eigen::MatrixXd U1 = makePdTorqueWarmStart(model, x0, x1, n1, dt, kp, kd);
  U.middleCols(offset, n1) = U1.leftCols(n1);
  offset += n1;

  const Eigen::MatrixXd U2 = makePdTorqueWarmStart(model, x1, x2, n2, dt, kp, kd);
  U.middleCols(offset, n2) = U2.leftCols(n2);
  offset += n2;

  const Eigen::MatrixXd U3 = makePdTorqueWarmStart(model, x2, x3, n3, dt, kp, kd);
  U.middleCols(offset, n3) = U3.leftCols(n3);

  return U;
}

inline Eigen::MatrixXd makeGridApproachReverseWarmStart(
    const ManipulatorDynamicsModel &model,
    const GridApproachTask &task,
    int horizon,
    double dt,
    double kp,
    double kd) {
  const Eigen::VectorXd x0 = makeGoalInsertState(model, task);
  const Eigen::VectorXd x1 = makeGoalRetractState(model, task);
  const Eigen::VectorXd x2 = makeFreeAlignedState(model, task);
  const Eigen::VectorXd x3 = makeFreeOppositeState(model, task);

  const int n1 = std::max(1, horizon / 3);
  const int n2 = std::max(1, horizon / 3);
  const int n3 = std::max(1, horizon - n1 - n2);

  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(model.dim_u, horizon);
  int offset = 0;

  const Eigen::MatrixXd U1 = makePdTorqueWarmStart(model, x0, x1, n1, dt, kp, kd);
  U.middleCols(offset, n1) = U1.leftCols(n1);
  offset += n1;

  const Eigen::MatrixXd U2 = makePdTorqueWarmStart(model, x1, x2, n2, dt, kp, kd);
  U.middleCols(offset, n2) = U2.leftCols(n2);
  offset += n2;

  const Eigen::MatrixXd U3 = makePdTorqueWarmStart(model, x2, x3, n3, dt, kp, kd);
  U.middleCols(offset, n3) = U3.leftCols(n3);

  return U;
}

inline void writeGridApproachMetadataHeader(std::ofstream &ofs) {
  ofs << "seed,goal_cell,goal_row,goal_col,"
         "free_opposite_x,free_opposite_y,free_opposite_z,"
         "free_aligned_x,free_aligned_y,free_aligned_z,"
         "goal_retract_x,goal_retract_y,goal_retract_z,"
         "goal_insert_x,goal_insert_y,goal_insert_z,"
         "free_opposite_collision,free_aligned_collision,"
         "goal_retract_collision,goal_insert_collision\n";
}

inline void writeGridApproachMetadataRow(std::ofstream &ofs,
                                         unsigned int seed,
                                         const GridApproachTask &task) {
  ofs << seed << ","
      << task.goal_id << ","
      << task.goal_cell.row << ","
      << task.goal_cell.col << ","
      << task.p_free_opposite.x() << "," << task.p_free_opposite.y()
      << "," << task.p_free_opposite.z() << ","
      << task.p_free_aligned.x() << "," << task.p_free_aligned.y()
      << "," << task.p_free_aligned.z() << ","
      << task.p_goal_retract.x() << "," << task.p_goal_retract.y()
      << "," << task.p_goal_retract.z() << ","
      << task.p_goal_insert.x() << "," << task.p_goal_insert.y()
      << "," << task.p_goal_insert.z() << ","
      << static_cast<int>(task.free_opposite_collision) << ","
      << static_cast<int>(task.free_aligned_collision) << ","
      << static_cast<int>(task.goal_retract_collision) << ","
      << static_cast<int>(task.goal_insert_collision) << "\n";
}

}  // namespace manipulator_grid_approach
