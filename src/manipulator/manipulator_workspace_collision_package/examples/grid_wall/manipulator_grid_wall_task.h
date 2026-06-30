#pragma once

#include "collision_checker.h"
#include "manipulator_dynamics_model.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace manipulator_grid_wall {

struct GridWallConfig {
  double x_insert = -0.56;
  double x_retract = -0.33;

  std::array<double, 3> y_centers{{-0.36, -0.16, 0.04}};
  std::array<double, 3> z_centers{{0.18, 0.36, 0.54}};

  double x_wall_min = -0.62;
  double x_wall_max = -0.42;

  double wall_thickness = 0.035;
  double wall_y_min = -0.46;
  double wall_y_max = 0.14;
  double wall_z_min = 0.09;
  double wall_z_max = 0.63;
  double wall_padding = 0.0;

  int ik_max_iter = 160;
  double ik_tol = 2.5e-3;
  double ik_damping = 2.0e-3;
  double ik_step_limit = 0.12;
  double ik_fd_eps = 1.0e-4;
  double ik_posture_reg = 2.0e-3;

  bool verbose = true;
};

struct WorkspaceBox {
  std::string name;
  std::array<double, 6> box;
};

struct GridCellPose {
  int id = -1;
  int row = -1;
  int col = -1;

  Eigen::Vector3d p_insert = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_retract = Eigen::Vector3d::Zero();

  Eigen::VectorXd q_insert;
  Eigen::VectorXd q_retract;

  double ik_error_insert = 0.0;
  double ik_error_retract = 0.0;
};

inline Eigen::VectorXd makeQ(std::initializer_list<double> values) {
  Eigen::VectorXd q(static_cast<int>(values.size()));
  int i = 0;
  for (double v : values) {
    q(i++) = v;
  }
  return q;
}

inline double clampScalar(double x, double lo, double hi) {
  return std::max(lo, std::min(x, hi));
}

inline Eigen::VectorXd clampJointLimits(const ManipulatorDynamicsModel &model,
                                        const Eigen::VectorXd &q_in) {
  Eigen::VectorXd q = q_in;
  if (model.q_min.size() == q.size() && model.q_max.size() == q.size()) {
    for (int i = 0; i < q.size(); ++i) {
      q(i) = clampScalar(q(i), model.q_min(i), model.q_max(i));
    }
  }
  return q;
}

inline Eigen::VectorXd makeZeroVelocityState(
    const ManipulatorDynamicsModel &model,
    const Eigen::VectorXd &q) {
  return model.makeState(q, Eigen::VectorXd::Zero(ManipulatorDynamicsModel::kDof));
}

inline Eigen::Matrix<double, 3, 6> finiteDifferenceEEJacobian(
    const ManipulatorDynamicsModel &model,
    const Eigen::VectorXd &q,
    double eps) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;
  Eigen::Matrix<double, 3, 6> J;
  J.setZero();

  for (int j = 0; j < dof; ++j) {
    Eigen::VectorXd qp = q;
    Eigen::VectorXd qm = q;
    qp(j) += eps;
    qm(j) -= eps;

    const Eigen::Vector3d pp =
        model.endEffectorPosition(makeZeroVelocityState(model, qp));
    const Eigen::Vector3d pm =
        model.endEffectorPosition(makeZeroVelocityState(model, qm));
    J.col(j) = (pp - pm) / (2.0 * eps);
  }

  return J;
}

inline Eigen::VectorXd solvePositionIKDampedLeastSquares(
    const ManipulatorDynamicsModel &model,
    const Eigen::Vector3d &target_pos,
    const Eigen::VectorXd &q_seed,
    const GridWallConfig &cfg,
    double *final_error = nullptr) {
  Eigen::VectorXd q = clampJointLimits(model, q_seed);
  const Eigen::VectorXd q_ref = q;

  for (int iter = 0; iter < cfg.ik_max_iter; ++iter) {
    const Eigen::VectorXd x = makeZeroVelocityState(model, q);
    const Eigen::Vector3d e = target_pos - model.endEffectorPosition(x);
    if (e.norm() < cfg.ik_tol) {
      break;
    }

    const Eigen::Matrix<double, 3, 6> J =
        finiteDifferenceEEJacobian(model, q, cfg.ik_fd_eps);
    Eigen::Matrix3d A = J * J.transpose();
    A.diagonal().array() += cfg.ik_damping * cfg.ik_damping;

    Eigen::VectorXd dq = J.transpose() * A.ldlt().solve(e);
    dq -= cfg.ik_posture_reg * (q - q_ref);
    const double n = dq.norm();
    if (n > cfg.ik_step_limit) {
      dq *= cfg.ik_step_limit / n;
    }

    q = clampJointLimits(model, q + dq);
  }

  const double err =
      (target_pos -
       model.endEffectorPosition(makeZeroVelocityState(model, q))).norm();
  if (final_error != nullptr) {
    *final_error = err;
  }
  return q;
}

inline std::vector<Eigen::VectorXd> defaultIkSeeds() {
  std::vector<Eigen::VectorXd> seeds;
  seeds.reserve(8);
  seeds.push_back(makeQ({0.05, -1.15, 1.25, 0.00, 0.85, 0.00}));
  seeds.push_back(makeQ({0.30, -1.00, 1.10, 0.00, 0.75, 0.00}));
  seeds.push_back(makeQ({-0.30, -1.00, 1.10, 0.00, 0.75, 0.00}));
  seeds.push_back(makeQ({0.65, -1.25, 1.35, 0.15, 0.70, 0.00}));
  seeds.push_back(makeQ({-0.65, -1.25, 1.35, -0.15, 0.70, 0.00}));
  seeds.push_back(makeQ({1.00, -1.05, 1.45, 0.20, 0.55, 0.00}));
  seeds.push_back(makeQ({-1.00, -1.05, 1.45, -0.20, 0.55, 0.00}));
  seeds.push_back(makeQ({0.00, -1.45, 1.55, 0.00, 0.45, 0.00}));
  return seeds;
}

inline Eigen::VectorXd solveCollisionFreePositionIK(
    const ManipulatorDynamicsModel &model,
    const Eigen::Vector3d &target_pos,
    const GridWallConfig &cfg,
    double *final_error = nullptr,
    bool *collision_free = nullptr) {
  Eigen::VectorXd best_q;
  double best_score = std::numeric_limits<double>::infinity();
  double best_err = std::numeric_limits<double>::infinity();
  bool best_free = false;

  for (const auto &seed : defaultIkSeeds()) {
    double err = 0.0;
    Eigen::VectorXd q =
        solvePositionIKDampedLeastSquares(model, target_pos, seed, cfg, &err);
    const Eigen::VectorXd x = makeZeroVelocityState(model, q);
    const bool coll = model.inWorkspaceCollision(x);
    const double score = err + (coll ? 10.0 : 0.0);
    if (score < best_score) {
      best_q = q;
      best_score = score;
      best_err = err;
      best_free = !coll;
    }
  }

  if (best_q.size() == 0) {
    throw std::runtime_error("failed to solve grid-wall IK");
  }
  if (final_error != nullptr) {
    *final_error = best_err;
  }
  if (collision_free != nullptr) {
    *collision_free = best_free;
  }
  return best_q;
}

inline std::vector<WorkspaceBox> makeGridWallWorkspaceBoxes(
    const GridWallConfig &cfg = GridWallConfig()) {
  const double y_split_01 = 0.5 * (cfg.y_centers[0] + cfg.y_centers[1]);
  const double y_split_12 = 0.5 * (cfg.y_centers[1] + cfg.y_centers[2]);
  const double z_split_01 = 0.5 * (cfg.z_centers[0] + cfg.z_centers[1]);
  const double z_split_12 = 0.5 * (cfg.z_centers[1] + cfg.z_centers[2]);
  const double h = 0.5 * cfg.wall_thickness;
  const double pad = cfg.wall_padding;

  std::vector<WorkspaceBox> boxes;
  boxes.reserve(4);
  boxes.push_back({"grid_wall_vertical_01",
                   {cfg.x_wall_min, cfg.x_wall_max,
                    y_split_01 - h - pad, y_split_01 + h + pad,
                    cfg.wall_z_min, cfg.wall_z_max}});
  boxes.push_back({"grid_wall_vertical_12",
                   {cfg.x_wall_min, cfg.x_wall_max,
                    y_split_12 - h - pad, y_split_12 + h + pad,
                    cfg.wall_z_min, cfg.wall_z_max}});
  boxes.push_back({"grid_wall_horizontal_01",
                   {cfg.x_wall_min, cfg.x_wall_max,
                    cfg.wall_y_min, cfg.wall_y_max,
                    z_split_01 - h - pad, z_split_01 + h + pad}});
  boxes.push_back({"grid_wall_horizontal_12",
                   {cfg.x_wall_min, cfg.x_wall_max,
                    cfg.wall_y_min, cfg.wall_y_max,
                    z_split_12 - h - pad, z_split_12 + h + pad}});
  return boxes;
}

inline void registerGridWallWorkspace(
    ManipulatorDynamicsModel &model,
    CollisionChecker &cc,
    const GridWallConfig &cfg = GridWallConfig()) {
  const auto boxes = makeGridWallWorkspaceBoxes(cfg);
  for (const auto &b : boxes) {
    model.addWorkspaceBoxMinMax(b.box[0], b.box[1], b.box[2], b.box[3],
                                b.box[4], b.box[5]);
    cc.addWorkspaceBoxMinMax(b.box[0], b.box[1], b.box[2], b.box[3],
                             b.box[4], b.box[5]);
  }
  cc.use_workspace_link_collision = true;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
}

inline Eigen::Vector3d cellInsertPosition(const GridWallConfig &cfg,
                                          int row,
                                          int col) {
  return Eigen::Vector3d(cfg.x_insert, cfg.y_centers[col], cfg.z_centers[row]);
}

inline Eigen::Vector3d cellRetractPosition(const GridWallConfig &cfg,
                                           int row,
                                           int col) {
  return Eigen::Vector3d(cfg.x_retract, cfg.y_centers[col], cfg.z_centers[row]);
}

inline std::vector<GridCellPose> makeGridCellPoseSet9(
    const ManipulatorDynamicsModel &model,
    const GridWallConfig &cfg = GridWallConfig()) {
  std::vector<GridCellPose> cells;
  cells.reserve(9);

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      GridCellPose cell;
      cell.id = 3 * row + col;
      cell.row = row;
      cell.col = col;
      cell.p_insert = cellInsertPosition(cfg, row, col);
      cell.p_retract = cellRetractPosition(cfg, row, col);

      double err_retract = 0.0;
      bool retract_free = false;
      cell.q_retract = solveCollisionFreePositionIK(
          model, cell.p_retract, cfg, &err_retract, &retract_free);

      double err_insert = 0.0;
      cell.q_insert = solvePositionIKDampedLeastSquares(
          model, cell.p_insert, cell.q_retract, cfg, &err_insert);

      cell.ik_error_retract = err_retract;
      cell.ik_error_insert = err_insert;
      cells.push_back(cell);
    }
  }

  if (cfg.verbose) {
    std::cout << "\n=== 3x3 grid-wall cell pose set ===\n";
    for (const auto &c : cells) {
      std::cout << "cell=" << c.id << " row=" << c.row << " col=" << c.col
                << " p_insert=[" << c.p_insert.transpose() << "]"
                << " p_retract=[" << c.p_retract.transpose() << "]"
                << " ik_insert=" << c.ik_error_insert
                << " ik_retract=" << c.ik_error_retract << "\n";
    }
  }

  return cells;
}

inline void validateGridCells(const std::vector<GridCellPose> &cells) {
  if (cells.size() != 9) {
    throw std::runtime_error("grid-wall benchmark must contain exactly 9 cells");
  }
  for (std::size_t i = 0; i < cells.size(); ++i) {
    if (cells[i].id != static_cast<int>(i)) {
      throw std::runtime_error("grid cell id must match its vector index");
    }
  }
}

inline void printGridWallBoxes(const std::vector<WorkspaceBox> &boxes) {
  std::cout << "\n=== Grid-wall workspace boxes ===\n";
  for (const auto &b : boxes) {
    std::cout << std::setw(24) << b.name << "  ["
              << b.box[0] << ", " << b.box[1] << ", "
              << b.box[2] << ", " << b.box[3] << ", "
              << b.box[4] << ", " << b.box[5] << "]\n";
  }
}

}  // namespace manipulator_grid_wall
