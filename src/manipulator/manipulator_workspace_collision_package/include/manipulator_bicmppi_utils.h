#pragma once

#include "manipulator_dynamics_model.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

inline Eigen::VectorXd clampVector(const Eigen::VectorXd &v,
                                   const Eigen::VectorXd &lo,
                                   const Eigen::VectorXd &hi) {
  return v.cwiseMax(lo).cwiseMin(hi);
}

inline Eigen::MatrixXd makePdTorqueWarmStart(const ManipulatorDynamicsModel &model,
                                             const Eigen::VectorXd &x0,
                                             const Eigen::VectorXd &x_goal,
                                             int T, double dt,
                                             double kp = 18.0,
                                             double kd = 7.0) {
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(model.dim_u, T);
  Eigen::VectorXd x = x0;
  for (int t = 0; t < T; ++t) {
    const Eigen::VectorXd q = x.head(ManipulatorDynamicsModel::kDof);
    const Eigen::VectorXd qd = x.segment(ManipulatorDynamicsModel::kDof,
                                         ManipulatorDynamicsModel::kDof);
    const Eigen::VectorXd q_goal = x_goal.head(ManipulatorDynamicsModel::kDof);
    Eigen::VectorXd tau = kp * (q_goal - q) - kd * qd;
    tau = clampVector(tau, -model.tau_max, model.tau_max);
    U.col(t) = tau;
    x = model.rk4Step(x, tau, dt);
  }
  return U;
}

inline Eigen::MatrixXd makeWaypointPdTorqueWarmStart(
    const ManipulatorDynamicsModel &model, const Eigen::VectorXd &x0,
    const Eigen::VectorXd &x_waypoint, const Eigen::VectorXd &x_goal, int T,
    double dt, double kp = 18.0, double kd = 7.0) {
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(model.dim_u, T);
  Eigen::VectorXd x = x0;
  const int waypoint_steps = std::max(1, T / 2);
  for (int t = 0; t < T; ++t) {
    const Eigen::VectorXd q = x.head(ManipulatorDynamicsModel::kDof);
    const Eigen::VectorXd qd = x.segment(ManipulatorDynamicsModel::kDof,
                                         ManipulatorDynamicsModel::kDof);
    const Eigen::VectorXd q_ref =
        (t < waypoint_steps ? x_waypoint : x_goal)
            .head(ManipulatorDynamicsModel::kDof);
    Eigen::VectorXd tau = kp * (q_ref - q) - kd * qd;
    tau = clampVector(tau, -model.tau_max, model.tau_max);
    U.col(t) = tau;
    x = model.rk4Step(x, tau, dt);
  }
  return U;
}

inline void writeMatrixCsv(const std::string &path, const Eigen::MatrixXd &M,
                           const std::string &prefix) {
  std::ofstream ofs(path);
  if (!ofs) {
    throw std::runtime_error("failed to open CSV: " + path);
  }
  ofs << "index";
  for (int r = 0; r < M.rows(); ++r) {
    ofs << "," << prefix << (r + 1);
  }
  ofs << "\n";
  ofs << std::setprecision(12);
  for (int c = 0; c < M.cols(); ++c) {
    ofs << c;
    for (int r = 0; r < M.rows(); ++r) {
      ofs << "," << M(r, c);
    }
    ofs << "\n";
  }
}

inline void writeManipulatorSummaryHeader(std::ofstream &ofs) {
  ofs << "iter,solver_elapsed,total_solver_elapsed,connection_distance,"
         "q_error,ee_error,qdot_norm,tau_norm,workspace_collision,"
         "q1,q2,q3,q4,q5,q6,qd1,qd2,qd3,qd4,qd5,qd6\n";
}

template <typename Solver>
inline void appendManipulatorSummaryRow(std::ofstream &ofs, int iter,
                                        double total_solver_elapsed,
                                        const Solver &solver,
                                        const ManipulatorDynamicsModel &model,
                                        const Eigen::VectorXd &x_goal) {
  const Eigen::VectorXd x = solver.x_init;
  const Eigen::VectorXd q = x.head(ManipulatorDynamicsModel::kDof);
  const Eigen::VectorXd q_goal = x_goal.head(ManipulatorDynamicsModel::kDof);
  const double q_error = (q - q_goal).norm();
  const double ee_error =
      (model.endEffectorPosition(x) - model.endEffectorPosition(x_goal)).norm();
  const double qdot_norm = x.segment(ManipulatorDynamicsModel::kDof,
                                     ManipulatorDynamicsModel::kDof).norm();
  const double tau_norm = solver.u0.norm();
  const bool collision = model.inWorkspaceCollision(x);

  ofs << std::setprecision(12) << iter << "," << solver.elapsed << ","
      << total_solver_elapsed << "," << solver.connectionDistance() << ","
      << q_error << "," << ee_error << "," << qdot_norm << "," << tau_norm
      << "," << static_cast<int>(collision);
  for (int i = 0; i < x.size(); ++i) {
    ofs << "," << x(i);
  }
  ofs << "\n";
}
