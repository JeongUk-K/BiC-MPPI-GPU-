#pragma once

#include "collision_checker.h"
#include "manipulator_bicmppi_utils.h"
#include "manipulator_dynamics_model.h"
#include "manipulator_pick_place_task.h"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct PinkNPlaceMppiConfig {
  std::string solver_label;
  std::string csv_prefix;
  std::uint_fast64_t seed = 260623ULL;

  float dt = 0.02f;
  int horizon = 95;
  int samples = 1024;
  double gamma_u = 0.0012;

  double sigma_q1 = 5.0;
  double sigma_q2 = 5.0;
  double sigma_q3 = 4.2;
  double sigma_q4 = 2.5;
  double sigma_q5 = 2.0;
  double sigma_q6 = 1.4;

  double warm_start_kp = 20.0;
  double warm_start_kd = 7.5;
  int max_total_steps = 650;
};

struct PinkNPlaceBiMppiConfig {
  std::string solver_label = "BiC-MPPI";
  std::string csv_prefix = "manipulator_pinkNplace_bicmppi_";
  std::uint_fast64_t seed = 260626ULL;

  float dt = 0.02f;
  int Tf = 95;
  int Tb = 95;
  int Nf = 1024;
  int Nb = 1024;
  int Nr = 1024;
  double gamma_u = 0.0012;

  double sigma_q1 = 5.0;
  double sigma_q2 = 5.0;
  double sigma_q3 = 4.2;
  double sigma_q4 = 2.5;
  double sigma_q5 = 2.0;
  double sigma_q6 = 1.4;

  double deviation_mu = 1.0;
  double cost_mu = 1.0;
  double epsilon = 4.2;
  int minpts = 8;
  double psi = 0.0;

  double warm_start_kp = 20.0;
  double warm_start_kd = 7.5;
  int max_total_steps = 650;
};

inline void applyPinkNPlaceModelDefaults(ManipulatorDynamicsModel &model) {
  model.w_tau = 1.0e-3;
  model.w_qdot = 2.0e-2;
  model.w_joint_limit = 20.0;
  model.w_workspace_obs = 2.0e3;
  model.w_ground = 2.0e4;
  model.w_terminal_q = 1.5e3;
  model.w_terminal_ee = 5.0e2;
  model.w_terminal_qdot = 1.0e2;
  model.link_radius = 0.045;
  model.obs_safe_margin = 0.10;
  model.hard_collision_margin = 0.02;
}

inline void writePinkNPlacePhaseSummaryHeader(std::ofstream &ofs) {
  ofs << "global_iter,phase_index,phase_name,phase_iter,solver_elapsed,"
         "total_solver_elapsed,connection_distance,q_error,ee_error,"
         "qdot_norm,tau_norm,workspace_collision,ee_x,ee_y,ee_z,"
         "target_ee_x,target_ee_y,target_ee_z,gripper_event\n";
}

template <typename Solver>
inline void appendPinkNPlacePhaseSummaryRow(
    std::ofstream &ofs, int global_iter, int phase_index,
    const PickPlacePhase &phase, int phase_iter, double total_solver_elapsed,
    const Solver &solver, const ManipulatorDynamicsModel &model,
    const std::string &event) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;
  const Eigen::VectorXd x = solver.x_init;
  const Eigen::VectorXd x_goal = phase.x_goal;
  const double q_error = (x.head(dof) - x_goal.head(dof)).norm();
  const double ee_error =
      (model.endEffectorPosition(x) - model.endEffectorPosition(x_goal)).norm();
  const double qdot_norm = x.segment(dof, dof).norm();
  const double tau_norm = solver.u0.norm();
  const bool collision = model.inWorkspaceCollision(x);
  const Eigen::Vector3d ee = model.endEffectorPosition(x);
  const Eigen::Vector3d target_ee = model.endEffectorPosition(x_goal);

  ofs << std::setprecision(12) << global_iter << "," << phase_index << ","
      << phase.name << "," << phase_iter << "," << solver.elapsed << ","
      << total_solver_elapsed << "," << solver.connectionDistance() << ","
      << q_error << "," << ee_error << "," << qdot_norm << "," << tau_norm
      << "," << static_cast<int>(collision) << "," << ee.x() << "," << ee.y()
      << "," << ee.z() << "," << target_ee.x() << "," << target_ee.y()
      << "," << target_ee.z() << "," << event << "\n";
}

inline void printPinkNPlaceWorkspaceBoxes() {
  std::cout << "\n=== pinkNplace workspace obstacles [base frame, m] ===\n";
  for (const auto &b : makePickPlaceWorkspaceBoxes()) {
    std::cout << std::setw(22) << b.name << "  x:[" << b.xmin << ", "
              << b.xmax << "] y:[" << b.ymin << ", " << b.ymax << "] z:["
              << b.zmin << ", " << b.zmax << "]\n";
  }
}

inline void printPinkNPlacePhaseTargets(
    const ManipulatorDynamicsModel &model,
    const std::vector<PickPlacePhase> &phases) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;
  std::cout << "\n=== pinkNplace target postures and FK EE positions ===\n";
  for (std::size_t i = 0; i < phases.size(); ++i) {
    const auto &phase = phases[i];
    const Eigen::Vector3d ee = model.endEffectorPosition(phase.x_goal);
    std::cout << i << "  " << std::setw(24) << phase.name << "  q=[";
    for (int j = 0; j < dof; ++j) {
      std::cout << phase.x_goal(j) << (j + 1 == dof ? "" : ", ");
    }
    std::cout << "]  ee=[" << ee.x() << ", " << ee.y() << ", " << ee.z()
              << "]";
    if (!phase.gripper_event_before.empty()) {
      std::cout << "  before:" << phase.gripper_event_before;
    }
    if (!phase.gripper_event_after.empty()) {
      std::cout << "  after:" << phase.gripper_event_after;
    }
    std::cout << "\n";
  }
}

template <typename Solver, typename ConfigureSolver>
int runPinkNPlaceMppiLikeExample(const PinkNPlaceMppiConfig &cfg,
                                 ConfigureSolver configure_solver) {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;

  Model model;
  applyPinkNPlaceModelDefaults(model);
  CollisionChecker cc;
  registerPickPlaceWorkspace(model, cc);

  const Eigen::VectorXd x_home = makePickPlaceHomeState(model);
  const auto phases = makePickPlacePhases(model);

  printPinkNPlaceWorkspaceBoxes();
  printPinkNPlacePhaseTargets(model, phases);

  MPPIParam param;
  param.dt = cfg.dt;
  param.T = cfg.horizon;
  param.N = cfg.samples;
  param.gamma_u = cfg.gamma_u;
  param.x_init = x_home;
  param.x_target = phases.front().x_goal;
  param.sigma_u = Eigen::MatrixXd::Zero(dof, dof);
  param.sigma_u.diagonal() << cfg.sigma_q1, cfg.sigma_q2, cfg.sigma_q3,
      cfg.sigma_q4, cfg.sigma_q5, cfg.sigma_q6;

  Solver solver(model);
  configure_solver(solver);
  solver.U_0 = makePdTorqueWarmStart(model, x_home, phases.front().x_goal,
                                     param.T, param.dt, cfg.warm_start_kp,
                                     cfg.warm_start_kd);
  solver.init(param);
  solver.setCollisionChecker(&cc);
  solver.setSeed(cfg.seed);

  std::ofstream summary(cfg.csv_prefix + "summary.csv");
  writePinkNPlacePhaseSummaryHeader(summary);

  Eigen::MatrixXd executed_x =
      Eigen::MatrixXd::Zero(model.dim_x, cfg.max_total_steps + 1);
  Eigen::MatrixXd executed_u =
      Eigen::MatrixXd::Zero(model.dim_u, cfg.max_total_steps);
  executed_x.col(0) = solver.x_init;

  int global_iter = 0;
  double total_solver_elapsed = 0.0;
  bool all_phases_success = true;

  for (std::size_t phase_idx = 0; phase_idx < phases.size(); ++phase_idx) {
    const auto &phase = phases[phase_idx];
    solver.x_target = phase.x_goal;
    solver.U_0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                       param.T, param.dt, cfg.warm_start_kp,
                                       cfg.warm_start_kd);

    if (!phase.gripper_event_before.empty()) {
      appendPinkNPlacePhaseSummaryRow(
          summary, global_iter, static_cast<int>(phase_idx), phase, -1,
          total_solver_elapsed, solver, model, phase.gripper_event_before);
      std::cout << "[gripper before] " << phase.name << ": "
                << phase.gripper_event_before << "\n";
    }

    bool phase_success = false;
    for (int iter = 0; iter < phase.max_iter &&
                       global_iter < cfg.max_total_steps;
         ++iter) {
      solver.solve();
      total_solver_elapsed += solver.elapsed;

      const Eigen::VectorXd x_now = solver.x_init;
      const double q_error = (x_now.head(dof) - phase.x_goal.head(dof)).norm();
      const double ee_error =
          (model.endEffectorPosition(x_now) -
           model.endEffectorPosition(phase.x_goal))
              .norm();
      const double qdot_norm = x_now.segment(dof, dof).norm();
      const bool collision = model.inWorkspaceCollision(x_now);

      appendPinkNPlacePhaseSummaryRow(summary, global_iter,
                                      static_cast<int>(phase_idx), phase, iter,
                                      total_solver_elapsed, solver, model, "");

      std::cout << std::fixed << std::setprecision(4) << cfg.solver_label
                << " phase=" << phase.name << " iter=" << iter
                << " global=" << global_iter << " q_err=" << q_error
                << " ee_err=" << ee_error << " qdot=" << qdot_norm
                << " coll=" << static_cast<int>(collision)
                << " elapsed=" << solver.elapsed << " s\n";

      if (!collision && q_error < phase.q_tol && ee_error < phase.ee_tol &&
          qdot_norm < phase.qdot_tol) {
        phase_success = true;
        break;
      }

      executed_u.col(global_iter) = solver.u0;
      solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
      executed_x.col(global_iter + 1) = solver.x_init;
      ++global_iter;

      if (solver.Uo.cols() >= param.T) {
        solver.U_0.leftCols(param.T - 1) = solver.Uo.middleCols(1, param.T - 1);
        solver.U_0.col(param.T - 1).setZero();
      } else {
        solver.U_0 = makePdTorqueWarmStart(
            model, solver.x_init, solver.x_target, param.T, param.dt,
            cfg.warm_start_kp, cfg.warm_start_kd);
      }
    }

    if (!phase_success) {
      all_phases_success = false;
      std::cerr << "[WARN] phase failed or reached max iterations: "
                << phase.name << "\n";
    }

    if (!phase.gripper_event_after.empty()) {
      appendPinkNPlacePhaseSummaryRow(summary, global_iter,
                                      static_cast<int>(phase_idx), phase,
                                      phase.max_iter, total_solver_elapsed,
                                      solver, model,
                                      phase.gripper_event_after);
      std::cout << "[gripper after] " << phase.name << ": "
                << phase.gripper_event_after << "\n";
    }
  }

  writeMatrixCsv(cfg.csv_prefix + "executed_x.csv",
                 executed_x.leftCols(global_iter + 1), "x");
  writeMatrixCsv(cfg.csv_prefix + "executed_u.csv",
                 executed_u.leftCols(std::max(0, global_iter)), "tau");
  if (solver.Xo.size() > 0) {
    writeMatrixCsv(cfg.csv_prefix + "last_plan_x.csv", solver.Xo, "x");
  }

  std::cout << "\n=== pinkNplace " << cfg.solver_label << " result ===\n"
            << "success: " << static_cast<int>(all_phases_success) << "\n"
            << "executed_steps: " << global_iter << "\n"
            << "final_workspace_collision: "
            << static_cast<int>(model.inWorkspaceCollision(solver.x_init))
            << "\n"
            << "total_solver_elapsed: " << total_solver_elapsed << " s\n"
            << "CSV prefix: " << cfg.csv_prefix << "\n";

  return 0;
}

template <typename Solver>
int runPinkNPlaceMppiLikeExample(const PinkNPlaceMppiConfig &cfg) {
  return runPinkNPlaceMppiLikeExample<Solver>(
      cfg, [](Solver &) {});
}

template <typename Solver>
int runPinkNPlaceBiMppiExample(const PinkNPlaceBiMppiConfig &cfg) {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;

  Model model;
  applyPinkNPlaceModelDefaults(model);
  CollisionChecker cc;
  registerPickPlaceWorkspace(model, cc);

  const Eigen::VectorXd x_home = makePickPlaceHomeState(model);
  const auto phases = makePickPlacePhases(model);

  printPinkNPlaceWorkspaceBoxes();
  printPinkNPlacePhaseTargets(model, phases);

  BiMPPIParam param;
  param.dt = cfg.dt;
  param.Tf = cfg.Tf;
  param.Tb = cfg.Tb;
  param.Nf = cfg.Nf;
  param.Nb = cfg.Nb;
  param.Nr = cfg.Nr;
  param.gamma_u = cfg.gamma_u;
  param.x_init = x_home;
  param.x_target = phases.front().x_goal;
  param.sigma_u = Eigen::MatrixXd::Zero(dof, dof);
  param.sigma_u.diagonal() << cfg.sigma_q1, cfg.sigma_q2, cfg.sigma_q3,
      cfg.sigma_q4, cfg.sigma_q5, cfg.sigma_q6;
  param.deviation_mu = cfg.deviation_mu;
  param.cost_mu = cfg.cost_mu;
  param.epsilon = cfg.epsilon;
  param.minpts = cfg.minpts;
  param.psi = cfg.psi;

  Solver solver(model);
  solver.init(param);
  solver.setCollisionChecker(&cc);
  solver.setSeed(cfg.seed);

  solver.x_init = x_home;
  solver.x_target = phases.front().x_goal;
  solver.U_f0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                      param.Tf, param.dt, cfg.warm_start_kp,
                                      cfg.warm_start_kd);
  solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                      param.Tb, param.dt, cfg.warm_start_kp,
                                      cfg.warm_start_kd);

  std::ofstream summary(cfg.csv_prefix + "summary.csv");
  writePinkNPlacePhaseSummaryHeader(summary);

  Eigen::MatrixXd executed_x =
      Eigen::MatrixXd::Zero(model.dim_x, cfg.max_total_steps + 1);
  Eigen::MatrixXd executed_u =
      Eigen::MatrixXd::Zero(model.dim_u, cfg.max_total_steps);
  executed_x.col(0) = solver.x_init;

  int global_iter = 0;
  double total_solver_elapsed = 0.0;
  bool all_phases_success = true;

  for (std::size_t phase_idx = 0; phase_idx < phases.size(); ++phase_idx) {
    const auto &phase = phases[phase_idx];
    solver.x_target = phase.x_goal;
    solver.U_f0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                        param.Tf, param.dt,
                                        cfg.warm_start_kp, cfg.warm_start_kd);
    solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                        param.Tb, param.dt,
                                        cfg.warm_start_kp, cfg.warm_start_kd);

    if (!phase.gripper_event_before.empty()) {
      appendPinkNPlacePhaseSummaryRow(
          summary, global_iter, static_cast<int>(phase_idx), phase, -1,
          total_solver_elapsed, solver, model, phase.gripper_event_before);
      std::cout << "[gripper before] " << phase.name << ": "
                << phase.gripper_event_before << "\n";
    }

    bool phase_success = false;
    for (int iter = 0; iter < phase.max_iter &&
                       global_iter < cfg.max_total_steps;
         ++iter) {
      solver.solve();
      total_solver_elapsed += solver.elapsed;

      const Eigen::VectorXd x_now = solver.x_init;
      const double q_error = (x_now.head(dof) - phase.x_goal.head(dof)).norm();
      const double ee_error =
          (model.endEffectorPosition(x_now) -
           model.endEffectorPosition(phase.x_goal))
              .norm();
      const double qdot_norm = x_now.segment(dof, dof).norm();
      const bool collision = model.inWorkspaceCollision(x_now);

      appendPinkNPlacePhaseSummaryRow(summary, global_iter,
                                      static_cast<int>(phase_idx), phase, iter,
                                      total_solver_elapsed, solver, model, "");

      std::cout << std::fixed << std::setprecision(4) << cfg.solver_label
                << " phase=" << phase.name << " iter=" << iter
                << " global=" << global_iter << " q_err=" << q_error
                << " ee_err=" << ee_error << " qdot=" << qdot_norm
                << " coll=" << static_cast<int>(collision)
                << " conn=" << solver.connectionDistance()
                << " elapsed=" << solver.elapsed << " s\n";

      if (!collision && q_error < phase.q_tol && ee_error < phase.ee_tol &&
          qdot_norm < phase.qdot_tol) {
        phase_success = true;
        break;
      }

      executed_u.col(global_iter) = solver.u0;
      solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
      executed_x.col(global_iter + 1) = solver.x_init;
      ++global_iter;

      if (solver.Uo.cols() >= param.Tf) {
        solver.U_f0.leftCols(param.Tf - 1) =
            solver.Uo.middleCols(1, param.Tf - 1);
        solver.U_f0.col(param.Tf - 1).setZero();
      } else {
        solver.U_f0 = makePdTorqueWarmStart(
            model, solver.x_init, solver.x_target, param.Tf, param.dt,
            cfg.warm_start_kp, cfg.warm_start_kd);
      }
      solver.U_b0 = makePdTorqueWarmStart(
          model, solver.x_init, solver.x_target, param.Tb, param.dt,
          cfg.warm_start_kp, cfg.warm_start_kd);
    }

    if (!phase_success) {
      all_phases_success = false;
      std::cerr << "[WARN] phase failed or reached max iterations: "
                << phase.name << "\n";
    }

    if (!phase.gripper_event_after.empty()) {
      appendPinkNPlacePhaseSummaryRow(summary, global_iter,
                                      static_cast<int>(phase_idx), phase,
                                      phase.max_iter, total_solver_elapsed,
                                      solver, model,
                                      phase.gripper_event_after);
      std::cout << "[gripper after] " << phase.name << ": "
                << phase.gripper_event_after << "\n";
    }
  }

  writeMatrixCsv(cfg.csv_prefix + "executed_x.csv",
                 executed_x.leftCols(global_iter + 1), "x");
  writeMatrixCsv(cfg.csv_prefix + "executed_u.csv",
                 executed_u.leftCols(std::max(0, global_iter)), "tau");
  if (solver.Xo.size() > 0) {
    writeMatrixCsv(cfg.csv_prefix + "last_plan_x.csv", solver.Xo, "x");
  }

  std::cout << "\n=== pinkNplace " << cfg.solver_label << " result ===\n"
            << "success: " << static_cast<int>(all_phases_success) << "\n"
            << "executed_steps: " << global_iter << "\n"
            << "final_workspace_collision: "
            << static_cast<int>(model.inWorkspaceCollision(solver.x_init))
            << "\n"
            << "total_solver_elapsed: " << total_solver_elapsed << " s\n"
            << "CSV prefix: " << cfg.csv_prefix << "\n";

  return 0;
}
