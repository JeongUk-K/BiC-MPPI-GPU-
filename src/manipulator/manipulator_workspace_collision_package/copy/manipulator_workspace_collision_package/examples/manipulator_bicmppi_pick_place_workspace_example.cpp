// manipulator_bicmppi_pick_place_workspace_example.cpp
// -----------------------------------------------------------------------------
// Pick-and-place BiC-MPPI example with workspace link collision.
//
// Model:
//   x = [q; qdot] in R^12, u = tau in R^6
//   qdot = v
//   vdot_i = (tau_i - d_i v_i - g_i sin(q_i)) / I_i
//
// Task:
//   Sequential pick-and-place phases with workspace AABB obstacles. Each phase
//   uses a target joint posture q_goal that fixes the corresponding EE pose via
//   FK. Gripper events are logged as discrete events; gripper dynamics are not
//   part of the BiC-MPPI state.
// -----------------------------------------------------------------------------

#include "bi_mppi_gpu.cuh"
#include "collision_checker.h"
#include "manipulator_bicmppi_utils.h"
#include "manipulator_dynamics_model.h"
#include "manipulator_pick_place_task.h"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void writePhaseSummaryHeader(std::ofstream &ofs) {
  ofs << "global_iter,phase_index,phase_name,phase_iter,solver_elapsed,"
         "total_solver_elapsed,connection_distance,q_error,ee_error,"
         "qdot_norm,tau_norm,workspace_collision,ee_x,ee_y,ee_z,"
         "target_ee_x,target_ee_y,target_ee_z,gripper_event\n";
}

template <typename Solver>
void appendPhaseSummaryRow(std::ofstream &ofs, int global_iter, int phase_index,
                           const PickPlacePhase &phase, int phase_iter,
                           double total_solver_elapsed, const Solver &solver,
                           const ManipulatorDynamicsModel &model,
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
      << "," << static_cast<int>(collision) << ","
      << ee.x() << "," << ee.y() << "," << ee.z() << ","
      << target_ee.x() << "," << target_ee.y() << "," << target_ee.z()
      << "," << event << "\n";
}

void printWorkspaceBoxes() {
  std::cout << "\n=== Pick-and-place workspace obstacles [base frame, m] ===\n";
  for (const auto &b : makePickPlaceWorkspaceBoxes()) {
    std::cout << std::setw(22) << b.name
              << "  x:[" << b.xmin << ", " << b.xmax << "]"
              << " y:[" << b.ymin << ", " << b.ymax << "]"
              << " z:[" << b.zmin << ", " << b.zmax << "]\n";
  }
}

void printPhaseTargets(const ManipulatorDynamicsModel &model,
                       const std::vector<PickPlacePhase> &phases) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;
  std::cout << "\n=== Pick-and-place target postures and FK EE positions ===\n";
  for (std::size_t i = 0; i < phases.size(); ++i) {
    const auto &p = phases[i];
    const Eigen::Vector3d ee = model.endEffectorPosition(p.x_goal);
    std::cout << i << "  " << std::setw(24) << p.name << "  q=[";
    for (int j = 0; j < dof; ++j) {
      std::cout << p.x_goal(j) << (j + 1 == dof ? "" : ", ");
    }
    std::cout << "]  ee=[" << ee.x() << ", " << ee.y() << ", " << ee.z()
              << "]";
    if (!p.gripper_event_before.empty()) {
      std::cout << "  before:" << p.gripper_event_before;
    }
    if (!p.gripper_event_after.empty()) {
      std::cout << "  after:" << p.gripper_event_after;
    }
    std::cout << "\n";
  }
}

}  // namespace

int main() {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;

  // ---------------------------------------------------------------------------
  // 1. Model and pick-and-place workspace
  // ---------------------------------------------------------------------------
  Model model;
  CollisionChecker cc;
  registerPickPlaceWorkspace(model, cc);

  const Eigen::VectorXd x_home = makeManipulatorStateFromQ(
      model, {0.00, -1.00, 1.20, 0.00, 0.80, 0.00});
  const auto phases = makePickPlacePhases(model);

  printWorkspaceBoxes();
  printPhaseTargets(model, phases);

  // ---------------------------------------------------------------------------
  // 2. BiC-MPPI parameters
  // ---------------------------------------------------------------------------
  BiMPPIParam param;
  param.dt = 0.02f;
  param.Tf = 95;
  param.Tb = 95;
  param.Nf = 1024;
  param.Nb = 1024;
  param.Nr = 1024;
  param.gamma_u = 0.0012;
  param.x_init = x_home;
  param.x_target = phases.front().x_goal;

  param.sigma_u = Eigen::MatrixXd::Zero(dof, dof);
  param.sigma_u.diagonal() << 5.0, 5.0, 4.2, 2.5, 2.0, 1.4;

  // For 6D torque input, DBSCAN epsilon must be larger than the 2D/3D vehicle
  // examples. This value usually gives a small number of behavior branches.
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.epsilon = 4.2;
  param.minpts = 8;
  param.psi = 0.0;

  // ---------------------------------------------------------------------------
  // 3. Solver setup
  // ---------------------------------------------------------------------------
  BiMPPI_GPU solver(model);
  solver.init(param);
  solver.setCollisionChecker(&cc);
  solver.setSeed(260623ULL);

  solver.x_init = x_home;
  solver.x_target = phases.front().x_goal;
  solver.U_f0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                      param.Tf, param.dt, 20.0, 7.5);
  solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                      param.Tb, param.dt, 20.0, 7.5);

  // ---------------------------------------------------------------------------
  // 4. Sequential pick-and-place execution
  // ---------------------------------------------------------------------------
  std::ofstream summary("manipulator_pick_place_bicmppi_summary.csv");
  writePhaseSummaryHeader(summary);

  const int max_total_steps = 650;
  Eigen::MatrixXd executed_x = Eigen::MatrixXd::Zero(model.dim_x, max_total_steps + 1);
  Eigen::MatrixXd executed_u = Eigen::MatrixXd::Zero(model.dim_u, max_total_steps);
  executed_x.col(0) = solver.x_init;

  int global_iter = 0;
  double total_solver_elapsed = 0.0;
  bool all_phases_success = true;

  for (std::size_t phase_idx = 0; phase_idx < phases.size(); ++phase_idx) {
    const auto &phase = phases[phase_idx];
    solver.x_target = phase.x_goal;

    solver.U_f0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                        param.Tf, param.dt, 20.0, 7.5);
    solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                        param.Tb, param.dt, 20.0, 7.5);

    if (!phase.gripper_event_before.empty()) {
      appendPhaseSummaryRow(summary, global_iter, static_cast<int>(phase_idx),
                            phase, -1, total_solver_elapsed, solver, model,
                            phase.gripper_event_before);
      std::cout << "[gripper before] " << phase.name << ": "
                << phase.gripper_event_before << "\n";
    }

    bool phase_success = false;
    for (int iter = 0; iter < phase.max_iter && global_iter < max_total_steps; ++iter) {
      solver.solve();
      total_solver_elapsed += solver.elapsed;

      const Eigen::VectorXd x_now = solver.x_init;
      const double q_error = (x_now.head(dof) - phase.x_goal.head(dof)).norm();
      const double ee_error =
          (model.endEffectorPosition(x_now) - model.endEffectorPosition(phase.x_goal)).norm();
      const double qdot_norm = x_now.segment(dof, dof).norm();
      const bool collision = model.inWorkspaceCollision(x_now);

      appendPhaseSummaryRow(summary, global_iter, static_cast<int>(phase_idx),
                            phase, iter, total_solver_elapsed, solver, model, "");

      std::cout << std::fixed << std::setprecision(4)
                << "phase=" << phase.name
                << " iter=" << iter
                << " global=" << global_iter
                << " q_err=" << q_error
                << " ee_err=" << ee_error
                << " qdot=" << qdot_norm
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

      // Warm-start forward sequence from previous selected plan when possible.
      if (solver.Uo.cols() >= param.Tf) {
        solver.U_f0.leftCols(param.Tf - 1) = solver.Uo.middleCols(1, param.Tf - 1);
        solver.U_f0.col(param.Tf - 1).setZero();
      } else {
        solver.U_f0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                            param.Tf, param.dt, 20.0, 7.5);
      }
      solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, solver.x_target,
                                          param.Tb, param.dt, 20.0, 7.5);
    }

    if (!phase_success) {
      all_phases_success = false;
      std::cerr << "[WARN] phase failed or reached max iterations: "
                << phase.name << "\n";
      break;
    }

    if (!phase.gripper_event_after.empty()) {
      appendPhaseSummaryRow(summary, global_iter, static_cast<int>(phase_idx),
                            phase, phase.max_iter, total_solver_elapsed, solver,
                            model, phase.gripper_event_after);
      std::cout << "[gripper after] " << phase.name << ": "
                << phase.gripper_event_after << "\n";
    }
  }

  writeMatrixCsv("manipulator_pick_place_executed_x.csv",
                 executed_x.leftCols(global_iter + 1), "x");
  writeMatrixCsv("manipulator_pick_place_executed_u.csv",
                 executed_u.leftCols(std::max(0, global_iter)), "tau");
  if (solver.Xo.size() > 0) {
    writeMatrixCsv("manipulator_pick_place_last_plan_x.csv", solver.Xo, "x");
  }

  std::cout << "\n=== Pick-and-place BiC-MPPI result ===\n"
            << "success: " << static_cast<int>(all_phases_success) << "\n"
            << "executed_steps: " << global_iter << "\n"
            << "final_workspace_collision: "
            << static_cast<int>(model.inWorkspaceCollision(solver.x_init)) << "\n"
            << "total_solver_elapsed: " << total_solver_elapsed << " s\n"
            << "CSV: manipulator_pick_place_bicmppi_summary.csv\n"
            << "CSV: manipulator_pick_place_executed_x.csv\n"
            << "CSV: manipulator_pick_place_executed_u.csv\n"
            << "CSV: manipulator_pick_place_last_plan_x.csv\n";

  return all_phases_success ? 0 : 2;
}
