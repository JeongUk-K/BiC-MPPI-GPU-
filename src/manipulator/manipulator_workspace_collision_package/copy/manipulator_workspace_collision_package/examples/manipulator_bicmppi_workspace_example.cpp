// manipulator_bicmppi_workspace_example.cpp
// -----------------------------------------------------------------------------
// BiC-MPPI workspace-link-collision example for a 6-DOF second-order manipulator model.
//
// Required project files from the existing BiC-MPPI repository:
//   - bi_mppi_gpu.cuh / bi_mppi_gpu.cu
//   - mppi_gpu.cuh / cuda_utils.cuh
//
// Drop-in files supplied in this package:
//   - manipulator_dynamics_model.h
//   - manipulator_bicmppi_utils.h
//   - collision_checker.h        (only if your repo does not already define one)
//   - mppi_param.h               (only if your repo does not already define one)
//   - cuda_legacy_model.cuh      (replace/patch the existing one for 12D dynamics)
// -----------------------------------------------------------------------------

#include "bi_mppi_gpu.cuh"
#include "collision_checker.h"
#include "manipulator_bicmppi_utils.h"
#include "manipulator_dynamics_model.h"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;

  // ---------------------------------------------------------------------------
  // 1. Model
  // ---------------------------------------------------------------------------
  Model model;

  // Workspace obstacle in the manipulator base frame.
  // The same box is registered in both the CPU model cost and the GPU collision
  // checker below, so rollout collision uses FK-based link-AABB tests.
  model.addWorkspaceBoxMinSize(0.25, -0.22, 0.05, 0.22, 0.44, 0.45);

  // ---------------------------------------------------------------------------
  // 2. Start and goal states x=[q;qdot]
  // ---------------------------------------------------------------------------
  Eigen::VectorXd q_init(dof), q_goal(dof), qdot_zero(dof);
  q_init << 0.05, -1.15, 1.25, 0.00, 0.85, 0.00;
  q_goal << 1.45, -0.80, 1.05, -0.20, 0.65, 0.25;
  qdot_zero.setZero();

  const Eigen::VectorXd x_init = model.makeState(q_init, qdot_zero);
  const Eigen::VectorXd x_goal = model.makeState(q_goal, qdot_zero);

  // ---------------------------------------------------------------------------
  // 3. BiC-MPPI parameters
  // ---------------------------------------------------------------------------
  BiMPPIParam param;
  param.dt = 0.02f;
  param.Tf = 90;
  param.Tb = 90;
  param.Nf = 1024;
  param.Nb = 1024;
  param.Nr = 1024;

  // The CUDA terminal cost is O(1e3 * ||q-qg||^2). Keep gamma small enough to
  // avoid all weights collapsing to zero.
  param.gamma_u = 0.0015;

  param.x_init = x_init;
  param.x_target = x_goal;

  param.sigma_u = Eigen::MatrixXd::Zero(dof, dof);
  param.sigma_u.diagonal() << 4.5, 4.5, 3.8, 2.2, 1.8, 1.2;

  // DBSCAN parameters for 3-block input-deviation features.
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.epsilon = 3.8;
  param.minpts = 8;
  param.psi = 0.0;

  // ---------------------------------------------------------------------------
  // 4. Workspace link-collision checker for GPU rollout branch generation
  // ---------------------------------------------------------------------------
  CollisionChecker cc;
  cc.clear();
  cc.resolution = 0.05;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
  cc.use_workspace_link_collision = true;

  // Same workspace box as the CPU model. The GPU kernel samples points along
  // each link segment and checks point-AABB distance minus link_radius.
  cc.addWorkspaceBoxMinSize(0.25, -0.22, 0.05, 0.22, 0.44, 0.45);

  // ---------------------------------------------------------------------------
  // 5. Solver setup
  // ---------------------------------------------------------------------------
  BiMPPI_GPU solver(model);
  solver.init(param);
  solver.setCollisionChecker(&cc);
  solver.setSeed(260623ULL);

  solver.U_f0 = makePdTorqueWarmStart(model, x_init, x_goal, param.Tf, param.dt);
  solver.U_b0 = makePdTorqueWarmStart(model, x_init, x_goal, param.Tb, param.dt);

  // ---------------------------------------------------------------------------
  // 6. Receding-horizon execution and logging
  // ---------------------------------------------------------------------------
  constexpr int max_iter = 120;
  constexpr double q_tol = 0.08;      // [rad]
  constexpr double ee_tol = 0.035;    // [m]
  constexpr double qdot_tol = 0.20;   // [rad/s]

  Eigen::MatrixXd executed_x = Eigen::MatrixXd::Zero(model.dim_x, max_iter + 1);
  Eigen::MatrixXd executed_u = Eigen::MatrixXd::Zero(model.dim_u, max_iter);
  executed_x.col(0) = solver.x_init;

  std::ofstream summary("manipulator_workspace_bicmppi_summary.csv");
  writeManipulatorSummaryHeader(summary);

  double total_solver_elapsed = 0.0;
  int executed_steps = 0;

  for (int iter = 0; iter < max_iter; ++iter) {
    solver.solve();
    total_solver_elapsed += solver.elapsed;

    appendManipulatorSummaryRow(summary, iter, total_solver_elapsed, solver,
                                model, x_goal);

    const Eigen::VectorXd x_now = solver.x_init;
    const double q_error = (x_now.head(dof) - x_goal.head(dof)).norm();
    const double ee_error =
        (model.endEffectorPosition(x_now) - model.endEffectorPosition(x_goal)).norm();
    const double qdot_norm = x_now.segment(dof, dof).norm();

    std::cout << std::fixed << std::setprecision(4)
              << "iter=" << iter
              << " q_err=" << q_error
              << " ee_err=" << ee_error
              << " qdot=" << qdot_norm
              << " conn=" << solver.connectionDistance()
              << " elapsed=" << solver.elapsed << " s\n";

    if (q_error < q_tol && ee_error < ee_tol && qdot_norm < qdot_tol) {
      executed_steps = iter;
      break;
    }

    executed_u.col(iter) = solver.u0;

    // Use RK4 for the simulated plant execution. The uploaded GPU rollout kernel
    // may still use Euler internally unless you patch it separately.
    solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
    executed_x.col(iter + 1) = solver.x_init;
    executed_steps = iter + 1;

    // Warm-start forward sequence from the previous selected sequence.
    if (solver.Uo.cols() >= param.Tf) {
      solver.U_f0.leftCols(param.Tf - 1) = solver.Uo.middleCols(1, param.Tf - 1);
      solver.U_f0.col(param.Tf - 1).setZero();
    } else {
      solver.U_f0 = makePdTorqueWarmStart(model, solver.x_init, x_goal,
                                          param.Tf, param.dt);
    }

    // Goal-side heuristic warm start. Recompute from current state to goal.
    solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, x_goal,
                                        param.Tb, param.dt);
  }

  writeMatrixCsv("manipulator_workspace_bicmppi_executed_x.csv",
                 executed_x.leftCols(executed_steps + 1), "x");
  writeMatrixCsv("manipulator_workspace_bicmppi_executed_u.csv",
                 executed_u.leftCols(std::max(0, executed_steps)), "tau");
  if (solver.Xo.size() > 0) {
    writeMatrixCsv("manipulator_workspace_bicmppi_last_plan_x.csv", solver.Xo, "x");
  }

  const Eigen::VectorXd x_final = solver.x_init;
  const double final_q_error = (x_final.head(dof) - x_goal.head(dof)).norm();
  const double final_ee_error =
      (model.endEffectorPosition(x_final) - model.endEffectorPosition(x_goal)).norm();
  const double final_qdot_norm = x_final.segment(dof, dof).norm();

  std::cout << "\n=== Workspace-link-collision manipulator BiC-MPPI result ===\n"
            << "executed_steps: " << executed_steps << "\n"
            << "final_q_error: " << final_q_error << " rad\n"
            << "final_ee_error: " << final_ee_error << " m\n"
            << "final_qdot_norm: " << final_qdot_norm << " rad/s\n"
            << "total_solver_elapsed: " << total_solver_elapsed << " s\n"
            << "CSV: manipulator_workspace_bicmppi_summary.csv\n"
            << "CSV: manipulator_workspace_bicmppi_executed_x.csv\n"
            << "CSV: manipulator_workspace_bicmppi_executed_u.csv\n"
            << "CSV: manipulator_workspace_bicmppi_last_plan_x.csv\n";

  return 0;
}
