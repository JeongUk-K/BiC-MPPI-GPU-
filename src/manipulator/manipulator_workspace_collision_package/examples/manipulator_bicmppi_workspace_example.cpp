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
#include "manipulator_pose_set.h"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

int main() {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;
  const unsigned long long seed = 260623ULL;

  // ---------------------------------------------------------------------------
  // 1. Model
  // ---------------------------------------------------------------------------
  Model model;

  // Mid-path workspace obstacle in the manipulator base frame.
  // The same box is registered in both the CPU model cost and the GPU collision
  // checker below, so rollout collision uses FK-based link-AABB tests.
  model.addWorkspaceBoxMinMax(-0.34, -0.20, -0.52, -0.38, 0.28, 0.42);

  // ---------------------------------------------------------------------------
  // 2. Random start and goal states from the fixed 16-pose set
  // ---------------------------------------------------------------------------
  const auto reference_poses = manipulator_pose_set::makeReferencePoseSet16();
  manipulator_pose_set::printPoseSet(model, reference_poses);
  manipulator_pose_set::printPoseCollisionCheck(model, reference_poses);

  std::mt19937 rng(static_cast<unsigned int>(seed));
  const auto pose_pair =
      manipulator_pose_set::sampleDifferentPosePair(reference_poses, rng);

  const Eigen::VectorXd x_init =
      manipulator_pose_set::makeStateFromPose(model, pose_pair.init_pose);
  const Eigen::VectorXd x_goal =
      manipulator_pose_set::makeStateFromPose(model, pose_pair.goal_pose);

std::cout << "\n=== Random pose-set benchmark ===\n"
          << "seed      : " << seed << "\n"
          << "init_idx  : " << pose_pair.init_idx << " ("
          << pose_pair.init_pose.name << ")\n"
          << "goal_idx  : " << pose_pair.goal_idx << " ("
          << pose_pair.goal_pose.name << ")\n"
          << "q_init    : " << x_init.head(dof).transpose() << "\n"
          << "q_goal    : " << x_goal.head(dof).transpose() << "\n"
          << "init_coll : "
          << static_cast<int>(model.inWorkspaceCollision(x_init)) << "\n"
          << "goal_coll : "
          << static_cast<int>(model.inWorkspaceCollision(x_goal)) << "\n";

  if (model.inWorkspaceCollision(x_init) ||
      model.inWorkspaceCollision(x_goal)) {
    std::cerr << "[WARN] Initial or goal pose is already in workspace collision. "
              << "Exclude this pair from paper-quality statistics.";
  }

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

  // Warm-start PD gains.
  const double warm_start_kp = 18.0;
  const double warm_start_kd = 7.0;

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
  cc.addWorkspaceBoxMinMax(-0.34, -0.20, -0.52, -0.38, 0.28, 0.42);

  // ---------------------------------------------------------------------------
  // 5. Solver setup
  // ---------------------------------------------------------------------------
  BiMPPI_GPU solver(model);
  solver.init(param);
  solver.setCollisionChecker(&cc);
  solver.setSeed(seed);

  solver.U_f0 = makePdTorqueWarmStart(model, x_init, x_goal, param.Tf,
                                      param.dt, warm_start_kp, warm_start_kd);
  solver.U_b0 = makePdTorqueWarmStart(model, x_init, x_goal, param.Tb,
                                      param.dt, warm_start_kp, warm_start_kd);

  // ---------------------------------------------------------------------------
  // 6. Receding-horizon execution and logging
  // ---------------------------------------------------------------------------
  constexpr int max_iter = 160;
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
                                          param.Tf, param.dt, warm_start_kp,
                                          warm_start_kd);
    }

    // Goal-side heuristic warm start. Recompute from current state to goal.
    solver.U_b0 = makePdTorqueWarmStart(model, solver.x_init, x_goal,
                                        param.Tb, param.dt, warm_start_kp,
                                        warm_start_kd);
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
