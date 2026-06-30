// manipulator_grid_approach_workspace_example.cpp
// -----------------------------------------------------------------------------
// BiC-MPPI grid-approach manipulator benchmark.
//
// Initial state:
//   A free-space posture whose wrist/tool side is intentionally flipped away
//   from the grid.
//
// Goal:
//   One randomly selected grid insertion cell.
//
// Expected route:
//   free_opposite -> free_aligned -> goal_retract -> goal_insert
// -----------------------------------------------------------------------------

#include "bi_mppi_gpu.cuh"
#include "collision_checker.h"
#include "manipulator_bicmppi_utils.h"
#include "manipulator_dynamics_model.h"
#include "manipulator_grid_approach_task.h"
#include "manipulator_grid_wall_task.h"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char **argv) {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;

  const std::string solver_label = "BiC-MPPI";
  const std::string csv_prefix = "manipulator_grid_approach_bicmppi_";

  std::uint_fast64_t seed64 = 260627ULL;
  int goal_cell_arg = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--seed" && i + 1 < argc) {
      seed64 = static_cast<std::uint_fast64_t>(std::stoull(argv[++i]));
    } else if (arg == "--goal" && i + 1 < argc) {
      goal_cell_arg = std::stoi(argv[++i]);
    }
  }

  const unsigned int seed = static_cast<unsigned int>(seed64);

  const float dt = 0.02f;
  const int Tf = 100;
  const int Tb = 100;
  const int Nf = 3000;
  const int Nb = 3000;
  const int Nr = 1024;
  const double gamma_u = 0.0013;

  const int max_iter = 300;
  const double ee_tol = 0.045;
  const double qdot_tol = 0.28;

  const double warm_start_kp = 18.0;
  const double warm_start_kd = 7.0;

  // ---------------------------------------------------------------------------
  // 1. Model and workspace
  // ---------------------------------------------------------------------------
  Model model;
  model.w_tau = 1.0e-3;
  model.w_qdot = 2.0e-2;
  model.w_joint_limit = 20.0;
  model.w_workspace_obs = 5.0e3;
  model.w_ground = 2.0e4;

  // Workspace task: do not overconstrain to one q; emphasize EE target.
  model.w_terminal_q = 5.0e1;
  model.w_terminal_ee = 4.0e3;
  model.w_terminal_qdot = 1.0e2;

  model.link_radius = 0.045;
  model.obs_safe_margin = 0.10;
  model.hard_collision_margin = 0.02;

  CollisionChecker cc;
  cc.clear();
  cc.resolution = 0.05;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
  cc.use_workspace_link_collision = true;

  manipulator_grid_approach::GridApproachConfig approach_cfg;
  approach_cfg.verbose = true;
  approach_cfg.grid_cfg.verbose = true;

  const auto boxes =
      manipulator_grid_wall::makeGridWallWorkspaceBoxes(approach_cfg.grid_cfg);
  manipulator_grid_wall::printGridWallBoxes(boxes);
  manipulator_grid_wall::registerGridWallWorkspace(
      model, cc, approach_cfg.grid_cfg);

  const auto cells =
      manipulator_grid_wall::makeGridCellPoseSet9(model, approach_cfg.grid_cfg);

  std::mt19937 rng(seed);
  manipulator_grid_approach::GridApproachTask task;
  if (goal_cell_arg >= 0) {
    task = manipulator_grid_approach::makeGridApproachTaskByGoal(
        model, cells, goal_cell_arg, approach_cfg);
  } else {
    task = manipulator_grid_approach::sampleValidGridApproachTask(
        model, cells, rng, approach_cfg);
  }

  const Eigen::VectorXd x_init =
      manipulator_grid_approach::makeFreeOppositeState(model, task);
  const Eigen::VectorXd x_goal =
      manipulator_grid_approach::makeGoalInsertState(model, task);

  if (!manipulator_grid_approach::gridApproachTaskKeyStatesCollisionFree(task)) {
    std::cerr << "[WARN] One or more approach key states are in collision. "
              << "The solver will run for debugging, but this should not be "
              << "used for paper-quality statistics.\n";
  }

  // ---------------------------------------------------------------------------
  // 2. BiC-MPPI parameters
  // ---------------------------------------------------------------------------
  BiMPPIParam param;
  param.dt = dt;
  param.Tf = Tf;
  param.Tb = Tb;
  param.Nf = Nf;
  param.Nb = Nb;
  param.Nr = Nr;
  param.gamma_u = gamma_u;
  param.x_init = x_init;
  param.x_target = x_goal;

  param.sigma_u = Eigen::MatrixXd::Zero(dof, dof);
  param.sigma_u.diagonal() << 5.0, 5.0, 4.2, 2.5, 2.0, 1.4;

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
  solver.setSeed(static_cast<unsigned long long>(seed64));

  solver.x_init = x_init;
  solver.x_target = x_goal;

  solver.U_f0 = manipulator_grid_approach::makeGridApproachWarmStart(
      model, task, param.Tf, param.dt, warm_start_kp, warm_start_kd);
  solver.U_b0 = manipulator_grid_approach::makeGridApproachReverseWarmStart(
      model, task, param.Tb, param.dt, warm_start_kp, warm_start_kd);

  // ---------------------------------------------------------------------------
  // 4. Receding-horizon execution and logging
  // ---------------------------------------------------------------------------
  Eigen::MatrixXd executed_x = Eigen::MatrixXd::Zero(model.dim_x, max_iter + 1);
  Eigen::MatrixXd executed_u = Eigen::MatrixXd::Zero(model.dim_u, max_iter);
  executed_x.col(0) = solver.x_init;

  std::ofstream summary(csv_prefix + "summary.csv");
  writeManipulatorSummaryHeader(summary);

  std::ofstream metadata(csv_prefix + "task_metadata.csv");
  manipulator_grid_approach::writeGridApproachMetadataHeader(metadata);
  manipulator_grid_approach::writeGridApproachMetadataRow(metadata, seed, task);

  double total_solver_elapsed = 0.0;
  int executed_steps = 0;
  bool success = false;

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
    const bool collision = model.inWorkspaceCollision(x_now);

    std::cout << std::fixed << std::setprecision(4)
              << solver_label << " iter=" << iter
              << " q_err=" << q_error
              << " ee_err=" << ee_error
              << " qdot=" << qdot_norm
              << " coll=" << static_cast<int>(collision)
              << " conn=" << solver.connectionDistance()
              << " elapsed=" << solver.elapsed << " s\n";

    if (!collision && ee_error < ee_tol && qdot_norm < qdot_tol) {
      executed_steps = iter;
      success = true;
      break;
    }

    executed_u.col(iter) = solver.u0;
    solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
    executed_x.col(iter + 1) = solver.x_init;
    executed_steps = iter + 1;

    if (solver.Uo.cols() >= param.Tf) {
      solver.U_f0.leftCols(param.Tf - 1) = solver.Uo.middleCols(1, param.Tf - 1);
      solver.U_f0.col(param.Tf - 1).setZero();
    } else {
      solver.U_f0 = manipulator_grid_approach::makeGridApproachWarmStart(
          model, task, param.Tf, param.dt, warm_start_kp, warm_start_kd);
    }

    solver.U_b0 = manipulator_grid_approach::makeGridApproachReverseWarmStart(
        model, task, param.Tb, param.dt, warm_start_kp, warm_start_kd);
  }

  writeMatrixCsv(csv_prefix + "executed_x.csv",
                 executed_x.leftCols(executed_steps + 1), "x");
  writeMatrixCsv(csv_prefix + "executed_u.csv",
                 executed_u.leftCols(std::max(0, executed_steps)), "tau");
  if (solver.Xo.size() > 0) {
    writeMatrixCsv(csv_prefix + "last_plan_x.csv", solver.Xo, "x");
  }

  const Eigen::VectorXd x_final = solver.x_init;
  const double final_ee_error =
      (model.endEffectorPosition(x_final) - model.endEffectorPosition(x_goal)).norm();
  const double final_qdot_norm = x_final.segment(dof, dof).norm();

  std::cout << "\n=== Grid-approach manipulator BiC-MPPI result ===\n"
            << "success: " << static_cast<int>(success) << "\n"
            << "seed: " << seed << "\n"
            << "goal_cell: " << task.goal_id << "\n"
            << "executed_steps: " << executed_steps << "\n"
            << "final_ee_error: " << final_ee_error << " m\n"
            << "final_qdot_norm: " << final_qdot_norm << " rad/s\n"
            << "final_collision: "
            << static_cast<int>(model.inWorkspaceCollision(x_final)) << "\n"
            << "total_solver_elapsed: " << total_solver_elapsed << " s\n"
            << "CSV prefix: " << csv_prefix << "\n";

  return success ? 0 : 2;
}
