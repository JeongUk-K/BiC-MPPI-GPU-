#include "collision_checker.h"
#include "manipulator_bicmppi_utils.h"
#include "manipulator_dynamics_model.h"
#include "manipulator_pose_set.h"
#include "mppi_gpu.cuh"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

int main() {
  using Model = ManipulatorDynamicsModel;
  constexpr int dof = Model::kDof;

  // ---------------------------------------------------------------------------
  // 1. Example-level knobs
  // ---------------------------------------------------------------------------
  const std::string solver_label = "MPPI";
  const std::string csv_prefix = "manipulator_workspace_mppi_";
  const std::uint_fast64_t seed = 260623ULL;

  const double obstacle_xmin = -0.34;
  const double obstacle_xmax = -0.20;
  const double obstacle_ymin = -0.52;
  const double obstacle_ymax = -0.38;
  const double obstacle_zmin = 0.28;
  const double obstacle_zmax = 0.42;

  const float dt = 0.02f;
  const int horizon = 90;
  const int samples = 1024;
  const double gamma_u = 0.0015;

  const double sigma_q1 = 4.5;
  const double sigma_q2 = 4.5;
  const double sigma_q3 = 3.8;
  const double sigma_q4 = 2.2;
  const double sigma_q5 = 1.8;
  const double sigma_q6 = 1.2;

  const int max_iter = 160;
  const double q_tol = 0.08;
  const double ee_tol = 0.035;
  const double qdot_tol = 0.20;

  const double warm_start_kp = 18.0;
  const double warm_start_kd = 7.0;

  // ---------------------------------------------------------------------------
  // 2. Model and workspace obstacle
  // ---------------------------------------------------------------------------
  Model model;
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
  model.addWorkspaceBoxMinMax(obstacle_xmin, obstacle_xmax, obstacle_ymin,
                              obstacle_ymax, obstacle_zmin, obstacle_zmax);

  // ---------------------------------------------------------------------------
  // 3. Random start and goal states from the fixed 16-pose set
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
  // 4. MPPI parameters
  // ---------------------------------------------------------------------------
  MPPIParam param;
  param.dt = dt;
  param.T = horizon;
  param.N = samples;
  param.gamma_u = gamma_u;
  param.x_init = x_init;
  param.x_target = x_goal;
  param.sigma_u = Eigen::MatrixXd::Zero(dof, dof);
  param.sigma_u.diagonal() << sigma_q1, sigma_q2, sigma_q3, sigma_q4,
      sigma_q5, sigma_q6;

  // ---------------------------------------------------------------------------
  // 5. Workspace link-collision checker for GPU rollouts
  // ---------------------------------------------------------------------------
  CollisionChecker cc;
  cc.clear();
  cc.resolution = 0.05;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
  cc.use_workspace_link_collision = true;
  cc.addWorkspaceBoxMinMax(obstacle_xmin, obstacle_xmax, obstacle_ymin,
                           obstacle_ymax, obstacle_zmin, obstacle_zmax);

  // ---------------------------------------------------------------------------
  // 6. Solver setup
  // ---------------------------------------------------------------------------
  MPPI_GPU solver(model);
  solver.U_0 = makePdTorqueWarmStart(model, x_init, x_goal, param.T,
                                      param.dt, warm_start_kp, warm_start_kd);
  solver.init(param);
  solver.setCollisionChecker(&cc);
  solver.setSeed(seed);

  // ---------------------------------------------------------------------------
  // 7. Receding-horizon execution and logging
  // ---------------------------------------------------------------------------
  Eigen::MatrixXd executed_x = Eigen::MatrixXd::Zero(model.dim_x, max_iter + 1);
  Eigen::MatrixXd executed_u = Eigen::MatrixXd::Zero(model.dim_u, max_iter);
  executed_x.col(0) = solver.x_init;

  std::ofstream summary(csv_prefix + "summary.csv");
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
        (model.endEffectorPosition(x_now) - model.endEffectorPosition(x_goal))
            .norm();
    const double qdot_norm = x_now.segment(dof, dof).norm();

    std::cout << std::fixed << std::setprecision(4)
              << solver_label << " iter=" << iter << " q_err=" << q_error
              << " ee_err=" << ee_error << " qdot=" << qdot_norm
              << " elapsed=" << solver.elapsed << " s\n";

    if (q_error < q_tol && ee_error < ee_tol && qdot_norm < qdot_tol) {
      executed_steps = iter;
      break;
    }

    executed_u.col(iter) = solver.u0;
    solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
    executed_x.col(iter + 1) = solver.x_init;
    executed_steps = iter + 1;

    if (solver.Uo.cols() >= param.T) {
      solver.U_0.leftCols(param.T - 1) = solver.Uo.middleCols(1, param.T - 1);
      solver.U_0.col(param.T - 1).setZero();
    } else {
      solver.U_0 = makePdTorqueWarmStart(model, solver.x_init, x_goal,
                                        param.T, param.dt, warm_start_kp,
                                        warm_start_kd);
    }
  }

  writeMatrixCsv(csv_prefix + "executed_x.csv",
                 executed_x.leftCols(executed_steps + 1), "x");
  writeMatrixCsv(csv_prefix + "executed_u.csv",
                 executed_u.leftCols(std::max(0, executed_steps)), "tau");
  if (solver.Xo.size() > 0) {
    writeMatrixCsv(csv_prefix + "last_plan_x.csv", solver.Xo, "x");
  }

  const Eigen::VectorXd x_final = solver.x_init;
  const double final_q_error = (x_final.head(dof) - x_goal.head(dof)).norm();
  const double final_ee_error =
      (model.endEffectorPosition(x_final) - model.endEffectorPosition(x_goal))
          .norm();
  const double final_qdot_norm = x_final.segment(dof, dof).norm();

  std::cout << "\n=== " << solver_label
            << " workspace-link-collision result ===\n"
            << "executed_steps: " << executed_steps << "\n"
            << "final_q_error: " << final_q_error << " rad\n"
            << "final_ee_error: " << final_ee_error << " m\n"
            << "final_qdot_norm: " << final_qdot_norm << " rad/s\n"
            << "total_solver_elapsed: " << total_solver_elapsed << " s\n"
            << "CSV prefix: " << csv_prefix << "\n";

  return 0;
}
