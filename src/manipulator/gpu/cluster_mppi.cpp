#include "cluster_mppi_gpu.cuh"
#include "manipulator_dynamics_model.h"
#include "collision_checker.h"
#include "mppi_param.h"

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

int main() {
  ManipulatorDynamicsModel model;
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

  model.addWorkspaceBoxMinMax(0.3, 0.5, -0.1, 0.1, 0.1, 0.4);

  MPPIParam param;
  param.dt = 0.02f;
  param.T = 90;
  param.N = 4096;
  param.gamma_u = 0.0015;
  param.x_init = Eigen::VectorXd::Zero(ManipulatorDynamicsModel::kDof * 2);
  param.x_init << 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  param.x_target = Eigen::VectorXd::Zero(ManipulatorDynamicsModel::kDof * 2);
  param.x_target << 0.5, -0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  Eigen::VectorXd sigma_u(ManipulatorDynamicsModel::kDof);
  sigma_u << 4.5, 4.5, 3.8, 2.2, 1.8, 1.2;
  param.sigma_u = sigma_u.asDiagonal();

  CollisionChecker cc;
  cc.resolution = 0.05;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
  cc.use_workspace_link_collision = true;
  cc.addWorkspaceBoxMinMax(0.3, 0.5, -0.1, 0.1, 0.1, 0.4);

  param.clustering_method = ClusteringMethod::KMeans;

  ClusterMPPI_GPU solver(model);
  solver.U_0 = Eigen::MatrixXd::Zero(ManipulatorDynamicsModel::kDof, param.T);
  solver.setClusteringMethod(ClusteringMethod::KMeans);
  solver.init(param);
  solver.setCollisionChecker(&cc);

  std::cout << "[Manipulator Cluster-MPPI] Initialized solver successfully.\n";
  for (int iter = 0; iter < 200; ++iter) {
    solver.solve();
    solver.move();
    const double q_err = (solver.x_init.head(6) - param.x_target.head(6)).norm();
    if (iter % 20 == 0) {
      std::cout << "Iter " << iter << " | q_error: " << q_err 
                << " | step elapsed: " << solver.elapsed << " s\n";
    }
    if (q_err < 0.03) {
      std::cout << "Target reached at iteration " << iter << "!\n";
      break;
    }
  }
  return 0;
}
