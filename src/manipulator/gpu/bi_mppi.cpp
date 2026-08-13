// Manipulator BiC-MPPI — uses the generic cuda-accel BiMPPI_GPU solver
#include "manipulator_model.h"
#include <bi_mppi_gpu.cuh>

#include <Eigen/Dense>
#include <iostream>

int main() {
  ManipulatorModel model;
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

  BiMPPIParam param;
  param.dt = 0.02f;
  param.Tf = 45;
  param.Tb = 45;
  param.Nf = 4096;
  param.Nb = 4096;
  param.Nr = 4096;
  param.gamma_u = 0.0015;
  param.clustering_method = ClusteringMethod::KMeans;
  param.x_init = Eigen::VectorXd::Zero(ManipulatorModel::kDof * 2);
  param.x_init << 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  param.x_target = Eigen::VectorXd::Zero(ManipulatorModel::kDof * 2);
  param.x_target << 0.5, -0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  Eigen::VectorXd sigma_u(ManipulatorModel::kDof);
  sigma_u << 4.5, 4.5, 3.8, 2.2, 1.8, 1.2;
  param.sigma_u = sigma_u.asDiagonal();

  BiMPPI_GPU solver(model);
  solver.init(param);

  std::cout << "[Manipulator BiC-MPPI] Initialized solver successfully.\n";
  for (int iter = 0; iter < 200; ++iter) {
    solver.solve();
    solver.move();
    const double q_err =
        (solver.x_init.head(6) - param.x_target.head(6)).norm();
    if (iter % 20 == 0)
      std::cout << "Iter " << iter << " | q_error: " << q_err
                << " | elapsed: " << solver.elapsed << " s\n";
    if (q_err < 0.03) {
      std::cout << "Target reached at iteration " << iter << "!\n";
      break;
    }
  }
  return 0;
}
