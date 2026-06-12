#include <svgd_mppi_gpu.cuh>
#include <wmrobot_map.h>

#include "wmrobot_gpu_stats.h"

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  auto model = WMRobotMap();

  using Solver = SVGDMPPI_GPU;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 50;
  param.Tb = 50;
  param.x_init.resize(model.dim_x);
  param.x_init << 2.5, 0.0, M_PI_2;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, M_PI_2;

  param.Nf = 200;
  param.Nb = 200;
  param.Ns = 10;
  param.istep = 5;

  param.Nr = 5000;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.6, 0.6;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;

  int maxiter = 200;

  const std::string variant = "SVGD-MPPI";
  std::vector<WmrobotGpuRunResult> runs;

  std::ofstream csv("result_svgd_mppi.csv");
  writeWmrobotGpuRunHeader(csv);

  for (int s = 0; s < 2; ++s) {
    switch (s) {
    case 0:
      param.x_init(0) = 0.5;
      break;
    case 1:
      param.x_init(0) = 2.5;
      break;
    default:
      break;
    }

    for (int map = 299; map >= 0; --map) {
      CollisionChecker collision_checker = CollisionChecker();
      collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                    std::to_string(map) + ".txt",
                                0.1);

      Solver solver(model);
      solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
      solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
      solver.init(param);
      solver.setCollisionChecker(&collision_checker);

      WmrobotGpuRunResult row;
      row.variant = variant;
      row.start_case = s;
      row.map_id = map;

      for (row.iter = 0; row.iter < maxiter; ++row.iter) {
        solver.solve();
        row.d_conn = solver.connectionDistance();
        solver.move();
        row.elapsed += solver.elapsed;
        row.elapsed_rollout += solver.elapsed_rollout;
        row.elapsed_clustering += solver.elapsed_clustering;
        row.elapsed_connection += solver.elapsed_connection;
        row.elapsed_guide += solver.elapsed_guide;

        if (collision_checker.getCollisionGrid(solver.x_init)) {
          row.is_collision = true;
          break;
        } else {
          row.d_goal = (solver.x_init - param.x_target).norm();
          if (row.d_goal < 0.1) {
            row.is_success = true;
            break;
          }
        }
      }
      if (row.iter >= maxiter) {
        row.iter = maxiter;
      }
      printWmrobotGpuRunRow(std::cout, row);
      writeWmrobotGpuRunRow(csv, row);
      runs.push_back(row);
    }
  }

  csv.close();

  const auto summary = summarizeWmrobotGpuRuns(variant, runs);
  std::ofstream summary_csv("result_svgd_mppi_summary.csv");
  writeWmrobotGpuSummaryHeader(summary_csv);
  writeWmrobotGpuSummaryRow(summary_csv, summary);
  return 0;
}
