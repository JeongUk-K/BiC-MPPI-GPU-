#include <quadrotor.h>
#include <svgd_mppi_gpu.cuh>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>

#include "quadrotor_path_logger.h"
#include "quadrotor_gpu_vis.h"

int main(int argc, char **argv) {
  const auto config = parseQuadrotorGpuArgs(argc, argv, true);
  auto model = Quadrotor();

  using Solver = SVGDMPPI_GPU;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = config.Tf;
  param.Tb = config.Tb;
  param.x_init.resize(model.dim_x);
  param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;

  param.Nf = config.Nf;
  param.Nb = config.Nb;
  param.Ns = config.Ns;
  param.istep = config.istep;

  param.Nr = config.Nr;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 1.5, 1.5, 1.5;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;

  int maxiter = config.maxiter;

  std::ofstream csv("result_quadrotor_svgd_mppi.csv");
  csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,elapsed_"
         "clustering,"
         "elapsed_connection,elapsed_guide,f_err\n";
  std::ofstream path_csv("path_quadrotor_svgd_mppi.csv");
  writeQuadrotorPathHeader(path_csv);
  std::ofstream progress_csv("result_quadrotor_svgd_mppi_progress.csv");
  csv << std::flush;
  progress_csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,elapsed_"
                  "clustering,"
                  "elapsed_connection,elapsed_guide,f_err\n";
  progress_csv << std::flush;

  for (int map = config.map_begin;
       map >= 0 && map > config.map_begin - config.num_maps; --map) {
    CollisionChecker collision_checker = CollisionChecker();
    collision_checker.loadMap(config.dataset_dir + "/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    Solver solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
    // quadrotor: gravity compensation on thrust channel
    solver.U_f0.row(2).array() += model.g;
    solver.U_b0.row(2).array() += model.g;
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);
    solver.dummy_u(2) += model.g;

    MPPIVisLogger vis_logger;
    initQuadrotorGpuVisLogger(vis_logger, config.save_rollouts, "svgd_mppi",
                              map, model.dim_x, param.Tf + param.Tb,
                              collision_checker, param.x_init,
                              param.x_target);
    solver.setVisLogger(&vis_logger);

    writeQuadrotorPathRow(path_csv, "svgd_mppi", map, 0, solver.x_init);

    bool is_landed = false;
    bool is_failed = true;
    int i = 0;
    double total_elapsed = 0.0;
    double total_rollout = 0.0;
    double total_clustering = 0.0;
    double total_connection = 0.0;
    double total_guide = 0.0;
    double f_err = 0.0;

    for (i = 0; i < maxiter; ++i) {
      beginQuadrotorGpuVisStep(vis_logger, config.save_rollouts,
                               config.vis_every, i);
      solver.solve();
      endQuadrotorGpuVisStep(vis_logger);
      solver.move();
      writeQuadrotorPathRow(path_csv, "svgd_mppi", map, i + 1,
                            solver.x_init);
      total_elapsed += solver.elapsed;
      total_rollout += solver.elapsed_rollout;
      total_clustering += solver.elapsed_clustering;
      total_connection += solver.elapsed_connection;
      total_guide += solver.elapsed_guide;

      f_err = (solver.x_init.head(2) - param.x_target.head(2)).norm();
      if (collision_checker.getCollisionGrid(solver.x_init)) {
        progress_csv << map << ',' << is_failed << ',' << is_landed << ','
                     << i << ',' << total_elapsed << ',' << total_rollout
                     << ',' << total_clustering << ',' << total_connection
                     << ',' << total_guide << ',' << f_err << '\n'
                     << std::flush;
        break;
      } else {
        if (solver.x_init(2) < 0) {
          is_landed = true;
          if (f_err < 0.3) {
            is_failed = false;
          }
          progress_csv << map << ',' << is_failed << ',' << is_landed << ','
                       << i << ',' << total_elapsed << ',' << total_rollout
                       << ',' << total_clustering << ',' << total_connection
                       << ',' << total_guide << ',' << f_err << '\n'
                       << std::flush;
          break;
        }
      }
      progress_csv << map << ',' << is_failed << ',' << is_landed << ',' << i
                   << ',' << total_elapsed << ',' << total_rollout << ','
                   << total_clustering << ',' << total_connection << ','
                   << total_guide << ',' << f_err << '\n'
                   << std::flush;
    }
    std::cout << map << '\t' << is_failed << '\t' << is_landed << '\t' << i
              << '\t' << total_elapsed << std::endl;
    csv << map << ',' << is_failed << ',' << is_landed << ',' << i << ','
        << total_elapsed << ',' << total_rollout << ',' << total_clustering
        << ',' << total_connection << ',' << total_guide << ',' << f_err
        << '\n'
        << std::flush;
  }

  csv.close();
  path_csv.close();
  progress_csv.close();
  return 0;
}
