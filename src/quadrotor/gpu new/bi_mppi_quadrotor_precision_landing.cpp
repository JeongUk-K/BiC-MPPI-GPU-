#include <bi_mppi_gpu.cuh>
#include <quadrotor.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#include "quadrotor_path_logger.h"

namespace {

double quietNaN() {
  return std::numeric_limits<double>::quiet_NaN();
}

double clamp01(double value) {
  if (value < 0.0) {
    return 0.0;
  }
  if (value > 1.0) {
    return 1.0;
  }
  return value;
}

}  // namespace

int main() {
  auto model = Quadrotor();

  using Solver = BiMPPI_GPU;
  using SolverParam = BiMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 50;
  param.Tb = 50;

  param.x_init.resize(model.dim_x);
  param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;

  param.x_target.resize(model.dim_x);
  param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;

  param.Nf = 10000;
  param.Nb = 10000;
  param.Nr = 5000;
  param.gamma_u = 10.0;
  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 1.5, 1.5, 1.5;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.01;
  param.psi = 0.6;

  const int maxiter = 200;

  // Precision landing thresholds.
  // Plane arrival is evaluated by crossing p_z = 0 without collision.
  // Precision landing additionally requires the interpolated touchdown state
  // to be close to the target and slow enough at touchdown.
  const double eps_xy = 0.30;   // [m]
  const double eps_vxy = 0.50;  // [m/s]
  const double eps_vz = 1.00;   // [m/s]

  std::ofstream csv("result_quadrotor_bi_mppi.csv");
  csv << "map,is_failed,is_landed,iter,elapsed,elapsed_rollout,"
         "elapsed_clustering,elapsed_connection,elapsed_guide,f_err,"
         "is_collision,is_arrived_plane,is_precision_landed,"
         "per_step_elapsed,per_step_rollout,per_step_clustering,"
         "per_step_connection,per_step_guide,"
         "touchdown_x,touchdown_y,touchdown_z,"
         "touchdown_vx,touchdown_vy,touchdown_vz,"
         "touchdown_xy_error,touchdown_vxy,touchdown_vz_abs,"
         "touchdown_speed\n";

  std::ofstream path_csv("path_quadrotor_bi_mppi.csv");
  writeQuadrotorPathHeader(path_csv);

  for (int map = 299; map >= 0; --map) {
    CollisionChecker collision_checker = CollisionChecker();
    collision_checker.loadMap("../BARN_dataset/txt_files/output_" +
                                  std::to_string(map) + ".txt",
                              0.1);

    Solver solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);

    // Quadrotor: gravity compensation on thrust channel.
    solver.U_f0.row(2).array() += model.g;
    solver.U_b0.row(2).array() += model.g;

    solver.init(param);
    solver.setCollisionChecker(&collision_checker);
    solver.dummy_u(2) += model.g;

    writeQuadrotorPathRow(path_csv, "bi_mppi", map, 0, solver.x_init);

    bool is_collision = false;
    bool is_arrived_plane = false;
    bool is_precision_landed = false;

    // Backward-compatible fields:
    // - is_landed is equivalent to landing-plane arrival.
    // - is_failed is evaluated under the stricter precision-landing criterion.
    bool is_landed = false;
    bool is_failed = true;

    int steps_executed = 0;
    double total_elapsed = 0.0;
    double total_rollout = 0.0;
    double total_clustering = 0.0;
    double total_connection = 0.0;
    double total_guide = 0.0;

    double f_err = quietNaN();

    Eigen::VectorXd x_touch =
        Eigen::VectorXd::Constant(model.dim_x, quietNaN());
    double touchdown_xy_error = quietNaN();
    double touchdown_vxy = quietNaN();
    double touchdown_vz_abs = quietNaN();
    double touchdown_speed = quietNaN();

    for (int i = 0; i < maxiter; ++i) {
      const Eigen::VectorXd x_prev = solver.x_init;

      solver.solve();
      solver.move();

      const Eigen::VectorXd x_next = solver.x_init;
      steps_executed = i + 1;

      writeQuadrotorPathRow(path_csv, "bi_mppi", map, steps_executed, x_next);

      total_elapsed += solver.elapsed;
      total_rollout += solver.elapsed_rollout;
      total_clustering += solver.elapsed_clustering;
      total_connection += solver.elapsed_connection;
      total_guide += solver.elapsed_guide;

      // Collision is checked on the propagated state first. If the state
      // crosses the landing plane, an additional touchdown-state collision
      // check is performed after interpolation.
      if (collision_checker.getCollisionGrid(x_next)) {
        is_collision = true;
        is_failed = true;
        break;
      }

      const bool crossed_ground = (x_prev(2) > 0.0 && x_next(2) <= 0.0);
      if (!crossed_ground) {
        continue;
      }

      // Interpolate the touchdown state at p_z = 0 instead of evaluating the
      // already-below-ground state x_next.
      double alpha = x_prev(2) / (x_prev(2) - x_next(2));
      alpha = clamp01(alpha);
      x_touch = x_prev + alpha * (x_next - x_prev);
      x_touch(2) = 0.0;

      if (collision_checker.getCollisionGrid(x_touch)) {
        is_collision = true;
        is_failed = true;
        break;
      }

      touchdown_xy_error =
          (x_touch.head(2) - param.x_target.head(2)).norm();
      touchdown_vxy = x_touch.segment(3, 2).norm();
      touchdown_vz_abs = std::abs(x_touch(5));
      touchdown_speed = x_touch.segment(3, 3).norm();

      f_err = touchdown_xy_error;

      is_arrived_plane = true;
      is_landed = true;

      is_precision_landed =
          (touchdown_xy_error < eps_xy) &&
          (touchdown_vxy < eps_vxy) &&
          (touchdown_vz_abs < eps_vz);

      is_failed = !is_precision_landed;
      break;
    }

    const double denom = static_cast<double>(std::max(1, steps_executed));
    const double per_step_elapsed = total_elapsed / denom;
    const double per_step_rollout = total_rollout / denom;
    const double per_step_clustering = total_clustering / denom;
    const double per_step_connection = total_connection / denom;
    const double per_step_guide = total_guide / denom;

    std::cout << map << '\t'
              << is_failed << '\t'
              << is_landed << '\t'
              << is_precision_landed << '\t'
              << steps_executed << '\t'
              << total_elapsed << std::endl;

    csv << map << ','
        << is_failed << ','
        << is_landed << ','
        << steps_executed << ','
        << total_elapsed << ','
        << total_rollout << ','
        << total_clustering << ','
        << total_connection << ','
        << total_guide << ','
        << f_err << ','
        << is_collision << ','
        << is_arrived_plane << ','
        << is_precision_landed << ','
        << per_step_elapsed << ','
        << per_step_rollout << ','
        << per_step_clustering << ','
        << per_step_connection << ','
        << per_step_guide << ','
        << x_touch(0) << ','
        << x_touch(1) << ','
        << x_touch(2) << ','
        << x_touch(3) << ','
        << x_touch(4) << ','
        << x_touch(5) << ','
        << touchdown_xy_error << ','
        << touchdown_vxy << ','
        << touchdown_vz_abs << ','
        << touchdown_speed
        << '\n';
  }

  csv.close();
  path_csv.close();
  return 0;
}
