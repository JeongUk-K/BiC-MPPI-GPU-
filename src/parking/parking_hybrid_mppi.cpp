#include "hybrid_parking_problem.h"

#include <hybrid_parking.h>
#include <mppi.h>

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  auto model = HybridParking();
  const std::uint_fast64_t seed =
      argc > 1 ? static_cast<std::uint_fast64_t>(std::stoull(argv[1])) : 1;

  using Solver = MPPI;
  using SolverParam = MPPIParam;

  SolverParam param;
  param.dt = 0.08;
  param.T = 70;
  param.x_init = hybridParkingStart();
  param.x_target = hybridParkingTarget();
  param.N = 1600;
  param.gamma_u = 1.0;

  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.70, 0.55;
  param.sigma_u = sigma_u.asDiagonal();

  const int maxiter = 120;
  const std::string solver_name = "mppi";

  std::ofstream result_csv("result_parking_hybrid_mppi.csv");
  writeHybridParkingResultHeader(result_csv);

  std::ofstream path_csv("path_parking_hybrid_mppi.csv");
  writeHybridParkingPathHeader(path_csv);
  path_csv << -1 << "," << param.x_init(0) << "," << param.x_init(1) << ","
           << param.x_init(2) << "," << param.x_init(3) << ","
           << param.x_init(4) << "\n";

  std::ofstream control_csv("control_parking_hybrid_mppi.csv");
  writeHybridParkingControlHeader(control_csv);

  CollisionChecker collision_checker;
  configureHybridParkingObstacles(collision_checker);

  Solver solver(model);
  solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
  solver.init(param);
  solver.setSeed(seed);
  solver.setCollisionChecker(&collision_checker);

  bool is_success = false;
  bool is_failed = false;
  int iter = 0;
  double total_elapsed = 0.0;
  ParkingMetrics metrics = calculateParkingMetrics(solver.x_init, param.x_target);

  for (iter = 0; iter < maxiter; ++iter) {
    solver.solve();
    const Eigen::VectorXd u = solver.Uo.col(0);
    control_csv << iter << "," << u(0) << "," << u(1) << "\n";

    solver.move();

    path_csv << iter << "," << solver.x_init(0) << "," << solver.x_init(1)
             << "," << solver.x_init(2) << "," << solver.x_init(3) << ","
             << solver.x_init(4) << "\n";

    total_elapsed += solver.elapsed;
    metrics = calculateParkingMetrics(solver.x_init, param.x_target);

    if (collision_checker.getCollisionGrid(solver.x_init)) {
      is_failed = true;
      break;
    }
    if (isHybridParkingSolved(metrics)) {
      is_success = true;
      break;
    }
  }

  const double score = hybridParkingScore(metrics);
  result_csv << solver_name << "," << is_failed << "," << is_success << ","
             << iter << "," << total_elapsed << "," << score << ","
             << metrics.pose_error << "," << metrics.position_error << ","
             << metrics.heading_error << "," << metrics.forward_distance << ","
             << metrics.reverse_distance << "," << metrics.missing_forward
             << "," << metrics.missing_reverse << "\n";

  std::cout << solver_name << '\t' << is_failed << '\t' << is_success << '\t'
            << iter << '\t' << total_elapsed << '\t' << score << '\t'
            << metrics.forward_distance << '\t' << metrics.reverse_distance
            << std::endl;

  return 0;
}
