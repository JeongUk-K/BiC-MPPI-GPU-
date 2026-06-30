#include <svgd_mppi_gpu.cuh>
#include <manipulator.h>
#include <mppi_vis_logger.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Cylinder {
  double cx, cy, r, z_min, z_max;
};

std::vector<Cylinder> loadCylinders(const std::string &path) {
  std::vector<Cylinder> cyls;
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "Warning: cannot open " << path << "\n";
    return cyls;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    Cylinder c;
    if (ss >> c.cx >> c.cy >> c.r >> c.z_min >> c.z_max)
      cyls.push_back(c);
  }
  return cyls;
}

double cylinderCost(double px, double py, double pz,
                    const std::vector<Cylinder> &cyls) {
  double cost = 0.0;
  for (const auto &c : cyls) {
    if (pz < c.z_min - 0.05 || pz > c.z_max + 0.05) continue;
    double dx = px - c.cx, dy = py - c.cy;
    double dist = std::sqrt(dx*dx + dy*dy);
    const double death_r = c.r + 0.12;
    const double warn_r  = c.r + 0.30;
    if (dist < death_r) {
      cost += 500000.0;
    } else if (dist < warn_r) {
      double excess = warn_r - dist;
      cost += 200000.0 * excess * excess;

      // Z-up helper cost
      double target_z = c.z_max + 0.15;
      if (pz < target_z) {
        cost += 1000.0 * (target_z - pz);
      }
    }
  }
  return cost;
}

int main(int argc, char **argv) {
  std::cout << "Starting Manipulator GPU SVGD-MPPI..." << std::endl;

  std::string cyl_file = "../obstacles/cylinder_test.txt";
  if (argc >= 2) cyl_file = argv[1];
  auto cyls = loadCylinders(cyl_file);
  std::cout << "Loaded " << cyls.size() << " cylinder(s) from " << cyl_file << "\n";

  CollisionChecker collision_checker = CollisionChecker();
  for (const auto &c : cyls)
    collision_checker.addCylinder(c.cx, c.cy, c.r, c.z_min, c.z_max);

  auto model = Manipulator();

  // Override stage cost for cylinder avoidance
  model.q = [&cyls, &model](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u) -> double {
    double ctrl = u.norm();
    double jlim = 0.0;
    for (int i = 0; i < model.dof; ++i) {
      double qr = (model.q_max(i) - model.q_min(i)) * 0.5;
      double qc = (model.q_max(i) + model.q_min(i)) * 0.5;
      double norm = (x(i) - qc) / (qr * 0.8);
      if (std::abs(norm) > 1.0)
        jlim += 5.0 * (std::abs(norm) - 1.0) * (std::abs(norm) - 1.0);
    }
    double obs = 0.0;
    if (!cyls.empty()) {
      auto T_fk = model.computeForwardKinematics(x);
      for (int k = 0; k < model.dof; ++k) {
        Eigen::Vector3d p_start = (k == 0) ? Eigen::Vector3d(0, 0, 0) : T_fk[k-1].block<3,1>(0,3);
        Eigen::Vector3d p_end = T_fk[k].block<3,1>(0,3);
        const int num_samples = 15;
        for (int s = 0; s <= num_samples; ++s) {
          double alpha = (double)s / num_samples;
          Eigen::Vector3d p = p_start * (1.0 - alpha) + p_end * alpha;
          obs += cylinderCost(p.x(), p.y(), p.z(), cyls);
        }
      }
    }
    return ctrl + jlim + obs;
  };

  using Solver = SVGDMPPI_GPU;
  using SolverParam = SVGDMPPIParam;

  SolverParam param;
  param.dt = 0.1;
  param.Tf = 40;
  param.Tb = 40;
  param.x_init.resize(model.dim_x);
  param.x_init.setZero();
  param.x_target.resize(model.dim_x);
  param.x_target.setZero();
  param.x_target(0) = M_PI / 2.0;

  param.Nf = 200;
  param.Nb = 200;
  param.Ns = 10;
  param.istep = 5;
  param.Nr = 2000;
  param.gamma_u = 0.1;

  Eigen::VectorXd sigma_u(model.dim_u);
  sigma_u << 0.5, 0.3, 0.3, 0.2, 0.2, 0.2;
  param.sigma_u = sigma_u.asDiagonal();
  param.deviation_mu = 1.0;
  param.cost_mu = 1.0;
  param.minpts = 5;
  param.epsilon = 0.5;
  param.psi = 0.3;

  int maxiter = 300;
  double tolerance = 0.15;

  MPPIVisLogger vis_logger;
  vis_logger.enabled = true;
  std::vector<std::vector<double>> empty_map;
  vis_logger.init("manipulator_svgd_mppi", model.dim_x, param.Tf, 0.1, empty_map, param.x_init, param.x_target);

  Solver solver(model);
  solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
  solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
  solver.init(param);
  solver.setCollisionChecker(&collision_checker);
  solver.setVisLogger(&vis_logger);

  std::ofstream csv_file("result_manipulator_gpu_svgd_mppi.csv");
  csv_file << "Iter";
  for (int j = 0; j < model.dim_x; ++j) csv_file << ",q_" << j;
  csv_file << ",err,elapsed\n";

  bool is_reached = false;
  double total_elapsed = 0.0;
  double f_err = 0.0;
  int i = 0;

  for (i = 0; i < maxiter; ++i) {
    vis_logger.beginStep(i);
    solver.solve();
    solver.move();
    total_elapsed += solver.elapsed;
    f_err = (solver.x_init - param.x_target).norm();

    csv_file << i;
    for (int j = 0; j < model.dim_x; ++j) csv_file << "," << solver.x_init(j);
    csv_file << "," << f_err << "," << solver.elapsed << "\n";

    if (f_err < tolerance) { is_reached = true; break; }

    if (i % 50 == 0)
      std::cout << "Iter: " << i << " | q0: " << solver.x_init(0)
                << " | Err: " << f_err << " | Elapsed: " << total_elapsed << "\n";
  }

  csv_file.close();
  std::cout << "====================================\n"
            << "Success: " << is_reached << "\t Iters: " << i
            << "\t Total Time: " << total_elapsed
            << "\t Final Err: " << f_err << "\n";
  return 0;
}
