#pragma once

#include <Eigen/Dense>

struct BiMPPIParam {
  float dt = 0.02f;
  int Tf = 80;
  int Tb = 80;
  int Nf = 1024;
  int Nb = 1024;
  int Nr = 1024;

  // Inverse temperature for the MPPI exponential weight.
  double gamma_u = 0.001;

  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd sigma_u;

  // DBSCAN feature scaling parameters used by the uploaded BiMPPI_GPU.
  double deviation_mu = 1.0;
  double cost_mu = 1.0;
  double epsilon = 0.5;
  int minpts = 8;

  // Reserved for compatibility with existing code.
  double psi = 0.0;
};
