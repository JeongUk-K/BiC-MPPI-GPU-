#include "bi_mppi_gpu.cuh"
#include "manipulator_pinkNplace_common.h"

int main() {
  PinkNPlaceBiMppiConfig cfg;
  cfg.solver_label = "BiC-MPPI";
  cfg.csv_prefix = "manipulator_pinkNplace_bicmppi_";
  cfg.seed = 260634ULL;

  cfg.dt = 0.02f;
  cfg.Tf = 95;
  cfg.Tb = 95;
  cfg.Nf = 1024;
  cfg.Nb = 1024;
  cfg.Nr = 1024;
  cfg.gamma_u = 0.0012;

  cfg.sigma_q1 = 5.0;
  cfg.sigma_q2 = 5.0;
  cfg.sigma_q3 = 4.2;
  cfg.sigma_q4 = 2.5;
  cfg.sigma_q5 = 2.0;
  cfg.sigma_q6 = 1.4;

  cfg.deviation_mu = 1.0;
  cfg.cost_mu = 1.0;
  cfg.epsilon = 4.2;
  cfg.minpts = 8;
  cfg.psi = 0.0;

  cfg.warm_start_kp = 20.0;
  cfg.warm_start_kd = 7.5;
  cfg.max_total_steps = 650;

  return runPinkNPlaceBiMppiExample<BiMPPI_GPU>(cfg);
}
