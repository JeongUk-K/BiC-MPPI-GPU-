#include "log_mppi_gpu.cuh"
#include "manipulator_pinkNplace_common.h"

int main() {
  PinkNPlaceMppiConfig cfg;
  cfg.solver_label = "Log-MPPI";
  cfg.csv_prefix = "manipulator_pinkNplace_logmppi_";
  cfg.seed = 260632ULL;

  cfg.dt = 0.02f;
  cfg.horizon = 95;
  cfg.samples = 1024;
  cfg.gamma_u = 0.0012;

  cfg.sigma_q1 = 5.0;
  cfg.sigma_q2 = 5.0;
  cfg.sigma_q3 = 4.2;
  cfg.sigma_q4 = 2.5;
  cfg.sigma_q5 = 2.0;
  cfg.sigma_q6 = 1.4;

  cfg.warm_start_kp = 20.0;
  cfg.warm_start_kd = 7.5;
  cfg.max_total_steps = 650;

  return runPinkNPlaceMppiLikeExample<LogMPPI_GPU>(
      cfg, [](LogMPPI_GPU &) {});
}
