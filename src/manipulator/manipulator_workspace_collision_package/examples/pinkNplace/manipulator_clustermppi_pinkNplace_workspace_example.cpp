#include "cluster_mppi_gpu.cuh"
#include "manipulator_pinkNplace_common.h"

int main() {
  PinkNPlaceMppiConfig cfg;
  cfg.solver_label = "Cluster-MPPI";
  cfg.csv_prefix = "manipulator_pinkNplace_clustermppi_";
  cfg.seed = 260633ULL;

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

  const double cluster_deviation_mu = 1.0;
  const double cluster_epsilon = 4.2;
  const int cluster_minpts = 8;

  cfg.warm_start_kp = 20.0;
  cfg.warm_start_kd = 7.5;
  cfg.max_total_steps = 650;

  return runPinkNPlaceMppiLikeExample<ClusterMPPI_GPU>(
      cfg, [&](ClusterMPPI_GPU &solver) {
        solver.setClusterParams(cluster_deviation_mu, cluster_epsilon,
                                cluster_minpts);
      });
}
