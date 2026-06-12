#pragma once

#include "stitched_barn_common.h"

namespace wmrobot_stitched {

inline Config makeCommonExperimentConfig() {
  Config cfg;

  // --------------------------------------------------------------------------
  // Edit shared map / waypoint / experiment parameters here.
  // --------------------------------------------------------------------------
  cfg.out_dir = "../results/wmrobot_waypoint_stitched";
  cfg.dataset_dir = "../BARN_dataset/txt_files";
  cfg.dataset_maps = 300;
  cfg.scenarios = 10;
  cfg.maps_per_scenario = 5;
  cfg.seed = 7;

  cfg.maxiter = 500;
  cfg.dt = 0.1;
  cfg.resolution = 0.1;

  cfg.start_x = 1.5;
  cfg.map_length = 5.0;
  cfg.waypoint_count = 6;
  cfg.waypoint_tolerance = 0.45;

  cfg.save_rollouts = true;
  cfg.vis_every = 10;

  return cfg;
}

inline Config makeMppiConfig() {
  Config cfg = makeCommonExperimentConfig();

  // MPPI uses one global target at the end of the 5 stitched maps.
  cfg.global_T = 500;
  cfg.N = 3000;

  return cfg;
}

inline Config makeLogMppiConfig() {
  Config cfg = makeCommonExperimentConfig();

  // Log-MPPI uses one global target at the end of the 5 stitched maps.
  cfg.global_T = 500;
  cfg.N = 3000;

  return cfg;
}

inline Config makeClusterMppiConfig() {
  Config cfg = makeCommonExperimentConfig();

  // Cluster-MPPI uses one global target at the end of the 5 stitched maps.
  cfg.global_T = 500;
  cfg.N = 3000;

  return cfg;
}

inline Config makeBiMppiConfig() {
  Config cfg = makeCommonExperimentConfig();

  // BiC-MPPI uses waypoint targets with bidirectional parallel rollouts.
  // Each waypoint connection rolls out forward and backward with these horizons.
  cfg.bic_Tf = 50;
  cfg.bic_Tb = 50;
  cfg.Nf = 3000;
  cfg.Nb = 3000;
  cfg.Nr = 1500;

  return cfg;
}

} // namespace wmrobot_stitched
