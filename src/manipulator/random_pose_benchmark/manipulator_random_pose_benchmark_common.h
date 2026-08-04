#pragma once

#include "collision_checker.h"
#include "manipulator_bicmppi_utils.h"
#include "manipulator_dynamics_model.h"
#include "manipulator_pose_set.h"
#include "mppi_param.h"

#include <Eigen/Dense>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <zstd.h>

namespace manipulator_random_pose_benchmark {

using JointNoiseSigma = std::array<double, ManipulatorDynamicsModel::kDof>;

struct RandomPoseBenchmarkSolverParams {
  struct Solver {
    float dt = 0.02f;
    int forward_horizon = 90;
    int forward_samples = 4096;
    int backward_horizon = 90;
    int backward_samples = 4096;
    int reverse_samples = 4096;
    double gamma_u = 0.0015;
    JointNoiseSigma sigma = {4.5, 4.5, 3.8, 2.2, 1.8, 1.2};
    ClusteringMethod clustering_method = ClusteringMethod::DBSCAN;
    int kmeans_clusters = 5;
    int kmeans_max_iterations = 100;
    double kmeans_threshold = 1e-6;

    double cluster_deviation_mu = 1.0;
    double cluster_epsilon = 3.8;
    int cluster_minpts = 8;

    double bic_deviation_mu = 1.0;
    double bic_cost_mu = 1.0;
    double bic_epsilon = 3.8;
    int bic_minpts = 8;
    double bic_psi = 0.0;
  };

  Solver mppi;
  Solver logmppi;
  Solver clustermppi;
  Solver bicmppi;
};

using SolverParams = RandomPoseBenchmarkSolverParams::Solver;

// Tune all random-pose benchmark solver parameters here.
inline RandomPoseBenchmarkSolverParams randomPoseSolverParams() {
  RandomPoseBenchmarkSolverParams params;

  params.mppi.dt = 0.02f;
  params.mppi.forward_horizon = 90;
  params.mppi.forward_samples = 4096;
  params.mppi.gamma_u = 0.0015;
  params.mppi.sigma = {4.5, 4.5, 3.8, 2.2, 1.8, 1.2};

  params.logmppi = params.mppi;

  params.clustermppi = params.mppi;
  params.clustermppi.cluster_deviation_mu = 1.0;
  params.clustermppi.cluster_epsilon = 3.8;
  params.clustermppi.cluster_minpts = 8;
  params.clustermppi.clustering_method = ClusteringMethod::KMeans;

  params.bicmppi.dt = 0.02f;
  params.bicmppi.forward_horizon = 45;
  params.bicmppi.backward_horizon = 45;
  params.bicmppi.forward_samples = 4096;
  params.bicmppi.backward_samples = 4096;
  params.bicmppi.reverse_samples = 4096;
  params.bicmppi.gamma_u = 0.0015;
  params.bicmppi.sigma = {4.5, 4.5, 3.8, 2.2, 1.8, 1.2};
  params.bicmppi.bic_deviation_mu = 1.0;
  params.bicmppi.bic_cost_mu = 1.0;
  params.bicmppi.bic_epsilon = 3.8;
  params.bicmppi.bic_minpts = 8;
  params.bicmppi.bic_psi = 0.0;
  params.bicmppi.clustering_method = ClusteringMethod::KMeans;

  return params;
}

struct BenchmarkConfig {
  static constexpr int kReferencePoseCount = 16;
  int scenario_count = kReferencePoseCount * (kReferencePoseCount - 1);
  std::uint_fast64_t rollout_seed_base = 260800ULL;
  double max_pair_q_distance = std::numeric_limits<double>::infinity();
  double min_pair_ee_distance = 0.0;

  double obstacle_half_x = 0.035;
  double obstacle_half_y = 0.035;
  double obstacle_half_z = 0.055;
  double obstacle_min_z = 0.05;

  int max_iter = 200;
  double q_tol = 0.03;  
  double ee_tol = 0.05;  
  double qdot_tol = 0.2;

  double warm_start_kp = 18.0;
  double warm_start_kd = 7.0;

  std::string output_root = "result/manipulator";
  bool save_rollout_ee = true;
  bool save_rollout_state = true;
  // Run all generated scenarios by default.
  std::vector<int> selected_scenario_ids;
};

inline BenchmarkConfig benchmarkConfigFromEnvironment() {
  BenchmarkConfig config;
  if (const char *value = std::getenv("MANIPULATOR_BENCHMARK_SCENARIOS")) {
    config.scenario_count = std::stoi(value);
    config.selected_scenario_ids.clear();
  }
  if (const char *value = std::getenv("MANIPULATOR_BENCHMARK_MAX_ITER")) {
    config.max_iter = std::stoi(value);
  }
  if (const char *value = std::getenv("MANIPULATOR_BENCHMARK_OUTPUT")) {
    config.output_root = value;
  }
  if (const char *value = std::getenv("MANIPULATOR_BENCHMARK_SAVE_EE")) {
    config.save_rollout_ee = std::string(value) != "0";
  }
  if (const char *value = std::getenv("MANIPULATOR_BENCHMARK_SAVE_STATE")) {
    config.save_rollout_state = std::string(value) != "0";
  }
  if (const char *value =
          std::getenv("MANIPULATOR_BENCHMARK_SCENARIO_IDS")) {
    config.selected_scenario_ids.clear();
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
      if (!token.empty()) config.selected_scenario_ids.push_back(std::stoi(token));
    }
    std::sort(config.selected_scenario_ids.begin(),
              config.selected_scenario_ids.end());
    if (std::adjacent_find(config.selected_scenario_ids.begin(),
                           config.selected_scenario_ids.end()) !=
        config.selected_scenario_ids.end()) {
      throw std::runtime_error("duplicate MANIPULATOR_BENCHMARK_SCENARIO_IDS");
    }
  }
  if (config.scenario_count <= 0 ||
      config.scenario_count > BenchmarkConfig::kReferencePoseCount *
                                  (BenchmarkConfig::kReferencePoseCount - 1)) {
    throw std::runtime_error("invalid MANIPULATOR_BENCHMARK_SCENARIOS");
  }
  if (config.max_iter <= 0) {
    throw std::runtime_error("invalid MANIPULATOR_BENCHMARK_MAX_ITER");
  }
  for (int id : config.selected_scenario_ids) {
    if (id < 0 || id >= config.scenario_count) {
      throw std::runtime_error("invalid MANIPULATOR_BENCHMARK_SCENARIO_IDS");
    }
  }
  return config;
}

struct BenchmarkScenario {
  int index = -1;
  std::uint_fast64_t rollout_seed = 0;
  manipulator_pose_set::PosePair pose_pair;
  ManipulatorBoxObstacle obstacle;
  bool direct_path_blocked = true;
};

struct BenchmarkResult {
  std::string solver_key;
  std::string solver_label;
  int scenario_index = -1;
  std::uint_fast64_t rollout_seed = 0;
  int init_idx = -1;
  int goal_idx = -1;
  std::string init_name;
  std::string goal_name;
  double obstacle_xmin = 0.0;
  double obstacle_xmax = 0.0;
  double obstacle_ymin = 0.0;
  double obstacle_ymax = 0.0;
  double obstacle_zmin = 0.0;
  double obstacle_zmax = 0.0;
  bool direct_path_blocked = false;
  bool init_collision = false;
  bool goal_collision = false;
  bool reached = false;
  bool success = false;
  int solve_iterations = 0;
  int executed_steps = 0;
  int max_iter = 0;
  double q_tol = 0.0;
  double ee_tol = 0.0;
  double qdot_tol = 0.0;
  double max_pair_q_distance = 0.0;
  double total_solver_elapsed = 0.0;
  double mean_solver_elapsed = 0.0;
  double total_rollout_elapsed = 0.0;
  double mean_rollout_elapsed = 0.0;
  double total_clustering_elapsed = 0.0;
  double mean_clustering_elapsed = 0.0;
  double total_connection_elapsed = 0.0;
  double mean_connection_elapsed = 0.0;
  double total_guide_elapsed = 0.0;
  double mean_guide_elapsed = 0.0;
  double wall_elapsed = 0.0;
  double final_q_error = 0.0;
  double final_ee_error = 0.0;
  double final_qdot_norm = 0.0;
  int collision_count = 0;
  double connection_distance = 0.0;
  int horizon = 0;
  int samples_per_branch = 0;
  int samples_total_per_iter = 0;
  std::string trajectory_csv;
};

inline std::string scenarioTag(int index) {
  std::ostringstream oss;
  oss << "scenario_" << std::setw(2) << std::setfill('0') << index;
  return oss.str();
}

inline std::filesystem::path rootPath(const BenchmarkConfig &config) {
  return std::filesystem::path(config.output_root);
}

inline std::filesystem::path solverStatsPath(const BenchmarkConfig &config,
                                             const std::string &solver_key) {
  return rootPath(config) /
         ("manipulator_random_pose_benchmark_" + solver_key + "_stats.csv");
}

inline std::filesystem::path scenarioCsvPath(const BenchmarkConfig &config) {
  return rootPath(config) / "manipulator_random_pose_benchmark_scenarios.csv";
}

inline std::filesystem::path trajectoryPath(const BenchmarkConfig &config,
                                            const std::string &solver_key,
                                            int scenario_index) {
  return rootPath(config) / "trajectories" / solver_key /
         (scenarioTag(scenario_index) + "_executed_x.csv");
}

inline std::filesystem::path rolloutEEPath(const BenchmarkConfig &config,
                                           const std::string &solver_key,
                                           int scenario_index) {
  return rootPath(config) / "rollout_ee" / solver_key /
         (scenarioTag(scenario_index) + "_ee_packed.bin.zst");
}

inline std::filesystem::path rolloutEEIndexPath(const BenchmarkConfig &config,
                                                const std::string &solver_key,
                                                int scenario_index) {
  return rootPath(config) / "rollout_ee" / solver_key /
         (scenarioTag(scenario_index) + "_index.csv");
}

inline std::filesystem::path rolloutStatePath(const BenchmarkConfig &config,
                                              const std::string &solver_key,
                                              int scenario_index) {
  return rootPath(config) / "rollout_state" / solver_key /
         (scenarioTag(scenario_index) + "_state_f16.bin.zst");
}

inline std::filesystem::path rolloutStateIndexPath(
    const BenchmarkConfig &config, const std::string &solver_key,
    int scenario_index) {
  return rootPath(config) / "rollout_state" / solver_key /
         (scenarioTag(scenario_index) + "_index.csv");
}

class RolloutEEWriter {
 public:
  RolloutEEWriter(const std::filesystem::path &data_path,
                  const std::filesystem::path &index_path)
      : data_(data_path, std::ios::binary), index_(index_path) {
    if (!data_ || !index_) {
      throw std::runtime_error("failed to open rollout EE output files");
    }
    context_ = ZSTD_createCCtx();
    if (!context_) throw std::runtime_error("failed to create Zstd context");
    checkZstd(ZSTD_CCtx_setParameter(context_, ZSTD_c_compressionLevel, 3));
    checkZstd(ZSTD_CCtx_setParameter(context_, ZSTD_c_checksumFlag, 1));
    output_buffer_.resize(ZSTD_CStreamOutSize());
    index_ << "iteration,batch,branch,rollout_count,point_count,dtype,"
              "coordinate_scale_m,layout,uncompressed_offset_bytes,"
              "uncompressed_bytes\n";
  }

  RolloutEEWriter(const RolloutEEWriter &) = delete;
  RolloutEEWriter &operator=(const RolloutEEWriter &) = delete;

  ~RolloutEEWriter() {
    if (context_) {
      try {
        finish();
      } catch (...) {
      }
      ZSTD_freeCCtx(context_);
    }
  }

  void setIteration(int iteration) {
    iteration_ = iteration;
    batch_ = 0;
  }

  void append(const std::string &branch,
              const std::vector<std::uint8_t> &positions,
              int rollout_count, int point_count) {
    const std::size_t byte_count = positions.size();
    index_ << iteration_ << ',' << batch_++ << ',' << branch << ','
           << rollout_count << ',' << point_count
           << ",mixed-i16-i8,0.001,delta-packed-rollout-time-xyz,"
           << uncompressed_offset_ << ','
           << byte_count << '\n';

    ZSTD_inBuffer input{positions.data(), byte_count, 0};
    while (input.pos < input.size) {
      ZSTD_outBuffer output{output_buffer_.data(), output_buffer_.size(), 0};
      checkZstd(ZSTD_compressStream2(context_, &output, &input,
                                    ZSTD_e_continue));
      data_.write(output_buffer_.data(),
                  static_cast<std::streamsize>(output.pos));
      if (!data_) throw std::runtime_error("failed to write rollout EE data");
    }
    uncompressed_offset_ += byte_count;
  }

 private:
  static void checkZstd(std::size_t code) {
    if (ZSTD_isError(code)) {
      throw std::runtime_error(std::string("Zstd error: ") +
                               ZSTD_getErrorName(code));
    }
  }

  void finish() {
    if (finished_) return;
    ZSTD_inBuffer input{nullptr, 0, 0};
    std::size_t remaining = 1;
    while (remaining != 0) {
      ZSTD_outBuffer output{output_buffer_.data(), output_buffer_.size(), 0};
      remaining = ZSTD_compressStream2(context_, &output, &input, ZSTD_e_end);
      checkZstd(remaining);
      data_.write(output_buffer_.data(),
                  static_cast<std::streamsize>(output.pos));
    }
    data_.flush();
    index_.flush();
    finished_ = true;
  }

  std::ofstream data_;
  std::ofstream index_;
  ZSTD_CCtx *context_ = nullptr;
  std::vector<char> output_buffer_;
  std::uint64_t uncompressed_offset_ = 0;
  int iteration_ = -1;
  int batch_ = 0;
  bool finished_ = false;
};

class RolloutStateWriter {
 public:
  RolloutStateWriter(const std::filesystem::path &data_path,
                     const std::filesystem::path &index_path)
      : data_(data_path, std::ios::binary), index_(index_path) {
    if (!data_ || !index_) {
      throw std::runtime_error("failed to open rollout state output files");
    }
    context_ = ZSTD_createCCtx();
    if (!context_) throw std::runtime_error("failed to create Zstd context");
    index_ << "iteration,batch,branch,rollout_count,point_count,state_dim,"
              "dtype,layout,compressed_offset_bytes,compressed_bytes,"
              "uncompressed_offset_bytes,uncompressed_bytes\n";
  }

  RolloutStateWriter(const RolloutStateWriter &) = delete;
  RolloutStateWriter &operator=(const RolloutStateWriter &) = delete;

  ~RolloutStateWriter() {
    if (context_) ZSTD_freeCCtx(context_);
  }

  void setIteration(int iteration) {
    iteration_ = iteration;
    batch_ = 0;
  }

  void append(const std::string &branch,
              const std::vector<std::uint16_t> &states,
              int rollout_count, int point_count, int state_dim) {
    const std::size_t byte_count = states.size() * sizeof(std::uint16_t);
    output_buffer_.resize(ZSTD_compressBound(byte_count));
    checkZstd(ZSTD_CCtx_setParameter(context_, ZSTD_c_compressionLevel, 3));
    checkZstd(ZSTD_CCtx_setParameter(context_, ZSTD_c_checksumFlag, 1));
    const std::size_t compressed_bytes =
        ZSTD_compress2(context_, output_buffer_.data(), output_buffer_.size(),
                       states.data(), byte_count);
    checkZstd(compressed_bytes);
    index_ << iteration_ << ',' << batch_++ << ',' << branch << ','
           << rollout_count << ',' << point_count << ',' << state_dim
           << ",float16,rollout-time-state," << compressed_offset_ << ','
           << compressed_bytes << ',' << uncompressed_offset_ << ','
           << byte_count << '\n';
    data_.write(output_buffer_.data(),
                static_cast<std::streamsize>(compressed_bytes));
    if (!data_) throw std::runtime_error("failed to write rollout state data");
    compressed_offset_ += compressed_bytes;
    uncompressed_offset_ += byte_count;
  }

 private:
  static void checkZstd(std::size_t code) {
    if (ZSTD_isError(code)) {
      throw std::runtime_error(std::string("Zstd error: ") +
                               ZSTD_getErrorName(code));
    }
  }

  std::ofstream data_;
  std::ofstream index_;
  ZSTD_CCtx *context_ = nullptr;
  std::vector<char> output_buffer_;
  std::uint64_t compressed_offset_ = 0;
  std::uint64_t uncompressed_offset_ = 0;
  int iteration_ = -1;
  int batch_ = 0;
};

inline void ensureOutputDirectories(const BenchmarkConfig &config,
                                    const std::string &solver_key) {
  std::filesystem::create_directories(rootPath(config));
  std::filesystem::create_directories(rootPath(config) / "trajectories" /
                                      solver_key);
  std::filesystem::create_directories(rootPath(config) / "rollout_ee" /
                                      solver_key);
  std::filesystem::create_directories(rootPath(config) / "rollout_state" /
                                      solver_key);
}

inline void configureWorkspaceModel(ManipulatorDynamicsModel &model,
                                    const BenchmarkConfig &config) {
  (void)config;
  model.w_tau = 1.0e-3;
  model.w_qdot = 2.0e-2;
  model.w_joint_limit = 20.0;
  model.w_workspace_obs = 2.0e3;
  model.w_ground = 2.0e4;
  model.w_terminal_q = 1.5e3;
  model.w_terminal_ee = 5.0e2;
  model.w_terminal_qdot = 1.0e2;
  model.link_radius = 0.045;
  model.obs_safe_margin = 0.10;
  model.hard_collision_margin = 0.02;
  model.workspace_boxes.clear();
}

inline void applyScenarioObstacle(ManipulatorDynamicsModel &model,
                                  const ManipulatorBoxObstacle &obstacle) {
  model.workspace_boxes.clear();
  model.addWorkspaceBoxMinMax(obstacle.xmin, obstacle.xmax, obstacle.ymin,
                              obstacle.ymax, obstacle.zmin, obstacle.zmax);
}

inline CollisionChecker makeWorkspaceCollisionChecker(
    const ManipulatorDynamicsModel &model,
    const ManipulatorBoxObstacle &obstacle) {
  CollisionChecker cc;
  cc.clear();
  cc.resolution = 0.05;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
  cc.use_workspace_link_collision = true;
  cc.addWorkspaceBoxMinMax(obstacle.xmin, obstacle.xmax, obstacle.ymin,
                           obstacle.ymax, obstacle.zmin, obstacle.zmax);
  return cc;
}

inline ManipulatorBoxObstacle makeMidPathObstacle(
    const ManipulatorDynamicsModel &model, const Eigen::VectorXd &x_init,
    const Eigen::VectorXd &x_goal, const BenchmarkConfig &config) {
  const Eigen::Vector3d init_ee = model.endEffectorPosition(x_init);
  const Eigen::Vector3d goal_ee = model.endEffectorPosition(x_goal);
  const Eigen::Vector3d center = 0.5 * (init_ee + goal_ee);

  ManipulatorBoxObstacle obstacle;
  obstacle.xmin = center.x() - config.obstacle_half_x;
  obstacle.xmax = center.x() + config.obstacle_half_x;
  obstacle.ymin = center.y() - config.obstacle_half_y;
  obstacle.ymax = center.y() + config.obstacle_half_y;
  obstacle.zmin = std::max(config.obstacle_min_z,
                           center.z() - config.obstacle_half_z);
  obstacle.zmax = std::max(obstacle.zmin + 0.02,
                           center.z() + config.obstacle_half_z);
  return obstacle;
}

inline Eigen::MatrixXd makeSigmaMatrix(const JointNoiseSigma &sigma_diag) {
  Eigen::MatrixXd sigma =
      Eigen::MatrixXd::Zero(ManipulatorDynamicsModel::kDof,
                            ManipulatorDynamicsModel::kDof);
  for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
    sigma(i, i) = sigma_diag[static_cast<std::size_t>(i)];
  }
  return sigma;
}

inline bool directJointPathBlocked(const ManipulatorDynamicsModel &model,
                                   const Eigen::VectorXd &x_init,
                                   const Eigen::VectorXd &x_goal);

inline std::vector<BenchmarkScenario> makeBenchmarkScenarios(
    ManipulatorDynamicsModel &model, const BenchmarkConfig &config) {
  const auto poses = manipulator_pose_set::makeReferencePoseSet16();
  manipulator_pose_set::validatePoseSet(poses);

  std::vector<BenchmarkScenario> scenarios;
  scenarios.reserve(static_cast<std::size_t>(config.scenario_count));

  for (std::size_t init_idx = 0; init_idx < poses.size(); ++init_idx) {
    for (std::size_t goal_idx = 0; goal_idx < poses.size(); ++goal_idx) {
      if (init_idx == goal_idx) {
        continue;
      }

      const auto pair = manipulator_pose_set::makePosePairByIndex(
          poses, static_cast<int>(init_idx), static_cast<int>(goal_idx));

      const Eigen::VectorXd x_init =
          manipulator_pose_set::makeStateFromPose(model, pair.init_pose);
      const Eigen::VectorXd x_goal =
          manipulator_pose_set::makeStateFromPose(model, pair.goal_pose);
      const double pair_q_distance =
          (x_init.head(ManipulatorDynamicsModel::kDof) -
           x_goal.head(ManipulatorDynamicsModel::kDof))
              .norm();
      if (pair_q_distance > config.max_pair_q_distance) {
        continue;
      }

      const Eigen::Vector3d init_ee = model.endEffectorPosition(x_init);
      const Eigen::Vector3d goal_ee = model.endEffectorPosition(x_goal);
      const double pair_ee_distance = (init_ee - goal_ee).norm();
      if (pair_ee_distance < config.min_pair_ee_distance) {
        continue;
      }

      const ManipulatorBoxObstacle obstacle =
          makeMidPathObstacle(model, x_init, x_goal, config);
      applyScenarioObstacle(model, obstacle);

      if (model.inWorkspaceCollision(x_init) ||
          model.inWorkspaceCollision(x_goal)) {
        continue;
      }

      BenchmarkScenario scenario;
      scenario.index = static_cast<int>(scenarios.size());
      scenario.rollout_seed = config.rollout_seed_base + scenario.index;
      scenario.pose_pair = pair;
      scenario.obstacle = obstacle;
      scenario.direct_path_blocked =
          directJointPathBlocked(model, x_init, x_goal);
      scenarios.push_back(scenario);
    }
  }

  if (scenarios.empty()) {
    throw std::runtime_error("failed to generate any valid benchmark scenario");
  }
  return scenarios;
}

inline std::vector<BenchmarkScenario> selectBenchmarkScenarios(
    const std::vector<BenchmarkScenario> &scenarios,
    const BenchmarkConfig &config) {
  if (config.selected_scenario_ids.empty()) return scenarios;
  std::vector<BenchmarkScenario> selected;
  selected.reserve(config.selected_scenario_ids.size());
  for (int id : config.selected_scenario_ids) {
    const auto found = std::find_if(
        scenarios.begin(), scenarios.end(),
        [id](const BenchmarkScenario &scenario) { return scenario.index == id; });
    if (found == scenarios.end()) {
      throw std::runtime_error("selected benchmark scenario was not generated");
    }
    selected.push_back(*found);
  }
  return selected;
}

inline void writeScenarioCsv(const BenchmarkConfig &config,
                             const ManipulatorDynamicsModel &model,
                             const std::vector<BenchmarkScenario> &scenarios) {
  std::ofstream ofs(scenarioCsvPath(config));
  if (!ofs) {
    throw std::runtime_error("failed to open scenario CSV");
  }
  ofs << "scenario,rollout_seed,init_idx,goal_idx,init_name,goal_name,"
         "init_ee_x,init_ee_y,init_ee_z,goal_ee_x,goal_ee_y,goal_ee_z,"
         "obstacle_xmin,obstacle_xmax,obstacle_ymin,obstacle_ymax,"
         "obstacle_zmin,obstacle_zmax,direct_path_blocked";
  for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
    ofs << ",init_q" << (i + 1);
  }
  for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
    ofs << ",goal_q" << (i + 1);
  }
  ofs << "\n";

  ofs << std::setprecision(12);
  for (const auto &scenario : scenarios) {
    const auto &pair = scenario.pose_pair;
    const Eigen::VectorXd x_init =
        manipulator_pose_set::makeStateFromPose(model, pair.init_pose);
    const Eigen::VectorXd x_goal =
        manipulator_pose_set::makeStateFromPose(model, pair.goal_pose);
    const Eigen::Vector3d init_ee = model.endEffectorPosition(x_init);
    const Eigen::Vector3d goal_ee = model.endEffectorPosition(x_goal);

    ofs << scenario.index << "," << scenario.rollout_seed << ","
        << pair.init_idx << "," << pair.goal_idx << ","
        << pair.init_pose.name << "," << pair.goal_pose.name << ","
        << init_ee.x() << "," << init_ee.y() << "," << init_ee.z() << ","
        << goal_ee.x() << "," << goal_ee.y() << "," << goal_ee.z() << ","
        << scenario.obstacle.xmin << "," << scenario.obstacle.xmax << ","
        << scenario.obstacle.ymin << "," << scenario.obstacle.ymax << ","
        << scenario.obstacle.zmin << "," << scenario.obstacle.zmax << ","
        << static_cast<int>(scenario.direct_path_blocked);
    for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
      ofs << "," << pair.init_pose.q[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
      ofs << "," << pair.goal_pose.q[static_cast<std::size_t>(i)];
    }
    ofs << "\n";
  }
}

inline void writeStatsHeader(std::ofstream &ofs) {
  ofs << "solver,solver_label,scenario,rollout_seed,init_idx,goal_idx,"
         "init_name,goal_name,obstacle_xmin,obstacle_xmax,obstacle_ymin,"
         "obstacle_ymax,obstacle_zmin,obstacle_zmax,direct_path_blocked,"
         "init_collision,goal_collision,reached,success,"
         "solve_iterations,executed_steps,max_iter,total_solver_elapsed_s,"
         "q_tol,ee_tol,qdot_tol,max_pair_q_distance,mean_solver_elapsed_s,"
         "total_rollout_elapsed_s,mean_rollout_elapsed_s,"
         "total_clustering_elapsed_s,mean_clustering_elapsed_s,"
         "total_connection_elapsed_s,mean_connection_elapsed_s,"
         "total_guide_elapsed_s,mean_guide_elapsed_s,"
         "wall_elapsed_s,final_q_error,final_ee_error,final_qdot_norm,"
         "collision_count,connection_distance,horizon,samples_per_branch,"
         "samples_total_per_iter,trajectory_csv\n";
}

inline void writeStatsRow(std::ofstream &ofs, const BenchmarkResult &r) {
  ofs << std::setprecision(12) << r.solver_key << "," << r.solver_label << ","
      << r.scenario_index << "," << r.rollout_seed << "," << r.init_idx << ","
      << r.goal_idx << "," << r.init_name << "," << r.goal_name << ","
      << r.obstacle_xmin << "," << r.obstacle_xmax << ","
      << r.obstacle_ymin << "," << r.obstacle_ymax << ","
      << r.obstacle_zmin << "," << r.obstacle_zmax << ","
      << static_cast<int>(r.direct_path_blocked) << ","
      << static_cast<int>(r.init_collision) << ","
      << static_cast<int>(r.goal_collision) << ","
      << static_cast<int>(r.reached) << "," << static_cast<int>(r.success)
      << "," << r.solve_iterations << "," << r.executed_steps << ","
      << r.max_iter << "," << r.total_solver_elapsed << ","
      << r.q_tol << "," << r.ee_tol << "," << r.qdot_tol << ","
      << r.max_pair_q_distance << "," << r.mean_solver_elapsed << ","
      << r.total_rollout_elapsed << "," << r.mean_rollout_elapsed << ","
      << r.total_clustering_elapsed << "," << r.mean_clustering_elapsed << ","
      << r.total_connection_elapsed << "," << r.mean_connection_elapsed << ","
      << r.total_guide_elapsed << "," << r.mean_guide_elapsed << ","
      << r.wall_elapsed << "," << r.final_q_error << ","
      << r.final_ee_error << "," << r.final_qdot_norm << ","
      << r.collision_count << "," << r.connection_distance << ","
      << r.horizon << "," << r.samples_per_branch << ","
      << r.samples_total_per_iter << "," << r.trajectory_csv << "\n";
}

inline int countWorkspaceCollisions(const ManipulatorDynamicsModel &model,
                                    const Eigen::MatrixXd &executed_x) {
  int count = 0;
  for (int c = 0; c < executed_x.cols(); ++c) {
    if (model.inWorkspaceCollision(executed_x.col(c))) {
      ++count;
    }
  }
  return count;
}

inline bool directJointPathBlocked(const ManipulatorDynamicsModel &model,
                                   const Eigen::VectorXd &x_init,
                                   const Eigen::VectorXd &x_goal) {
  constexpr int kSegments = 24;
  constexpr int dof = ManipulatorDynamicsModel::kDof;
  Eigen::VectorXd x = x_init;
  x.tail(dof).setZero();

  for (int i = 0; i <= kSegments; ++i) {
    const double alpha = static_cast<double>(i) / kSegments;
    x.head(dof) =
        (1.0 - alpha) * x_init.head(dof) + alpha * x_goal.head(dof);
    if (model.inWorkspaceCollision(x)) {
      return true;
    }
  }
  return false;
}

inline bool isReached(const ManipulatorDynamicsModel &model,
                      const Eigen::VectorXd &x, const Eigen::VectorXd &x_goal,
                      const BenchmarkConfig &config) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;
  const double q_error = (x.head(dof) - x_goal.head(dof)).norm();
  const double ee_error =
      (model.endEffectorPosition(x) - model.endEffectorPosition(x_goal)).norm();
  const double qdot_norm = x.segment(dof, dof).norm();
  return q_error < config.q_tol && ee_error < config.ee_tol &&
         qdot_norm < config.qdot_tol;
}

inline Eigen::VectorXd makeScenarioWaypointState(
    const ManipulatorDynamicsModel &model,
    const BenchmarkScenario &scenario) {
  const auto poses = manipulator_pose_set::makeReferencePoseSet16();
  const std::array<int, 8> candidate_ids = {1, 5, 12, 13, 10, 8, 6, 3};

  double best_penalty = std::numeric_limits<double>::infinity();
  Eigen::VectorXd best_state;
  bool found = false;

  for (const int id : candidate_ids) {
    if (id == scenario.pose_pair.init_idx || id == scenario.pose_pair.goal_idx) {
      continue;
    }

    const Eigen::VectorXd x_waypoint =
        manipulator_pose_set::makeStateFromPose(model, poses[id]);
    if (model.inWorkspaceCollision(x_waypoint)) {
      continue;
    }

    const double penalty = model.workspaceObstaclePenalty(x_waypoint);
    if (!found || penalty < best_penalty) {
      best_penalty = penalty;
      best_state = x_waypoint;
      found = true;
    }
  }

  if (!found) {
    return manipulator_pose_set::makeStateFromPose(model, poses[1]);
  }
  return best_state;
}

template <typename Solver>
BenchmarkResult executeForwardScenario(
    Solver &solver, const ManipulatorDynamicsModel &model,
    const BenchmarkConfig &config, const BenchmarkScenario &scenario,
    const Eigen::VectorXd &x_goal, const Eigen::VectorXd &x_waypoint,
    const MPPIParam &param,
    const std::string &solver_key, const std::string &solver_label,
    const std::filesystem::path &trajectory_csv,
    RolloutEEWriter *rollout_ee_writer,
    RolloutStateWriter *rollout_state_writer) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;

  Eigen::MatrixXd executed_x =
      Eigen::MatrixXd::Zero(model.dim_x, config.max_iter + 1);
  Eigen::MatrixXd executed_u =
      Eigen::MatrixXd::Zero(model.dim_u, config.max_iter);
  executed_x.col(0) = solver.x_init;

  double total_solver_elapsed = 0.0;
  double total_rollout_elapsed = 0.0;
  double total_clustering_elapsed = 0.0;
  double total_connection_elapsed = 0.0;
  double total_guide_elapsed = 0.0;
  int executed_steps = 0;
  int solve_iterations = 0;
  bool reached = false;

  const auto wall_start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < config.max_iter; ++iter) {
    if (rollout_ee_writer) rollout_ee_writer->setIteration(iter);
    if (rollout_state_writer) rollout_state_writer->setIteration(iter);
    solver.solve();
    ++solve_iterations;
    total_solver_elapsed += solver.elapsed;
    total_rollout_elapsed += solver.elapsed_rollout;
    total_clustering_elapsed += solver.elapsed_clustering;
    total_connection_elapsed += solver.elapsed_connection;
    total_guide_elapsed += solver.elapsed_guide;

    const Eigen::VectorXd x_now = solver.x_init;
    reached = isReached(model, x_now, x_goal, config);
    if (reached) {
      executed_steps = iter;
      break;
    }

    executed_u.col(iter) = solver.u0;
    solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
    executed_x.col(iter + 1) = solver.x_init;
    executed_steps = iter + 1;

    if (solver.Uo.cols() >= param.T) {
      solver.U_0.leftCols(param.T - 1) = solver.Uo.middleCols(1, param.T - 1);
      solver.U_0.col(param.T - 1).setZero();
    } else {
      solver.U_0 = makeWaypointPdTorqueWarmStart(
          model, solver.x_init, x_waypoint, x_goal, param.T, param.dt,
          config.warm_start_kp, config.warm_start_kd);
    }
  }
  const auto wall_finish = std::chrono::steady_clock::now();

  const Eigen::MatrixXd executed_x_trimmed =
      executed_x.leftCols(executed_steps + 1);
  writeMatrixCsv(trajectory_csv.string(), executed_x_trimmed, "x");

  const Eigen::VectorXd x_final = solver.x_init;
  const double final_q_error = (x_final.head(dof) - x_goal.head(dof)).norm();
  const double final_ee_error =
      (model.endEffectorPosition(x_final) - model.endEffectorPosition(x_goal))
          .norm();
  const double final_qdot_norm = x_final.segment(dof, dof).norm();
  const int collision_count =
      countWorkspaceCollisions(model, executed_x_trimmed);

  BenchmarkResult result;
  result.solver_key = solver_key;
  result.solver_label = solver_label;
  result.scenario_index = scenario.index;
  result.rollout_seed = scenario.rollout_seed;
  result.init_idx = scenario.pose_pair.init_idx;
  result.goal_idx = scenario.pose_pair.goal_idx;
  result.init_name = scenario.pose_pair.init_pose.name;
  result.goal_name = scenario.pose_pair.goal_pose.name;
  result.obstacle_xmin = scenario.obstacle.xmin;
  result.obstacle_xmax = scenario.obstacle.xmax;
  result.obstacle_ymin = scenario.obstacle.ymin;
  result.obstacle_ymax = scenario.obstacle.ymax;
  result.obstacle_zmin = scenario.obstacle.zmin;
  result.obstacle_zmax = scenario.obstacle.zmax;
  result.direct_path_blocked = scenario.direct_path_blocked;
  result.init_collision = model.inWorkspaceCollision(executed_x_trimmed.col(0));
  result.goal_collision = model.inWorkspaceCollision(x_goal);
  result.reached = reached;
  result.success = reached && collision_count == 0;
  result.solve_iterations = solve_iterations;
  result.executed_steps = executed_steps;
  result.max_iter = config.max_iter;
  result.q_tol = config.q_tol;
  result.ee_tol = config.ee_tol;
  result.qdot_tol = config.qdot_tol;
  result.max_pair_q_distance = config.max_pair_q_distance;
  result.total_solver_elapsed = total_solver_elapsed;
  result.mean_solver_elapsed =
      solve_iterations > 0 ? total_solver_elapsed / solve_iterations : 0.0;
  result.total_rollout_elapsed = total_rollout_elapsed;
  result.mean_rollout_elapsed =
      solve_iterations > 0 ? total_rollout_elapsed / solve_iterations : 0.0;
  result.total_clustering_elapsed = total_clustering_elapsed;
  result.mean_clustering_elapsed =
      solve_iterations > 0 ? total_clustering_elapsed / solve_iterations : 0.0;
  result.total_connection_elapsed = total_connection_elapsed;
  result.mean_connection_elapsed =
      solve_iterations > 0 ? total_connection_elapsed / solve_iterations : 0.0;
  result.total_guide_elapsed = total_guide_elapsed;
  result.mean_guide_elapsed =
      solve_iterations > 0 ? total_guide_elapsed / solve_iterations : 0.0;
  result.wall_elapsed =
      std::chrono::duration<double>(wall_finish - wall_start).count();
  result.final_q_error = final_q_error;
  result.final_ee_error = final_ee_error;
  result.final_qdot_norm = final_qdot_norm;
  result.collision_count = collision_count;
  result.connection_distance = solver.connectionDistance();
  result.horizon = param.T;
  result.samples_per_branch = param.N;
  result.samples_total_per_iter = param.N;
  result.trajectory_csv = trajectory_csv.string();
  return result;
}

template <typename Solver, typename ConfigureSolver>
int runForwardBenchmark(const std::string &solver_key,
                        const std::string &solver_label,
                        const SolverParams &solver_params,
                        ConfigureSolver configure_solver) {
  BenchmarkConfig config = benchmarkConfigFromEnvironment();
  ManipulatorDynamicsModel model;
  configureWorkspaceModel(model, config);

  const auto scenarios = selectBenchmarkScenarios(
      makeBenchmarkScenarios(model, config), config);
  ensureOutputDirectories(config, solver_key);
  writeScenarioCsv(config, model, scenarios);

  std::ofstream stats(solverStatsPath(config, solver_key));
  if (!stats) {
    throw std::runtime_error("failed to open solver stats CSV");
  }
  writeStatsHeader(stats);

  for (const auto &scenario : scenarios) {
    applyScenarioObstacle(model, scenario.obstacle);

    const Eigen::VectorXd x_init = manipulator_pose_set::makeStateFromPose(
        model, scenario.pose_pair.init_pose);
    const Eigen::VectorXd x_goal = manipulator_pose_set::makeStateFromPose(
        model, scenario.pose_pair.goal_pose);
    const Eigen::VectorXd x_waypoint =
        makeScenarioWaypointState(model, scenario);

    MPPIParam param;
    param.dt = solver_params.dt;
    param.T = solver_params.forward_horizon;
    param.N = solver_params.forward_samples;
    param.gamma_u = solver_params.gamma_u;
    param.x_init = x_init;
    param.x_target = x_goal;
    param.sigma_u = makeSigmaMatrix(solver_params.sigma);
    param.clustering_method = solver_params.clustering_method;
    param.kmeans_clusters = solver_params.kmeans_clusters;
    param.kmeans_max_iterations = solver_params.kmeans_max_iterations;
    param.kmeans_threshold = solver_params.kmeans_threshold;

    CollisionChecker cc = makeWorkspaceCollisionChecker(model, scenario.obstacle);
    Solver solver(model);
    configure_solver(solver, solver_params);
    solver.U_0 = makeWaypointPdTorqueWarmStart(
        model, x_init, x_waypoint, x_goal, param.T, param.dt,
        config.warm_start_kp, config.warm_start_kd);
    solver.init(param);
    solver.setCollisionChecker(&cc);
    solver.setSeed(scenario.rollout_seed);

    const std::filesystem::path traj_path =
        trajectoryPath(config, solver_key, scenario.index);
    std::unique_ptr<RolloutEEWriter> rollout_ee_writer;
    std::unique_ptr<RolloutStateWriter> rollout_state_writer;
    if (config.save_rollout_ee) {
      rollout_ee_writer = std::make_unique<RolloutEEWriter>(
          rolloutEEPath(config, solver_key, scenario.index),
          rolloutEEIndexPath(config, solver_key, scenario.index));
      solver.setRolloutEECallback(
          [&rollout_ee_writer](const std::string &branch,
                              const std::vector<std::uint8_t> &positions,
                              int rollout_count, int point_count) {
            rollout_ee_writer->append(branch, positions, rollout_count,
                                      point_count);
          });
    }
    if (config.save_rollout_state) {
      rollout_state_writer = std::make_unique<RolloutStateWriter>(
          rolloutStatePath(config, solver_key, scenario.index),
          rolloutStateIndexPath(config, solver_key, scenario.index));
      solver.setRolloutStateCallback(
          [&rollout_state_writer](const std::string &branch,
                                 const std::vector<std::uint16_t> &states,
                                 int rollout_count, int point_count,
                                 int state_dim) {
            rollout_state_writer->append(branch, states, rollout_count,
                                         point_count, state_dim);
          });
    }
    const BenchmarkResult result = executeForwardScenario(
        solver, model, config, scenario, x_goal, x_waypoint, param, solver_key,
        solver_label, traj_path, rollout_ee_writer.get(),
        rollout_state_writer.get());
    writeStatsRow(stats, result);

    std::cout << solver_label << " " << scenarioTag(scenario.index)
              << " init=" << scenario.pose_pair.init_pose.name
              << " goal=" << scenario.pose_pair.goal_pose.name
              << " success=" << static_cast<int>(result.success)
              << " iter=" << result.solve_iterations
              << " solver_time=" << result.total_solver_elapsed << " s"
              << " collisions=" << result.collision_count << "\n";
  }

  std::cout << solver_label << " stats: " << solverStatsPath(config, solver_key)
            << "\n";
  return 0;
}

template <typename Solver>
BenchmarkResult executeBidirectionalScenario(
    Solver &solver, const ManipulatorDynamicsModel &model,
    const BenchmarkConfig &config, const BenchmarkScenario &scenario,
    const Eigen::VectorXd &x_goal, const Eigen::VectorXd &x_waypoint,
    const BiMPPIParam &param,
    const std::string &solver_key, const std::string &solver_label,
    const std::filesystem::path &trajectory_csv,
    RolloutEEWriter *rollout_ee_writer,
    RolloutStateWriter *rollout_state_writer) {
  constexpr int dof = ManipulatorDynamicsModel::kDof;

  Eigen::MatrixXd executed_x =
      Eigen::MatrixXd::Zero(model.dim_x, config.max_iter + 1);
  Eigen::MatrixXd executed_u =
      Eigen::MatrixXd::Zero(model.dim_u, config.max_iter);
  executed_x.col(0) = solver.x_init;

  double total_solver_elapsed = 0.0;
  double total_rollout_elapsed = 0.0;
  double total_clustering_elapsed = 0.0;
  double total_connection_elapsed = 0.0;
  double total_guide_elapsed = 0.0;
  int executed_steps = 0;
  int solve_iterations = 0;
  bool reached = false;

  const auto wall_start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < config.max_iter; ++iter) {
    if (rollout_ee_writer) rollout_ee_writer->setIteration(iter);
    if (rollout_state_writer) rollout_state_writer->setIteration(iter);
    solver.solve();
    ++solve_iterations;
    total_solver_elapsed += solver.elapsed;
    total_rollout_elapsed += solver.elapsed_rollout;
    total_clustering_elapsed += solver.elapsed_clustering;
    total_connection_elapsed += solver.elapsed_connection;
    total_guide_elapsed += solver.elapsed_guide;

    const Eigen::VectorXd x_now = solver.x_init;
    reached = isReached(model, x_now, x_goal, config);
    if (reached) {
      executed_steps = iter;
      break;
    }

    executed_u.col(iter) = solver.u0;
    solver.x_init = model.rk4Step(solver.x_init, solver.u0, param.dt);
    executed_x.col(iter + 1) = solver.x_init;
    executed_steps = iter + 1;

    if (solver.Uo.cols() >= param.Tf) {
      solver.U_f0.leftCols(param.Tf - 1) =
          solver.Uo.middleCols(1, param.Tf - 1);
      solver.U_f0.col(param.Tf - 1).setZero();
    } else {
      solver.U_f0 = makeWaypointPdTorqueWarmStart(
          model, solver.x_init, x_waypoint, x_goal, param.Tf, param.dt,
          config.warm_start_kp, config.warm_start_kd);
    }
    solver.U_b0 = makePdTorqueWarmStart(
        model, solver.x_init, x_goal, param.Tb, param.dt,
        config.warm_start_kp, config.warm_start_kd);
  }
  const auto wall_finish = std::chrono::steady_clock::now();

  const Eigen::MatrixXd executed_x_trimmed =
      executed_x.leftCols(executed_steps + 1);
  writeMatrixCsv(trajectory_csv.string(), executed_x_trimmed, "x");

  const Eigen::VectorXd x_final = solver.x_init;
  const double final_q_error = (x_final.head(dof) - x_goal.head(dof)).norm();
  const double final_ee_error =
      (model.endEffectorPosition(x_final) - model.endEffectorPosition(x_goal))
          .norm();
  const double final_qdot_norm = x_final.segment(dof, dof).norm();
  const int collision_count =
      countWorkspaceCollisions(model, executed_x_trimmed);

  BenchmarkResult result;
  result.solver_key = solver_key;
  result.solver_label = solver_label;
  result.scenario_index = scenario.index;
  result.rollout_seed = scenario.rollout_seed;
  result.init_idx = scenario.pose_pair.init_idx;
  result.goal_idx = scenario.pose_pair.goal_idx;
  result.init_name = scenario.pose_pair.init_pose.name;
  result.goal_name = scenario.pose_pair.goal_pose.name;
  result.obstacle_xmin = scenario.obstacle.xmin;
  result.obstacle_xmax = scenario.obstacle.xmax;
  result.obstacle_ymin = scenario.obstacle.ymin;
  result.obstacle_ymax = scenario.obstacle.ymax;
  result.obstacle_zmin = scenario.obstacle.zmin;
  result.obstacle_zmax = scenario.obstacle.zmax;
  result.direct_path_blocked = scenario.direct_path_blocked;
  result.init_collision = model.inWorkspaceCollision(executed_x_trimmed.col(0));
  result.goal_collision = model.inWorkspaceCollision(x_goal);
  result.reached = reached;
  result.success = reached && collision_count == 0;
  result.solve_iterations = solve_iterations;
  result.executed_steps = executed_steps;
  result.max_iter = config.max_iter;
  result.q_tol = config.q_tol;
  result.ee_tol = config.ee_tol;
  result.qdot_tol = config.qdot_tol;
  result.max_pair_q_distance = config.max_pair_q_distance;
  result.total_solver_elapsed = total_solver_elapsed;
  result.mean_solver_elapsed =
      solve_iterations > 0 ? total_solver_elapsed / solve_iterations : 0.0;
  result.total_rollout_elapsed = total_rollout_elapsed;
  result.mean_rollout_elapsed =
      solve_iterations > 0 ? total_rollout_elapsed / solve_iterations : 0.0;
  result.total_clustering_elapsed = total_clustering_elapsed;
  result.mean_clustering_elapsed =
      solve_iterations > 0 ? total_clustering_elapsed / solve_iterations : 0.0;
  result.total_connection_elapsed = total_connection_elapsed;
  result.mean_connection_elapsed =
      solve_iterations > 0 ? total_connection_elapsed / solve_iterations : 0.0;
  result.total_guide_elapsed = total_guide_elapsed;
  result.mean_guide_elapsed =
      solve_iterations > 0 ? total_guide_elapsed / solve_iterations : 0.0;
  result.wall_elapsed =
      std::chrono::duration<double>(wall_finish - wall_start).count();
  result.final_q_error = final_q_error;
  result.final_ee_error = final_ee_error;
  result.final_qdot_norm = final_qdot_norm;
  result.collision_count = collision_count;
  result.connection_distance = solver.connectionDistance();
  result.horizon = param.Tf;
  result.samples_per_branch = param.Nf;
  result.samples_total_per_iter = param.Nf + param.Nb + param.Nr;
  result.trajectory_csv = trajectory_csv.string();
  return result;
}

template <typename Solver>
int runBidirectionalBenchmark(const std::string &solver_key,
                              const std::string &solver_label,
                              const SolverParams &solver_params) {
  BenchmarkConfig config = benchmarkConfigFromEnvironment();
  ManipulatorDynamicsModel model;
  configureWorkspaceModel(model, config);

  const auto scenarios = selectBenchmarkScenarios(
      makeBenchmarkScenarios(model, config), config);
  ensureOutputDirectories(config, solver_key);
  writeScenarioCsv(config, model, scenarios);

  std::ofstream stats(solverStatsPath(config, solver_key));
  if (!stats) {
    throw std::runtime_error("failed to open solver stats CSV");
  }
  writeStatsHeader(stats);

  for (const auto &scenario : scenarios) {
    applyScenarioObstacle(model, scenario.obstacle);

    const Eigen::VectorXd x_init = manipulator_pose_set::makeStateFromPose(
        model, scenario.pose_pair.init_pose);
    const Eigen::VectorXd x_goal = manipulator_pose_set::makeStateFromPose(
        model, scenario.pose_pair.goal_pose);
    const Eigen::VectorXd x_waypoint =
        makeScenarioWaypointState(model, scenario);

    BiMPPIParam param;
    param.dt = solver_params.dt;
    param.Tf = solver_params.forward_horizon;
    param.Tb = solver_params.backward_horizon;
    param.Nf = solver_params.forward_samples;
    param.Nb = solver_params.backward_samples;
    param.Nr = solver_params.reverse_samples;
    param.gamma_u = solver_params.gamma_u;
    param.x_init = x_init;
    param.x_target = x_goal;
    param.sigma_u = makeSigmaMatrix(solver_params.sigma);
    param.deviation_mu = solver_params.bic_deviation_mu;
    param.cost_mu = solver_params.bic_cost_mu;
    param.epsilon = solver_params.bic_epsilon;
    param.minpts = solver_params.bic_minpts;
    param.psi = solver_params.bic_psi;
    param.clustering_method = solver_params.clustering_method;
    param.kmeans_clusters = solver_params.kmeans_clusters;
    param.kmeans_max_iterations = solver_params.kmeans_max_iterations;
    param.kmeans_threshold = solver_params.kmeans_threshold;

    CollisionChecker cc = makeWorkspaceCollisionChecker(model, scenario.obstacle);
    Solver solver(model);
    solver.init(param);
    solver.setCollisionChecker(&cc);
    solver.setSeed(scenario.rollout_seed);
    solver.U_f0 = makeWaypointPdTorqueWarmStart(
        model, x_init, x_waypoint, x_goal, param.Tf, param.dt,
        config.warm_start_kp, config.warm_start_kd);
    solver.U_b0 = makePdTorqueWarmStart(model, x_init, x_goal, param.Tb,
                                        param.dt, config.warm_start_kp,
                                        config.warm_start_kd);

    const std::filesystem::path traj_path =
        trajectoryPath(config, solver_key, scenario.index);
    std::unique_ptr<RolloutEEWriter> rollout_ee_writer;
    std::unique_ptr<RolloutStateWriter> rollout_state_writer;
    if (config.save_rollout_ee) {
      rollout_ee_writer = std::make_unique<RolloutEEWriter>(
          rolloutEEPath(config, solver_key, scenario.index),
          rolloutEEIndexPath(config, solver_key, scenario.index));
      solver.setRolloutEECallback(
          [&rollout_ee_writer](const std::string &branch,
                              const std::vector<std::uint8_t> &positions,
                              int rollout_count, int point_count) {
            rollout_ee_writer->append(branch, positions, rollout_count,
                                      point_count);
          });
    }
    if (config.save_rollout_state) {
      rollout_state_writer = std::make_unique<RolloutStateWriter>(
          rolloutStatePath(config, solver_key, scenario.index),
          rolloutStateIndexPath(config, solver_key, scenario.index));
      solver.setRolloutStateCallback(
          [&rollout_state_writer](const std::string &branch,
                                 const std::vector<std::uint16_t> &states,
                                 int rollout_count, int point_count,
                                 int state_dim) {
            rollout_state_writer->append(branch, states, rollout_count,
                                         point_count, state_dim);
          });
    }
    const BenchmarkResult result = executeBidirectionalScenario(
        solver, model, config, scenario, x_goal, x_waypoint, param, solver_key,
        solver_label, traj_path, rollout_ee_writer.get(),
        rollout_state_writer.get());
    writeStatsRow(stats, result);

    std::cout << solver_label << " " << scenarioTag(scenario.index)
              << " init=" << scenario.pose_pair.init_pose.name
              << " goal=" << scenario.pose_pair.goal_pose.name
              << " success=" << static_cast<int>(result.success)
              << " iter=" << result.solve_iterations
              << " solver_time=" << result.total_solver_elapsed << " s"
              << " collisions=" << result.collision_count << "\n";
  }

  std::cout << solver_label << " stats: " << solverStatsPath(config, solver_key)
            << "\n";
  return 0;
}

}  // namespace manipulator_random_pose_benchmark
