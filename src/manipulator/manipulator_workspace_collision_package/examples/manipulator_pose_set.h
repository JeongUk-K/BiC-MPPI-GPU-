#pragma once

#include "manipulator_dynamics_model.h"

#include <Eigen/Dense>

#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// Manipulator reference pose set for BiC-MPPI random start-goal benchmarks.
// -----------------------------------------------------------------------------
// Purpose:
//   - Keep benchmark postures outside the main example cpp.
//   - Randomly sample two different postures from a fixed 16-pose set.
//   - Convert q posture to x=[q;qdot] target state with qdot=0.
//
// Convention:
//   - q is a 6-DOF joint posture [rad].
//   - The end-effector pose is induced by the manipulator FK.
//   - These are benchmark postures evaluated with the UR5e DH model.
// =============================================================================

namespace manipulator_pose_set {

struct ReferencePose {
  int id = -1;
  std::string name;
  std::array<double, ManipulatorDynamicsModel::kDof> q;
};

struct PosePair {
  int init_idx = -1;
  int goal_idx = -1;
  ReferencePose init_pose;
  ReferencePose goal_pose;
};

// Collision-free ordered pairs for the benchmark's midpoint obstacle setup.
// This is the 162-pair feasible set used by src/manipulator.
inline constexpr int kFeasiblePosePairCount = 162;

inline const std::array<std::pair<int, int>, kFeasiblePosePairCount> &
feasiblePosePairIndices162() {
  static const std::array<std::pair<int, int>, kFeasiblePosePairCount> pairs = {{
      {0, 1}, {0, 2}, {0, 4}, {0, 6}, {0, 9}, {0, 10}, {0, 11}, {0, 13},
      {0, 14}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7},
      {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15},
      {2, 0}, {2, 1}, {2, 3}, {2, 4}, {2, 5}, {2, 8}, {2, 9}, {2, 12},
      {2, 14}, {2, 15}, {3, 1}, {3, 2}, {3, 5}, {3, 6}, {3, 7}, {3, 9},
      {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 15}, {4, 0}, {4, 1}, {4, 2},
      {4, 5}, {4, 6}, {4, 7}, {4, 10}, {4, 11}, {4, 12}, {4, 13}, {4, 15},
      {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 8}, {5, 9}, {5, 11}, {5, 14},
      {6, 0}, {6, 1}, {6, 3}, {6, 4}, {6, 8}, {6, 9}, {6, 12}, {6, 14},
      {6, 15}, {7, 1}, {7, 3}, {7, 4}, {7, 8}, {7, 9}, {7, 10}, {7, 13},
      {7, 14}, {8, 1}, {8, 2}, {8, 5}, {8, 6}, {8, 7}, {8, 10}, {8, 11},
      {8, 12}, {8, 13}, {8, 15}, {9, 0}, {9, 1}, {9, 2}, {9, 3}, {9, 5},
      {9, 6}, {9, 7}, {9, 10}, {9, 11}, {9, 12}, {9, 13}, {9, 15}, {10, 0},
      {10, 1}, {10, 3}, {10, 4}, {10, 7}, {10, 8}, {10, 9}, {10, 12},
      {10, 14}, {10, 15}, {11, 0}, {11, 1}, {11, 3}, {11, 4}, {11, 5},
      {11, 8}, {11, 9}, {11, 12}, {11, 14}, {12, 1}, {12, 2}, {12, 3},
      {12, 4}, {12, 6}, {12, 8}, {12, 9}, {12, 10}, {12, 11}, {12, 13},
      {12, 14}, {13, 0}, {13, 1}, {13, 3}, {13, 4}, {13, 7}, {13, 8},
      {13, 9}, {13, 12}, {13, 14}, {14, 0}, {14, 1}, {14, 2}, {14, 5},
      {14, 6}, {14, 7}, {14, 10}, {14, 11}, {14, 12}, {14, 13}, {14, 15},
      {15, 1}, {15, 2}, {15, 3}, {15, 4}, {15, 6}, {15, 8}, {15, 9},
      {15, 10}, {15, 14},
  }};
  return pairs;
}

inline Eigen::VectorXd toEigenQ(const ReferencePose &pose) {
  Eigen::VectorXd q(ManipulatorDynamicsModel::kDof);
  for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
    q(i) = pose.q[static_cast<std::size_t>(i)];
  }
  return q;
}

inline Eigen::VectorXd makeStateFromQ(const ManipulatorDynamicsModel &model,
                                      const Eigen::VectorXd &q) {
  return model.makeState(q, Eigen::VectorXd::Zero(ManipulatorDynamicsModel::kDof));
}

inline Eigen::VectorXd makeStateFromPose(const ManipulatorDynamicsModel &model,
                                         const ReferencePose &pose) {
  return makeStateFromQ(model, toEigenQ(pose));
}

inline std::vector<ReferencePose> makeReferencePoseSet16() {
  std::vector<ReferencePose> poses;
  poses.reserve(16);

  poses.push_back({0,  "home_mid",        { 0.05, -1.15, 1.25,  0.00, 0.85,  0.00}});
  poses.push_back({1,  "detour_high",     { 1.20, -1.40, 0.50, -0.20, 0.70,  0.10}});
  poses.push_back({2,  "goal_right",      { 1.45, -0.80, 1.05, -0.20, 0.65,  0.25}});
  poses.push_back({3,  "pre_pick",        {-0.65, -1.18, 1.22,  0.00, 0.78,  0.00}});
  poses.push_back({4,  "pick_low",        {-0.65, -0.82, 1.52,  0.00, 0.38,  0.00}});
  poses.push_back({5,  "transfer_high",   { 0.10, -1.45, 1.25,  0.00, 1.05,  0.00}});
  poses.push_back({6,  "pre_place",       { 0.90, -1.18, 1.18,  0.00, 0.80,  0.00}});
  poses.push_back({7,  "place_low",       { 0.90, -0.78, 1.52,  0.00, 0.36,  0.00}});
  poses.push_back({8,  "left_high",       {-0.90, -1.25, 1.10,  0.15, 0.95,  0.05}});
  poses.push_back({9,  "left_low",        {-0.90, -0.90, 1.45,  0.10, 0.45,  0.05}});
  poses.push_back({10, "right_high",      { 1.10, -1.25, 1.10, -0.15, 0.95, -0.05}});
  poses.push_back({11, "right_low",       { 1.10, -0.90, 1.45, -0.10, 0.45, -0.05}});
  poses.push_back({12, "over_left",       {-0.35, -1.55, 1.35,  0.10, 1.10,  0.00}});
  poses.push_back({13, "over_right",      { 0.35, -1.55, 1.35, -0.10, 1.10,  0.00}});
  poses.push_back({14, "front_left_mid",  {-0.45, -0.95, 1.30,  0.10, 0.65,  0.10}});
  poses.push_back({15, "front_right_mid", { 0.45, -0.95, 1.30, -0.10, 0.65, -0.10}});

  return poses;
}

inline void validatePoseSet(const std::vector<ReferencePose> &poses) {
  if (poses.size() < 2) {
    throw std::runtime_error("Reference pose set must contain at least two poses.");
  }

  for (std::size_t i = 0; i < poses.size(); ++i) {
    if (poses[i].id != static_cast<int>(i)) {
      throw std::runtime_error("Reference pose id must match its vector index.");
    }
  }
}

inline PosePair sampleDifferentPosePair(const std::vector<ReferencePose> &poses,
                                        std::mt19937 &rng) {
  validatePoseSet(poses);

  std::uniform_int_distribution<int> dist(0, static_cast<int>(poses.size()) - 1);

  const int init_idx = dist(rng);
  int goal_idx = dist(rng);
  while (goal_idx == init_idx) {
    goal_idx = dist(rng);
  }

  return {init_idx, goal_idx, poses[static_cast<std::size_t>(init_idx)],
          poses[static_cast<std::size_t>(goal_idx)]};
}

inline PosePair makePosePairByIndex(const std::vector<ReferencePose> &poses,
                                    int init_idx, int goal_idx) {
  validatePoseSet(poses);

  if (init_idx < 0 || init_idx >= static_cast<int>(poses.size()) ||
      goal_idx < 0 || goal_idx >= static_cast<int>(poses.size())) {
    throw std::out_of_range("Pose index out of range.");
  }
  if (init_idx == goal_idx) {
    throw std::invalid_argument("Initial and goal pose indices must be different.");
  }

  return {init_idx, goal_idx, poses[static_cast<std::size_t>(init_idx)],
          poses[static_cast<std::size_t>(goal_idx)]};
}

inline void printPoseSet(const ManipulatorDynamicsModel &model,
                         const std::vector<ReferencePose> &poses) {
  std::cout << "\n=== Manipulator reference pose set ===\n";
  for (const auto &pose : poses) {
    const Eigen::VectorXd x = makeStateFromPose(model, pose);
    const Eigen::Vector3d ee = model.endEffectorPosition(x);

    std::cout << std::setw(2) << pose.id << "  " << std::setw(18) << pose.name
              << "  q=[";
    for (int j = 0; j < ManipulatorDynamicsModel::kDof; ++j) {
      std::cout << pose.q[static_cast<std::size_t>(j)]
                << (j + 1 == ManipulatorDynamicsModel::kDof ? "" : ", ");
    }
    std::cout << "]  ee=[" << ee.x() << ", " << ee.y() << ", " << ee.z()
              << "]\n";
  }
}

inline bool poseIsInWorkspaceCollision(const ManipulatorDynamicsModel &model,
                                       const ReferencePose &pose) {
  return model.inWorkspaceCollision(makeStateFromPose(model, pose));
}

inline void printPoseCollisionCheck(const ManipulatorDynamicsModel &model,
                                    const std::vector<ReferencePose> &poses) {
  std::cout << "\n=== Reference pose workspace-collision check ===\n";
  for (const auto &pose : poses) {
    std::cout << std::setw(2) << pose.id << "  " << std::setw(18) << pose.name
              << "  collision="
              << static_cast<int>(poseIsInWorkspaceCollision(model, pose))
              << "\n";
  }
}

}  // namespace manipulator_pose_set
