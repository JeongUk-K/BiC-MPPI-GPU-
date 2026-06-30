#pragma once

#include "collision_checker.h"
#include "manipulator_dynamics_model.h"

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

struct PickPlaceWorkspaceBox {
  std::string name;
  double xmin, xmax;
  double ymin, ymax;
  double zmin, zmax;
};

struct PickPlacePhase {
  std::string name;
  std::string gripper_event_before;
  std::string gripper_event_after;
  Eigen::VectorXd x_goal;
  int max_iter = 80;
  double q_tol = 0.075;
  double ee_tol = 0.035;
  double qdot_tol = 0.25;
};

inline Eigen::VectorXd makeManipulatorStateFromQ(
    const ManipulatorDynamicsModel &model,
    const std::array<double, ManipulatorDynamicsModel::kDof> &q_values) {
  Eigen::VectorXd q(ManipulatorDynamicsModel::kDof);
  for (int i = 0; i < ManipulatorDynamicsModel::kDof; ++i) {
    q(i) = q_values[i];
  }
  return model.makeState(q, Eigen::VectorXd::Zero(ManipulatorDynamicsModel::kDof));
}

inline std::vector<PickPlaceWorkspaceBox> makePickPlaceWorkspaceBoxes() {
  return {
      {"central_baffle", -0.48, -0.30, -0.22, 0.06, 0.05, 0.38},
      {"pick_bin_rear_wall", -0.72, -0.44, 0.30, 0.36, 0.02, 0.15},
      {"pick_bin_left_wall", -0.72, -0.66, 0.04, 0.36, 0.02, 0.15},
      {"place_bin_front_wall", -0.34, -0.04, -0.72, -0.66, 0.02, 0.16},
      {"place_bin_right_wall", -0.04, 0.02, -0.72, -0.42, 0.02, 0.16},
  };
}

inline void registerPickPlaceWorkspace(ManipulatorDynamicsModel &model,
                                       CollisionChecker &cc) {
  model.workspace_boxes.clear();
  cc.clear();
  cc.resolution = 0.05;
  cc.link_radius = model.link_radius;
  cc.workspace_safe_margin = model.obs_safe_margin;
  cc.workspace_hard_margin = model.hard_collision_margin;
  cc.use_workspace_link_collision = true;

  for (const auto &b : makePickPlaceWorkspaceBoxes()) {
    model.addWorkspaceBoxMinMax(b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax);
    cc.addWorkspaceBoxMinMax(b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax);
  }
}

inline Eigen::VectorXd makePickPlaceHomeState(
    const ManipulatorDynamicsModel &model) {
  return makeManipulatorStateFromQ(model,
                                   {0.00, -1.00, 1.20, 0.00, 0.80, 0.00});
}

inline std::vector<PickPlacePhase> makePickPlacePhases(
    const ManipulatorDynamicsModel &model) {
  const auto pre_pick = makeManipulatorStateFromQ(
      model, {-0.65, -1.18, 1.22, 0.00, 0.78, 0.00});
  const auto pick = makeManipulatorStateFromQ(
      model, {-0.65, -0.82, 1.52, 0.00, 0.38, 0.00});
  const auto lift = makeManipulatorStateFromQ(
      model, {-0.65, -1.18, 1.22, 0.00, 0.78, 0.00});
  const auto transfer_high = makeManipulatorStateFromQ(
      model, {0.10, -1.45, 1.25, 0.00, 1.05, 0.00});
  const auto pre_place = makeManipulatorStateFromQ(
      model, {0.90, -1.18, 1.18, 0.00, 0.80, 0.00});
  const auto place = makeManipulatorStateFromQ(
      model, {0.90, -0.78, 1.52, 0.00, 0.36, 0.00});
  const auto retreat = makeManipulatorStateFromQ(
      model, {0.90, -1.18, 1.18, 0.00, 0.80, 0.00});

  std::vector<PickPlacePhase> phases;
  phases.push_back({"home_to_pre_pick", "open", "", pre_pick, 85, 0.075, 0.035, 0.25});
  phases.push_back({"pre_pick_to_pick", "", "close", pick, 65, 0.060, 0.025, 0.25});
  phases.push_back({"pick_to_lift", "", "", lift, 65, 0.070, 0.035, 0.25});
  phases.push_back({"lift_to_transfer_high", "", "", transfer_high, 90, 0.090, 0.045, 0.30});
  phases.push_back({"transfer_to_pre_place", "", "", pre_place, 90, 0.080, 0.040, 0.25});
  phases.push_back({"pre_place_to_place", "", "release", place, 70, 0.060, 0.025, 0.25});
  phases.push_back({"place_to_retreat", "", "open", retreat, 70, 0.080, 0.040, 0.25});
  return phases;
}
