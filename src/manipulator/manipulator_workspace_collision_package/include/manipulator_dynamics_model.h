#pragma once

#include "model_base.h"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// =============================================================================
// ManipulatorDynamicsModel
// -----------------------------------------------------------------------------
// Paper/example convention:
//   x = [q; qdot] in R^12, u = tau in R^6
//   qdot     = v
//   vdot_i   = (tau_i - d_i v_i - g_i sin(q_i)) / I_i
//
// This is a decoupled joint-space second-order manipulator model. It is not a
// full rigid-body inverse-dynamics model M(q)qdd + C(q,qdot)qdot + G(q)=tau.
// It is deliberately chosen as a stable torque-level benchmark compatible with
// the uploaded BiMPPI_GPU legacy CUDA model without increasing kernel state size.
// =============================================================================

struct ManipulatorBoxObstacle {
  // Axis-aligned box bounds in workspace [m].
  double xmin = 0.0, xmax = 0.0;
  double ymin = 0.0, ymax = 0.0;
  double zmin = 0.0, zmax = 0.0;
};

class ManipulatorDynamicsModel : public ModelBase {
public:
  static constexpr int kDof = 6;

  ManipulatorDynamicsModel() { initialize(); }
  ~ManipulatorDynamicsModel() override = default;

  // Joint limits [rad], velocity limits [rad/s], torque limits [N m] or command units.
  Eigen::Matrix<double, kDof, 1> q_min;
  Eigen::Matrix<double, kDof, 1> q_max;
  Eigen::Matrix<double, kDof, 1> qdot_max;
  Eigen::Matrix<double, kDof, 1> tau_max;

  // Decoupled dynamics parameters.
  Eigen::Matrix<double, kDof, 1> inertia;
  Eigen::Matrix<double, kDof, 1> damping;
  Eigen::Matrix<double, kDof, 1> gravity_amp;

  // UR5e DH parameters derived from model/ur5e.xml.
  const double L2 = 0.425;
  const double L3 = 0.392;
  double dh_a[kDof] = {0.0, -0.425, -0.392, 0.0, 0.0, 0.0};
  double dh_d[kDof] = {0.163, 0.0, 0.0, 0.127, 0.1, 0.1};
  double dh_alpha[kDof] = {M_PI / 2.0, 0.0, 0.0, M_PI / 2.0, -M_PI / 2.0, 0.0};

  // Cost weights.
  double w_tau = 1.0e-3;
  double w_qdot = 2.0e-2;
  double w_joint_limit = 20.0;
  double w_workspace_obs = 2.0e3;
  double w_ground = 2.0e4;
  double w_terminal_q = 1.5e3;
  double w_terminal_ee = 5.0e2;
  double w_terminal_qdot = 1.0e2;

  // Link collision approximation.
  double link_radius = 0.045;      // [m]
  double obs_safe_margin = 0.10;   // [m]
  double hard_collision_margin = 0.02;

  std::vector<ManipulatorBoxObstacle> workspace_boxes;

  void addWorkspaceBoxMinMax(double xmin, double xmax, double ymin, double ymax,
                             double zmin, double zmax) {
    workspace_boxes.push_back({xmin, xmax, ymin, ymax, zmin, zmax});
  }

  void addWorkspaceBoxMinSize(double x, double y, double z, double w, double h,
                              double d) {
    addWorkspaceBoxMinMax(x, x + w, y, y + h, z, z + d);
  }

  Eigen::VectorXd makeState(const Eigen::VectorXd &q,
                            const Eigen::VectorXd &qdot) const {
    if (q.size() != kDof || qdot.size() != kDof) {
      throw std::runtime_error("ManipulatorDynamicsModel::makeState dimension mismatch");
    }
    Eigen::VectorXd x(2 * kDof);
    x << q, qdot;
    return x;
  }

  Eigen::VectorXd stateQ(const Eigen::VectorXd &x) const { return x.head(kDof); }
  Eigen::VectorXd stateQdot(const Eigen::VectorXd &x) const { return x.segment(kDof, kDof); }

  Eigen::VectorXd continuousDynamics(const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &u) const {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(2 * kDof);
    const Eigen::VectorXd q = x.head(kDof);
    const Eigen::VectorXd qd = x.segment(kDof, kDof);
    const Eigen::VectorXd tau = u.head(kDof).cwiseMax(-tau_max).cwiseMin(tau_max);

    dx.head(kDof) = qd;
    for (int i = 0; i < kDof; ++i) {
      const double g = gravity_amp(i) * std::sin(q(i));
      dx(kDof + i) = (tau(i) - damping(i) * qd(i) - g) / inertia(i);
    }
    return dx;
  }

  Eigen::VectorXd rk4Step(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
                          double dt) const {
    const Eigen::VectorXd k1 = continuousDynamics(x, u);
    const Eigen::VectorXd k2 = continuousDynamics(x + 0.5 * dt * k1, u);
    const Eigen::VectorXd k3 = continuousDynamics(x + 0.5 * dt * k2, u);
    const Eigen::VectorXd k4 = continuousDynamics(x + dt * k3, u);
    Eigen::VectorXd xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    projectStateInPlace(xn);
    return xn;
  }

  void projectStateInPlace(Eigen::VectorXd &x) const {
    x.head(kDof) = x.head(kDof).cwiseMax(q_min).cwiseMin(q_max);
    x.segment(kDof, kDof) = x.segment(kDof, kDof).cwiseMax(-qdot_max).cwiseMin(qdot_max);
  }

  std::array<Eigen::Matrix4d, kDof>
  computeForwardKinematics(const Eigen::VectorXd &q_in) const {
    if (q_in.size() < kDof) {
      throw std::runtime_error("FK requires at least 6 joint coordinates");
    }
    std::array<Eigen::Matrix4d, kDof> T_global;
    Eigen::Matrix4d T_prev = Eigen::Matrix4d::Identity();

    for (int i = 0; i < kDof; ++i) {
      const double theta = q_in(i);
      const double a = dh_a[i];
      const double d = dh_d[i];
      const double alpha = dh_alpha[i];

      const double ct = std::cos(theta);
      const double st = std::sin(theta);
      const double ca = std::cos(alpha);
      const double sa = std::sin(alpha);

      Eigen::Matrix4d T;
      T << ct, -st * ca, st * sa, a * ct,
           st,  ct * ca, -ct * sa, a * st,
           0.0, sa, ca, d,
           0.0, 0.0, 0.0, 1.0;
      T_global[i] = T_prev * T;
      T_prev = T_global[i];
    }
    return T_global;
  }

  std::vector<Eigen::Vector3d>
  getAllJointPositions(const Eigen::VectorXd &x) const override {
    const Eigen::VectorXd q = x.head(kDof);
    const auto T = computeForwardKinematics(q);
    std::vector<Eigen::Vector3d> points;
    points.reserve(kDof + 1);
    points.emplace_back(0.0, 0.0, 0.0);
    for (int i = 0; i < kDof; ++i) {
      points.push_back(T[i].block<3, 1>(0, 3));
    }
    return points;
  }

  Eigen::Vector3d endEffectorPosition(const Eigen::VectorXd &x) const {
    const auto T = computeForwardKinematics(x.head(kDof));
    return T[kDof - 1].block<3, 1>(0, 3);
  }

  double signedDistancePointAabb(const Eigen::Vector3d &p,
                                 const ManipulatorBoxObstacle &b) const {
    const double dx_out = std::max({b.xmin - p.x(), 0.0, p.x() - b.xmax});
    const double dy_out = std::max({b.ymin - p.y(), 0.0, p.y() - b.ymax});
    const double dz_out = std::max({b.zmin - p.z(), 0.0, p.z() - b.zmax});
    const double outside = std::sqrt(dx_out * dx_out + dy_out * dy_out + dz_out * dz_out);

    const bool inside = (p.x() >= b.xmin && p.x() <= b.xmax &&
                         p.y() >= b.ymin && p.y() <= b.ymax &&
                         p.z() >= b.zmin && p.z() <= b.zmax);
    if (!inside) {
      return outside;
    }

    const double dmin = std::min({p.x() - b.xmin, b.xmax - p.x(),
                                  p.y() - b.ymin, b.ymax - p.y(),
                                  p.z() - b.zmin, b.zmax - p.z()});
    return -dmin;
  }

  bool inWorkspaceCollision(const Eigen::VectorXd &x) const {
    if (workspace_boxes.empty()) {
      return false;
    }
    const auto pts = getAllJointPositions(x);
    for (std::size_t i = 1; i < pts.size(); ++i) {
      const Eigen::Vector3d p1 = pts[i - 1];
      const Eigen::Vector3d p2 = pts[i];
      const Eigen::Vector3d pmid = 0.5 * (p1 + p2);
      const std::array<Eigen::Vector3d, 3> checks = {p1, pmid, p2};
      for (const auto &box : workspace_boxes) {
        for (const auto &p : checks) {
          if (p.z() < 0.0) {
            return true;
          }
          const double sd = signedDistancePointAabb(p, box) - link_radius;
          if (sd < hard_collision_margin) {
            return true;
          }
        }
      }
    }
    return false;
  }

  double workspaceObstaclePenalty(const Eigen::VectorXd &x) const {
    if (workspace_boxes.empty()) {
      return 0.0;
    }
    double cost = 0.0;
    const auto pts = getAllJointPositions(x);
    for (std::size_t i = 1; i < pts.size(); ++i) {
      const Eigen::Vector3d p1 = pts[i - 1];
      const Eigen::Vector3d p2 = pts[i];
      const Eigen::Vector3d pmid = 0.5 * (p1 + p2);
      const std::array<Eigen::Vector3d, 3> checks = {p1, pmid, p2};
      for (const auto &box : workspace_boxes) {
        for (const auto &p : checks) {
          if (p.z() < 0.0) {
            cost += w_ground * p.z() * p.z();
          }
          const double sd = signedDistancePointAabb(p, box) - link_radius;
          if (sd < 0.0) {
            cost += 1.0e5 + 1.0e5 * sd * sd;
          } else if (sd < obs_safe_margin) {
            const double e = (obs_safe_margin - sd) / obs_safe_margin;
            cost += w_workspace_obs * e * e;
          }
        }
      }
    }
    return cost;
  }

  double jointLimitPenalty(const Eigen::VectorXd &x) const {
    const Eigen::VectorXd q = x.head(kDof);
    double cost = 0.0;
    for (int i = 0; i < kDof; ++i) {
      const double half_range = 0.5 * (q_max(i) - q_min(i));
      const double center = 0.5 * (q_max(i) + q_min(i));
      const double normalized = (q(i) - center) / (0.85 * half_range);
      const double excess = std::abs(normalized) - 1.0;
      if (excess > 0.0) {
        cost += w_joint_limit * excess * excess;
      }
    }
    return cost;
  }

  double stageCost(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {
    const Eigen::VectorXd qd = x.segment(kDof, kDof);
    const Eigen::VectorXd tau = u.head(kDof);
    return w_tau * tau.squaredNorm() + w_qdot * qd.squaredNorm() +
           jointLimitPenalty(x) + workspaceObstaclePenalty(x);
  }

  double terminalCost(const Eigen::VectorXd &x, const Eigen::VectorXd &x_goal) const {
    const Eigen::VectorXd q = x.head(kDof);
    const Eigen::VectorXd q_goal = x_goal.head(kDof);
    const Eigen::VectorXd qd = x.segment(kDof, kDof);
    const Eigen::Vector3d ee = endEffectorPosition(x);
    const Eigen::Vector3d ee_goal = endEffectorPosition(x_goal);
    return w_terminal_q * (q - q_goal).squaredNorm() +
           w_terminal_ee * (ee - ee_goal).squaredNorm() +
           w_terminal_qdot * qd.squaredNorm();
  }

private:
  void initialize() {
    dim_x = 2 * kDof;
    dim_u = kDof;

    q_min << -6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319;
    q_max <<  6.28319,  6.28319,  3.1415,  6.28319,  6.28319,  6.28319;
    qdot_max << 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    // ur5e.xml actuator force ranges: size3 joints and size1 wrist joints.
    tau_max << 150.0, 150.0, 150.0, 28.0, 28.0, 28.0;

    inertia << 4.0, 3.5, 2.5, 0.9, 0.6, 0.35;
    damping << 3.5, 3.2, 2.5, 0.8, 0.5, 0.35;
    gravity_amp << 0.0, 7.0, 4.5, 0.6, 0.3, 0.15;

    f = [this](const Eigen::VectorXd &x,
               const Eigen::VectorXd &u) -> Eigen::MatrixXd {
      Eigen::MatrixXd dx(dim_x, 1);
      dx.col(0) = continuousDynamics(x, u);
      return dx;
    };

    q = [this](const Eigen::VectorXd &x, const Eigen::VectorXd &u) -> double {
      return stageCost(x, u);
    };

    p = [this](const Eigen::VectorXd &x,
               const Eigen::VectorXd &x_goal) -> double {
      return terminalCost(x, x_goal);
    };

    h = [this](Eigen::Ref<Eigen::MatrixXd> U) -> void {
      for (int t = 0; t < U.cols(); ++t) {
        U.col(t) = U.col(t).cwiseMax(-tau_max).cwiseMin(tau_max);
      }
    };
  }
};
