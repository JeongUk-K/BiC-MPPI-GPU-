#pragma once
// =============================================================================
// manipulator_model.h
// -----------------------------------------------------------------------------
// Manipulator-specific model (ManipulatorModel) and workspace collision checker
// (ManipulatorCollisionChecker) for use with the generic cuda-accel solvers.
//
// Usage:
//   #include <mppi_gpu.cuh>           // from mppi/cuda-accel/
//   #include "manipulator_model.h"    // from model/
//
//   ManipulatorModel model;
//   MPPI_GPU solver(model);
// =============================================================================

#include "model_base.h"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Box obstacle ─────────────────────────────────────────────────────────────
struct ManipulatorBoxObstacle {
  double xmin, xmax, ymin, ymax, zmin, zmax;
};

// =============================================================================
// ManipulatorModel
// -----------------------------------------------------------------------------
// 6-DOF torque-controlled manipulator (UR5e-like parameters).
// State: x = [q(6); qdot(6)]  (12-dim)
// Control: u = tau(6)
//
// Inherits ModelBase so it can be passed directly to cuda-accel solvers.
// =============================================================================
class ManipulatorModel : public ModelBase {
public:
  static constexpr int kDof = 6;

  // ── Cost weights ────────────────────────────────────────────────────────────
  double w_tau             = 1.0e-3;
  double w_qdot            = 2.0e-2;
  double w_joint_limit     = 20.0;
  double w_workspace_obs   = 2.0e3;
  double w_ground          = 2.0e4;
  double w_terminal_q      = 1.5e3;
  double w_terminal_ee     = 5.0e2;
  double w_terminal_qdot   = 1.0e2;

  // ── Link geometry ───────────────────────────────────────────────────────────
  double link_radius           = 0.045;   // [m]
  double obs_safe_margin       = 0.10;    // [m]
  double hard_collision_margin = 0.02;    // [m]

  // ── Workspace obstacles ─────────────────────────────────────────────────────
  std::vector<ManipulatorBoxObstacle> workspace_boxes;

  void addWorkspaceBoxMinMax(double xmin, double xmax,
                             double ymin, double ymax,
                             double zmin, double zmax) {
    workspace_boxes.push_back({xmin, xmax, ymin, ymax, zmin, zmax});
  }

  // ── Constructor ─────────────────────────────────────────────────────────────
  ManipulatorModel() { initialize(); }

  // ─────────────────────────────────────────────────────────────────────────
  // Forward kinematics (UR5e DH parameters)
  // ─────────────────────────────────────────────────────────────────────────
  static constexpr double L2 = 0.425;
  static constexpr double L3 = 0.392;
  static const double* dh_a_arr()     { static const double a[kDof] = {0.0, -L2, -L3, 0.0, 0.0, 0.0}; return a; }
  static const double* dh_d_arr()     { static const double d[kDof] = {0.163, 0.0, 0.0, 0.127, 0.1, 0.1}; return d; }
  static const double* dh_alpha_arr() { static const double al[kDof] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, -M_PI/2.0, 0.0}; return al; }

  std::vector<Eigen::Vector3d>
  getAllJointPositions(const Eigen::VectorXd& x) const override {
    const double* a  = dh_a_arr();
    const double* d  = dh_d_arr();
    const double* al = dh_alpha_arr();
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(kDof + 1);
    pts.emplace_back(0.0, 0.0, 0.0);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < kDof; ++i) {
      T = T * dhMatrix(x(i), a[i], d[i], al[i]);
      pts.push_back(T.block<3,1>(0,3));
    }
    return pts;
  }

  Eigen::Vector3d endEffectorPosition(const Eigen::VectorXd& x) const {
    auto pts = getAllJointPositions(x);
    return pts.back();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cost functions (used by f/q/p lambdas)
  // ─────────────────────────────────────────────────────────────────────────
  double workspaceObstaclePenalty(const Eigen::VectorXd& x) const {
    if (workspace_boxes.empty()) return 0.0;
    const auto pts = getAllJointPositions(x);
    double cost = 0.0;
    constexpr int kSamples = 5;
    for (std::size_t ell = 1; ell < pts.size(); ++ell) {
      const Eigen::Vector3d p1 = pts[ell - 1];
      const Eigen::Vector3d p2 = pts[ell];
      for (int s = 0; s < kSamples; ++s) {
        const double r = static_cast<double>(s) / (kSamples - 1);
        const Eigen::Vector3d p = (1.0 - r) * p1 + r * p2;
        if (p.z() < 0.0)
          cost += w_ground * (1.0 + p.z() * p.z());
        for (const auto& box : workspace_boxes) {
          const double sd = signedDistanceAabb(p, box) - link_radius;
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

  double jointLimitPenalty(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd& q = x.head(kDof);
    double cost = 0.0;
    for (int i = 0; i < kDof; ++i) {
      const double half  = 0.5 * (q_max(i) - q_min(i));
      const double mid   = 0.5 * (q_max(i) + q_min(i));
      const double norm  = (q(i) - mid) / (0.85 * half);
      const double excess = std::abs(norm) - 1.0;
      if (excess > 0.0) cost += w_joint_limit * excess * excess;
    }
    return cost;
  }

  double stageCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
    const Eigen::VectorXd qd  = x.segment(kDof, kDof);
    const Eigen::VectorXd tau = u.head(kDof);
    return w_tau * tau.squaredNorm()
         + w_qdot * qd.squaredNorm()
         + jointLimitPenalty(x)
         + workspaceObstaclePenalty(x);
  }

  double terminalCost(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& x_goal) const {
    const Eigen::VectorXd q      = x.head(kDof);
    const Eigen::VectorXd q_goal = x_goal.head(kDof);
    const Eigen::VectorXd qd     = x.segment(kDof, kDof);
    const Eigen::Vector3d ee      = endEffectorPosition(x);
    const Eigen::Vector3d ee_goal = endEffectorPosition(x_goal);
    return w_terminal_q   * (q - q_goal).squaredNorm()
         + w_terminal_ee  * (ee - ee_goal).squaredNorm()
         + w_terminal_qdot * qd.squaredNorm();
  }

private:
  // ── Physical parameters ──────────────────────────────────────────────────
  Eigen::VectorXd q_min, q_max, qdot_max, tau_max;
  Eigen::VectorXd inertia, damping, gravity_amp;

  static Eigen::Matrix4d dhMatrix(double theta, double a, double d, double alpha) {
    const double ct = std::cos(theta), st = std::sin(theta);
    const double ca = std::cos(alpha),  sa = std::sin(alpha);
    Eigen::Matrix4d A;
    A << ct, -st*ca,  st*sa, a*ct,
         st,  ct*ca, -ct*sa, a*st,
        0.0,     sa,     ca,    d,
        0.0,    0.0,    0.0,  1.0;
    return A;
  }

  static double signedDistanceAabb(const Eigen::Vector3d& p,
                                   const ManipulatorBoxObstacle& b) {
    const double dx = std::max({b.xmin - p.x(), 0.0, p.x() - b.xmax});
    const double dy = std::max({b.ymin - p.y(), 0.0, p.y() - b.ymax});
    const double dz = std::max({b.zmin - p.z(), 0.0, p.z() - b.zmax});
    const double outside = std::sqrt(dx*dx + dy*dy + dz*dz);
    const bool inside = (p.x() >= b.xmin && p.x() <= b.xmax &&
                         p.y() >= b.ymin && p.y() <= b.ymax &&
                         p.z() >= b.zmin && p.z() <= b.zmax);
    if (!inside) return outside;
    return -std::min({p.x()-b.xmin, b.xmax-p.x(),
                      p.y()-b.ymin, b.ymax-p.y(),
                      p.z()-b.zmin, b.zmax-p.z()});
  }

  Eigen::VectorXd continuousDynamics(const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& u) const {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(dim_x);
    const Eigen::VectorXd q  = x.head(kDof);
    const Eigen::VectorXd qd = x.segment(kDof, kDof);
    const Eigen::VectorXd tau = u.head(kDof)
                                 .cwiseMax(-tau_max).cwiseMin(tau_max);
    dx.head(kDof) = qd;
    for (int i = 0; i < kDof; ++i) {
      const double g = gravity_amp(i) * std::sin(q(i));
      dx(kDof + i) = (tau(i) - damping(i) * qd(i) - g) / inertia(i);
    }
    return dx;
  }

  void initialize() {
    dim_x = 2 * kDof;
    dim_u = kDof;

    q_min.resize(kDof);    q_max.resize(kDof);
    qdot_max.resize(kDof); tau_max.resize(kDof);
    inertia.resize(kDof);  damping.resize(kDof); gravity_amp.resize(kDof);

    q_min    << -6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319;
    q_max    <<  6.28319,  6.28319,  3.1415,  6.28319,  6.28319,  6.28319;
    qdot_max << 2.0, 2.0, 2.0, 2.5, 2.5, 2.5;
    tau_max  << 150.0, 150.0, 150.0, 28.0, 28.0, 28.0;
    inertia      << 4.0, 3.5, 2.5, 0.9, 0.6, 0.35;
    damping      << 3.5, 3.2, 2.5, 0.8, 0.5, 0.35;
    gravity_amp  << 0.0, 7.0, 4.5, 0.6, 0.3, 0.15;

    // ── Callbacks for cuda-accel solvers ─────────────────────────────────
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u)
        -> Eigen::MatrixXd {
      Eigen::MatrixXd dx(dim_x, 1);
      dx.col(0) = continuousDynamics(x, u);
      return dx;
    };

    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
      return stageCost(x, u);
    };

    p = [this](const Eigen::VectorXd& x,
               const Eigen::VectorXd& xg) -> double {
      return terminalCost(x, xg);
    };

    h = [this](Eigen::Ref<Eigen::MatrixXd> U) {
      for (int t = 0; t < U.cols(); ++t)
        U.col(t) = U.col(t).cwiseMax(-tau_max).cwiseMin(tau_max);
    };
  }
};
