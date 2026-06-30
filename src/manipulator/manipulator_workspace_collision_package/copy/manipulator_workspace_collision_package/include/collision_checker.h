#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// =============================================================================
// CollisionChecker
// -----------------------------------------------------------------------------
// Backward-compatible checker for the existing BiMPPI_GPU interface.
//
// Supported modes:
//   1) Original 2D q-space / map collision for legacy examples.
//   2) Workspace link collision for a 6-DOF manipulator state x=[q;qdot].
//
// Workspace collision approximates each link by sampled points on the segment
// between adjacent FK joint positions. Obstacles are axis-aligned boxes in the
// manipulator base frame. GPU code receives the same boxes through BiMPPI_GPU.
// =============================================================================
class CollisionChecker {
public:
  bool with_map = false;
  double resolution = 0.05;
  std::vector<std::vector<double>> map;

  // Legacy 2D shapes: circles = {cx, cy, r, r^2}; rectangles = {xmin,xmax,ymin,ymax}.
  std::vector<std::array<double, 4>> circles;
  std::vector<std::array<double, 4>> rectangles;

  // Workspace AABB boxes: {xmin,xmax,ymin,ymax,zmin,zmax} [m].
  std::vector<std::array<double, 6>> workspace_boxes;

  // Link geometry / collision margins [m].
  double link_radius = 0.045;
  double workspace_safe_margin = 0.10;
  double workspace_hard_margin = 0.0;
  bool use_workspace_link_collision = true;

  void clear() {
    map.clear();
    circles.clear();
    rectangles.clear();
    workspace_boxes.clear();
    with_map = false;
  }

  void addQSpaceCircle(double q0, double q1, double radius) {
    circles.push_back({q0, q1, radius, radius * radius});
  }

  void addQSpaceRectangle(double q0_min, double q0_max, double q1_min,
                          double q1_max) {
    rectangles.push_back({q0_min, q0_max, q1_min, q1_max});
  }

  void addWorkspaceBoxMinMax(double xmin, double xmax, double ymin, double ymax,
                             double zmin, double zmax) {
    workspace_boxes.push_back({xmin, xmax, ymin, ymax, zmin, zmax});
  }

  void addWorkspaceBoxMinSize(double x, double y, double z, double w, double h,
                              double d) {
    addWorkspaceBoxMinMax(x, x + w, y, y + h, z, z + d);
  }

  bool getCollisionGrid(const Eigen::VectorXd &x) const {
    if (x.size() < 2) return false;

    if (legacyQSpaceCollision(x)) return true;

    if (use_workspace_link_collision && x.size() >= 6 && !workspace_boxes.empty()) {
      return manipulatorWorkspaceCollision(x);
    }
    return false;
  }

  double workspacePenalty(const Eigen::VectorXd &x) const {
    if (!use_workspace_link_collision || x.size() < 6 || workspace_boxes.empty()) {
      return 0.0;
    }
    const auto pts = manipulatorJointPositions(x);
    double c = 0.0;
    constexpr int kSamples = 5;
    for (std::size_t ell = 1; ell < pts.size(); ++ell) {
      const Eigen::Vector3d a = pts[ell - 1];
      const Eigen::Vector3d b = pts[ell];
      for (int s = 0; s < kSamples; ++s) {
        const double r = static_cast<double>(s) / static_cast<double>(kSamples - 1);
        const Eigen::Vector3d p = (1.0 - r) * a + r * b;
        if (p.z() < 0.0) {
          c += 1.0e5 + 1.0e5 * p.z() * p.z();
        }
        for (const auto &box : workspace_boxes) {
          const double sd = signedDistancePointAabb(p, box) - link_radius;
          if (sd < workspace_hard_margin) {
            c += 1.0e6;
          } else if (sd < workspace_safe_margin) {
            const double e = (workspace_safe_margin - sd) / workspace_safe_margin;
            c += 1.0e4 * e * e;
          }
        }
      }
    }
    return c;
  }

private:
  bool legacyQSpaceCollision(const Eigen::VectorXd &x) const {
    if (with_map && !map.empty() && !map.front().empty()) {
      const int row = static_cast<int>(std::round(x(0) / resolution));
      const int col = static_cast<int>(std::round(x(1) / resolution));
      if (row < 0 || row >= static_cast<int>(map.size())) return true;
      if (col < 0 || col >= static_cast<int>(map[row].size())) return false;
      return map[row][col] == 10.0;
    }

    for (const auto &c : circles) {
      const double dx = x(0) - c[0];
      const double dy = x(1) - c[1];
      if (dx * dx + dy * dy <= c[3]) return true;
    }

    for (const auto &r : rectangles) {
      if (x(0) >= r[0] && x(0) <= r[1] && x(1) >= r[2] && x(1) <= r[3]) {
        return true;
      }
    }
    return false;
  }

  static Eigen::Matrix4d dh(double theta, double a, double d, double alpha) {
    const double ct = std::cos(theta), st = std::sin(theta);
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    Eigen::Matrix4d A;
    A << ct, -st * ca,  st * sa, a * ct,
         st,  ct * ca, -ct * sa, a * st,
        0.0,       sa,       ca,      d,
        0.0,      0.0,      0.0,    1.0;
    return A;
  }

  static std::vector<Eigen::Vector3d> manipulatorJointPositions(const Eigen::VectorXd &x) {
    constexpr int kDof = 6;
    const double L2 = 0.427;
    const double L3 = 0.357;
    const double dh_a[kDof] = {0.0, -L2, -L3, 0.0, 0.0, 0.0};
    const double dh_d[kDof] = {0.15, 0.0, 0.0, 0.11, 0.09, 0.09};
    const double dh_alpha[kDof] = {M_PI / 2.0, 0.0, 0.0, M_PI / 2.0, -M_PI / 2.0, 0.0};

    std::vector<Eigen::Vector3d> pts;
    pts.reserve(kDof + 1);
    pts.emplace_back(0.0, 0.0, 0.0);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < kDof; ++i) {
      T = T * dh(x(i), dh_a[i], dh_d[i], dh_alpha[i]);
      pts.push_back(T.block<3, 1>(0, 3));
    }
    return pts;
  }

  static double signedDistancePointAabb(const Eigen::Vector3d &p,
                                        const std::array<double, 6> &b) {
    const double dx = std::max({b[0] - p.x(), 0.0, p.x() - b[1]});
    const double dy = std::max({b[2] - p.y(), 0.0, p.y() - b[3]});
    const double dz = std::max({b[4] - p.z(), 0.0, p.z() - b[5]});
    const double outside = std::sqrt(dx * dx + dy * dy + dz * dz);
    const bool inside = (p.x() >= b[0] && p.x() <= b[1] &&
                         p.y() >= b[2] && p.y() <= b[3] &&
                         p.z() >= b[4] && p.z() <= b[5]);
    if (!inside) return outside;
    const double din = std::min({p.x() - b[0], b[1] - p.x(),
                                 p.y() - b[2], b[3] - p.y(),
                                 p.z() - b[4], b[5] - p.z()});
    return -din;
  }

  bool manipulatorWorkspaceCollision(const Eigen::VectorXd &x) const {
    const auto pts = manipulatorJointPositions(x);
    constexpr int kSamples = 5;
    for (std::size_t ell = 1; ell < pts.size(); ++ell) {
      const Eigen::Vector3d a = pts[ell - 1];
      const Eigen::Vector3d b = pts[ell];
      for (int s = 0; s < kSamples; ++s) {
        const double r = static_cast<double>(s) / static_cast<double>(kSamples - 1);
        const Eigen::Vector3d p = (1.0 - r) * a + r * b;
        if (p.z() < 0.0) return true;
        for (const auto &box : workspace_boxes) {
          const double sd = signedDistancePointAabb(p, box) - link_radius;
          if (sd < workspace_hard_margin) return true;
        }
      }
    }
    return false;
  }
};
