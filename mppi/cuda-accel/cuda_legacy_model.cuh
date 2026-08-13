#pragma once
#ifndef CUDA_LEGACY_MODEL_CUH
#define CUDA_LEGACY_MODEL_CUH

#include <cmath>
#include <cstring>

#include "quadrotor_landing_cost.h"

enum LegacyCudaModelType {
  LEGACY_CUDA_WMROBOT = 0,
  LEGACY_CUDA_QUADROTOR = 1,
  LEGACY_CUDA_VELO = 2,
  LEGACY_CUDA_MANIPULATOR = 3,
  LEGACY_CUDA_BICYCLE = 4,
  LEGACY_CUDA_QUADROTOR_PRECISION_LANDING = 5,
};

inline int legacy_cuda_model_type_from_name(const char *type_name, int dim_x,
                                            int dim_u) {
  if (type_name && std::strstr(type_name, "QuadrotorPrecisionLanding")) {
    return LEGACY_CUDA_QUADROTOR_PRECISION_LANDING;
  }
  if (type_name && std::strstr(type_name, "Quadrotor")) {
    return LEGACY_CUDA_QUADROTOR;
  }
  if (type_name && std::strstr(type_name, "Velo")) {
    return LEGACY_CUDA_VELO;
  }
  if (type_name && std::strstr(type_name, "Manipulator")) {
    return LEGACY_CUDA_MANIPULATOR;
  }
  if (type_name && std::strstr(type_name, "Bicycle")) {
    return LEGACY_CUDA_BICYCLE;
  }
  if (type_name && std::strstr(type_name, "WMRobot")) {
    return LEGACY_CUDA_WMROBOT;
  }

  if (dim_x == 6 && dim_u == 3) {
    return LEGACY_CUDA_QUADROTOR;
  }
  if (dim_x == 6 && dim_u == 6) {
    return LEGACY_CUDA_MANIPULATOR;
  }
  if (dim_x == 4 && dim_u == 2) {
    return LEGACY_CUDA_VELO;
  }
  return LEGACY_CUDA_WMROBOT;
}

__device__ __forceinline__ double legacy_cuda_clamp(double v, double lo,
                                                    double hi) {
  return fmax(lo, fmin(hi, v));
}

__device__ __forceinline__ void
legacy_cuda_manipulator_fk(const double *q, double joint_pts[7][3]) {
  const double a[6] = {0.0, -0.425, -0.392, 0.0, 0.0, 0.0};
  const double d[6] = {0.163, 0.0, 0.0, 0.127, 0.1, 0.1};
  const double alpha[6] = {M_PI_2, 0.0, 0.0, M_PI_2, -M_PI_2, 0.0};

  double T[16] = {1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0};
  joint_pts[0][0] = 0.0; joint_pts[0][1] = 0.0; joint_pts[0][2] = 0.0;

  for (int i = 0; i < 6; ++i) {
    double th = q[i];
    double ct = cos(th), st = sin(th);
    double ca = cos(alpha[i]), sa = sin(alpha[i]);
    double ai = a[i], di = d[i];

    double A[16] = {
      ct, -st * ca,  st * sa, ai * ct,
      st,  ct * ca, -ct * sa, ai * st,
     0.0,       sa,       ca,      di,
     0.0,      0.0,      0.0,     1.0
    };

    double T_next[16];
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) {
          sum += T[r * 4 + k] * A[k * 4 + c];
        }
        T_next[r * 4 + c] = sum;
      }
    }
    for (int k = 0; k < 16; ++k) T[k] = T_next[k];
    joint_pts[i + 1][0] = T[3];
    joint_pts[i + 1][1] = T[7];
    joint_pts[i + 1][2] = T[11];
  }
}

__device__ __forceinline__ void
legacy_cuda_dynamics(const double *x, const double *u, double *x_dot, int dim_x,
                     int dim_u, int model_type) {
  for (int d = 0; d < dim_x; ++d) {
    x_dot[d] = 0.0;
  }

  if (model_type == LEGACY_CUDA_QUADROTOR ||
      model_type == LEGACY_CUDA_QUADROTOR_PRECISION_LANDING) {
    x_dot[0] = x[3];
    x_dot[1] = x[4];
    x_dot[2] = x[5];
    x_dot[3] = u[0];
    x_dot[4] = u[1];
    x_dot[5] = u[2] - 9.81;
    return;
  }

  if (model_type == LEGACY_CUDA_VELO) {
    x_dot[0] = x[2];
    x_dot[1] = x[3];
    x_dot[2] = u[0];
    x_dot[3] = u[1];
    return;
  }

  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    if (dim_x >= 12 && dim_u >= 6) {
      const double tau_max[6] = {150.0, 150.0, 150.0, 28.0, 28.0, 28.0};
      const double inertia[6] = {4.0, 3.5, 2.5, 0.9, 0.6, 0.35};
      const double damping[6] = {3.5, 3.2, 2.5, 0.8, 0.5, 0.35};
      const double gravity_amp[6] = {0.0, 7.0, 4.5, 0.6, 0.3, 0.15};

      for (int i = 0; i < 6; ++i) {
        x_dot[i] = x[6 + i];
        double tau_c = legacy_cuda_clamp(u[i], -tau_max[i], tau_max[i]);
        double g = gravity_amp[i] * sin(x[i]);
        x_dot[6 + i] = (tau_c - damping[i] * x[6 + i] - g) / inertia[i];
      }
    } else {
      for (int d = 0; d < dim_x && d < dim_u; ++d) {
        x_dot[d] = u[d];
      }
    }
    return;
  }

  if (model_type == LEGACY_CUDA_BICYCLE) {
    const double wheelbase = 0.2;
    x_dot[0] = x[3] * cos(x[2]);
    x_dot[1] = x[3] * sin(x[2]);
    x_dot[2] = x[3] * tan(u[1]) / wheelbase;
    x_dot[3] = u[0];
    return;
  }

  x_dot[0] = u[0] * cos(x[2]);
  x_dot[1] = u[0] * sin(x[2]);
  x_dot[2] = u[1];
}

__device__ __forceinline__ double
legacy_cuda_terminal_cost(const double *x, const double *x_target, int dim_x,
                          int model_type) {
  double s = 0.0;

  if (model_type == LEGACY_CUDA_QUADROTOR) {
    for (int d = 0; d < 3; ++d) {
      double diff = x[d] - x_target[d];
      s += diff * diff;
    }
    return sqrt(s);
  }

  if (model_type == LEGACY_CUDA_QUADROTOR_PRECISION_LANDING) {
    return quadrotor_landing_cost::terminalCost(x, x_target);
  }

  if (model_type == LEGACY_CUDA_VELO || model_type == LEGACY_CUDA_BICYCLE) {
    for (int d = 0; d < 2; ++d) {
      double diff = x[d] - x_target[d];
      s += diff * diff;
    }
    return sqrt(s);
  }

  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    if (dim_x >= 12) {
      double w_terminal_q = 1.5e3;
      double w_terminal_ee = 5.0e2;
      double w_terminal_qdot = 1.0e2;

      double sq_q = 0.0;
      for (int d = 0; d < 6; ++d) {
        double diff = x[d] - x_target[d];
        sq_q += diff * diff;
      }

      double sq_qdot = 0.0;
      for (int d = 6; d < 12; ++d) {
        sq_qdot += x[d] * x[d];
      }

      double pts_x[7][3], pts_t[7][3];
      legacy_cuda_manipulator_fk(x, pts_x);
      legacy_cuda_manipulator_fk(x_target, pts_t);

      double sq_ee = 0.0;
      for (int c = 0; c < 3; ++c) {
        double diff = pts_x[6][c] - pts_t[6][c];
        sq_ee += diff * diff;
      }

      return w_terminal_q * sq_q + w_terminal_ee * sq_ee + w_terminal_qdot * sq_qdot;
    }

    for (int d = 0; d < dim_x; ++d) {
      double diff = x[d] - x_target[d];
      s += diff * diff;
    }
    return 1000.0 * s;
  }

  for (int d = 0; d < dim_x; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

__device__ __forceinline__ double
legacy_cuda_manipulator_workspace_penalty(const double *x,
                                          const double *boxes, int n_boxes) {
  if (boxes == nullptr || n_boxes <= 0) {
    return 0.0;
  }

  double points[7][3];
  legacy_cuda_manipulator_fk(x, points);

  constexpr double kLinkRadius = 0.045;
  constexpr double kSafeMargin = 0.10;
  constexpr double kWorkspaceWeight = 2.0e3;
  constexpr double kGroundWeight = 2.0e4;
  double cost = 0.0;

  // Keep the same three samples as ManipulatorDynamicsModel::workspaceObstaclePenalty:
  // segment start, midpoint, and segment end.
  for (int ell = 1; ell <= 6; ++ell) {
    for (int b = 0; b < n_boxes; ++b) {
      const double *box = boxes + 6 * b;
      for (int sample = 0; sample < 3; ++sample) {
        const double r = 0.5 * static_cast<double>(sample);
        const double px = (1.0 - r) * points[ell - 1][0] + r * points[ell][0];
        const double py = (1.0 - r) * points[ell - 1][1] + r * points[ell][1];
        const double pz = (1.0 - r) * points[ell - 1][2] + r * points[ell][2];

        if (pz < 0.0) {
          cost += kGroundWeight * pz * pz;
        }

        const double dx = fmax(fmax(box[0] - px, 0.0), px - box[1]);
        const double dy = fmax(fmax(box[2] - py, 0.0), py - box[3]);
        const double dz = fmax(fmax(box[4] - pz, 0.0), pz - box[5]);
        double signed_distance = sqrt(dx * dx + dy * dy + dz * dz);
        const bool inside = px >= box[0] && px <= box[1] &&
                            py >= box[2] && py <= box[3] &&
                            pz >= box[4] && pz <= box[5];
        if (inside) {
          const double distance_to_boundary =
              fmin(fmin(px - box[0], box[1] - px),
                   fmin(fmin(py - box[2], box[3] - py),
                        fmin(pz - box[4], box[5] - pz)));
          signed_distance = -distance_to_boundary;
        }
        signed_distance -= kLinkRadius;

        if (signed_distance < 0.0) {
          cost += 1.0e5 + 1.0e5 * signed_distance * signed_distance;
        } else if (signed_distance < kSafeMargin) {
          const double excess =
              (kSafeMargin - signed_distance) / kSafeMargin;
          cost += kWorkspaceWeight * excess * excess;
        }
      }
    }
  }
  return cost;
}

__device__ __forceinline__ double
legacy_cuda_stage_cost(const double *x, const double *u, int dim_x, int dim_u,
                       int model_type, const double *boxes, int n_boxes) {
  if (model_type == LEGACY_CUDA_MANIPULATOR && dim_x >= 12 && dim_u >= 6) {
    constexpr double kJointLimitWeight = 20.0;
    constexpr double kTorqueWeight = 1.0e-3;
    constexpr double kQdotWeight = 2.0e-2;
    const double q_min[6] = {-6.28319, -6.28319, -3.1415,
                             -6.28319, -6.28319, -6.28319};
    const double q_max[6] = {6.28319, 6.28319, 3.1415,
                             6.28319, 6.28319, 6.28319};

    double torque_sq = 0.0;
    double qdot_sq = 0.0;
    double joint_limit_cost = 0.0;
    for (int i = 0; i < 6; ++i) {
      torque_sq += u[i] * u[i];
      qdot_sq += x[6 + i] * x[6 + i];

      const double half_range = 0.5 * (q_max[i] - q_min[i]);
      const double center = 0.5 * (q_max[i] + q_min[i]);
      const double normalized = (x[i] - center) / (0.85 * half_range);
      const double excess = fabs(normalized) - 1.0;
      if (excess > 0.0) {
        joint_limit_cost += kJointLimitWeight * excess * excess;
      }
    }

    return kTorqueWeight * torque_sq + kQdotWeight * qdot_sq +
           joint_limit_cost +
           legacy_cuda_manipulator_workspace_penalty(x, boxes, n_boxes);
  }

  double torque_sq = 0.0;
  for (int i = 0; i < dim_u; ++i) {
    torque_sq += u[i] * u[i];
  }
  return 1.0e-3 * torque_sq;
}

__device__ __forceinline__ void legacy_cuda_project_control(double *u, int dim_u,
                                                             int T,
                                                             int model_type) {
  if (model_type == LEGACY_CUDA_QUADROTOR ||
      model_type == LEGACY_CUDA_QUADROTOR_PRECISION_LANDING) {
    for (int t = 0; t < T; ++t) {
      double u0 = u[0 * T + t];
      double u1 = u[1 * T + t];
      double u2 = u[2 * T + t];

      double norm_u = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
      if (norm_u >= 20.0) {
        double mask = 20.0 / norm_u;
        u0 *= mask;
        u1 *= mask;
        u2 *= mask;
      }

      double V = sqrt(u0 * u0 + u1 * u1);
      double S = u2 * 1.7320508075688772935;
      if (V < -S) {
        u0 = 0.0;
        u1 = 0.0;
        u2 = 0.0;
        V = 0.0;
        S = 0.0;
      }
      if (V > fabs(S)) {
        double mul = 0.5 * (1.0 + S / V);
        u0 *= mul;
        u1 *= mul;
        u2 = V * mul;
      }

      u[0 * T + t] = u0;
      u[1 * T + t] = u1;
      u[2 * T + t] = u2;
    }
    return;
  }

  if (model_type == LEGACY_CUDA_VELO) {
    for (int t = 0; t < T; ++t) {
      u[0 * T + t] = legacy_cuda_clamp(u[0 * T + t], 0.0, 1.0);
      u[1 * T + t] = legacy_cuda_clamp(u[1 * T + t], 0.0, 1.0);
    }
    return;
  }

  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    const double tau_max[6] = {150.0, 150.0, 150.0, 28.0, 28.0, 28.0};
    for (int d = 0; d < dim_u && d < 6; ++d) {
      for (int t = 0; t < T; ++t) {
        u[d * T + t] =
            legacy_cuda_clamp(u[d * T + t], -tau_max[d], tau_max[d]);
      }
    }
    return;
  }

  if (model_type == LEGACY_CUDA_BICYCLE) {
    for (int t = 0; t < T; ++t) {
      u[0 * T + t] = legacy_cuda_clamp(u[0 * T + t], -3.0, 3.0);
      u[1 * T + t] =
          legacy_cuda_clamp(u[1 * T + t], -M_PI / 4.0, M_PI / 4.0);
    }
    return;
  }

  for (int t = 0; t < T; ++t) {
    u[0 * T + t] = legacy_cuda_clamp(u[0 * T + t], 0.0, 1.0);
    u[1 * T + t] = legacy_cuda_clamp(u[1 * T + t], -M_PI / 2.0, M_PI / 2.0);
  }
}

__device__ __forceinline__ bool
legacy_cuda_collision_grid_polygon(const double *x, const double *circles,
                                   int n_circles, const double *rects,
                                   int n_rects) {
  for (int i = 0; i < n_circles; ++i) {
    double dx = circles[i * 4 + 0] - x[0];
    double dy = circles[i * 4 + 1] - x[1];
    if (dx * dx + dy * dy <= circles[i * 4 + 3]) {
      return true;
    }
  }

  for (int i = 0; i < n_rects; ++i) {
    if (x[0] < rects[i * 4 + 0]) {
      continue;
    } else if (rects[i * 4 + 1] < x[0]) {
      continue;
    } else if (x[1] < rects[i * 4 + 2]) {
      continue;
    } else if (rects[i * 4 + 3] < x[1]) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ bool
legacy_cuda_collision_grid_map(const double *x, const double *d_map,
                               int max_row, int max_col, double resolution) {
  int nx = (int)round(x[0] / resolution);
  int ny = (int)round(x[1] / resolution);
  if (nx < 0 || max_row <= nx) {
    return true;
  }
  if (ny < 0 || max_col <= ny) {
    return false;
  }
  return d_map[nx * max_col + ny] == 10.0;
}

__device__ __forceinline__ bool
legacy_cuda_collision_grid(const double *x, bool with_map, const double *d_map,
                           int max_row, int max_col, double resolution,
                           const double *circles, int n_circles,
                           const double *rects, int n_rects,
                           int model_type = 0) {
  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    double pts[7][3];
    legacy_cuda_manipulator_fk(x, pts);
    const double link_radius = 0.045;
    const double margin = 0.02;

    // 1. Ground collision check
    for (int i = 1; i < 7; ++i) {
      if (pts[i][2] < link_radius) {
        return true;
      }
    }

    // 2. 3D Workspace Box (AABB) link collision check
    if (n_rects > 0 && rects != nullptr) {
      for (int ell = 1; ell < 7; ++ell) {
        const double p1[3] = {pts[ell - 1][0], pts[ell - 1][1], pts[ell - 1][2]};
        const double p2[3] = {pts[ell][0],     pts[ell][1],     pts[ell][2]};
        for (int s = 0; s < 5; ++s) {
          const double r = (double)s / 4.0;
          const double px = (1.0 - r) * p1[0] + r * p2[0];
          const double py = (1.0 - r) * p1[1] + r * p2[1];
          const double pz = (1.0 - r) * p1[2] + r * p2[2];

          for (int b = 0; b < n_rects; ++b) {
            const double xmin = rects[b * 6 + 0];
            const double xmax = rects[b * 6 + 1];
            const double ymin = rects[b * 6 + 2];
            const double ymax = rects[b * 6 + 3];
            const double zmin = rects[b * 6 + 4];
            const double zmax = rects[b * 6 + 5];

            const double dx = fmax(0.0, fmax(xmin - px, px - xmax));
            const double dy = fmax(0.0, fmax(ymin - py, py - ymax));
            const double dz = fmax(0.0, fmax(zmin - pz, pz - zmax));
            const double outside = sqrt(dx * dx + dy * dy + dz * dz);

            const bool inside = (px >= xmin && px <= xmax &&
                                 py >= ymin && py <= ymax &&
                                 pz >= zmin && pz <= zmax);
            double sd = 0.0;
            if (inside) {
              const double d1 = px - xmin, d2 = xmax - px;
              const double d3 = py - ymin, d4 = ymax - py;
              const double d5 = pz - zmin, d6 = zmax - pz;
              const double min_in = fmin(fmin(d1, d2), fmin(fmin(d3, d4), fmin(d5, d6)));
              sd = -min_in;
            } else {
              sd = outside;
            }

            if (sd - link_radius < margin) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  if (with_map) {
    return legacy_cuda_collision_grid_map(x, d_map, max_row, max_col,
                                          resolution);
  }
  return legacy_cuda_collision_grid_polygon(x, circles, n_circles, rects,
                                            n_rects);
}

#endif // CUDA_LEGACY_MODEL_CUH
