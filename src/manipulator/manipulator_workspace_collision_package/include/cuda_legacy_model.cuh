#pragma once

#include <cmath>
#include <cstring>

#ifdef BICMPPI_USE_QUADROTOR_LANDING_COST
#include "quadrotor_landing_cost.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  if (type_name && std::strstr(type_name, "QuadrotorPrecisionLanding")) return LEGACY_CUDA_QUADROTOR_PRECISION_LANDING;
  if (type_name && std::strstr(type_name, "Quadrotor")) return LEGACY_CUDA_QUADROTOR;
  if (type_name && std::strstr(type_name, "Velo")) return LEGACY_CUDA_VELO;
  if (type_name && std::strstr(type_name, "Manipulator")) return LEGACY_CUDA_MANIPULATOR;
  if (type_name && std::strstr(type_name, "Bicycle")) return LEGACY_CUDA_BICYCLE;
  if (type_name && std::strstr(type_name, "WMRobot")) return LEGACY_CUDA_WMROBOT;
  if ((dim_x == 12 && dim_u == 6) || (dim_x == 6 && dim_u == 6)) return LEGACY_CUDA_MANIPULATOR;
  if (dim_x == 6 && dim_u == 3) return LEGACY_CUDA_QUADROTOR;
  if (dim_x == 4 && dim_u == 2) return LEGACY_CUDA_VELO;
  return LEGACY_CUDA_WMROBOT;
}

__device__ __forceinline__ double legacy_cuda_clamp(double v, double lo, double hi) {
  return fmax(lo, fmin(hi, v));
}

// -----------------------------------------------------------------------------
// Dynamics
// -----------------------------------------------------------------------------
__device__ __forceinline__ void
legacy_cuda_dynamics(const double *x, const double *u, double *x_dot, int dim_x,
                     int dim_u, int model_type) {
  for (int d = 0; d < dim_x; ++d) x_dot[d] = 0.0;

  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    if (dim_x >= 12 && dim_u >= 6) {
      const double inertia[6] = {4.0, 3.5, 2.5, 0.9, 0.6, 0.35};
      const double damping[6] = {3.5, 3.2, 2.5, 0.8, 0.5, 0.35};
      const double gravity_amp[6] = {0.0, 7.0, 4.5, 0.6, 0.3, 0.15};
      const double tau_max[6] = {18.0, 18.0, 14.0, 8.0, 6.0, 4.0};
      for (int i = 0; i < 6; ++i) {
        const double tau = legacy_cuda_clamp(u[i], -tau_max[i], tau_max[i]);
        x_dot[i] = x[6 + i];
        x_dot[6 + i] = (tau - damping[i] * x[6 + i] - gravity_amp[i] * sin(x[i])) / inertia[i];
      }
      return;
    }
    // Legacy 6D velocity-controlled manipulator: x=q, u=qdot.
    for (int i = 0; i < dim_x && i < dim_u; ++i) x_dot[i] = u[i];
    return;
  }

  if (model_type == LEGACY_CUDA_QUADROTOR ||
      model_type == LEGACY_CUDA_QUADROTOR_PRECISION_LANDING) {
    x_dot[0] = x[3]; x_dot[1] = x[4]; x_dot[2] = x[5];
    x_dot[3] = u[0]; x_dot[4] = u[1]; x_dot[5] = u[2] - 9.81;
    return;
  }

  if (model_type == LEGACY_CUDA_VELO) {
    x_dot[0] = x[2]; x_dot[1] = x[3]; x_dot[2] = u[0]; x_dot[3] = u[1];
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

// -----------------------------------------------------------------------------
// Manipulator FK and workspace link collision utilities
// -----------------------------------------------------------------------------
__device__ __forceinline__ void legacy_cuda_mat4_mul(const double *A, const double *B, double *C) {
  double R[16];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      double v = 0.0;
      for (int k = 0; k < 4; ++k) v += A[4 * r + k] * B[4 * k + c];
      R[4 * r + c] = v;
    }
  }
  for (int i = 0; i < 16; ++i) C[i] = R[i];
}

__device__ __forceinline__ void legacy_cuda_dh(double theta, double a, double d,
                                               double alpha, double *A) {
  const double ct = cos(theta), st = sin(theta);
  const double ca = cos(alpha), sa = sin(alpha);
  A[0] = ct;   A[1] = -st * ca; A[2] =  st * sa; A[3] = a * ct;
  A[4] = st;   A[5] =  ct * ca; A[6] = -ct * sa; A[7] = a * st;
  A[8] = 0.0;  A[9] = sa;       A[10] = ca;      A[11] = d;
  A[12] = 0.0; A[13] = 0.0;     A[14] = 0.0;     A[15] = 1.0;
}

__device__ __forceinline__ void
legacy_cuda_manipulator_fk_points(const double *x, double *px, double *py, double *pz) {
  const double L2 = 0.427;
  const double L3 = 0.357;
  const double a[6] = {0.0, -L2, -L3, 0.0, 0.0, 0.0};
  const double d[6] = {0.15, 0.0, 0.0, 0.11, 0.09, 0.09};
  const double alpha[6] = {M_PI / 2.0, 0.0, 0.0, M_PI / 2.0, -M_PI / 2.0, 0.0};

  double T[16] = {1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0};
  px[0] = 0.0; py[0] = 0.0; pz[0] = 0.0;
  for (int i = 0; i < 6; ++i) {
    double A[16];
    legacy_cuda_dh(x[i], a[i], d[i], alpha[i], A);
    legacy_cuda_mat4_mul(T, A, T);
    px[i + 1] = T[3];
    py[i + 1] = T[7];
    pz[i + 1] = T[11];
  }
}

__device__ __forceinline__ double
legacy_cuda_signed_distance_point_aabb(double x, double y, double z, const double *b) {
  const double dx = fmax(fmax(b[0] - x, 0.0), x - b[1]);
  const double dy = fmax(fmax(b[2] - y, 0.0), y - b[3]);
  const double dz = fmax(fmax(b[4] - z, 0.0), z - b[5]);
  const double outside = sqrt(dx * dx + dy * dy + dz * dz);
  const bool inside = (x >= b[0] && x <= b[1] && y >= b[2] && y <= b[3] && z >= b[4] && z <= b[5]);
  if (!inside) return outside;
  double din = x - b[0];
  din = fmin(din, b[1] - x);
  din = fmin(din, y - b[2]);
  din = fmin(din, b[3] - y);
  din = fmin(din, z - b[4]);
  din = fmin(din, b[5] - z);
  return -din;
}

__device__ __forceinline__ bool
legacy_cuda_manipulator_workspace_collision(const double *x, const double *boxes,
                                            int n_boxes, double link_radius,
                                            double hard_margin) {
  if (n_boxes <= 0 || boxes == nullptr) return false;
  double px[7], py[7], pz[7];
  legacy_cuda_manipulator_fk_points(x, px, py, pz);
  constexpr int kSamples = 5;
  for (int ell = 1; ell <= 6; ++ell) {
    for (int s = 0; s < kSamples; ++s) {
      const double r = static_cast<double>(s) / static_cast<double>(kSamples - 1);
      const double qx = (1.0 - r) * px[ell - 1] + r * px[ell];
      const double qy = (1.0 - r) * py[ell - 1] + r * py[ell];
      const double qz = (1.0 - r) * pz[ell - 1] + r * pz[ell];
      if (qz < 0.0) return true;
      for (int b = 0; b < n_boxes; ++b) {
        const double sd = legacy_cuda_signed_distance_point_aabb(qx, qy, qz, boxes + 6 * b) - link_radius;
        if (sd < hard_margin) return true;
      }
    }
  }
  return false;
}

__device__ __forceinline__ double
legacy_cuda_workspace_obstacle_cost(const double *x, const double *boxes, int n_boxes,
                                    double link_radius, double safe_margin,
                                    double hard_margin) {
  if (n_boxes <= 0 || boxes == nullptr) return 0.0;
  double px[7], py[7], pz[7];
  legacy_cuda_manipulator_fk_points(x, px, py, pz);
  double cost = 0.0;
  constexpr int kSamples = 5;
  for (int ell = 1; ell <= 6; ++ell) {
    for (int s = 0; s < kSamples; ++s) {
      const double r = static_cast<double>(s) / static_cast<double>(kSamples - 1);
      const double qx = (1.0 - r) * px[ell - 1] + r * px[ell];
      const double qy = (1.0 - r) * py[ell - 1] + r * py[ell];
      const double qz = (1.0 - r) * pz[ell - 1] + r * pz[ell];
      if (qz < 0.0) cost += 1.0e5 + 1.0e5 * qz * qz;
      for (int b = 0; b < n_boxes; ++b) {
        const double sd = legacy_cuda_signed_distance_point_aabb(qx, qy, qz, boxes + 6 * b) - link_radius;
        if (sd < hard_margin) {
          cost += 1.0e6;
        } else if (sd < safe_margin) {
          const double e = (safe_margin - sd) / safe_margin;
          cost += 1.0e4 * e * e;
        }
      }
    }
  }
  return cost;
}

// -----------------------------------------------------------------------------
// Costs
// -----------------------------------------------------------------------------
__device__ __forceinline__ double
legacy_cuda_terminal_cost(const double *x, const double *x_target, int dim_x,
                          int model_type) {
  double s = 0.0;
  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    if (dim_x >= 12) {
      double q_cost = 0.0, v_cost = 0.0;
      for (int i = 0; i < 6; ++i) {
        const double dq = x[i] - x_target[i];
        const double dv = x[6 + i] - x_target[6 + i];
        q_cost += dq * dq;
        v_cost += dv * dv;
      }
      double px[7], py[7], pz[7], gx[7], gy[7], gz[7];
      legacy_cuda_manipulator_fk_points(x, px, py, pz);
      legacy_cuda_manipulator_fk_points(x_target, gx, gy, gz);
      const double ex = px[6] - gx[6];
      const double ey = py[6] - gy[6];
      const double ez = pz[6] - gz[6];
      const double ee = ex * ex + ey * ey + ez * ez;
      return 1500.0 * q_cost + 500.0 * ee + 100.0 * v_cost;
    }
    for (int d = 0; d < dim_x; ++d) { const double e = x[d] - x_target[d]; s += e * e; }
    return 1000.0 * s;
  }
  if (model_type == LEGACY_CUDA_QUADROTOR_PRECISION_LANDING) {
#ifdef BICMPPI_USE_QUADROTOR_LANDING_COST
    return quadrotor_landing_cost::terminalCost(x, x_target);
#else
    for (int d = 0; d < 3; ++d) { const double e = x[d] - x_target[d]; s += e * e; }
    return sqrt(s);
#endif
  }
  if (model_type == LEGACY_CUDA_QUADROTOR) {
    for (int d = 0; d < 3; ++d) { const double e = x[d] - x_target[d]; s += e * e; }
    return sqrt(s);
  }
  if (model_type == LEGACY_CUDA_VELO || model_type == LEGACY_CUDA_BICYCLE) {
    for (int d = 0; d < 2; ++d) { const double e = x[d] - x_target[d]; s += e * e; }
    return sqrt(s);
  }
  for (int d = 0; d < dim_x; ++d) { const double e = x[d] - x_target[d]; s += e * e; }
  return sqrt(s);
}

__device__ __forceinline__ double
legacy_cuda_stage_cost(const double *x, const double *u, int dim_x, int dim_u,
                       int model_type, const double *boxes = nullptr,
                       int n_boxes = 0, double link_radius = 0.045,
                       double safe_margin = 0.10, double hard_margin = 0.0) {
  if (model_type == LEGACY_CUDA_MANIPULATOR && dim_x >= 12 && dim_u >= 6) {
    double u2 = 0.0, v2 = 0.0, lim = 0.0;
    const double qmin[6] = {-2.0 * M_PI, -2.0 * M_PI, -165.0 * M_PI / 180.0, -2.0 * M_PI, -2.0 * M_PI, -2.0 * M_PI};
    const double qmax[6] = { 2.0 * M_PI,  2.0 * M_PI,  165.0 * M_PI / 180.0,  2.0 * M_PI,  2.0 * M_PI,  2.0 * M_PI};
    for (int i = 0; i < 6; ++i) {
      u2 += u[i] * u[i];
      v2 += x[6 + i] * x[6 + i];
      const double half = 0.5 * (qmax[i] - qmin[i]);
      const double center = 0.5 * (qmax[i] + qmin[i]);
      const double z = (x[i] - center) / (0.85 * half);
      const double excess = fabs(z) - 1.0;
      if (excess > 0.0) lim += excess * excess;
    }
    return 1.0e-3 * u2 + 2.0e-2 * v2 + 20.0 * lim +
           legacy_cuda_workspace_obstacle_cost(x, boxes, n_boxes, link_radius,
                                               safe_margin, hard_margin);
  }
  double u2 = 0.0;
  for (int i = 0; i < dim_u; ++i) u2 += u[i] * u[i];
  return 1.0e-3 * u2;
}

// -----------------------------------------------------------------------------
// Input projection
// -----------------------------------------------------------------------------
__device__ __forceinline__ void legacy_cuda_project_control(double *u, int dim_u,
                                                            int T, int model_type) {
  if (model_type == LEGACY_CUDA_MANIPULATOR) {
    const double tau_max[6] = {18.0, 18.0, 14.0, 8.0, 6.0, 4.0};
    for (int d = 0; d < dim_u && d < 6; ++d)
      for (int t = 0; t < T; ++t) u[d * T + t] = legacy_cuda_clamp(u[d * T + t], -tau_max[d], tau_max[d]);
    return;
  }
  if (model_type == LEGACY_CUDA_QUADROTOR || model_type == LEGACY_CUDA_QUADROTOR_PRECISION_LANDING) {
    for (int t = 0; t < T; ++t) {
      double u0 = u[0 * T + t], u1 = u[1 * T + t], u2 = u[2 * T + t];
      const double n = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
      if (n >= 20.0) { const double m = 20.0 / n; u0 *= m; u1 *= m; u2 *= m; }
      u[0 * T + t] = u0; u[1 * T + t] = u1; u[2 * T + t] = u2;
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
  if (model_type == LEGACY_CUDA_BICYCLE) {
    for (int t = 0; t < T; ++t) {
      u[0 * T + t] = legacy_cuda_clamp(u[0 * T + t], -3.0, 3.0);
      u[1 * T + t] = legacy_cuda_clamp(u[1 * T + t], -M_PI / 4.0, M_PI / 4.0);
    }
    return;
  }
  for (int t = 0; t < T; ++t) {
    u[0 * T + t] = legacy_cuda_clamp(u[0 * T + t], 0.0, 1.0);
    u[1 * T + t] = legacy_cuda_clamp(u[1 * T + t], -M_PI / 2.0, M_PI / 2.0);
  }
}

// -----------------------------------------------------------------------------
// Collision
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool
legacy_cuda_collision_grid_polygon(const double *x, const double *circles,
                                   int n_circles, const double *rects,
                                   int n_rects) {
  for (int i = 0; i < n_circles; ++i) {
    const double dx = circles[i * 4 + 0] - x[0];
    const double dy = circles[i * 4 + 1] - x[1];
    if (dx * dx + dy * dy <= circles[i * 4 + 3]) return true;
  }
  for (int i = 0; i < n_rects; ++i) {
    if (x[0] >= rects[i * 4 + 0] && x[0] <= rects[i * 4 + 1] &&
        x[1] >= rects[i * 4 + 2] && x[1] <= rects[i * 4 + 3]) return true;
  }
  return false;
}

__device__ __forceinline__ bool
legacy_cuda_collision_grid_map(const double *x, const double *d_map,
                               int max_row, int max_col, double resolution) {
  const int nx = static_cast<int>(round(x[0] / resolution));
  const int ny = static_cast<int>(round(x[1] / resolution));
  if (nx < 0 || max_row <= nx) return true;
  if (ny < 0 || max_col <= ny) return false;
  return d_map[nx * max_col + ny] == 10.0;
}

__device__ __forceinline__ bool
legacy_cuda_collision_grid(const double *x, bool with_map, const double *d_map,
                           int max_row, int max_col, double resolution,
                           const double *circles, int n_circles,
                           const double *rects, int n_rects,
                           const double *ws_boxes, int n_ws_boxes,
                           double link_radius, double safe_margin,
                           double hard_margin, int model_type) {
  if (with_map && legacy_cuda_collision_grid_map(x, d_map, max_row, max_col, resolution)) return true;
  if (legacy_cuda_collision_grid_polygon(x, circles, n_circles, rects, n_rects)) return true;
  if (model_type == LEGACY_CUDA_MANIPULATOR && n_ws_boxes > 0) {
    return legacy_cuda_manipulator_workspace_collision(x, ws_boxes, n_ws_boxes,
                                                       link_radius, hard_margin);
  }
  return false;
}

// Backward-compatible overload for old kernels. New BiMPPI kernels should call
// the extended signature above so workspace link collision is active on GPU.
__device__ __forceinline__ bool
legacy_cuda_collision_grid(const double *x, bool with_map, const double *d_map,
                           int max_row, int max_col, double resolution,
                           const double *circles, int n_circles,
                           const double *rects, int n_rects) {
  if (with_map) return legacy_cuda_collision_grid_map(x, d_map, max_row, max_col, resolution);
  return legacy_cuda_collision_grid_polygon(x, circles, n_circles, rects, n_rects);
}
