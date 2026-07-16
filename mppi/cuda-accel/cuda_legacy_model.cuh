#pragma once

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
    for (int d = 0; d < dim_x && d < dim_u; ++d) {
      x_dot[d] = u[d];
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
    const double q_dot_max[6] = {1.5, 1.5, 1.5, 2.0, 2.0, 2.0};
    for (int d = 0; d < dim_u && d < 6; ++d) {
      for (int t = 0; t < T; ++t) {
        u[d * T + t] =
            legacy_cuda_clamp(u[d * T + t], -q_dot_max[d], q_dot_max[d]);
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
                           const double *rects, int n_rects) {
  if (with_map) {
    return legacy_cuda_collision_grid_map(x, d_map, max_row, max_col,
                                          resolution);
  }
  return legacy_cuda_collision_grid_polygon(x, circles, n_circles, rects,
                                            n_rects);
}
