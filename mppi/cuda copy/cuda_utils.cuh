#pragma once

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>

// ============================================================
// CUDA Error Macros
// ============================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d  %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(_err));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(call)                                                     \
  do {                                                                         \
    curandStatus_t _err = (call);                                              \
    if (_err != CURAND_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuRAND error at %s:%d  code=%d\n", __FILE__, __LINE__,  \
              (int)_err);                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================
// WMRobot Device Functions (CUDA compiler only)
// State: x = [px, py, theta]  Control: u = [v, omega]
// Memory layout for controls: Ui[i * dim_u * T + d * T + t]
// ============================================================
#ifdef __CUDACC__

__device__ __forceinline__ void f_wmrobot(const double *x, double v,
                                          double omega, double *x_dot) {
  x_dot[0] = v * cos(x[2]);
  x_dot[1] = v * sin(x[2]);
  x_dot[2] = omega;
}

__device__ __forceinline__ double p_wmrobot(const double *x,
                                            const double *x_target, int dim_x) {
  double s = 0.0;
  for (int d = 0; d < dim_x; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

// Clamp a single (dim_u x T) block in-place
__device__ __forceinline__ void h_wmrobot(double *u, int T) {
  for (int t = 0; t < T; ++t) {
    u[0 * T + t] = fmax(0.0, fmin(1.0, u[0 * T + t]));
    u[1 * T + t] = fmax(-M_PI / 2.0, fmin(M_PI / 2.0, u[1 * T + t]));
  }
}

__device__ __forceinline__ void f_quadrotor(const double *x, const double *u,
                                            double *x_dot) {
  x_dot[0] = x[3];
  x_dot[1] = x[4];
  x_dot[2] = x[5];
  x_dot[3] = u[0];
  x_dot[4] = u[1];
  x_dot[5] = u[2] - 9.81;
}

__device__ __forceinline__ double
p_quadrotor(const double *x, const double *x_target, int dim_x) {
  double s = 0.0;
  for (int d = 0; d < 3; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

__device__ __forceinline__ void h_quadrotor(double *u, int T) {
  for (int t = 0; t < T; ++t) {
    double u0 = u[0 * T + t];
    double u1 = u[1 * T + t];
    double u2 = u[2 * T + t];

    double norm_u = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
    if (norm_u >= 20.0) {
      double mask = 1.0 / norm_u;
      u0 *= mask;
      u1 *= mask;
      u2 *= mask;
    }

    double V_val = sqrt(u0 * u0 + u1 * u1);
    double S_val = u2 * 1.73205080756887729; // tan(M_PI / 3.0)

    if (V_val < -S_val) {
      u0 = 0.0;
      u1 = 0.0;
      u2 = 0.0;
      V_val = 0.0;
      S_val = 0.0;
    }

    if (V_val > fabs(S_val)) {
      double mul = 0.5 * (1.0 + S_val / V_val);
      u0 = u0 * mul;
      u1 = u1 * mul;
      u2 = V_val * mul;
    }

    u[0 * T + t] = u0;
    u[1 * T + t] = u1;
    u[2 * T + t] = u2;
  }
}

__device__ __forceinline__ void f_velo(const double *x, const double *u,
                                       double *x_dot) {
  x_dot[0] = x[2];
  x_dot[1] = x[3];
  x_dot[2] = u[0];
  x_dot[3] = u[1];
}

__device__ __forceinline__ double p_velo(const double *x,
                                         const double *x_target, int dim_x) {
  double s = 0.0;
  for (int d = 0; d < 2; ++d) {
    double diff = x[d] - x_target[d];
    s += diff * diff;
  }
  return sqrt(s);
}

__device__ __forceinline__ void h_velo(double *u, int T) {
  for (int t = 0; t < T; ++t) {
    u[0 * T + t] = fmax(0.0, fmin(1.0, u[0 * T + t]));
    u[1 * T + t] = fmax(0.0, fmin(1.0, u[1 * T + t]));
  }
}

// ============================================================
// Manipulator Device Functions (CUDA compiler only)
// State: x = [q0..q5]  Control: u = [q_dot0..q_dot5]
// 1st-order kinematic: x_dot = u
// ============================================================
__device__ __forceinline__ void f_manipulator(const double *x, const double *u,
                                              double *x_dot, int dim_u) {
  for (int d = 0; d < dim_u; ++d)
    x_dot[d] = u[d];
}

__device__ __forceinline__ void compute_fk_positions_gpu(const double *q, double joint_pos[7][3]);

__device__ __forceinline__ double p_manipulator(const double *x,
                                                const double *x_target,
                                                int dim_x) {
  // 1) Joint space configuration error squared (weight = 1000.0)
  double config_err = 0.0;
  for (int d = 0; d < dim_x; ++d) {
    double diff = x[d] - x_target[d];
    config_err += diff * diff;
  }
  
  // 2) Task space end-effector position error squared (weight = 500.0)
  double joint_pos_curr[7][3];
  double joint_pos_tgt[7][3];
  compute_fk_positions_gpu(x, joint_pos_curr);
  compute_fk_positions_gpu(x_target, joint_pos_tgt);
  
  double dx = joint_pos_curr[6][0] - joint_pos_tgt[6][0];
  double dy = joint_pos_curr[6][1] - joint_pos_tgt[6][1];
  double dz = joint_pos_curr[6][2] - joint_pos_tgt[6][2];
  double task_err = dx * dx + dy * dy + dz * dz;

  // 3) J1 Direction Constraint: prevent backward sweep and overshoot
  double direction_cost = 0.0;
  if (x[0] < 0.0) {
    direction_cost = 50000.0 * x[0] * x[0];
  } else if (x[0] > 1.5707963267948966 + 0.3) {
    double excess = x[0] - (1.5707963267948966 + 0.3);
    direction_cost = 5000.0 * excess * excess;
  }

  return 1000.0 * config_err + 500.0 * task_err + direction_cost;
}


// q_dot_max: [1.5, 1.5, 1.5, 2.0, 2.0, 2.0] rad/s (RB5 limits)
__device__ __forceinline__ void h_manipulator(double *u, int dim_u, int T) {
  const double q_dot_max[6] = {1.5, 1.5, 1.5, 2.0, 2.0, 2.0};
  for (int d = 0; d < dim_u && d < 6; ++d)
    for (int t = 0; t < T; ++t)
      u[d * T + t] = fmax(-q_dot_max[d], fmin(q_dot_max[d], u[d * T + t]));
}

// Helper to check if a 3D point is inside a cylinder
__device__ __forceinline__ bool getCollisionCylinder3D_gpu(double px, double py, double pz,
                                                          const double *d_cylinders, int n_cylinders) {
  for (int i = 0; i < n_cylinders; ++i) {
    double cx = d_cylinders[i * 6 + 0];
    double cy = d_cylinders[i * 6 + 1];
    double r  = d_cylinders[i * 6 + 2];
    double z_min = d_cylinders[i * 6 + 4];
    double z_max = d_cylinders[i * 6 + 5];
    
    if (pz < z_min - 0.05 || pz > z_max + 0.05) continue;
    double dx = px - cx;
    double dy = py - cy;
    double death_r = r + 0.12; // Safety buffer
    if (dx * dx + dy * dy < death_r * death_r) {
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ void compute_fk_positions_gpu(const double *q, double joint_pos[7][3]) {
  const double L2 = 0.427;
  const double L3 = 0.357;
  const double dh_a[6] = {0.0, -L2, -L3, 0.0, 0.0, 0.0};
  const double dh_d[6] = {0.15, 0.0, 0.0, 0.11, 0.09, 0.09};
  const double dh_alpha[6] = {M_PI / 2.0, 0.0, 0.0, M_PI / 2.0, -M_PI / 2.0, 0.0};
  
  double T_prev[4][4];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      T_prev[r][c] = (r == c) ? 1.0 : 0.0;
    }
  }
  
  joint_pos[0][0] = 0.0;
  joint_pos[0][1] = 0.0;
  joint_pos[0][2] = 0.0;
  
  for (int i = 0; i < 6; ++i) {
    double theta = q[i];
    double a = dh_a[i];
    double d = dh_d[i];
    double alpha = dh_alpha[i];
    
    double c_theta = cos(theta);
    double s_theta = sin(theta);
    double c_alpha = cos(alpha);
    double s_alpha = sin(alpha);
    
    double T_local[4][4];
    T_local[0][0] = c_theta;
    T_local[0][1] = -s_theta * c_alpha;
    T_local[0][2] = s_theta * s_alpha;
    T_local[0][3] = a * c_theta;
    
    T_local[1][0] = s_theta;
    T_local[1][1] = c_theta * c_alpha;
    T_local[1][2] = -c_theta * s_alpha;
    T_local[1][3] = a * s_theta;
    
    T_local[2][0] = 0.0;
    T_local[2][1] = s_alpha;
    T_local[2][2] = c_alpha;
    T_local[2][3] = d;
    
    T_local[3][0] = 0.0;
    T_local[3][1] = 0.0;
    T_local[3][2] = 0.0;
    T_local[3][3] = 1.0;
    
    double T_global[4][4];
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) {
          sum += T_prev[r][k] * T_local[k][c];
        }
        T_global[r][c] = sum;
      }
    }
    
    joint_pos[i + 1][0] = T_global[0][3];
    joint_pos[i + 1][1] = T_global[1][3];
    joint_pos[i + 1][2] = T_global[2][3];
    
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        T_prev[r][c] = T_global[r][c];
      }
    }
  }
}

__device__ __forceinline__ bool check_manipulator_collision_gpu(const double *q,
                                                               const double *d_cylinders, int n_cylinders) {
  if (n_cylinders <= 0) return false;
  double joint_pos[7][3];
  compute_fk_positions_gpu(q, joint_pos);
  
  for (int k = 0; k < 6; ++k) {
    double p_start[3] = {joint_pos[k][0], joint_pos[k][1], joint_pos[k][2]};
    double p_end[3]   = {joint_pos[k+1][0], joint_pos[k+1][1], joint_pos[k+1][2]};
    
    const int num_samples = 15;
    for (int s = 0; s <= num_samples; ++s) {
      double alpha = (double)s / num_samples;
      double px = p_start[0] * (1.0 - alpha) + p_end[0] * alpha;
      double py = p_start[1] * (1.0 - alpha) + p_end[1] * alpha;
      double pz = p_start[2] * (1.0 - alpha) + p_end[2] * alpha;
      
      if (getCollisionCylinder3D_gpu(px, py, pz, d_cylinders, n_cylinders)) {
        return true;
      }
    }
  }
  return false;
}

__device__ __forceinline__ double cylinder_cost_gpu(double px, double py, double pz,
                                                    const double *d_cylinders, int n_cylinders) {
  double cost = 0.0;
  for (int i = 0; i < n_cylinders; ++i) {
    double cx = d_cylinders[i * 6 + 0];
    double cy = d_cylinders[i * 6 + 1];
    double r  = d_cylinders[i * 6 + 2];
    double z_min = d_cylinders[i * 6 + 4];
    double z_max = d_cylinders[i * 6 + 5];
    
    if (pz < z_min - 0.05 || pz > z_max + 0.05) continue;
    double dx = px - cx, dy = py - cy;
    double dist = sqrt(dx*dx + dy*dy);
    double death_r = r + 0.12; // Safety buffer
    double warn_r  = r + 0.30; // Wider gradient zone
    if (dist < death_r) {
      cost += 500000.0;
    } else if (dist < warn_r) {
      double excess = warn_r - dist;
      cost += 200000.0 * excess * excess;
      
      // Z-up helper cost
      double target_z = z_max + 0.15;
      if (pz < target_z) {
        cost += 1000.0 * (target_z - pz);
      }
    }
  }
  return cost;
}


__device__ __forceinline__ double compute_manipulator_obs_cost_gpu(const double joint_pos[7][3],
                                                                  const double *d_cylinders, int n_cylinders) {
  double obs_cost = 0.0;
  for (int k = 0; k < 6; ++k) {
    double p_start[3] = {joint_pos[k][0], joint_pos[k][1], joint_pos[k][2]};
    double p_end[3]   = {joint_pos[k+1][0], joint_pos[k+1][1], joint_pos[k+1][2]};
    
    const int num_samples = 15;
    for (int s = 0; s <= num_samples; ++s) {
      double alpha = (double)s / num_samples;
      double px = p_start[0] * (1.0 - alpha) + p_end[0] * alpha;
      double py = p_start[1] * (1.0 - alpha) + p_end[1] * alpha;
      double pz = p_start[2] * (1.0 - alpha) + p_end[2] * alpha;
      obs_cost += cylinder_cost_gpu(px, py, pz, d_cylinders, n_cylinders);
    }
  }
  return obs_cost;
}

__device__ __forceinline__ bool collision_grid_map(const double *x,
                                                   const double *d_map,
                                                   int max_row, int max_col,
                                                   double resolution) {
  int nx = (int)round(x[0] / resolution);
  int ny = (int)round(x[1] / resolution);
  if (nx < 0 || nx >= max_row)
    return true;
  if (ny < 0 || ny >= max_col)
    return false;
  return (d_map[nx * max_col + ny] == 10.0);
}

__device__ __forceinline__ bool
collision_polygon(const double *x, const double *circles, int n_circles,
                  const double *rects, int n_rects) {
  for (int i = 0; i < n_circles; ++i) {
    double dx = circles[i * 4 + 0] - x[0];
    double dy = circles[i * 4 + 1] - x[1];
    if (dx * dx + dy * dy <= circles[i * 4 + 3])
      return true;
  }
  for (int i = 0; i < n_rects; ++i) {
    if (x[0] >= rects[i * 4 + 0] && x[0] <= rects[i * 4 + 1] &&
        x[1] >= rects[i * 4 + 2] && x[1] <= rects[i * 4 + 3])
      return true;
  }
  return false;
}

__device__ __forceinline__ bool
check_collision(const double *x, bool with_map, const double *d_map,
                int max_row, int max_col, double resolution,
                const double *circles, int n_circles, const double *rects,
                int n_rects, int model_type) {
  if (model_type == 3) {
    return check_manipulator_collision_gpu(x, circles, n_circles);
  }
  if (with_map)
    return collision_grid_map(x, d_map, max_row, max_col, resolution);
  return collision_polygon(x, circles, n_circles, rects, n_rects);
}

__device__ __forceinline__ bool
check_collision(const double *x, bool with_map, const double *d_map,
                int max_row, int max_col, double resolution,
                const double *circles, int n_circles, const double *rects,
                int n_rects) {
  return check_collision(x, with_map, d_map, max_row, max_col, resolution,
                         circles, n_circles, rects, n_rects, 0);
}

// ============================================================
// GPU Reduction Utilities
// ============================================================

// Warp-level min reduction
__device__ __forceinline__ double warp_reduce_min(double val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

// Warp-level sum reduction
__device__ __forceinline__ double warp_reduce_sum(double val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// Shared forward/weighted-sum kernels are declared in shared_kernels.cuh
// and defined in shared_kernels.cu (single TU to avoid ODR violations).

// ============================================================
// Kernel body macros: each .cu defines its own local kernel
// using these macros to avoid ODR violations across TUs.
// ============================================================
#define DEFINE_FORWARD_ROLLOUT_KERNEL(KERNEL_NAME)                             \
  __global__ void KERNEL_NAME(                                                 \
      const double *__restrict__ d_U0, double *d_Ui,                           \
      const double *__restrict__ d_noise, const double *__restrict__ d_sigma,  \
      const double *__restrict__ d_x_init,                                     \
      const double *__restrict__ d_x_target, double *d_costs, double *d_Di,    \
      bool with_map, const double *d_map, int max_row, int max_col,            \
      double res, const double *d_circles, int n_circ, const double *d_rects,  \
      int n_rect, int N, int dim_u, int dim_x, int T, double dt,               \
      double gamma_u, int model_type) {                                        \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= N)                                                                \
      return;                                                                  \
    double *Ui_i = d_Ui + i * (dim_u * T);                                     \
    for (int d = 0; d < dim_u; ++d)                                            \
      for (int t = 0; t < T; ++t)                                              \
        Ui_i[d * T + t] = d_U0[d * T + t] +                                    \
                          d_sigma[d] * d_noise[i * (dim_u * T) + d * T + t];   \
    if (model_type == 0) {                                                     \
      h_wmrobot(Ui_i, T);                                                      \
    } else if (model_type == 1) {                                              \
      h_quadrotor(Ui_i, T);                                                    \
    } else if (model_type == 2) {                                              \
      h_velo(Ui_i, T);                                                         \
    } else if (model_type == 3) {                                              \
      h_manipulator(Ui_i, dim_u, T);                                           \
    }                                                                          \
    double *Di_i = d_Di + i * dim_u;                                           \
    for (int d = 0; d < dim_u; ++d) {                                          \
      double acc = 0.0;                                                        \
      for (int t = 0; t < T; ++t)                                              \
        acc += Ui_i[d * T + t] - d_U0[d * T + t];                              \
      Di_i[d] = acc / T;                                                       \
    }                                                                          \
    double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];            \
    for (int d = 0; d < dim_x; ++d)                                            \
      x[d] = d_x_init[d];                                                      \
    double cost = 0.0;                                                         \
    bool hit = false;                                                          \
    for (int t = 0; t < T; ++t) {                                              \
      if (model_type == 0) {                                                   \
        cost += p_wmrobot(x, d_x_target, dim_x);                               \
        double v = Ui_i[0 * T + t], omega = Ui_i[1 * T + t];                   \
        f_wmrobot(x, v, omega, xd_);                                           \
      } else if (model_type == 1) {                                            \
        cost += p_quadrotor(x, d_x_target, dim_x);                             \
        double u_val[3] = {Ui_i[0 * T + t], Ui_i[1 * T + t], Ui_i[2 * T + t]}; \
        f_quadrotor(x, u_val, xd_);                                            \
      } else if (model_type == 2) {                                            \
        cost += p_velo(x, d_x_target, dim_x);                                  \
        double u_val2[2] = {Ui_i[0 * T + t], Ui_i[1 * T + t]};                 \
        f_velo(x, u_val2, xd_);                                                \
      } else if (model_type == 3) {                                            \
        cost += p_manipulator(x, d_x_target, dim_x);                           \
        double joint_pos[7][3];                                                \
        compute_fk_positions_gpu(x, joint_pos);                                \
        cost += compute_manipulator_obs_cost_gpu(joint_pos, d_circles, n_circ); \
        const double q_min[6] = {-2.0 * M_PI, -2.0 * M_PI, -165.0 * M_PI / 180.0, -2.0 * M_PI, -2.0 * M_PI, -2.0 * M_PI}; \
        const double q_max[6] = {2.0 * M_PI, 2.0 * M_PI, 165.0 * M_PI / 180.0, 2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI}; \
        for (int j = 0; j < 6; ++j) {                                          \
          double qr = (q_max[j] - q_min[j]) * 0.5;                             \
          double qc = (q_max[j] + q_min[j]) * 0.5;                             \
          double norm = (x[j] - qc) / (qr * 0.8);                              \
          if (fabs(norm) > 1.0)                                                \
            cost += 5.0 * (fabs(norm) - 1.0) * (fabs(norm) - 1.0);             \
        }                                                                      \
        double u_manip[GPU_MAX_DIM_U];                                          \
        for (int _d = 0; _d < dim_u; ++_d)                                     \
          u_manip[_d] = Ui_i[_d * T + t];                                      \
        f_manipulator(x, u_manip, xd_, dim_u);                                 \
      }                                                                        \
      for (int d = 0; d < dim_x; ++d)                                          \
        xn[d] = x[d] + dt * xd_[d];                                            \
      if (!hit && check_collision(x, with_map, d_map, max_row, max_col, res,   \
                                  d_circles, n_circ, d_rects, n_rect, model_type)) { \
        hit = true;                                                            \
        cost = 1e8;                                                            \
      }                                                                        \
      for (int d = 0; d < dim_x; ++d)                                          \
        x[d] = xn[d];                                                          \
    }                                                                          \
    if (!hit) {                                                                \
      if (model_type == 0) {                                                   \
        cost += p_wmrobot(x, d_x_target, dim_x);                               \
      } else if (model_type == 1) {                                            \
        cost += p_quadrotor(x, d_x_target, dim_x);                             \
      } else if (model_type == 2) {                                            \
        cost += p_velo(x, d_x_target, dim_x);                                  \
      } else if (model_type == 3) {                                            \
        cost += p_manipulator(x, d_x_target, dim_x);                           \
      }                                                                        \
      if (check_collision(x, with_map, d_map, max_row, max_col, res,           \
                          d_circles, n_circ, d_rects, n_rect, model_type))     \
        cost = 1e8;                                                            \
    }                                                                          \
    d_costs[i] = cost;                                                         \
  }

#define DEFINE_WEIGHTED_SUM_KERNEL(KERNEL_NAME)                                \
  __global__ void KERNEL_NAME(const double *d_Ui, const double *d_costs,       \
                              double min_cost, double gamma_u, double *d_Uo,   \
                              int N, int dim_u, int T) {                       \
    int dt_idx = blockIdx.x;                                                   \
    int d = dt_idx / T;                                                        \
    int t = dt_idx % T;                                                        \
    if (d >= dim_u)                                                            \
      return;                                                                  \
    extern __shared__ double sdata[];                                          \
    double *s_wsum = sdata;                                                    \
    double *s_wctrl = sdata + blockDim.x;                                      \
    int tid = threadIdx.x;                                                     \
    double w_acc = 0.0, wc_acc = 0.0;                                          \
    for (int i = tid; i < N; i += blockDim.x) {                                \
      double w = exp(-gamma_u * (d_costs[i] - min_cost));                      \
      w_acc += w;                                                              \
      wc_acc += w * d_Ui[i * (dim_u * T) + d * T + t];                         \
    }                                                                          \
    s_wsum[tid] = w_acc;                                                       \
    s_wctrl[tid] = wc_acc;                                                     \
    __syncthreads();                                                           \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                             \
      if (tid < s) {                                                           \
        s_wsum[tid] += s_wsum[tid + s];                                        \
        s_wctrl[tid] += s_wctrl[tid + s];                                      \
      }                                                                        \
      __syncthreads();                                                         \
    }                                                                          \
    if (tid == 0)                                                              \
      d_Uo[dt_idx] = s_wctrl[0] / s_wsum[0];                                   \
  }

#endif // __CUDACC__
