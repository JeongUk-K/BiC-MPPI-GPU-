#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>

#include "cuda_legacy_model.cuh"

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
      fprintf(stderr, "cuRAND error at %s:%d  code=%d (%s)\n", __FILE__, __LINE__,  \
              (int)_err, #call);                                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#ifdef __CUDACC__

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
      double gamma_u, int model_type, bool check_initial_collision) {          \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= N)                                                                \
      return;                                                                  \
    double *Ui_i = d_Ui + i * (dim_u * T);                                     \
    for (int d = 0; d < dim_u; ++d)                                            \
      for (int t = 0; t < T; ++t)                                              \
        Ui_i[d * T + t] = d_U0[d * T + t] +                                    \
                          d_sigma[d] * d_noise[i * (dim_u * T) + d * T + t];   \
    legacy_cuda_project_control(Ui_i, dim_u, T, model_type);                   \
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
    if (check_initial_collision &&                                             \
        legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,  \
                                  d_circles, n_circ, d_rects, n_rect)) {       \
      hit = true;                                                              \
      cost = 1e8;                                                              \
    }                                                                          \
    for (int t = 0; t < T; ++t) {                                              \
      if (hit)                                                                 \
        break;                                                                 \
      cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);     \
      double u_local[GPU_MAX_DIM_U];                                           \
      for (int _d = 0; _d < dim_u; ++_d)                                       \
        u_local[_d] = Ui_i[_d * T + t];                                        \
      legacy_cuda_dynamics(x, u_local, xd_, dim_x, dim_u, model_type);         \
      for (int d = 0; d < dim_x; ++d)                                          \
        xn[d] = x[d] + dt * xd_[d];                                            \
      for (int d = 0; d < dim_x; ++d)                                          \
        x[d] = xn[d];                                                          \
      if (legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col,     \
                                     res, d_circles, n_circ, d_rects,          \
                                     n_rect)) {                                \
        hit = true;                                                            \
        cost = 1e8;                                                            \
      }                                                                        \
    }                                                                          \
    if (!hit) {                                                                \
      cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);     \
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

#endif // CUDA_UTILS_CUH
