#pragma once

#include "cuda_legacy_model.cuh"
#include "cuda_utils.cuh"

// =============================================================================
// Workspace-aware forward rollout kernel macro.
// -----------------------------------------------------------------------------
// This is a drop-in replacement for the forward rollout macro used by
// bi_mppi_gpu.cu. It keeps the original MPPI sampling/projection layout but
// extends collision and stage cost calls with workspace link-collision buffers.
// =============================================================================
#define DEFINE_FORWARD_ROLLOUT_KERNEL(NAME)                                                \
__global__ void NAME(                                                                       \
    const double* __restrict__ d_U0,                                                        \
    double* d_Ui,                                                                           \
    const double* __restrict__ d_noise,                                                     \
    const double* __restrict__ d_sigma,                                                     \
    const double* __restrict__ d_x_init,                                                    \
    const double* __restrict__ d_x_target,                                                  \
    double* d_costs, double* d_Di,                                                          \
    bool with_map, const double* d_map,                                                     \
    int max_row, int max_col, double res,                                                   \
    const double* d_circles, int n_circ,                                                    \
    const double* d_rects, int n_rect,                                                      \
    const double* d_ws_boxes, int n_ws_boxes,                                               \
    double ws_link_radius, double ws_safe_margin, double ws_hard_margin,                    \
    int N, int dim_u, int dim_x, int T, double dt, double gamma_u,                          \
    int model_type, bool forward) {                                                         \
  (void)gamma_u;                                                                            \
  (void)forward;                                                                            \
  const int i = blockIdx.x * blockDim.x + threadIdx.x;                                      \
  if (i >= N) return;                                                                       \
  double* Ui_i = d_Ui + i * (dim_u * T);                                                    \
  for (int d = 0; d < dim_u; ++d) {                                                         \
    for (int t = 0; t < T; ++t) {                                                           \
      const double raw = d_U0[d * T + t] +                                                  \
                         d_sigma[d] * d_noise[i * (dim_u * T) + d * T + t];                 \
      Ui_i[d * T + t] = raw;                                                                \
    }                                                                                       \
  }                                                                                         \
  legacy_cuda_project_control(Ui_i, dim_u, T, model_type);                                  \
  double* Di_i = d_Di + i * dim_u;                                                          \
  for (int d = 0; d < dim_u; ++d) {                                                         \
    double acc = 0.0;                                                                       \
    for (int t = 0; t < T; ++t) acc += Ui_i[d * T + t] - d_U0[d * T + t];                    \
    Di_i[d] = acc / static_cast<double>(T);                                                  \
  }                                                                                         \
  double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd[GPU_MAX_DIM_X];                            \
  for (int d = 0; d < dim_x; ++d) x[d] = d_x_init[d];                                       \
  double cost = 0.0;                                                                        \
  bool hit = legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,          \
                                        d_circles, n_circ, d_rects, n_rect,                 \
                                        d_ws_boxes, n_ws_boxes, ws_link_radius,             \
                                        ws_safe_margin, ws_hard_margin, model_type);        \
  if (hit) cost = 1.0e8;                                                                    \
  for (int t = 0; t < T && !hit; ++t) {                                                     \
    double u_local[GPU_MAX_DIM_U];                                                          \
    for (int d = 0; d < dim_u; ++d) u_local[d] = Ui_i[d * T + t];                            \
    cost += legacy_cuda_stage_cost(x, u_local, dim_x, dim_u, model_type,                    \
                                   d_ws_boxes, n_ws_boxes, ws_link_radius,                  \
                                   ws_safe_margin, ws_hard_margin);                         \
    cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);                    \
    legacy_cuda_dynamics(x, u_local, xd, dim_x, dim_u, model_type);                         \
    for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt * xd[d];                              \
    for (int d = 0; d < dim_x; ++d) x[d] = xn[d];                                           \
    hit = legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,             \
                                     d_circles, n_circ, d_rects, n_rect,                    \
                                     d_ws_boxes, n_ws_boxes, ws_link_radius,                \
                                     ws_safe_margin, ws_hard_margin, model_type);           \
    if (hit) cost = 1.0e8;                                                                  \
  }                                                                                         \
  if (!hit) cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);            \
  d_costs[i] = cost;                                                                        \
}
