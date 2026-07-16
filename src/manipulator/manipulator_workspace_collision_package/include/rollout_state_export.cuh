#pragma once

#include "cuda_utils.cuh"
#include "rollout_state_callback.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rollout_state_export {

__device__ __forceinline__ void saveState(const double *x,
                                          std::uint16_t *states,
                                          std::size_t base, int dim_x) {
  for (int d = 0; d < dim_x; ++d) {
    states[base + d] =
        __half_as_ushort(__float2half_rn(static_cast<float>(x[d])));
  }
}

static __global__ void exportKernel(
    const double *__restrict__ controls, const double *__restrict__ anchor,
    std::uint16_t *states, int rollout_count, int dim_u, int dim_x,
    int horizon, double dt, int model_type, bool backward) {
  const int rollout = blockIdx.x * blockDim.x + threadIdx.x;
  if (rollout >= rollout_count) return;

  const double *u_rollout = controls + rollout * dim_u * horizon;
  double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xdot[GPU_MAX_DIM_X];
  for (int d = 0; d < dim_x; ++d) x[d] = anchor[d];

  auto output_base = [&](int time_index) {
    return (static_cast<std::size_t>(rollout) * (horizon + 1) + time_index) *
           dim_x;
  };

  if (!backward) {
    saveState(x, states, output_base(0), dim_x);
    for (int t = 0; t < horizon; ++t) {
      double u[GPU_MAX_DIM_U];
      for (int d = 0; d < dim_u; ++d) u[d] = u_rollout[d * horizon + t];
      legacy_cuda_dynamics(x, u, xdot, dim_x, dim_u, model_type);
      for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt * xdot[d];
      for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
      saveState(x, states, output_base(t + 1), dim_x);
    }
    return;
  }

  saveState(x, states, output_base(horizon), dim_x);
  for (int t = horizon - 1; t >= 0; --t) {
    const int control_time = (t == horizon - 1) ? t : t + 1;
    double u[GPU_MAX_DIM_U];
    for (int d = 0; d < dim_u; ++d)
      u[d] = u_rollout[d * horizon + control_time];
    legacy_cuda_dynamics(x, u, xdot, dim_x, dim_u, model_type);
    for (int d = 0; d < dim_x; ++d) xn[d] = x[d] - dt * xdot[d];
    for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
    saveState(x, states, output_base(t), dim_x);
  }
}

inline void emit(const RolloutStateBatchCallback &callback,
                 const std::string &branch, const double *device_controls,
                 const double *device_anchor, int rollout_count, int dim_u,
                 int dim_x, int horizon, double dt, int model_type,
                 bool backward = false) {
  if (!callback || rollout_count <= 0 || horizon <= 0) return;

  const std::size_t value_count = static_cast<std::size_t>(rollout_count) *
                                  (horizon + 1) * dim_x;
  std::uint16_t *device_states = nullptr;
  CUDA_CHECK(cudaMalloc(&device_states,
                        value_count * sizeof(std::uint16_t)));
  const int block = 256;
  exportKernel<<<(rollout_count + block - 1) / block, block>>>(
      device_controls, device_anchor, device_states, rollout_count, dim_u,
      dim_x, horizon, dt, model_type, backward);
  CUDA_CHECK(cudaGetLastError());

  std::vector<std::uint16_t> host_states(value_count);
  CUDA_CHECK(cudaMemcpy(host_states.data(), device_states,
                        value_count * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_states));
  callback(branch, host_states, rollout_count, horizon + 1, dim_x);
}

}  // namespace rollout_state_export
