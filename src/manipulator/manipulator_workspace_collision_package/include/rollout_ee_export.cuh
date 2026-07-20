#pragma once

#include "cuda_utils.cuh"
#include "rollout_ee_callback.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rollout_ee_export {

#ifndef ROLLOUT_EE_COORDINATE_SCALE
#define ROLLOUT_EE_COORDINATE_SCALE 1.0e-3
#endif
// The offline 0.02 s archive keeps its 1 mm scale. Online controllers with a
// larger discretization may override this at compile time so an int8 delta can
// represent the larger EE displacement between adjacent samples.
constexpr double kCoordinateScale = ROLLOUT_EE_COORDINATE_SCALE;

__device__ __forceinline__ std::int16_t quantizeCoordinate(double value) {
  double scaled = nearbyint(value / kCoordinateScale);
  scaled = fmax(-32768.0, fmin(32767.0, scaled));
  return static_cast<std::int16_t>(scaled);
}

// Reconstruct sampled trajectories directly on the GPU and retain only the EE
// positions. This avoids copying all sampled controls to the CPU and repeating
// manipulator dynamics/FK there.
static __global__ void exportKernel(
    const double *__restrict__ controls, const double *__restrict__ anchor,
    std::int16_t *positions, int rollout_count, int dim_u, int dim_x,
    int horizon, double dt, int model_type, bool backward) {
  const int rollout = blockIdx.x * blockDim.x + threadIdx.x;
  if (rollout >= rollout_count) return;

  const double *u_rollout = controls + rollout * dim_u * horizon;
  double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xdot[GPU_MAX_DIM_X];
  for (int d = 0; d < dim_x; ++d) x[d] = anchor[d];

  auto save_ee = [&](int time_index) {
    double px[7], py[7], pz[7];
    legacy_cuda_manipulator_fk_points(x, px, py, pz);
    const std::size_t base =
        (static_cast<std::size_t>(rollout) * (horizon + 1) + time_index) * 3;
    positions[base] = quantizeCoordinate(px[6]);
    positions[base + 1] = quantizeCoordinate(py[6]);
    positions[base + 2] = quantizeCoordinate(pz[6]);
  };

  if (!backward) {
    save_ee(0);
    for (int t = 0; t < horizon; ++t) {
      double u[GPU_MAX_DIM_U];
      for (int d = 0; d < dim_u; ++d) u[d] = u_rollout[d * horizon + t];
      legacy_cuda_dynamics(x, u, xdot, dim_x, dim_u, model_type);
      for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt * xdot[d];
      for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
      save_ee(t + 1);
    }
    return;
  }

  save_ee(horizon);
  for (int t = horizon - 1; t >= 0; --t) {
    const int control_time = (t == horizon - 1) ? t : t + 1;
    double u[GPU_MAX_DIM_U];
    for (int d = 0; d < dim_u; ++d)
      u[d] = u_rollout[d * horizon + control_time];
    legacy_cuda_dynamics(x, u, xdot, dim_x, dim_u, model_type);
    for (int d = 0; d < dim_x; ++d) xn[d] = x[d] - dt * xdot[d];
    for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
    save_ee(t);
  }
}

static __global__ void deltaEncodeKernel(std::int16_t *positions,
                                         int rollout_count, int point_count) {
  const int rollout = blockIdx.x * blockDim.x + threadIdx.x;
  if (rollout >= rollout_count) return;
  const std::size_t rollout_base =
      static_cast<std::size_t>(rollout) * point_count * 3;
  // Descending order preserves the preceding absolute sample in-place.
  for (int t = point_count - 1; t >= 1; --t) {
    for (int axis = 0; axis < 3; ++axis) {
      const std::size_t current = rollout_base + t * 3 + axis;
      const std::size_t previous = current - 3;
      positions[current] = static_cast<std::int16_t>(
          static_cast<int>(positions[current]) -
          static_cast<int>(positions[previous]));
    }
  }
}

static __global__ void packDeltaKernel(const std::int16_t *positions,
                                       std::uint8_t *packed,
                                       unsigned int *overflow_count,
                                       int rollout_count, int point_count) {
  const int rollout = blockIdx.x * blockDim.x + threadIdx.x;
  if (rollout >= rollout_count) return;
  const std::size_t source_base =
      static_cast<std::size_t>(rollout) * point_count * 3;
  const std::size_t record_size = 6 + (point_count - 1) * 3;
  const std::size_t output_base = static_cast<std::size_t>(rollout) * record_size;
  for (int axis = 0; axis < 3; ++axis) {
    const std::uint16_t value =
        static_cast<std::uint16_t>(positions[source_base + axis]);
    packed[output_base + 2 * axis] = static_cast<std::uint8_t>(value & 0xffu);
    packed[output_base + 2 * axis + 1] =
        static_cast<std::uint8_t>((value >> 8) & 0xffu);
  }
  for (int t = 1; t < point_count; ++t) {
    for (int axis = 0; axis < 3; ++axis) {
      int delta = static_cast<int>(positions[source_base + t * 3 + axis]);
      if (delta < -128 || delta > 127) atomicAdd(overflow_count, 1u);
      delta = max(-128, min(127, delta));
      packed[output_base + 6 + (t - 1) * 3 + axis] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(delta));
    }
  }
}

inline void emit(const RolloutEEBatchCallback &callback,
                 const std::string &branch, const double *device_controls,
                 const double *device_anchor, int rollout_count, int dim_u,
                 int dim_x, int horizon, double dt, int model_type,
                 bool backward = false) {
  if (!callback || rollout_count <= 0 || horizon <= 0) return;

  const int point_count = horizon + 1;
  const std::size_t value_count =
      static_cast<std::size_t>(rollout_count) * point_count * 3;
  std::int16_t *device_positions = nullptr;
  CUDA_CHECK(cudaMalloc(&device_positions, value_count * sizeof(std::int16_t)));
  const int block = 256;
  exportKernel<<<(rollout_count + block - 1) / block, block>>>(
      device_controls, device_anchor, device_positions, rollout_count, dim_u,
      dim_x, horizon, dt, model_type, backward);
  CUDA_CHECK(cudaGetLastError());
  deltaEncodeKernel<<<(rollout_count + block - 1) / block, block>>>(
      device_positions, rollout_count, point_count);
  CUDA_CHECK(cudaGetLastError());

  const std::size_t packed_size = static_cast<std::size_t>(rollout_count) *
                                  (6 + horizon * 3);
  std::uint8_t *device_packed = nullptr;
  unsigned int *device_overflow_count = nullptr;
  CUDA_CHECK(cudaMalloc(&device_packed, packed_size));
  CUDA_CHECK(cudaMalloc(&device_overflow_count, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(device_overflow_count, 0, sizeof(unsigned int)));
  packDeltaKernel<<<(rollout_count + block - 1) / block, block>>>(
      device_positions, device_packed, device_overflow_count, rollout_count,
      point_count);
  CUDA_CHECK(cudaGetLastError());
  unsigned int overflow_count = 0;
  CUDA_CHECK(cudaMemcpy(&overflow_count, device_overflow_count,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_overflow_count));
  if (overflow_count != 0) {
#ifdef ROLLOUT_EE_CLAMP_OVERFLOW
    static bool overflow_warning_emitted = false;
    if (!overflow_warning_emitted) {
      fprintf(stderr,
              "rollout EE delta overflow: clamping out-of-range visualization "
              "values at scale %.7f (warning shown once)\n",
              kCoordinateScale);
      overflow_warning_emitted = true;
    }
#else
    fprintf(stderr,
            "rollout EE delta overflow: %u values exceed int8 at scale %.7f\n",
            overflow_count, kCoordinateScale);
    exit(EXIT_FAILURE);
#endif
  }
  std::vector<std::uint8_t> host_positions(packed_size);
  CUDA_CHECK(cudaMemcpy(host_positions.data(), device_packed, packed_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_packed));
  CUDA_CHECK(cudaFree(device_positions));
  callback(branch, host_positions, rollout_count, point_count);
}

}  // namespace rollout_ee_export
