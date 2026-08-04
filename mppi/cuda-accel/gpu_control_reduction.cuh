#pragma once

#include "cuda_utils.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>

#include <cstddef>
#include <vector>

namespace cuda_accel {
namespace detail {

struct WeightedControlCache {
  thrust::device_vector<double> weights;
  thrust::device_vector<double> weight_sum;
  thrust::device_vector<double> output;
};

static __global__ void computeWeights(
    const double *costs, const double *minimum_cost, int sample_count,
    double gamma, double *weights, double *weight_sum) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count) return;
  const double weight = exp(-gamma * (costs[sample] - *minimum_cost));
  weights[sample] = weight;
  atomicAdd(weight_sum, weight);
}

static __global__ void accumulateWeightedControls(
    const double *controls, const double *weights, int sample_count,
    int control_size, double *output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= sample_count * control_size) return;
  const int sample = index / control_size;
  const int control_index = index % control_size;
  atomicAdd(output + control_index,
            weights[sample] * controls[sample * control_size + control_index]);
}

static __global__ void normalizeWeightedControls(
    const double *weight_sum, int control_size, double *output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= control_size) return;
  const double denominator = *weight_sum;
  output[index] = denominator > 0.0 ? output[index] / denominator : 0.0;
}

}  // namespace detail

// Reduces N sampled control sequences to one MPPI weighted control sequence
// without copying N costs or N control sequences to the host.
inline void reduceWeightedControlsDevice(
    const double *device_controls, const double *device_costs,
    int sample_count, int dim_u, int horizon, double gamma,
    std::vector<double> &host_control) {
  host_control.clear();
  if (sample_count <= 0 || dim_u <= 0 || horizon <= 0) return;

  static thread_local detail::WeightedControlCache cache;
  const int control_size = dim_u * horizon;
  cache.weights.resize(sample_count);
  cache.weight_sum.resize(1);
  cache.output.resize(control_size);
  thrust::fill(cache.weight_sum.begin(), cache.weight_sum.end(), 0.0);
  thrust::fill(cache.output.begin(), cache.output.end(), 0.0);

  const auto costs = thrust::device_pointer_cast(device_costs);
  const auto minimum = thrust::min_element(costs, costs + sample_count);
  const double *minimum_cost = thrust::raw_pointer_cast(minimum);

  constexpr int block = 256;
  detail::computeWeights<<<(sample_count + block - 1) / block, block>>>(
      device_costs, minimum_cost, sample_count, gamma,
      thrust::raw_pointer_cast(cache.weights.data()),
      thrust::raw_pointer_cast(cache.weight_sum.data()));

  const int accumulation_count = sample_count * control_size;
  detail::accumulateWeightedControls<<<
      (accumulation_count + block - 1) / block, block>>>(
      device_controls, thrust::raw_pointer_cast(cache.weights.data()),
      sample_count, control_size,
      thrust::raw_pointer_cast(cache.output.data()));

  detail::normalizeWeightedControls<<<
      (control_size + block - 1) / block, block>>>(
      thrust::raw_pointer_cast(cache.weight_sum.data()), control_size,
      thrust::raw_pointer_cast(cache.output.data()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  host_control.resize(control_size);
  CUDA_CHECK(cudaMemcpy(
      host_control.data(), thrust::raw_pointer_cast(cache.output.data()),
      static_cast<std::size_t>(control_size) * sizeof(double),
      cudaMemcpyDeviceToHost));
}

}  // namespace cuda_accel
