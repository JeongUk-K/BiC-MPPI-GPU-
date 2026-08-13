#include "log_mppi_gpu.cuh"
#include "log_mppi_weights.h"

namespace {
constexpr double kCollisionCost = 1.0e8;

__global__ void log_mppi_logsumexp_weighted_sum_kernel(
    const double *controls, const double *costs, double min_valid_cost,
    double gamma_u, double *output, int samples, int dim_u, int horizon) {
  const int control_index = blockIdx.x;
  const int control_size = dim_u * horizon;
  extern __shared__ double partial[];
  double weight_sum = 0.0;
  double weighted_control_sum = 0.0;

  // Subtracting the best valid cost is the log-sum-exp shift:
  // exp(-gamma * J_i - max_j(-gamma * J_j)).
  for (int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const double cost = costs[sample];
    if (!isfinite(cost) || cost >= kCollisionCost) continue;

    const double log_weight = -gamma_u * (cost - min_valid_cost);
    if (!isfinite(log_weight)) continue;
    const double weight = exp(log_weight);
    if (!isfinite(weight)) continue;

    weight_sum += weight;
    weighted_control_sum +=
        weight * controls[sample * control_size + control_index];
  }

  partial[threadIdx.x] = weight_sum;
  partial[blockDim.x + threadIdx.x] = weighted_control_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
      partial[blockDim.x + threadIdx.x] +=
          partial[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double total_weight = partial[0];
    output[control_index] =
        (total_weight > 0.0 && isfinite(total_weight))
            ? partial[blockDim.x] / total_weight
            : 0.0;
  }
}

int reductionThreads(int samples) {
  int threads = 1;
  while (threads < 256 && threads * 2 <= samples) threads *= 2;
  return threads;
}
}  // namespace

void LogMPPI_GPU::weightedControlSum(const std::vector<double> &costs,
                                     Eigen::MatrixXd &Uo_out) {
  const auto copyFallback = [&]() {
    int fallback = leastFiniteCostIndex(costs);
    if (fallback < 0 || fallback >= N) fallback = 0;
    const std::size_t control_size = static_cast<std::size_t>(dim_u) * T;
    CUDA_CHECK(cudaMemcpy(d_Uo, d_Ui + static_cast<std::size_t>(fallback) *
                                      control_size,
                          control_size * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  };

  bool has_valid_cost = false;
  double min_valid_cost = std::numeric_limits<double>::infinity();
  if (costs.size() == static_cast<std::size_t>(N) &&
      std::isfinite(gamma_u) && gamma_u >= 0.0) {
    for (double cost : costs) {
      if (std::isfinite(cost) && cost < kCollisionCost) {
        has_valid_cost = true;
        min_valid_cost = std::min(min_valid_cost, cost);
      }
    }
  }

  if (!has_valid_cost) {
    copyFallback();
  } else {
    const int control_blocks = dim_u * T;
    const int threads = reductionThreads(N);
    const size_t shared_bytes = 2 * static_cast<size_t>(threads) *
                                 sizeof(double);
    log_mppi_logsumexp_weighted_sum_kernel<<<control_blocks, threads,
                                             shared_bytes>>>(
        d_Ui, d_costs, min_valid_cost, gamma_u, d_Uo, N, dim_u, T);
    CUDA_CHECK(cudaGetLastError());
  }

  std::vector<double> host_output(static_cast<std::size_t>(dim_u) * T);
  CUDA_CHECK(cudaMemcpy(host_output.data(), d_Uo,
                        host_output.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  Uo_out.resize(dim_u, T);
  for (int d = 0; d < dim_u; ++d)
    for (int t = 0; t < T; ++t)
      Uo_out(d, t) = host_output[d * T + t];
}
