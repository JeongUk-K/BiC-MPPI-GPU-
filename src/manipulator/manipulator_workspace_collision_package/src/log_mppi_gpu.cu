#include "log_mppi_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

DEFINE_FORWARD_ROLLOUT_KERNEL(log_mppi_rollout_kernel)

__global__ void log_mppi_logsumexp_weighted_sum_kernel(
    const double *d_Ui, const double *d_costs, double min_valid_cost,
    double gamma_u, double collision_cost, double *d_Uo, int N, int dim_u,
    int T) {
  const int dt_idx = blockIdx.x;
  const int d = dt_idx / T;
  const int t = dt_idx % T;
  if (d >= dim_u) {
    return;
  }

  extern __shared__ double sdata[];
  double *s_wsum = sdata;
  double *s_wctrl = sdata + blockDim.x;

  const int tid = threadIdx.x;
  double w_acc = 0.0;
  double wc_acc = 0.0;
  const double log_w_max = -gamma_u * min_valid_cost;

  for (int i = tid; i < N; i += blockDim.x) {
    const double cost = d_costs[i];
    if (isfinite(cost) && cost < collision_cost) {
      const double log_w = -gamma_u * cost;
      const double w = exp(log_w - log_w_max);
      w_acc += w;
      wc_acc += w * d_Ui[i * (dim_u * T) + d * T + t];
    }
  }

  s_wsum[tid] = w_acc;
  s_wctrl[tid] = wc_acc;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_wsum[tid] += s_wsum[tid + s];
      s_wctrl[tid] += s_wctrl[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_Uo[dt_idx] = (s_wsum[0] > 0.0 && isfinite(s_wsum[0]))
                       ? s_wctrl[0] / s_wsum[0]
                       : 0.0;
  }
}

void LogMPPI_GPU::solve() {
  constexpr double kCollisionCost = 1.0e8;

  start = std::chrono::high_resolution_clock::now();

  uploadControl();
  uploadState();
  generateNoise();

  const int block = 256;
  const int grid = (N + block - 1) / block;
  log_mppi_rollout_kernel<<<grid, block>>>(
      d_U0, d_Ui, d_noise, d_sigma, d_x_init, d_x_target, d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution, d_circles,
      n_circles, d_rects, n_rects, d_ws_boxes, n_ws_boxes, ws_link_radius,
      ws_safe_margin, ws_hard_margin, N, dim_u, dim_x, T,
      static_cast<double>(dt), gamma_u, model_type, false);
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs,
                        static_cast<size_t>(N) * sizeof(double),
                        cudaMemcpyDeviceToHost));

  bool has_valid = false;
  double min_valid_cost = std::numeric_limits<double>::infinity();
  double best_cost = std::numeric_limits<double>::infinity();
  int best_idx = 0;

  for (int i = 0; i < N; ++i) {
    const double cost = h_costs[i];
    if (std::isfinite(cost) && cost < best_cost) {
      best_cost = cost;
      best_idx = i;
    }
    if (std::isfinite(cost) && cost < kCollisionCost) {
      has_valid = true;
      min_valid_cost = std::min(min_valid_cost, cost);
    }
  }

  const size_t control_bytes =
      static_cast<size_t>(dim_u) * static_cast<size_t>(T) * sizeof(double);
  if (has_valid) {
    const int dt_blocks = dim_u * T;
    const int wsum_threads = std::min(256, N);
    const size_t shared_sz = 2 * wsum_threads * sizeof(double);
    log_mppi_logsumexp_weighted_sum_kernel<<<dt_blocks, wsum_threads,
                                             shared_sz>>>(
        d_Ui, d_costs, min_valid_cost, gamma_u, kCollisionCost, d_Uo, N, dim_u,
        T);
    CUDA_CHECK(cudaGetLastError());
  } else {
    CUDA_CHECK(cudaMemcpy(d_Uo, d_Ui + static_cast<size_t>(best_idx) * dim_u * T,
                          control_bytes, cudaMemcpyDeviceToDevice));
  }

  std::vector<double> h_Uo(static_cast<size_t>(dim_u) * T);
  CUDA_CHECK(cudaMemcpy(h_Uo.data(), d_Uo, control_bytes,
                        cudaMemcpyDeviceToHost));

  Uo.resize(dim_u, T);
  for (int d = 0; d < dim_u; ++d) {
    for (int t = 0; t < T; ++t) {
      Uo(d, t) = h_Uo[static_cast<size_t>(d) * T + t];
    }
  }
  h(Uo);

  CUDA_CHECK(cudaDeviceSynchronize());
  finish = std::chrono::high_resolution_clock::now();
  elapsed_1 = finish - start;
  elapsed_rollout = elapsed_1.count();
  elapsed_clustering = 0.0;
  elapsed_connection = 0.0;
  elapsed_guide = 0.0;
  elapsed = elapsed_rollout;

  u0 = Uo.col(0);

  Xo.col(0) = x_init;
  for (int t = 0; t < T; ++t) {
    Xo.col(t + 1) =
        Xo.col(t) + static_cast<double>(dt) * f(Xo.col(t), Uo.col(t));
  }

  visual_traj.push_back(x_init);
}
