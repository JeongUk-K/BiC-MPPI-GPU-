#include "cluster_mppi_gpu.cuh"

#include <algorithm>
#include <limits>
#include <numeric>

DEFINE_FORWARD_ROLLOUT_KERNEL(cluster_rollout_kernel)

ClusterMPPI_GPU::~ClusterMPPI_GPU() {}

void ClusterMPPI_GPU::dbscan(std::vector<std::vector<int>> &clusters,
                             const Eigen::MatrixXd &feature_source,
                             const Eigen::VectorXd &costs, int N_samples,
                             int T_steps) {
  clusters.clear();

  constexpr int kNumBlocks = 3;
  Eigen::MatrixXd block_feature;
  const Eigen::MatrixXd *feature_ptr = &feature_source;

  const bool source_is_control_sequence =
      (feature_source.rows() == N_samples * dim_u &&
       feature_source.cols() == T_steps && T_steps > 0);

  if (source_is_control_sequence) {
    block_feature = Eigen::MatrixXd::Zero(kNumBlocks * dim_u, N_samples);
    for (int i = 0; i < N_samples; ++i) {
      for (int b = 0; b < kNumBlocks; ++b) {
        const int t0 = (b * T_steps) / kNumBlocks;
        const int t1 = ((b + 1) * T_steps) / kNumBlocks;
        const int len = std::max(1, t1 - t0);
        for (int d = 0; d < dim_u; ++d) {
          double acc = 0.0;
          for (int t = t0; t < t1; ++t) {
            acc += feature_source(i * dim_u + d, t);
          }
          block_feature(b * dim_u + d, i) =
              acc / static_cast<double>(len);
        }
      }
    }
    feature_ptr = &block_feature;
  }

  const Eigen::MatrixXd &feature = *feature_ptr;
  std::vector<std::vector<int>> upper_neighbors(N_samples);
  std::vector<char> valid(N_samples, false);
  std::vector<int> valid_indices;
  valid_indices.reserve(std::max(0, N_samples));

  constexpr double J_col = 1e8;
  int best_idx = 0;
  double best_cost = (N_samples > 0) ? costs(0) : 0.0;
  for (int i = 0; i < N_samples; ++i) {
    if (costs(i) < best_cost) {
      best_cost = costs(i);
      best_idx = i;
    }
    valid[i] = (costs(i) < J_col);
    if (valid[i]) {
      valid_indices.push_back(i);
    }
  }

  if (valid_indices.empty()) {
    if (N_samples > 0) {
      clusters.push_back(std::vector<int>{best_idx});
    }
    return;
  }

  const double eps_scaled = epsilon / std::max(1.0e-12, deviation_mu);
  const double eps_scaled_sq = eps_scaled * eps_scaled;

#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < N_samples; ++i) {
    if (!valid[i]) {
      continue;
    }
    std::vector<int> &neighbors = upper_neighbors[i];
    for (int j = i + 1; j < N_samples; ++j) {
      if (!valid[j]) {
        continue;
      }
      double dist_sq = 0.0;
      for (int d = 0; d < feature.rows(); ++d) {
        const double diff = feature(d, i) - feature(d, j);
        dist_sq += diff * diff;
      }
      if (dist_sq < eps_scaled_sq) {
        neighbors.push_back(j);
      }
    }
  }

  std::vector<size_t> degrees(N_samples, 0);
  for (int i = 0; i < N_samples; ++i) {
    degrees[i] += upper_neighbors[i].size();
    for (int j : upper_neighbors[i]) {
      ++degrees[j];
    }
  }

  std::vector<std::vector<int>> tree(N_samples);
  for (int i = 0; i < N_samples; ++i) {
    tree[i].reserve(degrees[i]);
  }
  for (int i = 0; i < N_samples; ++i) {
    for (int j : upper_neighbors[i]) {
      tree[i].push_back(j);
      tree[j].push_back(i);
    }
  }

  std::vector<char> core(N_samples, false);
  for (int i = 0; i < N_samples; ++i) {
    if (static_cast<int>(tree[i].size()) > minpts) {
      core[i] = true;
    }
  }

  std::vector<char> visited(N_samples, false);
  for (int i = 0; i < N_samples; ++i) {
    if (!core[i] || visited[i]) {
      continue;
    }
    std::deque<int> branch;
    std::vector<int> cluster;
    branch.push_back(i);
    cluster.push_back(i);
    visited[i] = true;
    while (!branch.empty()) {
      const int now = branch.front();
      branch.pop_front();
      for (int nb : tree[now]) {
        if (visited[nb]) {
          continue;
        }
        visited[nb] = true;
        cluster.push_back(nb);
        if (core[nb]) {
          branch.push_back(nb);
        }
      }
    }
    clusters.push_back(cluster);
  }

  if (clusters.empty()) {
    clusters.push_back(std::move(valid_indices));
  }
}

void ClusterMPPI_GPU::calculateU(
    Eigen::MatrixXd &Uout, const std::vector<std::vector<int>> &clusters,
    const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui_cpu, int T_steps) {
  const int n_cl = static_cast<int>(clusters.size());
  Uout = Eigen::MatrixXd::Zero(n_cl * dim_u, T_steps);

#pragma omp parallel for
  for (int idx = 0; idx < n_cl; ++idx) {
    const int pts = static_cast<int>(clusters[idx].size());
    double min_c = std::numeric_limits<double>::max();
    for (int k : clusters[idx]) {
      min_c = std::min(min_c, costs(k));
    }

    std::vector<double> wts(pts);
    double tw = 0.0;
    for (int i = 0; i < pts; ++i) {
      wts[i] = std::exp(-gamma_u * (costs(clusters[idx][i]) - min_c));
      tw += wts[i];
    }
    for (int i = 0; i < pts; ++i) {
      const int k = clusters[idx][i];
      Uout.middleRows(idx * dim_u, dim_u) +=
          (wts[i] / tw) * Ui_cpu.middleRows(k * dim_u, dim_u);
    }
    Eigen::Ref<Eigen::MatrixXd> slice = Uout.middleRows(idx * dim_u, dim_u);
    h(slice);
  }
}

void ClusterMPPI_GPU::solve() {
  std::vector<int> full_cluster(N);
  std::iota(full_cluster.begin(), full_cluster.end(), 0);

  start = std::chrono::high_resolution_clock::now();

  uploadControl();
  uploadState();
  generateNoise();

  const int block = 256;
  const int grid = (N + block - 1) / block;
  cluster_rollout_kernel<<<grid, block>>>(
      d_U0, d_Ui, d_noise, d_sigma, d_x_init, d_x_target, d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution, d_circles,
      n_circles, d_rects, n_rects, d_ws_boxes, n_ws_boxes, ws_link_radius,
      ws_safe_margin, ws_hard_margin, N, dim_u, dim_x, T, static_cast<double>(dt),
      gamma_u, model_type, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto t_cluster_start = std::chrono::high_resolution_clock::now();
  elapsed_rollout =
      std::chrono::duration<double>(t_cluster_start - start).count();

  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs,
                        static_cast<size_t>(N) * sizeof(double),
                        cudaMemcpyDeviceToHost));

  std::vector<double> h_Ui_flat(static_cast<size_t>(N) * dim_u * T);
  CUDA_CHECK(cudaMemcpy(h_Ui_flat.data(), d_Ui,
                        static_cast<size_t>(N) * dim_u * T * sizeof(double),
                        cudaMemcpyDeviceToHost));

  Eigen::MatrixXd Ui_cpu(N * dim_u, T);
  for (int i = 0; i < N; ++i) {
    for (int d = 0; d < dim_u; ++d) {
      for (int t = 0; t < T; ++t) {
        Ui_cpu(i * dim_u + d, t) =
            h_Ui_flat[static_cast<size_t>(i) * (dim_u * T) + d * T + t];
      }
    }
  }

  Eigen::VectorXd costs_cpu = Eigen::Map<Eigen::VectorXd>(h_costs.data(), N);

  std::vector<std::vector<int>> clusters;
  dbscan(clusters, Ui_cpu, costs_cpu, N, T);
  if (clusters.empty()) {
    clusters.push_back(full_cluster);
  }
  calculateU(U, clusters, costs_cpu, Ui_cpu, T);

  double min_cost = std::numeric_limits<double>::max();
  int min_idx = 0;
  for (int ci = 0; ci < static_cast<int>(clusters.size()); ++ci) {
    Eigen::MatrixXd Xi(dim_x, T + 1);
    Xi.col(0) = x_init;
    double cost = 0.0;
    for (int t = 0; t < T; ++t) {
      cost += p(Xi.col(t), x_target);
      Xi.col(t + 1) =
          Xi.col(t) + static_cast<double>(dt) *
                          f(Xi.col(t), U.block(ci * dim_u, t, dim_u, 1));
    }
    cost += p(Xi.col(T), x_target);
    if (collision_checker) {
      for (int t = 1; t < T + 1; ++t) {
        if (collision_checker->getCollisionGrid(Xi.col(t))) {
          cost = 1e8;
          break;
        }
      }
    }
    if (cost < min_cost) {
      min_cost = cost;
      min_idx = ci;
    }
  }

  Uo = U.middleRows(min_idx * dim_u, dim_u);

  finish = std::chrono::high_resolution_clock::now();
  elapsed_clustering =
      std::chrono::duration<double>(finish - t_cluster_start).count();
  elapsed_connection = 0.0;
  elapsed_guide = 0.0;
  elapsed_1 = finish - start;
  elapsed = elapsed_rollout + elapsed_clustering;

  u0 = Uo.col(0);
  Xo.col(0) = x_init;
  for (int t = 0; t < T; ++t) {
    Xo.col(t + 1) =
        Xo.col(t) + static_cast<double>(dt) * f(Xo.col(t), Uo.col(t));
  }

  visual_traj.push_back(x_init);
}
