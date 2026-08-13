#include "cluster_mppi_gpu.cuh"
#include "gpu_kmeans.cuh"

#include <algorithm>
#include <limits>
#include <numeric>

// Vis helpers (defined in mppi_gpu.cu but local per TU, so redefine here)
static void cluster_vis_reconstruct_forward(
    const Eigen::MatrixXd &Ui_cpu, // (N*dim_u) x T (Eigen layout)
    const Eigen::VectorXd &x0, int N, int dim_u, int dim_x, int T, double dt,
    const std::vector<int> &indices,
    std::vector<Eigen::MatrixXd> &out,
    std::vector<double> &out_costs,
    const Eigen::VectorXd &all_costs,
    const std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> &f_func) {
  out.clear();
  out_costs.clear();
  for (int idx : indices) {
    Eigen::MatrixXd Xi(dim_x, T + 1);
    Xi.col(0) = x0;
    for (int t = 0; t < T; ++t) {
      Xi.col(t + 1) = Xi.col(t) + dt * f_func(Xi.col(t), Ui_cpu.middleRows(idx * dim_u, dim_u).col(t));
    }
    out.push_back(Xi);
    out_costs.push_back(all_costs(idx));
  }
}
static std::vector<int> cluster_vis_select(const Eigen::VectorXd &costs,
                                           int N, int max_save) {
  std::vector<int> si(N);
  std::iota(si.begin(), si.end(), 0);
  std::sort(si.begin(), si.end(),
            [&](int a, int b) { return costs(a) < costs(b); });
  std::vector<int> sel;
  int stride = std::max(1, N / max_save);
  for (int i = 0; i < N && (int)sel.size() < max_save; i += stride)
    sel.push_back(si[i]);
  return sel;
}

// Local rollout kernel (ODR-safe: unique name per TU)
DEFINE_FORWARD_ROLLOUT_KERNEL(cluster_rollout_kernel)


ClusterMPPI_GPU::~ClusterMPPI_GPU() {}

void ClusterMPPI_GPU::init(MPPIParam param) {
  MPPI_GPU::init(param);
  clustering_method = param.clustering_method;
  kmeans_clusters = param.kmeans_clusters;
  kmeans_max_iterations = param.kmeans_max_iterations;
  kmeans_threshold = param.kmeans_threshold;
}

// ---- fastsc K-means (GPU, using the same features as DBSCAN) ----
void ClusterMPPI_GPU::kmeansCluster(
    std::vector<std::vector<int>> &clusters,
    const Eigen::MatrixXd &feature_source, const Eigen::VectorXd &costs,
    int N_samples, int T_steps) {
  constexpr int kNumBlocks = 3;
  Eigen::MatrixXd block_feature;
  const Eigen::MatrixXd *feature_ptr = &feature_source;

  const bool source_is_control_sequence =
      feature_source.rows() == N_samples * dim_u &&
      feature_source.cols() == T_steps && T_steps > 0;
  if (source_is_control_sequence) {
    block_feature = Eigen::MatrixXd::Zero(kNumBlocks * dim_u, N_samples);
    for (int i = 0; i < N_samples; ++i) {
      for (int block = 0; block < kNumBlocks; ++block) {
        const int t0 = (block * T_steps) / kNumBlocks;
        const int t1 = ((block + 1) * T_steps) / kNumBlocks;
        const int length = std::max(1, t1 - t0);
        for (int d = 0; d < dim_u; ++d) {
          double sum = 0.0;
          for (int t = t0; t < t1; ++t)
            sum += feature_source(i * dim_u + d, t);
          block_feature(block * dim_u + d, i) = sum / length;
        }
      }
    }
    feature_ptr = &block_feature;
  }

  runGPUKMeans(clusters, *feature_ptr, costs,
               {kmeans_clusters, kmeans_max_iterations, kmeans_threshold});
}

// ---- DBSCAN (CPU, aligned with BiMPPI_GPU) ----
void ClusterMPPI_GPU::dbscan(std::vector<std::vector<int>> &clusters,
                              const Eigen::MatrixXd &feature_source,
                              const Eigen::VectorXd &costs,
                              int N_samples, int T_steps) {
  clusters.clear();

  // feature_source supports two layouts:
  //   1) dim_feature x N_samples       : use as a precomputed feature matrix.
  //   2) (N_samples * dim_u) x T_steps : sampled control sequences Ui_cpu.
  //      In this case, build a 3-block input feature internally:
  //        D_i = [mean_{0:T/3} W_u u_i,
  //               mean_{T/3:2T/3} W_u u_i,
  //               mean_{2T/3:T} W_u u_i].
  //      The shared warm-start control cancels in pairwise distances.
  constexpr int kNumBlocks = 3;
  Eigen::MatrixXd block_feature;
  const Eigen::MatrixXd *feature_ptr = &feature_source;

  const bool source_is_control_sequence =
      (feature_source.rows() == N_samples * dim_u &&
       feature_source.cols() == T_steps && T_steps > 0);

  if (source_is_control_sequence) {
    const int feature_T = static_cast<int>(feature_source.cols());
    block_feature = Eigen::MatrixXd::Zero(kNumBlocks * dim_u, N_samples);

    for (int i = 0; i < N_samples; ++i) {
      for (int b = 0; b < kNumBlocks; ++b) {
        const int t0 = (b * feature_T) / kNumBlocks;
        const int t1 = ((b + 1) * feature_T) / kNumBlocks;
        const int len = std::max(1, t1 - t0);

        for (int d = 0; d < dim_u; ++d) {
          double acc = 0.0;
          for (int t = t0; t < t1; ++t) {
            acc += feature_source(i * dim_u + d, t);
          }

          const double scale = 1.0;
          block_feature(b * dim_u + d, i) =
              (acc / static_cast<double>(len)) / scale;
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
    if (valid[i]) valid_indices.push_back(i);
  }

  if (valid_indices.empty()) {
    if (N_samples > 0) clusters.push_back(std::vector<int>{best_idx});
    return;
  }

  const bool use_squared_distance = (deviation_mu > 0.0 && epsilon > 0.0);
  const double eps_scaled = use_squared_distance ? epsilon / deviation_mu : 0.0;
  const double eps_scaled_sq = eps_scaled * eps_scaled;

#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < N_samples; ++i) {
    if (!valid[i]) continue;
    std::vector<int> &neighbors = upper_neighbors[i];
    for (int j = i + 1; j < N_samples; ++j) {
      if (!valid[j]) continue;
      double dist_sq = 0.0;
      for (int d = 0; d < feature.rows(); ++d) {
        const double diff = feature(d, i) - feature(d, j);
        dist_sq += diff * diff;
      }
      const bool is_neighbor = use_squared_distance
                                   ? (dist_sq < eps_scaled_sq)
                                   : (deviation_mu * std::sqrt(dist_sq) < epsilon);
      if (is_neighbor) neighbors.push_back(j);
    }
  }

  std::vector<size_t> degrees(N_samples, 0);
  for (int i = 0; i < N_samples; ++i) {
    degrees[i] += upper_neighbors[i].size();
    for (int j : upper_neighbors[i]) ++degrees[j];
  }

  std::vector<std::vector<int>> tree(N_samples);
  for (int i = 0; i < N_samples; ++i) tree[i].reserve(degrees[i]);
  for (int i = 0; i < N_samples; ++i) {
    for (int j : upper_neighbors[i]) {
      tree[i].push_back(j);
      tree[j].push_back(i);
    }
  }

  std::vector<char> core(N_samples, false);
  for (int i = 0; i < N_samples; ++i) {
    if ((int)tree[i].size() > minpts) core[i] = true;
  }

  std::vector<char> visited(N_samples, false);
  for (int i = 0; i < N_samples; ++i) {
    if (!core[i] || visited[i]) continue;
    std::deque<int> branch;
    std::vector<int> cluster;
    branch.push_back(i);
    cluster.push_back(i);
    visited[i] = true;
    while (!branch.empty()) {
      const int now = branch.front();
      branch.pop_front();
      for (int nb : tree[now]) {
        if (visited[nb]) continue;
        visited[nb] = true;
        cluster.push_back(nb);
        if (core[nb]) branch.push_back(nb);
      }
    }
    clusters.push_back(cluster);
  }

  if (clusters.empty()) {
    clusters.push_back(std::move(valid_indices));
  }
}

void ClusterMPPI_GPU::calculateU(Eigen::MatrixXd &Uout,
                                  const std::vector<std::vector<int>> &clusters,
                                  const Eigen::VectorXd &costs,
                                  const Eigen::MatrixXd &Ui_cpu,
                                  int T_steps) {
  int n_cl = (int)clusters.size();
  Uout = Eigen::MatrixXd::Zero(n_cl * dim_u, T_steps);

#pragma omp parallel for
  for (int idx = 0; idx < n_cl; ++idx) {
    int pts = (int)clusters[idx].size();
    double min_c = std::numeric_limits<double>::max();
    for (int k : clusters[idx]) min_c = std::min(min_c, costs(k));

    std::vector<double> wts(pts);
    double tw = 0.0;
    for (int i = 0; i < pts; ++i) {
      wts[i] = std::exp(-gamma_u * (costs(clusters[idx][i]) - min_c));
      tw += wts[i];
    }
    for (int i = 0; i < pts; ++i) {
      int k = clusters[idx][i];
      Uout.middleRows(idx * dim_u, dim_u) +=
          (wts[i] / tw) * Ui_cpu.middleRows(k * dim_u, dim_u);
    }
    // Clamp
    Eigen::Ref<Eigen::MatrixXd> slice = Uout.middleRows(idx * dim_u, dim_u);
    if (h) h(slice);
  }
}

// ---- Override solve() ----
void ClusterMPPI_GPU::solve() {
  std::vector<int> full_cluster(N);
  std::iota(full_cluster.begin(), full_cluster.end(), 0);

  start = std::chrono::high_resolution_clock::now();

  // --- GPU rollout ---
  uploadControl();
  uploadState();
  generateNoise();

  const int BLOCK = 256;
  int grid = (N + BLOCK - 1) / BLOCK;
  cluster_rollout_kernel<<<grid, BLOCK>>>(
      d_U0, d_Ui, d_noise, d_sigma,
      d_x_init, d_x_target,
      d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution,
      d_circles, n_circles, d_rects, n_rects,
      N, dim_u, dim_x, T, (double)dt, gamma_u, model_type, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t_cluster_start = std::chrono::high_resolution_clock::now();
  elapsed_rollout = std::chrono::duration<double>(t_cluster_start - start).count();

  // Copy costs + Ui to host. Both backends build the same 3-block feature
  // representation from the sampled control sequences.
  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Ui host copy: N x dim_u x T → Eigen (N*dim_u) x T
  std::vector<double> h_Ui_flat((size_t)N * dim_u * T);
  CUDA_CHECK(cudaMemcpy(h_Ui_flat.data(), d_Ui,
                        (size_t)N * dim_u * T * sizeof(double), cudaMemcpyDeviceToHost));
  Eigen::MatrixXd Ui_cpu(N * dim_u, T);
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < dim_u; ++d)
      for (int t = 0; t < T; ++t)
        Ui_cpu(i * dim_u + d, t) = h_Ui_flat[i * (dim_u * T) + d * T + t];

  Eigen::VectorXd costs_cpu = Eigen::Map<Eigen::VectorXd>(h_costs.data(), N);

  // --- Selected clustering backend ---
  std::vector<std::vector<int>> clusters;
  if (clustering_method == ClusteringMethod::KMeans)
    kmeansCluster(clusters, Ui_cpu, costs_cpu, N, T);
  else
    dbscan(clusters, Ui_cpu, costs_cpu, N, T);
  if (clusters.empty()) clusters.push_back(full_cluster);
  calculateU(U, clusters, costs_cpu, Ui_cpu, T);

  // Select best cluster
  double min_cost = std::numeric_limits<double>::max();
  int min_idx = 0;
  for (int ci = 0; ci < (int)clusters.size(); ++ci) {
    Eigen::MatrixXd Xi(dim_x, T + 1);
    Xi.col(0) = x_init;
    for (int j = 0; j < T; ++j) {
      Xi.col(j + 1) = Xi.col(j) + (double)dt * f(Xi.col(j), U.block(ci * dim_u, j, dim_u, 1));
    }
    const Eigen::MatrixXd Ui = U.middleRows(ci * dim_u, dim_u);
    const double cost = evaluateTrajectoryCost(Xi, Ui);
    if (cost < min_cost) { min_cost = cost; min_idx = ci; }
  }

  Uo = U.middleRows(min_idx * dim_u, dim_u);

  auto t_end = std::chrono::high_resolution_clock::now();
  finish = t_end;
  elapsed_clustering = std::chrono::duration<double>(t_end - t_cluster_start).count();
  elapsed_connection = 0.0;
  elapsed_guide      = 0.0;
  elapsed_1 = finish - start;
  elapsed   = elapsed_rollout + elapsed_clustering;

  u0 = Uo.col(0);

  Xo.col(0) = x_init;
  for (int j = 0; j < T; ++j) {
    Xo.col(j + 1) = Xo.col(j) + (double)dt * f(Xo.col(j), Uo.col(j));
  }
  cost = evaluateTrajectoryCost(Xo, Uo);

  // ── Visualization data export ──
  if (vis_logger && vis_logger->enabled) {
    // Rollout samples
    auto sel = cluster_vis_select(costs_cpu, N, 50);
    std::vector<Eigen::MatrixXd> sample_trajs;
    std::vector<double> sample_costs;
    cluster_vis_reconstruct_forward(Ui_cpu, x_init, N, dim_u, dim_x, T, dt,
                                    sel, sample_trajs, sample_costs, costs_cpu, f);
    vis_logger->saveTrajectories("rollouts", sample_trajs);
    vis_logger->saveCosts("rollouts", sample_costs);
    // Cluster representative trajectories
    std::vector<Eigen::MatrixXd> cluster_trajs;
    for (int ci = 0; ci < (int)clusters.size(); ++ci) {
      Eigen::MatrixXd Xi(dim_x, T + 1);
      Xi.col(0) = x_init;
      for (int j = 0; j < T; ++j) {
        Xi.col(j + 1) = Xi.col(j) + (double)dt * f(Xi.col(j), U.block(ci * dim_u, j, dim_u, 1));
      }
      cluster_trajs.push_back(Xi);
    }
    vis_logger->saveTrajectories("clusters", cluster_trajs);
    vis_logger->saveTrajectory("optimal", Xo);
    vis_logger->saveOptimalCost(cost);
    vis_logger->savePosition(x_init);
  }

  visual_traj.push_back(x_init);
}
