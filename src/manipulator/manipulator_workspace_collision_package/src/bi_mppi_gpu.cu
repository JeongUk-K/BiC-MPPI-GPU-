#include "bi_mppi_gpu.cuh"
#include "gpu_kmeans.cuh"

#include <algorithm>
#include <vector>

void BiMPPI_GPU::clusterControlsDevice(
    const double *device_controls, const double *device_costs,
    int sample_count, int horizon, Eigen::MatrixXd &clustered_controls,
    std::vector<std::vector<int>> &clusters) {
  std::vector<int> active_clusters;
  std::vector<double> host_controls;
  runGPUKMeansControlsDevice(
      device_controls, device_costs, sample_count, dim_u, horizon, gamma_u,
      {kmeans_clusters, kmeans_max_iterations, kmeans_threshold},
      active_clusters, host_controls);

  const int active_count = static_cast<int>(active_clusters.size());
  clusters.assign(active_count, {});
  clustered_controls =
      Eigen::MatrixXd::Zero(active_count * dim_u, horizon);
  const int control_size = dim_u * horizon;
  for (int cluster = 0; cluster < active_count; ++cluster) {
    clusters[cluster].push_back(active_clusters[cluster]);
    const double *source =
        host_controls.data() + static_cast<std::size_t>(cluster) * control_size;
    for (int d = 0; d < dim_u; ++d) {
      for (int t = 0; t < horizon; ++t) {
        clustered_controls(cluster * dim_u + d, t) =
            source[d * horizon + t];
      }
    }
    Eigen::Ref<Eigen::MatrixXd> control =
        clustered_controls.middleRows(cluster * dim_u, dim_u);
    h(control);
  }
}

void BiMPPI_GPU::appendDeviceVisRolloutSamples(
    const double *device_controls, int sample_count, int horizon,
    bool backward) {
  if (!vis_logger || !vis_logger->enabled || sample_count <= 0) return;
  const int copied_samples =
      std::min(sample_count, vis_rollout_samples_per_call);
  std::vector<double> flat(
      static_cast<std::size_t>(copied_samples) * dim_u * horizon);
  CUDA_CHECK(cudaMemcpy(flat.data(), device_controls,
                        flat.size() * sizeof(double), cudaMemcpyDeviceToHost));
  Eigen::MatrixXd controls(copied_samples * dim_u, horizon);
  for (int sample = 0; sample < copied_samples; ++sample) {
    for (int d = 0; d < dim_u; ++d) {
      for (int t = 0; t < horizon; ++t) {
        controls(sample * dim_u + d, t) =
            flat[(static_cast<std::size_t>(sample) * dim_u + d) * horizon + t];
      }
    }
  }
  appendVisRolloutSamples(controls, copied_samples, horizon, backward);
}

void BiMPPI_GPU::forwardRolloutDevice() {
  launchForwardSampling();
  const auto cluster_start = std::chrono::high_resolution_clock::now();

  appendDeviceVisRolloutSamples(d_Ufi, Nf, Tf, false);
  clusterControlsDevice(d_Ufi, d_costs_f, Nf, Tf, Uf, clusters_f);

  Xf.resize(static_cast<int>(clusters_f.size()) * dim_x, Tf + 1);
  for (int cluster = 0; cluster < static_cast<int>(clusters_f.size());
       ++cluster) {
    Xf.block(cluster * dim_x, 0, dim_x, 1) = x_init;
    for (int t = 0; t < Tf; ++t) {
      Xf.block(cluster * dim_x, t + 1, dim_x, 1) =
          Xf.block(cluster * dim_x, t, dim_x, 1) +
          static_cast<double>(dt) *
              f(Xf.block(cluster * dim_x, t, dim_x, 1),
                Uf.block(cluster * dim_u, t, dim_u, 1));
    }
  }
  elapsed_clustering +=
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - cluster_start)
          .count();
}

void BiMPPI_GPU::backwardRolloutDevice() {
  launchBackwardSampling();
  const auto cluster_start = std::chrono::high_resolution_clock::now();

  appendDeviceVisRolloutSamples(d_Ubi, Nb, Tb, true);
  clusterControlsDevice(d_Ubi, d_costs_b, Nb, Tb, Ub, clusters_b);

  Xb.resize(static_cast<int>(clusters_b.size()) * dim_x, Tb + 1);
  for (int cluster = 0; cluster < static_cast<int>(clusters_b.size());
       ++cluster) {
    Xb.block(cluster * dim_x, Tb, dim_x, 1) = x_target;
    for (int t = Tb - 1; t >= 0; --t) {
      const int control_column = (t == Tb - 1) ? t : t + 1;
      const Eigen::VectorXd control =
          Ub.block(cluster * dim_u, control_column, dim_u, 1);
      Xb.block(cluster * dim_x, t, dim_x, 1) =
          Xb.block(cluster * dim_x, t + 1, dim_x, 1) -
          static_cast<double>(dt) *
              f(Xb.block(cluster * dim_x, t + 1, dim_x, 1), control);
    }
  }
  elapsed_clustering +=
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - cluster_start)
          .count();
}

void BiMPPI_GPU::solve() {
  if (clustering_method != ClusteringMethod::KMeans) {
    BiMPPI_GPU_Legacy::solve();
    return;
  }

  elapsed_rollout = elapsed_clustering = 0.0;
  vis_rollout_samples.clear();
  start = std::chrono::high_resolution_clock::now();
  backwardRolloutDevice();
  forwardRolloutDevice();

  const auto connection_start = std::chrono::high_resolution_clock::now();
  selectConnection();
  concatenate();
  const auto guide_start = std::chrono::high_resolution_clock::now();
  elapsed_connection =
      std::chrono::duration<double>(guide_start - connection_start).count();
  guideMPPIDevice();
  const auto finish_time = std::chrono::high_resolution_clock::now();
  elapsed_guide =
      std::chrono::duration<double>(finish_time - guide_start).count();
  elapsed_1 = finish_time - start;
  partitioningControl();
  elapsed = elapsed_rollout + elapsed_clustering + elapsed_connection +
            elapsed_guide;

  if (vis_logger && vis_logger->enabled) {
    if (!vis_rollout_samples.empty())
      vis_logger->saveTrajectories("rollouts", vis_rollout_samples);
    std::vector<Eigen::MatrixXd> forward_trajectories;
    for (int cluster = 0; cluster < static_cast<int>(clusters_f.size());
         ++cluster)
      forward_trajectories.push_back(
          Xf.block(cluster * dim_x, 0, dim_x, Tf + 1));
    vis_logger->saveTrajectories("forward_clusters", forward_trajectories);
    std::vector<Eigen::MatrixXd> backward_trajectories;
    for (int cluster = 0; cluster < static_cast<int>(clusters_b.size());
         ++cluster)
      backward_trajectories.push_back(
          Xb.block(cluster * dim_x, 0, dim_x, Tb + 1));
    vis_logger->saveTrajectories("backward_clusters", backward_trajectories);
    if (!Xr.empty()) vis_logger->saveTrajectories("guide_candidates", Xr);
    vis_logger->saveTrajectory("optimal", Xo);
    vis_logger->savePosition(x_init);
  }

  visual_traj.push_back(x_init);
}
