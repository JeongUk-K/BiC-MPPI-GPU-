#pragma once

#include <kmeans.h>

#include <Eigen/Dense>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

struct GPUKMeansParam {
  GPUKMeansParam(int cluster_count = 5, int iterations = 100,
                 double convergence_threshold = 1e-6,
                 double collision_cost_threshold = 1e8)
      : clusters(cluster_count), max_iterations(iterations),
        threshold(convergence_threshold),
        collision_cost(collision_cost_threshold) {}
  int clusters;
  int max_iterations;
  double threshold;
  double collision_cost;
};

struct GPUKMeansDeviceCache {
  thrust::device_vector<double> data;
  thrust::device_vector<int> labels;
  thrust::device_vector<double> centroids;
  thrust::device_vector<double> distances;
  kmeans::Workspace<double> workspace;
};

// Runs fastsc K-means on the columns of feature. Collision-penalized samples
// are excluded in the same way as the DBSCAN implementations.
inline void runGPUKMeans(std::vector<std::vector<int>> &clusters,
                         const Eigen::Ref<const Eigen::MatrixXd> &feature,
                         const Eigen::Ref<const Eigen::VectorXd> &costs,
                         const GPUKMeansParam &param) {
  clusters.clear();
  const int sample_count = static_cast<int>(feature.cols());
  if (sample_count == 0) return;
  if (costs.size() != sample_count)
    throw std::invalid_argument("K-means feature/cost sample count mismatch");

  std::vector<int> valid_indices;
  valid_indices.reserve(sample_count);
  int best_index = 0;
  for (int i = 0; i < sample_count; ++i) {
    if (costs(i) < costs(best_index)) best_index = i;
    bool finite_feature = true;
    for (int d = 0; d < feature.rows(); ++d)
      finite_feature = finite_feature && std::isfinite(feature(d, i));
    if (std::isfinite(costs(i)) && costs(i) < param.collision_cost &&
        finite_feature)
      valid_indices.push_back(i);
  }
  if (valid_indices.empty()) {
    clusters.push_back({best_index});
    return;
  }

  const int dimensions = static_cast<int>(feature.rows());
  const int k = std::max(
      1, std::min(param.clusters, static_cast<int>(valid_indices.size())));
  const int iterations = std::max(1, param.max_iterations);
  const double threshold = std::max(0.0, param.threshold);

  thrust::host_vector<double> host_data(valid_indices.size() * dimensions);
  thrust::host_vector<int> host_labels(valid_indices.size(), 0);
  for (int row = 0; row < static_cast<int>(valid_indices.size()); ++row) {
    const int source = valid_indices[row];
    for (int d = 0; d < dimensions; ++d)
      host_data[row * dimensions + d] = feature(d, source);
  }

  // Deterministic farthest-point initialization avoids near-identical initial
  // centroids for shuffled MPPI samples and makes backend comparisons stable.
  std::vector<int> seeds;
  seeds.reserve(k);
  int first_seed = 0;
  for (int row = 1; row < static_cast<int>(valid_indices.size()); ++row)
    if (costs(valid_indices[row]) < costs(valid_indices[first_seed]))
      first_seed = row;
  seeds.push_back(first_seed);
  std::vector<double> min_distance(valid_indices.size(),
                                   std::numeric_limits<double>::infinity());
  while (static_cast<int>(seeds.size()) < k) {
    const int newest = seeds.back();
    int farthest = 0;
    double farthest_distance = -1.0;
    for (int row = 0; row < static_cast<int>(valid_indices.size()); ++row) {
      double distance = 0.0;
      for (int d = 0; d < dimensions; ++d) {
        const double delta = host_data[row * dimensions + d] -
                             host_data[newest * dimensions + d];
        distance += delta * delta;
      }
      min_distance[row] = std::min(min_distance[row], distance);
      if (min_distance[row] > farthest_distance) {
        farthest_distance = min_distance[row];
        farthest = row;
      }
    }
    seeds.push_back(farthest);
  }

  thrust::host_vector<double> host_centroids(k * dimensions);
  for (int cluster = 0; cluster < k; ++cluster)
    for (int d = 0; d < dimensions; ++d)
      host_centroids[cluster * dimensions + d] =
          host_data[seeds[cluster] * dimensions + d];

  // Reuse device capacities across closed-loop solver iterations.
  static thread_local GPUKMeansDeviceCache cache;
  cache.data = host_data;
  cache.labels = host_labels;
  cache.centroids = host_centroids;
  cache.distances.resize(valid_indices.size());
  kmeans::kmeans(iterations, static_cast<int>(valid_indices.size()), dimensions,
                 k, cache.data, cache.labels, cache.centroids, cache.distances,
                 false, threshold, false, &cache.workspace);

  host_labels = cache.labels;
  clusters.resize(k);
  for (int i = 0; i < static_cast<int>(valid_indices.size()); ++i) {
    const int label = host_labels[i];
    if (label < 0 || label >= k)
      throw std::runtime_error("fastsc K-means returned an invalid label");
    clusters[label].push_back(valid_indices[i]);
  }
  clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
                                [](const std::vector<int> &cluster) {
                                  return cluster.empty();
                                }),
                 clusters.end());
  if (clusters.empty()) clusters.push_back({best_index});
}
