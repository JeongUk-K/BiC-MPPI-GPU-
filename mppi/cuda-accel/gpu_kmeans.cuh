#pragma once

#include <kmeans.h>

#include <Eigen/Dense>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
  thrust::device_vector<int> valid_indices;
  thrust::device_vector<int> cluster_counts;
  thrust::device_vector<double> cluster_min_costs;
  thrust::device_vector<double> sample_weights;
  thrust::device_vector<double> cluster_weight_sums;
  thrust::device_vector<double> clustered_controls;
  thrust::device_vector<double> valid_costs;
  thrust::device_vector<double> min_seed_distances;
  kmeans::Workspace<double> workspace;
};

namespace gpu_kmeans_detail {

struct ValidCost {
  const double *costs;
  double collision_cost;

  __device__ bool operator()(int index) const {
    const double cost = costs[index];
    return isfinite(cost) && cost < collision_cost;
  }
};

static __global__ void buildBlockMeanFeatures(
    const double *controls, const double *costs, const int *valid_indices,
    int valid_count, int dim_u, int horizon, int block_count,
    double *features, double *valid_costs) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= valid_count) return;
  const int sample = valid_indices[row];
  valid_costs[row] = costs[sample];
  const double *control = controls + sample * dim_u * horizon;
  for (int block = 0; block < block_count; ++block) {
    const int t0 = (block * horizon) / block_count;
    const int t1 = ((block + 1) * horizon) / block_count;
    const int length = max(1, t1 - t0);
    for (int d = 0; d < dim_u; ++d) {
      double sum = 0.0;
      for (int t = t0; t < t1; ++t) sum += control[d * horizon + t];
      features[row * (block_count * dim_u) + block * dim_u + d] =
          sum / static_cast<double>(length);
    }
  }
}

static __global__ void updateSeedDistances(
    const double *features, const double *centroid, int sample_count,
    int dimensions, bool reset, double *minimum_distances) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= sample_count) return;
  double distance = 0.0;
  for (int d = 0; d < dimensions; ++d) {
    const double delta =
        features[row * dimensions + d] - centroid[d];
    distance += delta * delta;
  }
  minimum_distances[row] =
      reset ? distance : min(minimum_distances[row], distance);
}

static __global__ void clusterMinCosts(
    const int *labels, const int *valid_indices, const double *costs,
    int sample_count, int cluster_count, double *minimum_costs,
    int *cluster_counts) {
  const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
  if (cluster >= cluster_count) return;
  double minimum = INFINITY;
  int count = 0;
  for (int row = 0; row < sample_count; ++row) {
    if (labels[row] != cluster) continue;
    minimum = min(minimum, costs[valid_indices[row]]);
    ++count;
  }
  minimum_costs[cluster] = minimum;
  cluster_counts[cluster] = count;
}

static __global__ void computeSampleWeights(
    const int *labels, const int *valid_indices, const double *costs,
    const double *minimum_costs, int sample_count, double gamma,
    double *weights, double *weight_sums) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= sample_count) return;
  const int cluster = labels[row];
  const double weight =
      exp(-gamma * (costs[valid_indices[row]] - minimum_costs[cluster]));
  weights[row] = weight;
  atomicAdd(weight_sums + cluster, weight);
}

static __global__ void accumulateWeightedClusterControls(
    const double *controls, const int *labels, const int *valid_indices,
    const double *weights, int sample_count, int dim_u, int horizon,
    double *output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int control_size = dim_u * horizon;
  if (index >= sample_count * control_size) return;
  const int row = index / control_size;
  const int control_index = index % control_size;
  const int cluster = labels[row];
  const int sample = valid_indices[row];
  atomicAdd(output + cluster * control_size + control_index,
            weights[row] *
                controls[sample * control_size + control_index]);
}

static __global__ void normalizeWeightedClusterControls(
    const double *weight_sums, int control_size, int output_count,
    double *output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output_count) return;
  const int cluster = index / control_size;
  const double denominator = weight_sums[cluster];
  output[index] = denominator > 0.0 ? output[index] / denominator : 0.0;
}

}  // namespace gpu_kmeans_detail

// Device-resident control-sequence K-means path used by the optimized
// BiC-MPPI solver. Only k representative controls and k counts return to CPU.
inline void runGPUKMeansControlsDevice(
    const double *device_controls, const double *device_costs,
    int sample_count, int dim_u, int horizon, double gamma,
    const GPUKMeansParam &param, std::vector<int> &active_clusters,
    std::vector<double> &host_clustered_controls) {
  active_clusters.clear();
  host_clustered_controls.clear();
  if (sample_count <= 0 || dim_u <= 0 || horizon <= 0) return;

  static thread_local GPUKMeansDeviceCache cache;
  const bool profile = std::getenv("BIC_MPPI_PROFILE_GPU_KMEANS") != nullptr;
  const auto profile_start = std::chrono::high_resolution_clock::now();
  auto profile_mark = [&]() {
    if (profile) CUDA_CHECK(cudaDeviceSynchronize());
    return std::chrono::high_resolution_clock::now();
  };
  cache.valid_indices.resize(sample_count);
  const auto begin = thrust::make_counting_iterator<int>(0);
  const auto end = begin + sample_count;
  const auto valid_end = thrust::copy_if(
      begin, end, cache.valid_indices.begin(),
      gpu_kmeans_detail::ValidCost{device_costs, param.collision_cost});
  int valid_count =
      static_cast<int>(valid_end - cache.valid_indices.begin());

  if (valid_count == 0) {
    const auto costs_begin = thrust::device_pointer_cast(device_costs);
    const int best = static_cast<int>(
        thrust::min_element(costs_begin, costs_begin + sample_count) -
        costs_begin);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(cache.valid_indices.data()),
                          &best, sizeof(int), cudaMemcpyHostToDevice));
    valid_count = 1;
  }
  cache.valid_indices.resize(valid_count);
  const auto after_filter = profile_mark();

  constexpr int block_count = 3;
  const int dimensions = block_count * dim_u;
  const int cluster_count =
      std::max(1, std::min(param.clusters, valid_count));
  cache.data.resize(static_cast<std::size_t>(valid_count) * dimensions);
  cache.valid_costs.resize(valid_count);
  cache.min_seed_distances.resize(valid_count);
  cache.labels.resize(valid_count);
  cache.centroids.resize(cluster_count * dimensions);
  cache.distances.resize(valid_count);

  const int block = 256;
  gpu_kmeans_detail::buildBlockMeanFeatures<<<
      (valid_count + block - 1) / block, block>>>(
      device_controls, device_costs,
      thrust::raw_pointer_cast(cache.valid_indices.data()), valid_count,
      dim_u, horizon, block_count,
      thrust::raw_pointer_cast(cache.data.data()),
      thrust::raw_pointer_cast(cache.valid_costs.data()));
  CUDA_CHECK(cudaGetLastError());

  const auto first_seed_it =
      thrust::min_element(cache.valid_costs.begin(), cache.valid_costs.end());
  int seed = static_cast<int>(first_seed_it - cache.valid_costs.begin());
  for (int cluster = 0; cluster < cluster_count; ++cluster) {
    double *centroid =
        thrust::raw_pointer_cast(cache.centroids.data()) +
        cluster * dimensions;
    CUDA_CHECK(cudaMemcpy(
        centroid,
        thrust::raw_pointer_cast(cache.data.data()) + seed * dimensions,
        dimensions * sizeof(double), cudaMemcpyDeviceToDevice));
    gpu_kmeans_detail::updateSeedDistances<<<
        (valid_count + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(cache.data.data()), centroid, valid_count,
        dimensions, cluster == 0,
        thrust::raw_pointer_cast(cache.min_seed_distances.data()));
    CUDA_CHECK(cudaGetLastError());
    if (cluster + 1 < cluster_count) {
      const auto farthest_it = thrust::max_element(
          cache.min_seed_distances.begin(), cache.min_seed_distances.end());
      seed = static_cast<int>(
          farthest_it - cache.min_seed_distances.begin());
    }
  }
  thrust::fill(cache.labels.begin(), cache.labels.end(), 0);
  const auto after_features = profile_mark();

  kmeans::kmeans(
      std::max(1, param.max_iterations), valid_count, dimensions, cluster_count,
      cache.data, cache.labels, cache.centroids, cache.distances, false,
      std::max(0.0, param.threshold), false, &cache.workspace);
  const auto after_kmeans = profile_mark();

  cache.cluster_counts.resize(cluster_count);
  cache.cluster_min_costs.resize(cluster_count);
  cache.sample_weights.resize(valid_count);
  cache.cluster_weight_sums.resize(cluster_count);
  cache.clustered_controls.resize(
      static_cast<std::size_t>(cluster_count) * dim_u * horizon);
  thrust::fill(cache.cluster_weight_sums.begin(),
               cache.cluster_weight_sums.end(), 0.0);
  thrust::fill(cache.clustered_controls.begin(),
               cache.clustered_controls.end(), 0.0);

  gpu_kmeans_detail::clusterMinCosts<<<
      (cluster_count + block - 1) / block, block>>>(
      thrust::raw_pointer_cast(cache.labels.data()),
      thrust::raw_pointer_cast(cache.valid_indices.data()), device_costs,
      valid_count, cluster_count,
      thrust::raw_pointer_cast(cache.cluster_min_costs.data()),
      thrust::raw_pointer_cast(cache.cluster_counts.data()));
  gpu_kmeans_detail::computeSampleWeights<<<
      (valid_count + block - 1) / block, block>>>(
      thrust::raw_pointer_cast(cache.labels.data()),
      thrust::raw_pointer_cast(cache.valid_indices.data()), device_costs,
      thrust::raw_pointer_cast(cache.cluster_min_costs.data()), valid_count,
      gamma, thrust::raw_pointer_cast(cache.sample_weights.data()),
      thrust::raw_pointer_cast(cache.cluster_weight_sums.data()));
  const int output_count = cluster_count * dim_u * horizon;
  const int accumulation_count = valid_count * dim_u * horizon;
  gpu_kmeans_detail::accumulateWeightedClusterControls<<<
      (accumulation_count + block - 1) / block, block>>>(
      device_controls, thrust::raw_pointer_cast(cache.labels.data()),
      thrust::raw_pointer_cast(cache.valid_indices.data()),
      thrust::raw_pointer_cast(cache.sample_weights.data()), valid_count,
      dim_u, horizon,
      thrust::raw_pointer_cast(cache.clustered_controls.data()));
  gpu_kmeans_detail::normalizeWeightedClusterControls<<<
      (output_count + block - 1) / block, block>>>(
      thrust::raw_pointer_cast(cache.cluster_weight_sums.data()),
      dim_u * horizon, output_count,
      thrust::raw_pointer_cast(cache.clustered_controls.data()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto after_reduction = std::chrono::high_resolution_clock::now();

  const thrust::host_vector<int> counts = cache.cluster_counts;
  const thrust::host_vector<double> clustered_controls =
      cache.clustered_controls;
  const int control_size = dim_u * horizon;
  for (int cluster = 0; cluster < cluster_count; ++cluster) {
    if (counts[cluster] <= 0) continue;
    active_clusters.push_back(cluster);
    const auto first = clustered_controls.begin() + cluster * control_size;
    host_clustered_controls.insert(host_clustered_controls.end(), first,
                                    first + control_size);
  }
  if (profile) {
    const auto finish = std::chrono::high_resolution_clock::now();
    const auto seconds = [](auto first, auto last) {
      return std::chrono::duration<double>(last - first).count();
    };
    std::cerr << "[gpu-kmeans-profile] filter="
              << seconds(profile_start, after_filter)
              << " feature_init=" << seconds(after_filter, after_features)
              << " kmeans=" << seconds(after_features, after_kmeans)
              << " weighted=" << seconds(after_kmeans, after_reduction)
              << " d2h=" << seconds(after_reduction, finish)
              << " valid=" << valid_count << " k=" << cluster_count << "\n";
  }
}

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
