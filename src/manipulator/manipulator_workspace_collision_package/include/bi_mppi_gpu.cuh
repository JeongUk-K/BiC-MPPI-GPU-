#pragma once

#include "bi_mppi_gpu_legacy.cuh"

// Optimized BiC-MPPI. The public interface intentionally matches the original
// solver, while the K-means path keeps sampled controls, feature construction,
// clustering, and cluster-weighted control reduction on the GPU.
class BiMPPI_GPU : public BiMPPI_GPU_Legacy {
public:
  template <typename ModelClass>
  explicit BiMPPI_GPU(ModelClass model) : BiMPPI_GPU_Legacy(model) {}

  void solve();

private:
  void backwardRolloutDevice();
  void forwardRolloutDevice();
  void clusterControlsDevice(const double *device_controls,
                             const double *device_costs, int sample_count,
                             int horizon, Eigen::MatrixXd &clustered_controls,
                             std::vector<std::vector<int>> &clusters);
  void appendDeviceVisRolloutSamples(const double *device_controls,
                                     int sample_count, int horizon,
                                     bool backward);
};
