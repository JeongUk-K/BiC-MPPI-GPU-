#pragma once

#include "mppi_gpu.cuh"

#include <deque>
#include <numeric>

// ============================================================
// ClusterMPPI_GPU — GPU rollout + selectable DBSCAN/GPU K-means clustering
//
// Interface mirrors the CPU ClusterMPPI class.
// ============================================================
class ClusterMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass> ClusterMPPI_GPU(ModelClass model);
  ~ClusterMPPI_GPU();

  Eigen::MatrixXd U;  // dim_u x (clusters * T)  cluster controls
  Eigen::MatrixXd X;

  void solve() override;
  void init(MPPIParam param);

  void dbscan(std::vector<std::vector<int>> &clusters,
              const Eigen::MatrixXd &feature_source, const Eigen::VectorXd &costs,
              int N_samples, int T_steps);

  void kmeansCluster(std::vector<std::vector<int>> &clusters,
                     const Eigen::MatrixXd &feature_source,
                     const Eigen::VectorXd &costs, int N_samples,
                     int T_steps);

  void calculateU(Eigen::MatrixXd &Uout,
                  const std::vector<std::vector<int>> &clusters,
                  const Eigen::VectorXd &costs,
                  const Eigen::MatrixXd &Ui_cpu,
                  int T_steps);

private:
  double deviation_mu;
  double epsilon;
  int    minpts;
  ClusteringMethod clustering_method;
  int kmeans_clusters;
  int kmeans_max_iterations;
  double kmeans_threshold;
};

template <typename ModelClass>
ClusterMPPI_GPU::ClusterMPPI_GPU(ModelClass model) : MPPI_GPU(model) {
  deviation_mu = 1.0;
  epsilon      = 0.01;
  minpts       = 5;
  clustering_method = ClusteringMethod::DBSCAN;
  kmeans_clusters = 5;
  kmeans_max_iterations = 100;
  kmeans_threshold = 1e-6;
}
