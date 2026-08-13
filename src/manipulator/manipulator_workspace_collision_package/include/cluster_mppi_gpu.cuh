#pragma once

#include "mppi_gpu.cuh"

#include <deque>
#include <numeric>

class ClusterMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass> ClusterMPPI_GPU(ModelClass model);
  ~ClusterMPPI_GPU() override;

  void setClusterParams(double deviation_mu_value, double epsilon_value,
                        int minpts_value) {
    deviation_mu = deviation_mu_value;
    epsilon = epsilon_value;
    minpts = minpts_value;
  }

  Eigen::MatrixXd U;
  Eigen::MatrixXd X;

  void solve() override;
  void init(MPPIParam param);

  void dbscan(std::vector<std::vector<int>> &clusters,
              const Eigen::MatrixXd &feature_source,
              const Eigen::VectorXd &costs, int N_samples, int T_steps);

  void kmeansCluster(std::vector<std::vector<int>> &clusters,
                     const Eigen::MatrixXd &feature_source,
                     const Eigen::VectorXd &costs, int N_samples,
                     int T_steps);

  void calculateU(Eigen::MatrixXd &Uout,
                  const std::vector<std::vector<int>> &clusters,
                  const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui_cpu,
                  int T_steps);

private:
  double deviation_mu = 1.0;
  double epsilon = 3.8;
  int minpts = 8;
  ClusteringMethod clustering_method = ClusteringMethod::DBSCAN;
  int kmeans_clusters = 5;
  int kmeans_max_iterations = 100;
  double kmeans_threshold = 1e-6;
};

template <typename ModelClass>
ClusterMPPI_GPU::ClusterMPPI_GPU(ModelClass model) : MPPI_GPU(model) {}
