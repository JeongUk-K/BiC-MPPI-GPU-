#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

enum class ClusteringMethod {
  DBSCAN,
  KMeans,
};

inline ClusteringMethod parseClusteringMethod(const std::string &value) {
  if (value == "dbscan") return ClusteringMethod::DBSCAN;
  if (value == "kmeans") return ClusteringMethod::KMeans;
  throw std::invalid_argument(
      "clustering method must be either 'dbscan' or 'kmeans'");
}

inline const char *clusteringMethodName(ClusteringMethod method) {
  return method == ClusteringMethod::KMeans ? "kmeans" : "dbscan";
}

struct MPPIParam {
  float dt = 0.02f;
  int T = 200;
  int N = 4096;

  // Inverse temperature for the MPPI exponential weight.
  double gamma_u = 0.001;

  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd sigma_u;
  ClusteringMethod clustering_method = ClusteringMethod::DBSCAN;
  int kmeans_clusters = 5;
  int kmeans_max_iterations = 100;
  double kmeans_threshold = 1e-6;
};

struct BiMPPIParam {
  float dt = 0.02f;
  int Tf = 100;
  int Tb = 100;
  int Nf = 4096;
  int Nb = 4096;
  int Nr = 4096;

  // Inverse temperature for the MPPI exponential weight.
  double gamma_u = 0.001;

  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd sigma_u;

  // DBSCAN feature scaling parameters used by the uploaded BiMPPI_GPU.
  double deviation_mu = 1.0;
  double cost_mu = 1.0;
  double epsilon = 0.5;
  int minpts = 8;

  // Reserved for compatibility with existing code.
  double psi = 0.0;
  ClusteringMethod clustering_method = ClusteringMethod::DBSCAN;
  int kmeans_clusters = 5;
  int kmeans_max_iterations = 100;
  double kmeans_threshold = 1e-6;
};
