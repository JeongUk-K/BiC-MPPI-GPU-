#pragma once

#include "collision_checker.h"
#include "cuda_utils.cuh"
#include "mppi_vis_logger.h"
#include "model_base.h"
#include "mppi_param.h"
#include "rollout_ee_callback.h"
#include "rollout_state_callback.h"

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <deque>
#include <future>
#include <map>
#include <limits>
#include <mutex>
#include <numeric>
#include <typeinfo>
#include <vector>

// ============================================================
// BiMPPI_GPU — Bidirectional MPPI with GPU rollout
//
// Forward/backward rollout + guide MPPI on GPU.
// Clustering is selectable between CPU DBSCAN and fastsc GPU K-means.
// ============================================================
class BiMPPI_GPU {
public:
  enum class ConnectionMetric {
    Euclidean,
    SE2,
  };

  template <typename ModelClass> BiMPPI_GPU(ModelClass model);
  ~BiMPPI_GPU();

  void init(BiMPPIParam param);
  void setCollisionChecker(CollisionChecker *cc);
  void setSeed(std::uint_fast64_t seed);
  void setConnectionMetric(ConnectionMetric metric);
  void setSE2ConnectionWeights(double xy_weight, double theta_weight);
  void solve();
  void move();
  double connectionDistance() const;
  void guideReference(const Eigen::MatrixXd &Uref,
                      const Eigen::MatrixXd &Xref);
  void setVisLogger(MPPIVisLogger *logger) { vis_logger = logger; }
  void setRolloutEECallback(RolloutEEBatchCallback callback) {
    rollout_ee_callback = std::move(callback);
  }
  void setRolloutStateCallback(RolloutStateBatchCallback callback) {
    rollout_state_callback = std::move(callback);
  }

  // ---- Public state (mirrors CPU BiMPPI) ----
  Eigen::MatrixXd U_f0; // dim_u x Tf
  Eigen::MatrixXd U_b0; // dim_u x Tb
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::VectorXd dummy_u;
  Eigen::MatrixXd Uo;
  Eigen::MatrixXd Xo;
  Eigen::VectorXd u0;
  double cost = std::numeric_limits<double>::quiet_NaN();

  double trajectoryCost() const { return cost; }

  // Timing
  std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
  std::chrono::duration<double> elapsed_1, elapsed_2, elapsed_3, elapsed_4;
  double elapsed, elapsed_rollout, elapsed_clustering, elapsed_connection,
      elapsed_guide;
  std::vector<Eigen::VectorXd> visual_traj;

protected:
  int dim_x, dim_u;
  float dt;
  int Tf, Tb, Nf, Nb, Nr;
  double gamma_u;
  std::vector<double> sigma_diag;
  double deviation_mu, cost_mu, epsilon;
  int minpts;
  double psi;
  ClusteringMethod clustering_method = ClusteringMethod::DBSCAN;
  int kmeans_clusters = 5;
  int kmeans_max_iterations = 100;
  double kmeans_threshold = 1e-6;
  ConnectionMetric connection_metric = ConnectionMetric::Euclidean;
  double se2_connection_xy_weight = 0.2;
  double se2_connection_theta_weight = 1.0;

  CollisionChecker *collision_checker;
  MPPIVisLogger *vis_logger = nullptr;
  RolloutEEBatchCallback rollout_ee_callback;
  RolloutStateBatchCallback rollout_state_callback;

  // CPU cluster data
  std::vector<std::vector<int>> clusters_f, clusters_b;
  std::vector<int> full_cluster_f, full_cluster_b;
  Eigen::MatrixXd Uf, Ub, Xf, Xb;
  std::vector<std::vector<int>> joints;
  std::vector<Eigen::MatrixXd> Xc, Uc;
  std::vector<Eigen::MatrixXd> Ur, Xr;
  std::vector<double> Cr;
  std::vector<Eigen::MatrixXd> vis_rollout_samples;
  static constexpr int kMaxSavedBiRollouts = 128;
  int vis_rollout_samples_per_call = 64;

  // GPU buffers (forward)
  double *d_Uf0, *d_Ufi, *d_noise_f, *d_costs_f, *d_Uf_out, *d_Di_f;
  // GPU buffers (backward)
  double *d_Ub0, *d_Ubi, *d_noise_b, *d_costs_b, *d_Ub_out, *d_Di_b;
  // GPU buffers (guide)
  double *d_Ur0, *d_Uri, *d_noise_r, *d_costs_r, *d_Ur_out, *d_Xref;

  double *d_x_init, *d_x_target, *d_sigma;

  // Collision
  double *d_map, *d_circles, *d_rects;
  int map_max_row, map_max_col, n_circles, n_rects;
  double map_resolution;
  bool with_map;
  curandGenerator_t curand_gen;
  curandGenerator_t curand_gen_forward;
  curandGenerator_t curand_gen_backward;
  cudaStream_t stream_forward;
  cudaStream_t stream_backward;
  mutable std::mutex branch_state_mutex;

  int alloc_Nf, alloc_Nb, alloc_Tf, alloc_Tb; // last allocated sizes
  int alloc_Tr_guide;

  // ---- Model-independent callbacks & type ----
  int model_type;
  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
  std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

  void allocForward();
  void allocBackward();
  void allocGuide();
  void allocGuideFor(int Tr);
  void freeForward();
  void freeBackward();
  void freeGuide();
  void freeCommon();
  void uploadCollisionData();

  void launchBackwardSampling();
  void launchForwardSampling();
  void backwardRollout();
  void forwardRollout();
  void clusterControlsDevice(const double *device_controls,
                             const double *device_costs, int sample_count,
                             int horizon, Eigen::MatrixXd &clustered_controls,
                             std::vector<std::vector<int>> &clusters,
                             cudaStream_t stream = 0);
  void appendDeviceVisRolloutSamples(const double *device_controls,
                                     int sample_count, int horizon,
                                     bool backward,
                                     cudaStream_t stream = 0);
  Eigen::MatrixXd reduceControlsDevice(const double *device_controls,
                                      const double *device_costs,
                                      int sample_count, int horizon);
  void backwardRawRollout(Eigen::VectorXd &costs, Eigen::MatrixXd &Ui_cpu);
  void forwardRawRollout(Eigen::VectorXd &costs, Eigen::MatrixXd &Ui_cpu);
  void appendVisRolloutSamples(const Eigen::MatrixXd &Ui_cpu, int N_samples,
                               int T_steps, bool backward);
  double connectionMetricDistance(
      const Eigen::Ref<const Eigen::VectorXd> &xf,
      const Eigen::Ref<const Eigen::VectorXd> &xb) const;
  void selectConnection();
  void concatenate();
  void guideMPPI();
  void partitioningControl();

  void dbscan(std::vector<std::vector<int>> &clusters,
              const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs,
              int N_samples);
  void kmeansCluster(std::vector<std::vector<int>> &clusters,
                     const Eigen::MatrixXd &feature_source,
                     const Eigen::VectorXd &costs, int N_samples);
  void calculateU(Eigen::MatrixXd &Uout,
                  const std::vector<std::vector<int>> &clusters,
                  const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui_cpu,
                  int T_steps);
  double evaluateTrajectoryCost(const Eigen::MatrixXd &trajectory,
                                const Eigen::MatrixXd &controls) const;
};

template <typename ModelClass> BiMPPI_GPU::BiMPPI_GPU(ModelClass model) {
  dim_x = model.dim_x;
  dim_u = model.dim_u;
  this->f = model.f;
  this->q = model.q;
  this->p = model.p;
  this->h = model.h;

  model_type =
      legacy_cuda_model_type_from_name(typeid(ModelClass).name(), dim_x, dim_u);

  d_Uf0 = d_Ufi = d_noise_f = d_costs_f = d_Uf_out = d_Di_f = nullptr;
  d_Ub0 = d_Ubi = d_noise_b = d_costs_b = d_Ub_out = d_Di_b = nullptr;
  d_Ur0 = d_Uri = d_noise_r = d_costs_r = d_Ur_out = d_Xref = nullptr;
  d_x_init = d_x_target = d_sigma = nullptr;
  d_map = d_circles = d_rects = nullptr;
  n_circles = n_rects = 0;
  with_map = false;
  alloc_Nf = alloc_Nb = alloc_Tf = alloc_Tb = 0;
  alloc_Tr_guide = 0;
  curand_gen = nullptr;
  curand_gen_forward = nullptr;
  curand_gen_backward = nullptr;
  stream_forward = nullptr;
  stream_backward = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_forward, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_backward, cudaStreamNonBlocking));
  CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandCreateGenerator(&curand_gen_forward,
                                     CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandCreateGenerator(&curand_gen_backward,
                                     CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetStream(curand_gen_forward, stream_forward));
  CURAND_CHECK(curandSetStream(curand_gen_backward, stream_backward));
  const auto seed = static_cast<unsigned long long>(std::time(nullptr));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, seed));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_gen_forward, seed));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_gen_backward, seed + 1));
}

inline void BiMPPI_GPU::setSeed(std::uint_fast64_t seed) {
  if (!curand_gen) {
    CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
  }
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, static_cast<unsigned long long>(seed)));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen_forward, static_cast<unsigned long long>(seed)));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen_backward, static_cast<unsigned long long>(seed + 1)));
}
