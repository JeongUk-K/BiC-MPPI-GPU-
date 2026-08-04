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
#include <map>
#include <numeric>
#include <typeinfo>
#include <vector>

// ============================================================
// BiMPPI_GPU_Legacy — original bidirectional MPPI implementation
//
// Forward/backward rollout + guide MPPI on GPU.
// DBSCAN, selectConnection, concatenate stay on CPU.
// ============================================================
class BiMPPI_GPU_Legacy {
public:
  template <typename ModelClass> BiMPPI_GPU_Legacy(ModelClass model);
  ~BiMPPI_GPU_Legacy();

  void init(BiMPPIParam param);
  void setCollisionChecker(CollisionChecker *cc);
  void setSeed(std::uint_fast64_t seed);
  void solve();
  void move();
  double connectionDistance() const;
  void guideReference(const Eigen::MatrixXd &Uref,
                      const Eigen::MatrixXd &Xref);
  void setVisLogger(MPPIVisLogger *logger) { vis_logger = logger; }
  void setClusteringMethod(ClusteringMethod method) {
    clustering_method = method;
  }
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
  double *d_map, *d_circles, *d_rects, *d_ws_boxes;
  int map_max_row, map_max_col, n_circles, n_rects, n_ws_boxes;
  double map_resolution;
  double ws_link_radius, ws_safe_margin, ws_hard_margin;
  bool with_map;
  curandGenerator_t curand_gen;

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
  void backwardRawRollout(Eigen::VectorXd &costs, Eigen::MatrixXd &Ui_cpu);
  void forwardRawRollout(Eigen::VectorXd &costs, Eigen::MatrixXd &Ui_cpu);
  void appendVisRolloutSamples(const Eigen::MatrixXd &Ui_cpu, int N_samples,
                               int T_steps, bool backward);
  void selectConnection();
  void concatenate();
  void guideMPPI();
  void guideMPPIDevice();
  void guideMPPIImpl(bool device_reduction);
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
};

template <typename ModelClass> BiMPPI_GPU_Legacy::BiMPPI_GPU_Legacy(ModelClass model) {
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
  d_map = d_circles = d_rects = d_ws_boxes = nullptr;
  n_circles = n_rects = n_ws_boxes = 0;
  ws_link_radius = 0.045;
  ws_safe_margin = 0.10;
  ws_hard_margin = 0.0;
  with_map = false;
  alloc_Nf = alloc_Nb = alloc_Tf = alloc_Tb = 0;
  alloc_Tr_guide = 0;
  curand_gen = nullptr;
}

inline void BiMPPI_GPU_Legacy::setSeed(std::uint_fast64_t seed) {
  if (!curand_gen) {
    CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  }
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, static_cast<unsigned long long>(seed)));
}
