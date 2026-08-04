#pragma once

#include "collision_checker.h"
#include "cuda_utils.cuh"
#include "model_base.h"
#include "mppi_param.h"
#include "mppi_vis_logger.h"
#include "rollout_ee_callback.h"
#include "rollout_state_callback.h"

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <functional>
#include <typeinfo>
#include <vector>

// ============================================================
// MPPI_GPU — workspace-aware CUDA MPPI solver
// ============================================================
class MPPI_GPU {
public:
  template <typename ModelClass> MPPI_GPU(ModelClass model);
  virtual ~MPPI_GPU();

  void init(MPPIParam param);
  void setCollisionChecker(CollisionChecker *cc);
  void setSeed(std::uint_fast64_t seed);
  virtual void solve();
  void move();
  double connectionDistance() const { return 0.0; }
  void setVisLogger(MPPIVisLogger *logger) { vis_logger = logger; }
  void setRolloutEECallback(RolloutEEBatchCallback callback) {
    rollout_ee_callback = std::move(callback);
  }
  void setRolloutStateCallback(RolloutStateBatchCallback callback) {
    rollout_state_callback = std::move(callback);
  }

  Eigen::MatrixXd U_0;
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::MatrixXd Uo;
  Eigen::MatrixXd Xo;
  Eigen::VectorXd u0;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
  std::chrono::duration<double> elapsed_1;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;

protected:
  int dim_x = 0;
  int dim_u = 0;
  float dt = 0.02f;
  int T = 0;
  int N = 0;
  double gamma_u = 0.001;
  std::vector<double> sigma_diag;

  CollisionChecker *collision_checker = nullptr;
  MPPIVisLogger *vis_logger = nullptr;
  RolloutEEBatchCallback rollout_ee_callback;
  RolloutStateBatchCallback rollout_state_callback;
  std::vector<Eigen::VectorXd> visual_traj;

  double *d_U0 = nullptr;
  double *d_Ui = nullptr;
  double *d_noise = nullptr;
  double *d_costs = nullptr;
  double *d_Uo = nullptr;
  double *d_Di = nullptr;
  double *d_x_init = nullptr;
  double *d_x_target = nullptr;
  double *d_sigma = nullptr;

  double *d_map = nullptr;
  int map_max_row = 0;
  int map_max_col = 0;
  double map_resolution = 0.05;
  bool with_map = false;
  double *d_circles = nullptr;
  int n_circles = 0;
  double *d_rects = nullptr;
  int n_rects = 0;
  double *d_ws_boxes = nullptr;
  int n_ws_boxes = 0;
  double ws_link_radius = 0.045;
  double ws_safe_margin = 0.10;
  double ws_hard_margin = 0.0;

  curandGenerator_t curand_gen;
  int model_type = LEGACY_CUDA_WMROBOT;

  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
  std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

  void allocGPU();
  void freeGPU();
  void uploadCollisionData();
  void uploadControl();
  void uploadState();
  virtual void generateNoise();
};

template <typename ModelClass> MPPI_GPU::MPPI_GPU(ModelClass model) {
  dim_x = model.dim_x;
  dim_u = model.dim_u;
  this->f = model.f;
  this->q = model.q;
  this->p = model.p;
  this->h = model.h;
  model_type =
      legacy_cuda_model_type_from_name(typeid(ModelClass).name(), dim_x, dim_u);

  d_U0 = d_Ui = d_noise = d_costs = d_Uo = d_Di = nullptr;
  d_x_init = d_x_target = d_sigma = nullptr;
  d_map = d_circles = d_rects = d_ws_boxes = nullptr;
  n_circles = n_rects = n_ws_boxes = 0;
  with_map = false;
  collision_checker = nullptr;
  curand_gen = nullptr;
}

inline void MPPI_GPU::setSeed(std::uint_fast64_t seed) {
  if (!curand_gen) {
    CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  }
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
      curand_gen, static_cast<unsigned long long>(seed)));
}

// =============================================================================
// Workspace-aware forward rollout kernel macro.
// -----------------------------------------------------------------------------
// This is a drop-in replacement for the forward rollout macro used by
// bi_mppi_gpu.cu. It keeps the original MPPI sampling/projection layout but
// extends collision and stage cost calls with workspace link-collision buffers.
// =============================================================================
#undef DEFINE_FORWARD_ROLLOUT_KERNEL
#define DEFINE_FORWARD_ROLLOUT_KERNEL(NAME)                                                \
__global__ void NAME(                                                                       \
    const double* __restrict__ d_U0,                                                        \
    double* d_Ui,                                                                           \
    const double* __restrict__ d_noise,                                                     \
    const double* __restrict__ d_sigma,                                                     \
    const double* __restrict__ d_x_init,                                                    \
    const double* __restrict__ d_x_target,                                                  \
    double* d_costs, double* d_Di,                                                          \
    bool with_map, const double* d_map,                                                     \
    int max_row, int max_col, double res,                                                   \
    const double* d_circles, int n_circ,                                                    \
    const double* d_rects, int n_rect,                                                      \
    const double* d_ws_boxes, int n_ws_boxes,                                               \
    double ws_link_radius, double ws_safe_margin, double ws_hard_margin,                    \
    int N, int dim_u, int dim_x, int T, double dt, double gamma_u,                          \
    int model_type, bool forward) {                                                         \
  (void)gamma_u;                                                                            \
  (void)forward;                                                                            \
  const int i = blockIdx.x * blockDim.x + threadIdx.x;                                      \
  if (i >= N) return;                                                                       \
  double* Ui_i = d_Ui + i * (dim_u * T);                                                    \
  for (int d = 0; d < dim_u; ++d) {                                                         \
    for (int t = 0; t < T; ++t) {                                                           \
      const double raw = d_U0[d * T + t] +                                                  \
                         d_sigma[d] * d_noise[i * (dim_u * T) + d * T + t];                 \
      Ui_i[d * T + t] = raw;                                                                \
    }                                                                                       \
  }                                                                                         \
  legacy_cuda_project_control(Ui_i, dim_u, T, model_type);                                  \
  double* Di_i = d_Di + i * dim_u;                                                          \
  for (int d = 0; d < dim_u; ++d) {                                                         \
    double acc = 0.0;                                                                       \
    for (int t = 0; t < T; ++t) acc += Ui_i[d * T + t] - d_U0[d * T + t];                    \
    Di_i[d] = acc / static_cast<double>(T);                                                  \
  }                                                                                         \
  double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd[GPU_MAX_DIM_X];                            \
  for (int d = 0; d < dim_x; ++d) x[d] = d_x_init[d];                                       \
  double cost = 0.0;                                                                        \
  bool hit = legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,          \
                                        d_circles, n_circ, d_rects, n_rect,                 \
                                        d_ws_boxes, n_ws_boxes, ws_link_radius,             \
                                        ws_safe_margin, ws_hard_margin, model_type);        \
  if (hit) cost = 1.0e8;                                                                    \
  for (int t = 0; t < T && !hit; ++t) {                                                     \
    double u_local[GPU_MAX_DIM_U];                                                          \
    for (int d = 0; d < dim_u; ++d) u_local[d] = Ui_i[d * T + t];                            \
    cost += legacy_cuda_stage_cost(x, u_local, dim_x, dim_u, model_type,                    \
                                   d_ws_boxes, n_ws_boxes, ws_link_radius,                  \
                                   ws_safe_margin, ws_hard_margin);                         \
    cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);                    \
    legacy_cuda_dynamics(x, u_local, xd, dim_x, dim_u, model_type);                         \
    for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt * xd[d];                              \
    for (int d = 0; d < dim_x; ++d) x[d] = xn[d];                                           \
    hit = legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,             \
                                     d_circles, n_circ, d_rects, n_rect,                    \
                                     d_ws_boxes, n_ws_boxes, ws_link_radius,                \
                                     ws_safe_margin, ws_hard_margin, model_type);           \
    if (hit) cost = 1.0e8;                                                                  \
  }                                                                                         \
  if (!hit) cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);            \
  d_costs[i] = cost;                                                                        \
}
