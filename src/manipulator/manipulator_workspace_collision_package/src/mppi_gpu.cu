#include "mppi_gpu.cuh"
#include "rollout_ee_export.cuh"
#include "rollout_state_export.cuh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

DEFINE_FORWARD_ROLLOUT_KERNEL(mppi_rollout_kernel)
DEFINE_WEIGHTED_SUM_KERNEL(mppi_weighted_sum_kernel)

static void safe_cuda_free(double *&p) {
  if (p) {
    cudaFree(p);
    p = nullptr;
  }
}

MPPI_GPU::~MPPI_GPU() {
  freeGPU();
  if (curand_gen) {
    curandDestroyGenerator(curand_gen);
    curand_gen = nullptr;
  }
}

void MPPI_GPU::init(MPPIParam param) {
  dt = param.dt;
  T = param.T;
  N = param.N;
  gamma_u = param.gamma_u;
  x_init = param.x_init;
  x_target = param.x_target;

  sigma_diag.resize(dim_u);
  for (int d = 0; d < dim_u; ++d) {
    sigma_diag[d] = param.sigma_u(d, d);
  }

  u0 = Eigen::VectorXd::Zero(dim_u);
  Uo = Eigen::MatrixXd::Zero(dim_u, T);
  Xo = Eigen::MatrixXd::Zero(dim_x, T + 1);

  allocGPU();
}

void MPPI_GPU::setCollisionChecker(CollisionChecker *cc) {
  collision_checker = cc;
  uploadCollisionData();
}

void MPPI_GPU::allocGPU() {
  const size_t sz_U = static_cast<size_t>(dim_u) * T * sizeof(double);
  size_t noise_count = static_cast<size_t>(N) * dim_u * T;
  const size_t noise_count_even = (noise_count % 2 != 0) ? noise_count + 1 : noise_count;
  const size_t sz_Ui = noise_count * sizeof(double);
  const size_t sz_noise = noise_count_even * sizeof(double);
  const size_t sz_N = static_cast<size_t>(N) * sizeof(double);
  const size_t sz_Di = static_cast<size_t>(N) * dim_u * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_U0, sz_U));
  CUDA_CHECK(cudaMalloc(&d_Ui, sz_Ui));
  CUDA_CHECK(cudaMalloc(&d_noise, sz_noise));
  CUDA_CHECK(cudaMalloc(&d_costs, sz_N));
  CUDA_CHECK(cudaMalloc(&d_Uo, sz_U));
  CUDA_CHECK(cudaMalloc(&d_Di, sz_Di));
  CUDA_CHECK(cudaMalloc(&d_x_init, dim_x * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_x_target, dim_x * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sigma, dim_u * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_sigma, sigma_diag.data(), dim_u * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void MPPI_GPU::freeGPU() {
  safe_cuda_free(d_U0);
  safe_cuda_free(d_Ui);
  safe_cuda_free(d_noise);
  safe_cuda_free(d_costs);
  safe_cuda_free(d_Uo);
  safe_cuda_free(d_Di);
  safe_cuda_free(d_x_init);
  safe_cuda_free(d_x_target);
  safe_cuda_free(d_sigma);
  safe_cuda_free(d_map);
  safe_cuda_free(d_circles);
  safe_cuda_free(d_rects);
  safe_cuda_free(d_ws_boxes);
}

void MPPI_GPU::uploadCollisionData() {
  safe_cuda_free(d_map);
  safe_cuda_free(d_circles);
  safe_cuda_free(d_rects);
  safe_cuda_free(d_ws_boxes);
  with_map = false;
  map_max_row = 0;
  map_max_col = 0;
  n_circles = 0;
  n_rects = 0;
  n_ws_boxes = 0;

  if (!collision_checker) {
    return;
  }

  with_map = collision_checker->with_map;
  map_resolution = collision_checker->resolution;
  map_max_row = static_cast<int>(collision_checker->map.size());
  map_max_col = map_max_row > 0 ? static_cast<int>(collision_checker->map[0].size()) : 0;

  if (with_map && map_max_row > 0 && map_max_col > 0) {
    const size_t map_sz = static_cast<size_t>(map_max_row) * map_max_col * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_map, map_sz));
    std::vector<double> flat(static_cast<size_t>(map_max_row) * map_max_col);
    for (int r = 0; r < map_max_row; ++r) {
      for (int c = 0; c < map_max_col; ++c) {
        flat[static_cast<size_t>(r) * map_max_col + c] = collision_checker->map[r][c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_map, flat.data(), map_sz, cudaMemcpyHostToDevice));
  }

  n_circles = static_cast<int>(collision_checker->circles.size());
  if (n_circles > 0) {
    const size_t sz = static_cast<size_t>(n_circles) * 4 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_circles, sz));
    std::vector<double> cbuf(static_cast<size_t>(n_circles) * 4);
    for (int i = 0; i < n_circles; ++i) {
      for (int j = 0; j < 4; ++j) {
        cbuf[static_cast<size_t>(i) * 4 + j] = collision_checker->circles[i][j];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_circles, cbuf.data(), sz, cudaMemcpyHostToDevice));
  }

  n_rects = static_cast<int>(collision_checker->rectangles.size());
  if (n_rects > 0) {
    const size_t sz = static_cast<size_t>(n_rects) * 4 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_rects, sz));
    std::vector<double> rbuf(static_cast<size_t>(n_rects) * 4);
    for (int i = 0; i < n_rects; ++i) {
      for (int j = 0; j < 4; ++j) {
        rbuf[static_cast<size_t>(i) * 4 + j] = collision_checker->rectangles[i][j];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_rects, rbuf.data(), sz, cudaMemcpyHostToDevice));
  }

  n_ws_boxes = static_cast<int>(collision_checker->workspace_boxes.size());
  ws_link_radius = collision_checker->link_radius;
  ws_safe_margin = collision_checker->workspace_safe_margin;
  ws_hard_margin = collision_checker->workspace_hard_margin;
  if (collision_checker->use_workspace_link_collision && n_ws_boxes > 0) {
    const size_t sz = static_cast<size_t>(n_ws_boxes) * 6 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_ws_boxes, sz));
    std::vector<double> buf(static_cast<size_t>(n_ws_boxes) * 6);
    for (int i = 0; i < n_ws_boxes; ++i) {
      for (int j = 0; j < 6; ++j) {
        buf[static_cast<size_t>(i) * 6 + j] = collision_checker->workspace_boxes[i][j];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_ws_boxes, buf.data(), sz, cudaMemcpyHostToDevice));
  } else {
    n_ws_boxes = 0;
  }
}

void MPPI_GPU::uploadControl() {
  std::vector<double> u0_flat(static_cast<size_t>(dim_u) * T);
  for (int d = 0; d < dim_u; ++d) {
    for (int t = 0; t < T; ++t) {
      u0_flat[static_cast<size_t>(d) * T + t] = U_0(d, t);
    }
  }
  CUDA_CHECK(cudaMemcpy(d_U0, u0_flat.data(),
                        static_cast<size_t>(dim_u) * T * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void MPPI_GPU::uploadState() {
  CUDA_CHECK(cudaMemcpy(d_x_init, x_init.data(), dim_x * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x_target, x_target.data(), dim_x * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void MPPI_GPU::generateNoise() {
  if (!curand_gen) {
    CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
        curand_gen, static_cast<unsigned long long>(std::time(nullptr))));
  }
  size_t count = static_cast<size_t>(N) * dim_u * T;
  if (count % 2 != 0) {
    ++count;
  }
  CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_noise, count, 0.0, 1.0));
}

void MPPI_GPU::solve() {
  start = std::chrono::high_resolution_clock::now();

  uploadControl();
  uploadState();
  generateNoise();

  const int block = 256;
  const int grid = (N + block - 1) / block;
  mppi_rollout_kernel<<<grid, block>>>(
      d_U0, d_Ui, d_noise, d_sigma, d_x_init, d_x_target, d_costs, d_Di,
      with_map, d_map, map_max_row, map_max_col, map_resolution, d_circles,
      n_circles, d_rects, n_rects, d_ws_boxes, n_ws_boxes, ws_link_radius,
      ws_safe_margin, ws_hard_margin, N, dim_u, dim_x, T, static_cast<double>(dt),
      gamma_u, model_type, false);
  CUDA_CHECK(cudaGetLastError());

  rollout_ee_export::emit(rollout_ee_callback, "forward", d_Ui, d_x_init, N,
                          dim_u, dim_x, T, static_cast<double>(dt), model_type);
  rollout_state_export::emit(rollout_state_callback, "forward", d_Ui,
                             d_x_init, N, dim_u, dim_x, T,
                             static_cast<double>(dt), model_type);

  std::vector<double> h_costs(N);
  CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, static_cast<size_t>(N) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  const double min_cost = *std::min_element(h_costs.begin(), h_costs.end());

  const int dt_blocks = dim_u * T;
  const int wsum_threads = std::min(256, N);
  const size_t shared_sz = 2 * wsum_threads * sizeof(double);
  mppi_weighted_sum_kernel<<<dt_blocks, wsum_threads, shared_sz>>>(
      d_Ui, d_costs, min_cost, gamma_u, d_Uo, N, dim_u, T);
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> h_Uo(static_cast<size_t>(dim_u) * T);
  CUDA_CHECK(cudaMemcpy(h_Uo.data(), d_Uo,
                        static_cast<size_t>(dim_u) * T * sizeof(double),
                        cudaMemcpyDeviceToHost));

  Uo.resize(dim_u, T);
  for (int d = 0; d < dim_u; ++d) {
    for (int t = 0; t < T; ++t) {
      Uo(d, t) = h_Uo[static_cast<size_t>(d) * T + t];
    }
  }
  h(Uo);

  CUDA_CHECK(cudaDeviceSynchronize());
  finish = std::chrono::high_resolution_clock::now();
  elapsed_1 = finish - start;
  elapsed_rollout = elapsed_1.count();
  elapsed_clustering = 0.0;
  elapsed_connection = 0.0;
  elapsed_guide = 0.0;
  elapsed = elapsed_rollout;

  u0 = Uo.col(0);

  Xo.col(0) = x_init;
  for (int t = 0; t < T; ++t) {
    Xo.col(t + 1) = Xo.col(t) + static_cast<double>(dt) * f(Xo.col(t), Uo.col(t));
  }

  visual_traj.push_back(x_init);
}

void MPPI_GPU::move() {
  x_init = x_init + static_cast<double>(dt) * f(x_init, u0);
  if (T > 1 && Uo.cols() >= T) {
    U_0.leftCols(T - 1) = Uo.rightCols(T - 1);
    U_0.col(T - 1).setZero();
  }
}
