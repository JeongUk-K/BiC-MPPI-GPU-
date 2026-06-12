#include "log_mppi_gpu.cuh"

// Legacy CPU LogMPPI overrides getNoise() only:
//   (sigma_u * Normal(0, 1)).array() * Lognormal(0, 1).array()
// MPPI_GPU applies sigma inside the shared rollout kernel, so this override
// stores Normal(0, 1) * Lognormal(0, 1) in d_noise and reuses MPPI_GPU::solve().
__global__ void log_mppi_apply_lognormal_kernel(double *d_noise,
                                                const double *d_log_noise,
                                                size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  d_noise[idx] *= exp(d_log_noise[idx]);
}

LogMPPI_GPU::~LogMPPI_GPU() {
  if (d_log_noise) {
    cudaFree(d_log_noise);
    d_log_noise = nullptr;
  }
  d_log_noise_capacity = 0;
}

void LogMPPI_GPU::generateNoise() {
  const size_t count = static_cast<size_t>(N) * dim_u * T;
  const size_t count_even = (count % 2 != 0) ? count + 1 : count;

  CURAND_CHECK(
      curandGenerateNormalDouble(curand_gen, d_noise, count_even, 0.0, 1.0));

  if (!d_log_noise || d_log_noise_capacity < count_even) {
    if (d_log_noise) {
      CUDA_CHECK(cudaFree(d_log_noise));
    }
    CUDA_CHECK(cudaMalloc(&d_log_noise, count_even * sizeof(double)));
    d_log_noise_capacity = count_even;
  }

  CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_log_noise, count_even,
                                          0.0, 1.0));

  const int block = 256;
  const int grid = static_cast<int>((count + block - 1) / block);
  log_mppi_apply_lognormal_kernel<<<grid, block>>>(d_noise, d_log_noise, count);
  CUDA_CHECK(cudaGetLastError());
}
