#pragma once

#include "mppi_gpu.cuh"

// ============================================================
// LogMPPI_GPU — Log-MPPI solver with CUDA GPU rollout
//
// Inherits MPPI_GPU and mirrors the legacy CPU LogMPPI class:
// MPPI::solve() is reused, and only noise generation is changed to
//   η(d,t) * exp(ξ(d,t))
// where η ~ N(0,1), ξ ~ N(0,1). The base MPPI rollout then applies σ_d.
//
// Interface mirrors LogMPPI (CPU). Replace:
//   #include <log_mppi.h>       →  #include <log_mppi_gpu.cuh>
//   using Solver = LogMPPI;     →  using Solver = LogMPPI_GPU;
// ============================================================
class LogMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass>
  LogMPPI_GPU(ModelClass model);
  ~LogMPPI_GPU();

protected:
  // Additional lognormal buffer: same shape as d_noise [N * dim_u * T]
  double *d_log_noise;  // raw N(0,1) used as log-normal exponent
  size_t d_log_noise_capacity;

  void generateNoise() override;
};

// ---- Template constructor ----
template <typename ModelClass>
LogMPPI_GPU::LogMPPI_GPU(ModelClass model) : MPPI_GPU(model) {
  d_log_noise = nullptr;
  d_log_noise_capacity = 0;
}
