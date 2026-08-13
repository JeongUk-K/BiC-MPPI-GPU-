#pragma once

#include "mppi_gpu.cuh"

// ============================================================
// LogMPPI_GPU — Log-MPPI solver with CUDA GPU rollout
//
// Uses the same Gaussian rollout distribution as MPPI_GPU and differs only
// by normalizing trajectory weights with explicit log-sum-exp.
//
// Interface mirrors LogMPPI (CPU). Replace:
//   #include <log_mppi.h>       →  #include <log_mppi_gpu.cuh>
//   using Solver = LogMPPI;     →  using Solver = LogMPPI_GPU;
// ============================================================
class LogMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass>
  LogMPPI_GPU(ModelClass model);
  ~LogMPPI_GPU() override = default;

protected:
  void weightedControlSum(const std::vector<double> &costs,
                          Eigen::MatrixXd &Uo_out) override;
};

// ---- Template constructor ----
template <typename ModelClass>
LogMPPI_GPU::LogMPPI_GPU(ModelClass model) : MPPI_GPU(model) {}
