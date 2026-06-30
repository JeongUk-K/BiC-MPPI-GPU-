#pragma once

#include "mppi_gpu.cuh"

class LogMPPI_GPU : public MPPI_GPU {
public:
  template <typename ModelClass> LogMPPI_GPU(ModelClass model);
  ~LogMPPI_GPU() override = default;

  // Deprecated compatibility hook. Corrected Log-MPPI no longer applies
  // log-normal input-noise scaling; sampling is identical to MPPI.
  void setLogNoiseParams(double, double) {}

  void solve() override;
};

template <typename ModelClass>
LogMPPI_GPU::LogMPPI_GPU(ModelClass model) : MPPI_GPU(model) {}
