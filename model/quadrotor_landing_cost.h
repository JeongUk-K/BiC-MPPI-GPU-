#pragma once

#include <cmath>

#ifdef __CUDACC__
#define QUADROTOR_COST_HD __host__ __device__ __forceinline__
#else
#define QUADROTOR_COST_HD inline
#endif

namespace quadrotor_landing_cost {

constexpr double kPrecisionXyTolerance = 0.30;
constexpr double kPrecisionVxyTolerance = 0.50;
constexpr double kPrecisionVzTolerance = 1.00;
constexpr double kVxyWeight = kPrecisionXyTolerance / kPrecisionVxyTolerance;
constexpr double kVzWeight = kPrecisionXyTolerance / kPrecisionVzTolerance;

QUADROTOR_COST_HD double terminalCost(const double *x, const double *x_target) {
  const double dx = x[0] - x_target[0];
  const double dy = x[1] - x_target[1];
  const double dz = x[2] - x_target[2];
  const double position_error = ::sqrt(dx * dx + dy * dy + dz * dz);

  const double dvx = x[3] - x_target[3];
  const double dvy = x[4] - x_target[4];
  const double dvz = x[5] - x_target[5];
  const double vxy_error = ::sqrt(dvx * dvx + dvy * dvy);
  const double vz_error = ::fabs(dvz);

  const double landing_gate = 1.0 / (1.0 + position_error);
  return position_error +
         landing_gate * (kVxyWeight * vxy_error + kVzWeight * vz_error);
}

} // namespace quadrotor_landing_cost

#undef QUADROTOR_COST_HD
