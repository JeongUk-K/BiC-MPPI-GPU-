#pragma once

#include "model_base.h"

#include <algorithm>
#include <cmath>

class HybridParking : public ModelBase {
public:
  HybridParking();
  ~HybridParking();

  double L;
};

namespace {
inline double wrapParkingAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}
} // namespace

HybridParking::HybridParking() {
  // x = [px, py, theta, forward_distance, reverse_distance]
  dim_x = 5;

  // u = [signed_speed, steering_angle]
  dim_u = 2;

  L = 0.28;

  f = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &u) -> Eigen::MatrixXd {
    Eigen::VectorXd x_dot = Eigen::VectorXd::Zero(x.rows());
    const double v = u(0);
    const double delta = u(1);

    x_dot(0) = v * std::cos(x(2));
    x_dot(1) = v * std::sin(x(2));
    x_dot(2) = v * std::tan(delta) / L;
    x_dot(3) = std::max(0.0, v);
    x_dot(4) = std::max(0.0, -v);
    return x_dot;
  };

  q = [](const Eigen::VectorXd &, const Eigen::VectorXd &u) -> double {
    return 0.02 * u(0) * u(0) + 0.01 * u(1) * u(1);
  };

  p = [](const Eigen::VectorXd &x,
         const Eigen::VectorXd &x_target) -> double {
    const double pos_err = (x.head(2) - x_target.head(2)).norm();
    const double yaw_err = std::abs(wrapParkingAngle(x(2) - x_target(2)));
    const double missing_forward = std::max(0.0, x_target(3) - x(3));
    const double missing_reverse = std::max(0.0, x_target(4) - x(4));

    return 5.0 * pos_err + 1.8 * yaw_err + 2.5 * missing_forward +
           2.5 * missing_reverse;
  };

  h = [](Eigen::Ref<Eigen::MatrixXd> U) -> void {
    U.row(0) = U.row(0).cwiseMax(-0.8).cwiseMin(0.8);

    const double max_steer = 0.7;
    U.row(1) = U.row(1).cwiseMax(-max_steer).cwiseMin(max_steer);
  };
}

HybridParking::~HybridParking() {}
