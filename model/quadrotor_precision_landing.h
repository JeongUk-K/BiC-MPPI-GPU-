#pragma once

#include "quadrotor.h"
#include "quadrotor_landing_cost.h"

class QuadrotorPrecisionLanding : public Quadrotor {
public:
  QuadrotorPrecisionLanding();
  ~QuadrotorPrecisionLanding();
};

QuadrotorPrecisionLanding::QuadrotorPrecisionLanding() : Quadrotor() {
  p = [this](const Eigen::VectorXd &x,
             const Eigen::VectorXd &x_target) -> double {
    return quadrotor_landing_cost::terminalCost(x.data(), x_target.data());
  };
}

QuadrotorPrecisionLanding::~QuadrotorPrecisionLanding() {}
