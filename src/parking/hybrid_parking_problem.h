#pragma once

#include <collision_checker.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>

struct ParkingMetrics {
  double pose_error;
  double position_error;
  double heading_error;
  double forward_distance;
  double reverse_distance;
  double missing_forward;
  double missing_reverse;
};

inline double wrapHybridParkingAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

inline Eigen::VectorXd hybridParkingStart() {
  Eigen::VectorXd x(5);
  x << 0.55, 0.65, 0.0, 0.0, 0.0;
  return x;
}

inline Eigen::VectorXd hybridParkingTarget() {
  Eigen::VectorXd x(5);
  x << 2.55, 1.82, M_PI, 1.05, 0.95;
  return x;
}

inline void configureHybridParkingObstacles(CollisionChecker &checker) {
  checker.addRectangle(-0.50, -0.50, 5.00, 0.50);
  checker.addRectangle(-0.50, 3.00, 5.00, 0.50);
  checker.addRectangle(-0.50, -0.50, 0.50, 4.00);
  checker.addRectangle(4.00, -0.50, 0.50, 4.00);

  checker.addRectangle(1.05, 1.45, 0.82, 0.72);
  checker.addRectangle(3.15, 1.45, 0.82, 0.72);
  checker.addRectangle(1.95, 2.38, 1.10, 0.18);
}

inline ParkingMetrics calculateParkingMetrics(const Eigen::VectorXd &x,
                                               const Eigen::VectorXd &target) {
  ParkingMetrics metrics;
  metrics.position_error = (x.head(2) - target.head(2)).norm();
  metrics.heading_error = std::abs(wrapHybridParkingAngle(x(2) - target(2)));
  metrics.pose_error = metrics.position_error + 0.35 * metrics.heading_error;
  metrics.forward_distance = x(3);
  metrics.reverse_distance = x(4);
  metrics.missing_forward = std::max(0.0, target(3) - x(3));
  metrics.missing_reverse = std::max(0.0, target(4) - x(4));
  return metrics;
}

inline bool isHybridParkingSolved(const ParkingMetrics &metrics) {
  return metrics.position_error < 0.22 && metrics.heading_error < 0.25 &&
         metrics.missing_forward < 0.08 && metrics.missing_reverse < 0.08;
}

inline double hybridParkingScore(const ParkingMetrics &metrics) {
  return metrics.pose_error + 0.75 * metrics.missing_forward +
         0.75 * metrics.missing_reverse;
}

inline void writeHybridParkingResultHeader(std::ofstream &csv) {
  csv << "solver,is_failed,is_success,iter,elapsed,score,pose_error,"
         "position_error,heading_error,forward_distance,reverse_distance,"
         "missing_forward,missing_reverse\n";
}

inline void writeHybridParkingPathHeader(std::ofstream &csv) {
  csv << "iter,x,y,theta,forward_distance,reverse_distance\n";
}

inline void writeHybridParkingControlHeader(std::ofstream &csv) {
  csv << "iter,signed_speed,steering\n";
}
