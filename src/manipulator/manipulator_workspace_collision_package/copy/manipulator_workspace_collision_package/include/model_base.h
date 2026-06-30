#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>

// Maximum dimensions used by the uploaded CUDA rollout kernels.
// 12D manipulator dynamics x=[q;qdot], u=tau fits in these limits.
#ifndef GPU_MAX_DIM_X
#define GPU_MAX_DIM_X 14
#endif
#ifndef GPU_MAX_DIM_U
#define GPU_MAX_DIM_U 7
#endif
#ifndef GPU_MAX_T
#define GPU_MAX_T 200
#endif
#ifndef GPU_MAX_NS
#define GPU_MAX_NS 2048
#endif

class ModelBase {
public:
  ModelBase() = default;
  virtual ~ModelBase() = default;

  int dim_x = 0;
  int dim_u = 0;

  // Continuous-time dynamics derivative: x_dot = f(x,u).
  // The uploaded BiMPPI_GPU integrates this derivative as x += dt*f(x,u).
  std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;

  // Stage and terminal costs used by CPU-side post-rollout selection and logging.
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
  std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;

  // Projection onto admissible input-sequence set.
  std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

  // Optional geometry hook. Non-manipulator models may return an empty vector.
  virtual std::vector<Eigen::Vector3d>
  getAllJointPositions(const Eigen::VectorXd &x) const {
    (void)x;
    return {};
  }
};
