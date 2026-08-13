#include "altro/altro.hpp"
#include "altro/solver/solver.hpp"
#include "altrocpp_interface/altrocpp_interface.hpp"
#include "manipulator_random_pose_benchmark_common.h"
#include "rollout_ee_callback.h"
#include "rollout_state_callback.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

namespace {

constexpr int kDof = ManipulatorDynamicsModel::kDof;
constexpr int kStateDim = 2 * kDof;
constexpr int kInputDim = kDof;
constexpr int kWorkspaceConstraintDim = 2 * kDof * 3;
const Eigen::VectorXd kTauMax =
    (Eigen::VectorXd(kInputDim) << 18.0, 18.0, 14.0, 8.0, 6.0, 4.0)
        .finished();

Eigen::VectorXd workspaceConstraints(const ManipulatorDynamicsModel &model,
                                     const Eigen::VectorXd &x) {
  Eigen::VectorXd values(kWorkspaceConstraintDim);
  const auto points = model.getAllJointPositions(x);
  const auto &box = model.workspace_boxes.front();
  int row = 0;
  for (int link = 1; link <= kDof; ++link) {
    for (int sample = 0; sample < 3; ++sample) {
      const double alpha = 0.5 * sample;
      const Eigen::Vector3d p =
          (1.0 - alpha) * points[link - 1] + alpha * points[link];
      const double clearance =
          model.signedDistancePointAabb(p, box) - model.link_radius;
      values(row++) = model.hard_collision_margin - clearance;
      values(row++) = -p.z();
    }
  }
  return values;
}

class ManipulatorALTRO {
 public:
  explicit ManipulatorALTRO(ManipulatorDynamicsModel model)
      : model_(std::move(model)) {}

  void init(const MPPIParam &param) {
    dt_ = param.dt;
    horizon_ = param.T;
    x_init = param.x_init;
    x_target = param.x_target;
    Uo = Eigen::MatrixXd::Zero(kInputDim, horizon_);
    u0 = Eigen::VectorXd::Zero(kInputDim);
  }

  void setCollisionChecker(CollisionChecker *) {}
  void setSeed(std::uint_fast64_t) {}
  void setRolloutEECallback(RolloutEEBatchCallback) {}
  void setRolloutStateCallback(RolloutStateBatchCallback) {}
  double connectionDistance() const { return 0.0; }
  double trajectoryCost() const { return objective_; }

  void solve() {
    if (!solver_) buildSolver();

    solver_->SetInitialState(x_init.data(), kStateDim);
    setInitialGuess();

    const auto start = std::chrono::steady_clock::now();
    status_ = solver_->Solve();
    const auto finish = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration<double>(finish - start).count();
    elapsed_rollout = elapsed;
    elapsed_clustering = 0.0;
    elapsed_connection = 0.0;
    elapsed_guide = 0.0;
    objective_ = solver_->GetFinalObjective();

    for (int k = 0; k < horizon_; ++k) {
      Uo.col(k) = solver_->solver_->trajectory_->Control(k)
                      .cwiseMax(-kTauMax)
                      .cwiseMin(kTauMax);
    }
    u0 = Uo.col(0);
  }

  Eigen::MatrixXd U_0;
  Eigen::MatrixXd Uo;
  Eigen::VectorXd x_init;
  Eigen::VectorXd x_target;
  Eigen::VectorXd u0;
  double elapsed = 0.0;
  double elapsed_rollout = 0.0;
  double elapsed_clustering = 0.0;
  double elapsed_connection = 0.0;
  double elapsed_guide = 0.0;

 private:
  void buildSolver() {
    solver_ = std::make_unique<altro::ALTROSolver>(horizon_);
    solver_->SetDimension(kStateDim, kInputDim, 0, horizon_ + 1);
    solver_->SetTimeStep(dt_, 0, horizon_);

    auto dynamics = [this](double *next, const double *x, const double *u,
                           float dt) {
      Eigen::Map<Eigen::VectorXd>(next, kStateDim) = model_.rk4Step(
          Eigen::Map<const Eigen::VectorXd>(x, kStateDim),
          Eigen::Map<const Eigen::VectorXd>(u, kInputDim), dt);
    };
    auto dynamics_jacobian =
        [this](double *data, const double *x_data, const double *u_data,
               float dt) {
          Eigen::Map<Eigen::MatrixXd> jac(data, kStateDim,
                                          kStateDim + kInputDim);
          const Eigen::Map<const Eigen::VectorXd> x(x_data, kStateDim);
          const Eigen::Map<const Eigen::VectorXd> u(u_data, kInputDim);
          constexpr double eps = 1e-6;
          for (int col = 0; col < kStateDim + kInputDim; ++col) {
            Eigen::VectorXd xp = x;
            Eigen::VectorXd xm = x;
            Eigen::VectorXd up = u;
            Eigen::VectorXd um = u;
            if (col < kStateDim) {
              xp(col) += eps;
              xm(col) -= eps;
            } else {
              up(col - kStateDim) += eps;
              um(col - kStateDim) -= eps;
            }
            jac.col(col) =
                (model_.rk4Step(xp, up, dt) - model_.rk4Step(xm, um, dt)) /
                (2.0 * eps);
          }
        };
    solver_->SetExplicitDynamics(dynamics, dynamics_jacobian, 0, horizon_);

    Eigen::VectorXd stage_q = Eigen::VectorXd::Zero(kStateDim);
    stage_q.tail(kDof).setConstant(2.0 * model_.w_qdot);
    Eigen::VectorXd stage_r = Eigen::VectorXd::Constant(
        kInputDim, 2.0 * model_.w_tau);
    const Eigen::VectorXd zero_u = Eigen::VectorXd::Zero(kInputDim);
    solver_->SetLQRCost(stage_q.data(), stage_r.data(), x_target.data(),
                        zero_u.data(), 0, horizon_);
    setTerminalCost();

    addBoundConstraints();
    addWorkspaceConstraints();
    solver_->SetInitialState(x_init.data(), kStateDim);

    altro::AltroOptions options;
    options.tol_cost = 1e-4;
    options.tol_primal_feasibility = 1e-4;
    options.tol_stationarity = 1e-4;
    options.penalty_initial = 10.0;
    options.penalty_scaling = 10.0;
    options.penalty_max = 1e8;
    options.verbose = altro::Verbosity::Silent;
    solver_->SetOptions(options);
    solver_->Initialize();
  }

  Eigen::MatrixXd endEffectorJacobian(const Eigen::VectorXd &x) const {
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3, kStateDim);
    constexpr double eps = 1e-6;
    for (int col = 0; col < kDof; ++col) {
      Eigen::VectorXd xp = x;
      Eigen::VectorXd xm = x;
      xp(col) += eps;
      xm(col) -= eps;
      jac.col(col) =
          (model_.endEffectorPosition(xp) - model_.endEffectorPosition(xm)) /
          (2.0 * eps);
    }
    return jac;
  }

  void setTerminalCost() {
    auto cost = [this](const double *x, const double *) {
      return model_.terminalCost(
          Eigen::Map<const Eigen::VectorXd>(x, kStateDim), x_target);
    };
    auto gradient = [this](double *dx, double *du, const double *x_data,
                           const double *) {
      const Eigen::Map<const Eigen::VectorXd> x(x_data, kStateDim);
      Eigen::Map<Eigen::VectorXd> gx(dx, kStateDim);
      Eigen::Map<Eigen::VectorXd>(du, kInputDim).setZero();
      gx.setZero();
      gx.head(kDof) = 2.0 * model_.w_terminal_q *
                      (x.head(kDof) - x_target.head(kDof));
      gx.tail(kDof) = 2.0 * model_.w_terminal_qdot * x.tail(kDof);
      const Eigen::Vector3d error =
          model_.endEffectorPosition(x) -
          model_.endEffectorPosition(x_target);
      gx += 2.0 * model_.w_terminal_ee *
            endEffectorJacobian(x).transpose() * error;
    };
    auto hessian = [this](double *dxdx, double *dudu, double *dxdu,
                          const double *x_data, const double *) {
      const Eigen::Map<const Eigen::VectorXd> x(x_data, kStateDim);
      Eigen::Map<Eigen::MatrixXd> q(dxdx, kStateDim, kStateDim);
      Eigen::Map<Eigen::MatrixXd>(dudu, kInputDim, kInputDim).setZero();
      Eigen::Map<Eigen::MatrixXd>(dxdu, kStateDim, kInputDim).setZero();
      q.setZero();
      q.diagonal().head(kDof).setConstant(2.0 * model_.w_terminal_q);
      q.diagonal().tail(kDof).setConstant(2.0 * model_.w_terminal_qdot);
      const Eigen::MatrixXd jac = endEffectorJacobian(x);
      q += 2.0 * model_.w_terminal_ee * jac.transpose() * jac;
    };
    solver_->solver_->problem_.SetCostFunction(
        std::make_shared<altro::cpp_interface::GeneralCostFunction>(
            kStateDim, kInputDim, cost, gradient, hessian),
        horizon_);
  }

  void addBoundConstraints() {
    constexpr int path_dim = 2 * kStateDim + 2 * kInputDim;
    auto path = [this](double *out, const double *x, const double *u) {
      int row = 0;
      for (int i = 0; i < kDof; ++i) out[row++] = x[i] - model_.q_max(i);
      for (int i = 0; i < kDof; ++i) out[row++] = model_.q_min(i) - x[i];
      for (int i = 0; i < kDof; ++i)
        out[row++] = x[kDof + i] - model_.qdot_max(i);
      for (int i = 0; i < kDof; ++i)
        out[row++] = -model_.qdot_max(i) - x[kDof + i];
      for (int i = 0; i < kInputDim; ++i)
        out[row++] = u[i] - kTauMax(i);
      for (int i = 0; i < kInputDim; ++i)
        out[row++] = -kTauMax(i) - u[i];
    };
    auto path_jacobian = [](double *data, const double *, const double *) {
      Eigen::Map<Eigen::MatrixXd> jac(data, path_dim,
                                      kStateDim + kInputDim);
      jac.setZero();
      int row = 0;
      for (int i = 0; i < kDof; ++i) jac(row++, i) = 1.0;
      for (int i = 0; i < kDof; ++i) jac(row++, i) = -1.0;
      for (int i = 0; i < kDof; ++i) jac(row++, kDof + i) = 1.0;
      for (int i = 0; i < kDof; ++i) jac(row++, kDof + i) = -1.0;
      for (int i = 0; i < kInputDim; ++i)
        jac(row++, kStateDim + i) = 1.0;
      for (int i = 0; i < kInputDim; ++i)
        jac(row++, kStateDim + i) = -1.0;
    };
    solver_->SetConstraint(path, path_jacobian, path_dim,
                           altro::ConstraintType::INEQUALITY, "bounds", 0,
                           horizon_, nullptr);

    constexpr int terminal_dim = 2 * kStateDim;
    auto terminal = [this](double *out, const double *x, const double *) {
      int row = 0;
      for (int i = 0; i < kDof; ++i) out[row++] = x[i] - model_.q_max(i);
      for (int i = 0; i < kDof; ++i) out[row++] = model_.q_min(i) - x[i];
      for (int i = 0; i < kDof; ++i)
        out[row++] = x[kDof + i] - model_.qdot_max(i);
      for (int i = 0; i < kDof; ++i)
        out[row++] = -model_.qdot_max(i) - x[kDof + i];
    };
    auto terminal_jacobian =
        [](double *data, const double *, const double *) {
          Eigen::Map<Eigen::MatrixXd> jac(data, terminal_dim,
                                          kStateDim + kInputDim);
          jac.setZero();
          int row = 0;
          for (int i = 0; i < kDof; ++i) jac(row++, i) = 1.0;
          for (int i = 0; i < kDof; ++i) jac(row++, i) = -1.0;
          for (int i = 0; i < kDof; ++i) jac(row++, kDof + i) = 1.0;
          for (int i = 0; i < kDof; ++i) jac(row++, kDof + i) = -1.0;
        };
    solver_->SetConstraint(terminal, terminal_jacobian, terminal_dim,
                           altro::ConstraintType::INEQUALITY,
                           "terminal_bounds", horizon_, 0, nullptr);
  }

  void addWorkspaceConstraints() {
    auto constraint = [this](double *out, const double *x, const double *) {
      Eigen::Map<Eigen::VectorXd>(out, kWorkspaceConstraintDim) =
          workspaceConstraints(
              model_, Eigen::Map<const Eigen::VectorXd>(x, kStateDim));
    };
    auto jacobian = [this](double *data, const double *x_data, const double *) {
      Eigen::Map<Eigen::MatrixXd> jac(
          data, kWorkspaceConstraintDim, kStateDim + kInputDim);
      jac.setZero();
      const Eigen::Map<const Eigen::VectorXd> x(x_data, kStateDim);
      constexpr double eps = 1e-6;
      for (int col = 0; col < kDof; ++col) {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        xp(col) += eps;
        xm(col) -= eps;
        jac.col(col) =
            (workspaceConstraints(model_, xp) -
             workspaceConstraints(model_, xm)) /
            (2.0 * eps);
      }
    };
    solver_->SetConstraint(constraint, jacobian, kWorkspaceConstraintDim,
                           altro::ConstraintType::INEQUALITY, "workspace", 0,
                           horizon_ + 1, nullptr);
  }

  void setInitialGuess() {
    Eigen::MatrixXd guess = U_0;
    if (guess.rows() != kInputDim || guess.cols() != horizon_)
      guess = Eigen::MatrixXd::Zero(kInputDim, horizon_);
    for (int k = 0; k < horizon_; ++k)
      guess.col(k) = guess.col(k).cwiseMax(-kTauMax).cwiseMin(kTauMax);

    Eigen::VectorXd x = x_init;
    solver_->SetState(x.data(), kStateDim, 0);
    for (int k = 0; k < horizon_; ++k) {
      solver_->SetInput(guess.col(k).data(), kInputDim, k);
      x = model_.rk4Step(x, guess.col(k), dt_);
      solver_->SetState(x.data(), kStateDim, k + 1);
    }
  }

  ManipulatorDynamicsModel model_;
  std::unique_ptr<altro::ALTROSolver> solver_;
  altro::SolveStatus status_ = altro::SolveStatus::MaxIterations;
  float dt_ = 0.02f;
  int horizon_ = 0;
  double objective_ = 0.0;
};

}  // namespace

int main() {
  auto params =
      manipulator_random_pose_benchmark::randomPoseSolverParams();
  params.mppi.forward_samples = 1;
  return manipulator_random_pose_benchmark::runForwardBenchmark<
      ManipulatorALTRO>(
      "altro", "ALTRO", params.mppi,
      [](ManipulatorALTRO &,
         const manipulator_random_pose_benchmark::SolverParams &) {});
}
