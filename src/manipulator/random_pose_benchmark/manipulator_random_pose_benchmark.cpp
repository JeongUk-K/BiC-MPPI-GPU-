#include "manipulator_random_pose_benchmark_common.h"

// acados does not have an IPOPT backend.  This runner uses acados' native SQP
// NLP solver with partial-condensing HPIPM, and keeps the OCP in this file so
// no CasADi-generated solver is required.
extern "C" {
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_nlp/ocp_nlp_cost_external.h"
#include "acados/ocp_nlp/ocp_nlp_dynamics_disc.h"
#include "acados/utils/external_function_generic.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "blasfeo_d_aux.h"
#include "blasfeo_d_aux_ext_dep.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kNx = ManipulatorDynamicsModel::kDof * 2;
constexpr int kNu = ManipulatorDynamicsModel::kDof;
constexpr int kHorizon = 90;
constexpr double kDt = 0.02;

// These are the limits used by the existing CUDA rollout projector.
const Eigen::VectorXd kTauMin =
    (Eigen::VectorXd(6) << -18.0, -18.0, -14.0, -8.0, -6.0, -4.0).finished();
const Eigen::VectorXd kTauMax =
    (Eigen::VectorXd(6) << 18.0, 18.0, 14.0, 8.0, 6.0, 4.0).finished();

struct OcpContext {
  const ManipulatorDynamicsModel *model = nullptr;
  Eigen::VectorXd x_goal;
};

// acados' generic callback ABI has no user-data argument.  The benchmark owns
// one solver at a time, so a process-wide context also works for acados' OpenMP
// callback threads and avoids a second generated-code/configuration layer.
OcpContext *g_context = nullptr;

double stableSoftplus(double x) {
  if (x > 40.0) return x;
  if (x < -40.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

double smoothWorkspacePenalty(const ManipulatorDynamicsModel &model,
                              const Eigen::VectorXd &x) {
  if (model.workspace_boxes.empty()) return 0.0;

  const auto points = model.getAllJointPositions(x);
  constexpr double kSafeMargin = 0.10;
  constexpr double kHardMargin = 0.02;
  constexpr double kSharpness = 80.0;
  constexpr double kSafeWeight = 2.0e3;
  constexpr double kHardWeight = 1.0e5;
  double cost = 0.0;

  for (std::size_t link = 1; link < points.size(); ++link) {
    const Eigen::Vector3d a = points[link - 1];
    const Eigen::Vector3d b = points[link];
    for (int sample = 0; sample < 5; ++sample) {
      const double alpha = static_cast<double>(sample) / 4.0;
      const Eigen::Vector3d p = (1.0 - alpha) * a + alpha * b;
      if (p.z() < 0.0) {
        const double ground = stableSoftplus(-kSharpness * p.z()) / kSharpness;
        cost += 2.0e4 * ground * ground;
      }
      for (const auto &box : model.workspace_boxes) {
        const double sd = model.signedDistancePointAabb(p, box) - model.link_radius;
        const double safe = stableSoftplus(kSharpness * (kSafeMargin - sd)) /
                            kSharpness;
        const double hard = stableSoftplus(kSharpness * (kHardMargin - sd)) /
                            kSharpness;
        cost += kSafeWeight * safe * safe / (kSafeMargin * kSafeMargin);
        cost += kHardWeight * hard * hard;
      }
    }
  }
  return cost;
}

double stageCost(const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
  const auto &model = *g_context->model;
  const Eigen::VectorXd qdot = x.tail(ManipulatorDynamicsModel::kDof);
  return model.w_tau * u.squaredNorm() + model.w_qdot * qdot.squaredNorm() +
         model.jointLimitPenalty(x) + smoothWorkspacePenalty(model, x);
}

double terminalCost(const Eigen::VectorXd &x) {
  const auto &model = *g_context->model;
  const Eigen::VectorXd q = x.head(ManipulatorDynamicsModel::kDof);
  const Eigen::VectorXd q_goal = g_context->x_goal.head(ManipulatorDynamicsModel::kDof);
  const Eigen::VectorXd qdot = x.tail(ManipulatorDynamicsModel::kDof);
  const double ee_error =
      (model.endEffectorPosition(x) - model.endEffectorPosition(g_context->x_goal))
          .squaredNorm();
  return model.w_terminal_q * (q - q_goal).squaredNorm() +
         model.w_terminal_ee * ee_error +
         model.w_terminal_qdot * qdot.squaredNorm();
}

void readState(void **in, int input, Eigen::VectorXd &value) {
  auto *args = static_cast<blasfeo_dvec_args *>(in[input]);
  for (int i = 0; i < value.size(); ++i) {
    value(i) = BLASFEO_DVECEL(args->x, args->xi + i);
  }
}

void writeDynamicsJacobian(blasfeo_dmat *jac, int ai, int aj,
                           const Eigen::MatrixXd &A,
                           const Eigen::MatrixXd &B) {
  blasfeo_dgese(kNu + kNx, kNx, 0.0, jac, ai, aj);
  for (int r = 0; r < kNu; ++r)
    for (int c = 0; c < kNx; ++c)
      BLASFEO_DMATEL(jac, ai + r, aj + c) = B(c, r);
  for (int r = 0; r < kNx; ++r)
    for (int c = 0; c < kNx; ++c)
      BLASFEO_DMATEL(jac, ai + kNu + r, aj + c) = A(c, r);
}

void continuousDynamicsAndJacobians(const ManipulatorDynamicsModel &model,
                                    const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::VectorXd &dx, Eigen::MatrixXd &fx,
                                    Eigen::MatrixXd &fu) {
  dx.setZero(kNx);
  fx.setZero(kNx, kNx);
  fu.setZero(kNx, kNu);
  for (int i = 0; i < kNu; ++i) {
    const double inertia = model.inertia(i);
    dx(i) = x(kNu + i);
    dx(kNu + i) =
        (u(i) - model.damping(i) * x(kNu + i) -
         model.gravity_amp(i) * std::sin(x(i))) /
        inertia;
    fx(i, kNu + i) = 1.0;
    fx(kNu + i, i) =
        -model.gravity_amp(i) * std::cos(x(i)) / inertia;
    fx(kNu + i, kNu + i) = -model.damping(i) / inertia;
    fu(kNu + i, i) = 1.0 / inertia;
  }
}

void rk4DynamicsAndJacobians(const ManipulatorDynamicsModel &model,
                             const Eigen::VectorXd &x,
                             const Eigen::VectorXd &u, Eigen::VectorXd &next,
                             Eigen::MatrixXd &a, Eigen::MatrixXd &b) {
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(kNx, kNx);
  Eigen::VectorXd k1(kNx), k2(kNx), k3(kNx), k4(kNx);
  Eigen::MatrixXd f1(kNx, kNx), f2(kNx, kNx), f3(kNx, kNx), f4(kNx, kNx);
  Eigen::MatrixXd g1(kNx, kNu), g2(kNx, kNu), g3(kNx, kNu), g4(kNx, kNu);
  continuousDynamicsAndJacobians(model, x, u, k1, f1, g1);
  const Eigen::VectorXd x2 = x + 0.5 * kDt * k1;
  continuousDynamicsAndJacobians(model, x2, u, k2, f2, g2);
  const Eigen::VectorXd x3 = x + 0.5 * kDt * k2;
  continuousDynamicsAndJacobians(model, x3, u, k3, f3, g3);
  const Eigen::VectorXd x4 = x + kDt * k3;
  continuousDynamicsAndJacobians(model, x4, u, k4, f4, g4);

  next = x + (kDt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

  const Eigen::MatrixXd k2x = f2 * (identity + 0.5 * kDt * f1);
  const Eigen::MatrixXd k3x = f3 * (identity + 0.5 * kDt * k2x);
  const Eigen::MatrixXd k4x = f4 * (identity + kDt * k3x);
  const Eigen::MatrixXd k2u = f2 * (0.5 * kDt * g1) + g2;
  const Eigen::MatrixXd k3u = f3 * (0.5 * kDt * k2u) + g3;
  const Eigen::MatrixXd k4u = f4 * (kDt * k3u) + g4;
  a = identity + (kDt / 6.0) * (f1 + 2.0 * k2x + 2.0 * k3x + k4x);
  b = (kDt / 6.0) * (g1 + 2.0 * k2u + 2.0 * k3u + k4u);
}

void discreteDynamics(void *, ext_fun_arg_t *, void **in, ext_fun_arg_t *,
                      void **out) {
  Eigen::VectorXd x(kNx), u(kNu);
  readState(in, 0, x);
  readState(in, 1, u);

  Eigen::VectorXd next(kNx);
  Eigen::MatrixXd a(kNx, kNx), b(kNx, kNu);
  rk4DynamicsAndJacobians(*g_context->model, x, u, next, a, b);

  auto *f_args = static_cast<blasfeo_dvec_args *>(out[0]);
  for (int i = 0; i < kNx; ++i)
    BLASFEO_DVECEL(f_args->x, f_args->xi + i) = next(i);

  auto *jac_args = static_cast<blasfeo_dmat_args *>(out[1]);
  writeDynamicsJacobian(jac_args->A, jac_args->ai, jac_args->aj, a, b);

  // disc_dyn_fun_jac has only f and [J_u'; J_x'] outputs.  The dynamics
  // Hessian output belongs to disc_dyn_fun_jac_hess, which is not installed.
}

double evaluateCost(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
                    bool terminal) {
  return terminal ? terminalCost(x) : stageCost(x, u);
}

void externalCost(void **in, void **out, bool terminal) {
  auto *x_args = static_cast<blasfeo_dvec_args *>(in[0]);
  Eigen::VectorXd x(kNx);
  for (int i = 0; i < kNx; ++i)
    x(i) = BLASFEO_DVECEL(x_args->x, x_args->xi + i);

  Eigen::VectorXd u = Eigen::VectorXd::Zero(kNu);
  if (!terminal) {
    auto *u_args = static_cast<blasfeo_dvec_args *>(in[1]);
    for (int i = 0; i < kNu; ++i)
      u(i) = BLASFEO_DVECEL(u_args->x, u_args->xi + i);
  }

  const int nz = terminal ? kNx : kNx + kNu;
  const double h = 1.0e-5;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(nz);
  Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(nz);

  const double value = evaluateCost(x, u, terminal);
  for (int i = 0; i < nz; ++i) {
    Eigen::VectorXd xp = x;
    Eigen::VectorXd xm = x;
    Eigen::VectorXd up = u;
    Eigen::VectorXd um = u;
    if (terminal) {
      xp(i) += h;
      xm(i) -= h;
    } else if (i < kNu) {
      up(i) += h;
      um(i) -= h;
    } else {
      xp(i - kNu) += h;
      xm(i - kNu) -= h;
    }
    const double plus = evaluateCost(xp, up, terminal);
    const double minus = evaluateCost(xm, um, terminal);
    gradient(i) = (plus - minus) / (2.0 * h);
    diagonal(i) = std::max(1.0e-6, (plus + minus - 2.0 * value) / (h * h));
  }

  *static_cast<double *>(out[0]) = value;
  auto *grad = static_cast<blasfeo_dvec *>(out[1]);
  auto *hess = static_cast<blasfeo_dmat *>(out[2]);
  blasfeo_dgese(nz, nz, 0.0, hess, 0, 0);
  for (int i = 0; i < nz; ++i) {
    BLASFEO_DVECEL(grad, i) = gradient(i);
    BLASFEO_DMATEL(hess, i, i) = diagonal(i);
  }
}

void stageExternalCost(void *, ext_fun_arg_t *, void **in, ext_fun_arg_t *,
                       void **out) {
  externalCost(in, out, false);
}

void terminalExternalCost(void *, ext_fun_arg_t *, void **in, ext_fun_arg_t *,
                          void **out) {
  externalCost(in, out, true);
}

external_function_generic makeGenericFunction(
    void (*evaluate)(void *, ext_fun_arg_t *, void **, ext_fun_arg_t *,
                     void **)) {
  external_function_generic function{};
  function.evaluate = evaluate;
  function.get_external_workspace_requirement =
      &external_function_param_generic_get_external_workspace_requirement;
  function.set_external_workspace =
      &external_function_param_generic_set_external_workspace;
  return function;
}

class AcadosManipulatorOcp {
public:
  explicit AcadosManipulatorOcp(OcpContext &context) : context_(context) {
    g_context = &context_;
    create();
  }

  ~AcadosManipulatorOcp() {
    if (solver_) ocp_nlp_solver_destroy(solver_);
    if (opts_) ocp_nlp_solver_opts_destroy(opts_);
    if (out_) ocp_nlp_out_destroy(out_);
    if (in_) ocp_nlp_in_destroy(in_);
    if (dims_) ocp_nlp_dims_destroy(dims_);
    if (config_) ocp_nlp_config_destroy(config_);
    if (plan_) ocp_nlp_plan_destroy(plan_);
    if (g_context == &context_) g_context = nullptr;
  }

  void setInitialState(const Eigen::VectorXd &x0) {
    std::copy(x0.data(), x0.data() + kNx, lbx0_.begin());
    ubx0_ = lbx0_;
    auto *constraints =
        reinterpret_cast<ocp_nlp_constraints_bgh_model **>(in_->constraints);
    // d stores [lower(nb); upper(nb)], and the initial-state rows follow the
    // six stage-0 control rows in both halves.
    blasfeo_pack_dvec(kNx, lbx0_.data(), 1, &constraints[0]->d, kNu);
    blasfeo_pack_dvec(kNx, ubx0_.data(), 1, &constraints[0]->d,
                      kNu + kNu + kNx);
  }

  void initializeGuess(const Eigen::MatrixXd &u_guess,
                       const Eigen::MatrixXd &x_guess) {
    for (int stage = 0; stage < kHorizon; ++stage) {
      Eigen::VectorXd u = u_guess.col(stage);
      u = u.cwiseMax(kTauMin).cwiseMin(kTauMax);
      ocp_nlp_out_set(config_, dims_, out_, in_, stage, "u", u.data());
      ocp_nlp_out_set(config_, dims_, out_, in_, stage, "x",
                      const_cast<double *>(x_guess.col(stage).data()));
    }
    ocp_nlp_out_set(config_, dims_, out_, in_, kHorizon, "x",
                    const_cast<double *>(x_guess.col(kHorizon).data()));
  }

  int solve() { return ocp_nlp_solve(solver_, in_, out_); }

  Eigen::VectorXd firstControl() const {
    Eigen::VectorXd u(kNu);
    ocp_nlp_out_get(config_, dims_, out_, 0, "u", u.data());
    return u;
  }

  void shiftGuess(const Eigen::VectorXd &x0) {
    Eigen::MatrixXd x(kNx, kHorizon + 1);
    Eigen::MatrixXd u(kNu, kHorizon);
    for (int stage = 0; stage <= kHorizon; ++stage)
      ocp_nlp_out_get(config_, dims_, out_, stage, "x", x.col(stage).data());
    for (int stage = 0; stage < kHorizon; ++stage)
      ocp_nlp_out_get(config_, dims_, out_, stage, "u", u.col(stage).data());

    for (int stage = 0; stage < kHorizon; ++stage) {
      x.col(stage) = x.col(std::min(stage + 1, kHorizon));
      if (stage + 1 < kHorizon) u.col(stage) = u.col(stage + 1);
      else u.col(stage).setZero();
    }
    x.col(0) = x0;
    x.col(kHorizon) = x.col(kHorizon - 1);
    initializeGuess(u, x);
  }

  double timeSeconds() const {
    double value = 0.0;
    ocp_nlp_get(solver_, "time_tot", &value);
    return value;
  }

private:
  OcpContext &context_;
  ocp_nlp_plan_t *plan_ = nullptr;
  ocp_nlp_config *config_ = nullptr;
  ocp_nlp_dims *dims_ = nullptr;
  ocp_nlp_in *in_ = nullptr;
  ocp_nlp_out *out_ = nullptr;
  void *opts_ = nullptr;
  ocp_nlp_solver *solver_ = nullptr;
  external_function_generic dynamics_function_{};
  external_function_generic cost_function_{};
  external_function_generic terminal_cost_function_{};
  std::vector<double> lbx0_ = std::vector<double>(kNx);
  std::vector<double> ubx0_ = std::vector<double>(kNx);

  void create() {
    plan_ = ocp_nlp_plan_create(kHorizon);
    plan_->nlp_solver = SQP;
    plan_->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    for (int stage = 0; stage <= kHorizon; ++stage) {
      plan_->nlp_cost[stage] = EXTERNAL;
      plan_->nlp_constraints[stage] = BGH;
      if (stage < kHorizon) plan_->nlp_dynamics[stage] = DISCRETE_MODEL;
    }

    config_ = ocp_nlp_config_create(*plan_);
    dims_ = ocp_nlp_dims_create(config_);

    std::vector<int> nx(kHorizon + 1, kNx);
    std::vector<int> nu(kHorizon + 1, kNu);
    std::vector<int> nz(kHorizon + 1, 0);
    std::vector<int> ns(kHorizon + 1, 0);
    std::vector<int> nbx(kHorizon + 1, kNx);
    std::vector<int> nbu(kHorizon + 1, kNu);
    std::vector<int> ng(kHorizon + 1, 0);
    std::vector<int> nh(kHorizon + 1, 0);
    std::vector<int> nsbx(kHorizon + 1, 0);
    nu[kHorizon] = 0;
    nbu[kHorizon] = 0;

    ocp_nlp_dims_set_opt_vars(config_, dims_, "nx", nx.data());
    ocp_nlp_dims_set_opt_vars(config_, dims_, "nu", nu.data());
    ocp_nlp_dims_set_opt_vars(config_, dims_, "nz", nz.data());
    ocp_nlp_dims_set_opt_vars(config_, dims_, "ns", ns.data());
    for (int stage = 0; stage <= kHorizon; ++stage) {
      ocp_nlp_dims_set_constraints(config_, dims_, stage, "nbx", &nbx[stage]);
      ocp_nlp_dims_set_constraints(config_, dims_, stage, "nbu", &nbu[stage]);
      ocp_nlp_dims_set_constraints(config_, dims_, stage, "ng", &ng[stage]);
      ocp_nlp_dims_set_constraints(config_, dims_, stage, "nh", &nh[stage]);
      ocp_nlp_dims_set_constraints(config_, dims_, stage, "nsbx",
                                   &nsbx[stage]);
    }

    in_ = ocp_nlp_in_create(config_, dims_);
    out_ = ocp_nlp_out_create(config_, dims_);
    opts_ = ocp_nlp_solver_opts_create(config_, dims_);

    int max_sqp_iter = 20;
    double tol = 1.0e-4;
    int print_level = 0;
    ocp_nlp_solver_opts_set(config_, opts_, "max_iter", &max_sqp_iter);
    ocp_nlp_solver_opts_set(config_, opts_, "tol_stat", &tol);
    ocp_nlp_solver_opts_set(config_, opts_, "tol_eq", &tol);
    ocp_nlp_solver_opts_set(config_, opts_, "tol_ineq", &tol);
    ocp_nlp_solver_opts_set(config_, opts_, "tol_comp", &tol);
    ocp_nlp_solver_opts_set(config_, opts_, "print_level", &print_level);

    dynamics_function_ = makeGenericFunction(&discreteDynamics);
    cost_function_ = makeGenericFunction(&stageExternalCost);
    terminal_cost_function_ = makeGenericFunction(&terminalExternalCost);

    auto **dynamics =
        reinterpret_cast<ocp_nlp_dynamics_disc_model **>(in_->dynamics);
    for (int stage = 0; stage < kHorizon; ++stage)
      dynamics[stage]->disc_dyn_fun_jac = &dynamics_function_;

    auto **cost = reinterpret_cast<ocp_nlp_cost_external_model **>(in_->cost);
    for (int stage = 0; stage < kHorizon; ++stage)
      cost[stage]->ext_cost_fun_jac_hess = &cost_function_;
    cost[kHorizon]->ext_cost_fun_jac_hess = &terminal_cost_function_;

    auto **constraints =
        reinterpret_cast<ocp_nlp_constraints_bgh_model **>(in_->constraints);
    for (int stage = 0; stage <= kHorizon; ++stage) {
      const int nbu_stage = stage < kHorizon ? kNu : 0;
      idxb_[stage].resize(nbu_stage + kNx);
      lower_[stage].resize(nbu_stage + kNx);
      upper_[stage].resize(nbu_stage + kNx);
      for (int i = 0; i < nbu_stage; ++i) {
        idxb_[stage][i] = i;
        lower_[stage][i] = kTauMin(i);
        upper_[stage][i] = kTauMax(i);
      }
      for (int i = 0; i < kNx; ++i) idxb_[stage][nbu_stage + i] = nbu_stage + i;
      constraints[stage]->idxb = idxb_[stage].data();
    }

    for (int stage = 0; stage < kHorizon; ++stage) {
      double sampling_time = kDt;
      ocp_nlp_in_set(config_, dims_, in_, stage, "Ts",
                     &sampling_time);
    }
    setStateBounds();
    solver_ = ocp_nlp_solver_create(config_, dims_, opts_, in_);
    if (!solver_) throw std::runtime_error("acados solver creation failed");
    if (ocp_nlp_precompute(solver_, in_, out_) != ACADOS_SUCCESS)
      throw std::runtime_error("acados precompute failed");
  }

  std::vector<std::vector<int>> idxb_ =
      std::vector<std::vector<int>>(kHorizon + 1);
  std::vector<std::vector<double>> lower_ =
      std::vector<std::vector<double>>(kHorizon + 1);
  std::vector<std::vector<double>> upper_ =
      std::vector<std::vector<double>>(kHorizon + 1);

  void setStateBounds() {
    auto **constraints =
        reinterpret_cast<ocp_nlp_constraints_bgh_model **>(in_->constraints);
    for (int stage = 0; stage <= kHorizon; ++stage) {
      const int nbu_stage = stage < kHorizon ? kNu : 0;
      for (int i = 0; i < kNx; ++i) {
        const int row = nbu_stage + i;
        if (stage == 0) {
          lower_[stage][row] = 0.0;
          upper_[stage][row] = 0.0;
        } else if (i < kNu) {
          lower_[stage][row] = context_.model->q_min(i);
          upper_[stage][row] = context_.model->q_max(i);
        } else {
          const int j = i - kNu;
          lower_[stage][row] = -context_.model->qdot_max(j);
          upper_[stage][row] = context_.model->qdot_max(j);
        }
      }
      const int n = nbu_stage + kNx;
      blasfeo_pack_dvec(n, lower_[stage].data(), 1, &constraints[stage]->d, 0);
      blasfeo_pack_dvec(n, upper_[stage].data(), 1, &constraints[stage]->d, n);
    }
  }
};

Eigen::MatrixXd makeInitialGuess(const ManipulatorDynamicsModel &model,
                                 const Eigen::VectorXd &x_init,
                                 const Eigen::VectorXd &x_waypoint,
                                 const Eigen::VectorXd &x_goal) {
  Eigen::MatrixXd u = makeWaypointPdTorqueWarmStart(
      model, x_init, x_waypoint, x_goal, kHorizon, kDt, 18.0, 7.0);
  for (int t = 0; t < kHorizon; ++t)
    u.col(t) = u.col(t).cwiseMax(kTauMin).cwiseMin(kTauMax);
  return u;
}

}  // namespace

int main() {
  using namespace manipulator_random_pose_benchmark;

  try {
    BenchmarkConfig config = benchmarkConfigFromEnvironment();
    ManipulatorDynamicsModel model;
    configureWorkspaceModel(model, config);
    const auto scenarios = selectBenchmarkScenarios(
        makeBenchmarkScenarios(model, config), config);
    ensureOutputDirectories(config, "acados");
    writeScenarioCsv(config, model, scenarios);

    std::ofstream stats(solverStatsPath(config, "acados"));
    if (!stats) throw std::runtime_error("failed to open acados stats CSV");
    writeStatsHeader(stats);

    for (const auto &scenario : scenarios) {
      applyScenarioObstacle(model, scenario.obstacle);
      const Eigen::VectorXd x_goal = manipulator_pose_set::makeStateFromPose(
          model, scenario.pose_pair.goal_pose);
      Eigen::VectorXd x_now = manipulator_pose_set::makeStateFromPose(
          model, scenario.pose_pair.init_pose);
      const Eigen::VectorXd x_waypoint = makeScenarioWaypointState(model, scenario);

      OcpContext context{&model, x_goal};
      AcadosManipulatorOcp solver(context);
      Eigen::MatrixXd u_guess = makeInitialGuess(model, x_now, x_waypoint, x_goal);
      Eigen::MatrixXd x_guess = Eigen::MatrixXd::Zero(kNx, kHorizon + 1);
      x_guess.col(0) = x_now;
      for (int t = 0; t < kHorizon; ++t) {
        x_guess.col(t + 1) = model.rk4Step(x_guess.col(t), u_guess.col(t), kDt);
      }
      solver.initializeGuess(u_guess, x_guess);

      Eigen::MatrixXd executed_x = Eigen::MatrixXd::Zero(kNx, config.max_iter + 1);
      executed_x.col(0) = x_now;
      int executed_steps = 0;
      int solve_iterations = 0;
      int last_status = ACADOS_SUCCESS;
      double total_solver_elapsed = 0.0;
      bool reached = false;

      const auto wall_start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < config.max_iter; ++iter) {
        if (isReached(model, x_now, x_goal, config)) {
          reached = true;
          break;
        }
        solver.setInitialState(x_now);
        const auto solve_start = std::chrono::steady_clock::now();
        last_status = solver.solve();
        const auto solve_finish = std::chrono::steady_clock::now();
        ++solve_iterations;
        total_solver_elapsed +=
            std::chrono::duration<double>(solve_finish - solve_start).count();
        // SQP can return MAXITER/MINSTEP with a usable last iterate.  Keep
        // receding-horizon control alive for those statuses; a QP failure does
        // not provide a safe control to apply.
        if (last_status == ACADOS_QP_FAILURE) break;

        const Eigen::VectorXd u0 = solver.firstControl();
        x_now = model.rk4Step(x_now, u0, kDt);
        executed_x.col(iter + 1) = x_now;
        executed_steps = iter + 1;
        reached = isReached(model, x_now, x_goal, config);
        if (reached) break;
        solver.shiftGuess(x_now);
      }
      const auto wall_finish = std::chrono::steady_clock::now();

      const Eigen::MatrixXd executed_trimmed =
          executed_x.leftCols(executed_steps + 1);
      const auto trajectory = trajectoryPath(config, "acados", scenario.index);
      writeMatrixCsv(trajectory.string(), executed_trimmed, "x");
      const double final_q_error =
          (x_now.head(kNu) - x_goal.head(kNu)).norm();
      const double final_ee_error =
          (model.endEffectorPosition(x_now) - model.endEffectorPosition(x_goal))
              .norm();
      const double final_qdot_norm = x_now.tail(kNu).norm();
      const int collision_count = countWorkspaceCollisions(model, executed_trimmed);

      BenchmarkResult result;
      result.solver_key = "acados";
      result.solver_label = "acados SQP (HPIPM)";
      result.scenario_index = scenario.index;
      result.rollout_seed = scenario.rollout_seed;
      result.init_idx = scenario.pose_pair.init_idx;
      result.goal_idx = scenario.pose_pair.goal_idx;
      result.init_name = scenario.pose_pair.init_pose.name;
      result.goal_name = scenario.pose_pair.goal_pose.name;
      result.obstacle_xmin = scenario.obstacle.xmin;
      result.obstacle_xmax = scenario.obstacle.xmax;
      result.obstacle_ymin = scenario.obstacle.ymin;
      result.obstacle_ymax = scenario.obstacle.ymax;
      result.obstacle_zmin = scenario.obstacle.zmin;
      result.obstacle_zmax = scenario.obstacle.zmax;
      result.direct_path_blocked = scenario.direct_path_blocked;
      result.init_collision = model.inWorkspaceCollision(executed_trimmed.col(0));
      result.goal_collision = model.inWorkspaceCollision(x_goal);
      result.reached = reached;
      result.success = reached && last_status == ACADOS_SUCCESS &&
                       collision_count == 0;
      result.solve_iterations = solve_iterations;
      result.executed_steps = executed_steps;
      result.max_iter = config.max_iter;
      result.q_tol = config.q_tol;
      result.ee_tol = config.ee_tol;
      result.qdot_tol = config.qdot_tol;
      result.max_pair_q_distance = config.max_pair_q_distance;
      result.total_solver_elapsed = total_solver_elapsed;
      result.mean_solver_elapsed = solve_iterations > 0
                                      ? total_solver_elapsed / solve_iterations
                                      : 0.0;
      result.wall_elapsed =
          std::chrono::duration<double>(wall_finish - wall_start).count();
      result.final_q_error = final_q_error;
      result.final_ee_error = final_ee_error;
      result.final_qdot_norm = final_qdot_norm;
      result.collision_count = collision_count;
      result.connection_distance = 0.0;
      result.horizon = kHorizon;
      result.samples_per_branch = 1;
      result.samples_total_per_iter = 1;
      result.trajectory_csv = trajectory.string();
      writeStatsRow(stats, result);

      std::cout << result.solver_label << " " << scenarioTag(scenario.index)
                << " init=" << result.init_name << " goal=" << result.goal_name
                << " success=" << static_cast<int>(result.success)
                << " status=" << last_status << " iter=" << solve_iterations
                << " collisions=" << collision_count << "\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "acados manipulator benchmark failed: " << error.what() << "\n";
    return 1;
  }
}
