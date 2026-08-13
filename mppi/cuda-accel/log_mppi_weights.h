#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

inline bool computeLogMppiWeights(const std::vector<double> &costs,
                                  double gamma_u,
                                  double collision_cost,
                                  std::vector<double> &weights) {
  weights.assign(costs.size(), 0.0);
  if (costs.empty() || !std::isfinite(gamma_u) || gamma_u < 0.0 ||
      !std::isfinite(collision_cost))
    return false;

  double max_log_weight = -std::numeric_limits<double>::infinity();
  for (double cost : costs) {
    if (!std::isfinite(cost) || cost >= collision_cost) continue;
    const double log_weight = -gamma_u * cost;
    if (std::isfinite(log_weight))
      max_log_weight = std::max(max_log_weight, log_weight);
  }
  if (!std::isfinite(max_log_weight)) return false;

  double sum_exp = 0.0;
  for (double cost : costs) {
    if (!std::isfinite(cost) || cost >= collision_cost) continue;
    const double log_weight = -gamma_u * cost;
    const double shifted = log_weight - max_log_weight;
    if (std::isfinite(shifted)) sum_exp += std::exp(shifted);
  }
  if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) return false;

  for (std::size_t i = 0; i < costs.size(); ++i) {
    if (!std::isfinite(costs[i]) || costs[i] >= collision_cost) continue;
    const double log_weight = -gamma_u * costs[i];
    if (std::isfinite(log_weight))
      // exp(log_w - max_log_w) / sum(exp(log_w - max_log_w))
      weights[i] = std::exp(log_weight - max_log_weight) / sum_exp;
  }
  return true;
}

inline int leastFiniteCostIndex(const std::vector<double> &costs) {
  int best = -1;
  for (std::size_t i = 0; i < costs.size(); ++i) {
    if (std::isfinite(costs[i]) &&
        (best < 0 || costs[i] < costs[static_cast<std::size_t>(best)]))
      best = static_cast<int>(i);
  }
  return best;
}
