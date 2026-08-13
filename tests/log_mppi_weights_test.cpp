#include "log_mppi_weights.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {
constexpr double kCollisionCost = 1e8;

void assertNormalized(const std::vector<double> &weights) {
  for (double weight : weights) assert(std::isfinite(weight));
  assert(std::abs(std::accumulate(weights.begin(), weights.end(), 0.0) - 1.0) <
         1e-12);
}
}  // namespace

int main() {
  std::vector<double> weights;

  assert(computeLogMppiWeights({1000.0, 1001.0, 1002.0}, 10.0,
                               kCollisionCost, weights));
  assertNormalized(weights);
  assert(weights[0] > weights[1] && weights[1] > weights[2]);

  assert(computeLogMppiWeights({1.0, 2.0, 3.0}, 0.5, kCollisionCost,
                               weights));
  const double direct_sum =
      std::exp(-0.5) + std::exp(-1.0) + std::exp(-1.5);
  for (int i = 0; i < 3; ++i)
    assert(std::abs(weights[i] - std::exp(-0.5 * (i + 1)) / direct_sum) <
           1e-12);

  assert(computeLogMppiWeights(
      {1e8, 12.0, 15.0, std::numeric_limits<double>::infinity()}, 1.0,
      kCollisionCost, weights));
  assertNormalized(weights);
  assert(weights[0] == 0.0 && weights[3] == 0.0 && weights[1] > weights[2]);

  const std::vector<double> invalid = {
      1e8, 1e8, std::numeric_limits<double>::infinity()};
  assert(!computeLogMppiWeights(invalid, 1.0, kCollisionCost, weights));
  assert(std::accumulate(weights.begin(), weights.end(), 0.0) == 0.0);
  assert(leastFiniteCostIndex(invalid) == 0);

  assert(computeLogMppiWeights({1.0, 2.0, 3.0}, 0.0, kCollisionCost,
                               weights));
  assertNormalized(weights);
  for (double weight : weights) assert(std::abs(weight - 1.0 / 3.0) < 1e-12);

  assert(!computeLogMppiWeights({}, 1.0, kCollisionCost, weights));
  assert(weights.empty());
}
