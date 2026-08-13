#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Packed quantized end-effector positions. Each rollout stores its first xyz
// point as three little-endian int16 values, followed by int8 xyz deltas.
using RolloutEEBatchCallback = std::function<void(
    const std::string &branch, const std::vector<std::uint8_t> &positions,
    int rollout_count, int point_count)>;
