#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// IEEE-754 binary16 state values laid out as [rollout][time][state].
using RolloutStateBatchCallback = std::function<void(
    const std::string &branch, const std::vector<std::uint16_t> &states,
    int rollout_count, int point_count, int state_dim)>;
