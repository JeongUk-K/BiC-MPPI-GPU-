#pragma once

#include "mppi.h"

class LogMPPI : public MPPI {
public:
    template<typename ModelClass>
    LogMPPI(ModelClass model);
    ~LogMPPI();

    Eigen::Rand::LognormalGen<double> log_norm_gen{0.0, 1.0};

    Eigen::MatrixXd getNoise(const int &T) override {
        Eigen::MatrixXd log_distribution = log_norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
        return (sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng)).array() * log_distribution.array();
    }

    Eigen::MatrixXd getNoise(const int &T, const int &sample_index,
                             const int &solve_index,
                             const int &phase_index) override {
        std::uint_fast64_t mixed = noise_seed;
        mixed ^= 0x9e3779b97f4a7c15ULL +
                 static_cast<std::uint_fast64_t>(sample_index) + (mixed << 6) +
                 (mixed >> 2);
        mixed ^= 0xbf58476d1ce4e5b9ULL +
                 static_cast<std::uint_fast64_t>(solve_index) + (mixed << 6) +
                 (mixed >> 2);
        mixed ^= 0x94d049bb133111ebULL +
                 static_cast<std::uint_fast64_t>(phase_index) + (mixed << 6) +
                 (mixed >> 2);

        std::mt19937_64 local_urng(mixed);
        Eigen::Rand::NormalGen<double> local_norm_gen{0.0, 1.0};
        Eigen::Rand::LognormalGen<double> local_log_norm_gen{0.0, 1.0};
        Eigen::MatrixXd log_distribution =
            local_log_norm_gen.template generate<Eigen::MatrixXd>(dim_u, T,
                                                                  local_urng);
        return (sigma_u * local_norm_gen.template generate<Eigen::MatrixXd>(
                              dim_u, T, local_urng))
            .array() *
               log_distribution.array();
    }
};

template<typename ModelClass>
LogMPPI::LogMPPI(ModelClass model) : MPPI(model) {
}

LogMPPI::~LogMPPI() {
}
