#pragma once

#include <cstdint>
#include <vector>

#include "tsp/distance_matrix.hpp"

namespace tsp {

struct SAParams {
    int64_t iterations = 1000000;
    double initial_temperature = 1000.0;
    double final_temperature = 1e-3;
    uint64_t seed = 1;
    bool use_nearest_neighbor_init = true;
};

struct SAResult {
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length = 0;
    double elapsed_ms = 0.0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
};

SAResult run_sa_2opt(const DistanceMatrix& dm, const SAParams& params);

}  // namespace tsp
