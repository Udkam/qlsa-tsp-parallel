#pragma once

#include "tsp/distance_matrix.hpp"
#include "tsp/parallel.hpp"

namespace tsp {

[[nodiscard]] bool cuda_available() noexcept;

ParallelResult run_cuda_chains(const DistanceMatrix& dm, const ParallelParams& params);

}  // namespace tsp
