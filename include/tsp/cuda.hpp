#pragma once

#include "tsp/distance_matrix.hpp"
#include "tsp/parallel.hpp"

namespace tsp {

constexpr int kCudaMaxQlsaStateWindow = 64;

[[nodiscard]] bool cuda_available() noexcept;

// Applies the CUDA request contract before runtime device detection so CPU
// fallback and real GPU execution reject the same invalid configuration.
void validate_cuda_request(const DistanceMatrix& dm, const ParallelParams& params);
ParallelResult run_cuda_chains(const DistanceMatrix& dm, const ParallelParams& params);

}  // namespace tsp
