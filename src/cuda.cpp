#include "tsp/cuda.hpp"

#include <iostream>

namespace tsp {

#ifdef TSP_HAS_CUDA
bool cuda_available_impl() noexcept;
ParallelResult run_cuda_chains_impl(const DistanceMatrix& dm, const ParallelParams& params);
#endif

bool cuda_available() noexcept {
#ifdef TSP_HAS_CUDA
    return cuda_available_impl();
#else
    return false;
#endif
}

ParallelResult run_cuda_chains(const DistanceMatrix& dm, const ParallelParams& params) {
#ifdef TSP_HAS_CUDA
    if (!cuda_available_impl()) {
        std::cerr << "Warning: CUDA is not available at runtime; falling back to serial multi-chain execution.\n";
        ParallelParams fallback = params;
        fallback.cuda_enabled = false;
        fallback.threads = 1;
        return run_parallel_chains(dm, fallback);
    }
    return run_cuda_chains_impl(dm, params);
#else
    std::cerr << "Warning: CUDA was not enabled at build time; falling back to serial multi-chain execution.\n";
    ParallelParams fallback = params;
    fallback.cuda_enabled = false;
    fallback.threads = 1;
    return run_parallel_chains(dm, fallback);
#endif
}

}  // namespace tsp
