#include "tsp/cuda.hpp"

#include <iostream>
#include <stdexcept>

#include "tsp/timer.hpp"

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
    // Keep public API behavior independent of whether this process happens to
    // have a CUDA build/device. CPU fallback must not silently execute a
    // different QLSA variant than an available GPU would execute.
    if (params.algorithm == AlgorithmKind::QLSA && params.qlsa_params.variant != "current") {
        throw std::invalid_argument("CUDA QLSA currently supports variant current only");
    }
    Timer total_timer;
#ifdef TSP_HAS_CUDA
    if (!cuda_available_impl()) {
        std::cerr << "Warning: CUDA is not available at runtime; falling back to serial multi-chain execution.\n";
        ParallelParams fallback = params;
        fallback.cuda_enabled = false;
        fallback.threads = 1;
        ParallelResult result = run_parallel_chains(dm, fallback);
        result.requested_backend = ParallelBackend::Cuda;
        result.backend_fallback = true;
        result.backend_fallback_reason =
            "CUDA was requested but no CUDA device is available at runtime";
        result.total_elapsed_ms = total_timer.elapsed_ms();
        result.elapsed_ms = result.total_elapsed_ms;
        return result;
    }
    ParallelResult result = run_cuda_chains_impl(dm, params);
    result.total_elapsed_ms = total_timer.elapsed_ms();
    result.elapsed_ms = result.total_elapsed_ms;
    return result;
#else
    std::cerr << "Warning: CUDA was not enabled at build time; falling back to serial multi-chain execution.\n";
    ParallelParams fallback = params;
    fallback.cuda_enabled = false;
    fallback.threads = 1;
    ParallelResult result = run_parallel_chains(dm, fallback);
    result.requested_backend = ParallelBackend::Cuda;
    result.backend_fallback = true;
    result.backend_fallback_reason =
        "CUDA was requested but is not enabled in this build";
    result.total_elapsed_ms = total_timer.elapsed_ms();
    result.elapsed_ms = result.total_elapsed_ms;
    return result;
#endif
}

}  // namespace tsp
