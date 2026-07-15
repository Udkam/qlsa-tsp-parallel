#include "tsp/cuda.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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

void validate_cuda_request(const DistanceMatrix& dm, const ParallelParams& params) {
    if (dm.size() <= 0) {
        throw std::invalid_argument("CUDA run requires a non-empty distance matrix");
    }
    if (params.chains < 1) {
        throw std::invalid_argument("chains must be >= 1");
    }
    if (params.cuda_block_size < 1 || params.cuda_block_size > 1024) {
        throw std::invalid_argument("cuda_block_size must be in [1, 1024]");
    }
    if (params.cuda_candidates_per_iter <= 0) {
        throw std::invalid_argument("cuda_candidates_per_iter must be positive");
    }
    if (params.cuda_candidates_per_iter > params.cuda_block_size) {
        throw std::invalid_argument("cuda_candidates_per_iter must be <= cuda_block_size");
    }

    int64_t iterations = params.sa_params.iterations;
    if (params.algorithm == AlgorithmKind::QLSA) {
        validate_qlsa_params(params.qlsa_params);
        iterations = params.qlsa_params.sa.iterations;
        if (params.qlsa_params.variant != "current") {
            throw std::invalid_argument("CUDA QLSA currently supports variant current only");
        }
        if (params.qlsa_params.state_window > kCudaMaxQlsaStateWindow) {
            throw std::invalid_argument("CUDA QLSA state_window must be <= " +
                                        std::to_string(kCudaMaxQlsaStateWindow));
        }
        const size_t action_count = params.qlsa_params.actions.empty()
                                        ? default_qlsa_actions().size()
                                        : params.qlsa_params.actions.size();
        if (action_count < 1 || action_count > 8) {
            throw std::invalid_argument("CUDA QLSA supports 1..8 actions");
        }
    } else if (params.sa_params.initial_temperature <= 0.0 ||
               params.sa_params.final_temperature <= 0.0) {
        throw std::invalid_argument("temperatures must be positive");
    }
    if (iterations < 1) {
        throw std::invalid_argument("iterations must be >= 1");
    }
    if (iterations > std::numeric_limits<int64_t>::max() / params.chains) {
        throw std::overflow_error("iterations * chains overflows int64_t");
    }
}

ParallelResult run_cuda_chains(const DistanceMatrix& dm, const ParallelParams& params) {
    validate_cuda_request(dm, params);
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
