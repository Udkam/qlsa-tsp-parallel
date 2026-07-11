#include "tsp/parallel.hpp"

#include <limits>
#include <stdexcept>

#ifdef TSP_HAS_OPENMP
#include <omp.h>
#endif

#include "tsp/cuda.hpp"
#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {
namespace {

constexpr uint64_t kSeedStride = 0x9E3779B97F4A7C15ULL;

void validate_parallel_params(const ParallelParams& params) {
    if (params.chains < 1) {
        throw std::invalid_argument("chains must be >= 1");
    }
    if (params.threads < 1) {
        throw std::invalid_argument("threads must be >= 1");
    }
    if (params.cuda_block_size < 1) {
        throw std::invalid_argument("cuda_block_size must be >= 1");
    }
    if (params.cuda_candidates_per_iter <= 0) {
        throw std::invalid_argument("cuda_candidates_per_iter must be positive");
    }
    if (params.cuda_candidates_per_iter > params.cuda_block_size) {
        throw std::invalid_argument("cuda_candidates_per_iter must be <= cuda_block_size");
    }
    const int64_t iterations = (params.algorithm == AlgorithmKind::SA)
                                   ? params.sa_params.iterations
                                   : params.qlsa_params.sa.iterations;
    if (iterations < 1) {
        throw std::invalid_argument("iterations must be >= 1");
    }
    if (iterations > std::numeric_limits<int64_t>::max() / params.chains) {
        throw std::overflow_error("iterations * chains overflows int64_t");
    }
}

ChainResult run_one_chain(const DistanceMatrix& dm, const ParallelParams& params, int chain_id_value) {
    ChainResult chain;
    chain.chain_id = chain_id_value;
    chain.seed = chain_seed(params.base_seed, chain_id_value);

    if (params.algorithm == AlgorithmKind::SA) {
        SAParams sa_params = params.sa_params;
        sa_params.seed = chain.seed;
        const SAResult result = run_sa_2opt(dm, sa_params);
        chain.best_tour = result.best_tour;
        chain.best_length = result.best_length;
        chain.final_length = result.final_length;
        chain.elapsed_ms = result.elapsed_ms;
        chain.accepted_moves = result.accepted_moves;
        chain.improved_moves = result.improved_moves;
        chain.iterations_completed = result.iterations_completed;
        chain.deadline_reached = result.deadline_reached;
        return chain;
    }

    QLSAParams qlsa_params = params.qlsa_params;
    qlsa_params.sa.seed = chain.seed;
    const QLSAResult result = run_qlsa_2opt(dm, qlsa_params);
    chain.best_tour = result.best_tour;
    chain.best_length = result.best_length;
    chain.final_length = result.final_length;
    chain.elapsed_ms = result.elapsed_ms;
    chain.accepted_moves = result.accepted_moves;
    chain.improved_moves = result.improved_moves;
    chain.iterations_completed = result.iterations_completed;
    chain.deadline_reached = result.deadline_reached;
    return chain;
}

}  // namespace

bool openmp_available() noexcept {
#ifdef TSP_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

const char* parallel_backend_name(ParallelBackend backend) noexcept {
    switch (backend) {
        case ParallelBackend::CpuSerial:
            return "cpu_serial";
        case ParallelBackend::OpenMP:
            return "openmp";
        case ParallelBackend::Cuda:
            return "cuda";
    }
    return "unknown";
}

uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t chain_seed(uint64_t base_seed, int chain_id_value) noexcept {
    return splitmix64(base_seed + kSeedStride * static_cast<uint64_t>(chain_id_value + 1));
}

ParallelResult run_parallel_chains(const DistanceMatrix& dm, const ParallelParams& params) {
    validate_parallel_params(params);
    if (dm.size() <= 0) {
        throw std::invalid_argument("run_parallel_chains requires a non-empty distance matrix");
    }
    if (params.cuda_enabled) {
        return run_cuda_chains(dm, params);
    }

    Timer timer;
    ParallelResult result;
    result.requested_backend = params.threads > 1
                                   ? ParallelBackend::OpenMP
                                   : ParallelBackend::CpuSerial;
#ifndef TSP_HAS_OPENMP
    result.actual_backend = ParallelBackend::CpuSerial;
    if (params.threads > 1) {
        result.backend_fallback = true;
        result.backend_fallback_reason =
            "OpenMP was requested but is not available in this build";
    }
#endif
    result.chains = params.chains;
    result.threads = params.threads;
    result.base_seed = params.base_seed;
    result.chain_results.resize(static_cast<size_t>(params.chains));

#ifdef TSP_HAS_OPENMP
    if (params.threads > 1) {
#pragma omp parallel num_threads(params.threads)
        {
#pragma omp single
            result.actual_threads = omp_get_num_threads();
#pragma omp for schedule(static)
            for (int chain_id_value = 0; chain_id_value < params.chains; ++chain_id_value) {
                result.chain_results[static_cast<size_t>(chain_id_value)] =
                    run_one_chain(dm, params, chain_id_value);
            }
        }
        result.actual_backend = result.actual_threads > 1
                                    ? ParallelBackend::OpenMP
                                    : ParallelBackend::CpuSerial;
        if (result.actual_threads <= 1) {
            result.backend_fallback = true;
            result.backend_fallback_reason =
                "OpenMP was requested but the runtime created a single-thread team";
        }
    } else
#endif
    {
        result.actual_threads = 1;
        result.actual_backend = ParallelBackend::CpuSerial;
        for (int chain_id_value = 0; chain_id_value < params.chains; ++chain_id_value) {
            result.chain_results[static_cast<size_t>(chain_id_value)] =
                run_one_chain(dm, params, chain_id_value);
        }
    }

    int best_index = -1;
    int best_length = std::numeric_limits<int>::max();
    for (int i = 0; i < params.chains; ++i) {
        const ChainResult& chain = result.chain_results[static_cast<size_t>(i)];
        result.total_accepted_moves += chain.accepted_moves;
        result.total_improved_moves += chain.improved_moves;
        result.total_iterations_completed += chain.iterations_completed;
        result.deadline_reached = result.deadline_reached || chain.deadline_reached;
        if (chain.best_length < best_length) {
            best_length = chain.best_length;
            best_index = i;
        }
    }

    if (best_index < 0) {
        throw std::runtime_error("parallel run produced no chain results");
    }

    const ChainResult& best_chain = result.chain_results[static_cast<size_t>(best_index)];
    result.best_tour = best_chain.best_tour;
    result.best_length = best_chain.best_length;
    result.final_length_of_best_chain = best_chain.final_length;
    const int checked_best_length = tour_length(result.best_tour, dm);
    if (checked_best_length != result.best_length) {
        throw std::runtime_error("parallel best_length verification failed");
    }

    result.total_elapsed_ms = timer.elapsed_ms();
    result.elapsed_ms = result.total_elapsed_ms;

    return result;
}

}  // namespace tsp
