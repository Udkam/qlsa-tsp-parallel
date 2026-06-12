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
    const int64_t iterations = (params.algorithm == AlgorithmKind::SA)
                                   ? params.sa_params.iterations
                                   : params.qlsa_params.sa.iterations;
    if (iterations < 1) {
        throw std::invalid_argument("iterations must be >= 1");
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
    result.chains = params.chains;
    result.threads = params.threads;
    result.base_seed = params.base_seed;
    result.chain_results.resize(static_cast<size_t>(params.chains));

#ifdef TSP_HAS_OPENMP
    if (params.threads > 1) {
#pragma omp parallel for schedule(static) num_threads(params.threads)
        for (int chain_id_value = 0; chain_id_value < params.chains; ++chain_id_value) {
            result.chain_results[static_cast<size_t>(chain_id_value)] =
                run_one_chain(dm, params, chain_id_value);
        }
    } else
#endif
    {
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
    result.elapsed_ms = timer.elapsed_ms();

    const int checked_best_length = tour_length(result.best_tour, dm);
    if (checked_best_length != result.best_length) {
        throw std::runtime_error("parallel best_length verification failed");
    }

    return result;
}

}  // namespace tsp
