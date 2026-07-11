#ifdef NDEBUG
#error "Tests require assertions; NDEBUG must not be defined"
#endif
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tsp/cuda.hpp"
#include "tsp/distance_matrix.hpp"
#include "tsp/parallel.hpp"
#include "tsp/tour.hpp"
#include "tsp/tsplib_parser.hpp"

namespace {

tsp::DistanceMatrix load_fixture_dm() {
    const std::string fixture_path = std::string(TEST_SOURCE_DIR) + "/fixtures/square4.tsp";
    return tsp::DistanceMatrix(tsp::load_tsplib(fixture_path));
}

tsp::ParallelParams make_candidate_params(uint64_t seed) {
    tsp::ParallelParams params;
    params.algorithm = tsp::AlgorithmKind::SA;
    params.chains = 4;
    params.threads = 1;
    params.cuda_enabled = true;
    params.cuda_block_size = 128;
    params.cuda_mode = tsp::CudaMode::Candidate;
    params.cuda_candidates_per_iter = 64;
    params.cuda_reversal_mode = tsp::CudaReversalMode::Serial;
    params.base_seed = seed;
    params.sa_params.iterations = 500;
    params.sa_params.seed = seed;
    params.sa_params.use_nearest_neighbor_init = true;
    params.qlsa_params.sa = params.sa_params;
    params.qlsa_params.alpha = 0.1;
    params.qlsa_params.gamma = 0.9;
    params.qlsa_params.epsilon = 0.1;
    params.qlsa_params.policy = "epsilon-greedy";
    return params;
}

void assert_candidate_result(const tsp::DistanceMatrix& dm, const tsp::ParallelResult& result) {
    assert(result.chains == 4);
    assert(result.threads == 128);
    assert(result.chain_results.size() == 4);
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));
    assert(result.best_length == tsp::tour_length(result.best_tour, dm));
    assert(result.best_length == 40);
    assert(result.elapsed_ms == result.total_elapsed_ms);
    assert(result.total_elapsed_ms > 0.0);
    assert(result.cuda_kernel_elapsed_ms > 0.0);
    assert(result.total_elapsed_ms >= result.cuda_kernel_elapsed_ms);
    assert(result.requested_backend == tsp::ParallelBackend::Cuda);
    assert(result.actual_backend == tsp::ParallelBackend::Cuda);
    assert(!result.backend_fallback);
    assert(result.backend_fallback_reason.empty());
    for (const tsp::ChainResult& chain : result.chain_results) {
        assert(tsp::is_valid_tour(chain.best_tour, dm.size()));
        assert(chain.best_length == tsp::tour_length(chain.best_tour, dm));
    }
}

}  // namespace

int main() {
    const tsp::DistanceMatrix dm = load_fixture_dm();

    {
        tsp::ParallelParams bad = make_candidate_params(1);
        bad.cuda_candidates_per_iter = 0;
        bool threw = false;
        try {
            (void)tsp::run_parallel_chains(dm, bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    {
        tsp::ParallelParams bad = make_candidate_params(1);
        bad.cuda_candidates_per_iter = bad.cuda_block_size + 1;
        bool threw = false;
        try {
            (void)tsp::run_parallel_chains(dm, bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    if (!tsp::cuda_available()) {
        std::cout << "test_cuda_candidate skipped; CUDA is not available\n";
        return 0;
    }

    const tsp::ParallelParams params = make_candidate_params(11);
    const tsp::ParallelResult result = tsp::run_parallel_chains(dm, params);
    assert_candidate_result(dm, result);

    tsp::ParallelParams parallel_reversal = params;
    parallel_reversal.cuda_reversal_mode = tsp::CudaReversalMode::Parallel;
    const tsp::ParallelResult parallel_result = tsp::run_parallel_chains(dm, parallel_reversal);
    assert_candidate_result(dm, parallel_result);

    tsp::ParallelParams random_policy = params;
    random_policy.cuda_candidate_policy = tsp::CudaCandidatePolicy::Random;
    const tsp::ParallelResult random_result = tsp::run_parallel_chains(dm, random_policy);
    assert_candidate_result(dm, random_result);

    tsp::ParallelParams random_parallel = random_policy;
    random_parallel.cuda_reversal_mode = tsp::CudaReversalMode::Parallel;
    const tsp::ParallelResult random_parallel_result = tsp::run_parallel_chains(dm, random_parallel);
    assert_candidate_result(dm, random_parallel_result);

    tsp::ParallelParams hybrid_policy = params;
    hybrid_policy.cuda_candidate_policy = tsp::CudaCandidatePolicy::Hybrid;
    const tsp::ParallelResult hybrid_result = tsp::run_parallel_chains(dm, hybrid_policy);
    assert_candidate_result(dm, hybrid_result);

    tsp::ParallelParams hybrid_parallel = hybrid_policy;
    hybrid_parallel.cuda_reversal_mode = tsp::CudaReversalMode::Parallel;
    const tsp::ParallelResult hybrid_parallel_result = tsp::run_parallel_chains(dm, hybrid_parallel);
    assert_candidate_result(dm, hybrid_parallel_result);

    tsp::ParallelParams qlsa_candidate = params;
    qlsa_candidate.algorithm = tsp::AlgorithmKind::QLSA;
    const tsp::ParallelResult qlsa_result = tsp::run_parallel_chains(dm, qlsa_candidate);
    assert_candidate_result(dm, qlsa_result);

    const tsp::ParallelResult repeat = tsp::run_parallel_chains(dm, params);
    assert(repeat.best_length == result.best_length);
    assert(repeat.total_accepted_moves == result.total_accepted_moves);
    assert(repeat.total_improved_moves == result.total_improved_moves);

    std::cout << "test_cuda_candidate passed\n";
    return 0;
}
