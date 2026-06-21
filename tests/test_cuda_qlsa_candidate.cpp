#include <cassert>
#include <cstdint>
#include <iostream>
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

tsp::ParallelParams make_params(uint64_t seed, tsp::CudaReversalMode reversal_mode) {
    tsp::ParallelParams params;
    params.algorithm = tsp::AlgorithmKind::QLSA;
    params.chains = 4;
    params.threads = 1;
    params.cuda_enabled = true;
    params.cuda_block_size = 128;
    params.cuda_mode = tsp::CudaMode::Candidate;
    params.cuda_candidates_per_iter = 64;
    params.cuda_reversal_mode = reversal_mode;
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

void assert_valid_square4(const tsp::DistanceMatrix& dm, const tsp::ParallelResult& result) {
    assert(result.chains == 4);
    assert(result.threads == 128);
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));
    assert(result.best_length == tsp::tour_length(result.best_tour, dm));
    assert(result.best_length == 40);
    for (const tsp::ChainResult& chain : result.chain_results) {
        assert(tsp::is_valid_tour(chain.best_tour, dm.size()));
        assert(chain.best_length == tsp::tour_length(chain.best_tour, dm));
    }
}

}  // namespace

int main() {
    if (!tsp::cuda_available()) {
        std::cout << "test_cuda_qlsa_candidate skipped; CUDA is not available\n";
        return 0;
    }

    const tsp::DistanceMatrix dm = load_fixture_dm();

    const tsp::ParallelParams serial = make_params(101, tsp::CudaReversalMode::Serial);
    const tsp::ParallelResult serial_result = tsp::run_parallel_chains(dm, serial);
    assert_valid_square4(dm, serial_result);

    const tsp::ParallelParams parallel = make_params(101, tsp::CudaReversalMode::Parallel);
    const tsp::ParallelResult parallel_result = tsp::run_parallel_chains(dm, parallel);
    assert_valid_square4(dm, parallel_result);

    std::cout << "test_cuda_qlsa_candidate passed\n";
    return 0;
}
