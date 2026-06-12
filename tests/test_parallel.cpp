#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include "tsp/distance_matrix.hpp"
#include "tsp/parallel.hpp"
#include "tsp/tour.hpp"
#include "tsp/tsplib_parser.hpp"

namespace {

tsp::DistanceMatrix load_fixture_dm() {
    const std::string fixture_path = std::string(TEST_SOURCE_DIR) + "/fixtures/square4.tsp";
    return tsp::DistanceMatrix(tsp::load_tsplib(fixture_path));
}

tsp::ParallelParams make_params(tsp::AlgorithmKind algorithm, int chains, int threads, uint64_t seed) {
    tsp::ParallelParams params;
    params.algorithm = algorithm;
    params.chains = chains;
    params.threads = threads;
    params.base_seed = seed;
    params.sa_params.iterations = 1000;
    params.sa_params.seed = seed;
    params.sa_params.use_nearest_neighbor_init = false;
    params.qlsa_params.sa = params.sa_params;
    params.qlsa_params.alpha = 0.1;
    params.qlsa_params.gamma = 0.9;
    params.qlsa_params.epsilon = 0.1;
    params.qlsa_params.policy = "epsilon-greedy";
    return params;
}

void assert_parallel_result_valid(const tsp::DistanceMatrix& dm,
                                  const tsp::ParallelResult& result,
                                  int chains,
                                  uint64_t base_seed) {
    assert(result.chains == chains);
    assert(result.chain_results.size() == static_cast<size_t>(chains));
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));
    assert(result.best_length == tsp::tour_length(result.best_tour, dm));
    assert(result.best_length == 40);

    for (int i = 0; i < chains; ++i) {
        const tsp::ChainResult& chain = result.chain_results[static_cast<size_t>(i)];
        assert(chain.chain_id == i);
        assert(chain.seed == tsp::chain_seed(base_seed, i));
        assert(tsp::is_valid_tour(chain.best_tour, dm.size()));
        assert(chain.best_length == tsp::tour_length(chain.best_tour, dm));
    }
}

}  // namespace

int main() {
    const tsp::DistanceMatrix dm = load_fixture_dm();

    const tsp::ParallelParams sa_params = make_params(tsp::AlgorithmKind::SA, 4, 1, 123);
    const tsp::ParallelResult sa_result = tsp::run_parallel_chains(dm, sa_params);
    assert_parallel_result_valid(dm, sa_result, 4, 123);

    const tsp::ParallelParams qlsa_params = make_params(tsp::AlgorithmKind::QLSA, 4, 1, 123);
    const tsp::ParallelResult qlsa_result = tsp::run_parallel_chains(dm, qlsa_params);
    assert_parallel_result_valid(dm, qlsa_result, 4, 123);

    const tsp::ParallelResult repeat_result = tsp::run_parallel_chains(dm, sa_params);
    assert(repeat_result.best_length == sa_result.best_length);
    for (int i = 0; i < sa_params.chains; ++i) {
        assert(repeat_result.chain_results[static_cast<size_t>(i)].seed ==
               sa_result.chain_results[static_cast<size_t>(i)].seed);
        assert(repeat_result.chain_results[static_cast<size_t>(i)].best_length ==
               sa_result.chain_results[static_cast<size_t>(i)].best_length);
    }

    if (tsp::openmp_available()) {
        tsp::ParallelParams omp_params = make_params(tsp::AlgorithmKind::SA, 4, 2, 123);
        const tsp::ParallelResult omp_result = tsp::run_parallel_chains(dm, omp_params);
        assert_parallel_result_valid(dm, omp_result, 4, 123);

        tsp::ParallelParams omp_qlsa_params = make_params(tsp::AlgorithmKind::QLSA, 4, 2, 123);
        const tsp::ParallelResult omp_qlsa_result = tsp::run_parallel_chains(dm, omp_qlsa_params);
        assert_parallel_result_valid(dm, omp_qlsa_result, 4, 123);
    }

    std::cout << "test_parallel passed\n";
    return 0;
}
