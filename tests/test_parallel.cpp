#ifdef NDEBUG
#error "Tests require assertions; NDEBUG must not be defined"
#endif
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
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
    assert(result.elapsed_ms == result.total_elapsed_ms);
    assert(result.total_elapsed_ms > 0.0);
    assert(result.cuda_kernel_elapsed_ms == 0.0);
    assert(result.total_iterations_completed == 1000LL * chains);
    assert(!result.deadline_reached);

    for (int i = 0; i < chains; ++i) {
        const tsp::ChainResult& chain = result.chain_results[static_cast<size_t>(i)];
        assert(chain.chain_id == i);
        assert(chain.seed == tsp::chain_seed(base_seed, i));
        assert(tsp::is_valid_tour(chain.best_tour, dm.size()));
        assert(chain.best_length == tsp::tour_length(chain.best_tour, dm));
        assert(chain.iterations_completed == 1000);
        assert(!chain.deadline_reached);
    }
}

uint64_t base_seed_with_distinct_nn_starts(const tsp::DistanceMatrix& dm, int chains) {
    for (uint64_t base_seed = 0; base_seed < 100000; ++base_seed) {
        std::set<int> starts;
        for (int chain_id = 0; chain_id < chains; ++chain_id) {
            starts.insert(tsp::seeded_nearest_neighbor_start(
                tsp::chain_seed(base_seed, chain_id), dm.size()));
        }
        if (static_cast<int>(starts.size()) == chains) {
            return base_seed;
        }
    }
    assert(false && "expected a base seed with distinct seeded NN starts");
    return 0;
}

void test_seeded_nearest_neighbor_initialization(const tsp::DistanceMatrix& dm) {
    constexpr int kChains = 4;
    const uint64_t base_seed = base_seed_with_distinct_nn_starts(dm, kChains);

    tsp::ParallelParams params = make_params(tsp::AlgorithmKind::SA, kChains, 1, base_seed);
    params.sa_params.iterations = 1;
    params.sa_params.initial_temperature = 1e-12;
    params.sa_params.final_temperature = 1e-12;
    params.sa_params.use_nearest_neighbor_init = true;
    params.qlsa_params.sa = params.sa_params;

    const tsp::ParallelResult first = tsp::run_parallel_chains(dm, params);
    const tsp::ParallelResult repeat = tsp::run_parallel_chains(dm, params);
    std::set<int> starts;
    for (int chain_id = 0; chain_id < kChains; ++chain_id) {
        const tsp::ChainResult& chain = first.chain_results[static_cast<size_t>(chain_id)];
        const int expected_start = tsp::seeded_nearest_neighbor_start(chain.seed, dm.size());
        assert(chain.best_tour.front() == expected_start);
        assert(chain.best_tour == repeat.chain_results[static_cast<size_t>(chain_id)].best_tour);
        starts.insert(expected_start);
    }
    assert(static_cast<int>(starts.size()) == kChains);

    tsp::ParallelParams qlsa_params = params;
    qlsa_params.algorithm = tsp::AlgorithmKind::QLSA;
    qlsa_params.qlsa_params.sa = qlsa_params.sa_params;
    const tsp::ParallelResult qlsa = tsp::run_parallel_chains(dm, qlsa_params);
    for (int chain_id = 0; chain_id < kChains; ++chain_id) {
        const tsp::ChainResult& chain = qlsa.chain_results[static_cast<size_t>(chain_id)];
        assert(chain.best_tour.front() ==
               tsp::seeded_nearest_neighbor_start(chain.seed, dm.size()));
    }

    tsp::ParallelParams single = params;
    single.chains = 1;
    single.base_seed = base_seed;
    const tsp::ParallelResult standalone = tsp::run_parallel_chains(dm, single);
    assert(standalone.chain_results.front().best_tour.front() == 0);
}

}  // namespace

int main() {
    const tsp::DistanceMatrix dm = load_fixture_dm();

    test_seeded_nearest_neighbor_initialization(dm);

    assert(std::string(tsp::parallel_backend_name(tsp::ParallelBackend::CpuSerial)) ==
           "cpu_serial");
    assert(std::string(tsp::parallel_backend_name(tsp::ParallelBackend::OpenMP)) ==
           "openmp");
    assert(std::string(tsp::parallel_backend_name(tsp::ParallelBackend::Cuda)) ==
           "cuda");

    const tsp::ParallelParams sa_params = make_params(tsp::AlgorithmKind::SA, 4, 1, 123);
    const tsp::ParallelResult sa_result = tsp::run_parallel_chains(dm, sa_params);
    assert_parallel_result_valid(dm, sa_result, 4, 123);
    assert(sa_result.requested_backend == tsp::ParallelBackend::CpuSerial);
    assert(sa_result.actual_backend == tsp::ParallelBackend::CpuSerial);
    assert(!sa_result.backend_fallback);
    assert(sa_result.backend_fallback_reason.empty());
    assert(sa_result.actual_threads == 1);

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

    tsp::ParallelParams omp_params = make_params(tsp::AlgorithmKind::SA, 4, 2, 123);
    const tsp::ParallelResult omp_result = tsp::run_parallel_chains(dm, omp_params);
    assert_parallel_result_valid(dm, omp_result, 4, 123);
    assert(omp_result.requested_backend == tsp::ParallelBackend::OpenMP);
    if (tsp::openmp_available() && omp_result.actual_threads > 1) {
        assert(omp_result.actual_backend == tsp::ParallelBackend::OpenMP);
        assert(omp_result.actual_threads == 2);
        assert(!omp_result.backend_fallback);
        assert(omp_result.backend_fallback_reason.empty());
        tsp::ParallelParams omp_qlsa_params = make_params(tsp::AlgorithmKind::QLSA, 4, 2, 123);
        const tsp::ParallelResult omp_qlsa_result = tsp::run_parallel_chains(dm, omp_qlsa_params);
        assert_parallel_result_valid(dm, omp_qlsa_result, 4, 123);
    } else {
        assert(omp_result.actual_backend == tsp::ParallelBackend::CpuSerial);
        assert(omp_result.backend_fallback);
        assert(!omp_result.backend_fallback_reason.empty());
        assert(omp_result.actual_threads == 1);
    }

    tsp::Instance tiny_instance;
    tiny_instance.name = "tiny2";
    tiny_instance.type = "TSP";
    tiny_instance.dimension = 2;
    tiny_instance.edge_weight_type = "EUC_2D";
    tiny_instance.coords = {{0.0, 0.0}, {3.0, 4.0}};
    const tsp::DistanceMatrix tiny_dm(tiny_instance);
    const tsp::ParallelResult tiny_result = tsp::run_parallel_chains(
        tiny_dm, make_params(tsp::AlgorithmKind::SA, 2, 1, 456));
    assert(tiny_result.total_iterations_completed == 0);
    for (const tsp::ChainResult& chain : tiny_result.chain_results) {
        assert(chain.iterations_completed == 0);
    }

    tsp::ParallelParams overflowing = make_params(tsp::AlgorithmKind::SA, 2, 1, 789);
    overflowing.sa_params.iterations = std::numeric_limits<int64_t>::max();
    bool overflow_rejected = false;
    try {
        (void)tsp::run_parallel_chains(dm, overflowing);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    assert(overflow_rejected);

    std::cout << "test_parallel passed\n";
    return 0;
}
