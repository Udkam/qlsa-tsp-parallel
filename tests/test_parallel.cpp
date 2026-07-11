#ifdef NDEBUG
#error "Tests require assertions; NDEBUG must not be defined"
#endif
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
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

}  // namespace

int main() {
    const tsp::DistanceMatrix dm = load_fixture_dm();

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
