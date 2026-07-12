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

tsp::ParallelParams make_cuda_params(tsp::AlgorithmKind algorithm, uint64_t seed) {
    tsp::ParallelParams params;
    params.algorithm = algorithm;
    params.chains = 4;
    params.threads = 1;
    params.cuda_enabled = true;
    params.cuda_block_size = 64;
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

void assert_valid(const tsp::DistanceMatrix& dm, const tsp::ParallelResult& result) {
    assert(result.chains == 4);
    assert(result.chain_results.size() == 4);
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));
    assert(result.best_length == tsp::tour_length(result.best_tour, dm));
    assert(result.best_length == 40);
    assert(result.elapsed_ms == result.total_elapsed_ms);
    assert(result.total_elapsed_ms > 0.0);
    assert(result.requested_backend == tsp::ParallelBackend::Cuda);
    assert(result.total_iterations_completed == 4000);
    assert(!result.deadline_reached);
    for (int i = 0; i < result.chains; ++i) {
        const tsp::ChainResult& chain = result.chain_results[static_cast<size_t>(i)];
        assert(chain.chain_id == i);
        assert(tsp::is_valid_tour(chain.best_tour, dm.size()));
        assert(chain.best_length == tsp::tour_length(chain.best_tour, dm));
        assert(chain.iterations_completed == 1000);
    }
}

}  // namespace

int main() {
    const tsp::DistanceMatrix dm = load_fixture_dm();
    const bool cuda_is_available = tsp::cuda_available();

    tsp::ParallelParams unsupported_variant = make_cuda_params(tsp::AlgorithmKind::QLSA, 19);
    unsupported_variant.qlsa_params.variant = "paper-sb";
    bool rejected_unsupported_variant = false;
    try {
        (void)tsp::run_parallel_chains(dm, unsupported_variant);
    } catch (const std::invalid_argument& error) {
        rejected_unsupported_variant =
            std::string(error.what()).find("variant current only") != std::string::npos;
    }
    assert(rejected_unsupported_variant);

    const tsp::ParallelParams sa_params = make_cuda_params(tsp::AlgorithmKind::SA, 7);
    const tsp::ParallelResult sa_result = tsp::run_parallel_chains(dm, sa_params);
    assert_valid(dm, sa_result);
    if (cuda_is_available) {
        assert(sa_result.actual_backend == tsp::ParallelBackend::Cuda);
        assert(!sa_result.backend_fallback);
        assert(sa_result.backend_fallback_reason.empty());
        assert(sa_result.cuda_kernel_elapsed_ms > 0.0);
        assert(sa_result.total_elapsed_ms >= sa_result.cuda_kernel_elapsed_ms);
        assert(sa_result.actual_threads == sa_params.cuda_block_size);
    } else {
        assert(sa_result.actual_backend == tsp::ParallelBackend::CpuSerial);
        assert(sa_result.backend_fallback);
        assert(!sa_result.backend_fallback_reason.empty());
        assert(sa_result.cuda_kernel_elapsed_ms == 0.0);
        assert(sa_result.actual_threads == 1);
    }

    const tsp::ParallelParams qlsa_params = make_cuda_params(tsp::AlgorithmKind::QLSA, 7);
    const tsp::ParallelResult qlsa_result = tsp::run_parallel_chains(dm, qlsa_params);
    assert_valid(dm, qlsa_result);

    const tsp::ParallelResult repeat_result = tsp::run_parallel_chains(dm, sa_params);
    assert(repeat_result.best_length == sa_result.best_length);
    for (int i = 0; i < sa_result.chains; ++i) {
        assert(repeat_result.chain_results[static_cast<size_t>(i)].seed ==
               sa_result.chain_results[static_cast<size_t>(i)].seed);
        assert(repeat_result.chain_results[static_cast<size_t>(i)].best_length ==
               sa_result.chain_results[static_cast<size_t>(i)].best_length);
    }

    std::cout << "test_cuda passed; cuda_available=" << (cuda_is_available ? "true" : "false") << '\n';
    return 0;
}
