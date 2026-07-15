#ifdef NDEBUG
#error "Tests require assertions; NDEBUG must not be defined"
#endif

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include "tsp/mpi_parallel.hpp"
#include "tsp/tour.hpp"
#include "tsp/tsplib_parser.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const std::string fixture_path = std::string(TEST_SOURCE_DIR) + "/fixtures/square4.tsp";
    const tsp::DistanceMatrix dm(tsp::load_tsplib(fixture_path));

    tsp::ParallelParams params;
    params.algorithm = tsp::AlgorithmKind::SA;
    params.chains = 5;
    params.threads = 2;
    params.base_seed = 29;
    params.sa_params.iterations = 37;
    params.sa_params.initial_temperature = 20.0;
    params.sa_params.final_temperature = 0.01;
    params.sa_params.use_nearest_neighbor_init = true;

    const tsp::MpiParallelResult result = tsp::run_mpi_parallel_chains(dm, params);

    assert(result.mpi_enabled);
    assert(result.local.chain_results.size() == static_cast<size_t>(result.local_chains));
    int64_t counted_local_iterations = 0;
    for (const tsp::ChainResult& chain : result.local.chain_results) {
        assert(chain.iterations_completed == params.sa_params.iterations);
        assert(!chain.deadline_reached);
        counted_local_iterations += chain.iterations_completed;
    }
    assert(result.local.total_iterations_completed == counted_local_iterations);
    assert(!result.local.deadline_reached);
    assert(result.local.actual_threads == params.threads);

    long long expected_global_accepted = 0;
    long long expected_global_improved = 0;
    const long long local_accepted = result.local.total_accepted_moves;
    const long long local_improved = result.local.total_improved_moves;
    MPI_Allreduce(
        &local_accepted, &expected_global_accepted, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_improved, &expected_global_improved, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    assert(result.global.total_iterations_completed ==
           params.sa_params.iterations * static_cast<int64_t>(params.chains));
    assert(!result.global.deadline_reached);
    assert(result.global.actual_threads == params.threads);
    assert(result.global.total_accepted_moves == expected_global_accepted);
    assert(result.global.total_improved_moves == expected_global_improved);
    assert(tsp::is_valid_tour(result.global.best_tour, dm.size()));
    assert(result.global.best_length == tsp::tour_length(result.global.best_tour, dm));

    tsp::ParallelParams qlsa_params = params;
    qlsa_params.algorithm = tsp::AlgorithmKind::QLSA;
    qlsa_params.qlsa_params.sa = params.sa_params;
    qlsa_params.qlsa_params.sa.iterations = 23;
    const tsp::MpiParallelResult qlsa_result =
        tsp::run_mpi_parallel_chains(dm, qlsa_params);
    assert(qlsa_result.global.total_iterations_completed ==
           qlsa_params.qlsa_params.sa.iterations * static_cast<int64_t>(params.chains));
    assert(qlsa_result.global.actual_threads == params.threads);
    assert(!qlsa_result.global.deadline_reached);
    assert(tsp::is_valid_tour(qlsa_result.global.best_tour, dm.size()));

    if (world_rank == 0) {
        std::cout << "test_mpi_parallel passed with " << result.world_size << " ranks\n";
    }
    MPI_Finalize();
    return 0;
}
