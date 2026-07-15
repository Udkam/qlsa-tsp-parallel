#include "tsp/mpi_parallel.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef TSP_HAS_OPENMP
#include <omp.h>
#endif

#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {
namespace {

void validate_mpi_parallel_params(const ParallelParams& params) {
    if (params.chains < 1) {
        throw std::invalid_argument("chains must be >= 1");
    }
    if (params.threads < 1) {
        throw std::invalid_argument("threads must be >= 1");
    }
    const int64_t iterations = (params.algorithm == AlgorithmKind::SA)
                                   ? params.sa_params.iterations
                                   : params.qlsa_params.sa.iterations;
    if (iterations < 1) {
        throw std::invalid_argument("iterations must be >= 1");
    }
}

int local_chain_count(int total_chains, int world_size, int world_rank) {
    const int base = total_chains / world_size;
    const int rem = total_chains % world_size;
    return base + (world_rank < rem ? 1 : 0);
}

int chain_offset(int total_chains, int world_size, int world_rank) {
    const int base = total_chains / world_size;
    const int rem = total_chains % world_size;
    return world_rank * base + std::min(world_rank, rem);
}

ChainResult run_one_mpi_chain(const DistanceMatrix& dm,
                              const ParallelParams& params,
                              int local_chain_id,
                              int global_chain_id) {
    ChainResult chain;
    chain.chain_id = global_chain_id;
    chain.seed = chain_seed(params.base_seed, global_chain_id);

    if (params.algorithm == AlgorithmKind::SA) {
        SAParams sa_params = params.sa_params;
        sa_params.seed = chain.seed;
        if (params.chains > 1 && sa_params.use_nearest_neighbor_init) {
            sa_params.nearest_neighbor_start =
                seeded_nearest_neighbor_start(chain.seed, dm.size());
        }
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
    if (params.chains > 1 && qlsa_params.sa.use_nearest_neighbor_init) {
        qlsa_params.sa.nearest_neighbor_start =
            seeded_nearest_neighbor_start(chain.seed, dm.size());
    }
    const QLSAResult result = run_qlsa_2opt(dm, qlsa_params);
    chain.best_tour = result.best_tour;
    chain.best_length = result.best_length;
    chain.final_length = result.final_length;
    chain.elapsed_ms = result.elapsed_ms;
    chain.accepted_moves = result.accepted_moves;
    chain.improved_moves = result.improved_moves;
    chain.iterations_completed = result.iterations_completed;
    chain.deadline_reached = result.deadline_reached;
    (void)local_chain_id;
    return chain;
}

}  // namespace

bool mpi_runtime_available() noexcept {
    return true;
}

MpiParallelResult run_mpi_parallel_chains(const DistanceMatrix& dm, const ParallelParams& params) {
    validate_mpi_parallel_params(params);
    if (dm.size() <= 0) {
        throw std::invalid_argument("run_mpi_parallel_chains requires a non-empty distance matrix");
    }

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error("MPI runtime is not initialized");
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Timer total_timer;

    MpiParallelResult result;
    result.world_rank = world_rank;
    result.world_size = world_size;
    result.mpi_enabled = true;
    result.local_chains = local_chain_count(params.chains, world_size, world_rank);
    result.chain_offset = chain_offset(params.chains, world_size, world_rank);

    result.local.chains = result.local_chains;
    result.local.threads = params.threads;
    result.local.base_seed = params.base_seed;
    result.local.chain_results.resize(static_cast<size_t>(result.local_chains));

    Timer local_timer;
    int local_actual_threads = 1;
#ifdef TSP_HAS_OPENMP
    if (params.threads > 1 && result.local_chains > 1) {
#pragma omp parallel num_threads(params.threads)
        {
#pragma omp single
            local_actual_threads = omp_get_num_threads();
#pragma omp for schedule(static)
            for (int local_id = 0; local_id < result.local_chains; ++local_id) {
                const int global_id = result.chain_offset + local_id;
                result.local.chain_results[static_cast<size_t>(local_id)] =
                    run_one_mpi_chain(dm, params, local_id, global_id);
            }
        }
    } else
#endif
    {
        for (int local_id = 0; local_id < result.local_chains; ++local_id) {
            const int global_id = result.chain_offset + local_id;
            result.local.chain_results[static_cast<size_t>(local_id)] =
                run_one_mpi_chain(dm, params, local_id, global_id);
        }
    }
    result.local.actual_threads = local_actual_threads;
    result.local.elapsed_ms = local_timer.elapsed_ms();

    int local_best_index = -1;
    int local_best_length = std::numeric_limits<int>::max();
    for (int i = 0; i < result.local_chains; ++i) {
        const ChainResult& chain = result.local.chain_results[static_cast<size_t>(i)];
        result.local.total_accepted_moves += chain.accepted_moves;
        result.local.total_improved_moves += chain.improved_moves;
        result.local.total_iterations_completed += chain.iterations_completed;
        result.local.deadline_reached = result.local.deadline_reached || chain.deadline_reached;
        if (chain.best_length < local_best_length) {
            local_best_length = chain.best_length;
            local_best_index = i;
        }
    }

    if (local_best_index >= 0) {
        const ChainResult& best_chain =
            result.local.chain_results[static_cast<size_t>(local_best_index)];
        result.local.best_tour = best_chain.best_tour;
        result.local.best_length = best_chain.best_length;
        result.local.final_length_of_best_chain = best_chain.final_length;
    }

    const double comm_start = MPI_Wtime();

    struct {
        int value;
        int rank;
    } local_min{local_best_length, world_rank}, global_min{0, 0};

    MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    int winner_final_length = 0;
    if (world_rank == global_min.rank) {
        winner_final_length = result.local.final_length_of_best_chain;
    }
    MPI_Bcast(&winner_final_length, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);

    std::vector<int> global_best_tour(static_cast<size_t>(dm.size()), 0);
    if (world_rank == global_min.rank) {
        global_best_tour = result.local.best_tour;
    }
    MPI_Bcast(global_best_tour.data(), dm.size(), MPI_INT, global_min.rank, MPI_COMM_WORLD);

    long long local_accepted = static_cast<long long>(result.local.total_accepted_moves);
    long long local_improved = static_cast<long long>(result.local.total_improved_moves);
    long long local_iterations =
        static_cast<long long>(result.local.total_iterations_completed);
    int local_deadline = result.local.deadline_reached ? 1 : 0;
    long long global_accepted = 0;
    long long global_improved = 0;
    long long global_iterations = 0;
    int global_deadline = 0;
    int global_actual_threads = 1;
    MPI_Allreduce(
        &local_accepted, &global_accepted, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_improved, &global_improved, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_iterations, &global_iterations, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_deadline, &global_deadline, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_actual_threads, &global_actual_threads, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    result.communication_ms = (MPI_Wtime() - comm_start) * 1000.0;

    result.global.best_tour = global_best_tour;
    result.global.best_length = global_min.value;
    result.global.final_length_of_best_chain = winner_final_length;
    result.global.elapsed_ms = total_timer.elapsed_ms();
    result.global.chains = params.chains;
    result.global.threads = params.threads;
    result.global.base_seed = params.base_seed;
    result.global.actual_threads = global_actual_threads;
    result.global.total_iterations_completed = static_cast<int64_t>(global_iterations);
    result.global.deadline_reached = global_deadline != 0;
    result.global.total_accepted_moves = static_cast<int64_t>(global_accepted);
    result.global.total_improved_moves = static_cast<int64_t>(global_improved);

    if (!is_valid_tour(result.global.best_tour, dm.size())) {
        throw std::runtime_error("MPI reduction produced an invalid best tour");
    }
    const int checked_best_length = tour_length(result.global.best_tour, dm);
    if (checked_best_length != result.global.best_length) {
        throw std::runtime_error("MPI best_length verification failed");
    }

    return result;
}

}  // namespace tsp
