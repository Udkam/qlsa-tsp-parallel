#include "tsp/island.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <variant>

#ifdef TSP_HAS_OPENMP
#include <omp.h>
#endif

#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {
namespace {

using IslandState = std::variant<SAState, QLSAState>;

void validate_island_params(const DistanceMatrix& dm, const IslandParams& params) {
    if (dm.size() <= 0) {
        throw std::invalid_argument("run_openmp_islands requires a non-empty distance matrix");
    }
    if (params.island_count < 1) {
        throw std::invalid_argument("island_count must be >= 1");
    }
    if (params.threads < 1) {
        throw std::invalid_argument("threads must be >= 1");
    }
    if (params.migration_interval < 1) {
        throw std::invalid_argument("migration_interval must be >= 1");
    }
    if (params.deadline_check_interval < 1) {
        throw std::invalid_argument("deadline_check_interval must be >= 1");
    }
    if (params.time_limit_ms.has_value() && *params.time_limit_ms < 0) {
        throw std::invalid_argument("time_limit_ms must be non-negative");
    }
    const int64_t iterations = params.algorithm == IslandAlgorithm::SA
                                   ? params.sa_params.iterations
                                   : params.qlsa_params.sa.iterations;
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (iterations > std::numeric_limits<int64_t>::max() / params.island_count) {
        throw std::overflow_error("total island iteration budget overflows int64_t");
    }
}

std::optional<SearchClock::time_point> shared_deadline(const IslandParams& params) {
    std::optional<SearchClock::time_point> deadline = params.deadline;
    if (!params.time_limit_ms.has_value()) {
        return deadline;
    }
    const SearchClock::time_point now = SearchClock::now();
    const auto max_delta =
        std::chrono::duration_cast<std::chrono::milliseconds>(SearchClock::time_point::max() - now);
    const SearchClock::time_point relative_deadline =
        *params.time_limit_ms >= max_delta.count()
            ? SearchClock::time_point::max()
            : now + std::chrono::milliseconds(*params.time_limit_ms);
    if (!deadline.has_value() || relative_deadline < *deadline) {
        deadline = relative_deadline;
    }
    return deadline;
}

int64_t iteration_budget(const IslandState& state) {
    if (const auto* sa = std::get_if<SAState>(&state)) {
        return sa->params.iterations;
    }
    return std::get<QLSAState>(state).params.sa.iterations;
}

int64_t iterations_completed(const IslandState& state) {
    if (const auto* sa = std::get_if<SAState>(&state)) {
        return sa->iterations_completed;
    }
    return std::get<QLSAState>(state).iterations_completed;
}

const Tour& best_tour(const IslandState& state) {
    if (const auto* sa = std::get_if<SAState>(&state)) {
        return sa->best_tour;
    }
    return std::get<QLSAState>(state).best_tour;
}

int best_length(const IslandState& state) {
    if (const auto* sa = std::get_if<SAState>(&state)) {
        return sa->best_length;
    }
    return std::get<QLSAState>(state).best_length;
}

SearchChunkProgress run_state_chunk(const DistanceMatrix& dm,
                                    IslandState& state,
                                    const SearchChunkOptions& options) {
    if (auto* sa = std::get_if<SAState>(&state)) {
        return run_sa_chunk(dm, *sa, options);
    }
    return run_qlsa_chunk(dm, std::get<QLSAState>(state), options);
}

bool migrate_state(const DistanceMatrix& dm, IslandState& state, const Tour& migrant) {
    if (auto* sa = std::get_if<SAState>(&state)) {
        return migrate_sa_tour(dm, *sa, migrant);
    }
    return migrate_qlsa_tour(dm, std::get<QLSAState>(state), migrant);
}

bool deadline_expired(const std::optional<SearchClock::time_point>& deadline) {
    return deadline.has_value() && SearchClock::now() >= *deadline;
}

}  // namespace

uint64_t island_seed(uint64_t base_seed, int island_id) noexcept {
    constexpr uint64_t stride = 0x9E3779B97F4A7C15ULL;
    uint64_t value = base_seed + stride * static_cast<uint64_t>(island_id + 1);
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

IslandResult run_openmp_islands(const DistanceMatrix& dm,
                                const IslandParams& params,
                                MigrationTopology topology) {
    validate_island_params(dm, params);
    Timer timer;
    const std::optional<SearchClock::time_point> deadline = shared_deadline(params);
    const int64_t budget = params.algorithm == IslandAlgorithm::SA
                               ? params.sa_params.iterations
                               : params.qlsa_params.sa.iterations;

    std::vector<IslandState> states;
    states.reserve(static_cast<size_t>(params.island_count));
    for (int island_id = 0; island_id < params.island_count; ++island_id) {
        const uint64_t seed = island_seed(params.base_seed, island_id);
        if (params.algorithm == IslandAlgorithm::SA) {
            SAParams chain_params = params.sa_params;
            chain_params.seed = seed;
            if (params.island_count > 1 && chain_params.use_nearest_neighbor_init) {
                chain_params.nearest_neighbor_start =
                    seeded_nearest_neighbor_start(seed, dm.size());
            }
            states.emplace_back(initialize_sa_state(dm, chain_params));
        } else {
            QLSAParams chain_params = params.qlsa_params;
            chain_params.sa.seed = seed;
            if (params.island_count > 1 && chain_params.sa.use_nearest_neighbor_init) {
                chain_params.sa.nearest_neighbor_start =
                    seeded_nearest_neighbor_start(seed, dm.size());
            }
            states.emplace_back(initialize_qlsa_state(dm, chain_params));
        }
    }

    std::vector<int64_t> migration_attempts(static_cast<size_t>(params.island_count), 0);
    std::vector<int64_t> migrations_adopted(static_cast<size_t>(params.island_count), 0);
    std::vector<SearchChunkProgress> progress(static_cast<size_t>(params.island_count));
    int64_t migration_rounds = 0;
    bool reached_deadline = false;
    int minimum_actual_threads = 1;
    bool entered_openmp_region = false;

    for (;;) {
        bool any_remaining = false;
        for (const IslandState& state : states) {
            if (dm.size() >= 3 && iterations_completed(state) < iteration_budget(state)) {
                any_remaining = true;
                break;
            }
        }
        if (!any_remaining) {
            break;
        }
        if (deadline_expired(deadline)) {
            reached_deadline = true;
            break;
        }

#ifdef TSP_HAS_OPENMP
        if (params.threads > 1 && params.island_count > 1) {
            int round_actual_threads = 1;
#pragma omp parallel num_threads(params.threads)
            {
#pragma omp single
                round_actual_threads = omp_get_num_threads();
#pragma omp for schedule(static)
                for (int island_id = 0; island_id < params.island_count; ++island_id) {
                    const int64_t remaining =
                        iteration_budget(states[static_cast<size_t>(island_id)]) -
                        iterations_completed(states[static_cast<size_t>(island_id)]);
                    SearchChunkOptions options;
                    options.max_iterations = std::min(params.migration_interval, remaining);
                    options.deadline = deadline;
                    options.deadline_check_interval = params.deadline_check_interval;
                    progress[static_cast<size_t>(island_id)] =
                        run_state_chunk(dm, states[static_cast<size_t>(island_id)], options);
                }
            }
            minimum_actual_threads = entered_openmp_region
                                         ? std::min(minimum_actual_threads, round_actual_threads)
                                         : round_actual_threads;
            entered_openmp_region = true;
        } else
#endif
        {
            for (int island_id = 0; island_id < params.island_count; ++island_id) {
                const int64_t remaining =
                    iteration_budget(states[static_cast<size_t>(island_id)]) -
                    iterations_completed(states[static_cast<size_t>(island_id)]);
                SearchChunkOptions options;
                options.max_iterations = std::min(params.migration_interval, remaining);
                options.deadline = deadline;
                options.deadline_check_interval = params.deadline_check_interval;
                progress[static_cast<size_t>(island_id)] =
                    run_state_chunk(dm, states[static_cast<size_t>(island_id)], options);
            }
        }

        bool made_progress = false;
        for (const SearchChunkProgress& chunk : progress) {
            made_progress = made_progress || chunk.iterations_completed > 0;
            reached_deadline = reached_deadline || chunk.deadline_reached;
        }
        reached_deadline = reached_deadline || deadline_expired(deadline);
        if (reached_deadline || !made_progress) {
            break;
        }

        bool all_finished = true;
        for (const IslandState& state : states) {
            all_finished = all_finished && iterations_completed(state) >= iteration_budget(state);
        }
        if (all_finished) {
            break;
        }

        if (topology == MigrationTopology::Ring) {
            std::vector<Tour> snapshots;
            snapshots.reserve(states.size());
            for (const IslandState& state : states) {
                snapshots.push_back(best_tour(state));
            }
            for (int receiver = 0; receiver < params.island_count; ++receiver) {
                const int source = (receiver + params.island_count - 1) % params.island_count;
                ++migration_attempts[static_cast<size_t>(receiver)];
                if (migrate_state(dm,
                                  states[static_cast<size_t>(receiver)],
                                  snapshots[static_cast<size_t>(source)])) {
                    ++migrations_adopted[static_cast<size_t>(receiver)];
                }
            }
            ++migration_rounds;
        } else if (topology == MigrationTopology::GlobalBest) {
            int winner = 0;
            for (int island_id = 1; island_id < params.island_count; ++island_id) {
                if (best_length(states[static_cast<size_t>(island_id)]) <
                    best_length(states[static_cast<size_t>(winner)])) {
                    winner = island_id;
                }
            }
            const Tour snapshot = best_tour(states[static_cast<size_t>(winner)]);
            for (int receiver = 0; receiver < params.island_count; ++receiver) {
                if (receiver == winner) {
                    continue;
                }
                ++migration_attempts[static_cast<size_t>(receiver)];
                if (migrate_state(dm, states[static_cast<size_t>(receiver)], snapshot)) {
                    ++migrations_adopted[static_cast<size_t>(receiver)];
                }
            }
            ++migration_rounds;
        }
    }

    IslandResult result;
    result.island_count = params.island_count;
    result.threads = params.threads;
    result.base_seed = params.base_seed;
    result.topology = topology;
    result.actual_threads = entered_openmp_region ? minimum_actual_threads : 1;
    result.iteration_budget_per_island = budget;
    result.total_iteration_budget = budget * params.island_count;
    result.migration_rounds = migration_rounds;
    result.deadline_reached = reached_deadline;
#ifdef TSP_HAS_OPENMP
    result.used_openmp = result.actual_threads > 1;
#endif
    result.islands.reserve(states.size());

    int winner = -1;
    int winner_length = std::numeric_limits<int>::max();
    for (int island_id = 0; island_id < params.island_count; ++island_id) {
        IslandChainResult chain;
        chain.island_id = island_id;
        chain.seed = island_seed(params.base_seed, island_id);
        chain.migration_attempts = migration_attempts[static_cast<size_t>(island_id)];
        chain.migrations_adopted = migrations_adopted[static_cast<size_t>(island_id)];

        if (const auto* sa = std::get_if<SAState>(&states[static_cast<size_t>(island_id)])) {
            const SAResult finalized = finalize_sa_state(dm, *sa);
            chain.best_tour = finalized.best_tour;
            chain.final_tour = sa->current_tour;
            chain.best_length = finalized.best_length;
            chain.final_length = finalized.final_length;
            chain.iterations_completed = finalized.iterations_completed;
            chain.accepted_moves = finalized.accepted_moves;
            chain.improved_moves = finalized.improved_moves;
            chain.deadline_reached = finalized.deadline_reached;
        } else {
            const QLSAState& qlsa = std::get<QLSAState>(states[static_cast<size_t>(island_id)]);
            const QLSAResult finalized = finalize_qlsa_state(dm, qlsa);
            chain.best_tour = finalized.best_tour;
            chain.final_tour = qlsa.current_tour;
            chain.best_length = finalized.best_length;
            chain.final_length = finalized.final_length;
            chain.iterations_completed = finalized.iterations_completed;
            chain.accepted_moves = finalized.accepted_moves;
            chain.improved_moves = finalized.improved_moves;
            chain.deadline_reached = finalized.deadline_reached;
        }

        result.total_iterations_completed += chain.iterations_completed;
        result.total_accepted_moves += chain.accepted_moves;
        result.total_improved_moves += chain.improved_moves;
        result.migration_attempts += chain.migration_attempts;
        result.migrations_adopted += chain.migrations_adopted;
        if (chain.best_length < winner_length) {
            winner_length = chain.best_length;
            winner = island_id;
        }
        result.islands.push_back(std::move(chain));
    }

    if (winner < 0) {
        throw std::runtime_error("island run produced no result");
    }
    const IslandChainResult& best = result.islands[static_cast<size_t>(winner)];
    result.best_tour = best.best_tour;
    result.best_length = best.best_length;
    result.final_length_of_best_island = best.final_length;
    result.iteration_budget_exhausted =
        result.total_iterations_completed == result.total_iteration_budget;
    result.elapsed_ms = timer.elapsed_ms();

    if (!is_valid_tour(result.best_tour, dm.size()) ||
        tour_length(result.best_tour, dm) != result.best_length) {
        throw std::runtime_error("island best tour verification failed");
    }
    return result;
}

}  // namespace tsp
