#include "tsp/sa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "tsp/rng.hpp"
#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {
namespace {

void validate_sa_params(const DistanceMatrix& dm, const SAParams& params) {
    if (dm.size() <= 0) {
        throw std::invalid_argument("run_sa_2opt requires a non-empty distance matrix");
    }
    if (params.iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (params.initial_temperature <= 0.0 || params.final_temperature <= 0.0) {
        throw std::invalid_argument("temperatures must be positive");
    }
}

void validate_chunk_options(const SearchChunkOptions& options) {
    if (options.max_iterations < 0) {
        throw std::invalid_argument("chunk max_iterations must be non-negative");
    }
    if (options.time_limit_ms.has_value() && *options.time_limit_ms < 0) {
        throw std::invalid_argument("time_limit_ms must be non-negative");
    }
    if (options.deadline_check_interval <= 0) {
        throw std::invalid_argument("deadline_check_interval must be positive");
    }
}

std::optional<SearchClock::time_point> effective_deadline(const SearchChunkOptions& options) {
    std::optional<SearchClock::time_point> deadline = options.deadline;
    if (!options.time_limit_ms.has_value()) {
        return deadline;
    }

    const SearchClock::time_point now = SearchClock::now();
    const auto max_delta =
        std::chrono::duration_cast<std::chrono::milliseconds>(SearchClock::time_point::max() - now);
    const SearchClock::time_point relative_deadline =
        *options.time_limit_ms >= max_delta.count()
            ? SearchClock::time_point::max()
            : now + std::chrono::milliseconds(*options.time_limit_ms);
    if (!deadline.has_value() || relative_deadline < *deadline) {
        deadline = relative_deadline;
    }
    return deadline;
}

void verify_sa_state(const DistanceMatrix& dm, const SAState& state) {
    if (!state.initialized) {
        throw std::invalid_argument("SA state is not initialized");
    }
    if (state.params.iterations < state.iterations_completed) {
        throw std::runtime_error("SA state exceeded its iteration budget");
    }
    if (tour_length(state.best_tour, dm) != state.best_length) {
        throw std::runtime_error("SA best_length verification failed");
    }
    if (tour_length(state.current_tour, dm) != state.current_length) {
        throw std::runtime_error("SA final_length verification failed");
    }
}

}  // namespace

SAState initialize_sa_state(const DistanceMatrix& dm, const SAParams& params) {
    validate_sa_params(dm, params);
    Timer timer;

    SAState state;
    state.params = params;
    state.rng = Rng(params.seed);
    state.current_tour = params.use_nearest_neighbor_init ? nearest_neighbor_tour(dm)
                                                          : random_tour(dm.size(), state.rng);
    state.current_length = tour_length(state.current_tour, dm);
    state.best_tour = state.current_tour;
    state.best_length = state.current_length;
    state.temperature = params.initial_temperature;
    state.temperature_decay =
        params.iterations > 0
            ? std::pow(params.final_temperature / params.initial_temperature,
                       1.0 / static_cast<double>(params.iterations))
            : 1.0;
    state.initialized = true;
    state.elapsed_ms = timer.elapsed_ms();
    return state;
}

SearchChunkProgress run_sa_chunk(const DistanceMatrix& dm,
                                 SAState& state,
                                 const SearchChunkOptions& options) {
    validate_chunk_options(options);
    if (!state.initialized) {
        throw std::invalid_argument("SA state is not initialized");
    }
    if (static_cast<int>(state.current_tour.size()) != dm.size() ||
        static_cast<int>(state.best_tour.size()) != dm.size()) {
        throw std::invalid_argument("SA state does not match the distance matrix");
    }

    SearchChunkProgress progress;
    const int64_t remaining = state.params.iterations - state.iterations_completed;
    if (remaining < 0) {
        throw std::runtime_error("SA state exceeded its iteration budget");
    }
    progress.iterations_requested = std::min(options.max_iterations, remaining);

    const std::optional<SearchClock::time_point> deadline = effective_deadline(options);
    Timer timer;
    const int n = dm.size();
    while (n >= 3 && progress.iterations_completed < progress.iterations_requested) {
        if (deadline.has_value() &&
            progress.iterations_completed % options.deadline_check_interval == 0 &&
            SearchClock::now() >= *deadline) {
            progress.deadline_reached = true;
            state.deadline_reached = true;
            break;
        }

        int i = 0;
        int k = 0;
        do {
            i = state.rng.uniform_int(0, n - 1);
            k = state.rng.uniform_int(0, n - 1);
            if (i > k) {
                std::swap(i, k);
            }
        } while (!is_valid_2opt_move(n, i, k));

        const int delta = delta_2opt(state.current_tour, dm, i, k);
        bool accept = delta < 0;
        if (!accept) {
            const double probability = std::exp(-static_cast<double>(delta) / state.temperature);
            accept = state.rng.uniform01() < probability;
        }

        if (accept) {
            apply_2opt(state.current_tour, i, k);
            state.current_length += delta;
            ++state.accepted_moves;
            if (delta < 0) {
                ++state.improved_moves;
            }
            if (state.current_length < state.best_length) {
                state.best_length = state.current_length;
                state.best_tour = state.current_tour;
            }
        }

        state.temperature *= state.temperature_decay;
        ++state.iterations_completed;
        ++progress.iterations_completed;
    }

    state.elapsed_ms += timer.elapsed_ms();
    progress.total_iterations_completed = state.iterations_completed;
    progress.iteration_budget_exhausted =
        n < 3 || state.iterations_completed >= state.params.iterations;
    return progress;
}

SAResult finalize_sa_state(const DistanceMatrix& dm, const SAState& state) {
    verify_sa_state(dm, state);
    SAResult result;
    result.best_tour = state.best_tour;
    result.best_length = state.best_length;
    result.final_length = state.current_length;
    result.elapsed_ms = state.elapsed_ms;
    result.accepted_moves = state.accepted_moves;
    result.improved_moves = state.improved_moves;
    result.iterations_completed = state.iterations_completed;
    result.deadline_reached = state.deadline_reached;
    return result;
}

bool migrate_sa_tour(const DistanceMatrix& dm, SAState& state, const Tour& migrant) {
    if (!state.initialized) {
        throw std::invalid_argument("SA state is not initialized");
    }
    if (!is_valid_tour(migrant, dm.size())) {
        throw std::invalid_argument("SA migrant must be a legal tour");
    }
    const int migrant_length = tour_length(migrant, dm);
    if (migrant_length >= state.current_length) {
        return false;
    }
    state.current_tour = migrant;
    state.current_length = migrant_length;
    if (migrant_length < state.best_length) {
        state.best_tour = migrant;
        state.best_length = migrant_length;
    }
    return true;
}

SAResult run_sa_2opt(const DistanceMatrix& dm, const SAParams& params) {
    Timer timer;
    SAState state = initialize_sa_state(dm, params);
    SearchChunkOptions options;
    options.max_iterations = params.iterations;
    (void)run_sa_chunk(dm, state, options);
    SAResult result = finalize_sa_state(dm, state);
    result.elapsed_ms = timer.elapsed_ms();
    return result;
}

}  // namespace tsp
