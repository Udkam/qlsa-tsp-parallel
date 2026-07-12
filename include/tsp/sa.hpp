#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/rng.hpp"
#include "tsp/tour.hpp"

namespace tsp {

struct SAParams {
    int64_t iterations = 1000000;
    double initial_temperature = 1000.0;
    double final_temperature = 1e-3;
    uint64_t seed = 1;
    bool use_nearest_neighbor_init = true;
    // Direct/single-chain callers retain the historical city-0 start. Parallel
    // coordinators set this from each chain seed to diversify NN initial tours.
    int nearest_neighbor_start = 0;
};

struct SAResult {
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length = 0;
    double elapsed_ms = 0.0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    int64_t iterations_completed = 0;
    bool deadline_reached = false;
};

// Maps a chain seed to a valid NN starting city without consuming the search
// RNG stream. Parallel coordinators use it only when they have multiple
// chains, preserving the legacy city-0 start for standalone runs.
[[nodiscard]] int seeded_nearest_neighbor_start(uint64_t seed, int city_count);

using SearchClock = std::chrono::steady_clock;

// A chunk may be bounded by an iteration count, a relative time limit, an
// absolute deadline, or the earliest of the two time bounds. Parallel callers
// should compute one absolute deadline and share it between all workers.
struct SearchChunkOptions {
    int64_t max_iterations = 0;
    std::optional<int64_t> time_limit_ms;
    std::optional<SearchClock::time_point> deadline;
    int64_t deadline_check_interval = 64;
};

struct SearchChunkProgress {
    int64_t iterations_requested = 0;
    int64_t iterations_completed = 0;
    int64_t total_iterations_completed = 0;
    bool deadline_reached = false;
    bool iteration_budget_exhausted = false;
};

// Complete deterministic state for pausing and resuming one SA chain. Public
// fields intentionally make checkpoint serialization straightforward.
struct SAState {
    SAParams params;
    Rng rng{1};
    Tour current_tour;
    Tour best_tour;
    int current_length = 0;
    int best_length = 0;
    double temperature = 0.0;
    double temperature_decay = 1.0;
    int64_t iterations_completed = 0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    double elapsed_ms = 0.0;
    bool deadline_reached = false;
    bool initialized = false;
};

[[nodiscard]] SAState initialize_sa_state(const DistanceMatrix& dm, const SAParams& params);
SearchChunkProgress run_sa_chunk(const DistanceMatrix& dm,
                                 SAState& state,
                                 const SearchChunkOptions& options);
[[nodiscard]] SAResult finalize_sa_state(const DistanceMatrix& dm, const SAState& state);

// Adopt a legal migrant when it improves the current chain. The annealing
// schedule, RNG stream, iteration budget, and local move statistics are kept.
bool migrate_sa_tour(const DistanceMatrix& dm, SAState& state, const Tour& migrant);

SAResult run_sa_2opt(const DistanceMatrix& dm, const SAParams& params);

}  // namespace tsp
