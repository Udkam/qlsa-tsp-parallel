#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/sa.hpp"

namespace tsp {

enum class IslandAlgorithm {
    SA,
    QLSA,
};

enum class MigrationTopology {
    Independent,
    Ring,
    GlobalBest,
};

struct IslandParams {
    IslandAlgorithm algorithm = IslandAlgorithm::SA;
    SAParams sa_params;
    QLSAParams qlsa_params;
    int island_count = 4;
    int threads = 1;
    int64_t migration_interval = 100000;
    uint64_t base_seed = 1;

    // One shared deadline applies to the complete run. If both bounds are set,
    // the earlier deadline wins. This prevents queued OpenMP workers from each
    // receiving a fresh relative time budget.
    std::optional<int64_t> time_limit_ms;
    std::optional<SearchClock::time_point> deadline;
    int64_t deadline_check_interval = 64;
};

struct IslandChainResult {
    int island_id = 0;
    uint64_t seed = 0;
    Tour best_tour;
    Tour final_tour;
    int best_length = 0;
    int final_length = 0;
    int64_t iterations_completed = 0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    int64_t migration_attempts = 0;
    int64_t migrations_adopted = 0;
    bool deadline_reached = false;
};

struct IslandResult {
    Tour best_tour;
    int best_length = 0;
    int final_length_of_best_island = 0;
    double elapsed_ms = 0.0;
    int island_count = 0;
    int threads = 0;
    int actual_threads = 1;
    uint64_t base_seed = 0;
    MigrationTopology topology = MigrationTopology::Independent;
    int64_t iteration_budget_per_island = 0;
    int64_t total_iteration_budget = 0;
    int64_t total_iterations_completed = 0;
    int64_t total_accepted_moves = 0;
    int64_t total_improved_moves = 0;
    int64_t migration_rounds = 0;
    int64_t migration_attempts = 0;
    int64_t migrations_adopted = 0;
    bool deadline_reached = false;
    bool iteration_budget_exhausted = false;
    bool used_openmp = false;
    std::vector<IslandChainResult> islands;
};

[[nodiscard]] uint64_t island_seed(uint64_t base_seed, int island_id) noexcept;

// Runs independent SA/QLSA state machines in synchronized chunks. Ring and
// global-best migration copy a snapshot only between chunks, so workers never
// read another island while it is mutating its state.
IslandResult run_openmp_islands(const DistanceMatrix& dm,
                                const IslandParams& params,
                                MigrationTopology topology);

}  // namespace tsp
