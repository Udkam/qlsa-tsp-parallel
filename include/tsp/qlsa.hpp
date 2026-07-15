#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/rng.hpp"
#include "tsp/sa.hpp"

namespace tsp {

constexpr int kQLSAStateCount = 5;

struct QLSAAction {
    std::string name;
    double min_span_ratio = 0.0;
    double max_span_ratio = 1.0;
};

struct QLSAParams {
    SAParams sa;
    double alpha = 0.1;
    double gamma = 0.9;
    std::string policy = "epsilon-greedy";
    double epsilon = 0.1;
    double softmax_temperature = 1.0;
    // current: existing span/action engineering variant.
    // paper: stateless candidate-leader variant.
    // paper-sb: candidate-leader variant with diversity state.
    std::string variant = "current";
    int state_window = 8;
    double delta_scale = 10.0;
    double diversity_threshold = 0.5;
    // edge is the default engineering metric for symmetric TSP. hamming keeps
    // the paper's position-wise State-Based QLSA definition available for
    // faithful reproduction of historical experiments.
    std::string diversity_metric = "edge";
    std::vector<QLSAAction> actions;
};

struct QLSAResult {
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length = 0;
    double elapsed_ms = 0.0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    int64_t iterations_completed = 0;
    bool deadline_reached = false;
    std::vector<std::vector<double>> q_table;
    std::vector<int64_t> action_counts;
};

// Complete deterministic state for all current QLSA variants. Keeping the RNG,
// temperature, Q table, diversity state, and delta window makes chunked runs
// bit-for-bit equivalent to an uninterrupted run with the same parameters.
struct QLSAState {
    QLSAParams params;
    std::vector<QLSAAction> actions;
    Rng rng{1};
    Tour current_tour;
    Tour best_tour;
    int current_length = 0;
    int best_length = 0;
    double temperature = 0.0;
    double temperature_decay = 1.0;
    std::vector<std::vector<double>> q_table;
    std::vector<int64_t> action_counts;
    // Reused by softmax action selection. It is derived workspace rather than
    // learning state, but belongs to the chain so parallel chains never share
    // mutable scratch storage.
    std::vector<double> softmax_weights;
    std::vector<int> recent_deltas;
    double recent_delta_sum = 0.0;
    // Index of the oldest entry once recent_deltas reaches state_window. The
    // vector is then a fixed-capacity circular window, avoiding an O(window)
    // erase from the hot loop.
    size_t recent_delta_head = 0;
    // For paper-sb + edge, stores the two incident best-tour cities for each
    // city index. It is allocated once per search state so diversity updates do
    // not allocate a hash table on every iteration.
    std::vector<int> edge_diversity_best_neighbors;
    int learning_state = 0;
    int64_t iterations_completed = 0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    double elapsed_ms = 0.0;
    bool deadline_reached = false;
    bool initialized = false;
};

// Validates the parameter contract shared by CPU and accelerator backends.
// Backend-specific limits are checked by the corresponding backend entrypoint.
void validate_qlsa_params(const QLSAParams& params);
[[nodiscard]] std::vector<QLSAAction> default_qlsa_actions();
[[nodiscard]] int qlsa_state_from_average_delta(double average_delta, double delta_scale);
[[nodiscard]] double qlsa_reward_from_delta(int delta, bool accepted);
// Returns a normalized diversity ratio in [0, 1]. Supported metrics are
// "edge" (undirected cycle-edge difference) and "hamming" (paper-compatible
// position-wise difference).
[[nodiscard]] double qlsa_diversity_ratio(const Tour& current,
                                          const Tour& best,
                                          const std::string& metric);
void update_q_value(std::vector<std::vector<double>>& q_table,
                    int state,
                    int action,
                    int next_state,
                    double reward,
                    double alpha,
                    double gamma);
[[nodiscard]] int select_qlsa_action(const std::vector<double>& q_values,
                                      const QLSAParams& params,
                                      Rng& rng,
                                      std::vector<double>* softmax_weights = nullptr);

[[nodiscard]] QLSAState initialize_qlsa_state(const DistanceMatrix& dm,
                                              const QLSAParams& params);
SearchChunkProgress run_qlsa_chunk(const DistanceMatrix& dm,
                                   QLSAState& state,
                                   const SearchChunkOptions& options);
[[nodiscard]] QLSAResult finalize_qlsa_state(const DistanceMatrix& dm,
                                             const QLSAState& state);
bool migrate_qlsa_tour(const DistanceMatrix& dm, QLSAState& state, const Tour& migrant);

QLSAResult run_qlsa_2opt(const DistanceMatrix& dm, const QLSAParams& params);

}  // namespace tsp
