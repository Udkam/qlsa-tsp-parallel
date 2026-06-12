#pragma once

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
    int state_window = 8;
    double delta_scale = 10.0;
    std::vector<QLSAAction> actions;
};

struct QLSAResult {
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length = 0;
    double elapsed_ms = 0.0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    std::vector<std::vector<double>> q_table;
    std::vector<int64_t> action_counts;
};

[[nodiscard]] std::vector<QLSAAction> default_qlsa_actions();
[[nodiscard]] int qlsa_state_from_average_delta(double average_delta, double delta_scale);
[[nodiscard]] double qlsa_reward_from_delta(int delta, bool accepted);
void update_q_value(std::vector<std::vector<double>>& q_table,
                    int state,
                    int action,
                    int next_state,
                    double reward,
                    double alpha,
                    double gamma);
[[nodiscard]] int select_qlsa_action(const std::vector<double>& q_values,
                                     const QLSAParams& params,
                                     Rng& rng);

QLSAResult run_qlsa_2opt(const DistanceMatrix& dm, const QLSAParams& params);

}  // namespace tsp
