#include "tsp/qlsa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {
namespace {

struct Move {
    int i = 0;
    int k = 1;
};

void validate_params(const QLSAParams& params) {
    if (params.alpha < 0.0 || params.alpha > 1.0) {
        throw std::invalid_argument("QLSA alpha must be in [0, 1]");
    }
    if (params.gamma < 0.0 || params.gamma > 1.0) {
        throw std::invalid_argument("QLSA gamma must be in [0, 1]");
    }
    if (params.epsilon < 0.0 || params.epsilon > 1.0) {
        throw std::invalid_argument("QLSA epsilon must be in [0, 1]");
    }
    if (params.policy != "epsilon-greedy" && params.policy != "softmax") {
        throw std::invalid_argument("QLSA policy must be epsilon-greedy or softmax");
    }
    if (params.variant != "current" && params.variant != "paper" && params.variant != "paper-sb") {
        throw std::invalid_argument("QLSA variant must be current, paper, or paper-sb");
    }
    if (params.softmax_temperature <= 0.0) {
        throw std::invalid_argument("QLSA softmax_temperature must be positive");
    }
    if (params.state_window <= 0) {
        throw std::invalid_argument("QLSA state_window must be positive");
    }
    if (params.delta_scale <= 0.0) {
        throw std::invalid_argument("QLSA delta_scale must be positive");
    }
    if (params.diversity_threshold < 0.0 || params.diversity_threshold > 1.0) {
        throw std::invalid_argument("QLSA diversity_threshold must be in [0, 1]");
    }
    if (params.sa.iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (params.sa.initial_temperature <= 0.0 || params.sa.final_temperature <= 0.0) {
        throw std::invalid_argument("temperatures must be positive");
    }
}

std::vector<QLSAAction> normalized_actions(const QLSAParams& params) {
    std::vector<QLSAAction> actions = params.actions.empty() ? default_qlsa_actions() : params.actions;
    for (const QLSAAction& action : actions) {
        if (action.name.empty()) {
            throw std::invalid_argument("QLSA action name must not be empty");
        }
        if (action.min_span_ratio < 0.0 || action.max_span_ratio <= 0.0 ||
            action.min_span_ratio > 1.0 || action.max_span_ratio > 1.0 ||
            action.min_span_ratio > action.max_span_ratio) {
            throw std::invalid_argument("QLSA action span ratios must satisfy 0 <= min <= max <= 1");
        }
    }
    if (actions.empty()) {
        throw std::invalid_argument("QLSA requires at least one action");
    }
    return actions;
}

int span_bound(double ratio, int n, bool upper) {
    const double scaled = ratio * static_cast<double>(n);
    const int value = upper ? static_cast<int>(std::ceil(scaled)) : static_cast<int>(std::floor(scaled));
    return std::clamp(value, 2, n - 1);
}

Move sample_move_for_action(int n, const QLSAAction& action, Rng& rng) {
    if (n < 3) {
        return Move{};
    }

    int low = span_bound(action.min_span_ratio, n, false);
    int high = span_bound(action.max_span_ratio, n, true);
    if (low > high) {
        low = 2;
        high = n - 1;
    }

    for (int attempt = 0; attempt < 16; ++attempt) {
        const int span = rng.uniform_int(low, high);
        const int i = rng.uniform_int(0, n - span);
        const int k = i + span - 1;
        if (is_valid_2opt_move(n, i, k)) {
            return Move{i, k};
        }
    }

    for (;;) {
        int i = rng.uniform_int(0, n - 1);
        int k = rng.uniform_int(0, n - 1);
        if (i > k) {
            std::swap(i, k);
        }
        if (is_valid_2opt_move(n, i, k)) {
            return Move{i, k};
        }
    }
}

double max_q_value(const std::vector<double>& q_values) {
    if (q_values.empty()) {
        throw std::invalid_argument("empty Q row");
    }
    return *std::max_element(q_values.begin(), q_values.end());
}

int best_action_index(const std::vector<double>& q_values) {
    if (q_values.empty()) {
        throw std::invalid_argument("empty Q row");
    }
    int best = 0;
    for (int i = 1; i < static_cast<int>(q_values.size()); ++i) {
        if (q_values[static_cast<size_t>(i)] > q_values[static_cast<size_t>(best)]) {
            best = i;
        }
    }
    return best;
}

Move sample_random_2opt_move(int n, Rng& rng) {
    if (n < 3) {
        return Move{};
    }
    for (;;) {
        int i = rng.uniform_int(0, n - 1);
        int k = rng.uniform_int(0, n - 1);
        if (i > k) {
            std::swap(i, k);
        }
        if (is_valid_2opt_move(n, i, k)) {
            return Move{i, k};
        }
    }
}

int apply_one_2opt_metropolis(Tour& tour,
                              int length,
                              const DistanceMatrix& dm,
                              double temperature,
                              Rng& rng) {
    const int n = dm.size();
    if (n < 3) {
        return length;
    }
    const Move move = sample_random_2opt_move(n, rng);
    const int delta = delta_2opt(tour, dm, move.i, move.k);
    bool accept = delta <= 0;
    if (!accept) {
        accept = rng.uniform01() < std::exp(-static_cast<double>(delta) / std::max(temperature, 1e-12));
    }
    if (accept) {
        apply_2opt(tour, move.i, move.k);
        return length + delta;
    }
    return length;
}

int diversity_state(const Tour& current, const Tour& best, double threshold) {
    if (current.empty()) {
        return 0;
    }
    const double diversity =
        static_cast<double>(hamming_distance(current, best)) / static_cast<double>(current.size());
    return diversity >= threshold ? 1 : 0;
}

}  // namespace

std::vector<QLSAAction> default_qlsa_actions() {
    return {
        {"short-2opt", 0.0, 0.25},
        {"medium-2opt", 0.25, 0.60},
        {"long-2opt", 0.60, 1.0},
    };
}

int qlsa_state_from_average_delta(double average_delta, double delta_scale) {
    if (delta_scale <= 0.0) {
        throw std::invalid_argument("delta_scale must be positive");
    }
    if (average_delta <= -delta_scale) {
        return 0;
    }
    if (average_delta < 0.0) {
        return 1;
    }
    if (average_delta == 0.0) {
        return 2;
    }
    if (average_delta < delta_scale) {
        return 3;
    }
    return 4;
}

double qlsa_reward_from_delta(int delta, bool accepted) {
    if (accepted) {
        return static_cast<double>(-delta);
    }
    return delta > 0 ? -0.1 * static_cast<double>(delta) : 0.0;
}

void update_q_value(std::vector<std::vector<double>>& q_table,
                    int state,
                    int action,
                    int next_state,
                    double reward,
                    double alpha,
                    double gamma) {
    if (state < 0 || next_state < 0 || state >= static_cast<int>(q_table.size()) ||
        next_state >= static_cast<int>(q_table.size())) {
        throw std::out_of_range("QLSA state index out of range");
    }
    if (action < 0 || action >= static_cast<int>(q_table[static_cast<size_t>(state)].size())) {
        throw std::out_of_range("QLSA action index out of range");
    }
    if (alpha < 0.0 || alpha > 1.0 || gamma < 0.0 || gamma > 1.0) {
        throw std::invalid_argument("QLSA alpha/gamma out of range");
    }

    const double old_value = q_table[static_cast<size_t>(state)][static_cast<size_t>(action)];
    const double target = reward + gamma * max_q_value(q_table[static_cast<size_t>(next_state)]);
    q_table[static_cast<size_t>(state)][static_cast<size_t>(action)] =
        old_value + alpha * (target - old_value);
}

int select_qlsa_action(const std::vector<double>& q_values, const QLSAParams& params, Rng& rng) {
    if (q_values.empty()) {
        throw std::invalid_argument("empty Q row");
    }
    if (params.policy == "epsilon-greedy") {
        if (rng.uniform01() < params.epsilon) {
            return rng.uniform_int(0, static_cast<int>(q_values.size()) - 1);
        }
        return best_action_index(q_values);
    }
    if (params.policy == "softmax") {
        const double max_q = max_q_value(q_values);
        std::vector<double> weights(q_values.size(), 0.0);
        double total = 0.0;
        for (size_t i = 0; i < q_values.size(); ++i) {
            weights[i] = std::exp((q_values[i] - max_q) / params.softmax_temperature);
            total += weights[i];
        }
        if (total <= 0.0 || !std::isfinite(total)) {
            return best_action_index(q_values);
        }

        const double draw = rng.uniform01() * total;
        double cumulative = 0.0;
        for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
            cumulative += weights[static_cast<size_t>(i)];
            if (draw <= cumulative) {
                return i;
            }
        }
        return static_cast<int>(weights.size()) - 1;
    }

    throw std::invalid_argument("QLSA policy must be epsilon-greedy or softmax");
}

QLSAResult run_qlsa_current(const DistanceMatrix& dm, const QLSAParams& params) {
    const int n = dm.size();
    if (n <= 0) {
        throw std::invalid_argument("run_qlsa_2opt requires a non-empty distance matrix");
    }

    const std::vector<QLSAAction> actions = normalized_actions(params);
    std::vector<std::vector<double>> q_table(
        kQLSAStateCount, std::vector<double>(actions.size(), 0.0));

    Timer timer;
    Rng rng(params.sa.seed);

    Tour current = params.sa.use_nearest_neighbor_init ? nearest_neighbor_tour(dm) : random_tour(n, rng);
    int current_length = tour_length(current, dm);
    Tour best = current;
    int best_length = current_length;

    QLSAResult result;
    result.action_counts.assign(actions.size(), 0);

    double temperature = params.sa.initial_temperature;
    const double temp_decay = (params.sa.iterations > 0)
                                  ? std::pow(params.sa.final_temperature / params.sa.initial_temperature,
                                             1.0 / static_cast<double>(params.sa.iterations))
                                  : 1.0;

    std::vector<int> recent_deltas;
    recent_deltas.reserve(static_cast<size_t>(params.state_window));
    double recent_sum = 0.0;
    int state = qlsa_state_from_average_delta(0.0, params.delta_scale);

    for (int64_t iter = 0; n >= 3 && iter < params.sa.iterations; ++iter) {
        const int action_index = select_qlsa_action(q_table[static_cast<size_t>(state)], params, rng);
        ++result.action_counts[static_cast<size_t>(action_index)];

        const Move move = sample_move_for_action(n, actions[static_cast<size_t>(action_index)], rng);
        const int delta = delta_2opt(current, dm, move.i, move.k);

        bool accept = delta < 0;
        if (!accept) {
            const double probability = std::exp(-static_cast<double>(delta) / temperature);
            accept = rng.uniform01() < probability;
        }

        if (accept) {
            apply_2opt(current, move.i, move.k);
            current_length += delta;
            ++result.accepted_moves;

            if (delta < 0) {
                ++result.improved_moves;
            }
            if (current_length < best_length) {
                best_length = current_length;
                best = current;
            }
        }

        recent_deltas.push_back(delta);
        recent_sum += static_cast<double>(delta);
        if (static_cast<int>(recent_deltas.size()) > params.state_window) {
            recent_sum -= static_cast<double>(recent_deltas.front());
            recent_deltas.erase(recent_deltas.begin());
        }

        const double average_delta = recent_sum / static_cast<double>(recent_deltas.size());
        const int next_state = qlsa_state_from_average_delta(average_delta, params.delta_scale);
        const double reward = qlsa_reward_from_delta(delta, accept);
        update_q_value(q_table, state, action_index, next_state, reward, params.alpha, params.gamma);
        state = next_state;

        temperature *= temp_decay;
    }

    const int checked_best_length = tour_length(best, dm);
    if (checked_best_length != best_length) {
        throw std::runtime_error("QLSA best_length verification failed");
    }
    const int checked_final_length = tour_length(current, dm);
    if (checked_final_length != current_length) {
        throw std::runtime_error("QLSA final_length verification failed");
    }

    result.best_tour = std::move(best);
    result.best_length = best_length;
    result.final_length = current_length;
    result.elapsed_ms = timer.elapsed_ms();
    result.q_table = std::move(q_table);
    return result;
}

QLSAResult run_qlsa_paper_style(const DistanceMatrix& dm, const QLSAParams& params) {
    const int n = dm.size();
    if (n <= 0) {
        throw std::invalid_argument("run_qlsa_2opt requires a non-empty distance matrix");
    }

    const int state_count = (params.variant == "paper-sb") ? 2 : 1;
    constexpr int kCandidateLeaderActionCount = 4;
    enum CandidateLeaderAction {
        Current = 0,
        GlobalBest = 1,
        Random = 2,
        DoubleBridge = 3,
    };

    std::vector<std::vector<double>> q_table(
        static_cast<size_t>(state_count),
        std::vector<double>(static_cast<size_t>(kCandidateLeaderActionCount), 0.0));

    Timer timer;
    Rng rng(params.sa.seed);

    Tour current = params.sa.use_nearest_neighbor_init ? nearest_neighbor_tour(dm) : random_tour(n, rng);
    int current_length = tour_length(current, dm);
    Tour best = current;
    int best_length = current_length;

    QLSAResult result;
    result.action_counts.assign(static_cast<size_t>(kCandidateLeaderActionCount), 0);

    double temperature = params.sa.initial_temperature;
    const double temp_decay = (params.sa.iterations > 0)
                                  ? std::pow(params.sa.final_temperature / params.sa.initial_temperature,
                                             1.0 / static_cast<double>(params.sa.iterations))
                                  : 1.0;

    int state = (params.variant == "paper-sb")
                    ? diversity_state(current, best, params.diversity_threshold)
                    : 0;

    for (int64_t iter = 0; n >= 3 && iter < params.sa.iterations; ++iter) {
        const int action_index = select_qlsa_action(q_table[static_cast<size_t>(state)], params, rng);
        ++result.action_counts[static_cast<size_t>(action_index)];

        Tour leader;
        int leader_length = 0;
        switch (action_index) {
            case Current:
                leader = current;
                leader_length = current_length;
                break;
            case GlobalBest:
                leader = best;
                leader_length = best_length;
                break;
            case Random:
                leader = random_tour(n, rng);
                leader_length = tour_length(leader, dm);
                break;
            case DoubleBridge:
                leader = double_bridge(current, rng);
                leader_length = tour_length(leader, dm);
                break;
            default:
                throw std::runtime_error("invalid candidate-leader action");
        }

        const int previous_length = current_length;
        const int candidate_length =
            apply_one_2opt_metropolis(leader, leader_length, dm, temperature, rng);
        const int total_delta = candidate_length - current_length;

        bool accept = total_delta <= 0;
        if (!accept) {
            accept = rng.uniform01() < std::exp(-static_cast<double>(total_delta) /
                                                std::max(temperature, 1e-12));
        }

        if (accept) {
            current = std::move(leader);
            current_length = candidate_length;
            ++result.accepted_moves;
            if (current_length < previous_length) {
                ++result.improved_moves;
            }
            if (current_length < best_length) {
                best_length = current_length;
                best = current;
            }
        }

        const double reward = std::max(0, previous_length - current_length);
        const int next_state = (params.variant == "paper-sb")
                                   ? diversity_state(current, best, params.diversity_threshold)
                                   : 0;
        update_q_value(q_table, state, action_index, next_state, reward, params.alpha, params.gamma);
        state = next_state;

        temperature *= temp_decay;
    }

    const int checked_best_length = tour_length(best, dm);
    if (checked_best_length != best_length) {
        throw std::runtime_error("QLSA paper-style best_length verification failed");
    }
    const int checked_final_length = tour_length(current, dm);
    if (checked_final_length != current_length) {
        throw std::runtime_error("QLSA paper-style final_length verification failed");
    }

    result.best_tour = std::move(best);
    result.best_length = best_length;
    result.final_length = current_length;
    result.elapsed_ms = timer.elapsed_ms();
    result.q_table = std::move(q_table);
    return result;
}

QLSAResult run_qlsa_2opt(const DistanceMatrix& dm, const QLSAParams& params) {
    validate_params(params);
    if (params.variant == "current") {
        return run_qlsa_current(dm, params);
    }
    return run_qlsa_paper_style(dm, params);
}

}  // namespace tsp
