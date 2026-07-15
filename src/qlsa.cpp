#include "tsp/qlsa.hpp"

#include <algorithm>
#include <chrono>
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

void validate_params_impl(const QLSAParams& params) {
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
    if (params.diversity_metric != "edge" && params.diversity_metric != "hamming") {
        throw std::invalid_argument("QLSA diversity_metric must be edge or hamming");
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

void refresh_edge_diversity_cache(QLSAState& state) {
    if (state.params.diversity_metric != "edge") {
        return;
    }

    const size_t n = state.best_tour.size();
    std::vector<int>& neighbors = state.edge_diversity_best_neighbors;
    if (neighbors.size() != n * 2U) {
        neighbors.resize(n * 2U);
    }
    std::fill(neighbors.begin(), neighbors.end(), -1);

    for (size_t i = 0; i < n; ++i) {
        const int city = state.best_tour[i];
        if (city < 0 || static_cast<size_t>(city) >= n) {
            throw std::invalid_argument("QLSA edge diversity requires a legal best tour");
        }
        const size_t offset = static_cast<size_t>(city) * 2U;
        neighbors[offset] = state.best_tour[(i + n - 1U) % n];
        neighbors[offset + 1U] = state.best_tour[(i + 1U) % n];
    }
}

double cached_edge_diversity_ratio(const QLSAState& state) {
    const Tour& current = state.current_tour;
    if (current.empty()) {
        return 0.0;
    }
    const std::vector<int>& neighbors = state.edge_diversity_best_neighbors;
    if (neighbors.size() != current.size() * 2U) {
        throw std::logic_error("QLSA edge diversity cache does not match the current tour");
    }

    int differing_edges = 0;
    for (size_t i = 0; i < current.size(); ++i) {
        const int city = current[i];
        const int next = current[(i + 1U) % current.size()];
        if (city < 0 || static_cast<size_t>(city) >= current.size()) {
            throw std::invalid_argument("QLSA edge diversity requires a legal current tour");
        }
        const size_t offset = static_cast<size_t>(city) * 2U;
        if (neighbors[offset] != next && neighbors[offset + 1U] != next) {
            ++differing_edges;
        }
    }
    return static_cast<double>(differing_edges) / static_cast<double>(current.size());
}

int diversity_state(const QLSAState& state) {
    if (state.current_tour.empty()) {
        return 0;
    }
    const double diversity = state.params.diversity_metric == "edge"
                                 ? cached_edge_diversity_ratio(state)
                                 : qlsa_diversity_ratio(state.current_tour,
                                                        state.best_tour,
                                                        state.params.diversity_metric);
    return diversity >= state.params.diversity_threshold ? 1 : 0;
}

constexpr int kCandidateLeaderActionCount = 4;
enum CandidateLeaderAction {
    Current = 0,
    GlobalBest = 1,
    Random = 2,
    DoubleBridge = 3,
};

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

void verify_qlsa_state(const DistanceMatrix& dm, const QLSAState& state) {
    if (!state.initialized) {
        throw std::invalid_argument("QLSA state is not initialized");
    }
    if (state.params.sa.iterations < state.iterations_completed) {
        throw std::runtime_error("QLSA state exceeded its iteration budget");
    }
    if (tour_length(state.best_tour, dm) != state.best_length) {
        throw std::runtime_error("QLSA best_length verification failed");
    }
    if (tour_length(state.current_tour, dm) != state.current_length) {
        throw std::runtime_error("QLSA final_length verification failed");
    }
}

}  // namespace

void validate_qlsa_params(const QLSAParams& params) {
    validate_params_impl(params);
    if (params.variant == "current") {
        (void)normalized_actions(params);
    }
}

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

double qlsa_diversity_ratio(const Tour& current, const Tour& best, const std::string& metric) {
    if (current.size() != best.size()) {
        throw std::invalid_argument("QLSA diversity requires equal-length tours");
    }
    if (current.empty()) {
        return 0.0;
    }
    if (metric == "edge") {
        return static_cast<double>(undirected_edge_distance(current, best)) /
               static_cast<double>(current.size());
    }
    if (metric == "hamming") {
        return static_cast<double>(hamming_distance(current, best)) /
               static_cast<double>(current.size());
    }
    throw std::invalid_argument("QLSA diversity_metric must be edge or hamming");
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

int select_qlsa_action(const std::vector<double>& q_values,
                       const QLSAParams& params,
                       Rng& rng,
                       std::vector<double>* softmax_weights) {
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
        std::vector<double> local_weights;
        std::vector<double>& weights = softmax_weights != nullptr ? *softmax_weights : local_weights;
        if (weights.size() != q_values.size()) {
            weights.resize(q_values.size());
        }
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

QLSAState initialize_qlsa_state(const DistanceMatrix& dm, const QLSAParams& params) {
    validate_qlsa_params(params);
    if (dm.size() <= 0) {
        throw std::invalid_argument("run_qlsa_2opt requires a non-empty distance matrix");
    }
    Timer timer;

    QLSAState state;
    state.params = params;
    state.rng = Rng(params.sa.seed);
    state.current_tour = params.sa.use_nearest_neighbor_init
                             ? nearest_neighbor_tour(dm, params.sa.nearest_neighbor_start)
                             : random_tour(dm.size(), state.rng);
    state.current_length = tour_length(state.current_tour, dm);
    state.best_tour = state.current_tour;
    state.best_length = state.current_length;
    state.temperature = params.sa.initial_temperature;
    state.temperature_decay =
        params.sa.iterations > 0
            ? std::pow(params.sa.final_temperature / params.sa.initial_temperature,
                       1.0 / static_cast<double>(params.sa.iterations))
            : 1.0;

    if (params.variant == "current") {
        state.actions = normalized_actions(params);
        state.q_table.assign(static_cast<size_t>(kQLSAStateCount),
                             std::vector<double>(state.actions.size(), 0.0));
        state.action_counts.assign(state.actions.size(), 0);
        state.softmax_weights.resize(state.actions.size());
        state.recent_deltas.reserve(static_cast<size_t>(params.state_window));
        state.learning_state = qlsa_state_from_average_delta(0.0, params.delta_scale);
    } else {
        const int state_count = params.variant == "paper-sb" ? 2 : 1;
        state.q_table.assign(static_cast<size_t>(state_count),
                             std::vector<double>(static_cast<size_t>(kCandidateLeaderActionCount),
                                                 0.0));
        state.action_counts.assign(static_cast<size_t>(kCandidateLeaderActionCount), 0);
        state.softmax_weights.resize(static_cast<size_t>(kCandidateLeaderActionCount));
        if (params.variant == "paper-sb" && params.diversity_metric == "edge") {
            refresh_edge_diversity_cache(state);
        }
        state.learning_state = params.variant == "paper-sb" ? diversity_state(state) : 0;
    }

    state.initialized = true;
    state.elapsed_ms = timer.elapsed_ms();
    return state;
}

SearchChunkProgress run_qlsa_chunk(const DistanceMatrix& dm,
                                   QLSAState& state,
                                   const SearchChunkOptions& options) {
    validate_chunk_options(options);
    if (!state.initialized) {
        throw std::invalid_argument("QLSA state is not initialized");
    }
    if (static_cast<int>(state.current_tour.size()) != dm.size() ||
        static_cast<int>(state.best_tour.size()) != dm.size()) {
        throw std::invalid_argument("QLSA state does not match the distance matrix");
    }

    SearchChunkProgress progress;
    const int64_t remaining = state.params.sa.iterations - state.iterations_completed;
    if (remaining < 0) {
        throw std::runtime_error("QLSA state exceeded its iteration budget");
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

        if (state.params.variant == "current") {
            const int action_index =
                select_qlsa_action(state.q_table[static_cast<size_t>(state.learning_state)],
                                   state.params,
                                   state.rng,
                                   &state.softmax_weights);
            ++state.action_counts[static_cast<size_t>(action_index)];
            const Move move = sample_move_for_action(
                n, state.actions[static_cast<size_t>(action_index)], state.rng);
            const int delta = delta_2opt(state.current_tour, dm, move.i, move.k);

            bool accept = delta < 0;
            if (!accept) {
                const double probability =
                    std::exp(-static_cast<double>(delta) / state.temperature);
                accept = state.rng.uniform01() < probability;
            }
            if (accept) {
                apply_2opt(state.current_tour, move.i, move.k);
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

            const size_t state_window = static_cast<size_t>(state.params.state_window);
            if (state.recent_deltas.size() < state_window) {
                state.recent_deltas.push_back(delta);
                state.recent_delta_sum += static_cast<double>(delta);
                if (state.recent_deltas.size() == state_window) {
                    state.recent_delta_head = 0;
                }
            } else {
                const size_t overwritten = state.recent_delta_head;
                state.recent_delta_sum -=
                    static_cast<double>(state.recent_deltas[overwritten]);
                state.recent_deltas[overwritten] = delta;
                state.recent_delta_sum += static_cast<double>(delta);
                state.recent_delta_head = (overwritten + 1U) % state_window;
            }
            const double average_delta =
                state.recent_delta_sum / static_cast<double>(state.recent_deltas.size());
            const int next_state =
                qlsa_state_from_average_delta(average_delta, state.params.delta_scale);
            update_q_value(state.q_table,
                           state.learning_state,
                           action_index,
                           next_state,
                           qlsa_reward_from_delta(delta, accept),
                           state.params.alpha,
                           state.params.gamma);
            state.learning_state = next_state;
        } else {
            const int action_index =
                select_qlsa_action(state.q_table[static_cast<size_t>(state.learning_state)],
                                   state.params,
                                   state.rng,
                                   &state.softmax_weights);
            ++state.action_counts[static_cast<size_t>(action_index)];

            Tour leader;
            int leader_length = 0;
            switch (action_index) {
                case Current:
                    leader = state.current_tour;
                    leader_length = state.current_length;
                    break;
                case GlobalBest:
                    leader = state.best_tour;
                    leader_length = state.best_length;
                    break;
                case Random:
                    leader = random_tour(n, state.rng);
                    leader_length = tour_length(leader, dm);
                    break;
                case DoubleBridge:
                    leader = double_bridge(state.current_tour, state.rng);
                    leader_length = tour_length(leader, dm);
                    break;
                default:
                    throw std::runtime_error("invalid candidate-leader action");
            }

            const int previous_length = state.current_length;
            const int candidate_length = apply_one_2opt_metropolis(
                leader, leader_length, dm, state.temperature, state.rng);
            const int total_delta = candidate_length - state.current_length;
            bool accept = total_delta <= 0;
            if (!accept) {
                accept = state.rng.uniform01() <
                         std::exp(-static_cast<double>(total_delta) /
                                  std::max(state.temperature, 1e-12));
            }
            if (accept) {
                state.current_tour = std::move(leader);
                state.current_length = candidate_length;
                ++state.accepted_moves;
                if (state.current_length < previous_length) {
                    ++state.improved_moves;
                }
                if (state.current_length < state.best_length) {
                    state.best_length = state.current_length;
                    state.best_tour = state.current_tour;
                    if (state.params.variant == "paper-sb" &&
                        state.params.diversity_metric == "edge") {
                        refresh_edge_diversity_cache(state);
                    }
                }
            }

            const double reward = std::max(0, previous_length - state.current_length);
            const int next_state = state.params.variant == "paper-sb" ? diversity_state(state) : 0;
            update_q_value(state.q_table,
                           state.learning_state,
                           action_index,
                           next_state,
                           reward,
                           state.params.alpha,
                           state.params.gamma);
            state.learning_state = next_state;
        }

        state.temperature *= state.temperature_decay;
        ++state.iterations_completed;
        ++progress.iterations_completed;
    }

    state.elapsed_ms += timer.elapsed_ms();
    progress.total_iterations_completed = state.iterations_completed;
    progress.iteration_budget_exhausted =
        n < 3 || state.iterations_completed >= state.params.sa.iterations;
    return progress;
}

QLSAResult finalize_qlsa_state(const DistanceMatrix& dm, const QLSAState& state) {
    verify_qlsa_state(dm, state);
    QLSAResult result;
    result.best_tour = state.best_tour;
    result.best_length = state.best_length;
    result.final_length = state.current_length;
    result.elapsed_ms = state.elapsed_ms;
    result.accepted_moves = state.accepted_moves;
    result.improved_moves = state.improved_moves;
    result.iterations_completed = state.iterations_completed;
    result.deadline_reached = state.deadline_reached;
    result.q_table = state.q_table;
    result.action_counts = state.action_counts;
    return result;
}

bool migrate_qlsa_tour(const DistanceMatrix& dm, QLSAState& state, const Tour& migrant) {
    if (!state.initialized) {
        throw std::invalid_argument("QLSA state is not initialized");
    }
    if (!is_valid_tour(migrant, dm.size())) {
        throw std::invalid_argument("QLSA migrant must be a legal tour");
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
        if (state.params.variant == "paper-sb" && state.params.diversity_metric == "edge") {
            refresh_edge_diversity_cache(state);
        }
    }
    if (state.params.variant == "current") {
        state.recent_deltas.clear();
        state.recent_delta_sum = 0.0;
        state.recent_delta_head = 0;
        state.learning_state = qlsa_state_from_average_delta(0.0, state.params.delta_scale);
    } else if (state.params.variant == "paper-sb") {
        state.learning_state = diversity_state(state);
    } else {
        state.learning_state = 0;
    }
    return true;
}

QLSAResult run_qlsa_2opt(const DistanceMatrix& dm, const QLSAParams& params) {
    Timer timer;
    QLSAState state = initialize_qlsa_state(dm, params);
    SearchChunkOptions options;
    options.max_iterations = params.sa.iterations;
    (void)run_qlsa_chunk(dm, state, options);
    QLSAResult result = finalize_qlsa_state(dm, state);
    result.elapsed_ms = timer.elapsed_ms();
    return result;
}

}  // namespace tsp
