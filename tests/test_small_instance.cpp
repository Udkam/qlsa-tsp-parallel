#ifdef NDEBUG
#error "Tests require assertions; NDEBUG must not be defined"
#endif
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/rng.hpp"
#include "tsp/sa.hpp"
#include "tsp/tour.hpp"
#include "tsp/tsplib_parser.hpp"

namespace {

tsp::Instance make_square_instance() {
    tsp::Instance instance;
    instance.name = "square4";
    instance.type = "TSP";
    instance.dimension = 4;
    instance.edge_weight_type = "EUC_2D";
    instance.coords = {
        {0.0, 0.0},
        {10.0, 0.0},
        {10.0, 10.0},
        {0.0, 10.0},
    };
    return instance;
}

template <typename Function>
void assert_throws_with_message(Function&& function, const std::string& expected_fragment) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception& error) {
        threw = true;
        assert(std::string(error.what()).find(expected_fragment) != std::string::npos);
    }
    assert(threw);
}

}  // namespace

int main() {
    const tsp::Instance instance = make_square_instance();
    const tsp::DistanceMatrix dm(instance);

    assert(dm.size() == 4);
    assert(dm.dist(0, 0) == 0);
    assert(dm.dist(0, 1) == 10);
    assert(dm.dist(1, 2) == 10);
    assert(dm.dist(0, 2) == 14);
    assert(dm.dist(0, 2) == dm.dist(2, 0));

    const tsp::Tour square = {0, 1, 2, 3};
    assert(tsp::is_valid_tour(square, dm.size()));
    assert(tsp::tour_length(square, dm) == 40);

    tsp::Rng rng(123);
    const tsp::Tour random = tsp::random_tour(dm.size(), rng);
    assert(tsp::is_valid_tour(random, dm.size()));

    const tsp::Tour nn = tsp::nearest_neighbor_tour(dm);
    assert(tsp::is_valid_tour(nn, dm.size()));

    tsp::Tour crossing = {0, 2, 1, 3};
    const int old_length = tsp::tour_length(crossing, dm);
    const int delta = tsp::delta_2opt(crossing, dm, 1, 2);
    tsp::apply_2opt(crossing, 1, 2);
    const int new_length = tsp::tour_length(crossing, dm);
    assert(old_length + delta == new_length);
    assert(new_length == 40);

    tsp::SAParams params;
    params.iterations = 1000;
    params.initial_temperature = 100.0;
    params.final_temperature = 0.001;
    params.seed = 7;
    params.use_nearest_neighbor_init = false;
    const tsp::SAResult result = tsp::run_sa_2opt(dm, params);
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));
    assert(result.best_length == tsp::tour_length(result.best_tour, dm));

    std::vector<std::vector<double>> q_table(
        tsp::kQLSAStateCount, std::vector<double>(3, 0.0));
    tsp::update_q_value(q_table, 2, 1, 0, 10.0, 0.5, 0.9);
    assert(std::abs(q_table[2][1] - 5.0) < 1e-12);
    assert(tsp::qlsa_state_from_average_delta(-20.0, 10.0) == 0);
    assert(tsp::qlsa_state_from_average_delta(0.0, 10.0) == 2);
    assert(tsp::qlsa_state_from_average_delta(20.0, 10.0) == 4);
    assert(tsp::qlsa_reward_from_delta(-7, true) == 7.0);
    assert(tsp::qlsa_reward_from_delta(7, false) < 0.0);

    tsp::QLSAParams action_params;
    action_params.policy = "epsilon-greedy";
    action_params.epsilon = 0.0;
    tsp::Rng action_rng(1);
    assert(tsp::select_qlsa_action({1.0, 3.0, 2.0}, action_params, action_rng) == 1);

    action_params.policy = "softmax";
    action_params.softmax_temperature = 1.0;
    tsp::Rng softmax_rng(2);
    assert(tsp::select_qlsa_action({-1000.0, 1000.0, -1000.0}, action_params, softmax_rng) == 1);

    tsp::QLSAParams qlsa_params;
    qlsa_params.sa = params;
    qlsa_params.sa.iterations = 500;
    qlsa_params.alpha = 0.2;
    qlsa_params.gamma = 0.8;
    qlsa_params.epsilon = 0.05;
    qlsa_params.policy = "epsilon-greedy";
    const tsp::QLSAResult qlsa_result = tsp::run_qlsa_2opt(dm, qlsa_params);
    assert(tsp::is_valid_tour(qlsa_result.best_tour, dm.size()));
    assert(qlsa_result.best_length == tsp::tour_length(qlsa_result.best_tour, dm));
    assert(static_cast<int>(qlsa_result.q_table.size()) == tsp::kQLSAStateCount);
    assert(qlsa_result.action_counts.size() == tsp::default_qlsa_actions().size());
    const int64_t total_actions = std::accumulate(qlsa_result.action_counts.begin(),
                                                  qlsa_result.action_counts.end(),
                                                  int64_t{0});
    assert(total_actions == qlsa_params.sa.iterations);

    const std::string fixture_path = std::string(TEST_SOURCE_DIR) + "/fixtures/square4.tsp";
    const tsp::Instance parsed = tsp::load_tsplib(fixture_path);
    const tsp::DistanceMatrix parsed_dm(parsed);
    const tsp::QLSAResult parsed_result = tsp::run_qlsa_2opt(parsed_dm, qlsa_params);
    assert(tsp::is_valid_tour(parsed_result.best_tour, parsed_dm.size()));
    assert(parsed_result.best_length == 40);

    const std::string atsp_path = std::string(TEST_SOURCE_DIR) + "/fixtures/atsp3.tsp";
    assert_throws_with_message(
        [&] { (void)tsp::load_tsplib(atsp_path); },
        "TYPE ATSP");

    const std::string asymmetric_path =
        std::string(TEST_SOURCE_DIR) + "/fixtures/asymmetric_full_matrix3.tsp";
    assert_throws_with_message(
        [&] { (void)tsp::load_tsplib(asymmetric_path); },
        "asymmetric EXPLICIT FULL_MATRIX");

    tsp::Instance direct_atsp = make_square_instance();
    direct_atsp.type = "atsp";
    assert_throws_with_message(
        [&] { (void)tsp::DistanceMatrix(direct_atsp); },
        "TYPE ATSP");

    tsp::Instance symmetric_explicit;
    symmetric_explicit.name = "symmetric_explicit3";
    symmetric_explicit.type = "TSP";
    symmetric_explicit.dimension = 3;
    symmetric_explicit.edge_weight_type = "EXPLICIT";
    symmetric_explicit.edge_weight_format = "FULL_MATRIX";
    symmetric_explicit.raw_weights = {
        0, 7, 11,
        7, 0, 5,
        11, 5, 0,
    };
    const tsp::DistanceMatrix symmetric_explicit_dm(symmetric_explicit);
    assert(symmetric_explicit_dm.dist(0, 1) == 7);
    assert(symmetric_explicit_dm.dist(1, 0) == 7);

    tsp::Instance asymmetric_explicit = symmetric_explicit;
    asymmetric_explicit.name = "asymmetric_explicit3";
    asymmetric_explicit.raw_weights[1] = 9;
    assert_throws_with_message(
        [&] { (void)tsp::DistanceMatrix(asymmetric_explicit); },
        "asymmetric EXPLICIT FULL_MATRIX");

    std::cout << "test_small_instance passed\n";
    return 0;
}
