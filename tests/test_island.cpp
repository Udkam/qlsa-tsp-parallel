#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/island.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/sa.hpp"
#include "tsp/tour.hpp"
#include "tsp/tsplib_parser.hpp"

#ifdef NDEBUG
#error "NDEBUG must be undefined for assertion-based tests"
#endif

namespace {

tsp::DistanceMatrix make_irregular_dm() {
    tsp::Instance instance;
    instance.name = "irregular26";
    instance.type = "TSP";
    instance.dimension = 26;
    instance.edge_weight_type = "EUC_2D";
    for (int i = 0; i < instance.dimension; ++i) {
        instance.coords.push_back({static_cast<double>((i * 37 + i * i * 3) % 173),
                                   static_cast<double>((i * 61 + i * i * 5) % 191)});
    }
    return tsp::DistanceMatrix(instance);
}

struct MigrationTours {
    tsp::Tour best;
    tsp::Tour worse;
};

MigrationTours make_migration_tours(const tsp::DistanceMatrix& dm) {
    const tsp::Tour best = tsp::nearest_neighbor_tour(dm);
    const int best_length = tsp::tour_length(best, dm);
    tsp::Rng rng(713);
    for (int attempt = 0; attempt < 1024; ++attempt) {
        tsp::Tour candidate = tsp::random_tour(dm.size(), rng);
        if (tsp::tour_length(candidate, dm) > best_length) {
            return {best, candidate};
        }
    }
    assert(false && "fixture must provide a random tour worse than nearest-neighbor");
    return {};
}

void set_qlsa_state_tours(tsp::QLSAState& state,
                          const tsp::DistanceMatrix& dm,
                          const tsp::Tour& current,
                          const tsp::Tour& best) {
    state.current_tour = current;
    state.current_length = tsp::tour_length(current, dm);
    state.best_tour = best;
    state.best_length = tsp::tour_length(best, dm);
    assert(state.best_length < state.current_length);
}

void assert_sa_equal(const tsp::SAResult& lhs, const tsp::SAResult& rhs) {
    assert(lhs.best_tour == rhs.best_tour);
    assert(lhs.best_length == rhs.best_length);
    assert(lhs.final_length == rhs.final_length);
    assert(lhs.accepted_moves == rhs.accepted_moves);
    assert(lhs.improved_moves == rhs.improved_moves);
    assert(lhs.iterations_completed == rhs.iterations_completed);
}

void test_sa_chunk_equivalence(const tsp::DistanceMatrix& dm) {
    tsp::SAParams params;
    params.iterations = 4097;
    params.initial_temperature = 250.0;
    params.final_temperature = 1e-4;
    params.seed = 991;
    params.use_nearest_neighbor_init = false;

    const tsp::SAResult uninterrupted = tsp::run_sa_2opt(dm, params);
    tsp::SAState state = tsp::initialize_sa_state(dm, params);
    const std::vector<int64_t> chunk_sizes = {1, 17, 233, 64, 509};
    size_t chunk_index = 0;
    while (state.iterations_completed < params.iterations) {
        tsp::SearchChunkOptions options;
        options.max_iterations = chunk_sizes[chunk_index++ % chunk_sizes.size()];
        const tsp::SearchChunkProgress progress = tsp::run_sa_chunk(dm, state, options);
        assert(progress.iterations_completed > 0);
        assert(progress.total_iterations_completed == state.iterations_completed);
    }
    const tsp::SAResult chunked = tsp::finalize_sa_state(dm, state);
    assert_sa_equal(uninterrupted, chunked);

    // A deadline stop consumes no hidden iterations and the exact state can be
    // resumed later without changing the final deterministic result.
    tsp::SAState resumed = tsp::initialize_sa_state(dm, params);
    tsp::SearchChunkOptions prefix;
    prefix.max_iterations = 173;
    assert(tsp::run_sa_chunk(dm, resumed, prefix).iterations_completed == 173);
    tsp::SearchChunkOptions expired;
    expired.max_iterations = params.iterations;
    expired.deadline = tsp::SearchClock::now() - std::chrono::milliseconds(1);
    expired.deadline_check_interval = 1;
    const tsp::SearchChunkProgress stopped = tsp::run_sa_chunk(dm, resumed, expired);
    assert(stopped.iterations_completed == 0);
    assert(stopped.deadline_reached);
    assert(resumed.iterations_completed == 173);
    tsp::SearchChunkOptions suffix;
    suffix.max_iterations = params.iterations;
    assert(tsp::run_sa_chunk(dm, resumed, suffix).iteration_budget_exhausted);
    assert_sa_equal(uninterrupted, tsp::finalize_sa_state(dm, resumed));
}

void assert_qlsa_equal(const tsp::QLSAResult& lhs, const tsp::QLSAResult& rhs) {
    assert(lhs.best_tour == rhs.best_tour);
    assert(lhs.best_length == rhs.best_length);
    assert(lhs.final_length == rhs.final_length);
    assert(lhs.accepted_moves == rhs.accepted_moves);
    assert(lhs.improved_moves == rhs.improved_moves);
    assert(lhs.iterations_completed == rhs.iterations_completed);
    assert(lhs.q_table == rhs.q_table);
    assert(lhs.action_counts == rhs.action_counts);
}

void assert_island_semantics_equal(const tsp::IslandResult& lhs, const tsp::IslandResult& rhs) {
    assert(lhs.best_tour == rhs.best_tour);
    assert(lhs.best_length == rhs.best_length);
    assert(lhs.final_length_of_best_island == rhs.final_length_of_best_island);
    assert(lhs.island_count == rhs.island_count);
    assert(lhs.base_seed == rhs.base_seed);
    assert(lhs.topology == rhs.topology);
    assert(lhs.iteration_budget_per_island == rhs.iteration_budget_per_island);
    assert(lhs.total_iteration_budget == rhs.total_iteration_budget);
    assert(lhs.total_iterations_completed == rhs.total_iterations_completed);
    assert(lhs.total_accepted_moves == rhs.total_accepted_moves);
    assert(lhs.total_improved_moves == rhs.total_improved_moves);
    assert(lhs.migration_rounds == rhs.migration_rounds);
    assert(lhs.migration_attempts == rhs.migration_attempts);
    assert(lhs.migrations_adopted == rhs.migrations_adopted);
    assert(lhs.deadline_reached == rhs.deadline_reached);
    assert(lhs.iteration_budget_exhausted == rhs.iteration_budget_exhausted);
    assert(lhs.islands.size() == rhs.islands.size());
    for (size_t index = 0; index < lhs.islands.size(); ++index) {
        const tsp::IslandChainResult& left = lhs.islands[index];
        const tsp::IslandChainResult& right = rhs.islands[index];
        assert(left.island_id == right.island_id);
        assert(left.seed == right.seed);
        assert(left.best_tour == right.best_tour);
        assert(left.final_tour == right.final_tour);
        assert(left.best_length == right.best_length);
        assert(left.final_length == right.final_length);
        assert(left.iterations_completed == right.iterations_completed);
        assert(left.accepted_moves == right.accepted_moves);
        assert(left.improved_moves == right.improved_moves);
        assert(left.migration_attempts == right.migration_attempts);
        assert(left.migrations_adopted == right.migrations_adopted);
        assert(left.deadline_reached == right.deadline_reached);
    }
}

void test_qlsa_chunk_equivalence(const tsp::DistanceMatrix& dm, const std::string& variant) {
    tsp::QLSAParams params;
    params.sa.iterations = 3073;
    params.sa.initial_temperature = 300.0;
    params.sa.final_temperature = 1e-4;
    params.sa.seed = 12345;
    params.sa.use_nearest_neighbor_init = false;
    params.variant = variant;
    params.epsilon = 0.17;
    params.state_window = 3;

    const tsp::QLSAResult uninterrupted = tsp::run_qlsa_2opt(dm, params);
    tsp::QLSAState state = tsp::initialize_qlsa_state(dm, params);
    const std::vector<int64_t> chunks = {3, 101, 7, 512, 29};
    size_t chunk_index = 0;
    while (state.iterations_completed < params.sa.iterations) {
        tsp::SearchChunkOptions options;
        options.max_iterations = chunks[chunk_index++ % chunks.size()];
        const tsp::SearchChunkProgress progress = tsp::run_qlsa_chunk(dm, state, options);
        assert(progress.iterations_completed > 0);
    }
    if (variant == "current") {
        assert(state.recent_deltas.size() == 3);
        assert(state.recent_delta_head == 1);
        assert(state.softmax_weights.size() == state.actions.size());
    }
    assert_qlsa_equal(uninterrupted, tsp::finalize_qlsa_state(dm, state));
}

void test_qlsa_migration_learning_state(const tsp::DistanceMatrix& dm) {
    const MigrationTours tours = make_migration_tours(dm);

    tsp::QLSAParams current_params;
    current_params.sa.iterations = 20;
    // Keep the initialized best tour aligned with make_migration_tours() so
    // the paper-sb edge cache represents the same global-best cycle.
    current_params.sa.use_nearest_neighbor_init = true;
    current_params.variant = "current";
    tsp::QLSAState current = tsp::initialize_qlsa_state(dm, current_params);
    set_qlsa_state_tours(current, dm, tours.worse, tours.best);
    current.recent_deltas = {9, -3};
    current.recent_delta_sum = 6.0;
    current.learning_state = 0;
    assert(tsp::migrate_qlsa_tour(dm, current, tours.best));
    assert(current.recent_deltas.empty());
    assert(current.recent_delta_sum == 0.0);
    assert(current.learning_state ==
           tsp::qlsa_state_from_average_delta(0.0, current.params.delta_scale));

    tsp::QLSAParams paper_params = current_params;
    paper_params.variant = "paper";
    tsp::QLSAState paper = tsp::initialize_qlsa_state(dm, paper_params);
    set_qlsa_state_tours(paper, dm, tours.worse, tours.best);
    paper.learning_state = 1;
    assert(tsp::migrate_qlsa_tour(dm, paper, tours.best));
    assert(paper.learning_state == 0);

    tsp::Tour rotated_best = tours.best;
    std::rotate(rotated_best.begin(), rotated_best.begin() + 1, rotated_best.end());
    assert(tsp::tour_length(rotated_best, dm) == tsp::tour_length(tours.best, dm));

    tsp::QLSAParams edge_params = current_params;
    edge_params.variant = "paper-sb";
    edge_params.diversity_threshold = 0.5;
    edge_params.diversity_metric = "edge";
    tsp::QLSAState edge = tsp::initialize_qlsa_state(dm, edge_params);
    set_qlsa_state_tours(edge, dm, tours.worse, tours.best);
    assert(tsp::migrate_qlsa_tour(dm, edge, rotated_best));
    assert(edge.learning_state == 0);

    tsp::QLSAParams hamming_params = edge_params;
    hamming_params.diversity_metric = "hamming";
    tsp::QLSAState hamming = tsp::initialize_qlsa_state(dm, hamming_params);
    set_qlsa_state_tours(hamming, dm, tours.worse, tours.best);
    assert(tsp::migrate_qlsa_tour(dm, hamming, rotated_best));
    assert(hamming.learning_state == 1);
}

void assert_island_result_valid(const tsp::DistanceMatrix& dm,
                                const tsp::IslandResult& result,
                                int island_count,
                                int64_t budget) {
    assert(result.island_count == island_count);
    assert(result.islands.size() == static_cast<size_t>(island_count));
    assert(result.total_iteration_budget == budget * island_count);
    assert(result.total_iterations_completed == result.total_iteration_budget);
    assert(result.iteration_budget_exhausted);
    assert(!result.deadline_reached);
    assert(result.actual_threads >= 1);
    assert(result.actual_threads <= result.threads);
    assert(result.used_openmp == (result.actual_threads > 1));
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));
    assert(tsp::tour_length(result.best_tour, dm) == result.best_length);
    int64_t counted_iterations = 0;
    for (int i = 0; i < island_count; ++i) {
        const tsp::IslandChainResult& island = result.islands[static_cast<size_t>(i)];
        assert(island.island_id == i);
        assert(island.seed == tsp::island_seed(result.base_seed, i));
        assert(island.iterations_completed == budget);
        assert(tsp::is_valid_tour(island.best_tour, dm.size()));
        assert(tsp::is_valid_tour(island.final_tour, dm.size()));
        assert(tsp::tour_length(island.best_tour, dm) == island.best_length);
        assert(tsp::tour_length(island.final_tour, dm) == island.final_length);
        counted_iterations += island.iterations_completed;
    }
    assert(counted_iterations == result.total_iterations_completed);
}

void test_island_topologies(const tsp::DistanceMatrix& dm) {
    tsp::IslandParams params;
    params.algorithm = tsp::IslandAlgorithm::SA;
    params.island_count = 4;
    params.threads = 2;
    params.base_seed = 77;
    params.migration_interval = 250;
    params.sa_params.iterations = 1500;
    params.sa_params.initial_temperature = 200.0;
    params.sa_params.final_temperature = 1e-4;
    params.sa_params.use_nearest_neighbor_init = false;

    const tsp::IslandResult independent =
        tsp::run_openmp_islands(dm, params, tsp::MigrationTopology::Independent);
    assert_island_result_valid(dm, independent, params.island_count, params.sa_params.iterations);
    assert(independent.migration_rounds == 0);
    assert(independent.migration_attempts == 0);

    const tsp::IslandResult ring =
        tsp::run_openmp_islands(dm, params, tsp::MigrationTopology::Ring);
    assert_island_result_valid(dm, ring, params.island_count, params.sa_params.iterations);
    assert(ring.migration_rounds == 5);
    assert(ring.migration_attempts == ring.migration_rounds * params.island_count);

    // The persistent OpenMP team must preserve fixed-work island results and
    // migration ordering, independent of worker count or repeated execution.
    tsp::IslandParams serial_params = params;
    serial_params.threads = 1;
    const tsp::IslandResult serial_ring =
        tsp::run_openmp_islands(dm, serial_params, tsp::MigrationTopology::Ring);
    assert_island_semantics_equal(serial_ring, ring);
    const tsp::IslandResult ring_repeat =
        tsp::run_openmp_islands(dm, params, tsp::MigrationTopology::Ring);
    assert_island_semantics_equal(ring, ring_repeat);

    const tsp::IslandResult global =
        tsp::run_openmp_islands(dm, params, tsp::MigrationTopology::GlobalBest);
    assert_island_result_valid(dm, global, params.island_count, params.sa_params.iterations);
    assert(global.migration_rounds == 5);
    assert(global.migration_attempts == global.migration_rounds * (params.island_count - 1));

    // The same migration core supports every QLSA state-machine variant.
    tsp::IslandParams qlsa = params;
    qlsa.algorithm = tsp::IslandAlgorithm::QLSA;
    qlsa.qlsa_params.sa = params.sa_params;
    qlsa.qlsa_params.sa.iterations = 600;
    qlsa.migration_interval = 200;
    for (const std::string& variant : {"current", "paper", "paper-sb"}) {
        qlsa.qlsa_params.variant = variant;
        const tsp::IslandResult qlsa_global =
            tsp::run_openmp_islands(dm, qlsa, tsp::MigrationTopology::GlobalBest);
        assert_island_result_valid(
            dm, qlsa_global, qlsa.island_count, qlsa.qlsa_params.sa.iterations);
        assert(qlsa_global.migration_rounds == 2);
    }
}

void test_shared_time_boundary(const tsp::DistanceMatrix& dm) {
    tsp::IslandParams params;
    params.algorithm = tsp::IslandAlgorithm::SA;
    params.island_count = 4;
    params.threads = 4;
    params.sa_params.iterations = 100000;
    params.sa_params.use_nearest_neighbor_init = false;
    params.migration_interval = 1000;
    params.time_limit_ms = 0;
    params.deadline_check_interval = 1;

    const tsp::IslandResult result =
        tsp::run_openmp_islands(dm, params, tsp::MigrationTopology::Ring);
    assert(result.deadline_reached);
    assert(result.total_iterations_completed == 0);
    assert(result.actual_threads == 1);
    assert(!result.used_openmp);
    assert(!result.iteration_budget_exhausted);
    assert(result.migration_rounds == 0);
    assert(tsp::is_valid_tour(result.best_tour, dm.size()));

    params.threads = 2;
    params.sa_params.iterations = 1000000000;
    params.migration_interval = 1000;
    params.time_limit_ms = 20;
    params.deadline_check_interval = 64;
    const tsp::IslandResult positive_limit =
        tsp::run_openmp_islands(dm, params, tsp::MigrationTopology::Independent);
    assert(positive_limit.deadline_reached);
    assert(positive_limit.total_iterations_completed > 0);
    assert(positive_limit.total_iterations_completed < positive_limit.total_iteration_budget);
    assert(positive_limit.elapsed_ms >= 10.0);
    assert(positive_limit.actual_threads >= 1);
    assert(positive_limit.actual_threads <= 2);
}

}  // namespace

int main() {
    const tsp::DistanceMatrix dm = make_irregular_dm();
    test_sa_chunk_equivalence(dm);
    test_qlsa_chunk_equivalence(dm, "current");
    test_qlsa_chunk_equivalence(dm, "paper");
    test_qlsa_chunk_equivalence(dm, "paper-sb");
    test_qlsa_migration_learning_state(dm);
    test_island_topologies(dm);
    test_shared_time_boundary(dm);
    std::cout << "test_island passed\n";
    return 0;
}
