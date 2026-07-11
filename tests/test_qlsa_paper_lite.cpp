// Tests for paper-style QLSA building blocks: double-bridge perturbation and
// Hamming-distance diversity measure. These are the C++ utilities toward a
// candidate-leader / diversity-state variant; see
// docs/dev/paper_lite_qlsa_design.md.

#ifdef NDEBUG
#error "Tests require assertions; NDEBUG must not be defined"
#endif
#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/rng.hpp"
#include "tsp/tsplib_parser.hpp"
#include "tsp/tour.hpp"

int main() {
    // hamming_distance: position-wise difference count.
    const tsp::Tour a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    tsp::Tour b = a;
    assert(tsp::hamming_distance(a, b) == 0);
    std::swap(b[2], b[7]);
    assert(tsp::hamming_distance(a, b) == 2);

    // double_bridge on a sufficiently large tour stays a legal permutation and
    // generally differs from the input (segments are rearranged).
    tsp::Rng rng(42);
    const tsp::Tour base = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    bool any_changed = false;
    for (int trial = 0; trial < 50; ++trial) {
        const tsp::Tour db = tsp::double_bridge(base, rng);
        assert(tsp::is_valid_tour(db, static_cast<int>(base.size())));
        if (tsp::hamming_distance(base, db) > 0) {
            any_changed = true;
        }
    }
    assert(any_changed);

    // Small instance (n < 8): fall back keeps a legal permutation.
    const tsp::Tour small = {0, 1, 2, 3};
    const tsp::Tour small_db = tsp::double_bridge(small, rng);
    assert(tsp::is_valid_tour(small_db, static_cast<int>(small.size())));

    // Diversity bucket reasonableness: identical -> low, reversed -> high.
    const int n = static_cast<int>(base.size());
    tsp::Tour reversed(base.rbegin(), base.rend());
    const double identical_frac =
        static_cast<double>(tsp::hamming_distance(base, base)) / n;
    const double reversed_frac =
        static_cast<double>(tsp::hamming_distance(base, reversed)) / n;
    assert(identical_frac < 0.5);   // low diversity
    assert(reversed_frac >= 0.5);   // high diversity

    const std::filesystem::path square_path =
        std::filesystem::path(TEST_SOURCE_DIR) / "fixtures" / "square4.tsp";
    const tsp::Instance square = tsp::load_tsplib(square_path.string());
    const tsp::DistanceMatrix dm(square);

    tsp::SAParams sa_params;
    sa_params.iterations = 500;
    sa_params.seed = 7;
    sa_params.initial_temperature = 100.0;
    sa_params.final_temperature = 1e-3;
    sa_params.use_nearest_neighbor_init = true;

    tsp::QLSAParams paper_params;
    paper_params.sa = sa_params;
    paper_params.variant = "paper";
    paper_params.policy = "epsilon-greedy";
    paper_params.epsilon = 0.2;
    const tsp::QLSAResult paper = tsp::run_qlsa_2opt(dm, paper_params);
    assert(tsp::is_valid_tour(paper.best_tour, dm.size()));
    assert(paper.best_length == 40);
    assert(paper.q_table.size() == 1);
    assert(paper.q_table.front().size() == 4);
    assert(paper.action_counts.size() == 4);

    tsp::QLSAParams sb_params = paper_params;
    sb_params.variant = "paper-sb";
    sb_params.diversity_threshold = 0.5;
    sb_params.sa.seed = 8;
    const tsp::QLSAResult sb = tsp::run_qlsa_2opt(dm, sb_params);
    assert(tsp::is_valid_tour(sb.best_tour, dm.size()));
    assert(sb.best_length == 40);
    assert(sb.q_table.size() == 2);
    assert(sb.q_table.front().size() == 4);
    assert(sb.action_counts.size() == 4);

    std::cout << "test_qlsa_paper_lite passed\n";
    return 0;
}
