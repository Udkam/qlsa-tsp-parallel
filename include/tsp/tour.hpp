#pragma once

#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/rng.hpp"

namespace tsp {

using Tour = std::vector<int>;

[[nodiscard]] Tour identity_tour(int n);
[[nodiscard]] Tour random_tour(int n, Rng& rng);
[[nodiscard]] Tour nearest_neighbor_tour(const DistanceMatrix& dm, int start = 0);
[[nodiscard]] bool is_valid_tour(const Tour& tour, int n);
[[nodiscard]] int tour_length(const Tour& tour, const DistanceMatrix& dm);
[[nodiscard]] bool is_valid_2opt_move(int n, int i, int k);
[[nodiscard]] int delta_2opt(const Tour& tour, const DistanceMatrix& dm, int i, int k);
void apply_2opt(Tour& tour, int i, int k);

// Building blocks toward a paper-style (candidate-leader) QLSA variant.
// double_bridge applies a 4-opt double-bridge perturbation and returns a new
// legal permutation; hamming_distance gives the position-wise difference count
// used by the paper's diversity-state definition. See
// docs/dev/paper_lite_qlsa_design.md.
[[nodiscard]] Tour double_bridge(const Tour& tour, Rng& rng);
[[nodiscard]] int hamming_distance(const Tour& a, const Tour& b);

}  // namespace tsp
