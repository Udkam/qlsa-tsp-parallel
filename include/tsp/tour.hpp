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

}  // namespace tsp
