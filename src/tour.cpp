#include "tsp/tour.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tsp {

Tour identity_tour(int n) {
    Tour tour(static_cast<size_t>(n));
    std::iota(tour.begin(), tour.end(), 0);
    return tour;
}

Tour random_tour(int n, Rng& rng) {
    Tour tour = identity_tour(n);
    std::shuffle(tour.begin(), tour.end(), rng.engine());
    return tour;
}

Tour nearest_neighbor_tour(const DistanceMatrix& dm, int start) {
    const int n = dm.size();
    if (n <= 0) {
        return {};
    }
    if (start < 0 || start >= n) {
        throw std::out_of_range("nearest_neighbor_tour start index out of range");
    }

    Tour tour;
    tour.reserve(static_cast<size_t>(n));
    std::vector<unsigned char> used(static_cast<size_t>(n), 0);

    int current = start;
    tour.push_back(current);
    used[static_cast<size_t>(current)] = 1;

    for (int step = 1; step < n; ++step) {
        int best_city = -1;
        int best_dist = std::numeric_limits<int>::max();
        for (int city = 0; city < n; ++city) {
            if (used[static_cast<size_t>(city)] != 0) {
                continue;
            }
            const int d = dm.dist(current, city);
            if (d < best_dist || (d == best_dist && city < best_city)) {
                best_dist = d;
                best_city = city;
            }
        }
        current = best_city;
        used[static_cast<size_t>(current)] = 1;
        tour.push_back(current);
    }

    return tour;
}

bool is_valid_tour(const Tour& tour, int n) {
    if (n < 0 || static_cast<int>(tour.size()) != n) {
        return false;
    }
    std::vector<unsigned char> seen(static_cast<size_t>(n), 0);
    for (int city : tour) {
        if (city < 0 || city >= n) {
            return false;
        }
        if (seen[static_cast<size_t>(city)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(city)] = 1;
    }
    return true;
}

int tour_length(const Tour& tour, const DistanceMatrix& dm) {
    const int n = dm.size();
    if (!is_valid_tour(tour, n)) {
        throw std::invalid_argument("tour_length requires a legal tour");
    }
    if (n <= 1) {
        return 0;
    }
    int total = 0;
    for (int i = 0; i < n; ++i) {
        const int a = tour[static_cast<size_t>(i)];
        const int b = tour[static_cast<size_t>((i + 1) % n)];
        total += dm.dist(a, b);
    }
    return total;
}

bool is_valid_2opt_move(int n, int i, int k) {
    return n >= 3 && 0 <= i && i < k && k < n && !(i == 0 && k == n - 1);
}

int delta_2opt(const Tour& tour, const DistanceMatrix& dm, int i, int k) {
    const int n = dm.size();
    if (!is_valid_2opt_move(n, i, k) || static_cast<int>(tour.size()) != n) {
        throw std::invalid_argument("invalid 2-opt move");
    }

    const int a = tour[static_cast<size_t>((i - 1 + n) % n)];
    const int b = tour[static_cast<size_t>(i)];
    const int c = tour[static_cast<size_t>(k)];
    const int d = tour[static_cast<size_t>((k + 1) % n)];

    return dm.dist(a, c) + dm.dist(b, d) - dm.dist(a, b) - dm.dist(c, d);
}

void apply_2opt(Tour& tour, int i, int k) {
    if (!is_valid_2opt_move(static_cast<int>(tour.size()), i, k)) {
        throw std::invalid_argument("invalid 2-opt move");
    }
    std::reverse(tour.begin() + i, tour.begin() + k + 1);
}

Tour double_bridge(const Tour& tour, Rng& rng) {
    const int n = static_cast<int>(tour.size());
    if (n < 8) {
        // Too small for a clean double bridge: fall back to a single swap.
        Tour out = tour;
        if (n >= 2) {
            std::uniform_int_distribution<int> dist(0, n - 1);
            const int i = dist(rng.engine());
            const int j = dist(rng.engine());
            std::swap(out[static_cast<size_t>(i)], out[static_cast<size_t>(j)]);
        }
        return out;
    }
    // Choose 1 <= p1 < p2 < p3 < n, then reconnect as A + C + B + D.
    std::uniform_int_distribution<int> d1(1, n - 3);
    const int p1 = d1(rng.engine());
    std::uniform_int_distribution<int> d2(p1 + 1, n - 2);
    const int p2 = d2(rng.engine());
    std::uniform_int_distribution<int> d3(p2 + 1, n - 1);
    const int p3 = d3(rng.engine());

    Tour out;
    out.reserve(tour.size());
    out.insert(out.end(), tour.begin(), tour.begin() + p1);
    out.insert(out.end(), tour.begin() + p2, tour.begin() + p3);
    out.insert(out.end(), tour.begin() + p1, tour.begin() + p2);
    out.insert(out.end(), tour.begin() + p3, tour.end());
    return out;
}

int hamming_distance(const Tour& a, const Tour& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("hamming_distance requires equal-length tours");
    }
    int count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++count;
        }
    }
    return count;
}

}  // namespace tsp
