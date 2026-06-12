#include "tsp/sa.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tsp/rng.hpp"
#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {

SAResult run_sa_2opt(const DistanceMatrix& dm, const SAParams& params) {
    const int n = dm.size();
    if (n <= 0) {
        throw std::invalid_argument("run_sa_2opt requires a non-empty distance matrix");
    }
    if (params.iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (params.initial_temperature <= 0.0 || params.final_temperature <= 0.0) {
        throw std::invalid_argument("temperatures must be positive");
    }

    Timer timer;
    Rng rng(params.seed);

    Tour current = params.use_nearest_neighbor_init ? nearest_neighbor_tour(dm) : random_tour(n, rng);
    int current_length = tour_length(current, dm);
    Tour best = current;
    int best_length = current_length;

    SAResult result;

    double temperature = params.initial_temperature;
    const double alpha = (params.iterations > 0)
                             ? std::pow(params.final_temperature / params.initial_temperature,
                                        1.0 / static_cast<double>(params.iterations))
                             : 1.0;

    for (int64_t iter = 0; n >= 3 && iter < params.iterations; ++iter) {
        int i = 0;
        int k = 0;
        do {
            i = rng.uniform_int(0, n - 1);
            k = rng.uniform_int(0, n - 1);
            if (i > k) {
                std::swap(i, k);
            }
        } while (!is_valid_2opt_move(n, i, k));

        const int delta = delta_2opt(current, dm, i, k);
        bool accept = delta < 0;
        if (!accept) {
            const double probability = std::exp(-static_cast<double>(delta) / temperature);
            accept = rng.uniform01() < probability;
        }

        if (accept) {
            apply_2opt(current, i, k);
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

        temperature *= alpha;
    }

    const int checked_best_length = tour_length(best, dm);
    if (checked_best_length != best_length) {
        throw std::runtime_error("SA best_length verification failed");
    }
    const int checked_final_length = tour_length(current, dm);
    if (checked_final_length != current_length) {
        throw std::runtime_error("SA final_length verification failed");
    }

    result.best_tour = std::move(best);
    result.best_length = best_length;
    result.final_length = current_length;
    result.elapsed_ms = timer.elapsed_ms();
    return result;
}

}  // namespace tsp
