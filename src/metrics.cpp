#include "tsp/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tsp {

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double stddev_population(const std::vector<double>& values) {
    if (values.size() <= 1) {
        return 0.0;
    }
    const double avg = mean(values);
    double acc = 0.0;
    for (double value : values) {
        const double diff = value - avg;
        acc += diff * diff;
    }
    return std::sqrt(acc / static_cast<double>(values.size()));
}

int min_value(const std::vector<int>& values) {
    if (values.empty()) {
        throw std::invalid_argument("min_value requires a non-empty vector");
    }
    return *std::min_element(values.begin(), values.end());
}

}  // namespace tsp
