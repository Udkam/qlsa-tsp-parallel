#pragma once

#include <vector>

namespace tsp {

[[nodiscard]] double mean(const std::vector<double>& values);
[[nodiscard]] double stddev_population(const std::vector<double>& values);
[[nodiscard]] int min_value(const std::vector<int>& values);

}  // namespace tsp
