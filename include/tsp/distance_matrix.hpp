#pragma once

#include <vector>

#include "tsp/instance.hpp"

namespace tsp {

class DistanceMatrix {
public:
    explicit DistanceMatrix(const Instance& instance);

    [[nodiscard]] int size() const noexcept { return n_; }
    [[nodiscard]] int dist(int i, int j) const noexcept { return data_[static_cast<size_t>(i) * n_ + j]; }
    [[nodiscard]] const std::vector<int>& raw() const noexcept { return data_; }

private:
    int n_ = 0;
    std::vector<int> data_;
};

}  // namespace tsp
