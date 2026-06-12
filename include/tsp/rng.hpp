#pragma once

#include <cstdint>
#include <random>

namespace tsp {

class Rng {
public:
    explicit Rng(uint64_t seed);

    [[nodiscard]] int uniform_int(int low, int high);
    [[nodiscard]] double uniform01();
    [[nodiscard]] std::mt19937_64& engine() noexcept { return engine_; }

private:
    std::mt19937_64 engine_;
};

}  // namespace tsp
