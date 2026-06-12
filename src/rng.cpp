#include "tsp/rng.hpp"

namespace tsp {

Rng::Rng(uint64_t seed) : engine_(seed) {}

int Rng::uniform_int(int low, int high) {
    std::uniform_int_distribution<int> dist(low, high);
    return dist(engine_);
}

double Rng::uniform01() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(engine_);
}

}  // namespace tsp
