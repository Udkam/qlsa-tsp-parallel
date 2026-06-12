#pragma once

#include <chrono>

namespace tsp {

class Timer {
public:
    Timer();

    void reset();
    [[nodiscard]] double elapsed_ms() const;

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace tsp
