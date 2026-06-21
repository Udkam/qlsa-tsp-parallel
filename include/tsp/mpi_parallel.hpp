#pragma once

#include "tsp/parallel.hpp"

namespace tsp {

struct MpiParallelResult {
    ParallelResult global;
    ParallelResult local;

    int world_size = 1;
    int world_rank = 0;
    int local_chains = 0;
    int chain_offset = 0;
    double communication_ms = 0.0;
    bool mpi_enabled = false;
};

[[nodiscard]] bool mpi_runtime_available() noexcept;

MpiParallelResult run_mpi_parallel_chains(const DistanceMatrix& dm, const ParallelParams& params);

}  // namespace tsp
