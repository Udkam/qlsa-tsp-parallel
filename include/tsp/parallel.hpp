#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/sa.hpp"

namespace tsp {

enum class AlgorithmKind {
    SA,
    QLSA,
};

enum class CudaMode {
    Chain,
    Candidate,
};

enum class CudaReversalMode {
    Serial,
    Parallel,
};

enum class CudaCandidatePolicy {
    Best,
    Random,
    Hybrid,
};

enum class ParallelBackend {
    CpuSerial,
    OpenMP,
    Cuda,
};

struct ParallelParams {
    AlgorithmKind algorithm = AlgorithmKind::SA;

    SAParams sa_params;
    QLSAParams qlsa_params;

    int chains = 1;
    int threads = 1;
    bool cuda_enabled = false;
    int cuda_block_size = 128;
    CudaMode cuda_mode = CudaMode::Chain;
    int cuda_candidates_per_iter = 32;
    CudaReversalMode cuda_reversal_mode = CudaReversalMode::Serial;
    CudaCandidatePolicy cuda_candidate_policy = CudaCandidatePolicy::Best;
    uint64_t base_seed = 1;
};

struct ChainResult {
    int chain_id = 0;
    uint64_t seed = 0;
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length = 0;
    double elapsed_ms = 0.0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    int64_t iterations_completed = 0;
    bool deadline_reached = false;
};

struct ParallelResult {
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length_of_best_chain = 0;

    // Backward-compatible total wall-clock duration. New code should prefer
    // total_elapsed_ms, which is always kept identical to this field.
    double elapsed_ms = 0.0;
    double total_elapsed_ms = 0.0;

    // CUDA event time for the search kernel only. This remains zero for CPU
    // execution and for CUDA requests that fall back to the CPU.
    double cuda_kernel_elapsed_ms = 0.0;

    ParallelBackend requested_backend = ParallelBackend::CpuSerial;
    ParallelBackend actual_backend = ParallelBackend::CpuSerial;
    bool backend_fallback = false;
    std::string backend_fallback_reason;

    int chains = 1;
    int threads = 1;
    int actual_threads = 1;
    uint64_t base_seed = 1;

    int64_t total_accepted_moves = 0;
    int64_t total_improved_moves = 0;
    int64_t total_iterations_completed = 0;
    bool deadline_reached = false;

    std::vector<ChainResult> chain_results;
};

[[nodiscard]] bool openmp_available() noexcept;
[[nodiscard]] const char* parallel_backend_name(ParallelBackend backend) noexcept;
[[nodiscard]] uint64_t splitmix64(uint64_t x) noexcept;
[[nodiscard]] uint64_t chain_seed(uint64_t base_seed, int chain_id) noexcept;

ParallelResult run_parallel_chains(const DistanceMatrix& dm, const ParallelParams& params);

}  // namespace tsp
