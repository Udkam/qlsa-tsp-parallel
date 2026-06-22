#pragma once

#include <cstdint>
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
};

struct ParallelResult {
    std::vector<int> best_tour;
    int best_length = 0;
    int final_length_of_best_chain = 0;

    double elapsed_ms = 0.0;

    int chains = 1;
    int threads = 1;
    uint64_t base_seed = 1;

    int64_t total_accepted_moves = 0;
    int64_t total_improved_moves = 0;

    std::vector<ChainResult> chain_results;
};

[[nodiscard]] bool openmp_available() noexcept;
[[nodiscard]] uint64_t splitmix64(uint64_t x) noexcept;
[[nodiscard]] uint64_t chain_seed(uint64_t base_seed, int chain_id) noexcept;

ParallelResult run_parallel_chains(const DistanceMatrix& dm, const ParallelParams& params);

}  // namespace tsp
