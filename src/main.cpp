#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/cuda.hpp"
#include "tsp/island.hpp"
#include "tsp/metrics.hpp"
#include "tsp/parallel.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/sa.hpp"
#include "tsp/timer.hpp"
#include "tsp/tsplib_parser.hpp"
#include "tsp/tour.hpp"

namespace {

struct CliOptions {
    std::string input;
    int64_t iterations = 1000000;
    uint64_t seed = 1;
    double t0 = 1000.0;
    double tf = 1e-3;
    std::string init = "nn";
    int repeat = 1;
    bool csv_only = false;
    bool use_qlsa = false;
    std::string parallel = "none";
    int chains = 1;
    int threads = 1;
    int cuda_block_size = 128;
    std::string cuda_mode = "chain";
    int cuda_candidates_per_iter = 32;
    std::string cuda_reversal_mode = "serial";
    std::string cuda_candidate_policy = "best";
    double qlsa_alpha = 0.1;
    double qlsa_gamma = 0.9;
    double qlsa_epsilon = 0.1;
    std::string qlsa_policy = "epsilon-greedy";
    std::string qlsa_variant = "current";
    double qlsa_diversity_threshold = 0.5;
    std::optional<int64_t> time_limit_ms;
    std::string migration_topology = "disabled";
    int64_t migration_interval = 10000;
};

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --input data/berlin52.tsp --iterations 1000000 --seed 1 --t0 1000 --tf 0.001 --init nn [--repeat 1]\n"
        << "       " << program
        << " --qlsa --input data/berlin52.tsp --iterations 1000000 --seed 1 --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy\n"
        << "       " << program
        << " --qlsa --qlsa_variant paper-sb --input data/berlin52.tsp --iterations 1000000 --seed 1\n"
        << "       " << program
        << " --input data/berlin52.tsp --parallel omp --chains 8 --threads 4 --iterations 1000000 --seed 1\n"
        << "       " << program
        << " --input data/berlin52.tsp --qlsa --parallel cuda --chains 32 --cuda_block_size 128 --iterations 1000000 --seed 1\n"
        << "       " << program
        << " --input data/eil101.tsp --parallel omp --chains 8 --threads 8 --iterations 1000000000 --time-limit-ms 30000\n"
        << "       " << program
        << " --input data/eil101.tsp --parallel omp --chains 8 --threads 8 --migration-topology ring --migration-interval 10000\n";
    std::cerr
        << "CUDA options: --cuda_mode chain|candidate --cuda_candidates_per_iter 32 "
        << "--cuda_reversal_mode serial|parallel --cuda_candidate_policy best|random|hybrid\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--input") {
            options.input = require_value(arg);
        } else if (arg == "--iterations") {
            options.iterations = std::stoll(require_value(arg));
        } else if (arg == "--seed") {
            options.seed = std::stoull(require_value(arg));
        } else if (arg == "--t0") {
            options.t0 = std::stod(require_value(arg));
        } else if (arg == "--tf") {
            options.tf = std::stod(require_value(arg));
        } else if (arg == "--init") {
            options.init = require_value(arg);
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(require_value(arg));
        } else if (arg == "--csv-only") {
            options.csv_only = true;
        } else if (arg == "--qlsa") {
            options.use_qlsa = true;
        } else if (arg == "--parallel") {
            options.parallel = require_value(arg);
        } else if (arg == "--chains") {
            options.chains = std::stoi(require_value(arg));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value(arg));
        } else if (arg == "--cuda_block_size") {
            options.cuda_block_size = std::stoi(require_value(arg));
        } else if (arg == "--cuda_mode") {
            options.cuda_mode = require_value(arg);
        } else if (arg == "--cuda_candidates_per_iter") {
            options.cuda_candidates_per_iter = std::stoi(require_value(arg));
        } else if (arg == "--cuda_reversal_mode") {
            options.cuda_reversal_mode = require_value(arg);
        } else if (arg == "--cuda_candidate_policy") {
            options.cuda_candidate_policy = require_value(arg);
        } else if (arg == "--alpha") {
            options.qlsa_alpha = std::stod(require_value(arg));
        } else if (arg == "--gamma") {
            options.qlsa_gamma = std::stod(require_value(arg));
        } else if (arg == "--epsilon") {
            options.qlsa_epsilon = std::stod(require_value(arg));
        } else if (arg == "--policy") {
            options.qlsa_policy = require_value(arg);
        } else if (arg == "--qlsa_variant") {
            options.qlsa_variant = require_value(arg);
        } else if (arg == "--diversity_threshold") {
            options.qlsa_diversity_threshold = std::stod(require_value(arg));
        } else if (arg == "--time-limit-ms") {
            options.time_limit_ms = std::stoll(require_value(arg));
        } else if (arg == "--migration-topology") {
            options.migration_topology = require_value(arg);
        } else if (arg == "--migration-interval") {
            options.migration_interval = std::stoll(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.input.empty()) {
        throw std::invalid_argument("--input is required");
    }
    if (options.iterations < 1) {
        throw std::invalid_argument("--iterations must be >= 1");
    }
    if (options.repeat <= 0) {
        throw std::invalid_argument("--repeat must be positive");
    }
    if (options.init != "nn" && options.init != "random") {
        throw std::invalid_argument("--init must be nn or random");
    }
    if (options.parallel != "none" && options.parallel != "omp" && options.parallel != "cuda") {
        throw std::invalid_argument("--parallel must be none, omp, or cuda");
    }
    if (options.chains < 1) {
        throw std::invalid_argument("--chains must be >= 1");
    }
    if (options.threads < 1) {
        throw std::invalid_argument("--threads must be >= 1");
    }
    if (options.cuda_block_size < 1 || options.cuda_block_size > 1024) {
        throw std::invalid_argument("--cuda_block_size must be in [1, 1024]");
    }
    if (options.cuda_mode != "chain" && options.cuda_mode != "candidate") {
        throw std::invalid_argument("--cuda_mode must be chain or candidate");
    }
    if (options.cuda_candidates_per_iter <= 0) {
        throw std::invalid_argument("--cuda_candidates_per_iter must be positive");
    }
    if (options.cuda_candidates_per_iter > options.cuda_block_size) {
        throw std::invalid_argument("--cuda_candidates_per_iter must be <= --cuda_block_size");
    }
    if (options.cuda_reversal_mode != "serial" && options.cuda_reversal_mode != "parallel") {
        throw std::invalid_argument("--cuda_reversal_mode must be serial or parallel");
    }
    if (options.cuda_candidate_policy != "best" && options.cuda_candidate_policy != "random" &&
        options.cuda_candidate_policy != "hybrid") {
        throw std::invalid_argument("--cuda_candidate_policy must be best, random, or hybrid");
    }
    if (options.qlsa_alpha <= 0.0 || options.qlsa_alpha > 1.0) {
        throw std::invalid_argument("--alpha must be in (0, 1]");
    }
    if (options.qlsa_gamma < 0.0 || options.qlsa_gamma > 1.0) {
        throw std::invalid_argument("--gamma must be in [0, 1]");
    }
    if (options.qlsa_epsilon < 0.0 || options.qlsa_epsilon > 1.0) {
        throw std::invalid_argument("--epsilon must be in [0, 1]");
    }
    if (options.qlsa_policy != "epsilon-greedy" && options.qlsa_policy != "softmax") {
        throw std::invalid_argument("--policy must be epsilon-greedy or softmax");
    }
    if (options.qlsa_variant != "current" && options.qlsa_variant != "paper" &&
        options.qlsa_variant != "paper-sb") {
        throw std::invalid_argument("--qlsa_variant must be current, paper, or paper-sb");
    }
    if (options.qlsa_diversity_threshold < 0.0 || options.qlsa_diversity_threshold > 1.0) {
        throw std::invalid_argument("--diversity_threshold must be in [0, 1]");
    }
    if (options.time_limit_ms.has_value() && *options.time_limit_ms <= 0) {
        throw std::invalid_argument("--time-limit-ms must be positive");
    }
    if (options.migration_topology != "disabled" &&
        options.migration_topology != "independent" &&
        options.migration_topology != "ring" &&
        options.migration_topology != "global") {
        throw std::invalid_argument(
            "--migration-topology must be disabled, independent, ring, or global");
    }
    if (options.migration_interval <= 0) {
        throw std::invalid_argument("--migration-interval must be positive");
    }
    if (options.parallel == "cuda" && options.use_qlsa && options.qlsa_variant != "current") {
        throw std::invalid_argument("CUDA QLSA currently supports --qlsa_variant current only");
    }
    if (options.parallel == "cuda" && options.time_limit_ms.has_value()) {
        throw std::invalid_argument("--time-limit-ms currently supports CPU/OpenMP runs only");
    }
    if (options.parallel == "cuda" && options.migration_topology != "disabled") {
        throw std::invalid_argument("island migration currently supports CPU/OpenMP runs only");
    }
    if ((options.migration_topology == "ring" || options.migration_topology == "global") &&
        options.chains < 2) {
        throw std::invalid_argument("ring/global migration requires --chains >= 2");
    }
    return options;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

struct RunRow {
    std::string algorithm;
    int best_length = 0;
    int final_length = 0;
    double elapsed_ms = 0.0;
    int64_t accepted_moves = 0;
    int64_t improved_moves = 0;
    double total_elapsed_ms = 0.0;
    double cuda_kernel_elapsed_ms = 0.0;
    std::string requested_backend = "cpu_serial";
    std::string actual_backend = "cpu_serial";
    bool backend_fallback = false;
    std::string backend_fallback_reason;
    int64_t iterations_completed = 0;
    bool deadline_reached = false;
    std::string migration_topology = "disabled";
    int64_t migration_interval = 0;
    int64_t migration_rounds = 0;
    int64_t migration_attempts = 0;
    int64_t migrations_adopted = 0;
    int actual_threads = 1;
};

std::string algorithm_label(const CliOptions& options) {
    std::string base = options.use_qlsa ? "qlsa" : "sa";
    if (options.use_qlsa && options.qlsa_variant != "current") {
        base += "-" + options.qlsa_variant;
    }
    if (options.migration_topology != "disabled") {
        base += "-island-" + options.migration_topology;
        if (options.parallel == "omp") {
            base += "-omp";
        }
        return base;
    }
    if (options.parallel == "cuda") {
        if (options.cuda_mode == "candidate" && options.cuda_candidate_policy != "best") {
            return base + "-cuda-candidate-" + options.cuda_candidate_policy;
        }
        return base + "-cuda-" + options.cuda_mode;
    }
    if (options.parallel == "omp") {
        return base + "-omp";
    }
    if (options.chains > 1) {
        return base + "-multichain";
    }
    return base;
}

int effective_worker_count(const CliOptions& options) {
    if (options.parallel == "cuda") {
        return options.cuda_block_size;
    }
    if (options.parallel == "omp") {
        return options.threads;
    }
    return 1;
}

tsp::SAParams make_sa_params(const CliOptions& options, uint64_t seed) {
    tsp::SAParams params;
    params.iterations = options.iterations;
    params.initial_temperature = options.t0;
    params.final_temperature = options.tf;
    params.seed = seed;
    params.use_nearest_neighbor_init = options.init == "nn";
    return params;
}

tsp::QLSAParams make_qlsa_params(const CliOptions& options, const tsp::SAParams& sa_params) {
    tsp::QLSAParams params;
    params.sa = sa_params;
    params.alpha = options.qlsa_alpha;
    params.gamma = options.qlsa_gamma;
    params.epsilon = options.qlsa_epsilon;
    params.policy = options.qlsa_policy;
    params.variant = options.qlsa_variant;
    params.diversity_threshold = options.qlsa_diversity_threshold;
    return params;
}

std::string requested_backend_name(const CliOptions& options) {
    if (options.parallel == "cuda") {
        return "cuda";
    }
    if (options.parallel == "omp") {
        return "openmp";
    }
    return "cpu_serial";
}

tsp::MigrationTopology migration_topology_from_options(const CliOptions& options) {
    if (options.migration_topology == "ring") {
        return tsp::MigrationTopology::Ring;
    }
    if (options.migration_topology == "global") {
        return tsp::MigrationTopology::GlobalBest;
    }
    return tsp::MigrationTopology::Independent;
}

void print_csv_row(const tsp::Instance& instance,
                   const CliOptions& options,
                   uint64_t seed,
                   const RunRow& row) {
    std::cout << row.algorithm << ','
              << csv_escape(instance.name) << ','
              << instance.dimension << ','
              << options.iterations << ','
              << seed << ','
              << options.init << ','
              << options.chains << ','
              << effective_worker_count(options) << ','
              << options.parallel << ','
              << row.best_length << ','
              << row.final_length << ','
              << std::fixed << std::setprecision(3) << row.elapsed_ms << ','
              << row.accepted_moves << ','
              << row.improved_moves << ','
              << std::fixed << std::setprecision(3) << row.total_elapsed_ms << ','
              << std::fixed << std::setprecision(3) << row.cuda_kernel_elapsed_ms << ','
              << row.requested_backend << ','
              << row.actual_backend << ','
              << (row.backend_fallback ? "true" : "false") << ','
              << csv_escape(row.backend_fallback_reason) << ','
              << row.iterations_completed << ','
              << (row.deadline_reached ? "true" : "false") << ','
              << row.migration_topology << ','
              << row.migration_interval << ','
              << row.migration_rounds << ','
              << row.migration_attempts << ','
              << row.migrations_adopted << ','
              << row.actual_threads << '\n';
}

void print_human_run(int run_index,
                     int repeat,
                     uint64_t seed,
                     const CliOptions& options,
                     const RunRow& row) {
    std::cout << "Run " << run_index << '/' << repeat
              << " algorithm=" << row.algorithm
              << " seed=" << seed
              << " chains=" << options.chains
              << " threads=" << effective_worker_count(options)
              << " parallel=" << options.parallel
              << " best_length=" << row.best_length
              << " final_length=" << row.final_length
              << " elapsed_ms=" << std::fixed << std::setprecision(3) << row.elapsed_ms
              << " accepted=" << row.accepted_moves
              << " improved=" << row.improved_moves
              << " iterations_completed=" << row.iterations_completed
              << " requested_backend=" << row.requested_backend
              << " actual_backend=" << row.actual_backend
              << " actual_threads=" << row.actual_threads
              << " deadline_reached=" << (row.deadline_reached ? "true" : "false");
    if (row.migration_topology != "disabled") {
        std::cout << " migration_topology=" << row.migration_topology
                  << " migration_rounds=" << row.migration_rounds
                  << " migrations_adopted=" << row.migrations_adopted
                  << '/' << row.migration_attempts;
    }
    std::cout << '\n';
}

RunRow row_from_sa_result(const std::string& algorithm, const tsp::SAResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length;
    row.elapsed_ms = result.elapsed_ms;
    row.total_elapsed_ms = result.elapsed_ms;
    row.accepted_moves = result.accepted_moves;
    row.improved_moves = result.improved_moves;
    row.iterations_completed = result.iterations_completed;
    row.deadline_reached = result.deadline_reached;
    return row;
}

RunRow row_from_qlsa_result(const std::string& algorithm, const tsp::QLSAResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length;
    row.elapsed_ms = result.elapsed_ms;
    row.total_elapsed_ms = result.elapsed_ms;
    row.accepted_moves = result.accepted_moves;
    row.improved_moves = result.improved_moves;
    row.iterations_completed = result.iterations_completed;
    row.deadline_reached = result.deadline_reached;
    return row;
}

RunRow row_from_parallel_result(const std::string& algorithm,
                                const tsp::ParallelResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length_of_best_chain;
    row.elapsed_ms = result.elapsed_ms;
    row.total_elapsed_ms = result.total_elapsed_ms;
    row.cuda_kernel_elapsed_ms = result.cuda_kernel_elapsed_ms;
    row.requested_backend = tsp::parallel_backend_name(result.requested_backend);
    row.actual_backend = tsp::parallel_backend_name(result.actual_backend);
    row.backend_fallback = result.backend_fallback;
    row.backend_fallback_reason = result.backend_fallback_reason;
    row.accepted_moves = result.total_accepted_moves;
    row.improved_moves = result.total_improved_moves;
    row.iterations_completed = result.total_iterations_completed;
    row.deadline_reached = result.deadline_reached;
    row.actual_threads = result.actual_threads;
    return row;
}

RunRow row_from_island_result(const std::string& algorithm,
                              const CliOptions& options,
                              const tsp::IslandResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length_of_best_island;
    row.elapsed_ms = result.elapsed_ms;
    row.total_elapsed_ms = result.elapsed_ms;
    row.accepted_moves = result.total_accepted_moves;
    row.improved_moves = result.total_improved_moves;
    row.requested_backend = requested_backend_name(options);
    row.actual_backend = result.used_openmp ? "openmp" : "cpu_serial";
    row.backend_fallback = row.requested_backend != row.actual_backend;
    if (row.backend_fallback) {
        row.backend_fallback_reason =
            "OpenMP was requested but the island run executed on the serial CPU backend";
    }
    row.iterations_completed = result.total_iterations_completed;
    row.deadline_reached = result.deadline_reached;
    row.migration_topology = options.migration_topology == "disabled"
                                 ? "independent"
                                 : options.migration_topology;
    row.migration_interval = options.migration_interval;
    row.migration_rounds = result.migration_rounds;
    row.migration_attempts = result.migration_attempts;
    row.migrations_adopted = result.migrations_adopted;
    row.actual_threads = result.actual_threads;
    return row;
}

tsp::SearchClock::time_point saturated_deadline_after_ms(int64_t time_limit_ms) {
    const auto now = tsp::SearchClock::now();
    const auto maximum_delta = std::chrono::duration_cast<std::chrono::milliseconds>(
        tsp::SearchClock::time_point::max() - now);
    if (time_limit_ms >= maximum_delta.count()) {
        return tsp::SearchClock::time_point::max();
    }
    return now + std::chrono::milliseconds(time_limit_ms);
}

tsp::SAResult run_time_limited_sa(const tsp::DistanceMatrix& dm,
                                  const tsp::SAParams& params,
                                  int64_t time_limit_ms) {
    tsp::Timer timer;
    const auto deadline = saturated_deadline_after_ms(time_limit_ms);
    tsp::SAState state = tsp::initialize_sa_state(dm, params);
    tsp::SearchChunkOptions chunk;
    chunk.max_iterations = params.iterations;
    chunk.deadline = deadline;
    (void)tsp::run_sa_chunk(dm, state, chunk);
    tsp::SAResult result = tsp::finalize_sa_state(dm, state);
    result.elapsed_ms = timer.elapsed_ms();
    return result;
}

tsp::QLSAResult run_time_limited_qlsa(const tsp::DistanceMatrix& dm,
                                      const tsp::QLSAParams& params,
                                      int64_t time_limit_ms) {
    tsp::Timer timer;
    const auto deadline = saturated_deadline_after_ms(time_limit_ms);
    tsp::QLSAState state = tsp::initialize_qlsa_state(dm, params);
    tsp::SearchChunkOptions chunk;
    chunk.max_iterations = params.sa.iterations;
    chunk.deadline = deadline;
    (void)tsp::run_qlsa_chunk(dm, state, chunk);
    tsp::QLSAResult result = tsp::finalize_qlsa_state(dm, state);
    result.elapsed_ms = timer.elapsed_ms();
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        tsp::Instance instance = tsp::load_tsplib(options.input);
        tsp::DistanceMatrix dm(instance);

        if (options.parallel == "omp" && !tsp::openmp_available()) {
            std::cerr << "Warning: OpenMP was not enabled at build time; falling back to serial multi-chain execution.\n";
        }
        if (!options.csv_only) {
            std::cout << "Instance: " << instance.name << " (n=" << instance.dimension << ")\n";
            std::cout << "Input: " << std::filesystem::path(options.input).generic_string() << '\n';
            std::cout << "Algorithm: " << algorithm_label(options) << '\n';
            std::cout << "Parallel: " << options.parallel
                      << " chains=" << options.chains
                      << " threads=" << effective_worker_count(options);
            if (options.parallel == "cuda") {
                std::cout << " cuda_block_size=" << options.cuda_block_size;
                std::cout << " cuda_mode=" << options.cuda_mode
                          << " cuda_candidates_per_iter=" << options.cuda_candidates_per_iter
                          << " cuda_reversal_mode=" << options.cuda_reversal_mode
                          << " cuda_candidate_policy=" << options.cuda_candidate_policy;
            }
            if (options.time_limit_ms.has_value()) {
                std::cout << " time_limit_ms=" << *options.time_limit_ms;
            }
            if (options.migration_topology != "disabled") {
                std::cout << " migration_topology=" << options.migration_topology
                          << " migration_interval=" << options.migration_interval;
            }
            std::cout << '\n';
            if (options.use_qlsa) {
                std::cout << "QLSA params: alpha=" << options.qlsa_alpha
                          << " gamma=" << options.qlsa_gamma
                          << " epsilon=" << options.qlsa_epsilon
                          << " policy=" << options.qlsa_policy
                          << " variant=" << options.qlsa_variant
                          << " diversity_threshold=" << options.qlsa_diversity_threshold << '\n';
            }
            std::cout
                << "CSV: algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,"
                   "best_length,final_length,elapsed_ms,accepted_moves,improved_moves,total_elapsed_ms,"
                   "cuda_kernel_elapsed_ms,requested_backend,actual_backend,backend_fallback,"
                   "backend_fallback_reason,iterations_completed,deadline_reached,migration_topology,"
                   "migration_interval,migration_rounds,migration_attempts,migrations_adopted,"
                   "actual_threads\n";
        }

        std::vector<double> elapsed_values;
        std::vector<int> best_lengths;
        elapsed_values.reserve(static_cast<size_t>(options.repeat));
        best_lengths.reserve(static_cast<size_t>(options.repeat));

        for (int r = 0; r < options.repeat; ++r) {
            const uint64_t run_seed = options.seed + static_cast<uint64_t>(r);
            const std::string algorithm = algorithm_label(options);
            const tsp::SAParams sa_params = make_sa_params(options, run_seed);
            RunRow row;

            const bool explicit_island = options.migration_topology != "disabled";
            const bool time_limited_multichain =
                options.time_limit_ms.has_value() &&
                (options.parallel == "omp" || options.chains > 1);

            if (explicit_island || time_limited_multichain) {
                tsp::IslandParams island_params;
                island_params.algorithm = options.use_qlsa
                                              ? tsp::IslandAlgorithm::QLSA
                                              : tsp::IslandAlgorithm::SA;
                island_params.sa_params = sa_params;
                island_params.qlsa_params = make_qlsa_params(options, sa_params);
                island_params.island_count = options.chains;
                island_params.threads = options.parallel == "omp" ? options.threads : 1;
                island_params.migration_interval = options.migration_interval;
                island_params.base_seed = run_seed;
                island_params.time_limit_ms = options.time_limit_ms;
                const tsp::IslandResult result = tsp::run_openmp_islands(
                    dm, island_params, migration_topology_from_options(options));
                if (!tsp::is_valid_tour(result.best_tour, dm.size())) {
                    throw std::runtime_error("island run returned an invalid best tour");
                }
                row = row_from_island_result(algorithm, options, result);
            } else if (options.parallel == "omp" || options.parallel == "cuda" ||
                       options.chains > 1) {
                tsp::ParallelParams parallel_params;
                parallel_params.algorithm = options.use_qlsa ? tsp::AlgorithmKind::QLSA : tsp::AlgorithmKind::SA;
                parallel_params.sa_params = sa_params;
                parallel_params.qlsa_params = make_qlsa_params(options, sa_params);
                parallel_params.chains = options.chains;
                parallel_params.threads = (options.parallel == "omp") ? options.threads : 1;
                parallel_params.cuda_enabled = options.parallel == "cuda";
                parallel_params.cuda_block_size = options.cuda_block_size;
                parallel_params.cuda_mode = (options.cuda_mode == "candidate")
                                                ? tsp::CudaMode::Candidate
                                                : tsp::CudaMode::Chain;
                parallel_params.cuda_candidates_per_iter = options.cuda_candidates_per_iter;
                parallel_params.cuda_reversal_mode = (options.cuda_reversal_mode == "parallel")
                                                        ? tsp::CudaReversalMode::Parallel
                                                        : tsp::CudaReversalMode::Serial;
                if (options.cuda_candidate_policy == "random") {
                    parallel_params.cuda_candidate_policy = tsp::CudaCandidatePolicy::Random;
                } else if (options.cuda_candidate_policy == "hybrid") {
                    parallel_params.cuda_candidate_policy = tsp::CudaCandidatePolicy::Hybrid;
                } else {
                    parallel_params.cuda_candidate_policy = tsp::CudaCandidatePolicy::Best;
                }
                parallel_params.base_seed = run_seed;

                const tsp::ParallelResult result = tsp::run_parallel_chains(dm, parallel_params);
                if (!tsp::is_valid_tour(result.best_tour, dm.size())) {
                    throw std::runtime_error("parallel run returned an invalid best tour");
                }
                row = row_from_parallel_result(algorithm, result);
            } else if (options.use_qlsa) {
                const tsp::QLSAParams qlsa_params = make_qlsa_params(options, sa_params);
                const tsp::QLSAResult result = options.time_limit_ms.has_value()
                                                   ? run_time_limited_qlsa(
                                                         dm, qlsa_params, *options.time_limit_ms)
                                                   : tsp::run_qlsa_2opt(dm, qlsa_params);
                if (!tsp::is_valid_tour(result.best_tour, dm.size())) {
                    throw std::runtime_error("QLSA returned an invalid best tour");
                }
                row = row_from_qlsa_result(algorithm, result);
            } else {
                const tsp::SAResult result = options.time_limit_ms.has_value()
                                                 ? run_time_limited_sa(
                                                       dm, sa_params, *options.time_limit_ms)
                                                 : tsp::run_sa_2opt(dm, sa_params);
                if (!tsp::is_valid_tour(result.best_tour, dm.size())) {
                    throw std::runtime_error("SA returned an invalid best tour");
                }
                row = row_from_sa_result(algorithm, result);
            }

            elapsed_values.push_back(row.elapsed_ms);
            best_lengths.push_back(row.best_length);

            if (!options.csv_only) {
                print_human_run(r + 1, options.repeat, run_seed, options, row);
            }
            print_csv_row(instance, options, run_seed, row);
        }

        if (!options.csv_only && options.repeat > 1) {
            double elapsed_best = elapsed_values.front();
            for (double value : elapsed_values) {
                elapsed_best = std::min(elapsed_best, value);
            }
            std::cout << "Repeat summary: best_length=" << tsp::min_value(best_lengths)
                      << " elapsed_best_ms=" << std::fixed << std::setprecision(3) << elapsed_best
                      << " elapsed_mean_ms=" << tsp::mean(elapsed_values)
                      << " elapsed_std_ms=" << tsp::stddev_population(elapsed_values) << '\n';
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        print_usage(argc > 0 ? argv[0] : "tsp_sa");
        return 1;
    }
}
