#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsp/distance_matrix.hpp"
#include "tsp/cuda.hpp"
#include "tsp/metrics.hpp"
#include "tsp/parallel.hpp"
#include "tsp/qlsa.hpp"
#include "tsp/sa.hpp"
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
        << " --input data/berlin52.tsp --qlsa --parallel cuda --chains 32 --cuda_block_size 128 --iterations 1000000 --seed 1\n";
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
    if (options.parallel == "cuda" && options.use_qlsa && options.qlsa_variant != "current") {
        throw std::invalid_argument("CUDA QLSA currently supports --qlsa_variant current only");
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
};

std::string algorithm_label(const CliOptions& options) {
    std::string base = options.use_qlsa ? "qlsa" : "sa";
    if (options.use_qlsa && options.qlsa_variant != "current") {
        base += "-" + options.qlsa_variant;
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
              << row.improved_moves << '\n';
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
              << " improved=" << row.improved_moves << '\n';
}

RunRow row_from_sa_result(const std::string& algorithm, const tsp::SAResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length;
    row.elapsed_ms = result.elapsed_ms;
    row.accepted_moves = result.accepted_moves;
    row.improved_moves = result.improved_moves;
    return row;
}

RunRow row_from_qlsa_result(const std::string& algorithm, const tsp::QLSAResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length;
    row.elapsed_ms = result.elapsed_ms;
    row.accepted_moves = result.accepted_moves;
    row.improved_moves = result.improved_moves;
    return row;
}

RunRow row_from_parallel_result(const std::string& algorithm, const tsp::ParallelResult& result) {
    RunRow row;
    row.algorithm = algorithm;
    row.best_length = result.best_length;
    row.final_length = result.final_length_of_best_chain;
    row.elapsed_ms = result.elapsed_ms;
    row.accepted_moves = result.total_accepted_moves;
    row.improved_moves = result.total_improved_moves;
    return row;
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
        if (options.parallel == "cuda" && !tsp::cuda_available()) {
            std::cerr << "Warning: CUDA is not available at runtime; falling back to serial multi-chain execution.\n";
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
            std::cout << '\n';
            if (options.use_qlsa) {
                std::cout << "QLSA params: alpha=" << options.qlsa_alpha
                          << " gamma=" << options.qlsa_gamma
                          << " epsilon=" << options.qlsa_epsilon
                          << " policy=" << options.qlsa_policy
                          << " variant=" << options.qlsa_variant
                          << " diversity_threshold=" << options.qlsa_diversity_threshold << '\n';
            }
            std::cout << "CSV: algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves\n";
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

            if (options.parallel == "omp" || options.parallel == "cuda" || options.chains > 1) {
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
                const tsp::QLSAResult result = tsp::run_qlsa_2opt(dm, qlsa_params);
                if (!tsp::is_valid_tour(result.best_tour, dm.size())) {
                    throw std::runtime_error("QLSA returned an invalid best tour");
                }
                row = row_from_qlsa_result(algorithm, result);
            } else {
                const tsp::SAResult result = tsp::run_sa_2opt(dm, sa_params);
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
