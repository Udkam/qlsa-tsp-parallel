#include <mpi.h>

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
#include "tsp/metrics.hpp"
#include "tsp/mpi_parallel.hpp"
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
    std::string parallel = "mpi-omp";
    int chains = 1;
    int threads = 1;
    double qlsa_alpha = 0.1;
    double qlsa_gamma = 0.9;
    double qlsa_epsilon = 0.1;
    std::string qlsa_policy = "epsilon-greedy";
};

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --input data/berlin52.tsp --parallel mpi-omp --chains 32 --threads 8 --iterations 1000000 --seed 1\n"
        << "       mpiexec -n 2 " << program
        << " --qlsa --input data/berlin52.tsp --parallel mpi-omp --chains 32 --threads 4 --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy\n";
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
        } else if (arg == "--alpha") {
            options.qlsa_alpha = std::stod(require_value(arg));
        } else if (arg == "--gamma") {
            options.qlsa_gamma = std::stod(require_value(arg));
        } else if (arg == "--epsilon") {
            options.qlsa_epsilon = std::stod(require_value(arg));
        } else if (arg == "--policy") {
            options.qlsa_policy = require_value(arg);
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
    if (options.parallel != "mpi" && options.parallel != "mpi-omp") {
        throw std::invalid_argument("--parallel must be mpi or mpi-omp for tsp_sa_mpi");
    }
    if (options.chains < 1) {
        throw std::invalid_argument("--chains must be >= 1");
    }
    if (options.threads < 1) {
        throw std::invalid_argument("--threads must be >= 1");
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
    return params;
}

std::string algorithm_label(const CliOptions& options) {
    return options.use_qlsa ? "qlsa-mpi-omp" : "sa-mpi-omp";
}

void print_csv_row(const tsp::Instance& instance,
                   const CliOptions& options,
                   uint64_t seed,
                   const tsp::MpiParallelResult& result) {
    std::cout << algorithm_label(options) << ','
              << csv_escape(instance.name) << ','
              << instance.dimension << ','
              << options.iterations << ','
              << seed << ','
              << options.init << ','
              << options.chains << ','
              << options.threads << ','
              << "mpi-omp" << ','
              << result.global.best_length << ','
              << result.global.final_length_of_best_chain << ','
              << std::fixed << std::setprecision(3) << result.global.elapsed_ms << ','
              << result.global.total_accepted_moves << ','
              << result.global.total_improved_moves << ','
              << result.world_size << ','
              << std::fixed << std::setprecision(3) << result.communication_ms << ','
              << result.global.actual_threads << ','
              << result.global.total_iterations_completed << ','
              << (result.global.deadline_reached ? "true" : "false") << '\n';
}

void print_human_run(int run_index,
                     int repeat,
                     uint64_t seed,
                     const CliOptions& options,
                     const tsp::MpiParallelResult& result) {
    std::cout << "Run " << run_index << '/' << repeat
              << " algorithm=" << algorithm_label(options)
              << " seed=" << seed
              << " ranks=" << result.world_size
              << " chains=" << options.chains
              << " threads_per_rank=" << options.threads
              << " local_chains_rank0=" << result.local_chains
              << " best_length=" << result.global.best_length
              << " final_length=" << result.global.final_length_of_best_chain
              << " elapsed_ms=" << std::fixed << std::setprecision(3) << result.global.elapsed_ms
              << " communication_ms=" << result.communication_ms
              << " accepted=" << result.global.total_accepted_moves
              << " improved=" << result.global.total_improved_moves << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        const CliOptions options = parse_args(argc, argv);
        tsp::Instance instance = tsp::load_tsplib(options.input);
        tsp::DistanceMatrix dm(instance);

        if (rank == 0 && !options.csv_only) {
            std::cout << "Instance: " << instance.name << " (n=" << instance.dimension << ")\n";
            std::cout << "Input: " << std::filesystem::path(options.input).generic_string() << '\n';
            std::cout << "Algorithm: " << algorithm_label(options) << '\n';
            std::cout << "Parallel: mpi-omp chains=" << options.chains
                      << " threads_per_rank=" << options.threads << '\n';
            if (options.use_qlsa) {
                std::cout << "QLSA params: alpha=" << options.qlsa_alpha
                          << " gamma=" << options.qlsa_gamma
                          << " epsilon=" << options.qlsa_epsilon
                          << " policy=" << options.qlsa_policy << '\n';
            }
            std::cout << "CSV: algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves,mpi_ranks,communication_ms,actual_threads,iterations_completed,deadline_reached\n";
        }

        std::vector<double> elapsed_values;
        std::vector<int> best_lengths;
        if (rank == 0) {
            elapsed_values.reserve(static_cast<size_t>(options.repeat));
            best_lengths.reserve(static_cast<size_t>(options.repeat));
        }

        for (int r = 0; r < options.repeat; ++r) {
            const uint64_t run_seed = options.seed + static_cast<uint64_t>(r);
            const tsp::SAParams sa_params = make_sa_params(options, run_seed);

            tsp::ParallelParams parallel_params;
            parallel_params.algorithm = options.use_qlsa ? tsp::AlgorithmKind::QLSA : tsp::AlgorithmKind::SA;
            parallel_params.sa_params = sa_params;
            parallel_params.qlsa_params = make_qlsa_params(options, sa_params);
            parallel_params.chains = options.chains;
            parallel_params.threads = options.threads;
            parallel_params.cuda_enabled = false;
            parallel_params.base_seed = run_seed;

            const tsp::MpiParallelResult result = tsp::run_mpi_parallel_chains(dm, parallel_params);

            if (rank == 0) {
                if (!tsp::is_valid_tour(result.global.best_tour, dm.size())) {
                    throw std::runtime_error("MPI run returned an invalid best tour");
                }
                elapsed_values.push_back(result.global.elapsed_ms);
                best_lengths.push_back(result.global.best_length);

                if (!options.csv_only) {
                    print_human_run(r + 1, options.repeat, run_seed, options, result);
                }
                print_csv_row(instance, options, run_seed, result);
            }
        }

        if (rank == 0 && !options.csv_only && options.repeat > 1) {
            double elapsed_best = elapsed_values.front();
            for (double value : elapsed_values) {
                elapsed_best = std::min(elapsed_best, value);
            }
            std::cout << "Repeat summary: best_length=" << tsp::min_value(best_lengths)
                      << " elapsed_best_ms=" << std::fixed << std::setprecision(3) << elapsed_best
                      << " elapsed_mean_ms=" << tsp::mean(elapsed_values)
                      << " elapsed_std_ms=" << tsp::stddev_population(elapsed_values) << '\n';
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << '\n';
            print_usage(argc > 0 ? argv[0] : "tsp_sa_mpi");
        }
        MPI_Finalize();
        return 1;
    }
}
