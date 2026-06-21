#include "tsp/cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsp/timer.hpp"
#include "tsp/tour.hpp"

namespace tsp {
namespace {

constexpr uint64_t kSeedStride = 0x9E3779B97F4A7C15ULL;
constexpr int kMaxActions = 8;
constexpr int kStateCount = 5;
constexpr int kMaxStateWindow = 64;

struct DeviceAction {
    double min_span_ratio;
    double max_span_ratio;
};

struct DeviceChainSummary {
    int best_length;
    int final_length;
    long long accepted_moves;
    long long improved_moves;
    unsigned long long seed;
};

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__host__ __device__ uint64_t splitmix64_device(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ uint64_t rng_next(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

__device__ double rng_uniform01(uint64_t& state) {
    const uint64_t value = rng_next(state) >> 11;
    return static_cast<double>(value) * (1.0 / 9007199254740992.0);
}

__device__ int rng_uniform_int(uint64_t& state, int low, int high) {
    const uint64_t span = static_cast<uint64_t>(high - low + 1);
    return low + static_cast<int>(rng_next(state) % span);
}

__device__ int dm_dist(const int* dm, int n, int i, int j) {
    return dm[i * n + j];
}

__device__ bool valid_2opt(int n, int i, int k) {
    return n >= 3 && 0 <= i && i < k && k < n && !(i == 0 && k == n - 1);
}

__device__ int tour_length_device(const int* tour, const int* dm, int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += dm_dist(dm, n, tour[i], tour[(i + 1) % n]);
    }
    return total;
}

__device__ int delta_2opt_device(const int* tour, const int* dm, int n, int i, int k) {
    const int a = tour[(i - 1 + n) % n];
    const int b = tour[i];
    const int c = tour[k];
    const int d = tour[(k + 1) % n];
    return dm_dist(dm, n, a, c) + dm_dist(dm, n, b, d) -
           dm_dist(dm, n, a, b) - dm_dist(dm, n, c, d);
}

__device__ void reverse_segment(int* tour, int i, int k) {
    while (i < k) {
        const int tmp = tour[i];
        tour[i] = tour[k];
        tour[k] = tmp;
        ++i;
        --k;
    }
}

__device__ void reverse_segment_parallel(int* tour, int i, int k) {
    const int swaps = (k - i + 1) / 2;
    for (int offset = threadIdx.x; offset < swaps; offset += blockDim.x) {
        const int left = i + offset;
        const int right = k - offset;
        const int tmp = tour[left];
        tour[left] = tour[right];
        tour[right] = tmp;
    }
}

__device__ void copy_tour(int* dst, const int* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

__device__ double* aligned_double_ptr(int* after_ints) {
    uintptr_t address = reinterpret_cast<uintptr_t>(after_ints);
    address = (address + static_cast<uintptr_t>(alignof(double) - 1)) &
              ~static_cast<uintptr_t>(alignof(double) - 1);
    return reinterpret_cast<double*>(address);
}

__device__ void init_identity(int* tour, int n) {
    for (int i = 0; i < n; ++i) {
        tour[i] = i;
    }
}

__device__ void init_random_tour(int* tour, int n, uint64_t& rng_state) {
    init_identity(tour, n);
    for (int i = n - 1; i > 0; --i) {
        const int j = rng_uniform_int(rng_state, 0, i);
        const int tmp = tour[i];
        tour[i] = tour[j];
        tour[j] = tmp;
    }
}

__device__ void init_nearest_neighbor(int* tour, int* used, const int* dm, int n) {
    for (int i = 0; i < n; ++i) {
        used[i] = 0;
    }
    int current = 0;
    tour[0] = current;
    used[current] = 1;
    for (int step = 1; step < n; ++step) {
        int best_city = -1;
        int best_dist = INT_MAX;
        for (int city = 0; city < n; ++city) {
            if (used[city] != 0) {
                continue;
            }
            const int d = dm_dist(dm, n, current, city);
            if (d < best_dist || (d == best_dist && (best_city < 0 || city < best_city))) {
                best_dist = d;
                best_city = city;
            }
        }
        current = best_city;
        tour[step] = current;
        used[current] = 1;
    }
}

__device__ int state_from_average_delta(double average_delta, double delta_scale) {
    if (average_delta <= -delta_scale) {
        return 0;
    }
    if (average_delta < 0.0) {
        return 1;
    }
    if (average_delta == 0.0) {
        return 2;
    }
    if (average_delta < delta_scale) {
        return 3;
    }
    return 4;
}

__device__ double reward_from_delta(int delta, bool accepted) {
    if (accepted) {
        return static_cast<double>(-delta);
    }
    return delta > 0 ? -0.1 * static_cast<double>(delta) : 0.0;
}

__device__ int best_action(const double* q_row, int action_count) {
    int best = 0;
    for (int i = 1; i < action_count; ++i) {
        if (q_row[i] > q_row[best]) {
            best = i;
        }
    }
    return best;
}

__device__ int select_action(const double* q_table,
                             int state,
                             int action_count,
                             int policy_softmax,
                             double epsilon,
                             double softmax_temperature,
                             uint64_t& rng_state) {
    const double* row = q_table + state * action_count;
    if (policy_softmax == 0) {
        if (rng_uniform01(rng_state) < epsilon) {
            return rng_uniform_int(rng_state, 0, action_count - 1);
        }
        return best_action(row, action_count);
    }

    double max_q = row[0];
    for (int i = 1; i < action_count; ++i) {
        max_q = fmax(max_q, row[i]);
    }
    double total = 0.0;
    double weights[kMaxActions];
    for (int i = 0; i < action_count; ++i) {
        weights[i] = exp((row[i] - max_q) / softmax_temperature);
        total += weights[i];
    }
    if (!(total > 0.0) || !isfinite(total)) {
        return best_action(row, action_count);
    }
    const double draw = rng_uniform01(rng_state) * total;
    double cumulative = 0.0;
    for (int i = 0; i < action_count; ++i) {
        cumulative += weights[i];
        if (draw <= cumulative) {
            return i;
        }
    }
    return action_count - 1;
}

__device__ void update_q(double* q_table,
                         int action_count,
                         int state,
                         int action,
                         int next_state,
                         double reward,
                         double alpha,
                         double gamma) {
    double max_next = q_table[next_state * action_count];
    for (int i = 1; i < action_count; ++i) {
        max_next = fmax(max_next, q_table[next_state * action_count + i]);
    }
    double& old_value = q_table[state * action_count + action];
    const double target = reward + gamma * max_next;
    old_value += alpha * (target - old_value);
}

__device__ void sample_random_move(int n, int& out_i, int& out_k, uint64_t& rng_state) {
    do {
        int i = rng_uniform_int(rng_state, 0, n - 1);
        int k = rng_uniform_int(rng_state, 0, n - 1);
        if (i > k) {
            const int tmp = i;
            i = k;
            k = tmp;
        }
        out_i = i;
        out_k = k;
    } while (!valid_2opt(n, out_i, out_k));
}

__device__ int span_bound(double ratio, int n, bool upper) {
    const double scaled = ratio * static_cast<double>(n);
    const int value = upper ? static_cast<int>(ceil(scaled)) : static_cast<int>(floor(scaled));
    return min(max(value, 2), n - 1);
}

__device__ void sample_action_move(int n,
                                   DeviceAction action,
                                   int& out_i,
                                   int& out_k,
                                   uint64_t& rng_state) {
    int low = span_bound(action.min_span_ratio, n, false);
    int high = span_bound(action.max_span_ratio, n, true);
    if (low > high) {
        low = 2;
        high = n - 1;
    }
    for (int attempt = 0; attempt < 16; ++attempt) {
        const int span = rng_uniform_int(rng_state, low, high);
        const int i = rng_uniform_int(rng_state, 0, n - span);
        const int k = i + span - 1;
        if (valid_2opt(n, i, k)) {
            out_i = i;
            out_k = k;
            return;
        }
    }
    sample_random_move(n, out_i, out_k, rng_state);
}

__global__ void sa_kernel(const int* dm,
                          int n,
                          int chains,
                          long long iterations,
                          double initial_temperature,
                          double final_temperature,
                          unsigned long long base_seed,
                          int use_nearest_neighbor_init,
                          int* best_tours,
                          DeviceChainSummary* summaries) {
    const int chain_id = blockIdx.x;
    if (chain_id >= chains) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    int* current = reinterpret_cast<int*>(shared_bytes);
    int* best = current + n;
    int* used = best + n;

    if (threadIdx.x == 0) {
        uint64_t rng_state = splitmix64_device(base_seed + kSeedStride * static_cast<uint64_t>(chain_id + 1));
        const uint64_t chain_seed_value = rng_state;

        if (use_nearest_neighbor_init != 0) {
            init_nearest_neighbor(current, used, dm, n);
        } else {
            init_random_tour(current, n, rng_state);
        }
        int current_length = tour_length_device(current, dm, n);
        int best_length = current_length;
        copy_tour(best, current, n);

        double temperature = initial_temperature;
        const double temp_decay = pow(final_temperature / initial_temperature,
                                      1.0 / static_cast<double>(iterations));
        long long accepted = 0;
        long long improved = 0;

        for (long long iter = 0; n >= 3 && iter < iterations; ++iter) {
            int i = 0;
            int k = 1;
            sample_random_move(n, i, k, rng_state);
            const int delta = delta_2opt_device(current, dm, n, i, k);
            bool accept = delta < 0;
            if (!accept) {
                accept = rng_uniform01(rng_state) < exp(-static_cast<double>(delta) / temperature);
            }
            if (accept) {
                reverse_segment(current, i, k);
                current_length += delta;
                ++accepted;
                if (delta < 0) {
                    ++improved;
                }
                if (current_length < best_length) {
                    best_length = current_length;
                    copy_tour(best, current, n);
                }
            }
            temperature *= temp_decay;
        }

        for (int city = 0; city < n; ++city) {
            best_tours[chain_id * n + city] = best[city];
        }
        summaries[chain_id].best_length = best_length;
        summaries[chain_id].final_length = current_length;
        summaries[chain_id].accepted_moves = accepted;
        summaries[chain_id].improved_moves = improved;
        summaries[chain_id].seed = chain_seed_value;
    }
}

__global__ void sa_candidate_kernel(const int* dm,
                                    int n,
                                    int chains,
                                    long long iterations,
                                    double initial_temperature,
                                    double final_temperature,
                                    unsigned long long base_seed,
                                    int use_nearest_neighbor_init,
                                    int candidates_per_iter,
                                    int parallel_reversal,
                                    int* best_tours,
                                    DeviceChainSummary* summaries) {
    const int chain_id = blockIdx.x;
    if (chain_id >= chains) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    int* current = reinterpret_cast<int*>(shared_bytes);
    int* best = current + n;
    int* used = best + n;
    int* candidate_delta = used + n;
    int* candidate_i = candidate_delta + blockDim.x;
    int* candidate_k = candidate_i + blockDim.x;

    __shared__ int shared_current_length;
    __shared__ int shared_best_length;
    __shared__ long long shared_accepted;
    __shared__ long long shared_improved;
    __shared__ unsigned long long shared_seed_value;
    __shared__ double shared_temperature;
    __shared__ int shared_accept;
    __shared__ int shared_selected_i;
    __shared__ int shared_selected_k;
    __shared__ int shared_selected_delta;

    if (threadIdx.x == 0) {
        uint64_t rng_state = splitmix64_device(base_seed + kSeedStride * static_cast<uint64_t>(chain_id + 1));
        shared_seed_value = rng_state;

        if (use_nearest_neighbor_init != 0) {
            init_nearest_neighbor(current, used, dm, n);
        } else {
            init_random_tour(current, n, rng_state);
        }
        shared_current_length = tour_length_device(current, dm, n);
        shared_best_length = shared_current_length;
        shared_accepted = 0;
        shared_improved = 0;
        shared_temperature = initial_temperature;
        copy_tour(best, current, n);
    }
    __syncthreads();

    const double temp_decay = pow(final_temperature / initial_temperature,
                                  1.0 / static_cast<double>(iterations));

    for (long long iter = 0; n >= 3 && iter < iterations; ++iter) {
        const int tid = threadIdx.x;
        if (tid < candidates_per_iter) {
            uint64_t local_rng = splitmix64_device(
                base_seed ^
                (kSeedStride * static_cast<uint64_t>(chain_id + 1)) ^
                (0xBF58476D1CE4E5B9ULL * static_cast<uint64_t>(iter + 1)) ^
                (0x94D049BB133111EBULL * static_cast<uint64_t>(tid + 1)));
            int i = 0;
            int k = 1;
            sample_random_move(n, i, k, local_rng);
            candidate_i[tid] = i;
            candidate_k[tid] = k;
            candidate_delta[tid] = delta_2opt_device(current, dm, n, i, k);
        } else {
            candidate_i[tid] = 0;
            candidate_k[tid] = 1;
            candidate_delta[tid] = INT_MAX;
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride <<= 1) {
            const int period = stride << 1;
            if ((tid % period) == 0) {
                const int other = tid + stride;
                if (other < blockDim.x) {
                    const int other_delta = candidate_delta[other];
                    const int self_delta = candidate_delta[tid];
                    if (other_delta < self_delta ||
                        (other_delta == self_delta && candidate_i[other] < candidate_i[tid])) {
                        candidate_delta[tid] = other_delta;
                        candidate_i[tid] = candidate_i[other];
                        candidate_k[tid] = candidate_k[other];
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int i = candidate_i[0];
            const int k = candidate_k[0];
            const int delta = candidate_delta[0];
            uint64_t accept_rng = splitmix64_device(
                base_seed +
                kSeedStride * static_cast<uint64_t>(chain_id + 1) +
                0xD1B54A32D192ED03ULL * static_cast<uint64_t>(iter + 1));
            bool accept = delta < 0;
            if (!accept) {
                accept = rng_uniform01(accept_rng) < exp(-static_cast<double>(delta) / shared_temperature);
            }
            shared_accept = accept ? 1 : 0;
            shared_selected_i = i;
            shared_selected_k = k;
            shared_selected_delta = delta;
        }
        __syncthreads();

        if (shared_accept != 0) {
            if (parallel_reversal != 0) {
                reverse_segment_parallel(current, shared_selected_i, shared_selected_k);
            } else if (threadIdx.x == 0) {
                reverse_segment(current, shared_selected_i, shared_selected_k);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const int delta = shared_selected_delta;
            if (shared_accept != 0) {
                shared_current_length += delta;
                ++shared_accepted;
                if (delta < 0) {
                    ++shared_improved;
                }
                if (shared_current_length < shared_best_length) {
                    shared_best_length = shared_current_length;
                    copy_tour(best, current, n);
                }
            }
            shared_temperature *= temp_decay;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int city = 0; city < n; ++city) {
            best_tours[chain_id * n + city] = best[city];
        }
        summaries[chain_id].best_length = shared_best_length;
        summaries[chain_id].final_length = shared_current_length;
        summaries[chain_id].accepted_moves = shared_accepted;
        summaries[chain_id].improved_moves = shared_improved;
        summaries[chain_id].seed = shared_seed_value;
    }
}

__global__ void qlsa_kernel(const int* dm,
                            int n,
                            int chains,
                            long long iterations,
                            double initial_temperature,
                            double final_temperature,
                            unsigned long long base_seed,
                            int use_nearest_neighbor_init,
                            double alpha,
                            double gamma,
                            double epsilon,
                            int policy_softmax,
                            double softmax_temperature,
                            int state_window,
                            double delta_scale,
                            const DeviceAction* actions,
                            int action_count,
                            int* best_tours,
                            DeviceChainSummary* summaries) {
    const int chain_id = blockIdx.x;
    if (chain_id >= chains) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    int* current = reinterpret_cast<int*>(shared_bytes);
    int* best = current + n;
    int* used = best + n;
    double* q_table = aligned_double_ptr(used + n);

    if (threadIdx.x == 0) {
        uint64_t rng_state = splitmix64_device(base_seed + kSeedStride * static_cast<uint64_t>(chain_id + 1));
        const uint64_t chain_seed_value = rng_state;

        for (int i = 0; i < kStateCount * action_count; ++i) {
            q_table[i] = 0.0;
        }

        if (use_nearest_neighbor_init != 0) {
            init_nearest_neighbor(current, used, dm, n);
        } else {
            init_random_tour(current, n, rng_state);
        }
        int current_length = tour_length_device(current, dm, n);
        int best_length = current_length;
        copy_tour(best, current, n);

        double temperature = initial_temperature;
        const double temp_decay = pow(final_temperature / initial_temperature,
                                      1.0 / static_cast<double>(iterations));
        long long accepted = 0;
        long long improved = 0;
        int recent[kMaxStateWindow];
        int recent_count = 0;
        int recent_pos = 0;
        double recent_sum = 0.0;
        const int window = max(1, min(state_window, kMaxStateWindow));
        int state = state_from_average_delta(0.0, delta_scale);

        for (long long iter = 0; n >= 3 && iter < iterations; ++iter) {
            const int action_index = select_action(q_table, state, action_count, policy_softmax,
                                                   epsilon, softmax_temperature, rng_state);
            int i = 0;
            int k = 1;
            sample_action_move(n, actions[action_index], i, k, rng_state);
            const int delta = delta_2opt_device(current, dm, n, i, k);
            bool accept = delta < 0;
            if (!accept) {
                accept = rng_uniform01(rng_state) < exp(-static_cast<double>(delta) / temperature);
            }
            if (accept) {
                reverse_segment(current, i, k);
                current_length += delta;
                ++accepted;
                if (delta < 0) {
                    ++improved;
                }
                if (current_length < best_length) {
                    best_length = current_length;
                    copy_tour(best, current, n);
                }
            }

            if (recent_count < window) {
                recent[recent_count++] = delta;
                recent_sum += static_cast<double>(delta);
            } else {
                recent_sum -= static_cast<double>(recent[recent_pos]);
                recent[recent_pos] = delta;
                recent_sum += static_cast<double>(delta);
                recent_pos = (recent_pos + 1) % window;
            }
            const double average_delta = recent_sum / static_cast<double>(recent_count);
            const int next_state = state_from_average_delta(average_delta, delta_scale);
            const double reward = reward_from_delta(delta, accept);
            update_q(q_table, action_count, state, action_index, next_state, reward, alpha, gamma);
            state = next_state;

            temperature *= temp_decay;
        }

        for (int city = 0; city < n; ++city) {
            best_tours[chain_id * n + city] = best[city];
        }
        summaries[chain_id].best_length = best_length;
        summaries[chain_id].final_length = current_length;
        summaries[chain_id].accepted_moves = accepted;
        summaries[chain_id].improved_moves = improved;
        summaries[chain_id].seed = chain_seed_value;
    }
}

__global__ void qlsa_candidate_kernel(const int* dm,
                                      int n,
                                      int chains,
                                      long long iterations,
                                      double initial_temperature,
                                      double final_temperature,
                                      unsigned long long base_seed,
                                      int use_nearest_neighbor_init,
                                      double alpha,
                                      double gamma,
                                      double epsilon,
                                      int policy_softmax,
                                      double softmax_temperature,
                                      int state_window,
                                      double delta_scale,
                                      const DeviceAction* actions,
                                      int action_count,
                                      int candidates_per_iter,
                                      int parallel_reversal,
                                      int* best_tours,
                                      DeviceChainSummary* summaries) {
    const int chain_id = blockIdx.x;
    if (chain_id >= chains) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    int* current = reinterpret_cast<int*>(shared_bytes);
    int* best = current + n;
    int* used = best + n;
    double* q_table = aligned_double_ptr(used + n);
    int* candidate_delta = reinterpret_cast<int*>(q_table + kStateCount * action_count);
    int* candidate_i = candidate_delta + blockDim.x;
    int* candidate_k = candidate_i + blockDim.x;

    __shared__ int shared_current_length;
    __shared__ int shared_best_length;
    __shared__ long long shared_accepted;
    __shared__ long long shared_improved;
    __shared__ unsigned long long shared_seed_value;
    __shared__ double shared_temperature;
    __shared__ int shared_action_index;
    __shared__ int shared_state;
    __shared__ int shared_next_state;
    __shared__ int shared_accept;
    __shared__ int shared_selected_i;
    __shared__ int shared_selected_k;
    __shared__ int shared_selected_delta;
    __shared__ double shared_reward;

    if (threadIdx.x == 0) {
        uint64_t rng_state = splitmix64_device(base_seed + kSeedStride * static_cast<uint64_t>(chain_id + 1));
        shared_seed_value = rng_state;

        for (int i = 0; i < kStateCount * action_count; ++i) {
            q_table[i] = 0.0;
        }

        if (use_nearest_neighbor_init != 0) {
            init_nearest_neighbor(current, used, dm, n);
        } else {
            init_random_tour(current, n, rng_state);
        }
        shared_current_length = tour_length_device(current, dm, n);
        shared_best_length = shared_current_length;
        shared_accepted = 0;
        shared_improved = 0;
        shared_temperature = initial_temperature;
        shared_state = state_from_average_delta(0.0, delta_scale);
        copy_tour(best, current, n);
    }
    __syncthreads();

    const double temp_decay = pow(final_temperature / initial_temperature,
                                  1.0 / static_cast<double>(iterations));

    int recent[kMaxStateWindow];
    int recent_count = 0;
    int recent_pos = 0;
    double recent_sum = 0.0;
    const int window = max(1, min(state_window, kMaxStateWindow));
    uint64_t action_rng = splitmix64_device(
        base_seed + kSeedStride * static_cast<uint64_t>(chain_id + 1) +
        0xD1B54A32D192ED03ULL);

    for (long long iter = 0; n >= 3 && iter < iterations; ++iter) {
        if (threadIdx.x == 0) {
            shared_action_index = select_action(q_table, shared_state, action_count, policy_softmax,
                                                epsilon, softmax_temperature, action_rng);
        }
        __syncthreads();

        const int tid = threadIdx.x;
        if (tid < candidates_per_iter) {
            uint64_t local_rng = splitmix64_device(
                base_seed ^
                (kSeedStride * static_cast<uint64_t>(chain_id + 1)) ^
                (0xBF58476D1CE4E5B9ULL * static_cast<uint64_t>(iter + 1)) ^
                (0x94D049BB133111EBULL * static_cast<uint64_t>(tid + 1)) ^
                (0xD1B54A32D192ED03ULL * static_cast<uint64_t>(shared_action_index + 1)));
            int i = 0;
            int k = 1;
            sample_action_move(n, actions[shared_action_index], i, k, local_rng);
            candidate_i[tid] = i;
            candidate_k[tid] = k;
            candidate_delta[tid] = delta_2opt_device(current, dm, n, i, k);
        } else {
            candidate_i[tid] = 0;
            candidate_k[tid] = 1;
            candidate_delta[tid] = INT_MAX;
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride <<= 1) {
            const int period = stride << 1;
            if ((tid % period) == 0) {
                const int other = tid + stride;
                if (other < blockDim.x) {
                    const int other_delta = candidate_delta[other];
                    const int self_delta = candidate_delta[tid];
                    if (other_delta < self_delta ||
                        (other_delta == self_delta && candidate_i[other] < candidate_i[tid])) {
                        candidate_delta[tid] = other_delta;
                        candidate_i[tid] = candidate_i[other];
                        candidate_k[tid] = candidate_k[other];
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int delta = candidate_delta[0];
            uint64_t accept_rng = splitmix64_device(
                base_seed +
                kSeedStride * static_cast<uint64_t>(chain_id + 1) +
                0x94D049BB133111EBULL * static_cast<uint64_t>(iter + 1));
            bool accept = delta < 0;
            if (!accept) {
                accept = rng_uniform01(accept_rng) < exp(-static_cast<double>(delta) / shared_temperature);
            }
            shared_accept = accept ? 1 : 0;
            shared_selected_i = candidate_i[0];
            shared_selected_k = candidate_k[0];
            shared_selected_delta = delta;
            shared_reward = reward_from_delta(delta, accept);
        }
        __syncthreads();

        if (shared_accept != 0) {
            if (parallel_reversal != 0) {
                reverse_segment_parallel(current, shared_selected_i, shared_selected_k);
            } else if (threadIdx.x == 0) {
                reverse_segment(current, shared_selected_i, shared_selected_k);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const int delta = shared_selected_delta;
            if (shared_accept != 0) {
                shared_current_length += delta;
                ++shared_accepted;
                if (delta < 0) {
                    ++shared_improved;
                }
                if (shared_current_length < shared_best_length) {
                    shared_best_length = shared_current_length;
                    copy_tour(best, current, n);
                }
            }

            if (recent_count < window) {
                recent[recent_count++] = delta;
                recent_sum += static_cast<double>(delta);
            } else {
                recent_sum -= static_cast<double>(recent[recent_pos]);
                recent[recent_pos] = delta;
                recent_sum += static_cast<double>(delta);
                recent_pos = (recent_pos + 1) % window;
            }
            const double average_delta = recent_sum / static_cast<double>(recent_count);
            shared_next_state = state_from_average_delta(average_delta, delta_scale);
            update_q(q_table, action_count, shared_state, shared_action_index,
                     shared_next_state, shared_reward, alpha, gamma);
            shared_state = shared_next_state;
            shared_temperature *= temp_decay;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int city = 0; city < n; ++city) {
            best_tours[chain_id * n + city] = best[city];
        }
        summaries[chain_id].best_length = shared_best_length;
        summaries[chain_id].final_length = shared_current_length;
        summaries[chain_id].accepted_moves = shared_accepted;
        summaries[chain_id].improved_moves = shared_improved;
        summaries[chain_id].seed = shared_seed_value;
    }
}

std::vector<DeviceAction> make_device_actions(const QLSAParams& params) {
    const std::vector<QLSAAction> source = params.actions.empty() ? default_qlsa_actions() : params.actions;
    if (source.empty() || source.size() > kMaxActions) {
        throw std::invalid_argument("CUDA QLSA supports 1..8 actions");
    }
    std::vector<DeviceAction> actions;
    actions.reserve(source.size());
    for (const QLSAAction& action : source) {
        actions.push_back(DeviceAction{action.min_span_ratio, action.max_span_ratio});
    }
    return actions;
}

void validate_cuda_params(const DistanceMatrix& dm, const ParallelParams& params) {
    if (dm.size() <= 0) {
        throw std::invalid_argument("CUDA run requires a non-empty distance matrix");
    }
    if (params.chains < 1) {
        throw std::invalid_argument("chains must be >= 1");
    }
    if (params.cuda_block_size < 1 || params.cuda_block_size > 1024) {
        throw std::invalid_argument("cuda_block_size must be in [1, 1024]");
    }
    if (params.cuda_candidates_per_iter <= 0) {
        throw std::invalid_argument("cuda_candidates_per_iter must be positive");
    }
    if (params.cuda_candidates_per_iter > params.cuda_block_size) {
        throw std::invalid_argument("cuda_candidates_per_iter must be <= cuda_block_size");
    }
    const long long iterations = (params.algorithm == AlgorithmKind::SA)
                                     ? params.sa_params.iterations
                                     : params.qlsa_params.sa.iterations;
    if (iterations < 1) {
        throw std::invalid_argument("iterations must be >= 1");
    }
}

ParallelResult build_parallel_result(const DistanceMatrix& dm,
                                     const ParallelParams& params,
                                     const std::vector<int>& best_tours,
                                     const std::vector<DeviceChainSummary>& summaries,
                                     double elapsed_ms) {
    ParallelResult result;
    result.chains = params.chains;
    result.threads = params.cuda_block_size;
    result.base_seed = params.base_seed;
    result.elapsed_ms = elapsed_ms;
    result.chain_results.resize(static_cast<size_t>(params.chains));

    int best_index = -1;
    int best_length = std::numeric_limits<int>::max();
    const int n = dm.size();
    for (int chain_id = 0; chain_id < params.chains; ++chain_id) {
        ChainResult& chain = result.chain_results[static_cast<size_t>(chain_id)];
        chain.chain_id = chain_id;
        chain.seed = summaries[static_cast<size_t>(chain_id)].seed;
        chain.best_tour.assign(best_tours.begin() + static_cast<size_t>(chain_id) * n,
                               best_tours.begin() + static_cast<size_t>(chain_id + 1) * n);
        chain.best_length = summaries[static_cast<size_t>(chain_id)].best_length;
        chain.final_length = summaries[static_cast<size_t>(chain_id)].final_length;
        chain.elapsed_ms = 0.0;
        chain.accepted_moves = summaries[static_cast<size_t>(chain_id)].accepted_moves;
        chain.improved_moves = summaries[static_cast<size_t>(chain_id)].improved_moves;

        result.total_accepted_moves += chain.accepted_moves;
        result.total_improved_moves += chain.improved_moves;
        if (!is_valid_tour(chain.best_tour, n)) {
            throw std::runtime_error("CUDA chain returned an invalid best tour");
        }
        if (tour_length(chain.best_tour, dm) != chain.best_length) {
            throw std::runtime_error("CUDA chain best_length verification failed");
        }
        if (chain.best_length < best_length) {
            best_length = chain.best_length;
            best_index = chain_id;
        }
    }

    if (best_index < 0) {
        throw std::runtime_error("CUDA run produced no chain results");
    }
    const ChainResult& best_chain = result.chain_results[static_cast<size_t>(best_index)];
    result.best_tour = best_chain.best_tour;
    result.best_length = best_chain.best_length;
    result.final_length_of_best_chain = best_chain.final_length;

    const int checked = tour_length(result.best_tour, dm);
    if (checked != result.best_length) {
        throw std::runtime_error("CUDA best_length verification failed");
    }
    return result;
}

}  // namespace

bool cuda_available_impl() noexcept {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
}

ParallelResult run_cuda_chains_impl(const DistanceMatrix& dm, const ParallelParams& params) {
    validate_cuda_params(dm, params);
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("CUDA requested but no CUDA device is available");
    }

    const int n = dm.size();
    const long long iterations = (params.algorithm == AlgorithmKind::SA)
                                     ? params.sa_params.iterations
                                     : params.qlsa_params.sa.iterations;
    const double initial_temperature = (params.algorithm == AlgorithmKind::SA)
                                           ? params.sa_params.initial_temperature
                                           : params.qlsa_params.sa.initial_temperature;
    const double final_temperature = (params.algorithm == AlgorithmKind::SA)
                                         ? params.sa_params.final_temperature
                                         : params.qlsa_params.sa.final_temperature;
    const int use_nn = (params.algorithm == AlgorithmKind::SA)
                           ? (params.sa_params.use_nearest_neighbor_init ? 1 : 0)
                           : (params.qlsa_params.sa.use_nearest_neighbor_init ? 1 : 0);

    int max_shared = 0;
    check_cuda(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, 0),
               "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlock)");

    std::vector<DeviceAction> host_actions;
    DeviceAction* device_actions = nullptr;
    int action_count = 0;
    size_t shared_bytes = static_cast<size_t>(3 * n) * sizeof(int);
    if (params.algorithm == AlgorithmKind::QLSA) {
        host_actions = make_device_actions(params.qlsa_params);
        action_count = static_cast<int>(host_actions.size());
        shared_bytes += static_cast<size_t>(alignof(double)) +
                        static_cast<size_t>(kStateCount * action_count) * sizeof(double);
        check_cuda(cudaMalloc(&device_actions, host_actions.size() * sizeof(DeviceAction)),
                   "cudaMalloc(actions)");
        check_cuda(cudaMemcpy(device_actions, host_actions.data(),
                              host_actions.size() * sizeof(DeviceAction),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(actions)");
    }
    if (params.cuda_mode == CudaMode::Candidate) {
        shared_bytes += static_cast<size_t>(3 * params.cuda_block_size) * sizeof(int);
    }
    if (shared_bytes > static_cast<size_t>(max_shared)) {
        if (device_actions != nullptr) {
            cudaFree(device_actions);
        }
        throw std::runtime_error("CUDA shared memory requirement exceeds device limit");
    }

    int* device_dm = nullptr;
    int* device_best_tours = nullptr;
    DeviceChainSummary* device_summaries = nullptr;
    const size_t dm_bytes = dm.raw().size() * sizeof(int);
    const size_t tours_bytes = static_cast<size_t>(params.chains) * n * sizeof(int);
    const size_t summaries_bytes = static_cast<size_t>(params.chains) * sizeof(DeviceChainSummary);

    check_cuda(cudaMalloc(&device_dm, dm_bytes), "cudaMalloc(distance matrix)");
    check_cuda(cudaMalloc(&device_best_tours, tours_bytes), "cudaMalloc(best tours)");
    check_cuda(cudaMalloc(&device_summaries, summaries_bytes), "cudaMalloc(chain summaries)");
    check_cuda(cudaMemcpy(device_dm, dm.raw().data(), dm_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(distance matrix)");

    Timer timer;
    const int parallel_reversal = params.cuda_reversal_mode == CudaReversalMode::Parallel ? 1 : 0;
    if (params.algorithm == AlgorithmKind::SA) {
        if (params.cuda_mode == CudaMode::Candidate) {
            sa_candidate_kernel<<<params.chains, params.cuda_block_size, shared_bytes>>>(
                device_dm, n, params.chains, iterations, initial_temperature, final_temperature,
                params.base_seed, use_nn, params.cuda_candidates_per_iter, parallel_reversal,
                device_best_tours, device_summaries);
        } else {
            sa_kernel<<<params.chains, params.cuda_block_size, shared_bytes>>>(
                device_dm, n, params.chains, iterations, initial_temperature, final_temperature,
                params.base_seed, use_nn, device_best_tours, device_summaries);
        }
    } else {
        const QLSAParams& qlsa = params.qlsa_params;
        if (params.cuda_mode == CudaMode::Candidate) {
            qlsa_candidate_kernel<<<params.chains, params.cuda_block_size, shared_bytes>>>(
                device_dm, n, params.chains, iterations, initial_temperature, final_temperature,
                params.base_seed, use_nn, qlsa.alpha, qlsa.gamma, qlsa.epsilon,
                qlsa.policy == "softmax" ? 1 : 0, qlsa.softmax_temperature,
                qlsa.state_window, qlsa.delta_scale, device_actions, action_count,
                params.cuda_candidates_per_iter, parallel_reversal,
                device_best_tours, device_summaries);
        } else {
            qlsa_kernel<<<params.chains, params.cuda_block_size, shared_bytes>>>(
                device_dm, n, params.chains, iterations, initial_temperature, final_temperature,
                params.base_seed, use_nn, qlsa.alpha, qlsa.gamma, qlsa.epsilon,
                qlsa.policy == "softmax" ? 1 : 0, qlsa.softmax_temperature,
                qlsa.state_window, qlsa.delta_scale, device_actions, action_count,
                device_best_tours, device_summaries);
        }
    }
    check_cuda(cudaGetLastError(), "CUDA kernel launch");
    check_cuda(cudaDeviceSynchronize(), "CUDA kernel execution");
    const double elapsed_ms = timer.elapsed_ms();

    std::vector<int> host_best_tours(static_cast<size_t>(params.chains) * n);
    std::vector<DeviceChainSummary> host_summaries(static_cast<size_t>(params.chains));
    check_cuda(cudaMemcpy(host_best_tours.data(), device_best_tours, tours_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(best tours)");
    check_cuda(cudaMemcpy(host_summaries.data(), device_summaries, summaries_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(chain summaries)");

    cudaFree(device_dm);
    cudaFree(device_best_tours);
    cudaFree(device_summaries);
    if (device_actions != nullptr) {
        cudaFree(device_actions);
    }

    return build_parallel_result(dm, params, host_best_tours, host_summaries, elapsed_ms);
}

}  // namespace tsp
