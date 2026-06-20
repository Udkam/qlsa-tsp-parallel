# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化

## 0. 基本信息

| 项目 | 内容 |
|---|---|
| 课程名称 | 并行算法 |
| 项目题目 | 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化 |
| 团队人数 | 1 人 |
| 团队成员 | 陈乐浚 |
| 学号 | 22361054 |
| 学院/专业 | 中山大学计算机学院 / 信息与计算科学 |

## 1. 摘要

本项目围绕旅行推销员问题（Traveling Salesman Problem, TSP）展开，参考近期论文 “Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization” 中的算法思想，在 C++20 中实现了模拟退火（Simulated Annealing, SA）与 Q-Learning 辅助模拟退火（QLSA），并进一步实现了 OpenMP 多搜索链并行与 CUDA 多搜索链并行工程扩展。项目使用 TSPLIB95 标准实例作为实验数据，围绕路径质量、运行时间、加速比、并行效率等指标进行了多阶段实验。

工程实现方面，项目从 TSPLIB95 解析器、连续一维距离矩阵、路径表示、2-opt O(1) 增量计算、串行 SA/QLSA 基线开始，逐步扩展到 OpenMP 和 CUDA 后端。OpenMP 版本采用 chain-level parallelism，即每条搜索链独立维护路径、随机数状态和 Q 表，最后在主线程归约全局最优解。CUDA 版本也沿用多链抽象，将每条搜索链映射到 GPU block，并完成了 Ninja/CUDA 构建与运行验证。

实验结果表明，在默认参数多实例实验中，OpenMP 多链并行在 `berlin52`、`eil51`、`st70`、`eil76`、`rat99`、`eil101` 上取得稳定加速。SA OpenMP 平均加速约为 5.46x，QLSA OpenMP 平均加速约为 4.98x。进一步的调参和定向增强实验提高了较难实例上的解质量：在 `eil101` 上，SA 与 QLSA 均在定向增强实验中达到 BKS=629；在 `rat99` 上，QLSA high-budget 配置达到 BKS=1211，而 SA high-budget 最好结果为 1212，说明 QLSA 在该实例上表现出明确的解质量优势。CUDA 后端已完成并验证，但在当前小规模 TSPLIB 实例上不作为主要性能结论，最终报告中将其定位为工程扩展与后续优化方向。

## 2. 选题背景与参考论文

旅行推销员问题是组合优化领域的经典 NP-hard 问题。给定一组城市及城市间距离，TSP 要求找到一条访问每个城市一次并回到起点的最短回路。由于精确算法在规模增大时计算代价迅速上升，模拟退火、遗传算法、蚁群算法、局部搜索与强化学习辅助启发式算法常被用于求解较大规模实例的近似最优解。

模拟退火算法适合 TSP 这类组合优化问题。它通过随机扰动当前解生成邻域解，并用 Metropolis 接受准则在一定概率下接受较差解，从而在搜索早期保留跳出局部最优的能力，在温度逐渐下降后趋向稳定收敛。参考论文提出的 Q-Learning Assisted Simulated Annealing 进一步引入 Q-learning，用学习到的 Q 值辅助选择邻域搜索策略，使搜索过程能够根据近期搜索状态调整动作选择。

本项目并非只做脚本级复现，而是在论文算法思想基础上进行 C++ 底层实现和并行化优化。项目实现了 TSPLIB95 解析、路径和距离矩阵基础模块、SA/QLSA 核心算法、OpenMP 多链并行、CUDA 多链并行，以及可复现实验脚本和结果统计脚本。课程大作业要求鼓励复现近期论文算法或结合实际应用进行并行化，本项目将 QLSA 思想迁移到 C++/OpenMP/CUDA 工程实现中，并相对于自实现串行多链基线进行了加速比和并行效率分析。

需要说明的是，本项目没有声称全面超过参考论文的所有性能结果。由于硬件、实现语言、参数设置、随机种子和实验实例选择均可能不同，本文主要报告自实现串行多链基线、OpenMP 版本、CUDA 版本之间的公平对比，以及参数调优后在 TSPLIB95 实例上的解质量变化。

## 3. 算法原理

### 3.1 TSP 问题定义

给定 `n` 个城市和距离矩阵 `D`，其中 `D[i][j]` 表示城市 `i` 到城市 `j` 的距离。TSP 的目标是寻找一个城市排列：

```text
tour = (v0, v1, ..., v(n-1))
```

使得每个城市恰好出现一次，并最小化闭合回路长度：

```text
L(tour) = D[v0][v1] + D[v1][v2] + ... + D[v(n-1)][v0]
```

在本项目中，路径使用 `std::vector<int>` 表示城市排列，距离矩阵使用连续一维数组存储，路径长度计算和 2-opt delta 均基于该距离矩阵完成。

### 3.2 模拟退火 SA

模拟退火维护一个当前解 `current_tour` 和当前路径长度 `current_length`。每轮迭代中，算法通过 2-opt 生成一个邻域解，并计算其相对当前解的长度变化 `delta`。如果 `delta < 0`，说明邻域解更短，直接接受；如果 `delta >= 0`，则以温度 `T` 控制的概率接受该较差解：

```text
P(accept) = exp(-delta / T)
```

接受准则为：

```text
delta < 0: accept
delta >= 0: accept with probability exp(-delta / T)
```

温度采用指数退火：

```text
T(iter) = T0 * (Tf / T0)^(iter / iterations)
```

为了避免每轮计算 `pow`，实现中预先计算：

```text
alpha = (Tf / T0)^(1 / iterations)
T <- T * alpha
```

SA 的关键作用在于：高温阶段允许一定概率接受较差解，增强跳出局部最优的能力；低温阶段逐渐趋向贪心局部搜索，提高收敛稳定性。

### 3.3 QLSA

QLSA 在 SA 主循环基础上加入 Q-learning 动作选择层。普通 SA 每轮随机选择 2-opt move，而 QLSA 每轮先根据当前离散状态 `s` 和 Q 表选择一个邻域动作 `a`，再由动作决定 2-opt 候选区间的生成范围。当前实现中，动作集合对应不同尺度的 2-opt 反转区间，例如 short、medium、long 等。

状态由近期若干次路径长度变化或 delta 值离散化得到。奖励以路径长度减少为正向信号，路径变短时奖励较高，路径变差或拒绝 move 时给出较弱或负向反馈。Q 表更新公式为：

```text
Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

其中 `alpha` 为学习率，`gamma` 为折扣因子，`s'` 为执行 move 后的新状态。动作选择支持 `epsilon-greedy` 与 `softmax` 两种策略。`epsilon-greedy` 以 `epsilon` 概率随机探索，否则选择当前 Q 值最大的动作；`softmax` 则根据 Q 值对应的 softmax 概率分布采样动作。

QLSA 的额外开销主要来自动作选择、状态离散化和 Q 表更新。实验结果也说明，QLSA 不保证所有实例都优于 SA；在默认参数下，QLSA 有时会比 SA 慢且解质量不稳定。但在部分较难实例上，经过参数调优和预算增强后，QLSA 能获得更好的解质量，例如 `rat99` 的 high-budget 实验。

### 3.4 2-opt O(1) delta

2-opt move 通过反转路径中的一个区间 `[i, k]` 生成邻域解。该操作只改变两条边，因此无需每次重新计算完整路径长度。

设：

```text
a = tour[(i - 1 + n) % n]
b = tour[i]
c = tour[k]
d = tour[(k + 1) % n]
```

旧边为：

```text
old: a-b, c-d
```

新边为：

```text
new: a-c, b-d
```

路径长度变化为：

```text
delta = dist(a,c) + dist(b,d) - dist(a,b) - dist(c,d)
```

接受 move 后只需要执行一次区间反转，并令：

```text
current_length += delta
```

这样每轮 move 的长度评估为 O(1)，避免了 O(n) 的完整路径重算。最终算法结束时仍会重新计算 best tour 的完整长度进行校验，保证增量更新没有破坏结果正确性。

## 4. 工程实现

项目采用 C++20 与 CMake 构建，核心算法不依赖 Boost、Eigen、Concorde 等复杂第三方库。项目结构按模块划分，便于后续扩展和实验复现。

| 模块 | 文件 | 功能 |
|---|---|---|
| TSPLIB 解析 | `include/tsp/tsplib_parser.hpp`, `src/tsplib_parser.cpp` | 读取 `.tsp` 文件，解析坐标型和显式矩阵型实例 |
| 距离矩阵 | `include/tsp/distance_matrix.hpp`, `src/distance_matrix.cpp` | 使用一维连续数组存储 `n*n` 距离，便于 CPU cache 和 GPU 拷贝 |
| 路径操作 | `include/tsp/tour.hpp`, `src/tour.cpp` | tour 表示、合法性检查、identity/random/nearest-neighbor 初始化、2-opt delta |
| 随机数与计时 | `include/tsp/rng.hpp`, `include/tsp/timer.hpp`, `src/rng.cpp`, `src/timer.cpp` | 封装随机种子和运行时间统计 |
| SA | `include/tsp/sa.hpp`, `src/sa.cpp` | 串行模拟退火与 2-opt 搜索 |
| QLSA | `include/tsp/qlsa.hpp`, `src/qlsa.cpp` | Q-learning 辅助动作选择和 Q 表更新 |
| OpenMP | `include/tsp/parallel.hpp`, `src/parallel.cpp` | 多搜索链并行、chain 结果归约、OpenMP/串行 fallback |
| CUDA | `include/tsp/cuda.hpp`, `src/cuda.cpp`, `src/cuda_kernels.cu` | GPU 多链并行、距离矩阵拷贝、kernel 执行和 host 归约 |
| 命令行程序 | `src/main.cpp` | 统一 CLI，支持 SA/QLSA、serial/OpenMP/CUDA、repeat 和 CSV 输出 |
| 实验脚本 | `scripts/*.py`, `scripts/*.bat`, `scripts/*.sh` | 自动运行实验、调参、结果统计和报告表格生成 |
| 结果文档 | `docs/*.md`, `results/*.csv` | 保存阶段设计、实验分析和最终汇总 |

TSPLIB95 解析器支持 `NODE_COORD_SECTION`、`EDGE_WEIGHT_SECTION` 和 `EOF`，支持 `EUC_2D`、`CEIL_2D`、`GEO`、`ATT`、`EXPLICIT` 等距离类型，并支持 `FULL_MATRIX`、`UPPER_ROW`、`LOWER_ROW`、`UPPER_DIAG_ROW`、`LOWER_DIAG_ROW` 等显式矩阵格式。距离矩阵使用 `std::vector<int>` 存储连续一维数组，访问形式为 `matrix[i * n + j]`，这既提高了 CPU 缓存局部性，也方便 CUDA 版本直接将 `raw()` 拷贝到 GPU global memory。

## 5. 并行化设计

### 5.1 多搜索链并行思想

SA 和 QLSA 都是随机启发式搜索。单次运行结果会受到随机种子和搜索轨迹影响，因此实践中通常需要多次独立运行并取其中最优结果。本项目将这种多次独立运行抽象为多搜索链并行：一次实验启动多条 chain，每条 chain 独立维护当前路径、最优路径、随机数状态和统计指标，最后对所有 chain 的最优结果进行归约。

该并行粒度有三个优点。第一，各 chain 之间几乎不需要通信，适合多核 CPU 和 GPU 执行。第二，`DistanceMatrix` 是只读对象，可以在线程或 GPU block 间共享。第三，每条 QLSA chain 都拥有独立 Q table，不会出现并发写冲突。最终归约只发生在 chain 完成之后，因此同步开销较低。

### 5.2 OpenMP 实现

OpenMP 实现采用 chain-level `parallel for`。每个 `chain_id` 对应一条独立 SA 或 QLSA 搜索链，线程私有数据包括 RNG、tour、current length、best tour、accepted/improved move 统计，以及 QLSA 的 Q table。并行区内部不频繁更新全局 best，也不在内层 move 循环中加锁。

每条 chain 将结果写入 `chain_results[chain_id]`。由于每个 `chain_id` 只写自己的槽位，不需要互斥锁。并行区结束后，主线程串行遍历 `chain_results`，选择 best length 最小的 chain 作为全局最优，同时汇总所有 chain 的 accepted moves 和 improved moves。

相比 move-level 并行，chain-level 并行通信开销更低，也更容易保证随机过程可复现。move-level 并行虽然理论上可以同时评估多个候选 2-opt move，但会引入更复杂的同步和接受决策冲突；而本项目每次 2-opt delta 已经是 O(1)，单个 move 的计算量较小，因此当前阶段选择 chain-level 并行更合适。

### 5.3 CUDA 实现

CUDA 后端沿用多搜索链抽象，将每条 chain 映射到一个 GPU thread block。host 端将连续一维 `DistanceMatrix` 拷贝到 GPU global memory，kernel 内每个 block 维护独立 tour、RNG 状态、best 结果和 QLSA Q table。kernel 完成后，host 将每个 block 的结果拷回 CPU，并串行归约全局最优。

当前 CUDA 实现已经通过 Ninja + CUDA 构建，`src/cuda_kernels.cu` 能真实编译，且在 `square4` 和 `berlin52` 上可以运行。需要客观说明的是，在当前小规模 TSPLIB 实例上，CUDA 受 kernel 启动开销、调度开销、访存开销、每条 chain 工作量不足等因素影响，未作为主要加速结论。报告中将 CUDA 定位为高工程复杂度扩展和后续优化方向，而不是宣称其比 OpenMP 更快。

## 6. 实验设置

### 6.1 实验环境

本项目最终实验在 Windows 环境下完成，CUDA 构建使用 Ninja 生成器。实验环境如下。

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| CPU | 12th Gen Intel(R) Core(TM) i5-12600KF |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| 编译器 | MSVC 19.44 / nvcc 12.9.41 |
| 构建工具 | CMake + Ninja |
| 构建模式 | Release |
| OpenMP | 启用 |
| CUDA | 启用，Ninja 构建下成功编译 `cuda_kernels.cu` |

### 6.2 数据集、参数与评价指标

实验数据使用 TSPLIB95 标准实例。默认参数多实例实验选取：

```text
berlin52, eil51, st70, eil76, rat99, eil101
```

默认实验设置如下：

| 参数 | 设置 |
|---|---|
| iterations | 1000000 |
| chains | 32 |
| repeat | 3 |
| OpenMP threads | 8 |
| init | nearest-neighbor (`nn`) |
| SA 温度 | 默认命令行参数 |
| QLSA 默认参数 | `alpha=0.1`, `gamma=0.9`, `epsilon=0.1`, `policy=epsilon-greedy` |

评价指标包括：

```text
best length
Gap
elapsed_ms
speedup
efficiency
```

其中：

```text
Gap = (best length - BKS) / BKS * 100%
Speedup = T_serial / T_parallel
Efficiency = Speedup / thread_count * 100%
```

实验分为三个主要阶段。Step 5B 是默认参数多实例加速实验，用于评估 OpenMP 相对串行多链基线的性能提升。Step 6B 是调优参数独立验证，使用不同于调参搜索阶段的 seed=101 起始和 repeat=10，避免只报告搜索中挑出的最好结果。Step 6C 是定向增强高预算质量实验，在较优配置附近增加 chains 或 iterations，观察较难实例的解质量是否进一步改善。

### 6.3 构建与复现实验方式

CUDA/OpenMP Release 构建命令如下：

```powershell
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON
cmake --build build-cuda-ninja -j
ctest --test-dir build-cuda-ninja --output-on-failure
```

默认参数多实例实验命令如下：

```powershell
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\step5_multi_cpu_raw.csv
```

调优参数独立验证命令如下：

```bat
scripts\run_tuned_validation.bat
```

定向增强实验命令如下：

```bat
scripts\run_targeted_quality.bat
```

上述脚本会生成对应的 CSV 与 Markdown 分析文件。最终报告中的关键数据来自 `results/final_key_results.csv` 与 `docs/final_experiment_summary.md`，而不是人工编造的新实验数据。

## 7. 默认参数多实例加速实验

默认参数多实例实验结果如下。串行基线为 serial multi-chain，OpenMP 使用 32 chains 和 8 threads。

| Instance | Family | Serial Mean ms | OpenMP Mean ms | Speedup | Efficiency % | Best Length | Gap % |
|---|---|---:|---:|---:|---:|---:|---:|
| berlin52 | SA | 1043.053 | 202.402 | 5.153 | 64.417 | 7542 | 0.000 |
| berlin52 | QLSA | 2217.895 | 411.445 | 5.391 | 67.381 | 7542 | 0.000 |
| eil51 | SA | 1233.527 | 224.443 | 5.496 | 68.699 | 426 | 0.000 |
| eil51 | QLSA | 2371.958 | 428.979 | 5.529 | 69.116 | 426 | 0.000 |
| st70 | SA | 1243.066 | 247.472 | 5.023 | 62.788 | 675 | 0.000 |
| st70 | QLSA | 2356.846 | 560.222 | 4.207 | 52.587 | 675 | 0.000 |
| eil76 | SA | 1377.377 | 245.818 | 5.603 | 70.040 | 539 | 0.186 |
| eil76 | QLSA | 2439.045 | 490.690 | 4.971 | 62.133 | 542 | 0.743 |
| rat99 | SA | 1310.632 | 226.910 | 5.776 | 72.200 | 1215 | 0.330 |
| rat99 | QLSA | 2400.499 | 570.091 | 4.211 | 52.634 | 1225 | 1.156 |
| eil101 | SA | 1325.891 | 231.733 | 5.722 | 71.520 | 635 | 0.954 |
| eil101 | QLSA | 2490.072 | 445.503 | 5.589 | 69.867 | 637 | 1.272 |

从表中可以看出，OpenMP 多链并行在所有测试实例上均取得稳定加速。SA OpenMP 平均加速约 5.46x，平均并行效率约 68.28%；QLSA OpenMP 平均加速约 4.98x，平均并行效率约 62.29%。OpenMP 加速不改变单条搜索链的核心搜索逻辑，只是并行执行多条独立 chain，因此解质量与串行多链基线保持一致或同等可比。

默认参数下，`berlin52`、`eil51`、`st70` 能达到 BKS，而 `eil76`、`rat99`、`eil101` 仍存在不同程度 Gap。这说明 OpenMP 主要解决运行时间问题，解质量还需要通过参数调优或增加搜索预算进一步改善。

## 8. 调优与定向增强实验

### 8.1 Step 6B 独立验证

Step 6A 参数搜索发现，若调整温度参数、QLSA 学习率、折扣因子、探索概率和搜索预算，较难实例的 Gap 可以降低。为了避免只报告调参搜索中挑出的最好结果，Step 6B 将调优参数固化后使用独立 seed 进行验证。该阶段从 seed=101 开始，repeat=10。

| Instance | Family | Variant | Best Min | Min Gap % | Mean Gap % | Mean ms |
|---|---|---|---:|---:|---:|---:|
| eil101 | QLSA | tuned | 632 | 0.477 | 1.526 | 421.933 |
| eil101 | SA | tuned | 632 | 0.477 | 1.717 | 190.869 |
| eil76 | QLSA | tuned | 541 | 0.558 | 0.985 | 399.390 |
| eil76 | SA | tuned | 538 | 0.000 | 0.483 | 214.220 |
| rat99 | QLSA | quality-first | 1212 | 0.083 | 0.372 | 854.307 |
| rat99 | SA | tuned | 1213 | 0.165 | 0.875 | 206.171 |

结果显示，`eil76` 上 SA tuned 达到 BKS=538。`rat99` 上 QLSA quality-first 的最小 Gap 和平均 Gap 均优于 SA tuned，说明 QLSA 在该实例上具备潜在质量优势。`eil101` 上 SA/QLSA 的最小 Gap 均改善到 0.477%，但 repeat=10 下没有稳定复现 BKS，因此不能只根据 Step 6A 搜索阶段的最好结果下结论。

### 8.2 Step 6C 定向增强

Step 6C 不是重新进行全量参数搜索，而是在 Step 6B 较优配置附近增加搜索预算。具体做法是扩大 `chains` 和 `iterations`。增加 chains 表示一次实验启动更多独立搜索链，通常可以提高找到更好解的概率；增加 iterations 表示单条链搜索更充分，但会增加运行时间。因此该阶段结果必须同时讨论解质量和运行时间成本。

| Instance | Family | Config | Best | Min Gap % | Mean Gap % | Mean ms |
|---|---|---|---:|---:|---:|---:|
| eil101 | QLSA | best-quality | 629 | 0.000 | 0.254 | 3348.545 |
| eil101 | QLSA | best-time-quality-tradeoff | 629 | 0.000 | 0.763 | 787.724 |
| eil101 | SA | best-quality | 629 | 0.000 | 0.445 | 1867.987 |
| rat99 | QLSA | best-quality | 1211 | 0.000 | 0.099 | 3424.631 |
| rat99 | SA | best-quality | 1212 | 0.083 | 0.330 | 1804.426 |

定向增强结果表明，`eil101` 上 SA 与 QLSA 均能达到 BKS=629。其中 QLSA 在 `1e6 iterations + 64 chains` 配置下已经达到 BKS，平均运行时间约 787.724 ms，具有较好的时间-质量折中。`rat99` 上，QLSA 在 `2e6 iterations + 128 chains` 配置下达到 BKS=1211，而 SA high-budget 最好为 1212，未达到 BKS。因此，`rat99` 是本项目中 QLSA 相对 SA 改善解质量的明确案例。

需要强调的是，Step 6C 通过增加搜索预算提高了解质量，因此不应只报告“达到 BKS”，还应同时展示运行时间增加。最终报告中应把 Step 5B 作为主要性能加速结论，把 Step 6B/6C 作为解质量提升分析。

## 9. CUDA 实验与局限性

CUDA 实现过程中首先遇到 Windows Visual Studio CMake 生成器无法识别 CUDA toolset 的问题。后续改用 Ninja 构建后，CUDA language 能够启用，`cuda_kernels.cu` 能真实编译，CUDA enabled 版本可以运行。实验中，CUDA SA/QLSA 在 `square4` 和 `berlin52` 上均可运行，且 `berlin52` 上可以找到 BKS=7542。

然而，当前 CUDA 版本在小规模 TSPLIB 实例上未优于 OpenMP。以 `berlin52` 为例，SA CUDA 平均时间约 3540.677 ms，而 SA OpenMP 平均时间约 196.097 ms；QLSA CUDA 平均时间约 8127.686 ms，而 QLSA OpenMP 平均时间约 465.329 ms。虽然 CUDA 达到相同 best length，但运行时间波动明显且整体较慢。

原因主要包括：小实例每条 chain 的计算量有限，kernel 启动和调度开销占比高；当前 CUDA 基线主要完成多链工程映射，block 内候选 move 并行评估尚未充分展开；GPU global/shared memory 访问和 tour 反转操作仍有优化空间。因此，最终报告中 CUDA 不作为主要加速结论，而作为较高工程复杂度的扩展实现和后续优化方向。

## 10. 遇到的问题与解决方案

1. TSPLIB 数据下载问题。下载脚本在部分环境下会受到 WSL localhost 代理或外部链接可用性影响，导致自动下载失败。解决方式是在 `data/README.md` 和脚本提示中保留手动放置 `.tsp` 文件的路径，后续实验直接使用已经放入 `data/` 的 TSPLIB95 实例。

2. CUDA 构建问题。Visual Studio CMake 生成器未能正确启用 CUDA toolset，导致 CUDA language 不可用。解决方式是改用 Ninja + CUDA 工具链进行真实 CUDA 构建，使 `cuda_kernels.cu` 能够编译并运行。

3. Python 启动器问题。Windows 环境下 `python` 命令可能指向 Microsoft Store alias，出现无输出或启动异常。实验脚本和 bat 包装脚本改用 `py` 启动器运行 Python 脚本，提高了本机运行稳定性。

4. CUDA 小实例性能波动。CUDA 版本在 `berlin52` 这类小规模实例上达到 BKS，但运行时间明显慢于 OpenMP。最终将 OpenMP 作为主要加速结果，将 CUDA 作为工程扩展和后续优化对象。

5. QLSA 默认参数不稳定。默认 QLSA 在部分实例上并不优于 SA，甚至运行时间更长。通过 Step 6A 参数搜索、Step 6B 独立验证和 Step 6C 定向增强，项目明确区分了默认参数性能结论与调优后解质量结论，避免夸大 QLSA 的普适优势。

## 11. 总结

本项目完成了参考论文中 Q-Learning 辅助模拟退火思想的 C++ 工程化实现，构建了从 TSPLIB95 解析、距离矩阵、路径操作、2-opt delta、串行 SA、串行 QLSA 到 OpenMP/CUDA 多链并行的一整套实验框架。项目实现支持可复现 seed、批量实验脚本、CSV 结果输出、自动统计与 Markdown 报告生成。

实验上，OpenMP 多链并行在六个 TSPLIB95 实例上取得稳定加速，SA 平均加速约 5.46x，QLSA 平均加速约 4.98x，说明 chain-level 并行适合 SA/QLSA 这类多次独立随机搜索算法。默认参数下，部分较难实例仍存在 Gap；通过参数调优和定向增强实验，`eil76`、`eil101` 和 `rat99` 的解质量得到改善。其中 `rat99` 上 QLSA high-budget 达到 BKS，而 SA high-budget 未达到 BKS，体现了 QLSA 在特定实例上的质量优势。

CUDA 部分完成了工程实现和运行验证，但当前小规模实例上不优于 OpenMP。该结果说明 GPU 并行化不仅需要完成 kernel 移植，还需要设计足够细粒度且计算密集的 block 内并行策略，才能抵消 kernel 启动和访存开销。

## 12. 后续工作

后续可以从以下方向继续改进：

1. 优化 CUDA 内核，将 block 内线程用于批量候选 2-opt move 评估，而不仅仅执行单 chain 主循环。
2. 扩展到更大规模 TSPLIB95 实例，观察实例规模增大后 CUDA 的潜在收益。
3. 引入更丰富的邻域动作集合，例如不同候选数量、不同交换策略、混合 2-opt/3-opt 等。
4. 设计更系统的 Q-learning 状态表示，例如结合温度区间、接受率、近期改进率和搜索停滞程度。
5. 尝试自适应温度策略，使退火过程能根据实例和搜索状态自动调整。
6. 扩展到多 GPU 或分布式多进程多链搜索，进一步提高大规模实例的搜索覆盖率。

## 参考文献

1. Adil, N., Eddaoudi, F., Lakhbab, H., & Naimi, M. (2026). Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization. Statistics, Optimization & Information Computing, 15(5), 3706-3730. https://doi.org/10.19139/soic-2310-5070-3028
2. Reinelt, G. TSPLIB95: A Traveling Salesman Problem Library. ORSA Journal on Computing, 1991.
3. OpenMP Architecture Review Board. OpenMP Application Programming Interface Specification.
4. NVIDIA. CUDA C++ Programming Guide.
