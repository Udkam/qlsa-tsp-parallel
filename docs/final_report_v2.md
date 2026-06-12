# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化

## 0. 基本信息

- 课程名称：并行算法
- 项目题目：面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化
- 团队人数：1 人
- 团队成员：陈乐浚
- 学号：22361054
- 学院/专业：中山大学计算机学院 / 信息与计算科学

## 1. 摘要

本项目以 2026 年发表的论文 *Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization* 为参考对象，围绕旅行推销员问题（Traveling Salesman Problem, TSP）实现了模拟退火（Simulated Annealing, SA）与 Q-Learning 辅助模拟退火（QLSA），并在此基础上完成 C++20、OpenMP 与 CUDA 多后端工程扩展。项目不仅实现启发式算法本身，还实现了 TSPLIB95 解析器、一维连续距离矩阵、2-opt O(1) delta、命令行接口、自动实验脚本、结果统计脚本和报告图表生成流程。

实验部分分为四类：默认参数多实例加速实验、调优参数独立验证实验、定向增强高预算质量实验，以及与参考论文 Table 8 和 hard-instance 质量结果的参考对比。默认参数下，OpenMP multi-chain 在 `berlin52`、`eil51`、`st70`、`eil76`、`rat99`、`eil101` 六个 TSPLIB95 实例上取得稳定加速：SA OpenMP 平均 speedup 约 5.46x，平均 parallel efficiency 约 68.28%；QLSA OpenMP 平均 speedup 约 4.98x，平均 parallel efficiency 约 62.29%。在定向增强实验中，`eil101` 上 SA 与 QLSA 均达到 BKS=629；`rat99` 上 QLSA high-budget 达到 BKS=1211，而 SA high-budget 最好为 1212，说明 QLSA 在该实例上提供了明确的解质量提升案例。

CUDA backend 已在 Ninja + CUDA 构建下成功编译 `cuda_kernels.cu` 并运行，`berlin52` 上 SA/QLSA CUDA 版本均能找到 BKS。然而，在当前小规模 TSPLIB95 实例上，CUDA backend 受 kernel 启动、访存和每条 chain 工作粒度不足等因素影响，不作为本项目主要加速结论。本报告将 OpenMP multi-chain 作为主要并行性能结果，将 CUDA 定位为已完成的工程扩展与后续优化方向。

## 2. 课程要求与项目完成度对应关系

表 1 给出了本项目与课程大作业要求的对应关系。课程评分点强调完成情况、技术难度、并行性能分析和报告质量；本项目围绕近期论文算法完成了从复现、工程化、并行化到实验分析的完整流程。

| 课程要求/评分点 | 本项目对应完成内容 |
|---|---|
| 完成情况 50% | 完成 SA、QLSA、OpenMP multi-chain、CUDA backend、TSPLIB95 parser、CLI、测试、自动实验脚本和多实例实验 |
| 技术难度 30% | C++20 底层实现、O(1) 2-opt delta、OpenMP 粗粒度并行、CUDA kernel 工程、参数调优、结果自动化分析 |
| 近期论文复现/并行化 | 基于 2026 年 QLSA for TSP 论文，完成算法思想复现与并行化扩展 |
| 与近期论文对比 | 引入论文 Table 8 运行时间和 hard-instance 质量表作为参考对比 |
| 并行性能分析 | 统计 elapsed time、speedup、parallel efficiency、Gap、best length |
| 报告 20% | 提供小组报告、个人报告附录、实验图表、最终结果汇总和提交检查清单 |

本项目不是简单地对循环加 `parallel for`，而是先建立可复现实验基线，再将多条独立搜索链映射到 OpenMP 线程和 CUDA 后端，最后通过自动化脚本对 speedup、parallel efficiency 与解质量进行统计。

## 3. 参考论文精读与对比基准

### 3.1 论文基本信息

参考论文为：

Adil, N., Eddaoudi, F., Lakhbab, H., & Naimi, M. (2026). *Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization*. *Statistics, Optimization & Information Computing*, 15(5), 3706-3730. https://doi.org/10.19139/soic-2310-5070-3028

论文使用 TSPLIB95 实例评估 SA、QLSA 与 SB-QLSA。其实验环境为 64-bit Linux，Intel Xeon Gold 6130 CPU @ 2.10 GHz，Python 3.11.5，并使用 NumPy、Pandas 与 TSPLIB95 Python library。论文对每个实例进行 10 次独立运行，报告 Best、Worst、Mean、Std、Gap 和 computational time。Table 8 给出各算法运行时间，Table 9 进一步将 Q-learning variants 的时间与 SA 进行相对开销比较。

### 3.2 论文算法机制

论文中的 Paper-SA 采用 classical SA with 2-opt Metropolis。Paper-QLSA 在 SA 外层引入 stateless Q-learning，用于从候选解集合中选择引导搜索的解。候选集合可概括为：

\[
C=\{c_1,c_2,c_3,c_4\}
\]

其中包括当前解、全局最优解、随机解和 double-bridge 扰动解。论文同时讨论了 epsilon-greedy 与 Softmax/Boltzmann 两种 action selection 策略，并指出 Softmax variants 在其实验平均表现上更稳定。Paper-SB-QLSA 进一步引入 state-based Q-learning，用 diversity state 描述当前解与 best solution 的相似程度，从而构造 state-action Q table。

论文的贡献重点在于用 Q-learning 辅助选择搜索策略，使 SA 不再只依赖固定邻域扰动。论文实验表明，Q-learning variants 在多个 TSPLIB95 实例上相对于 classical SA 有更好的平均 Gap，但也带来额外运行时间开销。

### 3.3 本项目实现与论文的对应关系

表 2 总结了论文机制与本项目实现之间的对应关系。需要明确的是，本项目是基于论文思想进行工程复现和并行化扩展，并不声称逐行复刻论文所有 SB-QLSA 细节。

| 论文内容 | 本项目实现 | 说明 |
|---|---|---|
| TSPLIB95 实例 | 支持 `.tsp` parser | 覆盖 EUC_2D、CEIL_2D、GEO、ATT、EXPLICIT 及多种显式矩阵格式 |
| SA + 2-opt Metropolis | 已实现 SA + 2-opt delta | 使用 O(1) delta，避免每次 move 完整重算 tour length |
| QLSA | 已实现 QLSA | 本项目采用动作/状态离散化，与论文 candidate-leader 机制不完全相同 |
| epsilon-greedy / softmax | CLI 支持两者 | 默认实验主要使用 epsilon-greedy |
| SB-QLSA | 部分思想实现为 state/action discretization | 不声称完全复刻论文 SB-QLSA 的 candidate-leader + diversity-state 机制 |
| 论文串行 Python | 本项目 C++20/OpenMP/CUDA | 实现语言、硬件和并行后端均不同 |
| 论文 future parallel implementations | 本项目完成 OpenMP/CUDA backend | 是本项目相对论文的主要工程扩展点 |

## 4. 算法与实现设计

![图 1：项目整体架构](../figures/fig_architecture_pipeline.png)

图 1 展示了从 TSPLIB95 输入、C++ 核心算法、多后端执行到 CSV 结果分析和报告生成的完整流水线。

### 4.1 TSP 定义与 BKS/Gap

TSP 给定 \(n\) 个城市和距离矩阵 \(D\)，要求寻找一条访问每个城市恰好一次并回到起点的最短 Hamiltonian 回路。若路径表示为排列 \(\pi\)，则 tour length 为：

\[
L(\pi)=\sum_{i=0}^{n-1}D_{\pi_i,\pi_{(i+1)\bmod n}}
\tag{1}
\]

实验中使用 TSPLIB95 给出的 Best Known Solution（BKS）计算 Gap：

\[
\text{Gap}=\frac{\text{best length}-\text{BKS}}{\text{BKS}}\times 100\%
\tag{2}
\]

### 4.2 SA 与 2-opt Metropolis

SA 维护一个当前解，并通过 2-opt 反转区间生成邻域解。若 move 带来路径长度变化 \(\Delta\)，接受概率为：

\[
P(\Delta,T)=
\begin{cases}
1, & \Delta \le 0 \\
\exp(-\Delta/T), & \Delta > 0
\end{cases}
\tag{3}
\]

温度采用指数退火：

\[
T_k = T_0\left(\frac{T_f}{T_0}\right)^{k/N}
\tag{4}
\]

实现中预先计算每轮乘法因子，避免每次迭代调用 `pow`。

### 4.3 QLSA 与 Q-learning 更新

QLSA 在 SA 的邻域选择过程中加入 Q-learning。状态 \(s\) 由近期路径长度变化或 delta 离散化得到，动作 \(a\) 表示可选邻域策略。每次 move 接受或拒绝后，根据 reward 更新 Q table：

\[
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\tag{5}
\]

其中 \(\alpha\) 为 learning rate，\(\gamma\) 为 discount factor。本项目支持 epsilon-greedy 与 softmax 策略。QLSA 的额外开销主要来自动作选择、状态离散化和 Q table 更新，因此其 elapsed time 通常高于同等预算下的 SA；但在部分 harder instances 上，QLSA 能提高找到高质量解的概率。

### 4.4 O(1) 2-opt delta

对 tour 反转区间 \([i,k]\) 时，旧边为 \(a-b\)、\(c-d\)，新边为 \(a-c\)、\(b-d\)，其中：

\[
a=\text{tour}_{(i-1+n)\bmod n},\quad b=\text{tour}_{i},\quad c=\text{tour}_{k},\quad d=\text{tour}_{(k+1)\bmod n}
\]

因此 delta 可用常数时间计算：

\[
\Delta = D_{a,c}+D_{b,d}-D_{a,b}-D_{c,d}
\tag{6}
\]

这一设计是高性能 SA/QLSA 的关键，因为内层迭代次数可达到百万级或更高。

### 4.5 工程模块

表 3 给出了项目主要模块。核心数据结构均为后续 OpenMP 与 CUDA 后端服务，尤其是 `DistanceMatrix::raw()` 可直接拷贝到 GPU 全局内存。

| 模块 | 文件 | 功能 |
|---|---|---|
| TSPLIB 解析 | `include/tsp/tsplib_parser.hpp`, `src/tsplib_parser.cpp` | 读取 `.tsp` 文件，支持坐标型与显式矩阵型实例 |
| 距离矩阵 | `include/tsp/distance_matrix.hpp`, `src/distance_matrix.cpp` | 用一维连续数组存储 \(n\times n\) 距离 |
| 路径操作 | `include/tsp/tour.hpp`, `src/tour.cpp` | tour 表示、合法性检查、nearest-neighbor 初始化、2-opt delta |
| SA | `include/tsp/sa.hpp`, `src/sa.cpp` | 串行模拟退火 |
| QLSA | `include/tsp/qlsa.hpp`, `src/qlsa.cpp` | Q-learning 辅助模拟退火 |
| OpenMP | `include/tsp/parallel.hpp`, `src/parallel.cpp` | 多搜索链并行与结果归约 |
| CUDA | `include/tsp/cuda.hpp`, `src/cuda.cpp`, `src/cuda_kernels.cu` | GPU 多链并行工程扩展 |
| 实验脚本 | `scripts/*.py`, `scripts/*.bat` | 自动运行、统计、调参、图表生成 |

## 5. 并行化设计

### 5.1 并行机会分析

SA/QLSA 的单条搜索链具有明显的前后依赖，因为下一次 move 的接受概率依赖当前 tour 和当前温度。因此，直接对单条 chain 的迭代序列做 fine-grained 并行并不自然。相比之下，多搜索链之间相互独立：每条 chain 使用独立 seed、tour、best tour 和 Q table，共享只读 DistanceMatrix，最后仅需对各 chain 的 best result 做归约。这种并行粒度通信开销低，适合 OpenMP 和 CUDA。

### 5.2 OpenMP multi-chain 实现

OpenMP 后端采用 chain-level `parallel for`。每条 chain 在线程私有数据中执行完整 SA/QLSA 搜索，不在内层 move 循环中频繁加锁，结束后写入 `chain_results[chain_id]`。全局 best tour 由主线程在 parallel region 结束后串行归约。

伪代码如下：

```text
Algorithm: OpenMP Multi-chain SA/QLSA
Input: distance matrix D, chains C, threads p, base_seed
parallel for chain_id in [0, C):
    seed_i = derive_seed(base_seed, chain_id)
    result_i = run_sa_or_qlsa(D, seed_i)
serial reduction over result_i
return global best
```

该方案属于粗粒度并行，避免了共享写冲突，也避免了 move-level 并行中频繁同步的问题。由于 DistanceMatrix 只读共享、每条 chain 独立，OpenMP 在 8 线程下能取得约 60%-70% 的 parallel efficiency。

### 5.3 CUDA backend 实现

CUDA backend 将多条 chain 映射到 GPU 并行执行，DistanceMatrix 拷贝到 GPU 全局内存，Q table 在每条 chain 内独立维护。主机端负责启动 kernel、收集每条 chain 的 best result，并做最终归约。当前 CUDA 版本已经能通过 Ninja + CUDA 真实编译 `cuda_kernels.cu`，并在 `square4` 与 `berlin52` 等小实例上运行。

不过，当前 CUDA 后端仍偏工程验证性质。小规模 TSP 实例下，每条 chain 的计算量不足以抵消 kernel 启动、调度和访存开销。后续更合理的 GPU 优化方向是让 block 内线程批量评估候选 2-opt move，提升每次 kernel 调用的算术密度。

## 6. 实验设置

### 6.1 实验环境

表 4 给出了本项目实际实验环境。

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

### 6.2 实验分组与指标

默认参数加速实验使用 `berlin52`、`eil51`、`st70`、`eil76`、`rat99`、`eil101`。默认设置为 `iterations=1000000`、`chains=32`、`repeat=3`、OpenMP `threads=8`、`init=nn`。QLSA 默认参数为 `alpha=0.1`、`gamma=0.9`、`epsilon=0.1`、`policy=epsilon-greedy`。

实验指标包括：

- best length：多次运行中的最短路径长度；
- Gap：相对于 BKS 的偏差；
- elapsed time：运行时间；
- speedup：\(T_{serial}/T_{parallel}\)；
- parallel efficiency：\(\text{speedup}/p\times 100\%\)。

实验组严格区分如下：

| 实验组 | 目的 | 主要输出 |
|---|---|---|
| Step 5B 默认参数实验 | 评估 OpenMP multi-chain speedup | `results/step5_multi_cpu_summary.csv` |
| Step 6B 调优独立验证 | 使用独立 seed 验证调优参数质量 | `results/tuned_validation_summary.csv` |
| Step 6C 定向增强实验 | 增加 chains/iterations 观察 harder instances 质量 | `results/targeted_quality_summary.csv` |
| 论文参考对比 | 对照论文 Table 8 和 hard-instance quality | `results/paper_table8_runtime.csv`, `results/paper_hard_instance_quality.csv` |

### 6.3 构建与复现实验方式

推荐使用 Ninja 构建启用 CUDA 的版本：

```powershell
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON
cmake --build build-cuda-ninja -j
ctest --test-dir build-cuda-ninja --output-on-failure
```

默认参数多实例实验：

```powershell
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\step5_multi_cpu_raw.csv
```

调优验证与定向增强实验：

```powershell
scripts\run_tuned_validation.bat
scripts\run_targeted_quality.bat
```

报告图表可由以下命令生成：

```powershell
py scripts\make_report_figures.py
py scripts\check_report_assets.py
```

## 7. 与论文结果对比

### 7.1 与论文实验设计的差异

表 5 给出了参考论文与本项目实验设置的主要差异。由于实现语言、硬件和并行后端均不同，论文运行时间与本项目运行时间不构成严格同平台 benchmark；本报告将其作为课程要求下的论文基准参考对比。

| 项目 | 论文 | 本项目 |
|---|---|---|
| 实现语言 | Python 3.11.5 + NumPy/Pandas | C++20 |
| 硬件 | Intel Xeon Gold 6130 VM/HPC-MARWAN | i5-12600KF + RTX 4070 SUPER |
| 并行后端 | 未实现 parallel backend | OpenMP multi-chain + CUDA backend |
| 运行次数 | 每实例 10 次 | default repeat=3，tuned repeat=10，targeted repeat=5 |
| 算法 | SA, QLSA, SB-QLSA | SA, QLSA, OpenMP/CUDA multi-chain |
| 指标 | Best/Worst/Mean/Std/Gap/Time | best length/Gap/elapsed time/speedup/parallel efficiency |

### 7.2 论文 Table 8 运行时间 vs 本项目 OpenMP 时间

![图 6：论文与本项目运行时间参考对比](../figures/fig_paper_runtime_comparison_log.png)

图 6 展示了论文 Table 8 的秒级运行时间与本项目 OpenMP elapsed time 的参考对比，纵轴使用 log scale。该图说明 C++20 + OpenMP multi-chain 在本项目环境下具有明显工程时间优势，但该优势不能单独归因于算法本身，也包含语言、实现优化、硬件和并行化差异。

表 6 给出了精简数值。完整数据见 `results/paper_table8_runtime.csv` 与 `results/report_comparison_summary.csv`。

| Instance | Paper SA (s) | Paper QLSAε (s) | Our SA OpenMP (s) | Our QLSA OpenMP (s) |
|---|---:|---:|---:|---:|
| berlin52 | 600.56 | 644.12 | 0.202 | 0.411 |
| eil51 | 589.41 | 602.53 | 0.224 | 0.429 |
| st70 | 2460.60 | 2498.47 | 0.247 | 0.560 |
| eil76 | 2379.99 | 2450.12 | 0.246 | 0.491 |
| rat99 | 5027.88 | 5003.58 | 0.227 | 0.570 |
| eil101 | 151064.69 | 152305.01 | 0.232 | 0.446 |

### 7.3 论文解质量 vs 本项目调优/增强质量

![图 7：harder instances 解质量对比](../figures/fig_paper_quality_hard_instances.png)

图 7 比较了论文在 `eil76`、`rat99`、`eil101` 上的 mean Gap 与本项目调优/定向增强后的 mean Gap。论文结果显示 Q-learning variants 相对 Paper-SA 明显改善 mean Gap；本项目在 C++/OpenMP multi-chain 框架下，通过调优与增加搜索预算进一步接近 BKS。

在 `rat99` 上，本项目 QLSA targeted high-budget 达到 BKS=1211，mean Gap 为 0.099%；对应 SA high-budget 最好为 1212，mean Gap 为 0.330%。在 `eil101` 上，SA 与 QLSA targeted 均达到 BKS=629，其中 QLSA best-quality mean Gap 为 0.254%。这些结果说明，多链并行与参数调优能显著提高 harder instances 的搜索覆盖率。

需要注意，本项目 QLSA 与论文 SB-QLSA 的机制并不完全相同，因此该对比主要用于展示同一类 Q-learning-assisted SA 思路在工程化和并行化后的效果，而不是声明完全复刻论文所有变体。

## 8. 本项目内部并行性能分析

### 8.1 OpenMP speedup

![图 2：OpenMP 多实例加速比](../figures/fig_openmp_speedup.png)

图 2 展示了默认参数下 SA 与 QLSA 的 OpenMP speedup。表 7 给出关键数值。

| Instance | Family | Serial mean ms | OpenMP mean ms | Speedup | Efficiency | Best length | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| berlin52 | SA | 1043.053 | 202.402 | 5.153 | 64.417% | 7542 | 0.000% |
| berlin52 | QLSA | 2217.895 | 411.445 | 5.391 | 67.381% | 7542 | 0.000% |
| eil51 | SA | 1233.527 | 224.443 | 5.496 | 68.699% | 426 | 0.000% |
| eil51 | QLSA | 2371.958 | 428.979 | 5.529 | 69.116% | 426 | 0.000% |
| st70 | SA | 1243.066 | 247.472 | 5.023 | 62.788% | 675 | 0.000% |
| st70 | QLSA | 2356.846 | 560.222 | 4.207 | 52.587% | 675 | 0.000% |
| eil76 | SA | 1377.377 | 245.818 | 5.603 | 70.040% | 539 | 0.186% |
| eil76 | QLSA | 2439.045 | 490.690 | 4.971 | 62.133% | 542 | 0.743% |
| rat99 | SA | 1310.632 | 226.910 | 5.776 | 72.200% | 1215 | 0.330% |
| rat99 | QLSA | 2400.499 | 570.091 | 4.211 | 52.634% | 1225 | 1.156% |
| eil101 | SA | 1325.891 | 231.733 | 5.722 | 71.520% | 635 | 0.954% |
| eil101 | QLSA | 2490.072 | 445.503 | 5.589 | 69.867% | 637 | 1.272% |

SA OpenMP 平均 speedup 约 5.46x，QLSA OpenMP 平均 speedup 约 4.98x。OpenMP multi-chain 是本项目最稳定的性能提升来源。

### 8.2 OpenMP parallel efficiency

![图 3：OpenMP 并行效率](../figures/fig_openmp_efficiency.png)

图 3 展示了 8 线程 OpenMP 的 parallel efficiency。SA 平均 efficiency 约 68.28%，QLSA 平均 efficiency 约 62.29%。QLSA efficiency 略低，主要原因是 Q table update、action selection 和状态离散化增加了单条 chain 的额外开销，同时随机搜索本身也会带来不同实例间的时间波动。

### 8.3 默认参数 Gap

![图 4：默认参数 Gap](../figures/fig_default_gap.png)

图 4 展示了默认参数下的 best length Gap。`berlin52`、`eil51`、`st70` 在默认设置下达到 BKS；`eil76`、`rat99`、`eil101` 仍存在不同程度 Gap。因此，默认参数实验主要支撑并行加速结论，而 harder instances 的解质量需要进一步调参或增加搜索预算。

## 9. 参数调优与定向增强

![图 5：调优与定向增强 Gap 改善](../figures/fig_tuned_quality_improvement.png)

图 5 展示了 default、tuned validation 和 targeted high-budget 三个阶段的 Gap 改善。

### 9.1 Step 6B 调优参数独立验证

Step 6B 使用 Step 6A 调出的参数，但换用从 101 开始的独立 seed，并执行 repeat=10，避免只报告调参搜索中的最好样本。表 8 给出关键结果。

| Instance | Family | Variant | Best min | Min Gap | Mean Gap | Mean ms |
|---|---|---|---:|---:|---:|---:|
| eil76 | SA | tuned | 538 | 0.000% | 0.483% | 214.220 |
| eil76 | QLSA | tuned | 541 | 0.558% | 0.985% | 399.390 |
| rat99 | SA | tuned | 1213 | 0.165% | 0.875% | 206.171 |
| rat99 | QLSA | quality-first | 1212 | 0.083% | 0.372% | 854.307 |
| eil101 | SA | tuned | 632 | 0.477% | 1.717% | 190.869 |
| eil101 | QLSA | tuned | 632 | 0.477% | 1.526% | 421.933 |

结果表明，`eil76` 上 SA tuned 达到 BKS；`rat99` 上 QLSA quality-first 的 min Gap 和 mean Gap 均优于 SA tuned；`eil101` 上调优改善了 min Gap，但 repeat=10 下没有稳定达到 BKS。

### 9.2 Step 6C 定向增强实验

Step 6C 不是重新做全量网格搜索，而是在较优参数附近增加 chains 或 iterations。增加 chains 代表一次实验启动更多独立搜索链，通常提高找到更好解的概率；增加 iterations 代表单条 chain 搜索更充分，但会增加 elapsed time。

表 9 给出定向增强的关键结论。

| Instance | Family | Config | Best | Min Gap | Mean Gap | Mean ms |
|---|---|---|---:|---:|---:|---:|
| eil101 | QLSA | it=2e6, chains=128 | 629 | 0.000% | 0.254% | 3348.545 |
| eil101 | QLSA | it=1e6, chains=64 | 629 | 0.000% | 0.763% | 787.724 |
| eil101 | SA | it=2e6, chains=128 | 629 | 0.000% | 0.445% | 1867.987 |
| rat99 | QLSA | it=2e6, chains=128 | 1211 | 0.000% | 0.099% | 3424.631 |
| rat99 | SA | it=2e6, chains=128 | 1212 | 0.083% | 0.330% | 1804.426 |

`eil101` 上 SA 与 QLSA 均能在定向增强实验中达到 BKS=629，其中 QLSA `1e6 iterations + 64 chains` 已达到 BKS，具有较好的时间-质量折中。`rat99` 上，QLSA `2e6 iterations + 128 chains` 达到 BKS=1211，而 SA high-budget 最好为 1212，未达到 BKS。因此，`rat99` 是本项目中 QLSA 相对于 SA 改善 solution quality 的明确案例。

该结论需要与运行时间成本一起解读：targeted high-budget 通过增加搜索预算提升质量，不应与 default-parameter speedup 结论混为一谈。

## 10. CUDA 工程扩展与局限

![图 8：berlin52 CUDA/OpenMP/Serial 时间对比](../figures/fig_cuda_positioning.png)

图 8 展示了 `berlin52` 上 serial multi-chain、OpenMP multi-chain 和 CUDA backend 的 elapsed time。CUDA kernel 已通过 Ninja + CUDA 构建并运行，`berlin52` 上 SA/QLSA CUDA 版本均能达到 BKS=7542。

但从 elapsed time 看，CUDA backend 当前在小规模实例上慢于 OpenMP：SA CUDA mean time 约 3540.677 ms，而 SA OpenMP 约 196.097 ms；QLSA CUDA mean time 约 8127.686 ms，而 QLSA OpenMP 约 465.329 ms。这一现象符合当前实现粒度：每条 chain 的计算密度不足，GPU kernel 启动和访存成本占比较高。

因此，最终报告中 CUDA 的定位是“已完成并验证的工程扩展”，而非当前小规模实例上的主要 speedup 证据。后续如果要发挥 GPU 优势，应将 block 内线程用于候选 move 批量评估、共享内存缓存和更大规模实例。

## 11. 问题与解决方案

表 10 总结了项目过程中遇到的主要问题和处理方式。

| 问题 | 现象 | 解决方案 | 对项目的影响 |
|---|---|---|---|
| TSPLIB 下载 | WSL/代理环境下官方下载脚本可能因 localhost 代理失败 | 支持手动下载 `.tsp` 放入 `data/`，脚本缺失时跳过 | 保证项目不因数据下载失败而不可运行 |
| Visual Studio CUDA toolset | VS CMake 生成器未能启用 CUDA toolset | 改用 Ninja，成功启用 CUDA language | 真实编译 `cuda_kernels.cu` |
| Python Store alias | `python` 可能无输出或指向 Store alias | 改用 `py` launcher | 自动实验脚本可稳定运行 |
| CUDA 小实例波动 | `berlin52` 上 CUDA elapsed time 高于 OpenMP | 将 OpenMP 作为主要性能结论，CUDA 作为工程扩展 | 避免夸大 GPU 结果 |
| QLSA 默认参数不稳定 | 默认参数下 harder instances Gap 不如 SA | Step 6A/6B/6C 做调参与独立验证 | 得到 `rat99` QLSA 质量优势案例 |
| 论文机制与项目机制差异 | 本项目 QLSA 未完全复刻论文 SB-QLSA | 报告中明确“基于思想的工程复现和并行扩展” | 保证论文对比表述准确 |

## 12. 总结与贡献

本项目完成了一个围绕近期论文算法的并行算法工程系统，主要贡献如下：

- 算法贡献：实现 TSP 上的 SA 与 QLSA，支持 epsilon-greedy 与 softmax 策略，使用 O(1) 2-opt delta 提升内层迭代效率。
- 并行贡献：实现 OpenMP multi-chain 并行，在六个 TSPLIB95 实例上获得约 5x 平均 speedup；实现 CUDA backend 并通过真实 CUDA 构建与运行验证。
- 工程贡献：完成 TSPLIB95 parser、DistanceMatrix、Tour、CLI、测试、实验脚本、调参脚本、分析脚本和报告图表生成脚本。
- 实验贡献：区分 default speedup、tuned validation、targeted high-budget 与 paper reference comparison，避免将不同性质的结果混为一谈。
- 对论文的扩展：将论文中的 Q-learning-assisted SA 思路迁移到 C++20/OpenMP/CUDA 工程体系中，并补充了并行性能指标 speedup 与 parallel efficiency。

项目也存在明确局限：论文运行时间对比并非同硬件同语言公平 benchmark；CUDA backend 仍需进一步优化；QLSA 不是在每个实例上都优于 SA；本项目没有完全复刻论文 SB-QLSA 的所有 candidate-leader 与 diversity-state 机制。

## 13. 后续工作

后续可以从以下方向继续改进：

- 完整复刻 paper SB-QLSA 的 candidate-leader 与 diversity-state 机制；
- 系统比较 epsilon-greedy 与 softmax 策略，并增加统计显著性检验；
- 在 CUDA block 内并行批量评估候选 2-opt move，提高 GPU 算术密度；
- 扩展到更大规模 TSPLIB95 实例；
- 记录收敛曲线，分析不同算法在相同时间预算下的质量变化；
- 探索自适应温度策略、更丰富邻域动作集合和多 GPU 扩展。

## 参考文献

1. Adil, N., Eddaoudi, F., Lakhbab, H., & Naimi, M. (2026). Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization. *Statistics, Optimization & Information Computing*, 15(5), 3706-3730. https://doi.org/10.19139/soic-2310-5070-3028
2. Reinelt, G. TSPLIB: A Traveling Salesman Problem Library. *ORSA Journal on Computing*, 3(4), 376-384, 1991.
3. OpenMP Architecture Review Board. *OpenMP Application Programming Interface Specification*.
4. NVIDIA. *CUDA C++ Programming Guide*.
5. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. Optimization by Simulated Annealing. *Science*, 220(4598), 671-680, 1983.
6. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*. MIT Press.
