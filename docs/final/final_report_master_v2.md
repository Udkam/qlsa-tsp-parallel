# 1. 问题背景与课程目标映射

OpenMP multi-chain achieves about 5.46x stable speedup for SA and about 4.98x speedup for QLSA across six TSPLIB95 instances. This result is the central performance claim of the project: TSP local search is difficult to parallelize inside one chain, but it is highly suitable for independent multi-chain parallel search.

TSP asks for the shortest Hamiltonian cycle over a set of cities. Its NP-hard nature makes exact methods unsuitable as the primary course target once repeated experiments and parameter sweeps are required. SA is selected because 2-opt neighborhoods and Metropolis acceptance provide a compact, reproducible stochastic optimization baseline. QLSA is selected because the reference paper shows that reinforcement learning can guide search behavior and improve solution quality on harder instances. Parallelization is required because the scientific question is not only whether SA/QLSA can solve TSP, but whether the search process can be made faster without changing the quality evaluation protocol.

The design therefore starts from a concrete thesis: independent search chains are the natural parallel unit. This matters because each chain can keep its own random state, tour and Q table, while the distance matrix remains read-only. The resulting OpenMP implementation reduces elapsed time without changing the underlying acceptance rule or Gap definition.

表 1 将课程评分点、课程目标和报告证据直接对应，作为全文论证的索引。

表 1：课程评分点与论证证据。

| 课程目标 | 证据 | 论证作用 |
|---|---|---|
| 完成情况 | SA、QLSA、OpenMP、CUDA、TSPLIB95 parser、实验系统 | 证明任务不是单一算法 demo。 |
| 技术难度 | C++20、O(1) delta、CUDA pipeline、自动化分析系统 | 证明工程实现有底层复杂度。 |
| 并行性能 | SA 5.46x、QLSA 4.98x、efficiency 分析 | 支撑主性能结论。 |
| 近期论文对比 | 论文 Table 8 与 hard-instance quality 数据 | 建立算法来源和参考坐标。 |
| 报告质量 | 图表、索引、course/public 分离、复现命令 | 支撑提交级完整性。 |

# 2. 论文方法拆解

The paper validates the algorithmic idea; this project validates the engineering scalability of that idea. 本项目不是复现论文，而是工程化扩展：论文重点比较 SA、QLSA、SB-QLSA 的搜索质量和运行开销，工程侧则把这些思想迁移到 C++20、OpenMP 和 CUDA 后端下进行可复现实验。

## 2.1 SA

SA is the control baseline. The paper uses 2-opt Metropolis search: a shorter candidate tour is accepted directly, and a longer candidate may still be accepted with temperature-controlled probability. This mechanism matters because it provides both exploration and convergence, making it a suitable baseline for QLSA and parallel multi-chain evaluation.

工程侧保留这一核心机制，同时加入 O(1) 2-opt delta。该改动不改变 SA 的搜索定义，却显著降低百万级迭代中的内层计算成本。

## 2.2 QLSA

QLSA adds a learning layer over SA. The paper uses Q-Learning to select among candidate leaders such as current solution, global best solution, random solution and double-bridge solution. The intended effect is to make search policy adaptive instead of relying on a fixed neighborhood selection rule.

工程侧实现了 Q 表、epsilon-greedy、Softmax、状态离散化和奖励更新。差异在于 action 被定义为不同 2-opt 邻域策略，而不是论文 candidate leader 的逐项复刻。因此 QLSA results in this report demonstrate an engineering variant of the paper idea, not a line-by-line reproduction.

## 2.3 SB-QLSA

SB-QLSA in the paper introduces a diversity state based on the relation between current solution and best solution. Its role is to condition action values on search status. This is important because QLSA quality depends on whether the algorithm is exploring or intensifying.

工程侧采用 delta/state discretization 捕捉搜索状态变化，但没有完整实现论文中的 diversity-state candidate-leader mechanism。该限制必须前置，因为它决定了论文对比的解释边界：质量趋势可以比较，机制等价性不能声称。

## 2.4 论文实验结构

The paper reports quality and runtime through Best/Mean/Std/Gap and computational time tables. Table 4 and related hard-instance tables show that Q-learning variants improve mean Gap over Paper-SA on eil76, rat99 and eil101. Table 8 reports runtime. Table 9 discusses the extra cost of Q-learning variants.

表 2：论文机制与工程侧对应关系。

| Paper component | Engineering counterpart | Match level | Implication |
|---|---|---|---|
| SA + 2-opt Metropolis | SA baseline + O(1) delta | Direct | Baseline is comparable in search principle. |
| QLSA candidate leader | QLSA action selection | Partial | Learning idea retained, action semantics differ. |
| SB-QLSA diversity state | State discretization | Partial | State-aware idea retained, paper mechanism not fully reproduced. |
| Python implementation | C++20/OpenMP/CUDA system | Extension | Runtime comparison is reference-level only. |

# 3. 系统设计

The system is designed to make performance claims traceable. C++ is chosen because the inner loop contains millions of 2-opt moves, and the cost of distance lookup, delta computation and random acceptance directly determines elapsed time. A Python-level implementation would mix algorithmic behavior with interpreter overhead; C++ separates the algorithmic claim from avoidable runtime cost.

TSPLIB95 is chosen because it gives a common benchmark language shared with the paper. BKS values make Gap meaningful, and standard `.tsp` files prevent the experiment from being tied to one hand-written instance. Supporting both coordinate and explicit matrix cases also prevents the parser from becoming a berlin52-only shortcut.

DistanceMatrix is stored as one contiguous array. This decision matters twice: CPU search gets predictable memory access, and CUDA receives a raw buffer layout suitable for global memory transfer. The one-dimensional representation is therefore not only a micro-optimization; it is the data contract shared by serial, OpenMP and CUDA backends.

O(1) 2-opt delta is the decisive inner-loop design. Recomputing tour length after every candidate move would dominate runtime and weaken any parallel speedup claim. The implementation only updates the two removed and two added edges:

$$
\Delta=D_{a,c}+D_{b,d}-D_{a,b}-D_{c,d}
$$

CLI + CSV pipeline is treated as a system component, not as a convenience script. Parameters such as seed, repeat, chains, threads, policy and CUDA block size must be reproducible. CSV output creates a stable path from executable results to analysis, figures and final conclusions.

![System architecture and data flow](../../figures/final/fig01_architecture_pipeline.png)

图 1：系统总体架构与数据流。

# 4. 并行设计

The parallel design is deliberately chain-level because that is where SA has independence. A single SA chain is sequential: each tour update changes the next state. Multiple chains, however, are independent stochastic searches over the same read-only distance matrix. This gives a clean parallel boundary and a clean reduction step.

## 4.1 为什么 SA 可并行

SA is parallelizable at the multi-start level. Each chain can run with a different seed, accept or reject moves independently, and maintain its own best tour. The only shared object is the distance matrix, which is immutable during search. This property makes race-free OpenMP execution straightforward.

Why it matters: the parallel unit matches the algorithm's stochastic nature. The design accelerates exploration over multiple independent trajectories rather than forcing artificial parallelism into one dependent trajectory.

## 4.2 为什么选择 chain-level OpenMP

OpenMP chain-level execution is the primary performance mechanism. It produces about 5.46x average speedup for SA and about 4.98x for QLSA on 8 threads. The reason is structural: each chain writes to a private result slot, and global best reduction happens only after the parallel region.

This demonstrates that speedup comes from reduced synchronization, not from changing the optimization problem. It also preserves reproducibility because seed derivation and chain outputs remain explicit.

## 4.3 为什么不做 move-level

Move-level parallelism is rejected because it conflicts with the stateful nature of SA. Candidate moves are evaluated against the current tour, but accepting one move changes that tour and invalidates other candidates. It would also complicate Metropolis randomness and reproducibility.

The tradeoff is explicit: chain-level parallelism leaves single-chain computation serial, but it produces stable speedup and lower engineering risk. For a course project measured by correctness, speedup and reproducibility, this is the stronger design.

## 4.4 CUDA 为什么存在但不是主结果

CUDA is fully implemented but not competitive on small instances. The backend demonstrates a complete GPU path, but berlin52-level workloads are too small to amortize kernel launch, scheduling and memory overhead.

Why it matters: a negative CUDA speedup result is still informative. It shows that GPU parallelization requires enough per-kernel work or finer-grained candidate evaluation. The report therefore treats CUDA as engineering evidence and future direction, while OpenMP remains the performance result.

表 3：并行取舍结论。

| Design choice | Assertive conclusion | Risk controlled |
|---|---|---|
| OpenMP chain-level | Main speedup mechanism | Minimal synchronization |
| Move-level | Not selected | Avoids invalid candidate states |
| CUDA backend | Engineering extension | Avoids overstating small-instance performance |

# 5. 实验设计

The experiment system is built to answer separate questions, not to accumulate unrelated runs. Each group has a specific role in the argument.

Baseline exists to define the denominator of speedup. Without serial multi-chain timing, OpenMP speedup would have no internal reference.

Scaling exists to test whether speedup persists when thread count changes. It examines whether chain-level parallelism remains effective beyond a single 8-thread setting.

Tuning exists because default parameters do not fully solve harder instances. It separates algorithm speed from solution quality and identifies parameter regions worth validating.

Targeted enhancement exists to test whether quality gains survive independent seeds and larger budgets. This prevents the report from relying only on tuning-search best cases.

Policy comparison exists because the paper discusses epsilon-greedy and Softmax. The engineering implementation supports both, so the report must show how they behave under the implemented action/state definition.

Paper compare exists to place the project against the reference work. It is not a strict benchmark; it is a structured comparison of method, quality trend and runtime scale under different implementation conditions.

表 4：实验体系与科学问题。

| Experiment group | Question answered | Why it matters |
|---|---|---|
| Baseline | What is the serial multi-chain reference? | Enables speedup calculation. |
| Scaling | Does OpenMP remain effective across thread counts? | Tests parallel robustness. |
| Tuning | Can harder-instance Gap be reduced? | Separates quality from speed. |
| Targeted | Do tuned settings generalize under new seeds/budgets? | Avoids cherry-picking. |
| Policy | Does Softmax behave like epsilon-greedy in this implementation? | Tests QLSA design sensitivity. |
| Paper compare | How does engineering extension relate to the paper? | Establishes research context. |

# 6. 实验结果

## 6.1 OpenMP 性能结论

OpenMP multi-chain is the strongest result: SA reaches about 5.46x average speedup, and QLSA reaches about 4.98x average speedup across six TSPLIB95 instances.

![OpenMP speedup across TSPLIB95 instances](../../figures/final/fig02_openmp_speedup.png)

图 2：OpenMP 多实例 speedup。

表 5：OpenMP 主性能结果。

| Family | Average speedup | Average efficiency | Conclusion |
|---|---:|---:|---|
| SA | 5.46x | 68.28% | Stable chain-level acceleration. |
| QLSA | 4.98x | 62.29% | Stable but slightly more serial overhead. |

The data demonstrates that chain-level parallelism is effective because chains communicate only through final reduction. QLSA efficiency is lower because learning adds action selection and Q-table updates inside each chain. This is not a failure of parallelization; it is the expected serial overhead of the learning mechanism.

Why it matters: a 5x-level speedup on a stochastic local search establishes the project’s parallel algorithm contribution. The paper validates QLSA as an algorithmic idea; this result validates the same search family as a parallelizable engineering workload.

![OpenMP parallel efficiency across TSPLIB95 instances](../../figures/final/fig03_openmp_efficiency.png)

图 3：OpenMP 多实例 parallel efficiency。

## 6.2 默认参数质量结论

Default parameters solve the easier instances but leave measurable Gap on harder instances. This result is important because it prevents a misleading conclusion that speedup alone solves quality.

![Default-parameter Gap comparison](../../figures/final/fig04_default_gap.png)

图 4：默认参数下 SA 与 QLSA 的 Gap。

表 6：默认质量分层。

| Instance group | Result | Interpretation |
|---|---|---|
| berlin52, eil51, st70 | BKS reached | Default budget is sufficient. |
| eil76, rat99, eil101 | Gap remains | Quality requires tuning or more budget. |

The result indicates that OpenMP improves elapsed time without changing search behavior. Quality improvement must therefore come from parameter choice, QLSA policy or larger search budget, not from parallel execution itself.

## 6.3 QLSA 质量结论

rat99 QLSA reaches BKS=1211 while SA does not. This is the strongest algorithm-quality result in the project.

![Gap reduction after tuning and targeted enhancement](../../figures/final/fig05_tuning_curve.png)

图 5：调参与定向增强后的 Gap 改善。

表 7：定向增强关键结果。

| Instance | Family | Best length | Min Gap | Mean Gap | Mean ms |
|---|---|---:|---:|---:|---:|
| eil101 | SA | 629 | 0.000% | 0.445% | 1677.495 |
| eil101 | QLSA | 629 | 0.000% | 0.254% | 3348.545 |
| rat99 | SA | 1212 | 0.083% | 0.330% | 329.022 |
| rat99 | QLSA | 1211 | 0.000% | 0.099% | 1649.518 |

The data shows two distinct outcomes. On eil101, both SA and QLSA can reach BKS under targeted budget. On rat99, QLSA reaches BKS while SA remains one unit above BKS. This indicates that the learning-assisted strategy can improve search quality on a hard instance, but at a clear runtime cost.

Why it matters: this is the evidence that QLSA is not only an overhead layer. Under targeted settings, it can change the best achievable solution quality.

## 6.4 Policy comparison 结论

Softmax is not consistently superior in this engineering implementation. epsilon-greedy performs better on rat99, while the difference is smaller on eil76 and eil101.

![QLSA policy comparison](../../figures/final/fig06_policy_comparison.png)

图 6：QLSA epsilon-greedy 与 Softmax 对比。

表 8：policy comparison interpretation。

| Policy | Observed behavior | Meaning |
|---|---|---|
| epsilon-greedy | Lower mean Gap on rat99 | Direct exploration probability is more stable here. |
| Softmax | Competitive on some instances, weak on rat99 | Not equivalent to the paper’s candidate-leader Softmax. |

This demonstrates sensitivity to state/action design. The result should not be read as a contradiction of the paper, because the implemented Softmax acts on this project’s Q table actions rather than the paper’s full candidate leader set.

## 6.5 CUDA 定位结论

CUDA is fully implemented but not competitive on small instances.

![CUDA positioning on berlin52](../../figures/final/fig07_cuda_positioning.png)

图 7：berlin52 上 Serial、OpenMP 与 CUDA elapsed time。

表 9：CUDA result positioning。

| Dimension | Conclusion |
|---|---|
| Correctness | CUDA path builds, runs and emits CSV. |
| Performance | Small-instance runtime is worse than OpenMP. |
| Value | GPU backend exists for future larger-scale optimization. |

The main reason is workload granularity. One TSP chain on berlin52 does not provide enough work to offset GPU launch and scheduling overhead. This negative result is expected for small instances and does not weaken the OpenMP contribution.

# 7. 与论文对比

## 7.1 直接对比结论

The paper validates QLSA as an optimization method; this project validates the engineering scalability of SA/QLSA through C++ and OpenMP. Under the same TSPLIB instance names, the engineering implementation reports much lower elapsed time, but the comparison is not a strict benchmark because language and hardware differ. QLSA also shows a qualitatively similar behavior: it is most meaningful on harder instances rather than uniformly better everywhere.

![Runtime reference comparison with the paper](../../figures/final/fig08_paper_runtime_comparison.png)

图 8：论文 Table 8 与工程实现 OpenMP elapsed time 参考对比。

![Hard-instance mean Gap comparison](../../figures/final/fig09_paper_quality_comparison.png)

图 9：论文 hard-instance mean Gap 与工程实现调优/增强结果对比。

## 7.2 差异解释

表 10：论文与工程实现的差异。

| Aspect | Paper | Engineering implementation |
|---|---|---|
| Language | Python 3.11.5 | C++20 |
| Hardware | Xeon platform | i5-12600KF + RTX 4070 SUPER |
| Parallelism | Not the main backend | OpenMP + CUDA |
| QLSA mechanism | Candidate leader / SB diversity state | Action/state discretization |
| Runtime meaning | Computational time in paper environment | Elapsed time in local C++ environment |

The difference means that runtime cannot be interpreted as same-platform benchmark evidence. 换言之，论文与工程实现之间存在不同语言、不同硬件和不同实现，绝对时间不可直接比较。The valid claim is narrower and stronger: under a C++ multi-chain implementation, the paper’s search family becomes a parallelizable workload with stable OpenMP speedup.

## 7.3 意义

论文验证算法，本项目验证工程可扩展性。 The paper establishes that Q-learning variants can improve TSP search quality; the engineering system shows that the SA/QLSA family can be implemented with reproducible C++ infrastructure, accelerated by OpenMP, extended to CUDA, and evaluated through a complete CSV-to-report pipeline.

This distinction matters for course evaluation. The contribution is not a claim of replacing the paper, nor a claim of strict runtime superiority under identical conditions. The contribution is that a recent paper idea has been converted into an executable, parallel, reproducible experimental system.

# 8. 工程难度

The engineering contribution is a system, not a collection of scripts. The pipeline connects TSPLIB95 input, C++20 search kernels, OpenMP/CUDA backends, CSV output, statistical analysis, figures and final reports.

C++20 is used to control inner-loop cost. The parser must handle TSPLIB95 distance conventions rather than a single hand-written matrix. DistanceMatrix must support both CPU cache locality and GPU transfer. OpenMP must preserve independent chain states and deterministic seed derivation. CUDA must pass through CMake/Ninja/nvcc, device memory layout and host reduction.

Reproducibility is a core contribution. Every major claim is linked to existing CSV data or paper reference tables. This is why the pipeline matters: it converts stochastic runs into auditable evidence.

表 11：工程亮点与评分价值。

| Engineering element | Why it matters |
|---|---|
| C++20 core | Establishes a high-performance baseline. |
| TSPLIB95 parser | Enables standard-instance evaluation. |
| O(1) delta | Keeps local search inner loop efficient. |
| OpenMP backend | Produces the main speedup result. |
| CUDA backend | Provides a complete GPU extension path. |
| CSV-analysis-figures-report pipeline | Makes results reproducible and auditable. |

# 9. 局限性

CUDA remains under-optimized. The current backend is correct and complete enough for engineering validation, but not competitive on small TSPLIB95 instances.

QLSA remains sensitive to policy and parameter choices. The rat99 result is strong, but it does not generalize to a universal claim that QLSA dominates SA.

SB-QLSA is not fully reproduced. The implemented state/action discretization is an engineering variant, not the paper’s complete candidate-leader plus diversity-state mechanism.

Small-instance bias remains. berlin52 and similar instances are useful for correctness and timing, but they do not fully represent GPU-friendly workloads.

Benchmark non-equivalence remains. Different language, hardware and implementation details make paper runtime comparisons reference-level evidence, not strict benchmark evidence.

# 10. 总结

性能贡献：OpenMP multi-chain demonstrates stable parallel speedup, with SA at about 5.46x and QLSA at about 4.98x across six TSPLIB95 instances.

算法贡献：QLSA is implemented as a learning-assisted SA variant, and rat99 demonstrates its strongest quality case by reaching BKS=1211 while SA high-budget remains at 1212.

工程贡献：The project converts a recent QLSA-for-TSP idea into a reproducible C++20/OpenMP/CUDA experimental system with traceable CSV data, figures and submission-ready reports.
