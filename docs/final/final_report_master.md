# 1. 问题背景与课程目标映射

旅行推销员问题（Traveling Salesman Problem, TSP）要求在给定城市集合和距离矩阵的条件下，寻找访问每个城市一次并回到起点的最短回路。该问题属于经典 NP-hard 组合优化问题，精确算法在规模增加后难以直接承担大规模实验，因此模拟退火（Simulated Annealing, SA）和强化学习辅助启发式搜索成为合理选择。

TSP 适合并行化的原因不在于单次 2-opt move 本身容易拆分，而在于随机搜索可以从多个独立初始解或多个独立搜索链出发。每条搜索链拥有独立随机数、当前路径、最优路径和接受统计，链间只共享只读距离矩阵，最终对各链最优解进行归约。该结构为 OpenMP 多线程和 CUDA 后端提供了清晰并行边界。

表 1：课程评分点与报告证据映射。

| 课程目标 | 对应完成内容 | 报告证据 |
|---|---|---|
| 完成情况 | SA、QLSA、OpenMP、CUDA、TSPLIB95 parser、实验脚本均完成 | 第 3、4、5 节 |
| 技术难度 | C++20 底层实现、O(1) 2-opt delta、CUDA 编译链、自动化结果流水线 | 第 3、4、8 节 |
| 并行算法性能 | 使用 speedup、parallel efficiency、elapsed time 评价 OpenMP 多链并行 | 第 6 节 |
| 与近期论文对比 | 引入 2026 年 QLSA for TSP 论文机制、Table 8 时间和 hard-instance quality | 第 2、7 节 |
| 报告质量 | 图表编号、结果索引、course/public 版本、复现命令、已知限制 | `docs/final/` 与 `results/final/` |

# 2. 论文方法拆解

## 2.1 SA

参考论文中的 classical SA 使用 2-opt 邻域和 Metropolis 接受准则。给定当前解和候选解，若候选路径长度更短则直接接受；若更长，则按温度控制的概率接受。该机制允许高温阶段跳出局部最优，低温阶段逐渐收敛。工程实现侧采用相同 SA 主干，并加入 O(1) 2-opt delta，以避免每次 move 后完整重算 tour length。

## 2.2 QLSA

论文 QLSA 的核心思想是用 Q-Learning 辅助搜索策略选择。论文侧的 stateless QLSA 维护 candidate leader 集合，包括 current solution、global best solution、random solution 和 double-bridge solution，再通过 epsilon-greedy 或 Softmax/Boltzmann 策略选择 leader。本工程侧实现了 Q 表、状态离散化、动作选择、epsilon-greedy、Softmax 和奖励更新，但 action 定义为不同 2-opt 邻域策略，不是论文 candidate leader 机制的一一对应实现。

## 2.3 SB-QLSA

论文进一步提出 State-Based QLSA（SB-QLSA），使用当前解与最优解之间的 diversity state 扩展 Q 表。该机制试图区分探索状态与强化状态，使 Q 值与搜索多样性相关。工程侧采用 delta/state discretization 表示搜索状态，吸收了“状态影响策略选择”的思想，但不是论文 SB-QLSA 的等价实现。因此，报告中所有 QLSA 结果均表述为“基于论文思想的工程变体”。

## 2.4 论文实验结构

论文实验使用 TSPLIB95 实例，统计 Best、Mean、Std、Gap 和 computational time。Table 4 等 hard-instance quality 表格说明 Q-learning variants 相比 Paper-SA 在 eil76、rat99、eil101 上降低 mean Gap；Table 8 给出各算法运行时间；Table 9 分析 Q-learning variants 相对 SA 的额外开销。本工程侧整理了论文 Table 8 和 hard-instance quality 数据，分别存放在 `results/reference/paper_table8_runtime.csv` 和 `results/reference/paper_hard_instance_quality.csv`，用于参考对比。

表 2：论文方法与工程实现的关系。

| 论文内容 | 工程实现 | 对应关系 | 说明 |
|---|---|---|---|
| SA + 2-opt Metropolis | SA baseline | 对应 | 接受准则和邻域结构一致，底层增加 O(1) delta。 |
| QLSA candidate leader | QLSA action selection | 部分对应 | 保留 Q-learning 选择思想，action 定义不同。 |
| epsilon-greedy / Softmax | CLI policy 参数 | 对应 | 两种策略均可运行和比较。 |
| SB-QLSA diversity state | state discretization | 部分对应 | 状态离散化存在，但不是论文 diversity state 的完整工程复刻。 |
| Python serial implementation | C++20 + OpenMP/CUDA | 扩展 | 主要贡献在工程化和并行化。 |

# 3. 系统设计

系统设计的核心目标是让算法、数据、并行后端和实验分析解耦。C++20 被选为核心实现语言，是因为内层搜索循环包含百万级 2-opt move，解释器开销会直接影响 elapsed time；同时 C++ 的连续内存布局、内联函数和编译优化更适合构建性能基线。

TSPLIB95 被选为数据来源，是因为参考论文使用同一标准库，且 BKS 可查，便于使用 Gap 评价解质量。解析器支持坐标型和显式矩阵型实例，避免只针对 berlin52 这类简单坐标实例做特例实现。

DistanceMatrix 使用一维连续数组，而不是 `vector<vector<int>>`。该设计减少间接寻址，提高缓存局部性，并且 `raw()` 可直接拷贝到 CUDA 全局内存。Tour 模块只保存城市排列，路径合法性、nearest-neighbor 初始化、random 初始化和 2-opt delta 都围绕该表示实现。

2-opt delta 采用常数时间更新。反转区间 `[i,k]` 时，只替换两条边：旧边为 `a-b`、`c-d`，新边为 `a-c`、`b-d`。增量为：

$$
\Delta=D_{a,c}+D_{b,d}-D_{a,b}-D_{c,d}
$$

CLI + CSV pipeline 被选为实验接口，是因为课程实验需要 repeat、seed、threads、chains、policy、CUDA block size 等参数组合。程序统一输出 CSV 行，Python 脚本负责收集、统计、画图和写报告，避免手工复制终端结果导致不可追溯。

![System architecture and data flow](../../figures/final/fig01_architecture_pipeline.png)

图 1：系统总体架构与数据流。

# 4. 并行设计

## 4.1 为什么 SA 可并行

SA 单链内部是串行 Markov 过程，当前 tour 会影响下一次 move 和接受判断。因此，单链内部不适合作为首要并行目标。可并行性来自多链搜索：不同 chain 使用不同 seed 和独立状态，同时探索解空间，最后只需要归约全局 best tour。

## 4.2 为什么选 OpenMP chain-level

OpenMP chain-level parallel for 让每个线程执行一条或多条完整搜索链。线程私有数据包括 RNG、tour、current length、best tour、accepted/improved counters；QLSA 还包括独立 Q table。共享数据只有只读 DistanceMatrix。该设计几乎不需要锁，结果写入 `chain_results[chain_id]` 后再串行归约。

## 4.3 为什么不做 move-level

move-level 并行看似可以同时评价多个候选 2-opt move，但会引入三类问题：候选 move 对当前 tour 的依赖、接受后 tour 变更导致其他候选失效、随机过程复现困难。与之相比，chain-level 并行牺牲了单链内部并行度，但换来低同步成本和清晰统计口径，更适合课程实验中的 speedup 和 parallel efficiency 分析。

## 4.4 CUDA 为什么存在但不是主结论

CUDA 后端用于验证 GPU 多链搜索路径，包括距离矩阵拷贝、kernel 执行和 host 端归约。该路径提高了工程难度和扩展性，但在 berlin52 这类小规模实例上，kernel 启动、调度和每链工作量不足带来的开销占比较高。实验结果不支持将 CUDA 写成主要加速结论，因此报告将其定位为工程扩展和后续优化方向。

表 3：并行粒度取舍。

| 粒度 | 设计收益 | 主要风险 | 最终定位 |
|---|---|---|---|
| move-level | 理论上并行评价候选 move | 同步复杂、随机过程难复现 | 未作为主方案 |
| chain-level OpenMP | 链间独立、共享只读矩阵、归约简单 | 单链内部未加速 | 主性能方案 |
| CUDA multi-chain | GPU 路径完整，便于后续扩展 | 小实例开销高 | 工程验证 |

# 5. 实验设计

实验体系分为六组，而不是简单堆叠 CSV 文件。

表 4：实验体系。

| 实验组 | 目的 | 关键变量 | 结论用途 |
|---|---|---|---|
| baseline | 建立 serial multi-chain 基准 | algorithm、instance、seed | speedup 分母 |
| scaling | 测试 OpenMP threads/chains 扩展性 | threads、chains | 并行效率分析 |
| tuning | 搜索 SA/QLSA 参数 | t0、tf、alpha、gamma、epsilon | 发现 harder instances 的有效参数 |
| targeted | 固定较优参数后增加预算 | chains、iterations | 验证质量能否稳定接近 BKS |
| policy | 比较 epsilon-greedy 与 Softmax | policy | 说明本实现策略差异 |
| paper compare | 引入论文表格数据 | Paper-SA、Paper-QLSA、工程实现 | 形成参考对比 |

默认参数实验使用 `iterations=1000000`、`chains=32`、`repeat=3`、`threads=8`、`init=nn`。调优验证使用独立 seed，从 101 开始，避免只报告参数搜索中的最好结果。评价指标包括 best length、Gap、elapsed time、speedup 和 parallel efficiency：

$$
Gap=\frac{best\_length-BKS}{BKS}\times 100\%
$$

$$
Speedup=\frac{T_{serial}}{T_{parallel}}
$$

$$
Efficiency=\frac{Speedup}{thread\_count}\times 100\%
$$

# 6. 实验结果

## 6.1 默认参数 OpenMP 加速

![OpenMP speedup across TSPLIB95 instances](../../figures/final/fig02_openmp_speedup.png)

图 2：OpenMP 多实例 speedup。

表 5：默认参数 OpenMP 加速摘要。

| Family | Average speedup | Average efficiency | 主要说明 |
|---|---:|---:|---|
| SA | 5.46x | 68.28% | 多链之间通信少，OpenMP 加速稳定。 |
| QLSA | 4.98x | 62.29% | Q 表更新和策略选择增加链内开销。 |

分析：SA 和 QLSA 在 6 个实例上均获得约 5x 加速，说明 chain-level 粗粒度并行能够有效利用 8 线程。QLSA efficiency 低于 SA，并不是并行设计失效，而是 QLSA 单链内部的 action selection、state update 和 Q table update 提高了串行段比例。

结论：OpenMP multi-chain 是本工程最可靠的性能贡献。

![OpenMP parallel efficiency across TSPLIB95 instances](../../figures/final/fig03_openmp_efficiency.png)

图 3：OpenMP 多实例 parallel efficiency。

## 6.2 默认参数解质量

![Default-parameter Gap comparison](../../figures/final/fig04_default_gap.png)

图 4：默认参数下 SA 与 QLSA 的 Gap。

表 6：默认参数质量现象。

| Instance group | 观察结果 | 含义 |
|---|---|---|
| berlin52、eil51、st70 | SA/QLSA 达到 BKS | 默认参数足以解决较容易实例。 |
| eil76、rat99、eil101 | 仍存在 Gap | 加速不等于质量自动提升。 |

分析：默认参数的主要价值是建立可比性能基线。OpenMP 保持搜索逻辑不变，因此质量结果与 serial multi-chain 具有同一统计口径。harder instances 的 Gap 暴露出参数和搜索预算仍需优化。

结论：性能实验和质量实验必须分开解释，不能把并行加速直接解释为更优解。

## 6.3 参数调优与定向增强

![Gap reduction after tuning and targeted enhancement](../../figures/final/fig05_tuning_curve.png)

图 5：调参与定向增强后的 Gap 改善。

表 7：定向增强关键结果。

| Instance | Family | Best length | Min Gap | Mean Gap | Mean ms |
|---|---|---:|---:|---:|---:|
| eil101 | SA | 629 | 0.000% | 0.445% | 1677.495 |
| eil101 | QLSA | 629 | 0.000% | 0.254% | 3348.545 |
| rat99 | SA | 1212 | 0.083% | 0.330% | 329.022 |
| rat99 | QLSA | 1211 | 0.000% | 0.099% | 1649.518 |

分析：eil101 上 SA 与 QLSA 都能在高预算定向实验中达到 BKS。rat99 上 QLSA high-budget 达到 BKS=1211，而 SA high-budget 最好为 1212。该结果提供了 QLSA 在特定 harder instance 上改善解质量的证据，但运行时间也明显增加。

结论：QLSA 的优势应表述为“部分实例和特定预算下的质量收益”，而不是普遍优于 SA。

## 6.4 Policy comparison

![QLSA policy comparison](../../figures/final/fig06_policy_comparison.png)

图 6：QLSA epsilon-greedy 与 Softmax 对比。

表 8：policy comparison 解释。

| Policy | 观察 | 解释 |
|---|---|---|
| epsilon-greedy | rat99 上 mean Gap 更低 | 更直接的探索概率在该实现中更稳定。 |
| Softmax | eil76、eil101 差距较小，但 rat99 表现较差 | 本实现 Softmax 作用于动作 Q 值，不等同论文 candidate leader Softmax。 |

分析：policy comparison 是工程实现内部策略比较，不是论文 Softmax 机制的严格复现实验。该结果提示 QLSA 对策略和状态设计敏感。

结论：最终报告应保留 Softmax 支持与比较，但不能用该实验否定或复现论文 Softmax 结论。

## 6.5 CUDA 定位

![CUDA positioning on berlin52](../../figures/final/fig07_cuda_positioning.png)

图 7：berlin52 上 Serial、OpenMP 与 CUDA elapsed time。

表 9：CUDA 定位。

| 维度 | 结论 |
|---|---|
| 正确性 | CUDA 路径可构建、可运行、可输出 CSV。 |
| 性能 | 小规模实例上不优于 OpenMP。 |
| 工程价值 | 完成 GPU backend、数据拷贝和归约路径。 |

分析：CUDA 当前瓶颈来自 kernel 启动、访存、调度和每链工作量不足。若后续把 block 内线程用于批量候选 move 评价，GPU 才可能在更大实例上体现优势。

结论：CUDA 是工程扩展，不是本报告的主要性能证据。

# 7. 与论文对比

## 7.1 方法对比

表 10：论文侧与工程侧方法对比。

| 对比项 | 论文侧 | 工程侧 |
|---|---|---|
| 实现语言 | Python 3.11.5 | C++20 |
| 算法 | SA、QLSA、SB-QLSA | SA、QLSA、多链后端 |
| QLSA action | candidate leader | 2-opt 邻域动作 |
| 状态设计 | diversity state | delta/state discretization |
| 并行化 | 未作为主要实现 | OpenMP + CUDA |

## 7.2 时间对比

![Runtime reference comparison with the paper](../../figures/final/fig08_paper_runtime_comparison.png)

图 8：论文 Table 8 与工程实现 OpenMP elapsed time 参考对比。

表 11：时间对比口径。

| 项目 | 论文 | 工程实现 |
|---|---|---|
| 语言 | Python | C++20 |
| 硬件 | Xeon 平台 | i5-12600KF |
| 并行 | 未作为主要后端 | OpenMP 多链 |
| 结论口径 | computational time | elapsed time + speedup |

分析：图 8 不是严格 benchmark，而是参考对比。不同硬件、不同语言和不同实现会显著影响 elapsed time，绝对时间不可直接比较。该图只能说明工程化 C++ 和 OpenMP 多链并行在共同实例上具有实际运行效率优势。

## 7.3 质量对比

![Hard-instance mean Gap comparison](../../figures/final/fig09_paper_quality_comparison.png)

图 9：论文 hard-instance mean Gap 与工程实现调优/增强结果对比。

分析：论文中 Q-learning variants 明显降低 Paper-SA 的 mean Gap。工程实现通过调参与定向增强，在 eil76、rat99、eil101 上进一步接近 BKS，其中 rat99 QLSA high-budget 达到 BKS。由于机制不完全相同，该对比只说明工程实现具备竞争力和并行扩展收益。

## 7.4 风险说明

表 12：论文对比风险。

| 风险 | 规避方式 |
|---|---|
| 时间不是严格 benchmark | 明确标注为参考对比。 |
| QLSA 机制不完全一致 | 单独列出 candidate leader 与 action discretization 差异。 |
| SB-QLSA 未等价实现 | 使用“基于思想的工程变体”表述。 |
| CUDA 不作为主性能结论 | 仅展示工程定位实验。 |

# 8. 工程难度

工程难度体现在系统闭环，而不是单个 `parallel for`。C++20 算法核心需要处理 TSPLIB95 parser、距离类型、显式矩阵、Tour 合法性、RNG 可复现、O(1) delta 和断言校验。OpenMP 后端需要保持 chain result、seed derivation 和 serial baseline 的输出一致。CUDA 后端需要处理 CMake/Ninja/nvcc 编译链、GPU 数据布局和 host 归约。

实验自动化同样是工程难度的一部分。批量脚本覆盖 default、tuning、targeted、scaling、policy 和 paper compare，所有结论均从 raw CSV 或 reference CSV 派生。`results/final/RESULTS_INDEX.md` 说明每类结果来源，`docs/final/REPORT_MANIFEST.md` 说明报告入口和废弃版本，course/public 双版本区分了课程提交与公开仓库展示。

表 13：工程完成质量。

| 工程模块 | 技术价值 |
|---|---|
| C++20 核心 | 降低解释器开销，建立高性能 baseline。 |
| TSPLIB95 parser | 支持标准实例与 BKS/GAP 实验体系。 |
| O(1) 2-opt delta | 保证百万级迭代的内层效率。 |
| OpenMP backend | 提供稳定约 5x 的主要 speedup。 |
| CUDA backend | 完成 GPU 工程路径和后续扩展基础。 |
| CSV pipeline | 保证结果可追溯，避免手工复制。 |

# 9. 局限性

1. CUDA 后端仍未充分优化，小规模实例上不优于 OpenMP。
2. QLSA 对参数、policy 和状态设计敏感，默认参数并不稳定。
3. 实验实例以中小规模 TSPLIB95 为主，大规模实例仍需补充。
4. 工程实现不是论文 SB-QLSA 的等价实现，candidate leader 与 diversity state 机制未逐项落地。
5. 与论文时间表的比较只具有参考意义，不能作为跨平台严格 benchmark。
6. 预算扫描不能表述为真实逐迭代 trace。

# 10. 总结

性能贡献：OpenMP chain-level multi-chain 是主要并行成果，默认参数多实例实验中 SA 平均 speedup 约 5.46x，QLSA 平均 speedup 约 4.98x。

算法贡献：SA baseline 和 QLSA 工程变体均已实现；调参与定向增强使 harder instances 更接近 BKS，rat99 上 QLSA high-budget 达到 BKS=1211，而 SA high-budget 最好为 1212。

工程贡献：C++20 核心、TSPLIB95 parser、OpenMP/CUDA 双后端、自动化实验、论文参考对比、图表和提交级文档形成完整闭环，能够支撑课程对完成情况、技术难度、并行性能和报告质量的评分要求。
