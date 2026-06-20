# Paper Mechanism Alignment

本文档用于说明参考论文 *Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization* 与本项目实现之间的对应关系、可比指标和对比风险。结论口径为：本项目是“基于论文思想的工程复现与并行化扩展”，不是逐行复刻论文全部 SB-QLSA 机制。

## 1. 论文机制核查摘要

论文提出将 Q-learning 引入模拟退火（SA）框架。其 stateless QLSA 不再总是从当前解出发，而是在每次迭代中维护候选 leader 集合，并用 Q-learning 选择哪个候选解引导下一次 2-opt Metropolis 搜索。候选集合包括：

1. current solution；
2. global best solution；
3. randomly generated solution；
4. double-bridge perturbed solution。

论文还进一步提出 State-Based QLSA（SB-QLSA）。SB-QLSA 不只维护全局动作价值，而是根据当前解与历史最优解之间的 Hamming distance 定义 diversity state，使 Q table 变为 state-action value function。论文实验使用 epsilon-greedy 和 Softmax/Boltzmann 两种策略，并报告 Table 4 的质量统计、Table 8 的运行时间和 Table 9 的学习机制相对 SA 的额外开销。

论文未来工作中提到需要提高 scalability，并提到 richer state representations、dynamic candidate-set generation、hybridization with complementary metaheuristics 和 parallel implementations。因此，本项目实现 OpenMP/CUDA 多链并行可以视为对论文未来并行化方向的工程扩展。

## 2. 表 A：论文机制摘要

| 论文模块 | 论文做法 | 本项目是否实现 | 本项目实现方式 | 差异与说明 |
|---|---|---|---|---|
| TSPLIB95 输入 | 使用 TSPLIB95 benchmark instances | 已实现 | 自研 `.tsp` parser 和 DistanceMatrix | 本项目不依赖 Python TSPLIB95 library |
| Classical SA | 2-opt Metropolis + cooling schedule | 已实现 | C++20 SA，支持 seed、温度参数和 nearest-neighbor 初始化 | 与论文机制一致，参数不完全相同 |
| 2-opt Metropolis operator | 2-opt move 后用 Metropolis criterion 接受 | 已实现 | O(1) delta 计算，接受后反转区间 | 本项目更强调底层性能实现 |
| Stateless QLSA | Q-learning 选择 candidate leader | 部分实现 | 当前 QLSA 使用状态/动作离散化选择邻域策略 | 未完整实现论文 candidate leader 集合 |
| Candidate set | current, global best, random, double-bridge | 未完整实现 | 当前实现以邻域动作集合为主 | 后续可作为 paper-lite 变体扩展 |
| Double-bridge solution | 用于提高搜索多样性 | 未确认实现 | 当前代码审查阶段未发现完整 double-bridge candidate leader | 不应在报告中声称已复刻该机制 |
| epsilon-greedy | 按 ε 随机探索，否则选最大 Q | 已实现 | CLI 支持 `--policy epsilon-greedy` | 默认实验主要使用该策略 |
| Softmax/Boltzmann | 按 Q-value 和温度分布采样 | 已实现 | CLI 支持 `--policy softmax` | 与论文 Softmax 温度耦合细节可能不同 |
| SB-QLSA diversity state | 用 Hamming distance 定义低/高 diversity state | 部分实现 | 当前 QLSA 有状态离散化思想 | 不完全等同论文 SB-QLSA |
| Table 4 质量统计 | Best/Worst/Mean/Std/Gap | 部分对齐 | 本项目统计 best/min/mean Gap 和时间 | 指标口径接近，但运行次数和实现环境不同 |
| Table 8 运行时间 | 报告 SA/QLSA/SB-QLSA 秒级时间 | 已纳入参考对比 | `results/paper_table8_runtime.csv` | 非同硬件、非同语言公平 benchmark |
| Table 9 开销分析 | Q-learning variants 相对 SA 的 runtime overhead | 已在报告中讨论 | 用本项目 SA/QLSA 时间差和效率差解释 | 未逐项复现论文 Table 9 |
| Future parallel implementations | 论文未来工作提到 parallel implementations | 已扩展 | OpenMP multi-chain 与 CUDA backend | 本项目主要贡献之一 |

## 3. 表 B：论文实验指标与本项目指标对照

| 指标 | 论文定义/使用方式 | 本项目定义/使用方式 | 是否可直接对比 |
|---|---|---|---|
| Best | 10 次运行中的最短路径长度 | repeat 运行中的最短路径长度 | 可参考对比，但 repeat 次数不同 |
| Worst/Max | 10 次运行中的最差路径长度 | 大部分 summary 未保留该字段 | 不直接对比 |
| Mean | 10 次运行路径长度平均值 | 部分实验统计 best length mean 或 Gap mean | 可参考，需说明口径 |
| Std | 路径长度或时间标准差 | 部分 summary 记录 elapsed std | 不完全对齐 |
| Gap | `(Mean - BKS) / BKS` 或相关平均偏差 | default 多用 best Gap，tuned/targeted 同时记录 min/mean Gap | 必须说明 min Gap 与 mean Gap 区别 |
| Computational time | Python 实现秒级时间 | C++/OpenMP/CUDA 毫秒级 elapsed time | 仅作参考对比 |
| Runtime overhead | Q-learning variants 相对 SA 的额外时间 | 可通过 QLSA/SA 时间差、speedup/efficiency 分析 | 可定性对照 |
| Speedup | 论文未重点报告并行 speedup | `T_serial / T_OpenMP` | 本项目内部指标 |
| Parallel efficiency | 论文未报告 | `speedup / threads * 100%` | 本项目内部指标 |
| Reproducibility | 论文进行多次独立运行 | 本项目所有随机算法支持 seed | 本项目更强调工程复现 |

## 4. 表 C：论文对比风险说明

| 风险 | 原因 | 报告中如何规避误导 |
|---|---|---|
| 把运行时间对比写成公平 benchmark | 论文是 Python/Xeon，本项目是 C++/OpenMP/CUDA/i5+RTX | 明确称为“参考对比”，不称为同平台公平 benchmark |
| 夸大 QLSA 优势 | 本项目中 QLSA 只在部分实例/参数下优于 SA | 写成“rat99 上的明确质量案例”，不写“总是优于 SA” |
| 声称复刻 SB-QLSA | 当前代码没有完整 candidate-leader + diversity-state 机制 | 写明“部分思想实现”，不写“完整复刻” |
| 混淆 min Gap 与 mean Gap | targeted high-budget 可达到 BKS，但平均 Gap 仍可能大于 0 | 同时报告 min Gap、mean Gap 和运行时间 |
| CUDA 结果被误读 | CUDA 已构建运行，但小实例时间不优于 OpenMP | 将 CUDA 定位为工程扩展和后续优化方向 |
| tuning search best 被误用 | 调参搜索结果可能有选择偏差 | 最终报告优先引用 Step 6B 独立 seed 验证和 Step 6C repeat=5 |
| 论文机制差异被忽略 | 本项目 QLSA 与论文 stateless/SB-QLSA 细节不同 | 在论文对比节单独声明实现差异 |

## 5. 对最终报告的建议用语

建议使用：

- “本项目基于论文 QLSA 思想完成 C++20 工程复现与并行化扩展。”
- “论文运行时间对比为参考对比，不是同平台公平 benchmark。”
- “本项目 QLSA 当前实现吸收了状态/动作离散化思想，但未完整实现论文 SB-QLSA 的 candidate-leader + diversity-state 机制。”
- “OpenMP multi-chain 是本项目最稳定的性能提升来源。”
- “CUDA backend 已完成工程验证，但当前小规模实例上不作为主要加速证据。”

应避免使用：

- “全面超过论文。”
- “CUDA 比 OpenMP 更快。”
- “QLSA 总是优于 SA。”
- “完全复刻论文 SB-QLSA。”
