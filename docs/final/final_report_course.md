# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化

## 摘要

旅行推销员问题（Traveling Salesman Problem, TSP）是典型 NP-hard 组合优化问题，也是检验启发式搜索与并行优化方法的经典对象。本次大作业参考 2026 年论文 *Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization*，围绕其中 SA、QLSA 和 SB-QLSA 的思想，完成 C++20 工程化实现，并进一步实现 OpenMP 多链并行和 CUDA 后端工程验证。

实现层面，系统包含 TSPLIB95 parser、连续一维 DistanceMatrix、Tour 与 2-opt delta、串行 SA、Q-Learning 辅助 SA、OpenMP multi-chain backend、CUDA backend、命令行参数系统和实验分析脚本。OpenMP 是本报告的主性能结论：默认参数多实例实验中，SA OpenMP 平均 speedup 约 5.46x，QLSA OpenMP 平均 speedup 约 4.98x，并且没有改变搜索逻辑和结果统计口径。

解质量方面，默认参数下 berlin52、eil51、st70 达到 BKS，eil76、rat99、eil101 仍存在一定 Gap。因此后续进行参数调优、独立 seed 验证和定向增强实验。定向增强结果中，eil101 的 SA 和 QLSA 均达到 BKS=629；rat99 上 QLSA high-budget 达到 BKS=1211，而 SA high-budget 最好为 1212，说明 QLSA 在部分 harder instances 上具备质量收益，但不能外推为所有实例均由 QLSA 占优。

CUDA 后端已完成真实构建和运行验证，但在 berlin52 等小规模实例上不优于 OpenMP。报告中将 CUDA 定位为高工程难度扩展与后续优化方向，而非主要加速证据。与参考论文的时间和质量对比均明确标注为参考对比：论文环境为 Python 与 Xeon 平台，本实现为 Windows、C++20、OpenMP/CUDA，绝对时间不能视作同一平台下的严格性能比较。

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 课程名称 | 并行算法 |
| 项目题目 | 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化 |
| 团队人数 | 1 人 |
| 团队成员 | 陈乐浚 |
| 学号 | 22361054 |
| 学院/专业 | 中山大学计算机学院 / 信息与计算科学 |

## 2. 预期目标与实际完成情况

这一节对应课程小组报告的完成情况说明。选题报告的原始目标是复现近期论文中的 QLSA 思想，并在 TSP 场景下进行并行化优化。实际完成内容不仅包括算法本身，还包括可复现实验系统、结果分析脚本、论文参考对比和提交级文档整理。

表 1：选题报告预期目标与实际完成情况。

| 预期目标 | 实际完成情况 | 说明 |
|---|---|---|
| 读取 TSPLIB95 标准实例 | 完成 | 支持坐标型与显式矩阵型 `.tsp` 文件。 |
| 实现串行 SA baseline | 完成 | 使用 2-opt、Metropolis 准则和指数退火。 |
| 实现 QLSA | 完成 | 支持状态/动作离散化、Q 表更新、epsilon-greedy 与 Softmax。 |
| 实现 State-Based QLSA | 部分完成 | 采用状态/动作离散化思想，但不等同于论文完整 SB-QLSA candidate-leader 与 diversity-state 机制。 |
| OpenMP 并行优化 | 完成 | 使用 chain-level multi-chain 并行，是主要性能提升来源。 |
| CUDA 并行扩展 | 完成工程验证 | 可以构建和运行，但小实例上不作为主要加速结论。 |
| 实验自动化 | 完成 | 包含批量运行、调参、验证、图表和索引脚本。 |
| 与近期论文对比 | 完成 | 纳入论文 Table 8 运行时间和 hard-instance quality 数据。 |
| 个人报告 | 完成 | 单人团队，个人工作说明见附录 A。 |

表 2：课程评分点与本项目支撑材料。

| 课程评分点 | 支撑材料 |
|---|---|
| 完成情况 | SA、QLSA、OpenMP、CUDA、TSPLIB95 parser、CLI 和实验脚本均已实现。 |
| 技术难度 | C++20 底层实现、O(1) 2-opt delta、CUDA 编译链、OpenMP/CUDA 双后端、自动化实验流水线。 |
| 并行算法性能 | 使用 speedup、parallel efficiency、elapsed time、Gap 等指标进行分析。 |
| 近期论文对比 | 参考 2026 年 QLSA for TSP 论文，并整理论文运行时间和质量表格。 |
| 报告质量 | course/public 双报告、图表、结果索引、复现命令和已知限制说明。 |

## 3. 参考论文方法与本项目定位

参考论文提出了将 Q-Learning 与模拟退火结合求解 TSP 的思路。论文中的 classical SA 使用 2-opt Metropolis 搜索；stateless QLSA 不仅从 current solution 出发，还维护 candidate leader 集合，包括 current solution、global best solution、random solution 和 double-bridge solution，再由 Q-learning 策略选择候选 leader；State-Based QLSA 进一步使用当前解与最优解之间的 diversity state 扩展 Q 表，使 action value 与搜索状态相关。

论文实验使用 TSPLIB95，统计 Best、Worst、Mean、Std、Gap 和 computational time。其中 Table 8 给出各算法运行时间，hard-instance quality 表格展示了 QLSA/SB-QLSA 在 eil76、rat99、eil101 等较难实例上相对于 classical SA 的质量改进。本报告将这些数据整理到 `results/reference/`，用于参考对比。

表 3：论文机制与本实现对应关系。

| 论文机制 | 本项目实现 | 是否完全对应 | 说明 |
|---|---|---|---|
| TSPLIB95 实例 | 自实现 parser | 基本对应 | 支持主要距离类型和显式矩阵格式。 |
| Classical SA | SA + 2-opt Metropolis | 对应 | 使用 O(1) delta 更新路径长度。 |
| QLSA candidate leader | QLSA 动作/状态离散化 | 不完全对应 | 本实现选择不同 2-opt 邻域动作，不是完整 candidate leader。 |
| epsilon-greedy | CLI 支持 | 对应 | 可通过 `--policy epsilon-greedy` 运行。 |
| Softmax | CLI 支持 | 部分对应 | 本实现的 Softmax 作用于当前 Q 表动作，不等同论文完整机制。 |
| SB-QLSA diversity state | 状态离散化 | 部分对应 | 未声明完整复刻论文 SB-QLSA。 |
| 论文未来 parallel implementations | OpenMP/CUDA 后端 | 扩展 | 本项目将多链搜索工程化为并行后端。 |

因此，本实现的定位是：基于论文思想进行 C++ 工程复现与并行化扩展，而不是逐行复刻论文全部算法细节。时间对比涉及不同硬件、不同语言和不同实现，只能作为参考。

## 4. 方案设计

### 4.1 总体架构

![System architecture and data flow](../../figures/final/fig01_architecture_pipeline.png)

图 1：系统总体架构与数据流。

系统从 TSPLIB95 `.tsp` 文件开始，先由 parser 转换为统一 Instance，再构造连续一维 DistanceMatrix。SA 和 QLSA 共用底层 Tour、RNG、Timer 和 metrics 模块，执行后由 CLI 输出标准 CSV。后续 Python 脚本读取 raw CSV，生成 summary、图表、Markdown 分析和最终报告。这样做的目的不是堆叠脚本，而是把算法实现、实验运行和结果分析分离，保证每条结论能追溯到 CSV 数据。

### 4.2 数据结构设计

DistanceMatrix 使用连续一维数组保存 $$n \times n$$ 距离矩阵。该设计比 `vector<vector<int>>` 更适合缓存访问，也方便将 `raw()` 数据直接传给 CUDA 后端。TSP tour 使用城市排列表示，路径合法性通过排列检查验证。初始化支持 identity、random permutation 和 nearest-neighbor，其中默认实验采用 nearest-neighbor，以减少低质量随机初始解带来的噪声。

### 4.3 SA / QLSA 算法流程

SA 每次随机选择合法 2-opt move，计算增量 $$\Delta$$，若 $$\Delta \le 0$$ 则接受，否则以 Metropolis 概率接受：

$$
P(\Delta,T)=\exp(-\Delta/T)
$$

温度采用指数退火，避免每轮调用 `pow`。QLSA 在 SA 框架上加入 Q 表、动作选择和奖励更新。Q 更新公式为：

$$
Q(s,a) \leftarrow Q(s,a)+\alpha [r+\gamma \max Q(s',a')-Q(s,a)]
$$

奖励以路径长度减少为正向信号。QLSA 的收益来自对搜索策略的自适应选择，但额外开销包括 action selection、状态更新和 Q 表更新。

### 4.4 O(1) 2-opt delta

2-opt 反转区间 $$[i,k]$$ 时，只改变两条边。若旧边为 $$a-b$$、$$c-d$$，新边为 $$a-c$$、$$b-d$$，则：

$$
\Delta=D_{a,c}+D_{b,d}-D_{a,b}-D_{c,d}
$$

因此每次 move 不需要完整重算路径长度。对百万级迭代和多链实验而言，这是串行与并行版本都必须具备的底层优化。

### 4.5 实验流水线设计

实验流水线采用“可执行程序输出 CSV 行，Python 脚本收集与分析”的方式。所有批量实验均保存 raw CSV、summary CSV 和日志。最终图表由 `scripts/make_report_figures.py` 从已有 CSV 生成，缺失数据只记录到 `MISSING_FIGURES.md`，不生成虚假图。

## 5. 并行方案设计

### 5.1 并行机会分析

SA/QLSA 单条链内部存在强时序依赖：下一步 tour、长度和温度依赖当前状态。如果直接对 move-level 做并行，会产生大量同步，并且随机过程难以复现。相反，多初始解、多搜索链之间天然独立，只在最后需要归约 best tour，因此适合作为主要并行粒度。

### 5.2 OpenMP 多链并行

OpenMP 后端采用 chain-level parallel for。每条 chain 拥有独立 seed、tour、current length、best tour、accepted/improved counters；QLSA 链还拥有独立 Q table。DistanceMatrix 只读共享，不需要锁。每条链将结果写入 `chain_results[chain_id]`，并行区结束后在 host 端串行归约全局最优。

该方案能取得约 5x 加速，主要原因是链之间通信极少，锁竞争低，且每条链工作量较大。相比 move-level 并行，chain-level 并行更稳定、复现性更好，也更符合课程实验中 speedup 和 efficiency 的统计方式。

### 5.3 CUDA 后端

CUDA 后端完成了多链搜索的 GPU 工程验证。DistanceMatrix 拷贝到 GPU，全局最优在 host 端归约。当前 CUDA 实现能在 smoke test 和 berlin52 上运行，但小规模 TSPLIB95 实例中，kernel 启动、调度、访存和每链工作量不足带来的开销较明显，因此不作为主要加速结论。

### 5.4 并行粒度选择理由

表 4：chain-level 与 move-level 并行粒度比较。

| 粒度 | 优点 | 局限 | 本项目选择 |
|---|---|---|---|
| move-level | 理论上可并行评价多个候选 move | 同步复杂，随机过程难复现，链内依赖强 | 未作为主方案 |
| chain-level | 链间独立，通信少，易复现，适合 OpenMP | 单链内部未加速 | 作为主方案 |
| CUDA block-level | 可扩展到大量链 | 小实例每链工作量不足 | 工程扩展 |

## 6. 实施过程与解决的问题

表 5：实施问题、解决方案与影响。

| 问题 | 现象 | 解决方案 | 对结果影响 |
|---|---|---|---|
| TSPLIB 下载受 WSL/代理影响 | 官方链接下载失败或 localhost 代理不可用 | 支持手动放置 `.tsp` 文件，并在 data README 说明 | 保证实验数据可复现但不依赖单一下载源 |
| Visual Studio CUDA toolset 失败 | CMake 无法识别 CUDA toolset | 改用 Ninja + nvcc 构建 | CUDA kernel 成功真实编译 |
| Python Store alias | `python` 无输出或失败 | 使用 Windows `py` 启动器 | 实验脚本和检查脚本可稳定运行 |
| QLSA 默认参数不稳定 | harder instances Gap 偏高 | 执行 Step 6A/6B/6C 调参和独立验证 | 提高 eil76、rat99、eil101 解质量 |
| CUDA 小实例慢 | berlin52 上 CUDA elapsed time 高于 OpenMP | 报告中定位为工程扩展 | 避免夸大 CUDA 性能 |
| 论文机制与实现机制差异 | 本实现不是完整 SB-QLSA | 在报告中单独说明差异 | 降低对比误导风险 |
| 结果和图表过多 | 文档结构混乱 | 整理为 final/raw/summary/reference/archive | 提高提交可读性 |

## 7. 实验设计

### 7.1 数据集与环境

实验使用 TSPLIB95 实例，默认参数多实例实验包含 berlin52、eil51、st70、eil76、rat99、eil101。硬件环境为 Windows，CPU 为 12th Gen Intel(R) Core(TM) i5-12600KF，GPU 为 NVIDIA GeForce RTX 4070 SUPER。编译器为 MSVC 19.44 与 nvcc 12.9.41，构建工具为 CMake + Ninja，构建模式为 Release，OpenMP 和 CUDA 均启用。

### 7.2 指标定义

Gap 定义为：

$$
Gap=\frac{best\_length-BKS}{BKS}\times 100\%
$$

Speedup 定义为：

$$
Speedup=\frac{T_{serial}}{T_{parallel}}
$$

Parallel efficiency 定义为：

$$
Efficiency=\frac{Speedup}{thread\_count}\times 100\%
$$

### 7.3 实验分组

表 6：实验分组。

| 分组 | 目的 | 主要输出 |
|---|---|---|
| baseline | 建立 serial multi-chain 基准 | raw CSV 与 summary CSV |
| OpenMP scaling | 分析多线程加速与效率 | speedup、efficiency 图表 |
| tuning | 搜索 SA/QLSA 参数 | tuning summary |
| targeted enhancement | 扩大 chains/iterations 观察质量 | hard-instance 质量结果 |
| policy comparison | 比较 epsilon-greedy 与 Softmax | policy summary |
| CUDA positioning | 说明 CUDA 工程状态与局限 | berlin52 CUDA/OpenMP/serial 对比 |

默认参数实验使用 iterations=1,000,000、chains=32、repeat=3、OpenMP threads=8、init=nn。调优验证使用独立 seed，避免只报告搜索阶段挑出的最好结果。

## 8. 实验结果与分析

### 8.1 默认参数下 OpenMP 加速效果

![OpenMP speedup across TSPLIB95 instances](../../figures/final/fig02_openmp_speedup.png)

图 2：OpenMP 多实例加速比。

![OpenMP parallel efficiency across TSPLIB95 instances](../../figures/final/fig03_openmp_efficiency.png)

图 3：OpenMP 多实例并行效率。

表 7：SA 默认参数 OpenMP 加速结果。

| Instance | Serial ms | OpenMP ms | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| berlin52 | 1043.053 | 202.402 | 5.153x | 64.417% |
| eil51 | 1233.527 | 224.443 | 5.496x | 68.699% |
| st70 | 1243.066 | 247.472 | 5.023x | 62.788% |
| eil76 | 1377.377 | 245.818 | 5.603x | 70.040% |
| rat99 | 1310.632 | 226.910 | 5.776x | 72.200% |
| eil101 | 1325.891 | 231.733 | 5.722x | 71.520% |

表 8：QLSA 默认参数 OpenMP 加速结果。

| Instance | Serial ms | OpenMP ms | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| berlin52 | 2217.895 | 411.445 | 5.391x | 67.381% |
| eil51 | 2371.958 | 428.979 | 5.529x | 69.116% |
| st70 | 2356.846 | 560.222 | 4.207x | 52.587% |
| eil76 | 2439.045 | 490.690 | 4.971x | 62.133% |
| rat99 | 2400.499 | 570.091 | 4.211x | 52.634% |
| eil101 | 2490.072 | 445.503 | 5.589x | 69.867% |

SA 平均 speedup 约 5.46x，平均 parallel efficiency 约 68.28%；QLSA 平均 speedup 约 4.98x，平均 parallel efficiency 约 62.29%。QLSA 效率略低，主要因为 Q 表更新、策略选择和状态离散化增加了每链内部常数开销。OpenMP 多链并行不改变搜索逻辑，因此可作为主性能提升结果。

### 8.2 默认参数下解质量

![Default-parameter Gap comparison](../../figures/final/fig04_default_gap.png)

图 4：默认参数下 SA 与 QLSA 的 Gap 对比。

默认参数下，berlin52、eil51、st70 达到 BKS；eil76、rat99、eil101 仍存在 Gap。该结果说明并行加速不等于自动获得更优解：OpenMP 主要减少 elapsed time，而 harder instances 还需要参数调优或更高搜索预算。

### 8.3 参数调优与定向增强

![Gap reduction after tuning and targeted enhancement](../../figures/final/fig05_tuning_curve.png)

图 5：调参与定向增强后的 Gap 改善。

表 9：定向增强关键结果。

| Instance | Family | Best length | Min Gap | Mean Gap | Mean ms |
|---|---|---:|---:|---:|---:|
| eil101 | SA | 629 | 0.000% | 0.445% | 1677.495 |
| eil101 | QLSA | 629 | 0.000% | 0.254% | 3348.545 |
| rat99 | SA | 1212 | 0.083% | 0.330% | 329.022 |
| rat99 | QLSA | 1211 | 0.000% | 0.099% | 1649.518 |

定向增强不是重新做全量网格搜索，而是在较优参数附近增加 chains 或 iterations。eil101 上 SA 与 QLSA 均达到 BKS=629；rat99 上 QLSA 达到 BKS=1211，而 SA high-budget 最好为 1212。这是 QLSA 在本实验中最明确的质量收益案例。但该收益伴随更高 elapsed time，最终报告需要同时呈现质量和时间成本。

### 8.4 policy comparison

![QLSA policy comparison](../../figures/final/fig06_policy_comparison.png)

图 6：QLSA epsilon-greedy 与 Softmax 策略对比。

本实现中的 policy comparison 表明，Softmax 并非稳定优于 epsilon-greedy。rat99 上 epsilon-greedy mean Gap 更低，而 eil76、eil101 上两者差距较小。由于本实现的 action/state 设计不同于论文 candidate-leader 机制，该实验只能说明当前工程实现中的策略差异，不能直接等同于论文 Softmax 结论。

### 8.5 CUDA 定位实验

![CUDA positioning on berlin52](../../figures/final/fig07_cuda_positioning.png)

图 7：berlin52 上 Serial、OpenMP 与 CUDA elapsed time 对比。

CUDA 后端可以真实运行并找到 BKS，但 berlin52 这类小规模实例上不优于 OpenMP。主要原因是每条链工作量不足以抵消 kernel 启动、调度和访存成本。CUDA 的价值在本阶段体现在完整编译链、GPU kernel、数据拷贝和结果归约路径已经打通，为更大规模实例和 block 内候选 move 并行留下扩展空间。

## 9. 与近期论文结果对比

### 9.1 方法对比

表 10：论文方法与本实现对比。

| 项目 | 参考论文 | 本实现 |
|---|---|---|
| 实现语言 | Python 3.11.5 | C++20 |
| 主要算法 | SA、QLSA、SB-QLSA | SA、QLSA、多链并行后端 |
| QLSA action | candidate leader | 2-opt 邻域动作选择 |
| 状态机制 | SB-QLSA diversity state | delta/state discretization |
| 并行化 | 未作为主要实现 | OpenMP 与 CUDA 后端 |

### 9.2 运行时间参考对比

![Runtime reference comparison with the paper](../../figures/final/fig08_paper_runtime_comparison.png)

图 8：论文 Table 8 与本实现 OpenMP elapsed time 参考对比。

论文使用 Python、NumPy/Pandas 和 TSPLIB95 Python library，硬件环境为 Xeon 平台；本实现使用 Windows、C++20、OpenMP/CUDA 和 i5-12600KF/RTX 4070 SUPER。因此，绝对时间不可直接比较，不能称为同一平台下的严格性能比较。该图的作用是说明：在共同 TSPLIB95 实例上，C++ 工程化和 OpenMP 多链并行带来了实际运行效率上的数量级差异，但差异来源包括语言、硬件、实现和并行方式。

### 9.3 解质量参考对比

![Hard-instance mean Gap comparison](../../figures/final/fig09_paper_quality_comparison.png)

图 9：论文 hard-instance mean Gap 与本实现调优/增强结果对比。

hard-instance quality 对比使用共同 BKS 和 Gap 指标。参考论文中，QLSA/SB-QLSA 相比 Paper-SA 明显降低 mean Gap。本实现通过调参与定向增强，在 eil76、rat99、eil101 上进一步接近 BKS，其中 rat99 QLSA high-budget 达到 BKS。该对比可以支持“工程化实现、并行多链和调参增强提高了解质量”，但不能表述为完整复刻论文 SB-QLSA 或严格全面超过论文。

### 9.4 对比风险说明

表 11：论文对比风险与规避方式。

| 风险 | 原因 | 报告处理方式 |
|---|---|---|
| 时间不是公平 benchmark | 硬件、语言、实现均不同 | 使用“参考对比”表述 |
| QLSA 机制不完全一致 | 本实现不是完整 candidate-leader SB-QLSA | 单独列出机制差异 |
| 运行次数不同 | 默认实验 repeat=3，调优验证 repeat=10 | 按实验分组解释统计口径 |
| CUDA 未优于 OpenMP | 小实例 GPU 开销占比高 | 定位为工程扩展和后续方向 |

## 10. 工程难度与完成质量说明

工程难度主要体现在五个方面。第一，C++20 模块化实现替代 Python 脚本式复现，需要自己处理 parser、distance matrix、随机数、计时、CLI 和 CSV 输出。第二，TSPLIB95 parser 支持多种 edge weight type 与 explicit matrix format，保证实验数据输入可靠。第三，O(1) 2-opt delta 与连续 DistanceMatrix 为性能优化提供基础。第四，OpenMP/CUDA 双后端要求保持相同 seed 规则、输出字段和结果归约接口。第五，实验系统形成 raw CSV 到 summary、figures、report 的完整 pipeline，支持后续复现实验而不是手工复制终端结果。

同时，项目对课程提交和公开展示进行了区分：course 版报告保留课程要求的个人信息，public 版和 public submission 目录脱敏。结果文件按 final、raw、summary、reference、archive 分层，避免报告数据来源混乱。

## 11. 局限性

1. CUDA 后端仍未充分优化。当前实现更偏工程验证，尚未把 block 内线程用于大批量候选 move 评价，因此小规模实例上不优于 OpenMP。
2. QLSA 没有完整复刻论文 SB-QLSA。当前实现采用状态/动作离散化，不包含完整 candidate leader 和 diversity state 机制。
3. policy comparison 不是论文 Softmax 机制的严格复现。它只比较当前实现中的 epsilon-greedy 与 Softmax 行为。
4. 预算扫描曲线不是逐迭代 trace。若报告使用相关补充图，只能描述为预算扫描或外部采样结果。
5. 与论文运行时间对比不是同一平台下的严格性能比较。不同语言、硬件和实现会显著影响 elapsed time。
6. 实例规模仍偏小。CUDA 和更大规模 TSP 的潜力需要在更多 TSPLIB95 实例上继续验证。

## 12. 总结

性能结论方面，OpenMP chain-level multi-chain 是最可靠的并行优化成果。默认参数多实例实验中，SA 平均 speedup 约 5.46x，QLSA 平均 speedup 约 4.98x，证明粗粒度多链并行适合 SA/QLSA 这类随机搜索算法。

解质量结论方面，默认参数已经能在 berlin52、eil51、st70 上达到 BKS，但 harder instances 需要调参和更高预算。定向增强中，eil101 的 SA/QLSA 均达到 BKS，rat99 的 QLSA high-budget 达到 BKS 而 SA 未达到，说明 QLSA 在部分实例上具有实际质量收益。

工程结论方面，系统完成了从 TSPLIB95 输入、C++20 算法核心、OpenMP/CUDA 后端、实验脚本、CSV 分析、图表生成到 course/public 双报告的完整闭环。该完成度覆盖课程对算法实现、并行优化、实验评价、近期论文对比和报告质量的主要要求。

## 参考文献

1. Adil, N., Eddaoudi, F., Lakhbab, H., & Naimi, M. (2026). Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization. *Statistics, Optimization & Information Computing*, 15(5), 3706-3730. https://doi.org/10.19139/soic-2310-5070-3028
2. Reinelt, G. TSPLIB: A Traveling Salesman Problem Library. *ORSA Journal on Computing*, 3(4), 376-384, 1991.
3. OpenMP Architecture Review Board. OpenMP Application Programming Interface Specification.
4. NVIDIA. CUDA C++ Programming Guide.
5. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. Optimization by Simulated Annealing. *Science*, 220(4598), 671-680, 1983.
6. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*. MIT Press.

## 附录 A：个人工作说明

本课程大作业为单人团队完成，团队成员为陈乐浚，学号 22361054。因此，从选题、论文阅读、工程实现、并行化设计、实验执行、结果分析到报告撰写，均由本人独立承担。

在选题阶段，本人阅读并整理了 2026 年发表的 Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization 论文，重点分析其中 classical SA、stateless QLSA、State-Based QLSA、candidate leader、epsilon-greedy、Softmax 和 diversity state 等机制。基于课程对近期论文复现和并行优化的要求，本人将题目确定为面向 TSP 的 Q-Learning 辅助模拟退火并行优化，并明确不做简单脚本复现，而是进行 C++20 工程化实现和多后端并行扩展。

在工程框架搭建阶段，本人完成了 CMake 项目结构、模块化头文件和源文件组织、CLI 参数系统、可复现实验 seed 设计，以及用于后续实验的 Python/Bat 自动化脚本。底层数据结构方面，本人实现了 TSPLIB95 parser，支持坐标型和显式矩阵型实例，并实现了 EUC_2D、CEIL_2D、GEO、ATT、EXPLICIT 等距离类型。DistanceMatrix 采用一维连续数组存储，既提高 CPU 缓存友好性，也为 CUDA 端拷贝提供了直接数据布局。Tour 模块实现了路径合法性检查、nearest-neighbor 初始化、random 初始化、路径长度计算和 O(1) 2-opt delta 计算。

在算法实现阶段，本人完成了串行 SA 基线，并在此基础上实现 QLSA 变体。SA 使用 2-opt 邻域、Metropolis 接受准则和指数退火。QLSA 通过状态离散化、动作选择、Q 表更新、epsilon-greedy 与 Softmax 策略对搜索行为进行辅助调节。实现过程中本人保持了对论文机制的谨慎对应关系：当前 QLSA 体现了 Q-learning 辅助搜索策略选择思想，但没有声称完整复刻论文中的 SB-QLSA candidate-leader 与 diversity-state 机制。

在并行化阶段，本人实现了 OpenMP 多链并行，将多条独立 SA/QLSA 搜索链映射到不同线程，链内维护独立 RNG、tour、best tour 和 Q 表，线程间只共享只读 DistanceMatrix，结束后进行串行归约。该设计避免了 move-level 并行带来的频繁同步和随机过程复现困难，是本项目最主要的并行性能来源。本人还实现了 CUDA 后端工程扩展，完成 Ninja + CUDA 构建、kernel 编译和 smoke test。由于小规模 TSPLIB 实例上 CUDA 受 kernel 启动、调度和每链工作量不足影响，最终报告将其定位为工程扩展和后续优化方向，而不作为主要加速结论。

在实验阶段，本人完成了默认参数多实例实验、调参搜索、独立 seed 验证、定向增强实验、policy comparison、OpenMP scaling 和 CUDA positioning，并建立了 raw CSV、summary CSV、日志、图表和报告的流水线。实验结果显示，OpenMP 多链并行在多个 TSPLIB95 实例上取得稳定加速；调参与定向增强提高了 harder instances 的解质量；rat99 上 QLSA high-budget 达到 BKS，而 SA high-budget 未达到 BKS，形成了 QLSA 相对 SA 的一个明确质量案例。

在报告和提交整理阶段，本人对文档、图表、结果文件进行了提交级整理，区分 course 版和 public 版报告，保护姓名、学号等私人信息，并编写了复现命令、结果索引、项目结构说明和已知限制说明。最终报告采用正式课程报告写法，强调预期目标与实际完成情况、实施方案、问题解决、并行性能、论文对比和局限性，避免夸大 CUDA 或 QLSA 的结果。
