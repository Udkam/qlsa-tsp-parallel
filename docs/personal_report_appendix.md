# 个人报告附录

## 基本信息

本项目团队人数为 1 人，项目成员为陈乐浚，学号 22361054，学院/专业为中山大学计算机学院 / 信息与计算科学。本项目从选题、论文阅读、工程设计、算法实现、并行化、实验脚本、调参分析、图表生成到最终报告撰写均由本人独立完成。

## 个人承担工作

### 1. 选题与论文阅读

本人首先根据并行算法课程大作业要求确定选题方向，选择旅行推销员问题作为组合优化对象，并阅读参考论文 *Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization*。阅读过程中重点分析了论文中 SA、QLSA 与 SB-QLSA 的关系，理解 Q-learning 如何用于搜索策略选择，以及论文如何在 TSPLIB95 实例上统计 Best、Mean、Std、Gap 和运行时间。随后，本人将项目目标确定为：在 C++20 中实现 SA 与 QLSA，并在此基础上完成 OpenMP 与 CUDA 多链并行化。

### 2. 工程框架搭建

本人负责搭建 C++20 + CMake 工程结构，设计 `include/tsp/` 与 `src/` 的模块划分。工程从第一阶段就考虑后续并行扩展，将实例数据、距离矩阵、路径表示、随机数、计时器、SA、QLSA、并行执行接口和 CUDA 后端拆分为相对独立的模块。这样的结构使 OpenMP 与 CUDA 可以复用相同的 `DistanceMatrix`、tour 表示、参数结构和结果结构，降低了后续扩展的复杂度。

### 3. TSPLIB95 解析器与底层数据结构

本人实现了 TSPLIB95 `.tsp` 文件解析器，支持坐标型实例和显式矩阵型实例，覆盖 `EUC_2D`、`CEIL_2D`、`GEO`、`ATT`、`EXPLICIT` 等距离类型，以及 `FULL_MATRIX`、`UPPER_ROW`、`LOWER_ROW`、`UPPER_DIAG_ROW`、`LOWER_DIAG_ROW` 等显式矩阵格式。本人同时实现了使用一维连续数组存储的 `DistanceMatrix`，使距离查询具有较低开销，也便于后续将矩阵直接拷贝到 GPU。

### 4. SA 与 QLSA 实现

本人实现了串行 SA 基线，包括 nearest-neighbor 初始化、随机 tour 初始化、2-opt 邻域、O(1) delta 计算、Metropolis 接受准则、指数退火和最终路径长度校验。在此基础上，本人实现了 QLSA，加入离散状态、动作集合、奖励设计、Q table 更新、epsilon-greedy 和 softmax 动作选择策略。QLSA 实现保持与 SA 主循环结构一致，使后续 OpenMP 与 CUDA 后端可以共享相同的 multi-chain 抽象。

### 5. OpenMP 多链并行

本人设计并实现了 OpenMP multi-chain 并行方案。该方案将多条独立 SA/QLSA 搜索链映射到 OpenMP `parallel for`，每条 chain 维护独立随机数状态、tour、best tour 和 Q table。并行区内不在 move 内层循环频繁加锁，也不直接竞争全局 best，而是在所有 chain 结束后由主线程归约结果。该方案最终在多个 TSPLIB95 实例上取得约 5x 的平均 speedup，是本项目最主要的性能提升来源。

### 6. CUDA 多链工程实现

本人完成了 CUDA backend 的工程实现，包括 CUDA 参数接口、host wrapper、`cuda_kernels.cu`、距离矩阵 GPU 拷贝、每条 chain 独立执行和 kernel 结束后的 host 端结果归约。实现过程中曾遇到 Visual Studio CMake 生成器无法启用 CUDA toolset 的问题，后续改用 Ninja + CUDA 成功完成真实编译与运行验证。当前 CUDA backend 已作为工程扩展完成，但本人在报告中明确说明其在小规模实例上不是主要加速证据。

### 7. 实验自动化与参数调优

本人编写了多阶段实验脚本，包括默认参数多实例实验、`berlin52` 手动结果归档、CPU/OpenMP 多实例分析、参数搜索、调优结果分析、独立 seed 验证、定向增强实验和最终结果汇总。实验过程中，本人区分了默认参数加速结果、调优后独立验证结果和高预算质量结果，避免只引用 tuning search 中的最好样本。结果显示，OpenMP multi-chain 提供主要运行时间加速，调参和定向增强能改善 `eil76`、`rat99`、`eil101` 等较难实例的解质量，其中 `rat99` 上 QLSA high-budget 达到 BKS，而 SA high-budget 未达到 BKS。

### 8. 图表与报告撰写

本人整理了各阶段设计文档、实验结果表格、最终实验汇总和课程报告，并进一步生成论文对比 CSV 与报告图表。报告中明确区分主结论与补充分析：OpenMP 是主要性能提升结果；CUDA 是已完成的工程扩展，但不在当前小规模实例上作为主要 speedup 证据；QLSA 在部分实例上具有解质量优势，但不能概括为所有实例均占优。本人同时撰写个人报告附录和最终提交检查清单，确保最终提交材料完整、数据来源清晰、结论表述谨慎。

## 个人总结

通过本项目，本人完整经历了从论文算法思想理解、C++ 工程化实现、并行化设计、实验自动化、参数调优到结果分析和报告撰写的全过程。项目不仅实现了 SA、QLSA、OpenMP 和 CUDA，还建立了可复现实验脚本和多阶段分析文档。该过程加深了本人对随机启发式搜索、chain-level 并行、GPU 工程实现、实验指标统计和论文对比边界的理解，也体现了并行算法课程中从算法思想到工程性能分析的完整实践链条。
