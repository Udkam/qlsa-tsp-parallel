# 个人工作说明

本课程大作业为单人团队完成，团队成员为陈乐浚，学号 22361054。因此，从选题、论文阅读、工程实现、并行化设计、实验执行、结果分析到报告撰写，均由本人独立承担。

在选题阶段，本人阅读并整理了 2026 年发表的 Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization 论文，重点分析其中 classical SA、stateless QLSA、State-Based QLSA、candidate leader、epsilon-greedy、Softmax 和 diversity state 等机制。基于课程对近期论文复现和并行优化的要求，本人将题目确定为面向 TSP 的 Q-Learning 辅助模拟退火并行优化，并明确不做简单脚本复现，而是进行 C++20 工程化实现和多后端并行扩展。

在工程框架搭建阶段，本人完成了 CMake 项目结构、模块化头文件和源文件组织、CLI 参数系统、可复现实验 seed 设计，以及用于后续实验的 Python/Bat 自动化脚本。底层数据结构方面，本人实现了 TSPLIB95 parser，支持坐标型和显式矩阵型实例，并实现了 EUC_2D、CEIL_2D、GEO、ATT、EXPLICIT 等距离类型。DistanceMatrix 采用一维连续数组存储，既提高 CPU 缓存友好性，也为 CUDA 端拷贝提供了直接数据布局。Tour 模块实现了路径合法性检查、nearest-neighbor 初始化、random 初始化、路径长度计算和 O(1) 2-opt delta 计算。

在算法实现阶段，本人完成了串行 SA 基线，并在此基础上实现 QLSA 变体。SA 使用 2-opt 邻域、Metropolis 接受准则和指数退火。QLSA 通过状态离散化、动作选择、Q 表更新、epsilon-greedy 与 Softmax 策略对搜索行为进行辅助调节。实现过程中本人保持了对论文机制的谨慎对应关系：当前 QLSA 体现了 Q-learning 辅助搜索策略选择思想，但没有声称完整复刻论文中的 SB-QLSA candidate-leader 与 diversity-state 机制。

在并行化阶段，本人实现了 OpenMP 多链并行，将多条独立 SA/QLSA 搜索链映射到不同线程，链内维护独立 RNG、tour、best tour 和 Q 表，线程间只共享只读 DistanceMatrix，结束后进行串行归约。该设计避免了 move-level 并行带来的频繁同步和随机过程复现困难，是本项目最主要的并行性能来源。本人还实现了 CUDA 后端工程扩展，完成 Ninja + CUDA 构建、kernel 编译和 smoke test。由于小规模 TSPLIB 实例上 CUDA 受 kernel 启动、调度和每链工作量不足影响，最终报告将其定位为工程扩展和后续优化方向，而不作为主要加速结论。

在实验阶段，本人完成了默认参数多实例实验、调参搜索、独立 seed 验证、定向增强实验、policy comparison、OpenMP scaling 和 CUDA positioning，并建立了 raw CSV、summary CSV、日志、图表和报告的流水线。实验结果显示，OpenMP 多链并行在多个 TSPLIB95 实例上取得稳定加速；调参与定向增强提高了 harder instances 的解质量；rat99 上 QLSA high-budget 达到 BKS，而 SA high-budget 未达到 BKS，形成了 QLSA 相对 SA 的一个明确质量案例。

在报告和提交整理阶段，本人对文档、图表、结果文件进行了提交级整理，区分 course 版和 public 版报告，保护姓名、学号等私人信息，并编写了复现命令、结果索引、项目结构说明和已知限制说明。最终报告采用正式课程报告写法，强调预期目标与实际完成情况、实施方案、问题解决、并行性能、论文对比和局限性，避免夸大 CUDA 或 QLSA 的结果。
