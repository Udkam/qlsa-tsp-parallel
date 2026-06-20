# 个人报告附录

## 基本信息

本项目团队人数为 1 人，项目成员为陈乐浚，学号 22361054，学院/专业为中山大学计算机学院 / 信息与计算科学。本项目从选题、论文阅读、工程设计、算法实现、并行化、实验自动化、调参分析、图表生成到最终报告撰写均由本人独立完成。

## 个人工作说明

在选题阶段，本人根据并行算法课程大作业要求，选择旅行推销员问题作为组合优化对象，并阅读参考论文 *Q-Learning-Assisted Simulated Annealing for Traveling Salesman Problem Optimization*。阅读过程中重点分析了论文中的 SA、QLSA、SB-QLSA、candidate set、Softmax/epsilon-greedy 策略以及 Table 4、Table 8、Table 9 的实验结论。随后，本人将项目定位为：在论文思想基础上完成 C++20 工程复现，并进一步实现 OpenMP 与 CUDA 并行化扩展。

在工程实现阶段，本人搭建了 C++20 + CMake 项目结构，设计 `include/tsp/` 与 `src/` 的模块划分，实现 TSPLIB95 parser、DistanceMatrix、Tour、随机数、计时器、SA、QLSA、OpenMP parallel 和 CUDA backend。TSPLIB95 parser 支持坐标型和显式矩阵型实例；DistanceMatrix 使用一维连续数组，既提升 CPU 端访问效率，也便于后续拷贝到 GPU；Tour 模块实现路径合法性检查、nearest-neighbor 初始化和 2-opt O(1) delta。SA 实现包括 Metropolis 接受准则、指数退火、seed 可复现和最终路径长度校验。QLSA 实现包括状态/动作离散化、Q 表更新、epsilon-greedy 和 softmax 策略。

在并行化阶段，本人设计了 OpenMP 多链并行方案。该方案将多条独立 SA/QLSA 搜索链映射到 OpenMP 线程，每条 chain 维护独立 seed、tour、best tour 和 Q table，并在并行结束后串行归约全局最优。该方案避免在 move 内层循环中频繁加锁，是本项目最稳定的性能提升来源。CUDA 部分由本人完成 host wrapper、CUDA 参数接口、`cuda_kernels.cu`、距离矩阵 GPU 拷贝和结果归约。实现过程中曾遇到 Windows Visual Studio 生成器无法启用 CUDA toolset 的问题，后续改用 Ninja + CUDA 成功完成真实编译和运行验证。

在实验与分析阶段，本人编写了默认参数多实例实验、`berlin52` 手动结果归档、参数调优、独立 seed 验证、定向增强、policy comparison、OpenMP scaling 和预算扫描脚本。所有实验均保存 raw CSV、summary CSV 和日志，并生成对应 Markdown 分析文档。由于 TSPLIB 下载脚本在部分网络/代理环境下可能失败，本人保留了手动下载 `.tsp` 文件放入 `data/` 的复现方式。实验结果表明，OpenMP 多链并行提供约 5x 平均加速；调优和定向增强能改善 `eil76`、`rat99`、`eil101` 等较难实例的解质量；`rat99` 上 QLSA high-budget 达到 BKS，而 SA high-budget 未达到 BKS。

在报告撰写阶段，本人整理了论文机制对齐、代码质量审查、最终实验汇总、图表生成脚本和提交包。报告中本人坚持谨慎表述：OpenMP 是主要性能结论；CUDA 是已完成的工程扩展，但当前小规模实例上不作为主要加速证据；QLSA 在部分实例上具有质量优势，但不能声称总是优于 SA；本项目吸收论文 SB-QLSA 的部分思想，但未逐项实现论文全部 candidate-leader 和 diversity-state 机制。通过本项目，本人完整实践了从论文阅读、算法实现、并行化设计、实验自动化到最终报告交付的全过程。
