# Final Report v2 Review Notes

## 1. 相较旧版报告的主要变化

新版 `docs/final_report_v2.md` 不再只是按项目阶段顺序描述工作，而是围绕“课程要求 -> 参考论文 -> 本项目实现 -> 并行化扩展 -> 实验对比 -> 局限性”的逻辑重构。主要改动包括：

- 新增“课程要求与项目完成度对应关系”，将完成情况、技术难度、论文复现/并行化、性能分析和报告要求逐项映射到本项目交付物。
- 新增“参考论文精读与对比基准”，补充论文实验环境、算法机制、QLSA/SB-QLSA 差异、Table 8 和 Table 9 的作用。
- 明确说明本项目是基于论文思想的 C++20/OpenMP/CUDA 工程复现与扩展，不声称完全复刻论文 SB-QLSA 的所有 candidate-leader 和 diversity-state 机制。
- 将实验结果分为 default speedup、tuned validation、targeted high-budget 和 paper reference comparison 四类，避免不同性质的数据混用。
- 新增大量图表引用，减少纯文字和大表堆叠。

## 2. 新增图表

本次新增并生成以下图表，统一放在 `figures/`：

1. `fig_architecture_pipeline.png`：项目整体架构与实验流水线。
2. `fig_openmp_speedup.png`：默认参数多实例 OpenMP speedup。
3. `fig_openmp_efficiency.png`：默认参数 OpenMP parallel efficiency。
4. `fig_default_gap.png`：默认参数下 SA/QLSA Gap。
5. `fig_tuned_quality_improvement.png`：调优和定向增强前后 Gap 改善。
6. `fig_paper_runtime_comparison_log.png`：论文 Table 8 时间与本项目 OpenMP 时间参考对比。
7. `fig_paper_quality_hard_instances.png`：论文 hard-instance mean Gap 与本项目调优/增强结果对比。
8. `fig_cuda_positioning.png`：`berlin52` 上 serial/OpenMP/CUDA 时间定位。

这些图由 `scripts/make_report_figures.py` 自动生成。若 matplotlib 不可用，脚本会输出明确 warning 并退化生成 SVG，不会静默失败。

## 3. 纳入对比的论文表格

本次新增两个论文基准数据文件：

- `results/paper_table8_runtime.csv`：来自论文 Table 8，包含 `eil51`、`berlin52`、`st70`、`eil76`、`rat99`、`eil101` 的 SA、QLSA 和 SB-QLSA 秒级运行时间。
- `results/paper_hard_instance_quality.csv`：来自论文质量表，包含 `eil76`、`rat99`、`eil101` 上 Paper-SA、Paper-QLSA 和 Paper-SB-QLSA 的 best、mean、std、max、mean Gap。

同时生成 `results/report_comparison_summary.csv`，用于报告中精简展示论文参考对比和本项目结果。

## 4. 可以安全写入最终报告的结论

- 本项目基于 2026 年 QLSA for TSP 论文思想，完成了 C++20 工程实现。
- 本项目实现了 SA、QLSA、OpenMP multi-chain 和 CUDA backend。
- OpenMP multi-chain 在六个 TSPLIB95 实例上取得稳定加速，SA 平均 speedup 约 5.46x，QLSA 平均 speedup 约 4.98x。
- 默认参数实验主要支撑并行性能结论；harder instances 的解质量需要调参或增加搜索预算。
- Step 6B 独立验证避免了直接报告 tuning search best。
- Step 6C targeted high-budget 中，`eil101` 上 SA/QLSA 均达到 BKS=629；`rat99` 上 QLSA 达到 BKS=1211，而 SA high-budget 最好为 1212。
- CUDA backend 已完成真实构建和运行验证，但当前小规模实例上不作为主要 speedup 证据。
- 论文 Table 8 和质量表可以作为参考对比，但必须说明硬件和语言不同。

## 5. 不能写入最终报告的结论

- 不能写本项目与论文运行时间对比是严格同平台公平 benchmark。
- 不能写 CUDA backend 是当前主要加速来源。
- 不能写 QLSA 在所有实例或所有参数设置下都优于 SA。
- 不能写默认参数下所有实例均达到 BKS。
- 不能写本项目完全复刻了论文 SB-QLSA 的所有细节。
- 不能只引用 tuning search 中的最好结果作为最终质量结论。

## 6. 若继续提升报告质量，可补充的工作

- 增加 softmax policy 与 epsilon-greedy 的系统对比，特别是在 `eil76`、`rat99`、`eil101` 上做 repeat=5 或 repeat=10。
- 记录 SA/QLSA 的收敛曲线，展示相同时间预算下的 best length 随时间变化。
- 对 repeated runs 做 Wilcoxon 或 Friedman 等非参数统计检验，提高质量比较的严谨性。
- 更接近论文地实现 SB-QLSA candidate-leader + diversity-state 机制，再与当前 QLSA 做消融实验。
- 在更大规模 TSPLIB95 实例上测试 CUDA backend，验证 GPU 是否能在更高工作量下体现优势。
