# Final Known Issues

本文件记录最终阶段仍未完成或不应夸大的事项。以下内容不是阻塞项，但应在答辩或报告解释中保持诚实。

## 1. paper-lite QLSA 未实现

当前 QLSA 实现采用状态/动作离散化策略，能够支持 epsilon-greedy 和 softmax，但没有完整实现论文中的 candidate leader 集合：

- current solution；
- global best solution；
- random solution；
- double-bridge solution。

也没有逐项实现论文 SB-QLSA 的 Hamming-distance diversity state。因此最终报告只能称为“基于论文思想的工程复现与并行化扩展”，不能称为完整复刻论文 SB-QLSA。

## 2. 收敛曲线是 budget-sweep proxy

本轮没有修改核心 CLI 添加 `--trace` 和 `--trace_interval`。为避免最终阶段引入核心代码风险，`fig_convergence_berlin52.png` 和 `fig_convergence_rat99.png` 来自不同 iteration budget 的独立运行，不是同一次运行内部的逐迭代 trace。

报告中应称为“预算扫描曲线”或“收敛近似”，不要称为严格逐迭代收敛轨迹。

## 3. CUDA chains/block 补充图来自 smoke scaling

`fig_cuda_chains_blocks.png` 使用已有 `results/cuda_scaling.csv` 中的 `square4` smoke scaling 数据，不是 `berlin52`/`eil101` 上完整 CUDA grid。该图只用于说明 CUDA 参数通路和工程验证，不作为 GPU 性能结论。

## 4. 论文时间对比不是同环境基准

论文使用 Python/NumPy/Pandas + Xeon 环境，本项目使用 C++20 + OpenMP/CUDA + i5-12600KF/RTX 4070 SUPER。因此 `paper_table8_runtime.csv` 与本项目运行时间只能作为参考对比，不能解释为严格同平台性能比较。

## 5. EXPLICIT TSPLIB 测试覆盖仍可加强

代码支持 EXPLICIT 距离矩阵格式，但当前自动测试主要覆盖坐标型小实例。若继续完善，可增加一个 4 城市 EXPLICIT fixture，覆盖 `FULL_MATRIX` 或 `UPPER_ROW`。

## 6. CUDA 性能仍有优化空间

CUDA backend 已完成并可运行，但当前小规模 TSPLIB95 实例上不优于 OpenMP。后续需要在 block 内并行候选 2-opt move、优化共享内存使用，并在更大实例上重新评估。
