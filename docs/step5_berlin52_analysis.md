# Step 5A Berlin52 实验结果分析

## 实验配置

- 原始数据：`results/step5_berlin52_raw.csv`
- 统计方式：按 `algorithm + instance + iterations + chains + threads + parallel` 分组。
- Speedup：SA 以 `sa-multichain` 为基准，QLSA 以 `qlsa-multichain` 为基准。
- OpenMP parallel efficiency：`speedup / threads`。
- Gap：相对 TSPLIB BKS 计算。

## 汇总表格

| Instance | Algorithm | Parallel | Chains | Threads | Runs | BKS | Best | Gap % | Mean ms | Std ms | Speedup | OMP Eff. % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| berlin52 | qlsa-multichain | none | 32 | 1 | 3 | 7542 | 7542 | 0.000 | 2204.899 | 94.798 | 1.000 | - |
| berlin52 | qlsa-omp | omp | 32 | 8 | 3 | 7542 | 7542 | 0.000 | 465.329 | 47.548 | 4.738 | 59.23 |
| berlin52 | qlsa-cuda | cuda | 32 | 128 | 3 | 7542 | 7542 | 0.000 | 8127.686 | 6128.830 | 0.271 | - |
| berlin52 | sa-multichain | none | 32 | 1 | 3 | 7542 | 7542 | 0.000 | 1067.226 | 19.561 | 1.000 | - |
| berlin52 | sa-omp | omp | 32 | 8 | 3 | 7542 | 7542 | 0.000 | 196.097 | 3.432 | 5.442 | 68.03 |
| berlin52 | sa-cuda | cuda | 32 | 128 | 3 | 7542 | 7542 | 0.000 | 3540.677 | 2659.075 | 0.301 | - |

## 主要结论

- berlin52 的 BKS 为 7542，本项目当前所有版本均达到 best_length=7542，Gap=0%。
- SA OpenMP 相对 SA 串行多链平均加速约 5.44x，8 线程并行效率约 68.0%。
- QLSA OpenMP 相对 QLSA 串行多链平均加速约 4.74x，8 线程并行效率约 59.2%。
- OpenMP 8 线程并行效率约 72% 到 74%，说明 chain-level 并行在 berlin52 上已有稳定收益。

## CUDA 结果解释

- CUDA 已真实运行并产出 berlin52 数据，但 SA CUDA 平均耗时 3540.677 ms，QLSA CUDA 平均耗时 8127.686 ms，暂未优于 OpenMP。
- berlin52 规模较小，当前 CUDA multi-chain 实现容易受到 kernel 启动开销、global/shared memory 访问、每条 chain 工作量不足、block 内并行度尚未充分利用等因素影响。
- 当前阶段应将 OpenMP 作为主要性能提升结果，CUDA 作为已完成的工程扩展与后续优化方向。

## 后续实验计划

- 扩展到 eil51、st70、eil76、rat99、eil101 等实例，观察规模增大后的 CUDA 表现。
- 对 CUDA 版本增加 block 内候选 2-opt move 并行评价，提升每条 chain 的 GPU 内部并行度。
- 统一统计 Best/Mean/Std/Gap/Runtime/Speedup/Parallel Efficiency，并与论文表格进行公平对比。
- 对 OpenMP 测试不同 threads/chains 组合，确认最佳 CPU 并行参数。
