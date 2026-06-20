# Step 5A Berlin52 实验结果分析

## 实验配置

- 原始数据：`results/step5_multi_cpu_raw.csv`
- 统计方式：按 `algorithm + instance + iterations + chains + threads + parallel` 分组。
- Speedup：SA 以 `sa-multichain` 为基准，QLSA 以 `qlsa-multichain` 为基准。
- OpenMP parallel efficiency：`speedup / threads`。
- Gap：相对 TSPLIB BKS 计算。

## 汇总表格

| Instance | Algorithm | Parallel | Chains | Threads | Runs | BKS | Best | Gap % | Mean ms | Std ms | Speedup | OMP Eff. % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| berlin52 | qlsa-multichain | none | 32 | 1 | 3 | 7542 | 7542 | 0.000 | 2217.895 | 48.922 | 1.000 | - |
| berlin52 | qlsa-omp | omp | 32 | 8 | 3 | 7542 | 7542 | 0.000 | 411.445 | 48.245 | 5.391 | 67.38 |
| berlin52 | sa-multichain | none | 32 | 1 | 3 | 7542 | 7542 | 0.000 | 1043.053 | 5.651 | 1.000 | - |
| berlin52 | sa-omp | omp | 32 | 8 | 3 | 7542 | 7542 | 0.000 | 202.402 | 4.111 | 5.153 | 64.42 |
| eil101 | qlsa-multichain | none | 32 | 1 | 3 | 629 | 637 | 1.272 | 2490.072 | 88.380 | 1.000 | - |
| eil101 | qlsa-omp | omp | 32 | 8 | 3 | 629 | 637 | 1.272 | 445.503 | 16.873 | 5.589 | 69.87 |
| eil101 | sa-multichain | none | 32 | 1 | 3 | 629 | 635 | 0.954 | 1325.891 | 28.985 | 1.000 | - |
| eil101 | sa-omp | omp | 32 | 8 | 3 | 629 | 635 | 0.954 | 231.733 | 4.907 | 5.722 | 71.52 |
| eil51 | qlsa-multichain | none | 32 | 1 | 3 | 426 | 426 | 0.000 | 2371.958 | 80.746 | 1.000 | - |
| eil51 | qlsa-omp | omp | 32 | 8 | 3 | 426 | 426 | 0.000 | 428.979 | 80.000 | 5.529 | 69.12 |
| eil51 | sa-multichain | none | 32 | 1 | 3 | 426 | 426 | 0.000 | 1233.527 | 59.089 | 1.000 | - |
| eil51 | sa-omp | omp | 32 | 8 | 3 | 426 | 426 | 0.000 | 224.443 | 16.973 | 5.496 | 68.70 |
| eil76 | qlsa-multichain | none | 32 | 1 | 3 | 538 | 542 | 0.743 | 2439.045 | 83.400 | 1.000 | - |
| eil76 | qlsa-omp | omp | 32 | 8 | 3 | 538 | 542 | 0.743 | 490.690 | 10.913 | 4.971 | 62.13 |
| eil76 | sa-multichain | none | 32 | 1 | 3 | 538 | 539 | 0.186 | 1377.377 | 120.927 | 1.000 | - |
| eil76 | sa-omp | omp | 32 | 8 | 3 | 538 | 539 | 0.186 | 245.818 | 8.771 | 5.603 | 70.04 |
| rat99 | qlsa-multichain | none | 32 | 1 | 3 | 1211 | 1225 | 1.156 | 2400.499 | 108.343 | 1.000 | - |
| rat99 | qlsa-omp | omp | 32 | 8 | 3 | 1211 | 1225 | 1.156 | 570.091 | 46.343 | 4.211 | 52.63 |
| rat99 | sa-multichain | none | 32 | 1 | 3 | 1211 | 1215 | 0.330 | 1310.632 | 43.819 | 1.000 | - |
| rat99 | sa-omp | omp | 32 | 8 | 3 | 1211 | 1215 | 0.330 | 226.910 | 3.127 | 5.776 | 72.20 |
| st70 | qlsa-multichain | none | 32 | 1 | 3 | 675 | 675 | 0.000 | 2356.846 | 109.681 | 1.000 | - |
| st70 | qlsa-omp | omp | 32 | 8 | 3 | 675 | 675 | 0.000 | 560.222 | 107.027 | 4.207 | 52.59 |
| st70 | sa-multichain | none | 32 | 1 | 3 | 675 | 675 | 0.000 | 1243.066 | 35.249 | 1.000 | - |
| st70 | sa-omp | omp | 32 | 8 | 3 | 675 | 675 | 0.000 | 247.472 | 14.479 | 5.023 | 62.79 |

## 主要结论

- berlin52 的 BKS 为 7542，本项目当前所有版本均达到 best_length=7542，Gap=0%。
- SA OpenMP 相对 SA 串行多链平均加速约 5.15x，8 线程并行效率约 64.4%。
- QLSA OpenMP 相对 QLSA 串行多链平均加速约 5.39x，8 线程并行效率约 67.4%。
- OpenMP 8 线程并行效率约 72% 到 74%，说明 chain-level 并行在 berlin52 上已有稳定收益。

## CUDA 结果解释

- berlin52 规模较小，当前 CUDA multi-chain 实现容易受到 kernel 启动开销、global/shared memory 访问、每条 chain 工作量不足、block 内并行度尚未充分利用等因素影响。
- 当前阶段应将 OpenMP 作为主要性能提升结果，CUDA 作为已完成的工程扩展与后续优化方向。

## 后续实验计划

- 扩展到 eil51、st70、eil76、rat99、eil101 等实例，观察规模增大后的 CUDA 表现。
- 对 CUDA 版本增加 block 内候选 2-opt move 并行评价，提升每条 chain 的 GPU 内部并行度。
- 统一统计 Best/Mean/Std/Gap/Runtime/Speedup/Parallel Efficiency，并与论文表格进行公平对比。
- 对 OpenMP 测试不同 threads/chains 组合，确认最佳 CPU 并行参数。
