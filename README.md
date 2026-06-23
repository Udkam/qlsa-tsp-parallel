# Parallel TSP SA/QLSA

本仓库是并行算法课程大作业工程，主题为“面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化”。项目以 TSP 为应用背景，实现了 SA、QLSA、OpenMP 多搜索链并行、CUDA 后端和 MPI + OpenMP 双虚拟机分布式实验。

## 工程内容

- C++20 TSPLIB95 数据解析；
- 一维连续存储的 `DistanceMatrix`；
- 路径表示、最近邻初始化和 2-opt O(1) 增量计算；
- 串行 SA 与 QLSA；
- OpenMP 多搜索链并行；
- CUDA 多链模式、候选批量评价、候选策略和并行路径反转；
- QLSA 机制对齐变体：`current`、`paper`、`paper-sb`；
- MPI + OpenMP 分布式多链搜索；
- 自动实验、CSV 汇总、图表生成和报告检查脚本。

## 构建与测试

Windows + Ninja + CUDA：

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON -DTSP_ENABLE_MPI=ON"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build build-cuda-ninja -j"
ctest --test-dir build-cuda-ninja --output-on-failure
```

如果本机没有 MPI，CMake 会禁用 MPI 目标；OpenMP/CUDA 主程序仍可构建。

## 运行示例

OpenMP：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --repeat 3 --seed 1 --init nn
```

QLSA + OpenMP：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data\berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --repeat 3 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

论文机制对齐变体：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input data\berlin52.tsp --iterations 100000 --seed 1 --init nn
```

CUDA 候选批量评价：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn
```

CUDA 候选策略：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidate_policy hybrid --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn
```

QLSA + CUDA 候选批量评价：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

## 关键结果位置

- 默认参数 OpenMP 多实例实验：`results/summary/step5_multi_cpu_summary.csv`
- 调优验证与定向增强：`results/summary/tuned_validation_summary.csv`，`results/summary/targeted_quality_summary.csv`
- QLSA 机制对齐实验：`results/summary/qlsa_variant_alignment_summary.csv`
- CUDA QLSA candidate：`results/summary/cuda_qlsa_candidate_summary.csv`
- CUDA parallel reversal：`results/summary/cuda_reversal_summary.csv`
- CUDA candidate sweep：`results/summary/cuda_candidate_sweep_aggressive_summary.csv`
- 大实例 OpenMP：`results/summary/large_openmp_l1_summary.csv`
- 大实例 CUDA：`results/summary/large_cuda_formal_summary.csv`
- 双虚拟机 MPI：`results/summary/mpi_vm_scaling_formal_summary.csv`
- 论文参考数据：`results/reference/paper_table8_runtime.csv`，`results/reference/paper_hard_instance_quality.csv`

## 结论边界

- OpenMP 多搜索链并行是当前主要性能结论；
- CUDA 后端体现 GPU 工程扩展和候选批量评价能力，实验结论按数据解释；
- MPI + OpenMP 结果来自双 Ubuntu 虚拟机和真实 `mpirun`，用于说明分布式执行路径；
- QLSA 在部分实例和参数设置下改善解质量，不作为所有实例的普遍结论；
- 不把不同硬件、语言和实现方式下的论文运行时间对比写成同平台公平评测。

## 提交前检查

```powershell
py scripts\check_report_format.py
py scripts\check_report_assets.py
py scripts\check_privacy_and_encoding.py
py scripts\check_final_submission.py
ctest --test-dir build-cuda-ninja --output-on-failure
```
