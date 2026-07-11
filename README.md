# Parallel TSP SA/QLSA

本仓库是并行算法课程大作业工程。项目以 TSP 为应用背景，实现了 SA、QLSA、OpenMP 多搜索链并行、CUDA 后端和 MPI + OpenMP 双虚拟机分布式实验。

- 复现实验命令：`docs/final/reproduction_commands.md`
- 局限性说明：`docs/final/known_limitations.md`
- 关键结果索引：`results/final/RESULTS_INDEX.md`
- 报告图片：`figures/`

## 工程内容

- C++20 TSPLIB95 数据解析；
- 一维连续存储的 `DistanceMatrix`；
- 路径表示、最近邻初始化和 2-opt O(1) 增量计算；
- 串行 SA 与 QLSA；
- OpenMP 多搜索链并行；
- CUDA 多链模式、候选批量评价、候选策略和并行路径反转；
- QLSA 机制对齐变体：`current`、`paper`、`paper-sb`；
- SA/QLSA 可暂停、分块续跑和共享 wall-clock deadline；
- OpenMP island 模型：`independent`、`ring`、`global` 三种拓扑；
- MPI + OpenMP 分布式多链搜索；
- 严格后端/线程核验、paired-seed 公平实验、统计分析和报告检查脚本。

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

共享 30 秒 solver 预算（所有链使用同一个绝对 deadline）：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\eil101.tsp --parallel omp --chains 8 --threads 8 --iterations 1000000000 --time-limit-ms 30000 --migration-interval 10000 --seed 1 --csv-only
```

OpenMP island migration：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input data\eil101.tsp --parallel omp --chains 8 --threads 8 --iterations 1000000 --migration-topology ring --migration-interval 10000 --seed 1 --csv-only
```

CSV 保留原有 14 列前缀，并追加 total/kernel 计时、requested/actual backend、fallback 原因、实际迭代数、deadline、迁移统计和 `actual_threads`。CUDA 的 `total_elapsed_ms` 覆盖首次设备探测、分配、传输、kernel、回传、释放与结果校验；`cuda_kernel_elapsed_ms` 只表示 kernel event 时间。

## 公平实验与岛模型消融

正式 runner 要求显式指定刚构建的可执行文件，避免误用旧构建。默认公平矩阵使用 20 个 paired seeds，比较等搜索迭代与固定 solver wall time 两种互补预算；算法在每个 seed 内按可复现循环顺序执行。

```powershell
python scripts\run_fair_experiments.py --executable build-cuda-ninja\tsp_sa.exe --budget equal-iterations --instances eil76 rat99 eil101 --run-id fair_equal_iterations
python scripts\run_fair_experiments.py --executable build-cuda-ninja\tsp_sa.exe --budget fixed-time --instances eil76 rat99 eil101 --run-id fair_fixed_time
python scripts\analyze_paired_experiments.py --input results\fair_experiments\fair_equal_iterations --output-dir results\fair_experiments\fair_equal_iterations_analysis
```

分析器默认只接受四算法完整的共同 seed block，并输出 bootstrap CI、配对 Wilcoxon、精确 sign test、Friedman 与 Holm 校正；只有探索性分析才使用 `--allow-incomplete-pairs`。

岛模型消融默认比较 SA 与 `paper-sb` 在 `independent`、`ring`、`global` 下的结果，并统计迁移尝试、采纳率、质量差与运行时：

```powershell
python scripts\run_island_ablation.py --executable build-cuda-ninja\tsp_sa.exe --run-id island_ablation
python scripts\analyze_island_ablation.py results\island_ablation\island_ablation --output-dir results\island_ablation\island_ablation_analysis
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
python -S -m unittest -v tests.test_fair_experiment_pipeline tests.test_fair_runner_validation tests.test_fair_analyzer_integrity tests.test_island_ablation_pipeline tests.test_island_runner_hardening tests.test_island_analyzer_hardening
```
