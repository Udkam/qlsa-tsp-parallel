# Parallel TSP SA/QLSA

本仓库是并行算法课程大作业工程：面向旅行推销员问题（TSP）的模拟退火（SA）、Q-Learning 辅助模拟退火（QLSA）及其 OpenMP / CUDA / MPI + OpenMP 并行实现。

## 当前保留入口

- 课程报告：`docs/final/final_report_course.md`
- 课程报告 PDF：`docs/final/final_report_course.pdf`
- 唯一提交包：`submission/course/`
- 关键结果索引：`results/final/RESULTS_INDEX.md`
- 复现命令：`docs/final/reproduction_commands.md`

旧版 master/public 报告、历史 dev 文档、临时日志、浏览器/GPT 材料、旧 submission 目录和构建产物已清理。

## 工程内容

- C++20 TSPLIB95 parser
- 连续一维 `DistanceMatrix`
- Tour 表示、nearest-neighbor 初始化、2-opt O(1) delta
- 串行 SA 与 QLSA
- OpenMP multi-chain
- CUDA chain mode
- CUDA candidate-level 2-opt evaluation
- CUDA candidate policy (`best` / `random` / `hybrid`)
- CUDA QLSA candidate mode
- 可选 QLSA 机制对齐变体：`current` / `paper` / `paper-sb`
- MPI + OpenMP hybrid backend
- 自动实验脚本、CSV 汇总、图表生成和报告检查脚本

## 构建

Windows + Ninja + CUDA：

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON -DTSP_ENABLE_MPI=ON"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build build-cuda-ninja -j"
ctest --test-dir build-cuda-ninja --output-on-failure
```

如果本机没有 MPI，CMake 会自动禁用 MPI target；CUDA/OpenMP 主程序仍可构建。

## 运行示例

OpenMP：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --repeat 3 --seed 1 --init nn
```

QLSA + OpenMP：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data\berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --repeat 3 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

QLSA candidate-leader / diversity-state 变体：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input data\berlin52.tsp --iterations 100000 --seed 1 --init nn
```

CUDA candidate：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn
```

CUDA candidate random policy：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidate_policy random --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn
```

CUDA candidate hybrid policy：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidate_policy hybrid --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn
```

QLSA CUDA candidate：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data\a280.tsp --parallel cuda --cuda_mode candidate --cuda_candidates_per_iter 128 --cuda_block_size 128 --cuda_reversal_mode parallel --chains 64 --iterations 500000 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

## 关键实验结果位置

- 默认多实例 OpenMP：`results/summary/step5_multi_cpu_summary.csv`
- 调优与定向增强：`results/summary/tuned_validation_summary.csv`，`results/summary/targeted_quality_summary.csv`
- CUDA QLSA candidate：`results/summary/cuda_qlsa_candidate_summary.csv`
- CUDA parallel reversal：`results/summary/cuda_reversal_summary.csv`
- 大实例 OpenMP：`results/summary/large_openmp_l1_summary.csv`
- 大实例 CUDA：`results/summary/large_cuda_formal_summary.csv`
- CUDA hybrid quick：`results/summary/cuda_candidate_hybrid_quick_summary.csv`
- 双 VM MPI：`results/summary/mpi_vm_scaling_formal_summary.csv`，`results/summary/large_mpi_vm_formal_aggressive_summary.csv`
- 论文参考数据：`results/reference/paper_table8_runtime.csv`，`results/reference/paper_hard_instance_quality.csv`

## 结论边界

- OpenMP multi-chain 是当前主要性能结论。
- CUDA candidate-level evaluation 是工程扩展和质量/搜索覆盖率探索，不作为默认主性能后端。
- CUDA candidate policy 中 `best` 更偏向批量择优，`random` 更接近随机候选提案，`hybrid` 用于观察两者折中；三者都不改变 OpenMP 是主性能结论的定位。
- MPI + OpenMP 结果来自双 Ubuntu VM 和真实 `mpirun`，用于证明分布式工程链路，不等同生产 HPC benchmark。
- QLSA 在部分实例上提升解质量，但不能表述为总是优于 SA。
- C++ 已提供可选 `paper` / `paper-sb` 机制对齐变体；已有主实验仍主要使用 `current` 变体，不能把历史结果改写成 paper-sb 结论。

## 提交前检查

```powershell
py scripts\check_report_format.py docs\final\final_report_course.md
py scripts\check_report_assets.py
py scripts\check_privacy_and_encoding.py
```

若需要重新验证程序：

```powershell
ctest --test-dir build-cuda-ninja --output-on-failure
```
