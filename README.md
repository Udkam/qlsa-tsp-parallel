# TSP Q-Learning Assisted Simulated Annealing Parallelization

面向旅行推销员问题（TSP）的 Q-Learning 辅助模拟退火算法并行化实现与性能优化。项目基于 2026 年 QLSA for TSP 论文思想，完成 C++20 工程化实现，并提供 OpenMP 多链并行、CUDA 后端工程验证、自动化实验与最终报告材料。

## 核心结论

- **OpenMP 是主要性能结论**：默认参数多实例实验中，SA OpenMP 平均 speedup 约 **5.46x**，QLSA OpenMP 平均 speedup 约 **4.98x**。
- **QLSA 是质量增强结论**：在 targeted high-budget 实验中，`rat99` 上 QLSA 达到 BKS=1211，而 SA high-budget 最好为 1212。
- **CUDA 是工程扩展结论**：CUDA 后端已完成真实构建和运行验证，但在 berlin52 等小规模实例上不优于 OpenMP，因此不作为主要加速证据。
- **论文对比是参考对比**：参考论文使用 Python/Xeon 环境，本项目使用 C++20/OpenMP/CUDA 和本地硬件，运行时间不可解释为同平台严格 benchmark。

## 工程内容

- C++20 + CMake 工程；
- TSPLIB95 `.tsp` parser，支持坐标型和显式矩阵型实例；
- `EUC_2D`、`CEIL_2D`、`GEO`、`ATT`、`EXPLICIT` 距离支持；
- 一维连续数组 `DistanceMatrix`，便于 CPU cache 访问和 CUDA 拷贝；
- Tour 合法性检查、nearest-neighbor 初始化、random 初始化、O(1) 2-opt delta；
- 串行 SA baseline；
- Q-Learning Assisted SA，支持 epsilon-greedy 和 Softmax policy；
- OpenMP chain-level multi-chain 并行；
- CUDA multi-chain 后端；
- 实验脚本、CSV 汇总、图表生成、报告检查和提交包整理。

## 目录结构

```text
include/tsp/        C++ 头文件
src/                C++/CUDA 实现
tests/              assert-based 测试
scripts/            实验、分析、图表和检查脚本
configs/            调优与定向增强配置
data/               本地 TSPLIB95 数据目录，.tsp 文件默认不入库
results/final/      最终报告关键 CSV 与 RESULTS_INDEX.md
results/raw/        原始实验 CSV
results/summary/    汇总 CSV
results/reference/  论文表格参考数据
figures/final/      最终报告图表
docs/final/         最终报告、manifest、复现命令和限制说明
submission/public/  脱敏公开展示包
```

详细结构说明见：

- `docs/final/PROJECT_STRUCTURE.md`
- `results/final/RESULTS_INDEX.md`
- `docs/final/REPORT_MANIFEST.md`

## 构建

推荐 CUDA/Ninja 构建：

```powershell
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON
cmake --build build-cuda-ninja -j
```

CPU/OpenMP 构建：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

若 CUDA 构建失败，应先确认 Ninja、MSVC 环境和 `nvcc` 是否可用。CUDA fallback 或无法真实编译 kernel 的结果不能作为 GPU 性能证据。

## 测试

```powershell
ctest --test-dir build-cuda-ninja --output-on-failure
```

测试不依赖外部 TSPLIB 数据，包含小实例、并行路径和 CUDA smoke test。

## 数据准备

`data/*.tsp` 默认不提交到仓库。可以使用脚本尝试下载：

```powershell
bash scripts/download_tsplib_subset.sh
```

也可以手动下载 TSPLIB95 `.tsp` 文件并放入 `data/`，例如：

```text
data/berlin52.tsp
data/eil76.tsp
data/rat99.tsp
data/eil101.tsp
```

## 单次运行示例

SA + OpenMP：

```powershell
.\build-cuda-ninja\tsp_sa.exe --input data/berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --seed 1 --init nn
```

QLSA + OpenMP：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data/berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

QLSA + CUDA：

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --input data/berlin52.tsp --parallel cuda --chains 32 --cuda_block_size 128 --iterations 1000000 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

程序输出包含人类可读摘要和 CSV 行。CSV 字段包括：

```text
algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves
```

## 实验复现

最终报告相关复现命令集中记录在：

```text
docs/final/reproduction_commands.md
```

常用命令：

```powershell
py scripts\make_report_figures.py
py scripts\check_privacy_and_encoding.py
py scripts\check_report_assets.py
py scripts\check_report_format.py docs\final\final_report_master_v2.md
ctest --test-dir build-cuda-ninja --output-on-failure
```

默认参数多实例实验：

```powershell
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\raw\step5_multi_cpu_raw.csv
```

调优验证与定向增强：

```powershell
scripts\run_tuned_validation.bat
scripts\run_targeted_quality.bat
```

## 最终报告与提交包

主报告入口：

```text
docs/final/final_report_master_v2.md
```

报告 manifest：

```text
docs/final/REPORT_MANIFEST.md
```

公开展示包：

```text
submission/public/
```

课程提交包：

```text
submission/course/
```

`submission/course/` 可能包含课程要求的个人信息，默认不建议提交到公开仓库。公开仓库应使用 `submission/public/` 和 `docs/final/final_report_public.md`。

## 已知限制

- 当前 CUDA 后端未充分优化，小规模 TSPLIB95 实例上不优于 OpenMP。
- 当前 QLSA 是基于论文思想的工程化变体，不是完整 SB-QLSA candidate-leader + diversity-state 复刻。
- policy comparison 比较的是本实现中的 epsilon-greedy 与 Softmax，不等同于论文 Softmax 机制复现。
- 与论文 Table 8 的运行时间对比涉及不同语言、不同硬件和不同实现，只能作为参考对比。
- 预算扫描不能表述为真实逐迭代 trace。

完整限制见：

```text
docs/final/known_limitations.md
```

## 提交前检查

```powershell
py scripts\check_privacy_and_encoding.py
py scripts\check_report_assets.py
py scripts\check_report_format.py docs\final\final_report_master_v2.md
py scripts\check_final_submission.py
ctest --test-dir build-cuda-ninja --output-on-failure
```
