# TSP Q-Learning Assisted SA Parallelization

本项目用于并行算法期末大作业：**面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化**。

目标是在 C++ 底层实现 TSP 的模拟退火、Q-Learning 辅助模拟退火，并逐步加入 OpenMP/CUDA 多链并行优化。当前阶段已完成串行 SA、串行 QLSA、OpenMP 多链并行、CUDA 多链并行工程实现，以及 Step 5A 的实验自动化与 berlin52 结果统计。

## 当前完成内容

- C++20 + CMake 工程骨架；
- TSPLIB95 `.tsp` 解析器；
- `EUC_2D`、`CEIL_2D`、`GEO`、`ATT`、`EXPLICIT` 距离支持；
- 一维连续数组 `DistanceMatrix`；
- Tour 合法性检查、nearest-neighbor 初始化、路径长度计算、2-opt delta；
- 串行 SA + 2-opt 基线；
- 串行 Q-Learning-Assisted SA (QLSA)；
- OpenMP 多搜索链并行 SA/QLSA；
- CUDA 多搜索链并行 SA/QLSA；
- 实验脚本、日志归档、统计汇总和 Markdown 分析文档生成。

## 构建

默认尝试启用 OpenMP 和 CUDA：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Visual Studio 多配置生成器下建议显式构建 Release：

```bash
cmake --build build --config Release --parallel
```

CUDA 构建推荐使用 Ninja，例如已有的 `build-cuda-ninja/tsp_sa.exe`。如果检测到 CUDA fallback warning，则该结果不能作为真实 GPU 数据。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

测试不依赖外部 TSPLIB 数据，会使用 `tests/fixtures/square4.tsp`。

## 基本运行

串行 SA：

```bash
./build/tsp_sa --input data/berlin52.tsp --iterations 1000000 --seed 1 --init nn
```

串行 QLSA：

```bash
./build/tsp_sa --qlsa --input data/berlin52.tsp --iterations 1000000 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

OpenMP 多链 SA：

```bash
./build/tsp_sa --input data/berlin52.tsp --parallel omp --chains 32 --threads 8 --iterations 1000000 --seed 1 --init nn
```

CUDA 多链 QLSA：

```bash
./build/tsp_sa --qlsa --input data/berlin52.tsp --parallel cuda --chains 32 --cuda_block_size 128 --iterations 1000000 --seed 1 --init nn --alpha 0.1 --gamma 0.9 --epsilon 0.1 --policy epsilon-greedy
```

CSV 字段：

```text
algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves
```

## Step 5A 实验结果与自动化

手动 berlin52 结果已归档：

```text
results/berlin52_manual_raw.csv
results/berlin52_summary.csv
docs/step5_berlin52_analysis.md
```

重新统计手动结果：

```bash
python scripts/analyze_results.py --input results/berlin52_manual_raw.csv --output results/berlin52_summary.csv --markdown docs/step5_berlin52_analysis.md
```

一键运行 quick smoke test：

```bash
python scripts/run_step5_experiments.py --quick
```

在 Windows 上如果 `python` 指向 Microsoft Store alias，可改用：

```powershell
py scripts\run_step5_experiments.py --quick
```

Windows 包装脚本：

```bat
scripts\run_step5_quick.bat
```

一键运行 berlin52 完整实验：

```bat
scripts\run_step5_berlin52.bat
```

一键运行多个实例：

```bash
python scripts/run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --cuda-block-size 128 --output results/step5_multi_raw.csv
```

实验脚本会：

- 优先使用 `build-cuda-ninja/tsp_sa.exe`；
- 保存完整日志到 `results/logs/`；
- 从 stdout 抽取 CSV 数据行；
- 检测 fallback warning；
- 自动调用 `scripts/analyze_results.py` 生成 summary CSV 和 Markdown。

当前 berlin52 结果说明：

- berlin52 BKS 为 7542；
- 当前 SA/QLSA 串行多链、OpenMP、CUDA 版本均达到 best_length=7542，Gap=0%；
- OpenMP 是当前 berlin52 上的主要加速结果；
- CUDA 已完成工程扩展并可真实运行，但在 berlin52 小规模实例上可能受 kernel 启动、访存和每链工作量不足影响，暂未优于 OpenMP。

## Step 6A 参数调优与 OpenMP 扩展性实验

快速调参 smoke test：

```powershell
py scripts\tune_params.py --quick
```

分析调参结果：

```powershell
py scripts\analyze_tuning.py --input results\tuning_raw.csv
```

SA/QLSA 全量调参：

```powershell
py scripts\tune_params.py --instances eil76 rat99 eil101 --algorithm both --stage all --iterations 1000000 --repeat 3 --chains 32 --threads 8 --output results\tuning_raw.csv
```

OpenMP 扩展性实验：

```powershell
py scripts\run_openmp_scaling_grid.py
```

OpenMP 扩展性 quick smoke test：

```powershell
py scripts\run_openmp_scaling_grid.py --quick
```

输出文件：

```text
results/tuning_raw.csv
results/tuning_summary.csv
docs/step6A_tuning_analysis.md
results/openmp_scaling_grid_raw.csv
results/openmp_scaling_grid_summary.csv
docs/step6A_openmp_scaling_analysis.md
```

说明：quick 只用于验证脚本链路，不代表调参已经完成。完整结论需要运行全量调参和完整 OpenMP scaling grid。

## Step 6B 调优参数固化与独立验证

Step 6A 是参数搜索；Step 6B 将搜索得到的参数固化到配置文件，并使用独立 seed 进行验证，避免只引用 tuning search 中挑出来的 best result。最终报告应优先引用 Step 6B 的验证结果。

调优参数文件：
```text
configs/tuned_params.json
```

快速验证：
```bat
scripts\run_tuned_validation_quick.bat
```

完整独立验证：
```bat
scripts\run_tuned_validation.bat
```

输出文件：
```text
results/tuned_validation_raw.csv
results/tuned_validation_summary.csv
docs/step6B_tuned_validation_analysis.md
```

说明：完整验证默认使用 `repeat=10`、`seed=101`，即每个配置使用独立的 seed=101 到 seed=110；quick 模式只运行 eil76 的 SA/QLSA 各一次，用于验证脚本链路。

## Step 6C 定向增强实验

Step 6C 不做新的全量网格调参，而是在 Step 6B 较优配置附近增加搜索预算，重点观察 `eil101` 和 `rat99` 在更大 `chains` / `iterations` 下的解质量稳定性。`chains` 增大表示一次实验启动更多独立搜索链，`iterations` 增大表示单条链搜索更充分，两者都可能提高找到更好解的概率，但会增加运行时间。

配置文件：
```text
configs/targeted_quality_configs.json
```

快速测试：
```bat
scripts\run_targeted_quality_quick.bat
```

完整运行：
```bat
scripts\run_targeted_quality.bat
```

输出文件：
```text
results/targeted_quality_raw.csv
results/targeted_quality_summary.csv
docs/step6C_targeted_quality_analysis.md
```

说明：quick 模式只运行 `eil101 SA` 的一个小预算配置，用于验证脚本链路；完整结论需要运行 `repeat=5` 的完整定向增强实验。

## Step 6D 最终实验结果汇总

最终报告应优先引用以下汇总文件，而不是从各阶段 raw CSV 中手动摘抄：

```text
docs/final_experiment_summary.md
results/final_key_results.csv
```

其中 `docs/final_experiment_summary.md` 汇总 Step 5B 默认参数 OpenMP 加速结果、Step 6B 独立验证质量结果、Step 6C 定向增强质量结果，并明确 CUDA 结果在最终报告中的定位；`results/final_key_results.csv` 提供可直接复制到报告表格中的关键数值。

## 其他批量脚本

串行 baseline：

```bash
bash scripts/run_baseline.sh
```

OpenMP scaling：

```bash
bash scripts/run_omp_scaling.sh
```

CUDA scaling：

```bash
bash scripts/run_cuda_scaling.sh
```

## 后续计划

- 扩展 eil51、st70、eil76、rat99、eil101 等实例实验；
- 优化 CUDA block 内候选 2-opt 并行评价；
- 整理论文对比表、speedup、parallel efficiency、Gap 和最终报告。
