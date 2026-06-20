# 复现实验命令

本文档给出最终报告相关结果的主要复现命令。所有命令均在项目根目录执行。

## 1. 构建与测试

```bat
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON
cmake --build build-cuda-ninja -j
ctest --test-dir build-cuda-ninja --output-on-failure
```

如果当前环境无法启用 CUDA，可先使用 CPU/OpenMP 构建验证主体功能：

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 2. 默认参数多实例实验

```bat
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\step5_multi_cpu_raw.csv
```

该命令生成默认参数下 serial multi-chain 与 OpenMP multi-chain 的 CSV，并自动调用分析脚本生成 summary 和 Markdown 分析。

## 3. 调优参数独立验证

```bat
scripts\run_tuned_validation.bat
```

该命令读取 `configs/tuned_params.json`，使用独立 seed 从 101 开始进行 repeat=10 验证，输出：

- `results/tuned_validation_raw.csv`
- `results/tuned_validation_summary.csv`
- `docs/step6B_tuned_validation_analysis.md`

## 4. 定向增强实验

```bat
scripts\run_targeted_quality.bat
```

该命令读取 `configs/targeted_quality_configs.json`，扩大 chains 或 iterations 预算，输出：

- `results/targeted_quality_raw.csv`
- `results/targeted_quality_summary.csv`
- `docs/step6C_targeted_quality_analysis.md`

## 5. Policy Comparison 补充实验

```bat
py scripts\run_policy_comparison_selected.py
py scripts\analyze_policy_comparison.py --input results\policy_comparison_raw.csv --summary results\policy_comparison_summary.csv --markdown docs\policy_comparison_analysis.md --figure figures\final\fig_policy_comparison.png
```

该实验比较本实现中 QLSA 的 epsilon-greedy 与 softmax 策略。由于本实现的 action/state 机制与论文 candidate leader 机制不同，该结果仅作为本项目内部策略对比。

## 6. 图表生成与报告检查

```bat
py scripts\make_report_figures.py
py scripts\plot_convergence_traces.py
py scripts\check_report_format.py docs\final\final_report.md
py scripts\check_report_assets.py
```

最终报告主文件为：

- `docs/final/final_report.md`

提交级目录为：

- `docs/final_submission_v2/`
