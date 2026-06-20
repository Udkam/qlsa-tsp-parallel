# Final Submission README

## 1. 最终报告文件

推荐提交以下报告文件：

- `docs/final_report_extreme.md`：最终主报告。
- `docs/personal_report_appendix.md`：个人报告附录。
- `docs/final_known_issues.md`：已知限制与谨慎说明。

提交包中对应文件位于：

- `submission/final_report_extreme.md`
- `submission/personal_report_appendix.md`
- `submission/final_submission_readme.md`

## 2. 如何构建

推荐使用 Ninja + CUDA 构建：

```powershell
cmake -S . -B build-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DTSP_ENABLE_CUDA=ON
cmake --build build-cuda-ninja -j
```

如果本机 CUDA toolset 不可用，可使用普通 Release 构建：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 3. 如何运行测试

```powershell
ctest --test-dir build-cuda-ninja --output-on-failure
```

如果使用普通构建目录：

```powershell
ctest --test-dir build --output-on-failure
```

## 4. 如何复现实验

默认参数多实例实验：

```powershell
py scripts\run_step5_experiments.py --instances berlin52 eil51 st70 eil76 rat99 eil101 --iterations 1000000 --repeat 3 --chains 32 --threads 8 --no-cuda --output results\step5_multi_cpu_raw.csv
```

调优验证：

```powershell
scripts\run_tuned_validation.bat
```

定向增强：

```powershell
scripts\run_targeted_quality.bat
```

policy comparison：

```powershell
py scripts\run_policy_comparison_selected.py --output results\policy_comparison_raw.csv
py scripts\analyze_policy_comparison.py
```

OpenMP scaling：

```powershell
py scripts\run_openmp_scaling_grid.py --instances berlin52 eil101 --iterations 1000000 --repeat 3 --chains 32 64 --threads 1 2 4 8 12 16 --raw-output results\openmp_scaling_final_raw.csv --summary-output results\openmp_scaling_final_summary.csv --markdown docs\openmp_scaling_final_analysis.md
```

## 5. 关键结果文件

`submission/results_key/` 中包含：

- `final_key_results.csv`
- `step5_multi_cpu_summary.csv`
- `tuned_validation_summary.csv`
- `targeted_quality_summary.csv`
- `paper_table8_runtime.csv`
- `paper_hard_instance_quality.csv`
- `report_comparison_summary.csv`
- `policy_comparison_summary.csv`
- `openmp_scaling_final_summary.csv`

## 6. 图表目录

最终报告引用的图片位于：

- `submission/figures/`

根目录原始图表位于：

- `figures/`

## 7. 注意事项

- CUDA backend 已完成工程实现，但当前小规模实例上不是主要加速结论。
- QLSA 在 `rat99` high-budget 配置上达到 BKS，但不能声称所有实例都优于 SA。
- 本项目不是逐项实现论文全部 SB-QLSA 机制。
- 论文运行时间对比是参考对比，不是同硬件同语言严格性能比较。
- 最终提交前建议运行 `py scripts\check_final_submission.py` 和 `ctest`。
