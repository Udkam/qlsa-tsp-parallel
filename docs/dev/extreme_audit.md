# Extreme Optimization Audit

生成时间：2026-06-13  
当前分支：`final-extreme-polish`  
Git 初始状态：`main` 分支工作区干净；已切换到新分支 `final-extreme-polish`。  
归档目录：已创建 `docs/archive/`、`figures/archive/`、`results/archive/`。  
旧报告归档：已复制 `final_report.md`、`final_report_v2.md`、`final_report_v3.md`、`final_report_v4.md` 到 `docs/archive/`。

## 1. 当前代码模块列表

核心头文件位于 `include/tsp/`：

- `instance.hpp`：TSP 实例数据结构。
- `tsplib_parser.hpp`：TSPLIB95 parser 接口。
- `distance_matrix.hpp`：一维连续距离矩阵。
- `tour.hpp`：路径表示、初始化、合法性检查和 2-opt delta。
- `rng.hpp`：随机数封装。
- `timer.hpp`：计时工具。
- `sa.hpp`：串行 SA 参数与结果结构。
- `qlsa.hpp`：QLSA 参数与结果结构。
- `parallel.hpp`：多链并行参数、结果和调度接口。
- `cuda.hpp`：CUDA 后端接口。
- `metrics.hpp`：Gap 等指标工具。

核心源文件位于 `src/`：

- `tsplib_parser.cpp`
- `distance_matrix.cpp`
- `tour.cpp`
- `rng.cpp`
- `timer.cpp`
- `sa.cpp`
- `qlsa.cpp`
- `parallel.cpp`
- `cuda.cpp`
- `cuda_kernels.cu`
- `metrics.cpp`
- `main.cpp`

测试文件位于 `tests/`：

- `test_small_instance.cpp`
- `test_parallel.cpp`
- `test_cuda.cpp`
- `fixtures/square4.tsp`

## 2. 当前实验 CSV 列表

主要结果文件：

- `results/final_key_results.csv`
- `results/report_comparison_summary.csv`
- `results/step5_multi_cpu_raw.csv`
- `results/step5_multi_cpu_summary.csv`
- `results/step5_berlin52_raw.csv`
- `results/step5_berlin52_summary.csv`
- `results/tuning_raw.csv`
- `results/tuning_summary.csv`
- `results/tuned_validation_raw.csv`
- `results/tuned_validation_summary.csv`
- `results/targeted_quality_raw.csv`
- `results/targeted_quality_summary.csv`
- `results/openmp_scaling_grid_raw.csv`
- `results/openmp_scaling_grid_summary.csv`
- `results/cuda_scaling.csv`
- `results/omp_scaling.csv`
- `results/paper_table8_runtime.csv`
- `results/paper_hard_instance_quality.csv`
- `results/berlin52_manual_raw.csv`
- `results/berlin52_summary.csv`

## 3. 当前报告版本列表

- `docs/final_report.md`
- `docs/final_report_v2.md`
- `docs/final_report_v3.md`
- `docs/final_report_v4.md`
- `docs/final_experiment_summary.md`
- `docs/personal_report_appendix.md`
- `docs/final_submission_checklist.md`
- `docs/archive/final_report.md`
- `docs/archive/final_report_v2.md`
- `docs/archive/final_report_v3.md`
- `docs/archive/final_report_v4.md`

当前已有 `docs/final_report_v5_package/`，其中包含已打包的 v5 报告材料。后续本轮工作将新建 `docs/final_report_extreme.md` 和 `submission/`，不覆盖旧版本。

## 4. 当前图表列表

现有报告图表位于 `figures/`：

- `fig_architecture_pipeline.png`
- `fig_openmp_speedup.png`
- `fig_openmp_efficiency.png`
- `fig_default_gap.png`
- `fig_tuned_quality_improvement.png`
- `fig_paper_runtime_comparison_log.png`
- `fig_paper_quality_hard_instances.png`
- `fig_cuda_positioning.png`

短板：当前没有收敛曲线、完整 OpenMP threads scaling 图、policy comparison 图和 CUDA chains/block 补充图。若时间允许，可补充；若数据不足，应在 `figures/MISSING_FIGURES.md` 中说明。

## 5. 当前测试状态

已执行：

```powershell
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：

- `test_small_instance`：Passed
- `test_parallel`：Passed
- `test_cuda`：Passed
- 总计：3/3 tests passed

当前 `build-cuda-ninja` 目录存在，测试可直接运行。测试结果说明现有小实例、并行接口和 CUDA-facing smoke test 均通过。

## 6. 当前与论文对比数据状态

已具备论文对比数据：

- `results/paper_table8_runtime.csv`：来自论文 Table 8 的运行时间。
- `results/paper_hard_instance_quality.csv`：来自论文 hard-instance 质量表。
- `results/report_comparison_summary.csv`：论文数据与本项目结果的精简汇总。

当前报告已经说明：论文使用 Python/NumPy/Pandas + Xeon 环境，本项目使用 C++20 + OpenMP/CUDA + i5-12600KF/RTX 4070 SUPER；运行时间对比是参考对比，不是同硬件同语言公平 benchmark。

## 7. 当前最明显短板

1. 报告已有 v3/v4，但还可以进一步整合为最终 `extreme` 版，减少版本分散和重复表述。
2. 论文机制对齐仍可更系统，特别是 QLSA candidate set、SB-QLSA diversity state 与本项目实现差异。
3. 代码质量审查尚未形成独立文档，需要明确哪些已验证、哪些是测试缺口。
4. 当前没有收敛曲线和 policy comparison 图，报告的搜索过程分析仍偏弱。
5. CUDA 有工程实现和 smoke test，但当前小实例性能不优于 OpenMP，应保持谨慎定位。
6. 提交包需要统一整理，确保最终报告、图表、关键 CSV 和个人附录集中可交付。

## 8. 下一步改进计划

优先级按风险和收益排序：

1. 生成 `docs/paper_mechanism_alignment.md`，加强与参考论文机制对齐。
2. 生成 `docs/code_quality_review.md`，审查 parser、DistanceMatrix、2-opt、SA、QLSA、OpenMP、CUDA 和测试覆盖。
3. 不优先修改核心算法；若 paper-lite QLSA 风险较高，则生成 `docs/paper_lite_design_proposal.md` 和 `docs/qlsa_variant_design.md` 作为未来扩展说明。
4. 更新报告图脚本，若缺少新实验数据则生成 `figures/MISSING_FIGURES.md`。
5. 生成最终报告 `docs/final_report_extreme.md`，保持结论可追溯。
6. 强化 `docs/personal_report_appendix.md`。
7. 整理 `submission/` 提交包。
8. 新增 `scripts/check_final_submission.py` 并运行最终检查。
