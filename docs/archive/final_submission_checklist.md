# 最终提交检查清单

## 1. 基本信息

- [ ] 已在 `docs/final_report_v2.md` 中填写姓名：陈乐浚。
- [ ] 已在 `docs/final_report_v2.md` 中填写学号：22361054。
- [ ] 已在 `docs/final_report_v2.md` 中填写学院/专业：中山大学计算机学院 / 信息与计算科学。
- [ ] 已确认团队人数为 1 人。
- [ ] 已补充 CPU 型号：12th Gen Intel(R) Core(TM) i5-12600KF。
- [ ] 已补充 GPU 型号：NVIDIA GeForce RTX 4070 SUPER。
- [ ] 已确认报告中不存在 `<你的姓名>`、`<你的学号>`、`<你的专业>`、TODO、请补充等占位符。

## 2. 文件完整性

- [ ] 已确认 `docs/final_report_v2.md` 存在。
- [ ] 已确认 `docs/final_report_v2_review_notes.md` 存在。
- [ ] 已确认 `docs/personal_report_appendix.md` 存在。
- [ ] 已确认 `docs/final_submission_checklist.md` 存在。
- [ ] 已确认 `docs/final_experiment_summary.md` 存在。
- [ ] 已确认 `results/final_key_results.csv` 存在。
- [ ] 已确认 `results/paper_table8_runtime.csv` 存在。
- [ ] 已确认 `results/paper_hard_instance_quality.csv` 存在。
- [ ] 已确认 `results/report_comparison_summary.csv` 存在。
- [ ] 已确认 `figures/` 下 8 张报告图均已生成。
- [ ] 已确认代码、配置文件、实验 CSV 和 Markdown 分析文档均已保存。

## 3. 报告图表与资产

- [ ] 已运行 `py scripts\make_report_figures.py`。
- [ ] 已运行 `py scripts\check_report_assets.py`。
- [ ] 已确认 `docs/final_report_v2.md` 中所有图片路径都能解析到本地文件。
- [ ] 已确认图表没有引用未实际生成的数据。
- [ ] 已确认大表格只保留关键列，完整数据位于 `results/*.csv`。

## 4. 实验结论边界

- [ ] 已确认报告没有宣称默认参数下全部实例均达到 BKS。
- [ ] 已确认报告没有宣称 CUDA backend 是当前主要加速来源。
- [ ] 已确认报告没有宣称 QLSA 在所有实例上都优于 SA。
- [ ] 已确认报告没有把不同硬件下的论文运行时间对比写成严格公平 benchmark。
- [ ] 已确认报告没有把 tuning search best 直接当作独立验证结论。
- [ ] 已确认报告没有夸大 Step 6C targeted high-budget 结果，并同时说明了运行时间成本。
- [ ] 已确认报告明确说明本项目未完全复刻论文 SB-QLSA 的全部 candidate-leader 与 diversity-state 机制。
- [ ] 已确认 OpenMP multi-chain 被作为最终主要性能提升结论。
- [ ] 已确认 CUDA backend 被描述为已完成并验证的工程扩展。

## 5. 数据一致性

- [ ] 已确认 SA OpenMP 平均 speedup 约 5.46x。
- [ ] 已确认 QLSA OpenMP 平均 speedup 约 4.98x。
- [ ] 已确认 SA OpenMP 平均 parallel efficiency 约 68.28%。
- [ ] 已确认 QLSA OpenMP 平均 parallel efficiency 约 62.29%。
- [ ] 已确认 `rat99` QLSA high-budget 达到 BKS=1211。
- [ ] 已确认 `rat99` SA high-budget 最好为 1212。
- [ ] 已确认 `eil101` SA/QLSA 在定向增强实验中均达到 BKS=629。
- [ ] 已确认 `docs/final_report_v2.md` 中表格数值与 `results/final_key_results.csv`、`results/report_comparison_summary.csv` 和阶段分析文档一致。

## 6. 个人报告

- [ ] 已附上 `docs/personal_report_appendix.md`。
- [ ] 已说明团队人数为 1 人。
- [ ] 已说明本人承担选题、论文阅读、C++ 工程设计、TSPLIB95 parser、SA/QLSA 实现、OpenMP 并行、CUDA 工程实现、实验脚本、参数调优、结果分析和报告撰写。

## 7. 最终提交前建议命令

如果使用普通构建目录：

```powershell
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```

如果使用 Ninja/CUDA 构建目录：

```powershell
cmake --build build-cuda-ninja --parallel
ctest --test-dir build-cuda-ninja --output-on-failure
```

报告资产检查：

```powershell
py scripts\make_report_figures.py
py scripts\check_report_assets.py
```

提交前检查工作区：

```powershell
git status --short
```
