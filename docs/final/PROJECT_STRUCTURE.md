# 项目结构说明

本文档说明提交级目录整理后的文件职责，便于课程提交和公开仓库维护。

## 文档目录

- `docs/final/`：最终文档目录，只保留课程报告、公开报告、个人报告、复现命令、已知限制和结构说明。
- `docs/dev/`：过程性设计与实验分析文档，包括各阶段设计文档、调参分析、论文机制对齐和代码质量审查。
- `docs/archive/`：旧版报告、历史提交包和不再作为主入口使用的文档。

最终报告以母版加两份派生版组织：

- `docs/final/final_report_master.md`：唯一主报告母版（12 节结构），基本信息脱敏，是 course/public 的内容来源。
- `docs/final/final_report_course.md`：课程提交版，允许包含姓名、学号等课程要求信息。
- `docs/final/final_report_public.md`：公开仓库版，不包含姓名、学号、邮箱等私人信息。

`docs/final/OPUS_AUDIT.md` 记录最近一次审计与整理过程。

## 图表目录

- `figures/final/`：最终报告正文引用的图表，文件名按正文顺序编号。
- `figures/archive/`：旧命名图、临时图和历史图表。

最终报告图表顺序为：

1. `fig01_architecture_pipeline.png`
2. `fig02_openmp_speedup.png`
3. `fig03_openmp_efficiency.png`
4. `fig04_default_gap.png`
5. `fig05_tuning_curve.png`
6. `fig06_policy_comparison.png`
7. `fig07_cuda_positioning.png`
8. `fig08_paper_runtime_comparison.png`
9. `fig09_paper_quality_comparison.png`

## 结果目录

- `results/final/`：最终报告主结论引用的关键 CSV 和结果索引。
- `results/raw/`：原始实验输出 CSV。
- `results/summary/`：由分析脚本生成的汇总 CSV。
- `results/reference/`：参考论文表格整理数据。
- `results/archive/`：历史索引、quick/smoke 或不作为主结论的归档结果。
- `results/traces/`：预算扫描或收敛相关补充数据。
- `results/logs/`：实验运行日志。

## 提交目录

- `submission/course/`：课程提交包，允许包含课程所需个人信息。
- `submission/public/`：公开仓库展示包，必须脱敏。

公开版本中不得出现姓名、学号、私人邮箱或其他个人联系方式。
