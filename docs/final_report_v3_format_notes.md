# final_report_v3 格式重排说明

## 1. 本次重排目标

`docs/final_report_v3.md` 是在 `docs/final_report_v2.md` 基础上整理出的正式提交版。重排重点不是新增实验结论，而是优化报告结构、图表格式、表格宽度和论文对比表述，使其更适合课程大作业提交和 PDF 导出。

## 2. 相比 v2 的主要修改

- 将“摘要”调整为不编号章节，避免出现“0. 基本信息”这类不正式结构。
- 将“基本信息”改为表格形式，便于提交材料检查。
- 新增“预期目标与实际完成情况对照”，明确区分已完成、部分完成和工程扩展。
- 将工程实现与并行化设计合并，减少重复描述。
- 将默认参数实验的大表拆分为 SA 和 QLSA 两张表，避免 PDF 中横向过宽。
- 为所有图片补充独立图注，不仅依赖 Markdown alt text。
- 将论文对比改为独立大节，并增加“对比口径声明”和“小结”，强调非同硬件、非同语言的参考对比。
- 将 CUDA 表述进一步收敛为“工程扩展完成，性能优化仍有空间”，避免将其写成主性能结论。

## 3. 图表格式处理

报告中使用的 8 张图均来自 `figures/`：

1. `fig_architecture_pipeline.png`
2. `fig_paper_runtime_comparison_log.png`
3. `fig_paper_quality_hard_instances.png`
4. `fig_openmp_speedup.png`
5. `fig_openmp_efficiency.png`
6. `fig_default_gap.png`
7. `fig_tuned_quality_improvement.png`
8. `fig_cuda_positioning.png`

所有图片引用路径均保持为 `../figures/xxx.png`，因为报告位于 `docs/` 目录。

## 4. 表格格式处理

本次重排遵循以下规则：

- 每张表前均有“表 X：...”说明。
- 每张图下均有“图 X：...”图注。
- Markdown 表格尽量控制在 7 列以内。
- 宽表拆分为更小表格，例如默认参数结果拆成 SA 与 QLSA 两张表。
- 不在正文堆叠完整 CSV，只保留报告需要的关键列。

## 5. 结论边界

v3 报告继续保留以下谨慎边界：

- 不声称 CUDA 快于 OpenMP。
- 不声称 QLSA 总是优于 SA。
- 不声称默认参数下所有实例均达到 BKS。
- 不声称完全复刻论文 SB-QLSA。
- 不把论文运行时间对比描述为严格同平台性能基准。
- 不新增未实际运行过的实验数据。

## 6. 建议检查命令

```powershell
py scripts\check_report_assets.py
py scripts\check_report_format.py docs\final_report_v3.md
```

若后续继续修改报告，建议修改后重新运行以上两个检查脚本。
