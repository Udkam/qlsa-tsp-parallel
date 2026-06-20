# final_report_v3 PDF 导出说明

## 1. 推荐导出方式

推荐使用 Typora 或 Pandoc 将 `docs/final_report_v3.md` 导出为 PDF。若使用 Typora，可直接打开 Markdown 文件后选择导出 PDF。若使用 Pandoc，可在项目根目录执行：

```powershell
pandoc docs\final_report_v3.md -o docs\final_report_v3.pdf
```

如需更稳定的中文字体效果，建议在本机安装常见中文字体，并在 Pandoc 模板或 Typora 主题中指定中文字体。

## 2. 图片路径要求

报告位于 `docs/` 目录，图片位于 `figures/` 目录，因此 Markdown 中图片路径使用：

```text
../figures/xxx.png
```

导出 PDF 前应确认 `figures/` 目录与 `docs/` 目录相对位置没有改变。

## 3. 表格过宽处理

`final_report_v3.md` 已将主要宽表拆分为 SA 和 QLSA 两张表，正常情况下不需要横向页面。如果导出后仍出现表格过宽，可以采用以下方式：

- 在 Typora 中选择较小字号或更紧凑主题；
- 将页面方向改为横向；
- 继续拆分表格，只保留关键列；
- 将完整数据放在 `results/*.csv`，正文仅展示摘要表。

## 4. 导出前检查命令

建议在导出 PDF 前运行：

```powershell
py scripts\make_report_figures.py
py scripts\check_report_assets.py
py scripts\check_report_format.py docs\final_report_v3.md
```

这些命令用于确认图片存在、报告不存在占位符、没有明显错误结论，并检查是否存在超过 7 列的 Markdown 表格。

## 5. 字体与页面建议

建议使用以下设置：

- 正文字体：宋体、微软雅黑或 Noto Sans CJK SC；
- 英文与代码字体：Consolas 或 JetBrains Mono；
- 页面边距：上下左右 2.0 cm 至 2.5 cm；
- 正文字号：10.5 pt 或 11 pt；
- 图表居中显示，图注紧跟图片下方；
- 若 PDF 中图片过大，可适当缩放图片或调整导出主题。

最终提交前建议人工翻阅 PDF，重点检查标题层级、图表分页、公式显示、表格换行和参考文献格式。
