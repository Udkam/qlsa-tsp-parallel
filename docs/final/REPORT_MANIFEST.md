# Report Manifest

本文档定义最终报告入口和历史版本使用规则，避免继续出现多个 `final_report_vX.md` 并存导致的提交混乱。

## 唯一主报告入口

| 文件 | 状态 | 用途 |
|---|---|---|
| `docs/final/final_report_master.md` | 主入口 | 最终报告叙事母版。所有课程版、公开版和提交包报告均应以此为内容来源。 |

## 最终派生输出

| 文件 | 状态 | 用途 |
|---|---|---|
| `docs/final/final_report_course.md` | 派生输出 | 课程提交版，可以包含姓名、学号等课程要求信息。 |
| `docs/final/final_report_public.md` | 派生输出 | 公开仓库版，必须移除姓名、学号、邮箱等私人信息。 |
| `submission/course/final_report.md` | 提交包输出 | 课程提交包中的报告副本。 |
| `submission/public/final_report_public.md` | 提交包输出 | 公开展示包中的脱敏报告副本。 |

## 历史版本

以下文件或目录只作为历史记录，不再作为最终报告入口使用：

- `docs/archive/final_report.md`
- `docs/archive/final_report_v2.md`
- `docs/archive/final_report_v3.md`
- `docs/archive/final_report_v4.md`
- `docs/archive/final_report_v5_package/`
- `docs/archive/final_report_extreme.md`
- `docs/archive/final_submission_v2/`

如果后续需要修改最终报告，应优先修改 `docs/final/final_report_master.md`，再同步生成 course/public 派生版本。

## 不再使用的入口

| 文件 | 处理方式 |
|---|---|
| `final_report_extreme.md` | 已废弃，若存在只能放在 `docs/archive/`。 |
| `final_report_v3.md` / `final_report_v4.md` / `final_report_v5.md` | 已废弃，若存在只能放在 `docs/archive/`。 |
| `docs/final/final_report_course.md` | 不是主入口，只是 course 派生输出。 |
| `docs/final/final_report_public.md` | 不是主入口，只是 public 派生输出。 |

## 图表与数据来源

- 最终报告图表统一来自 `figures/final/fig01_*.png` 到 `figures/final/fig09_*.png`。
- 最终报告主结果数据来自 `results/final/`、`results/summary/` 和 `results/reference/`。
- 任何新结论必须能追溯到现有 CSV 或参考论文表格，不得手工新增未运行数据。
