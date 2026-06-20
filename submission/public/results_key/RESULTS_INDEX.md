# 实验结果索引

本文档说明最终报告中使用的 CSV 数据来源、用途和归档位置。所有 CSV 均来自已经运行的实验脚本或参考论文表格整理，不包含编造数据。

## 主结论数据

| 文件 | 位置 | 用途 |
|---|---|---|
| `final_key_results.csv` | `results/final/` | 汇总 Step 5B、Step 6B、Step 6C 的关键结论，用于最终报告主表格。 |
| `report_comparison_summary.csv` | `results/final/` | 汇总论文参考数据与本实现结果的对比数据。 |

## 汇总数据

| 文件 | 位置 | 用途 |
|---|---|---|
| `step5_multi_cpu_summary.csv` | `results/summary/` | 默认参数多实例 serial multi-chain 与 OpenMP multi-chain 加速实验。 |
| `step5_berlin52_summary.csv` | `results/summary/` | berlin52 上 serial、OpenMP、CUDA 定位实验汇总。 |
| `tuned_validation_summary.csv` | `results/summary/` | Step 6B 调优参数独立 seed 验证结果。 |
| `targeted_quality_summary.csv` | `results/summary/` | Step 6C 定向增强高预算质量实验结果。 |
| `policy_comparison_summary.csv` | `results/summary/` | QLSA epsilon-greedy 与 softmax 策略补充对比。 |
| `openmp_scaling_final_summary.csv` | `results/summary/` | OpenMP 线程扩展性补充实验。 |
| `tuning_summary.csv` | `results/summary/` | Step 6A 参数搜索汇总，仅作为调参过程证据。 |

## 论文参考数据

| 文件 | 位置 | 用途 |
|---|---|---|
| `paper_table8_runtime.csv` | `results/reference/` | 参考论文 Table 8 运行时间数据。 |
| `paper_hard_instance_quality.csv` | `results/reference/` | 参考论文 hard-instance 解质量表格整理数据。 |
| `qlearning_paper_text.txt` | `results/reference/` | 参考论文 PDF 文本提取，仅用于核对论文机制与表格来源。 |
| `pa_final_text.txt` | `results/reference/` | 课程作业说明文本提取，仅用于核对课程要求。 |

## 原始数据

| 文件 | 位置 | 用途 |
|---|---|---|
| `step5_multi_cpu_raw.csv` | `results/raw/` | 默认参数多实例 CPU/OpenMP 原始输出。 |
| `tuned_validation_raw.csv` | `results/raw/` | 调优参数独立验证原始输出。 |
| `targeted_quality_raw.csv` | `results/raw/` | 定向增强实验原始输出。 |
| `policy_comparison_raw.csv` | `results/raw/` | QLSA policy comparison 原始输出。 |
| `openmp_scaling_final_raw.csv` | `results/raw/` | OpenMP 扩展性实验原始输出。 |
| `step5_berlin52_raw.csv` | `results/raw/` | berlin52 CUDA/OpenMP/serial 定位实验原始输出。 |

## 补充与归档数据

- `results/traces/`：预算扫描与收敛相关补充数据。报告中明确说明这不是逐迭代 trace。
- `results/archive/`：历史索引和不作为主结论的归档结果。
- `results/logs/`：实验运行日志，不作为表格数据源。

## 使用原则

1. 最终报告的主结论优先引用 `results/final/` 和 `results/summary/`。
2. 与论文相关的对比只引用 `results/reference/`。
3. 原始实验可通过 `results/raw/` 和 `results/logs/` 追溯。
4. CUDA 结果仅作为工程定位，不作为主要加速结论。
