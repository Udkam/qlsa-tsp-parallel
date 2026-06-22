# 实验结果索引

本索引只列出当前项目保留的关键 CSV 与最终图表。历史 raw、日志、quick/smoke 中间文件和旧版本报告已清理。

## 1. OpenMP 主性能

| 文件 | 用途 |
|---|---|
| `results/summary/step5_multi_cpu_summary.csv` | 6 个 TSPLIB95 实例默认参数 SA/QLSA OpenMP speedup 与 parallel efficiency。 |
| `results/summary/openmp_scaling_final_summary.csv` | 默认实例 OpenMP 线程扩展性。 |
| `results/summary/openmp_scaling_large_summary.csv` | a280/rat575 大实例 OpenMP 线程扩展性。 |
| `figures/fig_course_01_openmp_speedup.png` | OpenMP 多链并行加速比图。 |
| `figures/fig_course_02_openmp_efficiency.png` | OpenMP 多链并行效率图。 |
| `figures/fig_course_10_openmp_thread_scaling.png` | OpenMP 线程扩展曲线图。 |

## 2. 解质量与调优

| 文件 | 用途 |
|---|---|
| `results/summary/tuned_validation_summary.csv` | 调优参数独立 seed 验证。 |
| `results/summary/targeted_quality_summary.csv` | 定向增强 high-budget 质量结果。 |
| `results/summary/policy_comparison_summary.csv` | QLSA epsilon-greedy / softmax 对比。 |
| `results/summary/qlsa_variant_alignment_summary.csv` | `current` / `paper` / `paper-sb` 三种 QLSA 入口的代表实例对齐实验。 |
| `figures/fig_course_03_default_gap.png` | 默认参数 Gap 图。 |
| `figures/fig_course_04_targeted_quality.png` | 定向增强质量图。 |
| `figures/fig_course_05_policy_comparison.png` | QLSA 策略对比图。 |
| `figures/fig_qlsa_variant_alignment.png` | QLSA 机制对齐变体最小偏差图。 |

## 3. CUDA chain / candidate

| 文件 | 用途 |
|---|---|
| `results/summary/cuda_qlsa_candidate_summary.csv` | SA/QLSA CUDA chain/candidate formal subset。 |
| `results/summary/cuda_reversal_summary.csv` | CUDA candidate serial/parallel reversal 对比。 |
| `results/summary/cuda_candidate_sweep_aggressive_summary.csv` | CUDA candidate aggressive 参数扫描。 |
| `results/summary/large_cuda_formal_summary.csv` | 大实例 CUDA chain/candidate formal subset。 |
| `results/logs/nsight/cuda_candidate_a280_hybrid_nsys_ascii.log` | a280 CUDA candidate-hybrid 的 Nsight Systems 文本摘要。 |
| `figures/fig_course_06_cuda_boundary.png` | CUDA 多链模式与候选批量评价对比图。 |
| `figures/fig_cuda_candidate_policy_formal.png` | CUDA SA 候选策略运行时间对比图。 |

## 4. MPI + OpenMP

| 文件 | 用途 |
|---|---|
| `results/summary/mpi_vm_smoke_summary.csv` | 双 VM MPI smoke 证据。 |
| `results/summary/mpi_vm_scaling_formal_summary.csv` | berlin52 双 VM formal scaling。 |
| `results/summary/large_mpi_vm_formal_aggressive_summary.csv` | ch130/a280 大实例双 VM formal subset。 |
| `figures/fig_course_07_mpi_scaling.png` | 双虚拟机 MPI + OpenMP 扩展性图。 |

## 5. 大实例压力测试

| 文件 | 用途 |
|---|---|
| `results/final/large_instance_download_status.csv` | L1/L2/L3 数据下载来源、状态、SHA256。 |
| `results/final/large_instance_inventory.csv` | 大实例存在性、dimension、EDGE_WEIGHT_TYPE、BKS。 |
| `results/summary/large_openmp_l1_summary.csv` | L1 formal OpenMP 结果。 |
| `results/summary/large_openmp_l2_formal_summary.csv` | L2 formal subset OpenMP 结果。 |
| `results/summary/large_openmp_l3_quick_summary.csv` | L3 quick OpenMP 结果。 |
| `figures/fig_course_11_representative_openmp.png` | 代表实例 OpenMP 压力测试图。 |

## 6. 论文与 Python 参考

| 文件 | 用途 |
|---|---|
| `results/reference/paper_table8_runtime.csv` | 参考论文 Table 8 时间数据。 |
| `results/reference/paper_hard_instance_quality.csv` | 参考论文 hard-instance quality 数据。 |
| `results/final/report_comparison_summary.csv` | 论文参考数据与本项目结果对比摘要。 |
| `results/summary/python_reference_summary.csv` | Python faithful baseline 与 C++ 对照。 |
| `figures/fig_course_09_paper_quality.png` | 论文 hard-instance quality 对比图。 |

## 7. 结论边界

- OpenMP multi-chain 是当前主要性能结论。
- CUDA candidate-level evaluation 是显式启用的 batch proposal 变体，不替代默认 `cuda_mode=chain`。
- MPI 结果来自真实双 VM `mpirun`，但环境是 VMware NAT，不等同生产 HPC benchmark。
- QLSA 只在部分实例/预算下优于 SA，不能写成总是优于 SA。
- C++ 主线没有声称完整复刻论文 SB-QLSA。
