# 实验结果索引

本索引只列出当前项目保留的关键 CSV 与最终图表。历史 raw、日志、quick/smoke 中间文件和旧版本报告已清理。

## 1. OpenMP 主性能

| 文件 | 用途 |
|---|---|
| `results/summary/step5_multi_cpu_summary.csv` | 6 个 TSPLIB95 实例默认参数 SA/QLSA OpenMP speedup 与 parallel efficiency。 |
| `results/summary/openmp_scaling_final_summary.csv` | 默认实例 OpenMP 线程扩展性。 |
| `results/summary/openmp_scaling_large_summary.csv` | a280/rat575 大实例 OpenMP 线程扩展性。 |
| `figures/final/fig02_openmp_speedup.png` | OpenMP speedup 图。 |
| `figures/final/fig03_openmp_efficiency.png` | OpenMP parallel efficiency 图。 |
| `figures/final/fig19_openmp_large_scaling.png` | 大实例 OpenMP scaling 图。 |

## 2. 解质量与调优

| 文件 | 用途 |
|---|---|
| `results/summary/tuned_validation_summary.csv` | 调优参数独立 seed 验证。 |
| `results/summary/targeted_quality_summary.csv` | 定向增强 high-budget 质量结果。 |
| `results/summary/policy_comparison_summary.csv` | QLSA epsilon-greedy / softmax 对比。 |
| `results/summary/qlsa_variant_alignment_summary.csv` | `current` / `paper` / `paper-sb` 三种 QLSA 入口的代表实例对齐实验。 |
| `figures/final/fig04_default_gap.png` | 默认参数 Gap 图。 |
| `figures/final/fig05_tuning_curve.png` | 调优与增强 Gap 改善图。 |
| `figures/final/fig06_policy_comparison.png` | policy comparison 图。 |
| `figures/final/fig_qlsa_variant_alignment.png` | QLSA 机制对齐变体最小偏差图。 |

## 3. CUDA chain / candidate

| 文件 | 用途 |
|---|---|
| `results/summary/cuda_qlsa_candidate_summary.csv` | SA/QLSA CUDA chain/candidate formal subset。 |
| `results/summary/cuda_reversal_summary.csv` | CUDA candidate serial/parallel reversal 对比。 |
| `results/summary/cuda_candidate_sweep_aggressive_summary.csv` | CUDA candidate aggressive 参数扫描。 |
| `results/summary/large_cuda_formal_summary.csv` | 大实例 CUDA chain/candidate formal subset。 |
| `results/logs/nsight/cuda_candidate_a280_hybrid_nsys_ascii.log` | a280 CUDA candidate-hybrid 的 Nsight Systems 文本摘要。 |
| `figures/final/fig15_cuda_candidate_mode.png` | CUDA chain/candidate 定位图。 |
| `figures/final/fig17_large_cuda_chain_vs_candidate.png` | 大实例 CUDA chain/candidate 对比。 |
| `figures/final/fig21_cuda_qlsa_candidate.png` | CUDA SA/QLSA candidate 对比。 |
| `figures/final/fig22_cuda_parallel_reversal.png` | CUDA parallel reversal 图。 |
| `figures/final/fig23_cuda_candidate_sweep_tradeoff.png` | CUDA candidate sweep 时间-Gap 折中图。 |

## 4. MPI + OpenMP

| 文件 | 用途 |
|---|---|
| `results/summary/mpi_vm_smoke_summary.csv` | 双 VM MPI smoke 证据。 |
| `results/summary/mpi_vm_scaling_formal_summary.csv` | berlin52 双 VM formal scaling。 |
| `results/summary/large_mpi_vm_formal_aggressive_summary.csv` | ch130/a280 大实例双 VM formal subset。 |
| `figures/final/fig13_mpi_vm_scaling_formal.png` | MPI formal speedup 图。 |
| `figures/final/fig14_hpc_hybrid_architecture.png` | MPI + OpenMP + CUDA 架构图。 |
| `figures/final/fig18_large_mpi_vm_scaling.png` | 大实例 MPI speedup 图。 |

## 5. 大实例压力测试

| 文件 | 用途 |
|---|---|
| `results/final/large_instance_download_status.csv` | L1/L2/L3 数据下载来源、状态、SHA256。 |
| `results/final/large_instance_inventory.csv` | 大实例存在性、dimension、EDGE_WEIGHT_TYPE、BKS。 |
| `results/summary/large_openmp_l1_summary.csv` | L1 formal OpenMP 结果。 |
| `results/summary/large_openmp_l2_formal_summary.csv` | L2 formal subset OpenMP 结果。 |
| `results/summary/large_openmp_l3_quick_summary.csv` | L3 quick OpenMP 结果。 |
| `figures/final/fig16_large_openmp_gap_time.png` | 大实例 OpenMP Gap 与时间图。 |

## 6. 论文与 Python 参考

| 文件 | 用途 |
|---|---|
| `results/reference/paper_table8_runtime.csv` | 参考论文 Table 8 时间数据。 |
| `results/reference/paper_hard_instance_quality.csv` | 参考论文 hard-instance quality 数据。 |
| `results/final/report_comparison_summary.csv` | 论文参考数据与本项目结果对比摘要。 |
| `results/summary/python_reference_summary.csv` | Python faithful baseline 与 C++ 对照。 |
| `figures/final/fig08_paper_runtime_comparison.png` | 论文运行时间参考对比图。 |
| `figures/final/fig09_paper_quality_comparison.png` | 论文 hard-instance quality 对比图。 |
| `figures/final/fig11_python_cpp_reference.png` | Python faithful baseline 对比图。 |

## 7. 结论边界

- OpenMP multi-chain 是当前主要性能结论。
- CUDA candidate-level evaluation 是显式启用的 batch proposal 变体，不替代默认 `cuda_mode=chain`。
- MPI 结果来自真实双 VM `mpirun`，但环境是 VMware NAT，不等同生产 HPC benchmark。
- QLSA 只在部分实例/预算下优于 SA，不能写成总是优于 SA。
- C++ 主线没有声称完整复刻论文 SB-QLSA。
