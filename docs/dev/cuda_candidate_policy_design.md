# CUDA 候选策略与路径操作设计记录

## 背景

CUDA 多链模式已经能够在 GPU 上同时运行多条搜索链，但链内一次迭代仍只处理一个 2-opt 候选，GPU 端计算密度有限。候选批量评价模式采用“一条搜索链对应一个线程块”的映射方式：块内多个线程同时生成 2-opt 候选，分别计算增量，再在共享内存中完成候选选择。

本轮在原有 `best` 和 `random` 策略基础上补充 `hybrid` 策略，并保留并行路径反转和最优路径协作复制。目标是把 CUDA 后端从“只验证可运行”推进到“可比较候选选择、路径操作和同步成本”的实验状态。

## 策略定义

- `best`：从候选批中选择增量最小的移动，偏向解质量。
- `random`：从候选批中按可复现随机方式选择一个移动，更接近随机提案。
- `hybrid`：在 `best` 与 `random` 之间交替，用于观察质量收益和提案随机性的折中。

默认值仍为 `best`。历史命令如果没有显式传入 `--cuda_candidate_policy`，行为不变。

## 实现位置

- `include/tsp/parallel.hpp`：`CudaCandidatePolicy` 增加 `Hybrid`。
- `src/main.cpp`：命令行参数支持 `--cuda_candidate_policy best|random|hybrid`。
- `src/cuda_kernels.cu`：SA/QLSA candidate kernel 中增加 hybrid 分支。
- `tests/test_cuda_candidate.cpp`：覆盖 SA candidate 的 hybrid 策略。
- `tests/test_cuda_qlsa_candidate.cpp`：覆盖 QLSA candidate 的 hybrid 策略。
- `scripts/run_cuda_candidate_experiments.py`：实验矩阵支持 `hybrid`。
- `scripts/run_cuda_candidate_sweep.py`：参数扫描支持 `hybrid`。
- `scripts/analyze_cuda_candidate.py`：汇总和图表支持 `hybrid`。
- `scripts/run_cuda_nsight_profile.py`：补充 Nsight Compute / Nsight Systems 捕获入口。

## 共享内存与路径操作

CUDA candidate kernel 当前使用共享内存保存候选增量、候选下标和路径副本。候选选择阶段不访问全局内存中的候选数组；路径更新时支持串行路径反转和块内并行路径反转两种模式。最优路径更新也使用块内线程协作复制，减少单线程长循环。

该实现没有把完整路径长期驻留在共享内存中。对于 a280、rat575 这类中等实例，完整路径仍需要全局内存存储。后续如果继续优化，可以考虑只缓存当前反转片段或引入分段路径布局，但这会增加实现复杂度和合法性检查成本。

## 语义边界

候选批量评价会改变普通 SA 的单候选提案过程：

- `best` 更像批量择优；
- `random` 更接近随机提案，但候选仍是批量生成；
- `hybrid` 是两者之间的工程折中。

因此报告中应把该功能写成 CUDA 端搜索变体，而不是普通 SA 的完全等价加速。实验结论必须同时报告运行时间和路径质量，不能只选择有利指标。

## quick 验证

已运行：

```powershell
py scripts\run_cuda_candidate_experiments.py --instances berlin52 a280 --algorithms sa qlsa --iterations 100000 --repeat 1 --chains 32 --block-sizes 128 --candidates-per-iter 128 --reversal-modes parallel --candidate-policies best random hybrid --output results\raw\cuda_candidate_hybrid_quick_raw.csv

py scripts\analyze_cuda_candidate.py --input results\raw\cuda_candidate_hybrid_quick_raw.csv --output results\summary\cuda_candidate_hybrid_quick_summary.csv --markdown docs\dev\cuda_candidate_hybrid_quick_analysis.md --figure figures\final\fig_cuda_candidate_hybrid_quick.png
```

在 berlin52 上，多链模式和三种 candidate 策略均达到 BKS。a280 上，candidate-best 和 candidate-hybrid 明显降低偏差，但运行时间高于 chain 模式。该结果支持“候选批量评价改善搜索质量但增加 GPU 端同步与路径操作成本”的解释。

## Nsight 状态

已运行：

```powershell
py scripts\run_cuda_nsight_profile.py --instance a280 --iterations 20000 --policy hybrid
```

当前 Windows PATH 中未找到 Nsight Systems；Nsight Compute 可用，并生成 `results/logs/nsight/cuda_candidate_a280_hybrid_ncu.ncu-rep`。由于本轮只建立 profiling 入口和报告文件，不从该文件中推导 occupancy、带宽或 CUDA 性能优势。
