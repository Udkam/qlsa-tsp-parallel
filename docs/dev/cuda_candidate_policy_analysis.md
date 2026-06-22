# CUDA candidate-level evaluation 实验分析

本实验比较 CUDA chain mode 与新增 candidate mode。candidate mode 使用 one block per chain、block 内线程并行评价多个 2-opt 候选 move，并在 shared memory 中做最小 delta 归约。该模式改变了单步 proposal：它是 batch proposal 变体，不等同于原始 SA 的单候选采样。

| instance | family | mode | policy | chains | block | candidates | runs | best | gap | mean ms | speedup vs chain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| berlin52 | qlsa | chain |  | 32 | 128 | 64 | 1 | 7542 | 0.0000 | 200.1210 | 1.0000 |
| berlin52 | qlsa | candidate | best | 32 | 128 | 64 | 1 | 7542 | 0.0000 | 459.2310 | 0.4358 |
| berlin52 | qlsa | candidate | random | 32 | 128 | 64 | 1 | 7542 | 0.0000 | 269.1920 | 0.7434 |
| berlin52 | sa | chain |  | 32 | 128 | 64 | 1 | 7542 | 0.0000 | 112.7470 | 1.0000 |
| berlin52 | sa | candidate | best | 32 | 128 | 64 | 1 | 7542 | 0.0000 | 288.3050 | 0.3911 |
| berlin52 | sa | candidate | random | 32 | 128 | 64 | 1 | 7542 | 0.0000 | 184.2290 | 0.6120 |

## 结论边界

- SA CUDA candidate mode 已验证为可运行路径，并保留 chain mode 作为默认兼容模式。
- `best` policy 在每轮候选中选最小 delta，`random` policy 从候选批中按可复现随机方式选一个候选，用于比较批量择优与随机提案语义。
- 若 speedup_vs_cuda_chain 小于 1，说明 batch candidate evaluation 在该实例/预算下没有抵消 reduction、shared memory 同步和 thread 0 reversal 的开销。
- SA/QLSA candidate mode 均已接入主线 CUDA 后端，但仍应标记为 batch proposal 变体。
- 本实验不改变 OpenMP 作为主性能结论的定位。
