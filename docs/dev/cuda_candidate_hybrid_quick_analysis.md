# CUDA candidate-level evaluation 实验分析

本实验比较 CUDA chain mode 与新增 candidate mode。candidate mode 使用 one block per chain、block 内线程并行评价多个 2-opt 候选 move，并在 shared memory 中做最小 delta 归约。该模式改变了单步 proposal：它是 batch proposal 变体，不等同于原始 SA 的单候选采样。

| instance | family | mode | policy | chains | block | candidates | runs | best | gap | mean ms | speedup vs chain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a280 | qlsa | chain |  | 32 | 128 | 128 | 1 | 3059 | 18.6119 | 243.3390 | 1.0000 |
| a280 | qlsa | candidate | best | 32 | 128 | 128 | 1 | 2587 | 0.3102 | 440.3740 | 0.5526 |
| a280 | qlsa | candidate | random | 32 | 128 | 128 | 1 | 3036 | 17.7200 | 320.0600 | 0.7603 |
| a280 | qlsa | candidate | hybrid | 32 | 128 | 128 | 1 | 2600 | 0.8143 | 376.3100 | 0.6466 |
| a280 | sa | chain |  | 32 | 128 | 128 | 1 | 3157 | 22.4118 | 138.7100 | 1.0000 |
| a280 | sa | candidate | best | 32 | 128 | 128 | 1 | 2591 | 0.4653 | 308.3890 | 0.4498 |
| a280 | sa | candidate | random | 32 | 128 | 128 | 1 | 3157 | 22.4118 | 209.7320 | 0.6614 |
| a280 | sa | candidate | hybrid | 32 | 128 | 128 | 1 | 2606 | 1.0469 | 256.7560 | 0.5402 |
| berlin52 | qlsa | chain |  | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 206.1080 | 1.0000 |
| berlin52 | qlsa | candidate | best | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 443.8330 | 0.4644 |
| berlin52 | qlsa | candidate | random | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 286.8940 | 0.7184 |
| berlin52 | qlsa | candidate | hybrid | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 345.6320 | 0.5963 |
| berlin52 | sa | chain |  | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 118.8890 | 1.0000 |
| berlin52 | sa | candidate | best | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 305.2960 | 0.3894 |
| berlin52 | sa | candidate | random | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 200.2060 | 0.5938 |
| berlin52 | sa | candidate | hybrid | 32 | 128 | 128 | 1 | 7542 | 0.0000 | 250.7870 | 0.4741 |

## 结论边界

- SA CUDA candidate mode 已验证为可运行路径，并保留 chain mode 作为默认兼容模式。
- `best` policy 在每轮候选中选最小 delta，`random` policy 从候选批中按可复现随机方式选一个候选，`hybrid` policy 在 best/random 之间交替，用于比较批量择优、随机提案和组合策略。
- 若 speedup_vs_cuda_chain 小于 1，说明 batch candidate evaluation 在该实例/预算下没有抵消 reduction、shared memory 同步和 thread 0 reversal 的开销。
- SA/QLSA candidate mode 均已接入主线 CUDA 后端，但仍应标记为 batch proposal 变体。
- 本实验不改变 OpenMP 作为主性能结论的定位。
