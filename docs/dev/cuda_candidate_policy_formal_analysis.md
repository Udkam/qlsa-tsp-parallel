# CUDA candidate-level evaluation 实验分析

本实验比较 CUDA chain mode 与新增 candidate mode。candidate mode 使用 one block per chain、block 内线程并行评价多个 2-opt 候选 move，并在 shared memory 中做最小 delta 归约。该模式改变了单步 proposal：它是 batch proposal 变体，不等同于原始 SA 的单候选采样。

| instance | family | mode | policy | chains | block | candidates | runs | best | gap | mean ms | speedup vs chain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a280 | qlsa | chain |  | 64 | 128 | 128 | 3 | 2781 | 7.8325 | 658.2997 | 1.0000 |
| a280 | qlsa | candidate | best | 64 | 128 | 128 | 6 | 2583 | 0.1551 | 1495.5983 | 0.4402 |
| a280 | qlsa | candidate | random | 64 | 128 | 128 | 6 | 2772 | 7.4835 | 929.1390 | 0.7085 |
| a280 | sa | chain |  | 64 | 128 | 128 | 3 | 2766 | 7.2509 | 366.9503 | 1.0000 |
| a280 | sa | candidate | best | 64 | 128 | 128 | 6 | 2579 | 0.0000 | 889.1128 | 0.4127 |
| a280 | sa | candidate | random | 64 | 128 | 128 | 6 | 2749 | 6.5917 | 608.9413 | 0.6026 |
| berlin52 | qlsa | chain |  | 64 | 128 | 128 | 3 | 7542 | 0.0000 | 568.7287 | 1.0000 |
| berlin52 | qlsa | candidate | best | 64 | 128 | 128 | 6 | 7542 | 0.0000 | 1326.8860 | 0.4286 |
| berlin52 | qlsa | candidate | random | 64 | 128 | 128 | 6 | 7542 | 0.0000 | 816.0577 | 0.6969 |
| berlin52 | sa | chain |  | 64 | 128 | 128 | 3 | 7542 | 0.0000 | 294.2277 | 1.0000 |
| berlin52 | sa | candidate | best | 64 | 128 | 128 | 6 | 7542 | 0.0000 | 867.6718 | 0.3391 |
| berlin52 | sa | candidate | random | 64 | 128 | 128 | 6 | 7542 | 0.0000 | 570.9175 | 0.5154 |
| lin318 | qlsa | chain |  | 64 | 128 | 128 | 3 | 44818 | 6.6359 | 616.5700 | 1.0000 |
| lin318 | qlsa | candidate | best | 64 | 128 | 128 | 6 | 42208 | 0.4259 | 1466.1027 | 0.4206 |
| lin318 | qlsa | candidate | random | 64 | 128 | 128 | 6 | 44900 | 6.8310 | 908.2948 | 0.6788 |
| lin318 | sa | chain |  | 64 | 128 | 128 | 3 | 44864 | 6.7453 | 326.6177 | 1.0000 |
| lin318 | sa | candidate | best | 64 | 128 | 128 | 6 | 42205 | 0.4188 | 894.5818 | 0.3651 |
| lin318 | sa | candidate | random | 64 | 128 | 128 | 6 | 45111 | 7.3330 | 598.8335 | 0.5454 |
| rat575 | qlsa | chain |  | 64 | 128 | 128 | 3 | 8103 | 19.6368 | 729.9123 | 1.0000 |
| rat575 | qlsa | candidate | best | 64 | 128 | 128 | 6 | 6951 | 2.6281 | 1652.8787 | 0.4416 |
| rat575 | qlsa | candidate | random | 64 | 128 | 128 | 6 | 8197 | 21.0247 | 1056.8762 | 0.6906 |
| rat575 | sa | chain |  | 64 | 128 | 128 | 3 | 8576 | 26.6204 | 421.0127 | 1.0000 |
| rat575 | sa | candidate | best | 64 | 128 | 128 | 6 | 6977 | 3.0120 | 933.4095 | 0.4510 |
| rat575 | sa | candidate | random | 64 | 128 | 128 | 6 | 8605 | 27.0486 | 644.6417 | 0.6531 |

## 结论边界

- SA CUDA candidate mode 已验证为可运行路径，并保留 chain mode 作为默认兼容模式。
- `best` policy 在每轮候选中选最小 delta，`random` policy 从候选批中按可复现随机方式选一个候选，用于比较批量择优与随机提案语义。
- 若 speedup_vs_cuda_chain 小于 1，说明 batch candidate evaluation 在该实例/预算下没有抵消 reduction、shared memory 同步和 thread 0 reversal 的开销。
- SA/QLSA candidate mode 均已接入主线 CUDA 后端，但仍应标记为 batch proposal 变体。
- 本实验不改变 OpenMP 作为主性能结论的定位。
