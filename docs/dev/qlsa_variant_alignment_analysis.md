# QLSA 变体对齐实验分析

本实验比较 `current`、`paper` 与 `paper-sb` 三种 QLSA 入口。`current` 是已有工程化状态/动作版本，`paper` 使用 candidate-leader 来源选择，`paper-sb` 在 candidate-leader 上加入路径多样性状态。

历史 raw CSV 没有 diversity metric 列时，`paper-sb` 行按当时实现标为 `hamming`；`edge` 样本以独立条件汇总。

三种变体使用统一的 C++/OpenMP 执行合同和共同种子，表中结果展示参数网格内达到的质量和时间代价。

## 各变体与度量条件的最佳配置

| 实例 | 变体 | 多样性度量 | 策略 | 阈值 | 最短路径 | 最小偏差 | 平均时间(ms) |
|---|---|---|---|---:|---:|---:|---:|
| berlin52 | current | - | epsilon-greedy | - | 7542 | 0.0000% | 150.75 |
| berlin52 | paper | - | epsilon-greedy | - | 7542 | 0.0000% | 596.25 |
| berlin52 | paper-sb | hamming | softmax | 0.70 | 7658 | 1.5381% | 640.11 |
| eil101 | current | - | softmax | - | 647 | 2.8617% | 281.92 |
| eil101 | paper | - | softmax | - | 637 | 1.2719% | 809.63 |
| eil101 | paper-sb | hamming | softmax | 0.50 | 632 | 0.4769% | 1130.82 |
| eil76 | current | - | epsilon-greedy | - | 548 | 1.8587% | 150.63 |
| eil76 | paper | - | softmax | - | 538 | 0.0000% | 710.15 |
| eil76 | paper-sb | hamming | epsilon-greedy | 0.70 | 538 | 0.0000% | 599.72 |
| rat99 | current | - | epsilon-greedy | - | 1231 | 1.6515% | 165.70 |
| rat99 | paper | - | softmax | - | 1236 | 2.0644% | 777.59 |
| rat99 | paper-sb | hamming | epsilon-greedy | 0.30 | 1237 | 2.1470% | 715.46 |

## 解释

- `paper` / `paper-sb` 的动作对象是候选来源，而不是 2-opt 跨度范围，因此它们与 `current` 的搜索行为不同。
- `paper-sb` 的 diversity state 让 Q 表能区分当前路径与历史最优路径是否相近；Hamming 与 edge 是不同实验条件，分别报告。
- 若某个实例上 `current` 仍然更好，说明已有工程化动作设计在该预算下更合适；若 `paper` 或 `paper-sb` 更好，则说明 candidate-leader 机制值得进一步调参。
