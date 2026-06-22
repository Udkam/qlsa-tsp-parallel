# QLSA 变体对齐实验分析

本实验比较 `current`、`paper` 与 `paper-sb` 三种 QLSA 入口。`current` 是已有工程化状态/动作版本，`paper` 使用 candidate-leader 来源选择，`paper-sb` 在 candidate-leader 上加入路径多样性状态。

实验目的不是替换已有 Step 5/6 结果，而是确认论文机制对齐变体已经进入同一 C++/OpenMP 实验流程，并观察其在代表实例上的质量和时间代价。

## 每个实例的最佳配置

| 实例 | 变体 | 策略 | 阈值 | 最短路径 | 最小偏差 | 平均时间(ms) |
|---|---|---|---:|---:|---:|---:|
| berlin52 | current | epsilon-greedy | - | 7542 | 0.0000% | 150.75 |
| eil101 | paper-sb | softmax | 0.50 | 632 | 0.4769% | 1130.82 |
| eil76 | paper-sb | epsilon-greedy | 0.70 | 538 | 0.0000% | 599.72 |
| rat99 | current | epsilon-greedy | - | 1231 | 1.6515% | 165.70 |

## 解释

- `paper` / `paper-sb` 的动作对象是候选来源，而不是 2-opt 跨度范围，因此它们与 `current` 的搜索行为不同。
- `paper-sb` 的 diversity state 让 Q 表能区分当前路径与历史最优路径是否相近，适合观察论文状态机制的工程化效果。
- 若某个实例上 `current` 仍然更好，说明已有工程化动作设计在该预算下更合适；若 `paper` 或 `paper-sb` 更好，则说明 candidate-leader 机制值得进一步调参。
