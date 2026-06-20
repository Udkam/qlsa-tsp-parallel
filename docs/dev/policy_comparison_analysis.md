# Policy Comparison Analysis

本实验比较当前 QLSA 实现中的 `epsilon-greedy` 与 `softmax` 策略。需要注意，本项目 softmax 是当前工程实现中的 action-selection policy，不完全等同论文 candidate-leader softmax 机制。

## Experiment Setup

- instances: eil76, rat99, eil101
- algorithm: QLSA
- parallel: OpenMP
- chains: 32
- threads: 8
- iterations: 1,000,000
- repeat: 5
- alpha=0.1, gamma=0.9, epsilon=0.1

## Summary

| Instance | Policy | Runs | Best | Min Gap % | Mean Gap % | Mean ms |
|---|---|---:|---:|---:|---:|---:|
| eil101 | epsilon-greedy | 5 | 631 | 0.318 | 1.335 | 368.517 |
| eil101 | softmax | 5 | 631 | 0.318 | 1.176 | 675.579 |
| eil76 | epsilon-greedy | 5 | 540 | 0.372 | 0.892 | 398.948 |
| eil76 | softmax | 5 | 539 | 0.186 | 0.855 | 653.067 |
| rat99 | epsilon-greedy | 5 | 1213 | 0.165 | 0.512 | 456.452 |
| rat99 | softmax | 5 | 1248 | 3.055 | 3.815 | 666.716 |

## Interpretation

该实验用于补充策略对比，而不是替代 Step 6B/6C 的 tuned validation 和 targeted high-budget 质量结论。最终报告可引用该图作为 QLSA 策略敏感性的辅助材料，但不应据此声称某一策略在所有实例上稳定占优。
