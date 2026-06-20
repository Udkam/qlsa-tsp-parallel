# QLSA Variant Design Review

本文件评估当前 QLSA 实现与论文 QLSA/SB-QLSA 机制的差异，并给出可选 `paper-lite` 变体设计。由于该变体会改动核心算法接口和实验口径，本轮最终冲刺不直接实现，以避免破坏已有稳定结果。

## 1. 当前实现状态

| 功能 | 当前状态 | 说明 |
|---|---|---|
| action set | 已实现 | 动作为不同 2-opt span 策略 |
| state discretization | 已实现 | 根据近期 delta 平均值离散为 5 个状态 |
| epsilon-greedy | 已实现 | `--policy epsilon-greedy` |
| softmax | 已实现 | `--policy softmax` |
| Q table | 已实现 | state-action Q table |
| reward | 已实现 | 改进 move 给正奖励，拒绝/变差给负奖励 |
| double-bridge kick | 未确认实现 | 未发现论文式 double-bridge candidate |
| candidate leader | 未完整实现 | 当前选择邻域动作，不选择 candidate leader |
| diversity state | 部分思想 | 当前状态不是 Hamming-distance diversity state |

## 2. 当前实现的合理定位

当前 QLSA 更适合描述为“基于 Q-learning 的邻域策略选择变体”。它保留了 SA 主循环和 2-opt Metropolis，并使用 Q table 在不同状态下选择不同 span 的 2-opt 动作。该设计有工程实现简单、速度较快、易于多链并行的优点，但与论文 stateless candidate-leader QLSA 和 SB-QLSA 仍存在机制差异。

因此最终报告应采用以下措辞：

- 可写：“本项目实现了 Q-learning-assisted SA 的工程变体。”
- 可写：“当前 QLSA 采用状态/动作离散化，与论文 candidate-leader 机制不完全相同。”
- 不可写：“本项目完整复刻了论文 SB-QLSA。”

## 3. paper-lite 变体建议

若后续继续实现 `paper-lite`，建议新增 CLI 参数：

```text
--qlsa-variant current
--qlsa-variant paper-lite
```

其中：

- `current`：保持当前实现和已有实验兼容；
- `paper-lite`：尽量贴近论文 candidate-leader 思想，但仍不声称完整 SB-QLSA。

## 4. paper-lite 算法草案

每次迭代构造候选集合：

1. `c1`: current tour；
2. `c2`: current best tour；
3. `c3`: random perturbation tour；
4. `c4`: double-bridge perturbation tour。

Q-learning 对四个 candidate/action 选择一个 leader，然后对该 leader 应用 2-opt Metropolis move。若 move 被接受，则更新 current tour；若产生更优路径，则更新 best tour。

Q 更新仍使用：

\[
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\]

reward 可定义为：

\[
r = L_{\text{before}} - L_{\text{after}}
\]

即路径长度下降为正奖励，变差为负奖励。

## 5. double-bridge perturbation 设计

double-bridge 可在 tour 上选取四个非相邻切点，将四段重新连接，用于产生较大扰动。实现时需要保证：

- 不破坏 tour 合法排列；
- 切点不相邻；
- 小规模实例 fallback 到随机扰动；
- seed 完全由链内 RNG 控制；
- 该操作只用于候选生成，不直接覆盖 best tour。

## 6. diversity state 设计

若要更接近论文 SB-QLSA，可用当前 tour 与 best tour 的 Hamming distance 定义 diversity：

\[
D_t=\frac{1}{n}\sum_{i=0}^{n-1} I(\pi_t(i)\ne \pi^*(i))
\]

然后根据阈值 \(\delta\) 将状态分为低多样性和高多样性：

\[
s_t = I[D_t \ge \delta]
\]

这会得到一个 \(2\times 4\) 的 state-action Q table。

## 7. 为什么本轮不直接实现

本轮目标是最终提交质量，而不是重新引入大范围算法风险。paper-lite 需要修改：

- `QLSAParams`；
- `run_qlsa_2opt` 主循环；
- CLI；
- CSV algorithm 命名；
- CUDA kernel 或至少 CUDA fallback 口径；
- 新测试；
- 新实验。

若在最终阶段匆忙实现，容易造成已有 Step 5/6 实验口径失效。因此本轮选择保留当前稳定实现，并在报告中诚实说明差异，将 paper-lite 作为后续工作。

## 8. 建议测试

未来实现后至少新增：

- `tests/test_qlsa_variants.cpp`；
- double-bridge 保持 tour 合法；
- `current` 与旧 QLSA 输出兼容；
- `paper-lite` 在 `square4.tsp` 上可运行；
- epsilon-greedy 与 softmax 均 smoke test；
- same seed reproducibility；
- OpenMP multi-chain 下每条 chain 独立 Q table。
