# Step 2 Design: Serial QLSA

## 1. 设计目标

Step 2 在 Step 1 的串行 SA/2-opt 基线上加入 Q-Learning 动作选择层，形成串行 Q-Learning-Assisted Simulated Annealing (QLSA) 实现。

当前阶段仍保持单线程执行，不做 OpenMP 或 CUDA 优化。目标是先固定状态、动作、奖励和 Q 表更新接口，为后续多搜索链并行提供稳定基础。

## 2. 与原串行 SA 的关系

QLSA 复用原 SA 的核心结构：

- 初始化 tour；
- 维护当前路径长度和 best tour；
- 使用 2-opt 邻域；
- 使用 O(1) delta；
- 使用 Metropolis 接受准则；
- 指数降温；
- 结束时完整重算路径长度并校验。

不同点在于：普通 SA 每轮随机选择一个 2-opt move；QLSA 每轮先根据 Q 表选择一个“邻域动作”，再由该动作生成 2-opt move。

## 3. 状态设计

当前实现使用最近 `state_window` 次候选 move 的 delta 平均值离散化为 5 个状态：

| 状态 | 含义 |
|---:|---|
| 0 | 强改善趋势 |
| 1 | 弱改善趋势 |
| 2 | 近似不变 |
| 3 | 弱变差趋势 |
| 4 | 强变差趋势 |

离散阈值由 `delta_scale` 控制。该状态设计只依赖最近搜索行为，不依赖实例规模以外的额外信息，便于后续每条并行搜索链独立维护。

## 4. 动作设计

动作集合由 `QLSAAction` 表示：

```cpp
struct QLSAAction {
    std::string name;
    double min_span_ratio;
    double max_span_ratio;
};
```

默认动作对应三类 2-opt 反转区间长度：

- `short-2opt`：短区间扰动；
- `medium-2opt`：中等区间扰动；
- `long-2opt`：长区间扰动。

动作只决定候选 2-opt 区间的长度范围，路径长度计算、接受准则和 best 更新仍沿用 SA 逻辑。

## 5. 奖励设计

奖励使用路径长度变化量 delta：

```text
reward = -delta, if move accepted
reward = -0.1 * delta, if worsening move rejected
```

因此路径长度减少时 `delta < 0`，奖励为正；接受变差 move 时奖励为负；拒绝变差 move 时给较小负反馈。这个设计保持“更短路径更好”的方向一致。

## 6. Q 表更新

每轮 move 接受或拒绝后执行：

```text
Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

其中：

- `s` 是执行动作前的离散状态；
- `a` 是本轮选择的邻域动作；
- `s'` 是加入本轮 delta 后的新离散状态；
- `alpha` 是学习率；
- `gamma` 是折扣因子。

## 7. 动作选择策略

当前支持两种策略：

- `epsilon-greedy`：以 `epsilon` 概率随机探索，否则选择当前状态下 Q 值最大的动作；
- `softmax`：按 Q 值的 softmax 分布随机选择动作。

所有随机选择都使用 `seed` 控制，`repeat` 模式下仍使用 `seed, seed+1, ...`。

## 8. 后续并行扩展

OpenMP 多链并行：

- 每个线程维护独立 tour、Q 表、RNG 和局部 best；
- 结束后归约 best tour 和统计指标。

CUDA 多链并行：

- 将距离矩阵一维数组复制到 GPU；
- 每条链维护较小的 Q 表和随机状态；
- action selection、2-opt delta、接受判断可在 GPU 上并行执行。

Q-Learning 策略扩展：

- 当前动作为不同尺度 2-opt；
- 后续可扩展为不同邻域类型、候选数量、温度扰动或混合策略。
