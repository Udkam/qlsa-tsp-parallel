# Step 4 Design: CUDA Multi-Chain Parallelism

## 1. CUDA 并行粒度

Step 4 沿用 Step 3 的 multi-chain 抽象：一次运行包含 `chains` 条独立搜索链。CUDA 后端将每条 chain 映射到一个 GPU thread block：

- grid size = `chains`
- block size = `cuda_block_size`
- 每个 block 维护一条 SA 或 QLSA 搜索链

当前 CUDA 基线重点是先固定 GPU 后端接口、数据拷贝、seed 派生和结果归约。后续可以进一步把 block 内多个线程用于批量候选 move 评价。

## 2. Block 内数据和线程职责

每个 block 使用 shared memory 存放：

- current tour；
- best tour；
- nearest-neighbor 初始化时的 used 标记；
- QLSA 的 Q table。

当前实现由 block 内 `threadIdx.x == 0` 执行完整 chain 的初始化、move 采样、Metropolis 接受判断、2-opt reverse、best 更新和 Q 表更新。这样可以先保证与现有串行/多链协议一致，并让多个 block 并行运行多条独立链。block 内并行候选评价会作为后续性能优化项。

## 3. DistanceMatrix

`DistanceMatrix::raw()` 已经是一维连续 `n*n` 数组。CUDA host wrapper 将该数组一次性复制到 GPU global memory，kernel 内只读访问：

```text
dist(i, j) = dm[i * n + j]
```

这和 Step 1 的设计保持一致，也便于后续 CUDA 版本优化访存。

## 4. QLSA Q Table

QLSA 每条 chain 独立维护 Q table。当前支持默认动作集合，也支持最多 8 个动作的 host 参数传入。Q table 在 block shared memory 中初始化为 0，并在每轮 move 后按：

```text
Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

进行更新。

## 5. Seed 派生规则

CUDA 与 OpenMP 使用同一 seed 派生规则：

```cpp
chain_seed = splitmix64(base_seed + 0x9E3779B97F4A7C15ULL * (chain_id + 1));
```

GPU 端每条 chain 使用独立的轻量 64-bit RNG 状态。相同 `base_seed` 和 `chains` 下，chain seed 序列稳定可复现。

## 6. 全局最优归约

kernel 结束后，每个 block 输出：

- best tour；
- best length；
- final length；
- accepted moves；
- improved moves；
- chain seed。

host 将所有 chain 结果复制回 CPU，串行选择 `best_length` 最小的 chain，并汇总 accepted/improved move 统计。最终 best tour 会用 CPU 完整 tour length 再校验一次。

## 7. CSV 字段

CUDA 复用 Step 3 的 CSV 字段：

```text
algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves
```

CUDA 模式下：

- `algorithm` 为 `sa-cuda` 或 `qlsa-cuda`；
- `parallel` 为 `cuda`；
- `threads` 字段记录 `cuda_block_size`；
- `elapsed_ms` 是整个 CUDA kernel 与数据回传的 wall-clock time。

## 8. 构建回退

CUDA 是可选能力。若当前 CMake 生成器无法启用 CUDA language，或运行时没有 CUDA device，项目仍可编译并回退到串行 multi-chain 执行，同时输出 warning。真实 CUDA 性能实验必须确认配置阶段显示 CUDA enabled，且运行时 `cuda_available=true`。

当前 Windows Visual Studio 生成器如果没有安装 CUDA VS toolset，会自动跳过 CUDA language。可安装 CUDA VS integration，或改用支持 CUDA language 的生成器后再做真实性能测试。

## 9. 后续性能对比方法

后续报告中应至少区分：

1. 串行 SA/QLSA；
2. OpenMP multi-chain SA/QLSA；
3. CUDA multi-chain SA/QLSA；
4. 相同 `iterations`、`chains`、`seed` 设置下的运行时间；
5. best/mean/std、Gap、speedup 和并行效率。

不要把 fallback 结果作为 CUDA 性能结果。
