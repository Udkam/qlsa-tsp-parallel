# Step 3 Design: OpenMP Multi-Chain Parallelism

## 1. 为什么选择多搜索链并行

SA 和 QLSA 都是随机启发式搜索。一次实验通常需要多次独立运行，才能获得更稳定的 best、mean、std 等统计指标。因此 Step 3 采用 parallel multi-start / multi-chain 粒度：一次运行启动多条独立搜索链，每条链独立维护当前 tour、随机数状态和局部最优解。

这种粒度同步开销低，不需要在每一步 move 后通信，适合 OpenMP，也能自然迁移到后续 CUDA 多链版本。

## 2. OpenMP 并行粒度

`run_parallel_chains` 将 `chains` 条搜索链映射到 OpenMP `parallel for`：

```cpp
#pragma omp parallel for schedule(static) num_threads(params.threads)
```

每个 `chain_id` 只写入自己的 `chain_results[chain_id]`。并行区域内不更新全局 best，不输出日志，不使用临界区。

如果 OpenMP 不可用，或者 `threads <= 1`，同一接口会使用普通 `for` 循环执行多链。这样构建不会依赖 OpenMP，同时 `chains` 的语义保持一致。

## 3. 线程私有数据

每条 chain 都通过串行 `run_sa_2opt` 或 `run_qlsa_2opt` 执行，因此以下数据天然私有：

- RNG；
- tour；
- Q table；
- current length；
- best tour；
- accepted/improved move 统计。

`DistanceMatrix` 是只读共享对象，可以在线程之间安全共享。

## 4. 为什么不做 move 级并行

本阶段不在内层 2-opt move 级别并行，原因是：

1. 单次 2-opt delta 是 O(1)，计算量很小；
2. move 级并行会引入频繁同步和接受决策冲突；
3. chain 级并行更容易保证随机过程可复现；
4. 后续 CUDA 多链实现也可以复用同样的抽象。

## 5. 结果归约

并行区域结束后，主线程串行遍历 `chain_results`：

- 选择 `best_length` 最小的 chain；
- 汇总所有 chain 的 `accepted_moves`；
- 汇总所有 chain 的 `improved_moves`；
- 将 best chain 的 `best_tour` 作为全局 best；
- 用完整 tour length 校验最终 `best_length`。

整体 `elapsed_ms` 记录多链 wall-clock time，不等于各 chain 时间之和。

## 6. Seed 可复现

每次 repeat 使用 `base_seed = seed + repeat_id`。每条 chain 使用稳定的 splitmix64 派生 seed：

```cpp
chain_seed = splitmix64(base_seed + 0x9E3779B97F4A7C15ULL * (chain_id + 1));
```

因此相同 `base_seed`、`chains`、`iterations` 下，`threads=1` 与 OpenMP 多线程运行具有相同的 chain seed 序列。每条 chain 的随机状态互不共享。

## 7. CSV 字段

当前 CSV 字段为：

```text
algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves
```

`algorithm` 可取：

- `sa`
- `qlsa`
- `sa-multichain`
- `qlsa-multichain`
- `sa-omp`
- `qlsa-omp`

`parallel` 可取：

- `none`
- `omp`

多链版本中，`best_length` 是所有 chain 的最短路径长度，`final_length` 是 best chain 的 final length，`accepted_moves` 和 `improved_moves` 是所有 chain 的总和。

## 8. CUDA 扩展方向

后续 CUDA 多链版本可以沿用当前结构：

- 每个 GPU thread 或 block 负责一条 chain；
- 距离矩阵一维数组复制到 GPU 全局内存；
- 每条 chain 独立 RNG；
- 每条 QLSA chain 独立 Q table；
- block/global reduction 找到全局 best chain。

当前 OpenMP 版本先固定多链接口、seed 规则和 CSV 指标，避免后续 CUDA 实现时重新设计实验协议。
