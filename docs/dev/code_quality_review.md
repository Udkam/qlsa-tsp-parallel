# Code Quality Review

本文件记录本轮“极致优化”阶段的代码静态审查结果。审查原则是：不盲目重写核心算法；对低风险问题可修复，对高风险算法改动先记录建议。当前本轮未修改 `src/`、`include/`、`CMakeLists.txt` 中的核心算法代码。

## 1. 审查范围

审查文件包括：

- `include/tsp/*.hpp`
- `src/*.cpp`
- `src/cuda_kernels.cu`
- `tests/test_small_instance.cpp`
- `tests/test_parallel.cpp`
- `tests/test_cuda.cpp`
- `src/main.cpp`

已运行测试：

```powershell
ctest --test-dir build-cuda-ninja --output-on-failure
```

结果：3/3 tests passed。

## 2. TSPLIB parser

确认点：

- `tsplib_parser.cpp` 能识别 `NODE_COORD_SECTION`、`EDGE_WEIGHT_SECTION` 和 `EOF`。
- `DistanceMatrix` 阶段支持 `EUC_2D`、`CEIL_2D`、`ATT`、`GEO` 和 `EXPLICIT`。
- 显式矩阵格式在需求文档中要求支持 `FULL_MATRIX`、`UPPER_ROW`、`LOWER_ROW`、`UPPER_DIAG_ROW`、`LOWER_DIAG_ROW`。

风险与缺口：

- 当前测试主要使用坐标型 `square4.tsp`，尚未看到单独的 EXPLICIT 小实例测试。
- 建议后续新增一个 4 城市 `EXPLICIT FULL_MATRIX` 或 `UPPER_ROW` fixture，验证 parser 到 DistanceMatrix 的完整路径。

结论：parser 工程能力较完整，但 EXPLICIT 测试覆盖仍是缺口。

## 3. DistanceMatrix

确认点：

- `DistanceMatrix` 使用一维连续数组存储 \(n \times n\) 距离矩阵。
- `raw()` 可直接返回底层连续数组，适合 CUDA 拷贝。
- 对角线为 0，对称实例按对称矩阵使用。

风险与缺口：

- 大实例内存为 \(O(n^2)\)，对当前 TSPLIB 小中型实例足够；若扩展到更大规模，需要考虑压缩存储或分块策略。

结论：当前设计符合高性能 C++ 和 CUDA 后端需求。

## 4. 2-opt delta 与 Tour

确认点：

- `delta_2opt` 使用旧边 `a-b`、`c-d` 和新边 `a-c`、`b-d` 做 O(1) delta。
- 已处理非法 move，包括整圈反转等边界。
- `test_small_instance.cpp` 已验证 delta 与完整重算长度一致。
- `tour_length` 会检查 tour 合法性并重算完整长度，用于结果校验。

风险与缺口：

- 当前测试只覆盖小型正方形实例。若要更严谨，可增加随机 tour 多组 2-opt delta fuzz test。

结论：核心 2-opt delta 逻辑正确性已有基础测试支撑。

## 5. SA 实现

确认点：

- SA 使用 O(1) delta，不在每次 move 后完整重算路径。
- 接受准则符合 Metropolis：`delta < 0` 直接接受，否则按 `exp(-delta/T)` 接受。
- 温度使用预计算 alpha 的指数退火。
- 结束时对 best 和 final length 进行完整 tour length 校验。

风险与缺口：

- 默认参数适合当前实验集，但不同实例仍需调参。
- 未记录收敛曲线；如果要分析搜索过程，需要新增 trace 或外部采样脚本。

结论：SA 是当前项目最稳定的算法基线。

## 6. QLSA 实现

确认点：

- `QLSAParams` 支持 `alpha`、`gamma`、`policy`、`epsilon`、`softmax_temperature`、`state_window`、`delta_scale` 和动作集合。
- 支持 epsilon-greedy 和 softmax 两种策略。
- `update_q_value` 实现标准 Q-learning 更新。
- `test_small_instance.cpp` 已覆盖 Q 表更新、状态离散化、reward、epsilon-greedy 和 softmax smoke。

风险与缺口：

- 当前 QLSA 是状态/动作离散化的工程变体，不是论文中完整 candidate-leader QLSA。
- 未发现完整 double-bridge candidate leader 机制。
- 未发现论文 SB-QLSA 的 Hamming-distance diversity state 完整机制。

结论：QLSA 实现可作为“基于论文思想的工程化变体”，不应在报告中声称完全复刻论文 SB-QLSA。

## 7. OpenMP 并行

确认点：

- `run_parallel_chains` 使用 chain-level parallel for。
- 每条 chain 使用 `chain_seed(base_seed, chain_id)` 派生 seed。
- 每条 chain 写入独立 `chain_results[chain_id]`，避免共享写冲突。
- 全局最优在并行区结束后串行归约。
- `test_parallel.cpp` 验证多链结果合法性、seed 派生和重复运行可复现。

风险与缺口：

- `chains=1` 与串行单链的比较需要注意：parallel path 会使用 `chain_seed(base_seed, 0)`，而直接串行 SA 使用原始 `seed`，因此默认不应要求二者结果完全一致。若要比较，需要显式使用派生后的 seed。

结论：OpenMP 并行实现稳健，是最终报告的主要性能结论。

## 8. RNG 与可复现性

确认点：

- 串行 SA/QLSA 支持 seed。
- 多链使用 `splitmix64(base_seed + stride * (chain_id+1))` 派生 chain seed。
- `test_parallel.cpp` 和 `test_cuda.cpp` 均验证重复运行的 chain seed 与 best length 可复现。

风险与缺口：

- CUDA kernel 内 RNG 与 CPU RNG 不同，因此不能要求 CUDA 与 CPU 同 seed 逐步一致；只能要求同一后端同参数可复现。

结论：项目满足实验可复现的基本要求。

## 9. CUDA

确认点：

- 项目包含 `src/cuda_kernels.cu`，当前 `ctest` 中 `test_cuda` 通过。
- `cuda.cpp` 保留 runtime 或 build-time 不可用时的 fallback 路径，并输出 warning。
- CUDA result 在 host 端重新校验 best length。

风险与缺口：

- CUDA 小实例性能不优于 OpenMP，报告中应定位为工程扩展。
- 如果发生 fallback，不能把结果作为真实 GPU 性能。
- 由于 CUDA 和 CPU RNG/执行路径不同，不应要求二者完全同轨迹。

结论：CUDA-facing plumbing 与 kernel 工程已完成，但性能优化仍有空间。

## 10. CLI 与 CSV

确认点：

- CLI 支持 `--qlsa`、`--parallel none|omp|cuda`、`--chains`、`--threads`、`--cuda_block_size`、`--seed`、`--repeat`、`--csv-only`、`--policy` 等参数。
- CSV 输出包含 algorithm、instance、dimension、iterations、seed、init、chains、threads、parallel、best_length、final_length、elapsed_ms、accepted_moves、improved_moves。
- Step 5/6 脚本基于该 CSV schema 做自动分析。

风险与缺口：

- 当前 CLI 没有 `--trace`、`--trace_interval`；若要做真实收敛曲线，需要新增核心代码。考虑到当前优先级和风险，本轮不强行改核心代码。
- 当前 CLI 没有 `--qlsa-variant paper-lite`；实现该变体会影响算法接口和实验口径，建议作为后续扩展。

结论：CLI 足以支撑当前报告结果，但不包含收敛曲线与 paper-lite 变体。

## 11. 测试覆盖总结

已覆盖：

- 4 城市正方形实例距离矩阵和 tour length；
- random tour 与 nearest-neighbor tour 合法性；
- 2-opt delta 与完整重算一致；
- SA 小实例运行；
- QLSA Q 表更新、动作选择策略和小实例运行；
- OpenMP 多链合法性、seed 派生和重复运行可复现；
- CUDA-facing smoke test 与重复运行。

缺口：

- TSPLIB EXPLICIT 小型 fixture 测试；
- 更广泛的 2-opt delta randomized fuzz test；
- softmax policy 在真实 TSPLIB 实例上的系统实验；
- convergence trace；
- paper-lite QLSA 变体测试。

## 12. 审查结论

当前代码可以支撑最终课程报告的核心结论：

1. SA/QLSA 串行与多链并行实现完整；
2. OpenMP 是主要稳定性能提升；
3. CUDA 已完成工程实现和 smoke test，但不作为小实例主加速结论；
4. QLSA 当前实现是论文思想的工程变体，不是完整 SB-QLSA；
5. 所有强结论都应追溯到已有 CSV、测试或论文表格数据。
