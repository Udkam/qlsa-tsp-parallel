# QLSA 论文机制与 C++ 实现对齐记录

## 目标

参考论文中的 QLSA 和 State-Based QLSA 以 candidate-leader 选择和 diversity state 为核心。本项目早期 C++ QLSA 采用工程化状态/动作设计，便于 OpenMP、CUDA 和调参实验。本轮补充可选 C++ 变体，使同一可执行文件中同时保留历史实现和更接近论文机制的实现。

## 变体说明

| 变体 | CLI 参数 | 状态数 | 动作含义 | 用途 |
|---|---|---:|---|---|
| 工程化当前版本 | `--qlsa_variant current` | 5 | short / medium / long 2-opt span | 兼容已有实验和默认结论 |
| 论文 stateless 版本 | `--qlsa_variant paper` | 1 | current、global best、random、double-bridge | 对齐论文 candidate-leader 思想 |
| 论文 state-based 版本 | `--qlsa_variant paper-sb` | 2 | current、global best、random、double-bridge | 在 candidate-leader 上加入 diversity state |

默认值是 `current`，因此历史命令和已有 CSV 不会改变。

## candidate-leader 机制

`paper` 与 `paper-sb` 每轮先由 Q 表选择候选来源：

1. 当前路径；
2. 当前链历史最优路径；
3. 随机路径；
4. double-bridge 扰动路径。

选出 leader 后，程序对 leader 执行一次 2-opt Metropolis 更新，再将更新后的 leader 与当前路径比较并按 Metropolis 准则决定是否替换当前路径。奖励定义为当前路径长度的相对改善，Q 表按标准 Q-learning 公式更新。

## diversity state

`paper-sb` 使用当前路径与历史最优路径的 Hamming 距离划分状态：

- 距离比例低于 `--diversity_threshold`：低多样性状态；
- 距离比例不低于阈值：高多样性状态。

默认阈值为 `0.5`。该设计与论文中的 diversity state 思想对齐，但具体状态划分仍是本项目的工程实现口径。

## 与当前工程化 QLSA 的差异

| 对比项 | 当前工程化 QLSA | paper / paper-sb |
|---|---|---|
| 动作对象 | 选择 2-opt 跨度范围 | 选择候选 leader 来源 |
| 状态来源 | 近期 delta 均值离散化 | 无状态或路径多样性状态 |
| 搜索动作 | 每轮直接对当前路径采样 2-opt | 先构造 leader，再做一次 2-opt refinement |
| 与已有实验关系 | Step 5/6/large 默认结果使用该口径 | 新增机制对齐和后续对比入口 |
| CUDA 支持 | 支持 QLSA candidate | CUDA 仍只支持 current 变体 |

## 实现位置

- `include/tsp/qlsa.hpp`：新增 `variant` 和 `diversity_threshold` 参数。
- `src/qlsa.cpp`：新增 `run_qlsa_paper_style`，实现 candidate-leader 和 diversity state。
- `src/main.cpp`：新增 `--qlsa_variant current|paper|paper-sb` 与 `--diversity_threshold`。
- `tests/test_qlsa_paper_lite.cpp`：覆盖 paper 和 paper-sb 在 square4 上的 smoke。

## 验证命令

```powershell
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input tests\fixtures\square4.tsp --iterations 500 --seed 1 --init random --csv-only
.\build-cuda-ninja\tsp_sa.exe --qlsa --qlsa_variant paper-sb --input data\berlin52.tsp --iterations 100000 --seed 1 --init nn --csv-only
ctest --test-dir build-cuda-ninja --output-on-failure
```

验证结果：

- `square4` 上 paper-sb 返回最短路径长度 40；
- `berlin52` 上 paper-sb 可运行并输出 CSV；
- `ctest` 6/6 通过。

## 报告边界

可以写：C++ 端已经提供可选 paper / paper-sb 机制对齐变体。  
不应写：已有 Step 5/6 主实验全部自动变成 paper-sb 结果。  
原因是历史结果仍来自 `current` 变体，paper / paper-sb 需要单独实验和参数调优后才能进入主结论。
