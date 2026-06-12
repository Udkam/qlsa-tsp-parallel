# Step 1 Design

## 1. 为什么先做串行 C++ 基线

本项目最终要比较串行版本、OpenMP 多线程版本和 CUDA GPU 版本。第一阶段先建立串行 C++ 基线，是为了先固定三个基础条件：

1. TSPLIB95 数据解析结果可靠；
2. 路径长度、2-opt delta 和随机种子可复现；
3. 后续并行版本有一个可直接对照的运行时间和解质量基准。

当前阶段不实现 CUDA，也不声称超过论文性能，只建立可编译、可测试、可重复运行的工程基线。

## 2. TSPLIB95 支持范围

已支持的 section：

- `NODE_COORD_SECTION`
- `EDGE_WEIGHT_SECTION`
- `EOF`

已支持的 `EDGE_WEIGHT_TYPE`：

- `EUC_2D`
- `CEIL_2D`
- `GEO`
- `ATT`
- `EXPLICIT`

已支持的 `EDGE_WEIGHT_FORMAT`：

- `FULL_MATRIX`
- `UPPER_ROW`
- `LOWER_ROW`
- `UPPER_DIAG_ROW`
- `LOWER_DIAG_ROW`

暂未支持：

- `EUC_3D`、`MAN_2D`、`MAX_2D` 等其他距离类型；
- `LOWER_COL`、`UPPER_COL` 等列优先显式矩阵格式；
- `.opt.tour` 最优路径文件解析；
- 非对称 ATSP 的专用格式。

这些格式不是第一阶段论文复现实例的主要需求，后续如实验数据需要再扩展。

## 3. 距离矩阵使用一维连续数组

`DistanceMatrix` 使用 `std::vector<int>` 存储 `n * n` 个距离，访问方式为：

```cpp
matrix[i * n + j]
```

这样做有三个原因：

1. 比 `vector<vector<int>>` 更连续，缓存局部性更好；
2. `dist(i, j)` 可以接近零开销；
3. 后续 CUDA 版本可以直接把 `raw()` 返回的一维数组拷贝到 GPU。

所有坐标型对称实例都会填充 `matrix[i*n+j] == matrix[j*n+i]`，对角线距离固定为 0。

## 4. SA 2-opt 的 O(1) delta

2-opt move 反转 tour 的区间 `[i, k]`。它只改变两条边：

- 旧边：`a-b`、`c-d`
- 新边：`a-c`、`b-d`

其中：

```text
a = tour[(i - 1 + n) % n]
b = tour[i]
c = tour[k]
d = tour[(k + 1) % n]
```

因此长度变化为：

```text
delta = dist(a,c) + dist(b,d) - dist(a,b) - dist(c,d)
```

接受 move 后只执行一次 `std::reverse`，并用 `current_length += delta` 更新当前路径长度。每轮不重新遍历完整 tour，最终再用完整路径长度校验 `best_length` 和 `final_length`。

## 5. 后续扩展方向

OpenMP 多线程多搜索链：

- 每个线程运行独立 SA/QLSA 搜索链；
- 每条链使用独立 seed 和本地最优解；
- 最后归约得到全局 best tour 和统计指标。

CUDA GPU 多搜索链：

- 将距离矩阵的一维数组复制到 GPU；
- 每个 thread 或 block 维护独立搜索链；
- 使用 GPU 端随机状态生成 2-opt move 和接受概率；
- 通过 block/global reduction 汇总最优解。

Q-Learning 策略选择：

- 在现有 SA 主循环中加入动作选择层；
- 动作可以对应不同邻域策略、候选 move 生成策略或温度/接受控制策略；
- 奖励由路径改进、接受情况、稳定性等指标给出；
- 串行 QLSA 稳定后再复用同一结构做 OpenMP/CUDA 多链并行。

## 6. 当前阶段边界

本阶段只完成：

- C++20 + CMake 工程骨架；
- TSPLIB95 解析器；
- 连续距离矩阵；
- Tour 工具函数；
- 串行 SA + 2-opt 基线；
- 可复现实验脚本和最小测试。

当前阶段不做论文性能结论，不写 CUDA，不引入复杂第三方库。
