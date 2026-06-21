# Python 忠实基线（python_ref）

本目录提供一个**忠实于参考论文机制**的 Python 实现，用于与 C++ 工程实现在同一台机器、同一组 TSPLIB95 数据、同一迭代预算下做直接对比。它求清晰而非求快，不参与 C++ 构建，也不替换主报告结果。

## 文件

| 文件 | 作用 |
|---|---|
| `tsplib_loader.py` | TSPLIB95 解析（EUC_2D/CEIL_2D/ATT/GEO/EXPLICIT）与共享 tour 工具（2-opt、double-bridge、Hamming） |
| `sa_paper.py` | 论文风格串行 SA（2-opt Metropolis + 指数退火） |
| `qlsa_paper.py` | candidate-leader QLSA：动作=选择 current/best/random/double-bridge leader，Q-learning + epsilon-greedy/softmax |
| `sb_qlsa_paper.py` | SB-QLSA：基于 current 与 best 的 Hamming 距离的 diversity-state，state-action Q 表 |
| `run_python_baseline.py` | 单次运行入口，输出与 C++ 完全一致的 CSV 字段 |

## 与 C++ 的一致性

- 距离使用 TSPLIB `nint` 取整（`int(x+0.5)`），与 C++ 整数距离一致；Python SA 在 berlin52 上提高预算可达 BKS=7542，佐证距离函数一致。
- CSV 字段：`algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves`。

## 用法

单次运行（在本目录内）：

```bat
py run_python_baseline.py --input ../data/berlin52.tsp --algorithm sa --iterations 100000 --seed 1 --repeat 3 --csv-only
py run_python_baseline.py --input ../data/rat99.tsp --algorithm qlsa-paper --iterations 100000 --seed 1 --repeat 3 --csv-only
py run_python_baseline.py --input ../data/eil101.tsp --algorithm sb-qlsa --iterations 100000 --seed 1 --repeat 3 --csv-only
```

同机对比实验（在项目根目录）：

```bat
py scripts\run_python_reference_comparison.py
```

输出：`results/raw/python_reference_raw.csv`、`results/summary/python_reference_summary.csv`、`figures/final/fig11_python_cpp_reference.png`、`docs/dev/python_reference_comparison.md`。

## 定位与限制

- 这是**缩小预算的同机参考实验**：迭代数（默认 100000）远低于主报告的 1,000,000，因此此处解质量不是本项目最优结果。
- Python 侧给出 candidate-leader QLSA 与 Hamming-diversity SB-QLSA 的忠实实现，用于弥补 C++ 侧 QLSA 为工程变体的机制差距；它是参考与对照，不声称 C++ 主结果改用了该机制。
