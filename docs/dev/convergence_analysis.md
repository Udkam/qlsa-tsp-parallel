# Convergence Proxy Analysis

本阶段未修改核心 CLI 添加逐迭代 `--trace`。为避免在最终冲刺阶段引入核心代码风险，本实验采用 increasing iteration budgets 的方式近似观察搜索质量随预算增加的变化。

每个点是相同 seed 下的一次独立运行，而不是同一次运行内部的逐迭代 trace。因此图中曲线应解释为 budget-sweep proxy，不应解释为严格收敛轨迹。

## Outputs

- `results/traces/berlin52_budget_sweep.csv`
- `figures/fig_convergence_berlin52.png`
- `results/traces/rat99_budget_sweep.csv`
- `figures/fig_convergence_rat99.png`
