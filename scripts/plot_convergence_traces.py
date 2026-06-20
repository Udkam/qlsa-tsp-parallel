#!/usr/bin/env python3
"""Plot budget-sweep convergence proxy figures."""

from __future__ import annotations

import csv
from pathlib import Path


BKS = {"berlin52": 7542, "rat99": 1211}


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gap(inst: str, best: float) -> float:
    return (best - BKS[inst]) / BKS[inst] * 100.0


def write_doc(root: Path, instances):
    lines = [
        "# Convergence Proxy Analysis",
        "",
        "本阶段未修改核心 CLI 添加逐迭代 `--trace`。为避免在最终冲刺阶段引入核心代码风险，本实验采用 increasing iteration budgets 的方式近似观察搜索质量随预算增加的变化。",
        "",
        "每个点是相同 seed 下的一次独立运行，而不是同一次运行内部的逐迭代 trace。因此图中曲线应解释为 budget-sweep proxy，不应解释为严格收敛轨迹。",
        "",
        "## Outputs",
        "",
    ]
    for inst in instances:
        lines.append(f"- `results/traces/{inst}_budget_sweep.csv`")
        lines.append(f"- `figures/fig_convergence_{inst}.png`")
    (root / "docs" / "convergence_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    trace_dir = root / "results" / "traces"
    fig_dir = root / "figures" / "final"
    fig_dir.mkdir(exist_ok=True)
    files = sorted(trace_dir.glob("*_budget_sweep.csv"))
    if not files:
        print("[warning] no trace files found")
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        missing = root / "figures" / "MISSING_FIGURES.md"
        missing.write_text(f"- convergence figures: matplotlib unavailable: {exc}\n", encoding="utf-8")
        return 0

    instances = []
    for path in files:
        rows = read_rows(path)
        if not rows:
            continue
        inst = rows[0]["instance"]
        instances.append(inst)
        by_alg = {"sa": [], "qlsa": []}
        for row in rows:
            by_alg[row["algorithm"]].append(row)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for alg, color in [("sa", "#1f77b4"), ("qlsa", "#ff7f0e")]:
            points = sorted(by_alg[alg], key=lambda r: int(r["iteration_budget"]))
            xs = [int(r["iteration_budget"]) for r in points]
            ys = [gap(inst, float(r["best_length"])) for r in points]
            ax.plot(xs, ys, marker="o", label=alg.upper(), color=color)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"{inst} 搜索预算扫描曲线")
        ax.set_xlabel("Iteration budget (log scale)")
        ax.set_ylabel("Best Gap (%)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        out = fig_dir / f"fig_convergence_{inst}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[figure] {out.relative_to(root)}")
    write_doc(root, instances)
    print("[done] wrote docs/convergence_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
