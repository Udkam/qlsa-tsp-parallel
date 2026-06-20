#!/usr/bin/env python3
"""Analyze QLSA epsilon-greedy vs softmax comparison."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BKS = {"eil76": 538, "rat99": 1211, "eil101": 629}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/policy_comparison_raw.csv")
    p.add_argument("--summary", default="results/policy_comparison_summary.csv")
    p.add_argument("--markdown", default="docs/policy_comparison_analysis.md")
    p.add_argument("--figure", default="figures/final/fig_policy_comparison.png")
    return p.parse_args()


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_figure(summary_rows, figure_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        figure_path.with_suffix(".svg").write_text(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='900' height='400'>"
            f"<text x='30' y='50'>matplotlib unavailable: {exc}</text></svg>",
            encoding="utf-8",
        )
        return

    instances = sorted({r["instance"] for r in summary_rows})
    policies = ["epsilon-greedy", "softmax"]
    values = {
        (r["instance"], r["policy"]): float(r["gap_mean_percent"]) for r in summary_rows
    }
    x = list(range(len(instances)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = {"epsilon-greedy": "#ff7f0e", "softmax": "#ffbb78"}
    for idx, policy in enumerate(policies):
        data = [values.get((inst, policy), 0.0) for inst in instances]
        offs = [v + (idx - 0.5) * width for v in x]
        bars = ax.bar(offs, data, width, label=policy, color=colors[policy])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("QLSA 策略对比：epsilon-greedy vs softmax")
    ax.set_ylabel("Mean Gap (%)")
    ax.set_xticks(x, instances)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = read_rows(root / args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["instance"], row["policy"])].append(row)

    summary_rows = []
    for (inst, policy), items in sorted(groups.items()):
        bks = BKS.get(inst, 0)
        bests = [int(r["best_length"]) for r in items]
        elapsed = [float(r["elapsed_ms"]) for r in items]
        accepted = [float(r["accepted_moves"]) for r in items]
        improved = [float(r["improved_moves"]) for r in items]
        best_min = min(bests)
        best_mean = mean(bests)
        gap_min = (best_min - bks) / bks * 100.0 if bks else 0.0
        gap_mean = (best_mean - bks) / bks * 100.0 if bks else 0.0
        summary_rows.append(
            {
                "instance": inst,
                "policy": policy,
                "runs": len(items),
                "bks": bks,
                "best_length_min": f"{best_min}",
                "best_length_mean": f"{best_mean:.3f}",
                "gap_min_percent": f"{gap_min:.6f}",
                "gap_mean_percent": f"{gap_mean:.6f}",
                "elapsed_ms_mean": f"{mean(elapsed):.3f}",
                "elapsed_ms_std": f"{std(elapsed):.3f}",
                "accepted_moves_mean": f"{mean(accepted):.3f}",
                "improved_moves_mean": f"{mean(improved):.3f}",
            }
        )

    summary_path = root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary_rows[0].keys()) if summary_rows else [
        "instance", "policy", "runs", "bks", "best_length_min", "best_length_mean",
        "gap_min_percent", "gap_mean_percent", "elapsed_ms_mean", "elapsed_ms_std",
        "accepted_moves_mean", "improved_moves_mean",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_figure(summary_rows, root / args.figure)

    md = root / args.markdown
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Policy Comparison Analysis",
        "",
        "本实验比较当前 QLSA 实现中的 `epsilon-greedy` 与 `softmax` 策略。需要注意，本项目 softmax 是当前工程实现中的 action-selection policy，不完全等同论文 candidate-leader softmax 机制。",
        "",
        "## Experiment Setup",
        "",
        "- instances: eil76, rat99, eil101",
        "- algorithm: QLSA",
        "- parallel: OpenMP",
        "- chains: 32",
        "- threads: 8",
        "- iterations: 1,000,000",
        "- repeat: 5",
        "- alpha=0.1, gamma=0.9, epsilon=0.1",
        "",
        "## Summary",
        "",
        "| Instance | Policy | Runs | Best | Min Gap % | Mean Gap % | Mean ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['instance']} | {r['policy']} | {r['runs']} | {r['best_length_min']} | "
            f"{float(r['gap_min_percent']):.3f} | {float(r['gap_mean_percent']):.3f} | "
            f"{float(r['elapsed_ms_mean']):.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "该实验用于补充策略对比，而不是替代 Step 6B/6C 的 tuned validation 和 targeted high-budget 质量结论。最终报告可引用该图作为 QLSA 策略敏感性的辅助材料，但不应据此声称某一策略在所有实例上稳定占优。",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {summary_path.relative_to(root)}, {md.relative_to(root)}, {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
