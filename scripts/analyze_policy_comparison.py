#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze QLSA epsilon-greedy vs softmax comparison."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BKS = {"eil76": 538, "rat99": 1211, "eil101": 629}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/policy_comparison_raw.csv")
    parser.add_argument("--summary", default="results/policy_comparison_summary.csv")
    parser.add_argument("--markdown", default="docs/policy_comparison_analysis.md")
    parser.add_argument("--figure", default="figures/fig_policy_comparison.png")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_figure(summary_rows, figure_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                "axes.unicode_minus": False,
                "font.size": 11,
            }
        )
    except Exception as exc:
        figure_path.with_suffix(".svg").write_text(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='900' height='400'>"
            f"<text x='30' y='50'>matplotlib unavailable: {exc}</text></svg>",
            encoding="utf-8",
        )
        return

    instances = sorted({row["instance"] for row in summary_rows})
    policies = ["epsilon-greedy", "softmax"]
    values = {
        (row["instance"], row["policy"]): float(row["gap_mean_percent"])
        for row in summary_rows
    }

    x_values = list(range(len(instances)))
    width = 0.35
    colors = {"epsilon-greedy": "#ff7f0e", "softmax": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for idx, policy in enumerate(policies):
        data = [values.get((inst, policy), 0.0) for inst in instances]
        offsets = [value + (idx - 0.5) * width for value in x_values]
        bars = ax.bar(offsets, data, width, label=policy, color=colors[policy])
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_title("QLSA 策略对比：epsilon-greedy 与 softmax")
    ax.set_ylabel("平均 Gap (%)")
    ax.set_xticks(x_values, instances)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#3b82f6")
    ax.legend(frameon=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = read_rows(root / args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["instance"], row["policy"])].append(row)

    summary_fields = [
        "instance",
        "policy",
        "runs",
        "best_length_min",
        "best_length_mean",
        "gap_min_percent",
        "gap_mean_percent",
        "elapsed_ms_mean",
        "elapsed_ms_std",
        "accepted_moves_mean",
        "improved_moves_mean",
    ]

    summary_rows = []
    for (instance, policy), items in sorted(groups.items()):
        bks = BKS[instance]
        best_lengths = [float(row["best_length"]) for row in items]
        elapsed = [float(row["elapsed_ms"]) for row in items]
        accepted = [float(row.get("accepted_moves", 0) or 0) for row in items]
        improved = [float(row.get("improved_moves", 0) or 0) for row in items]
        best_min = min(best_lengths)
        best_mean = mean(best_lengths)
        summary_rows.append(
            {
                "instance": instance,
                "policy": policy,
                "runs": str(len(items)),
                "best_length_min": f"{best_min:.3f}",
                "best_length_mean": f"{best_mean:.3f}",
                "gap_min_percent": f"{(best_min - bks) / bks * 100.0:.4f}",
                "gap_mean_percent": f"{(best_mean - bks) / bks * 100.0:.4f}",
                "elapsed_ms_mean": f"{mean(elapsed):.3f}",
                "elapsed_ms_std": f"{std(elapsed):.3f}",
                "accepted_moves_mean": f"{mean(accepted):.3f}",
                "improved_moves_mean": f"{mean(improved):.3f}",
            }
        )

    summary_path = root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_path = root / args.markdown
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# QLSA 策略对比分析",
        "",
        "本实验比较当前 C++ 实现中 epsilon-greedy 与 softmax 两种动作选择策略的平均偏差和运行时间。",
        "",
        "| instance | policy | runs | min Gap (%) | mean Gap (%) | mean ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['instance']} | {row['policy']} | {row['runs']} | "
            f"{row['gap_min_percent']} | {row['gap_mean_percent']} | {row['elapsed_ms_mean']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_figure(summary_rows, root / args.figure)
    print(f"[summary] {summary_path}")
    print(f"[markdown] {markdown_path}")
    print(f"[figure] {root / args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
