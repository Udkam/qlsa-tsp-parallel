#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze QLSA current/paper/paper-sb variant comparison experiments."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BKS = {
    "berlin52": 7542,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "a280": 2579,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "qlsa_variant_alignment_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "qlsa_variant_alignment_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "qlsa_variant_alignment_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig_qlsa_variant_alignment.png"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def group_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["instance"],
            row.get("qlsa_variant", ""),
            row.get("policy", ""),
            row.get("diversity_threshold", ""),
        )
        groups[key].append(row)

    summary: list[dict[str, str]] = []
    for (instance, variant, policy, threshold), items in sorted(groups.items()):
        bks = BKS.get(instance)
        best_values = [int(float(item["best_length"])) for item in items]
        elapsed = [float(item["elapsed_ms"]) for item in items]
        accepted = [float(item["accepted_moves"]) for item in items]
        improved = [float(item["improved_moves"]) for item in items]
        best_min = min(best_values)
        best_mean = mean([float(v) for v in best_values])
        gap_min = ((best_min - bks) / bks * 100.0) if bks else 0.0
        gap_mean = ((best_mean - bks) / bks * 100.0) if bks else 0.0
        summary.append({
            "instance": instance,
            "qlsa_variant": variant,
            "policy": policy,
            "diversity_threshold": threshold,
            "runs": str(len(items)),
            "iterations": items[0].get("iterations", ""),
            "chains": items[0].get("chains", ""),
            "threads": items[0].get("threads", ""),
            "bks": str(bks or ""),
            "best_length_min": str(best_min),
            "best_length_mean": f"{best_mean:.4f}",
            "gap_min": f"{gap_min:.4f}",
            "gap_mean": f"{gap_mean:.4f}",
            "elapsed_ms_mean": f"{mean(elapsed):.4f}",
            "elapsed_ms_std": f"{stddev(elapsed):.4f}",
            "accepted_moves_mean": f"{mean(accepted):.4f}",
            "improved_moves_mean": f"{mean(improved):.4f}",
        })
    return summary


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance",
        "qlsa_variant",
        "policy",
        "diversity_threshold",
        "runs",
        "iterations",
        "chains",
        "threads",
        "bks",
        "best_length_min",
        "best_length_mean",
        "gap_min",
        "gap_mean",
        "elapsed_ms_mean",
        "elapsed_ms_std",
        "accepted_moves_mean",
        "improved_moves_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_per_instance(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for instance in sorted({row["instance"] for row in rows}):
        candidates = [row for row in rows if row["instance"] == instance]
        candidates.sort(key=lambda r: (float(r["gap_min"]), float(r["elapsed_ms_mean"])))
        selected.append(candidates[0])
    return selected


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = best_per_instance(rows)
    lines = [
        "# QLSA 变体对齐实验分析",
        "",
        "本实验比较 `current`、`paper` 与 `paper-sb` 三种 QLSA 入口。`current` 是已有工程化状态/动作版本，`paper` 使用 candidate-leader 来源选择，`paper-sb` 在 candidate-leader 上加入路径多样性状态。",
        "",
        "实验目的不是替换已有 Step 5/6 结果，而是确认论文机制对齐变体已经进入同一 C++/OpenMP 实验流程，并观察其在代表实例上的质量和时间代价。",
        "",
        "## 每个实例的最佳配置",
        "",
        "| 实例 | 变体 | 策略 | 阈值 | 最短路径 | 最小偏差 | 平均时间(ms) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        threshold = row["diversity_threshold"] or "-"
        lines.append(
            f"| {row['instance']} | {row['qlsa_variant']} | {row['policy']} | {threshold} | "
            f"{row['best_length_min']} | {float(row['gap_min']):.4f}% | {float(row['elapsed_ms_mean']):.2f} |"
        )
    lines.extend([
        "",
        "## 解释",
        "",
        "- `paper` / `paper-sb` 的动作对象是候选来源，而不是 2-opt 跨度范围，因此它们与 `current` 的搜索行为不同。",
        "- `paper-sb` 的 diversity state 让 Q 表能区分当前路径与历史最优路径是否相近，适合观察论文状态机制的工程化效果。",
        "- 若某个实例上 `current` 仍然更好，说明已有工程化动作设计在该预算下更合适；若 `paper` 或 `paper-sb` 更好，则说明 candidate-leader 机制值得进一步调参。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception:
        lines = ["# Missing figure", "", "matplotlib 不可用，未生成 QLSA 变体图。"]
        path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    best_by_variant: dict[tuple[str, str], float] = {}
    instances = sorted({row["instance"] for row in rows})
    variants = ["current", "paper", "paper-sb"]
    for instance in instances:
        for variant in variants:
            candidates = [row for row in rows if row["instance"] == instance and row["qlsa_variant"] == variant]
            if candidates:
                best_by_variant[(instance, variant)] = min(float(row["gap_min"]) for row in candidates)

    x = list(range(len(instances)))
    width = 0.24
    colors = {
        "current": "#1f77b4",
        "paper": "#ff7f0e",
        "paper-sb": "#2ca02c",
    }
    labels = {
        "current": "当前工程化 QLSA",
        "paper": "候选来源 QLSA",
        "paper-sb": "带状态候选来源 QLSA",
    }

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
    for idx, variant in enumerate(variants):
        values = [best_by_variant.get((instance, variant), 0.0) for instance in instances]
        offset = (idx - 1) * width
        ax.bar([v + offset for v in x], values, width=width, label=labels[variant], color=colors[variant])
    ax.set_title("QLSA 机制对齐变体的最小偏差对比", fontsize=12)
    ax.set_xlabel("TSPLIB95 实例")
    ax.set_ylabel("最小偏差（%）")
    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.grid(axis="y", color="#d8e6f3", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    markdown_path = Path(args.markdown)
    if not markdown_path.is_absolute():
        markdown_path = ROOT / markdown_path
    figure_path = Path(args.figure)
    if not figure_path.is_absolute():
        figure_path = ROOT / figure_path

    rows = read_rows(input_path)
    summary = group_rows(rows)
    write_summary(output_path, summary)
    write_markdown(markdown_path, summary)
    write_figure(figure_path, summary)
    print(f"[ok] wrote {output_path.relative_to(ROOT)}")
    print(f"[ok] wrote {markdown_path.relative_to(ROOT)}")
    print(f"[ok] wrote {figure_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
