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

BASE_CONTRACT_FIELDS = [
    "algorithm",
    "dimension",
    "iterations",
    "init",
    "chains",
    "threads",
    "parallel",
]
EXTENDED_CONTRACT_FIELDS = [
    "requested_backend",
    "actual_backend",
    "backend_fallback",
    "iterations_completed",
    "deadline_reached",
    "actual_threads",
]
SUMMARY_CONTRACT_FIELDS = BASE_CONTRACT_FIELDS + EXTENDED_CONTRACT_FIELDS
UNIFORM_CONTRACT_FIELDS = [
    field for field in SUMMARY_CONTRACT_FIELDS if field not in {"algorithm", "dimension"}
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "qlsa_variant_alignment_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "qlsa_variant_alignment_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "qlsa_variant_alignment_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "fig_qlsa_variant_alignment.png"))
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


def normalized_diversity_metric(row: dict[str, str]) -> str:
    """Return the paper-sb metric while preserving legacy raw CSV semantics."""
    if row.get("qlsa_variant", "") != "paper-sb":
        return ""
    metric = row.get("diversity_metric", "").strip().lower()
    # Before the metric column existed, paper-sb implemented the paper's
    # position-Hamming state. Treat those archived rows explicitly as hamming
    # instead of silently mixing them with a new edge run.
    if not metric:
        return "hamming"
    if metric not in {"edge", "hamming"}:
        raise ValueError(f"unsupported paper-sb diversity_metric: {metric}")
    return metric


def normalized_contract(row: dict[str, str]) -> tuple[str, ...]:
    """Return the execution contract used to keep unlike runs separate."""
    return tuple(row.get(field, "").strip() for field in SUMMARY_CONTRACT_FIELDS)


def condition_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["instance"],
        row.get("qlsa_variant", ""),
        row.get("policy", ""),
        row.get("diversity_threshold", ""),
        normalized_diversity_metric(row),
        *normalized_contract(row),
    )


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Reject incomplete paired grids and heterogeneous execution contracts."""
    if not rows:
        raise ValueError("QLSA variant input contains no rows")

    required = {
        "instance",
        "seed",
        "best_length",
        "elapsed_ms",
        "accepted_moves",
        "improved_moves",
        *BASE_CONTRACT_FIELDS,
    }
    for index, row in enumerate(rows, start=2):
        missing = sorted(field for field in required if not row.get(field, "").strip())
        if missing:
            raise ValueError(f"row {index} is missing required fields: {', '.join(missing)}")

    # A single comparison file represents one execution contract. Parameter
    # scan dimensions (variant/policy/threshold/metric) are deliberately not
    # part of this check; backend, budget and initialization must be uniform.
    contracts = {
        tuple(row.get(field, "").strip() for field in UNIFORM_CONTRACT_FIELDS)
        for row in rows
    }
    if len(contracts) != 1:
        differing = [
            field
            for offset, field in enumerate(UNIFORM_CONTRACT_FIELDS)
            if len({contract[offset] for contract in contracts}) > 1
        ]
        raise ValueError(
            "QLSA variant input mixes execution contracts: " + ", ".join(differing)
        )

    dimensions_by_instance: dict[str, set[str]] = defaultdict(set)
    algorithms_by_variant: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        dimensions_by_instance[row["instance"]].add(row["dimension"])
        algorithms_by_variant[row.get("qlsa_variant", "")].add(row["algorithm"])
    if any(len(values) != 1 for values in dimensions_by_instance.values()):
        raise ValueError("QLSA variant input maps an instance to multiple dimensions")
    if any(len(values) != 1 for values in algorithms_by_variant.values()):
        raise ValueError("QLSA variant input maps a variant to multiple algorithms")

    seeds_by_instance_condition: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    row_count_by_instance_condition: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
    for row in rows:
        key = condition_key(row)
        instance_condition = (row["instance"], key[1:5])
        seed = row["seed"].strip()
        seeds_by_instance_condition[instance_condition].add(seed)
        row_count_by_instance_condition[instance_condition] += 1

    for key, seeds in seeds_by_instance_condition.items():
        if len(seeds) != row_count_by_instance_condition[key]:
            raise ValueError(f"duplicate seed in QLSA condition: instance={key[0]}, condition={key[1]}")

    by_instance: dict[str, list[set[str]]] = defaultdict(list)
    for (instance, _), seeds in seeds_by_instance_condition.items():
        by_instance[instance].append(seeds)
    for instance, seed_sets in by_instance.items():
        expected = seed_sets[0]
        if any(seeds != expected for seeds in seed_sets[1:]):
            raise ValueError(f"QLSA variant conditions for {instance} do not share paired seeds")


def group_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    validate_rows(rows)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = condition_key(row)
        groups[key].append(row)

    summary: list[dict[str, str]] = []
    for key, items in sorted(groups.items()):
        instance, variant, policy, threshold, metric = key[:5]
        bks = BKS.get(instance)
        best_values = [int(float(item["best_length"])) for item in items]
        elapsed = [float(item["elapsed_ms"]) for item in items]
        accepted = [float(item["accepted_moves"]) for item in items]
        improved = [float(item["improved_moves"]) for item in items]
        best_min = min(best_values)
        best_mean = mean([float(v) for v in best_values])
        gap_min = ((best_min - bks) / bks * 100.0) if bks else 0.0
        gap_mean = ((best_mean - bks) / bks * 100.0) if bks else 0.0
        contract_values = {
            field: key[5 + offset]
            for offset, field in enumerate(SUMMARY_CONTRACT_FIELDS)
        }
        summary.append({
            "instance": instance,
            "qlsa_variant": variant,
            "policy": policy,
            "diversity_threshold": threshold,
            "diversity_metric": metric,
            "runs": str(len(items)),
            **contract_values,
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
        "diversity_metric",
        "runs",
        *SUMMARY_CONTRACT_FIELDS,
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


def best_per_variant_and_metric(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    conditions = sorted(
        {
            (row["instance"], row["qlsa_variant"], row.get("diversity_metric", ""))
            for row in rows
        }
    )
    for instance, variant, metric in conditions:
        candidates = [
            row
            for row in rows
            if row["instance"] == instance
            and row["qlsa_variant"] == variant
            and row.get("diversity_metric", "") == metric
        ]
        candidates.sort(key=lambda r: (float(r["gap_min"]), float(r["elapsed_ms_mean"])))
        selected.append(candidates[0])
    return selected


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = best_per_variant_and_metric(rows)
    lines = [
        "# QLSA 变体对齐实验分析",
        "",
        "本实验比较 `current`、`paper` 与 `paper-sb` 三种 QLSA 入口。`current` 是已有工程化状态/动作版本，`paper` 使用 candidate-leader 来源选择，`paper-sb` 在 candidate-leader 上加入路径多样性状态。",
        "",
        "历史 raw CSV 没有 diversity metric 列时，`paper-sb` 行按当时实现标为 `hamming`；`edge` 样本以独立条件汇总。",
        "",
        "三种变体使用统一的 C++/OpenMP 执行合同和共同种子，表中结果展示参数网格内达到的质量和时间代价。",
        "",
        "## 各变体与度量条件的最佳配置",
        "",
        "| 实例 | 变体 | 多样性度量 | 策略 | 阈值 | 最短路径 | 最小偏差 | 平均时间(ms) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        threshold = row["diversity_threshold"] or "-"
        metric = row.get("diversity_metric", "") or "-"
        lines.append(
            f"| {row['instance']} | {row['qlsa_variant']} | {metric} | {row['policy']} | {threshold} | "
            f"{row['best_length_min']} | {float(row['gap_min']):.4f}% | {float(row['elapsed_ms_mean']):.2f} |"
        )
    lines.extend([
        "",
        "## 解释",
        "",
        "- `paper` / `paper-sb` 的动作对象是候选来源，而不是 2-opt 跨度范围，因此它们与 `current` 的搜索行为不同。",
        "- `paper-sb` 的 diversity state 让 Q 表能区分当前路径与历史最优路径是否相近；Hamming 与 edge 是不同实验条件，分别报告。",
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

    best_by_variant: dict[tuple[str, str, str], float] = {}
    instances = sorted({row["instance"] for row in rows})
    series: list[tuple[str, str]] = [("current", ""), ("paper", "")]
    paper_sb_metrics = sorted(
        {row.get("diversity_metric", "") for row in rows if row["qlsa_variant"] == "paper-sb"}
    )
    series.extend(("paper-sb", metric) for metric in paper_sb_metrics)
    for instance in instances:
        for variant, metric in series:
            candidates = [
                row
                for row in rows
                if row["instance"] == instance
                and row["qlsa_variant"] == variant
                and row.get("diversity_metric", "") == metric
            ]
            if candidates:
                best_by_variant[(instance, variant, metric)] = min(
                    float(row["gap_min"]) for row in candidates
                )

    x = list(range(len(instances)))
    width = min(0.24, 0.8 / max(1, len(series)))
    colors = {
        ("current", ""): "#1f77b4",
        ("paper", ""): "#ff7f0e",
        ("paper-sb", "hamming"): "#2ca02c",
        ("paper-sb", "edge"): "#9467bd",
    }
    labels = {
        ("current", ""): "当前工程化 QLSA",
        ("paper", ""): "候选来源 QLSA",
        ("paper-sb", "hamming"): "带状态候选来源 QLSA（Hamming）",
        ("paper-sb", "edge"): "带状态候选来源 QLSA（edge）",
    }

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
    for idx, (variant, metric) in enumerate(series):
        values = [best_by_variant.get((instance, variant, metric), math.nan) for instance in instances]
        offset = (idx - (len(series) - 1) / 2.0) * width
        key = (variant, metric)
        ax.bar(
            [v + offset for v in x],
            values,
            width=width,
            label=labels[key],
            color=colors[key],
        )
    if paper_sb_metrics == ["hamming"]:
        title = "QLSA 机制对齐变体的最小偏差对比（paper-sb：Hamming）"
    elif paper_sb_metrics:
        title = "QLSA 机制对齐变体的最小偏差对比（paper-sb 按度量分列）"
    else:
        title = "QLSA 机制对齐变体的最小偏差对比"
    ax.set_title(title, fontsize=12)
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
