#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze representative OpenMP runs selected from the full data/ rerun.

The raw experiment covers every .tsp file in data/. The report-facing view uses
a curated subset so the report can compare scale classes without turning into a
38-instance table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LARGE_CONFIG = ROOT / "configs" / "large_tsplib_instances.json"

SMALL_BKS = {
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "dantzig42": 699,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "gr17": 2085,
    "gr24": 1272,
    "gr48": 5046,
    "hk48": 11461,
    "pr76": 108159,
    "rat99": 1211,
    "st70": 675,
    "swiss42": 1273,
    "ulysses16": 6859,
    "ulysses22": 7013,
}

SUMMARY_FIELDS = [
    "instance",
    "tier",
    "dimension",
    "algorithm",
    "iterations",
    "chains",
    "threads",
    "runs",
    "bks",
    "best_length_min",
    "gap_min_percent",
    "gap_mean_percent",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "moves_per_second",
    "accepted_moves_mean",
    "improved_moves_mean",
]

DEFAULT_REPRESENTATIVES = [
    "berlin52",
    "eil76",
    "rat99",
    "eil101",
    "a280",
    "rat575",
    "rat783",
    "dsj1000",
    "u1060",
    "vm1084",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "final_all_data_openmp_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "final_representative_openmp_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "final_representative_openmp_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig_course_11_representative_openmp.png"))
    parser.add_argument(
        "--representatives",
        nargs="+",
        default=DEFAULT_REPRESENTATIVES,
        help="Instance names to keep in the report-facing analysis. Default: ten instances across small, hard, L1/L2, and L3 cases.",
    )
    return parser.parse_args()


def load_bks() -> dict[str, int]:
    bks = dict(SMALL_BKS)
    if LARGE_CONFIG.exists():
        payload = json.loads(LARGE_CONFIG.read_text(encoding="utf-8"))
        bks.update({k: int(v) for k, v in payload.get("bks", {}).items()})
    return bks


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def family(algorithm: str) -> str:
    return "qlsa" if algorithm.startswith("qlsa") else "sa"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))


def summarize(rows: list[dict[str, str]], representatives: set[str]) -> list[dict[str, str]]:
    bks = load_bks()
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok" or not row.get("best_length"):
            continue
        if row["instance"] not in representatives:
            continue
        key = (
            row["instance"],
            row.get("tier", "custom"),
            row["dimension"],
            family(row["algorithm"]),
            row["iterations"],
            row["chains"],
            row["threads"],
        )
        groups[key].append(row)

    out: list[dict[str, str]] = []
    for (instance, tier, dimension, alg, iterations, chains, threads), group in sorted(groups.items()):
        bests = [float(r["best_length"]) for r in group]
        elapsed = [float(r["elapsed_ms"]) for r in group]
        accepted = [float(r["accepted_moves"]) for r in group]
        improved = [float(r["improved_moves"]) for r in group]
        bks_value = bks.get(instance, 0)
        gaps = [((x - bks_value) / bks_value * 100.0) if bks_value else 0.0 for x in bests]
        elapsed_s = mean(elapsed) / 1000.0 if elapsed else 0.0
        moves = float(iterations) * float(chains)
        out.append(
            {
                "instance": instance,
                "tier": tier,
                "dimension": dimension,
                "algorithm": alg,
                "iterations": iterations,
                "chains": chains,
                "threads": threads,
                "runs": str(len(group)),
                "bks": str(bks_value) if bks_value else "",
                "best_length_min": f"{min(bests):.0f}",
                "gap_min_percent": f"{min(gaps):.4f}",
                "gap_mean_percent": f"{mean(gaps):.4f}",
                "elapsed_ms_mean": f"{mean(elapsed):.4f}",
                "elapsed_ms_std": f"{std(elapsed):.4f}",
                "moves_per_second": f"{(moves / elapsed_s) if elapsed_s > 0 else 0.0:.2f}",
                "accepted_moves_mean": f"{mean(accepted):.4f}",
                "improved_moves_mean": f"{mean(improved):.4f}",
            }
        )
    return out


def write_summary(path: Path, summary: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)


def make_figure(summary: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 11
    except Exception:
        path.with_suffix(".svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='260'>"
            "<text x='30' y='80' font-size='22'>缺少 matplotlib，无法生成全实例图</text></svg>",
            encoding="utf-8",
        )
        return

    rows = [r for r in summary if r.get("bks")]
    by_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_instance[row["instance"]][row["algorithm"]] = row
    instances = sorted(by_instance, key=lambda name: int(by_instance[name].get("sa", by_instance[name].get("qlsa"))["dimension"]))
    x = list(range(len(instances)))
    width = 0.36

    def series(metric: str, alg: str) -> list[float]:
        values: list[float] = []
        for name in instances:
            row = by_instance[name].get(alg)
            values.append(float(row[metric]) if row else 0.0)
        return values

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=300)
    axes[0].bar([i - width / 2 for i in x], series("gap_min_percent", "sa"), width, label="SA", color="#1f77b4")
    axes[0].bar([i + width / 2 for i in x], series("gap_min_percent", "qlsa"), width, label="QLSA", color="#ff7f0e")
    axes[0].set_title("代表实例最小偏差")
    axes[0].set_ylabel("最小偏差（%）")
    axes[0].set_xticks(x, instances, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)

    axes[1].bar([i - width / 2 for i in x], [v / 1000.0 for v in series("elapsed_ms_mean", "sa")], width, label="SA", color="#1f77b4")
    axes[1].bar([i + width / 2 for i in x], [v / 1000.0 for v in series("elapsed_ms_mean", "qlsa")], width, label="QLSA", color="#ff7f0e")
    axes[1].set_title("代表实例平均运行时间")
    axes[1].set_ylabel("运行时间（秒）")
    axes[1].set_xticks(x, instances, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False)

    fig.suptitle("OpenMP 代表实例结果（64 条搜索链，8 线程）")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(
    path: Path,
    summary: list[dict[str, str]],
    raw_path: Path,
    figure: Path,
    raw_rows: list[dict[str, str]],
    representatives: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_instances = sorted({r["instance"] for r in raw_rows if r.get("status", "ok") == "ok"})
    instances = [name for name in representatives if any(r["instance"] == name for r in summary)]
    dims = [int(r["dimension"]) for r in summary] or [0]
    zero_gap = [r for r in summary if r.get("bks") and abs(float(r["gap_min_percent"])) < 1e-9]
    lines = [
        "# 代表实例 OpenMP 压力测试分析",
        "",
        f"- 原始结果：`{raw_path.as_posix()}`",
            f"- 原始结果覆盖 `data/` 中 {len(raw_instances)} 个实例；本分析选取 {len(instances)} 个代表实例进入报告。",
        f"- 代表实例：{', '.join(instances)}",
        f"- 代表实例维度范围：{min(dims)} 到 {max(dims)}",
        f"- 代表实例中达到 BKS 的算法-实例组合数：{len(zero_gap)}",
        f"- 图表：`{figure.as_posix()}`",
        "",
        "## 代表实例结果",
        "",
        "| 实例 | 算法 | 城市数 | 最短路径长度 | 最小偏差 | 平均偏差 | 平均运行时间 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    order = {name: i for i, name in enumerate(representatives)}
    for r in sorted(summary, key=lambda row: (order.get(row["instance"], 999), row["algorithm"])):
        lines.append(
            f"| {r['instance']} | {r['algorithm'].upper()} | {r['dimension']} | {r['best_length_min']} | "
            f"{float(r['gap_min_percent']):.3f}% | {float(r['gap_mean_percent']):.3f}% | "
            f"{float(r['elapsed_ms_mean']) / 1000.0:.3f} s |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本轮实验实际覆盖 `data/` 目录下所有可用 `.tsp` 文件，统一使用 OpenMP、64 条搜索链、8 线程、1,000,000 次迭代和 3 次重复运行。",
            f"为避免正文堆叠过多实例，报告只选取 {len(instances)} 个代表实例进行分析；完整原始结果保留在原始结果表中。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_path = Path(args.input)
    summary_path = Path(args.output)
    markdown_path = Path(args.markdown)
    figure_path = Path(args.figure)

    raw_rows = read_rows(raw_path)
    representatives = list(dict.fromkeys(args.representatives))
    summary = summarize(raw_rows, set(representatives))
    write_summary(summary_path, summary)
    make_figure(summary, figure_path)
    write_markdown(markdown_path, summary, raw_path, figure_path, raw_rows, representatives)

    print(f"[ok] wrote {summary_path}")
    print(f"[ok] wrote {markdown_path}")
    print(f"[ok] wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
