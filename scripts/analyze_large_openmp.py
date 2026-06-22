#!/usr/bin/env python3
"""Analyze large-instance OpenMP runs and generate a report figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
SUMMARY_HEADER = [
    "instance", "tier", "dimension", "algorithm", "iterations", "chains", "threads", "repeat",
    "bks", "best_length_min", "gap_min", "gap_mean", "elapsed_ms_mean", "elapsed_ms_std",
    "moves_per_second", "accepted_moves_mean", "improved_moves_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "large_openmp_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "large_openmp_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "large_openmp_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "fig16_large_openmp_gap_time.png"))
    return parser.parse_args()


def load_bks() -> dict[str, int]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in config["bks"].items()}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def family(algorithm: str) -> str:
    return "qlsa" if algorithm.startswith("qlsa") else "sa"


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    bks = load_bks()
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok" and r.get("best_length")]
    skipped = [r for r in rows if r.get("status", "ok") != "ok"]
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok_rows:
        key = (
            row["instance"], row.get("tier", ""), row["dimension"], family(row["algorithm"]),
            row["iterations"], row["chains"], row["threads"],
        )
        groups[key].append(row)

    summary: list[dict[str, str]] = []
    for (instance, tier, dimension, alg, iterations, chains, threads), group in sorted(groups.items()):
        bks_value = bks.get(instance, 0)
        bests = [float(r["best_length"]) for r in group]
        elapsed = [float(r["elapsed_ms"]) for r in group]
        accepted = [float(r["accepted_moves"]) for r in group]
        improved = [float(r["improved_moves"]) for r in group]
        gaps = [((x - bks_value) / bks_value * 100.0) if bks_value else 0.0 for x in bests]
        repeat = len(group)
        moves = float(iterations) * float(chains)
        elapsed_s = mean(elapsed) / 1000.0 if elapsed else 0.0
        mps = moves / elapsed_s if elapsed_s > 0 else 0.0
        summary.append({
            "instance": instance,
            "tier": tier,
            "dimension": dimension,
            "algorithm": alg,
            "iterations": iterations,
            "chains": chains,
            "threads": threads,
            "repeat": str(repeat),
            "bks": str(bks_value) if bks_value else "",
            "best_length_min": f"{min(bests):.0f}",
            "gap_min": f"{min(gaps):.4f}",
            "gap_mean": f"{mean(gaps):.4f}",
            "elapsed_ms_mean": f"{mean(elapsed):.4f}",
            "elapsed_ms_std": f"{std(elapsed):.4f}",
            "moves_per_second": f"{mps:.2f}",
            "accepted_moves_mean": f"{mean(accepted):.4f}",
            "improved_moves_mean": f"{mean(improved):.4f}",
        })
    return summary, skipped


def write_summary(path: Path, summary: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
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
    except Exception:
        path.with_suffix(".svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='280'>"
            "<text x='30' y='80' font-size='24'>No matplotlib available for large OpenMP figure</text></svg>",
            encoding="utf-8",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    if not summary:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "未获得可统计的大实例 OpenMP 数据\n请先放置 L1/L2/L3 .tsp 文件", ha="center", va="center", fontsize=12)
        fig.suptitle("大实例 OpenMP 压力测试状态")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    labels = [f"{r['instance']}-{r['algorithm']}" for r in summary]
    gaps = [float(r["gap_min"]) for r in summary]
    times = [float(r["elapsed_ms_mean"]) / 1000.0 for r in summary]
    colors = ["#1f77b4" if r["algorithm"] == "sa" else "#ff7f0e" for r in summary]
    axes[0].bar(labels, gaps, color=colors)
    axes[0].set_title("最小 Gap")
    axes[0].set_ylabel("Gap（%）")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(labels, times, color=colors)
    axes[1].set_title("平均运行时间")
    axes[1].set_ylabel("运行时间（秒）")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("大规模 TSPLIB95 OpenMP 运行时间与 Gap")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(path: Path, raw_path: Path, summary: list[dict[str, str]], skipped: list[dict[str, str]], figure: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 大规模 OpenMP 实验分析",
        "",
        f"- 输入 raw CSV：`{raw_path.relative_to(ROOT) if raw_path.exists() else raw_path}`",
        f"- 输出图：`{figure.relative_to(ROOT)}`",
        "",
    ]
    if not summary:
        lines.extend([
            "本次没有获得 `status=ok` 的大实例 OpenMP 数据。当前最常见原因是 L1/L2/L3 `.tsp` 文件尚未放入 `data/`。",
            "",
        ])
    else:
        lines.extend([
            "| instance | tier | algorithm | best | min Gap | mean ms | moves/s |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for row in summary:
            lines.append(
                f"| {row['instance']} | {row['tier']} | {row['algorithm']} | {row['best_length_min']} | "
                f"{row['gap_min']} | {row['elapsed_ms_mean']} | {row['moves_per_second']} |"
            )
        lines.append("")
    if skipped:
        lines.extend([
            "## 跳过或异常记录",
            "",
            "| instance | tier | status | error |",
            "|---|---|---|---|",
        ])
        for row in skipped:
            lines.append(f"| {row.get('instance','')} | {row.get('tier','')} | {row.get('status','')} | {row.get('error','')} |")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 大实例实验的目标是验证工程可扩展性、时间趋势和 Gap 趋势，不追求所有实例达到 BKS。",
        "- 若 raw CSV 中存在 `missing` 或 `timeout`，不能把该实例写入性能结论。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_path = Path(args.input)
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    markdown = Path(args.markdown)
    if not markdown.is_absolute():
        markdown = ROOT / markdown
    figure = Path(args.figure)
    if not figure.is_absolute():
        figure = ROOT / figure

    rows = read_rows(raw_path)
    summary, skipped = summarize(rows)
    write_summary(output, summary)
    make_figure(summary, figure)
    write_markdown(markdown, raw_path, summary, skipped, figure)
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    print(f"[ok] wrote {markdown.relative_to(ROOT)}")
    print(f"[ok] wrote {figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
