#!/usr/bin/env python3
"""Analyze large-instance CUDA chain vs candidate runs."""

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
    "instance", "dimension", "cuda_mode", "chains", "block_size", "candidates_per_iter",
    "iterations", "repeat", "bks", "best_length_min", "gap_min", "elapsed_ms_mean",
    "speedup_candidate_vs_chain", "speedup_vs_openmp", "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "large_cuda_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "large_cuda_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "large_cuda_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "fig17_large_cuda_chain_vs_candidate.png"))
    parser.add_argument("--openmp-summary", default=str(ROOT / "results" / "summary" / "large_openmp_summary.csv"))
    return parser.parse_args()


def load_bks() -> dict[str, int]:
    return {k: int(v) for k, v in json.loads(CONFIG.read_text(encoding="utf-8"))["bks"].items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, str]], openmp_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    bks = load_bks()
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok" and r.get("best_length")]
    skipped = [r for r in rows if r.get("status", "ok") != "ok"]
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok_rows:
        key = (
            row["instance"], row["dimension"], row.get("cuda_mode", ""),
            row["chains"], row.get("cuda_block_size", row.get("threads", "")),
            row.get("cuda_candidates_per_iter", ""), row["iterations"],
        )
        groups[key].append(row)

    chain_times: dict[tuple[str, str, str, str], float] = {}
    for key, group in groups.items():
        instance, _dimension, mode, chains, block, _candidates, iterations = key
        if mode == "chain":
            chain_times[(instance, chains, block, iterations)] = mean([float(r["elapsed_ms"]) for r in group])

    openmp_times: dict[tuple[str, str], float] = {}
    for row in openmp_rows:
        if row.get("algorithm") == "sa" and row.get("elapsed_ms_mean"):
            openmp_times[(row["instance"], row["iterations"])] = float(row["elapsed_ms_mean"])

    summary: list[dict[str, str]] = []
    for key, group in sorted(groups.items()):
        instance, dimension, mode, chains, block, candidates, iterations = key
        bks_value = bks.get(instance, 0)
        bests = [float(r["best_length"]) for r in group]
        elapsed = [float(r["elapsed_ms"]) for r in group]
        elapsed_mean = mean(elapsed)
        gap_min = ((min(bests) - bks_value) / bks_value * 100.0) if bks_value else 0.0
        chain_mean = chain_times.get((instance, chains, block, iterations), 0.0)
        speed_candidate = (chain_mean / elapsed_mean) if mode == "candidate" and elapsed_mean > 0 and chain_mean > 0 else (1.0 if mode == "chain" else 0.0)
        openmp_mean = openmp_times.get((instance, iterations), 0.0)
        speed_openmp = (openmp_mean / elapsed_mean) if openmp_mean > 0 and elapsed_mean > 0 else 0.0
        notes = []
        if mode == "candidate":
            notes.append("batch proposal variant")
        if speed_candidate and speed_candidate < 1.0 and mode == "candidate":
            notes.append("slower than chain")
        summary.append({
            "instance": instance,
            "dimension": dimension,
            "cuda_mode": mode,
            "chains": chains,
            "block_size": block,
            "candidates_per_iter": candidates,
            "iterations": iterations,
            "repeat": str(len(group)),
            "bks": str(bks_value) if bks_value else "",
            "best_length_min": f"{min(bests):.0f}",
            "gap_min": f"{gap_min:.4f}",
            "elapsed_ms_mean": f"{elapsed_mean:.4f}",
            "speedup_candidate_vs_chain": f"{speed_candidate:.4f}",
            "speedup_vs_openmp": f"{speed_openmp:.4f}" if speed_openmp else "",
            "notes": "; ".join(notes),
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
        path.with_suffix(".svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><text x='20' y='40'>No matplotlib</text></svg>", encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    if not summary:
        ax.axis("off")
        ax.text(0.5, 0.5, "未获得可统计的大实例 CUDA 数据\n请先放置 L1/L2 .tsp 文件", ha="center", va="center", fontsize=12)
        fig.suptitle("大实例 CUDA chain/candidate 状态")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    labels = [f"{r['instance']}-{r['cuda_mode']}" for r in summary]
    times = [float(r["elapsed_ms_mean"]) / 1000.0 for r in summary]
    colors = ["#d62728" if r["cuda_mode"] == "candidate" else "#2ca02c" for r in summary]
    ax.bar(labels, times, color=colors)
    ax.set_title("大实例 CUDA chain 与 candidate 用时")
    ax.set_ylabel("运行时间（秒）")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(path: Path, raw_path: Path, summary: list[dict[str, str]], skipped: list[dict[str, str]], figure: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 大规模 CUDA 实验分析",
        "",
        f"- 输入 raw CSV：`{raw_path.relative_to(ROOT) if raw_path.exists() else raw_path}`",
        f"- 输出图：`{figure.relative_to(ROOT)}`",
        "",
    ]
    if not summary:
        lines.extend(["本次没有获得 `status=ok` 的大实例 CUDA 数据。请先将对应 `.tsp` 文件放入 `data/`。", ""])
    else:
        lines.extend([
            "| instance | mode | best | min Gap | mean ms | speedup candidate vs chain | notes |",
            "|---|---|---:|---:|---:|---:|---|",
        ])
        for row in summary:
            lines.append(
                f"| {row['instance']} | {row['cuda_mode']} | {row['best_length_min']} | {row['gap_min']} | "
                f"{row['elapsed_ms_mean']} | {row['speedup_candidate_vs_chain']} | {row['notes']} |"
            )
        lines.append("")
    if skipped:
        lines.extend(["## 跳过或异常记录", "", "| instance | mode | status | error |", "|---|---|---|---|"])
        for row in skipped:
            lines.append(f"| {row.get('instance','')} | {row.get('cuda_mode','')} | {row.get('status','')} | {row.get('error','')} |")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- candidate mode 是 SA batch proposal 变体，不等同于单候选 SA proposal。",
        "- 若 candidate 慢于 chain 或 OpenMP，必须如实写入报告，不能声称 CUDA 是主性能结果。",
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
    openmp_path = Path(args.openmp_summary)
    if not openmp_path.is_absolute():
        openmp_path = ROOT / openmp_path
    rows = read_csv(raw_path)
    openmp_rows = read_csv(openmp_path)
    summary, skipped = summarize(rows, openmp_rows)
    write_summary(output, summary)
    make_figure(summary, figure)
    write_markdown(markdown, raw_path, summary, skipped, figure)
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    print(f"[ok] wrote {markdown.relative_to(ROOT)}")
    print(f"[ok] wrote {figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
