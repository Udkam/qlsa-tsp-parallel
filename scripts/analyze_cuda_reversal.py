#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze CUDA candidate serial-vs-parallel reversal results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BKS = {
    "berlin52": 7542,
    "eil101": 629,
    "ch130": 6110,
    "a280": 2579,
    "lin318": 42029,
    "rat575": 6773,
}
HEADER = [
    "instance",
    "algorithm",
    "cuda_mode",
    "block_size",
    "candidates_per_iter",
    "iterations",
    "serial_ms_mean",
    "parallel_ms_mean",
    "parallel_speedup_vs_serial",
    "serial_best",
    "parallel_best",
    "serial_gap",
    "parallel_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "cuda_reversal_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "cuda_reversal_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "cuda_reversal_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig22_cuda_parallel_reversal.png"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("cuda_mode") == "candidate"]


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        reversal = row.get("cuda_reversal_mode", "serial")
        key = (
            row["instance"],
            row["algorithm"],
            row.get("cuda_mode", "candidate"),
            row.get("cuda_block_size", row.get("threads", "")),
            row.get("cuda_candidates_per_iter", ""),
            row["iterations"],
        )
        groups[key][reversal].append(row)

    out: list[dict[str, str]] = []
    for key, by_reversal in groups.items():
        if "serial" not in by_reversal or "parallel" not in by_reversal:
            continue
        instance, algorithm, mode, block_size, candidates, iterations = key
        serial_rows = by_reversal["serial"]
        parallel_rows = by_reversal["parallel"]
        serial_times = [float(r["elapsed_ms"]) for r in serial_rows]
        parallel_times = [float(r["elapsed_ms"]) for r in parallel_rows]
        serial_best = min(int(r["best_length"]) for r in serial_rows)
        parallel_best = min(int(r["best_length"]) for r in parallel_rows)
        bks = BKS.get(instance)
        serial_gap = math.nan if bks is None else (serial_best - bks) / bks * 100.0
        parallel_gap = math.nan if bks is None else (parallel_best - bks) / bks * 100.0
        serial_ms = statistics.fmean(serial_times)
        parallel_ms = statistics.fmean(parallel_times)
        out.append({
            "instance": instance,
            "algorithm": algorithm,
            "cuda_mode": mode,
            "block_size": block_size,
            "candidates_per_iter": candidates,
            "iterations": iterations,
            "serial_ms_mean": f"{serial_ms:.4f}",
            "parallel_ms_mean": f"{parallel_ms:.4f}",
            "parallel_speedup_vs_serial": f"{serial_ms / parallel_ms:.4f}" if parallel_ms > 0 else "",
            "serial_best": str(serial_best),
            "parallel_best": str(parallel_best),
            "serial_gap": "" if math.isnan(serial_gap) else f"{serial_gap:.4f}",
            "parallel_gap": "" if math.isnan(parallel_gap) else f"{parallel_gap:.4f}",
        })
    return sorted(out, key=lambda r: (r["instance"], r["algorithm"], int(r["block_size"])))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CUDA parallel reversal 分析",
        "",
        "本实验比较 CUDA candidate mode 中 thread-0 serial reversal 与 block cooperative parallel reversal。该优化不改变 candidate 选择规则，但改变 accepted 2-opt segment reversal 的执行方式。",
        "",
        "| instance | algorithm | block | candidates | serial ms | parallel ms | speedup | serial Gap | parallel Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['algorithm']} | {row['block_size']} | {row['candidates_per_iter']} | "
            f"{row['serial_ms_mean']} | {row['parallel_ms_mean']} | {row['parallel_speedup_vs_serial']} | "
            f"{row['serial_gap']} | {row['parallel_gap']} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- speedup 大于 1 才说明 parallel reversal 在该配置下降低 reversal 相关时间。",
        "- 若 speedup 不稳定或小于 1，说明同步开销和 memory access 可能抵消收益。",
        "- 该实验只评价 CUDA candidate 内部优化，不改变 OpenMP 作为主性能后端的定位。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.size": 11,
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        })
        labels = [f"{r['instance']}\n{r['algorithm'].split('-')[0]}" for r in rows]
        speedups = [float(r["parallel_speedup_vs_serial"]) for r in rows]
        fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=300)
        ax.bar(range(len(labels)), speedups, color="#d62728", alpha=0.82)
        ax.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("parallel / serial reversal speedup")
        ax.set_title("CUDA parallel reversal 相对 serial reversal")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".MISSING.md").write_text(f"无法生成图表：{exc}\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    rows = summarize(read_rows(input_path))
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    markdown = Path(args.markdown)
    if not markdown.is_absolute():
        markdown = ROOT / markdown
    figure = Path(args.figure)
    if not figure.is_absolute():
        figure = ROOT / figure
    write_csv(output, rows)
    write_markdown(markdown, rows)
    write_figure(figure, rows)
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    print(f"[ok] wrote {markdown.relative_to(ROOT)}")
    print(f"[ok] wrote {figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
