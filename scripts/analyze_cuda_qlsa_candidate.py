#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze SA/QLSA CUDA candidate and reversal-mode experiments."""

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
    "dimension",
    "algorithm",
    "cuda_mode",
    "cuda_reversal_mode",
    "chains",
    "block_size",
    "candidates_per_iter",
    "iterations",
    "runs",
    "bks",
    "best_length_min",
    "gap_min_percent",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "accepted_moves_mean",
    "improved_moves_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "cuda_qlsa_candidate_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "cuda_qlsa_candidate_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "cuda_qlsa_candidate_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig21_cuda_qlsa_candidate.png"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["instance"],
            row["dimension"],
            row["algorithm"],
            row.get("cuda_mode", ""),
            row.get("cuda_reversal_mode", "serial"),
            row["chains"],
            row.get("cuda_block_size", row.get("threads", "")),
            row.get("cuda_candidates_per_iter", ""),
            row["iterations"],
        )
        groups[key].append(row)

    out: list[dict[str, str]] = []
    for key, items in groups.items():
        instance, dimension, algorithm, mode, reversal, chains, block_size, candidates, iterations = key
        lengths = [int(r["best_length"]) for r in items]
        elapsed = [float(r["elapsed_ms"]) for r in items]
        accepted = [float(r["accepted_moves"]) for r in items]
        improved = [float(r["improved_moves"]) for r in items]
        bks = BKS.get(instance)
        best = min(lengths)
        gap = math.nan if bks is None else (best - bks) / bks * 100.0
        out.append({
            "instance": instance,
            "dimension": dimension,
            "algorithm": algorithm,
            "cuda_mode": mode,
            "cuda_reversal_mode": reversal,
            "chains": chains,
            "block_size": block_size,
            "candidates_per_iter": candidates,
            "iterations": iterations,
            "runs": str(len(items)),
            "bks": "" if bks is None else str(bks),
            "best_length_min": str(best),
            "gap_min_percent": "" if math.isnan(gap) else f"{gap:.4f}",
            "elapsed_ms_mean": f"{mean(elapsed):.4f}",
            "elapsed_ms_std": f"{stdev(elapsed):.4f}",
            "accepted_moves_mean": f"{mean(accepted):.4f}",
            "improved_moves_mean": f"{mean(improved):.4f}",
        })
    return sorted(out, key=lambda r: (r["instance"], r["algorithm"], r["cuda_mode"], r["cuda_reversal_mode"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CUDA QLSA candidate 分析",
        "",
        "本实验验证 `--qlsa --parallel cuda --cuda_mode candidate` 已接入。该模式由 thread 0 维护 Q table、状态和 action selection，block 内线程批量评价 action 约束下的 2-opt 候选 move，因此属于 batch proposal QLSA 变体，不等同论文完整 SB-QLSA。",
        "",
        "| instance | algorithm | mode | reversal | best | Gap (%) | mean ms | runs |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['algorithm']} | {row['cuda_mode']} | {row['cuda_reversal_mode']} | "
            f"{row['best_length_min']} | {row['gap_min_percent']} | {row['elapsed_ms_mean']} | {row['runs']} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 该实验只证明 QLSA candidate path 可构建、可运行并返回合法 tour。",
        "- 是否改善质量或时间必须按同实例、同预算的 CSV 结果解释。",
        "- 不能据此写成 CUDA 是主性能后端，也不能写成完整 SB-QLSA 复现。",
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
        labels = [f"{r['algorithm']}\n{r['cuda_mode']}\n{r['cuda_reversal_mode']}" for r in rows]
        times = [float(r["elapsed_ms_mean"]) for r in rows]
        gaps = [float(r["gap_min_percent"] or "nan") for r in rows]
        x = list(range(len(rows)))
        fig, ax1 = plt.subplots(figsize=(8.6, 4.6), dpi=300)
        ax1.bar(x, times, color="#d62728", alpha=0.82, label="平均时间")
        ax1.set_ylabel("平均时间 (ms)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha="right")
        ax1.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax2 = ax1.twinx()
        ax2.plot(x, gaps, color="#ff7f0e", marker="o", linewidth=2.0, label="最小 Gap")
        ax2.set_ylabel("最小 Gap (%)")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False)
        ax1.set_title("CUDA QLSA candidate quick 对比")
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
