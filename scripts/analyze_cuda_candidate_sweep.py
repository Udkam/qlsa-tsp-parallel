#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze CUDA candidate parameter sweep results."""

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
    "cuda_candidate_policy",
    "chains",
    "cuda_block_size",
    "cuda_candidates_per_iter",
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
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "cuda_candidate_sweep_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "cuda_candidate_sweep_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "cuda_candidate_sweep_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig23_cuda_candidate_sweep_tradeoff.png"))
    return parser.parse_args()


def fmean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def fstd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("cuda_mode") != "candidate":
            continue
        key = (
            row["instance"],
            row["dimension"],
            row["algorithm"],
            row.get("cuda_mode", "candidate"),
            row.get("cuda_reversal_mode", "serial"),
            row.get("cuda_candidate_policy") or "best",
            row["chains"],
            row["cuda_block_size"],
            row["cuda_candidates_per_iter"],
            row["iterations"],
        )
        groups[key].append(row)

    out: list[dict[str, str]] = []
    for key, items in groups.items():
        instance, dimension, algorithm, mode, reversal, candidate_policy, chains, block_size, candidates, iterations = key
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
            "cuda_candidate_policy": candidate_policy,
            "chains": chains,
            "cuda_block_size": block_size,
            "cuda_candidates_per_iter": candidates,
            "iterations": iterations,
            "runs": str(len(items)),
            "bks": "" if bks is None else str(bks),
            "best_length_min": str(best),
            "gap_min_percent": "" if math.isnan(gap) else f"{gap:.4f}",
            "elapsed_ms_mean": f"{fmean(elapsed):.4f}",
            "elapsed_ms_std": f"{fstd(elapsed):.4f}",
            "accepted_moves_mean": f"{fmean(accepted):.4f}",
            "improved_moves_mean": f"{fmean(improved):.4f}",
        })
    return sorted(
        out,
        key=lambda r: (
            r["instance"],
            r["algorithm"],
            r["cuda_reversal_mode"],
            r["cuda_candidate_policy"],
            int(r["cuda_block_size"]),
            int(r["cuda_candidates_per_iter"]),
        ),
    )


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_instance_algorithm: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_instance_algorithm[(row["instance"], row["algorithm"])].append(row)
    chosen = []
    for _, items in sorted(by_instance_algorithm.items()):
        chosen.append(min(items, key=lambda r: (float(r["gap_min_percent"] or "inf"), float(r["elapsed_ms_mean"]))))
    return chosen


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CUDA candidate 参数扫描分析",
        "",
        "本实验比较 CUDA candidate-level mode 的 block size、candidate 数量、algorithm 和 reversal mode。该模式属于 batch proposal 变体，不等同于默认单候选 SA/QLSA，也不替代 OpenMP 主性能结论。",
        "",
        "## 最佳质量配置",
        "",
        "| instance | algorithm | reversal | policy | block | candidates | best | Gap (%) | mean ms | runs |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows(rows):
        lines.append(
            f"| {row['instance']} | {row['algorithm']} | {row['cuda_reversal_mode']} | {row['cuda_candidate_policy']} | "
            f"{row['cuda_block_size']} | {row['cuda_candidates_per_iter']} | "
            f"{row['best_length_min']} | {row['gap_min_percent']} | {row['elapsed_ms_mean']} | {row['runs']} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- candidate 数量增加通常会提高每轮搜索覆盖率，但也增加 block 内同步、shared memory reduction 和 reversal 开销。",
        "- best policy 更偏向批量择优，random policy 更接近单候选随机提案语义，hybrid policy 在二者之间交替，可用于判断质量提升是否主要来自批量择优。",
        "- parallel reversal 是显式 opt-in 模式，只有在同一实例和同一预算下实际更快时，才可写为优化收益。",
        "- 当前实验用于质量-时间折中分析，不能写成 CUDA 是主性能后端。",
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

        chosen = best_rows(rows)
        labels = [f"{r['instance']}\n{r['algorithm'].split('-')[0]}\n{r['cuda_reversal_mode']}\n{r['cuda_candidate_policy']}" for r in chosen]
        times = [float(r["elapsed_ms_mean"]) for r in chosen]
        gaps = [float(r["gap_min_percent"] or "nan") for r in chosen]

        fig, ax1 = plt.subplots(figsize=(8.6, 4.6), dpi=300)
        x = list(range(len(labels)))
        ax1.bar(x, times, color="#d62728", alpha=0.82, label="平均时间")
        ax1.set_ylabel("平均时间 (ms)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.grid(axis="y", color="#d6e8ff", linewidth=0.6)

        ax2 = ax1.twinx()
        ax2.plot(x, gaps, color="#1f77b4", marker="o", linewidth=2.0, label="最小 Gap")
        ax2.set_ylabel("最小 Gap (%)")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False)
        ax1.set_title("CUDA candidate 参数扫描最佳质量配置")
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

    write_summary(output, rows)
    write_markdown(markdown, rows)
    write_figure(figure, rows)
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    print(f"[ok] wrote {markdown.relative_to(ROOT)}")
    print(f"[ok] wrote {figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
