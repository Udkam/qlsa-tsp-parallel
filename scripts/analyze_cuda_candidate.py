#!/usr/bin/env python3
"""Analyze CUDA candidate-mode experiments and write summary/markdown/figure."""

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
    "eil51": 426,
    "st70": 675,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "pr76": 108159,
    "ch130": 6110,
    "ch150": 6528,
    "kroA150": 26524,
    "d198": 15780,
    "tsp225": 3916,
    "pr226": 80369,
    "gil262": 2378,
    "a280": 2579,
    "lin318": 42029,
    "rd400": 15281,
    "pcb442": 50778,
    "d493": 35002,
    "att532": 27686,
    "u574": 36905,
    "rat575": 6773,
    "d657": 48912,
    "u724": 41910,
    "rat783": 8806,
}
SUMMARY_HEADER = [
    "instance",
    "algorithm_family",
    "cuda_mode",
    "chains",
    "cuda_block_size",
    "cuda_candidates_per_iter",
    "cuda_candidate_policy",
    "iterations",
    "runs",
    "bks",
    "best_length_min",
    "gap_min",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "speedup_vs_cuda_chain",
    "speedup_vs_openmp_if_available",
    "accepted_moves_mean",
    "improved_moves_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "cuda_candidate_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "cuda_candidate_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "cuda_candidate_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig15_cuda_candidate_mode.png"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def load_openmp_baselines() -> dict[tuple[str, str], float]:
    baselines: dict[tuple[str, str], float] = {}
    for path in [ROOT / "results" / "summary" / "step5_multi_cpu_summary.csv",
                 ROOT / "results" / "final" / "final_key_results.csv"]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                instance = row.get("instance", "")
                algorithm = row.get("algorithm", "")
                family = row.get("family", "")
                parallel = row.get("parallel", "")
                elapsed = row.get("elapsed_ms_mean") or row.get("omp_ms")
                if not elapsed:
                    continue
                if parallel == "omp" or algorithm in {"sa-omp", "qlsa-omp"}:
                    fam = family or ("qlsa" if algorithm.startswith("qlsa") else "sa")
                    try:
                        baselines[(instance, fam)] = float(elapsed)
                    except ValueError:
                        pass
    return baselines


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        algorithm = row["algorithm"]
        family = "qlsa" if algorithm.startswith("qlsa") else "sa"
        candidate_policy = row.get("cuda_candidate_policy") or ("best" if row.get("cuda_mode") == "candidate" else "")
        key = (
            row["instance"],
            family,
            row["cuda_mode"],
            row["chains"],
            row["cuda_block_size"],
            row["cuda_candidates_per_iter"],
            candidate_policy,
            row["iterations"],
        )
        groups[key].append(row)

    openmp_baselines = load_openmp_baselines()
    interim: list[dict[str, object]] = []
    for key, items in groups.items():
        instance, family, mode, chains, block_size, candidates, candidate_policy, iterations = key
        lengths = [int(r["best_length"]) for r in items]
        elapsed = [float(r["elapsed_ms"]) for r in items]
        accepted = [float(r["accepted_moves"]) for r in items]
        improved = [float(r["improved_moves"]) for r in items]
        bks = BKS.get(instance)
        best_min = min(lengths)
        gap = math.nan if bks is None else (best_min - bks) / bks * 100.0
        elapsed_mean = mean(elapsed)
        openmp = openmp_baselines.get((instance, family))
        interim.append({
            "instance": instance,
            "algorithm_family": family,
            "cuda_mode": mode,
            "chains": int(chains),
            "cuda_block_size": int(block_size),
            "cuda_candidates_per_iter": int(candidates),
            "cuda_candidate_policy": candidate_policy,
            "iterations": int(iterations),
            "runs": len(items),
            "bks": "" if bks is None else bks,
            "best_length_min": best_min,
            "gap_min": gap,
            "elapsed_ms_mean": elapsed_mean,
            "elapsed_ms_std": std(elapsed),
            "speedup_vs_cuda_chain": math.nan,
            "speedup_vs_openmp_if_available": math.nan if not openmp else openmp / elapsed_mean,
            "accepted_moves_mean": mean(accepted),
            "improved_moves_mean": mean(improved),
        })

    chain_lookup = {
        (r["instance"], r["algorithm_family"], r["chains"], r["cuda_block_size"], r["iterations"]): r["elapsed_ms_mean"]
        for r in interim if r["cuda_mode"] == "chain"
    }
    for row in interim:
        baseline = chain_lookup.get((row["instance"], row["algorithm_family"], row["chains"],
                                     row["cuda_block_size"], row["iterations"]))
        if baseline and row["elapsed_ms_mean"]:
            row["speedup_vs_cuda_chain"] = baseline / row["elapsed_ms_mean"]

    output_rows: list[dict[str, str]] = []
    mode_order = {"chain": 0, "candidate": 1}
    for row in sorted(interim, key=lambda r: (r["instance"], r["algorithm_family"], r["cuda_block_size"], r["cuda_candidates_per_iter"], mode_order.get(r["cuda_mode"], 9))):
        out = {}
        for field in SUMMARY_HEADER:
            value = row[field]
            if isinstance(value, float):
                out[field] = "" if math.isnan(value) else f"{value:.4f}"
            else:
                out[field] = str(value)
        output_rows.append(out)
    return output_rows


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CUDA candidate-level evaluation 实验分析",
        "",
        "本实验比较 CUDA chain mode 与新增 candidate mode。candidate mode 使用 one block per chain、block 内线程并行评价多个 2-opt 候选 move，并在 shared memory 中做最小 delta 归约。该模式改变了单步 proposal：它是 batch proposal 变体，不等同于原始 SA 的单候选采样。",
        "",
        "| instance | family | mode | policy | chains | block | candidates | runs | best | gap | mean ms | speedup vs chain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instance']} | {r['algorithm_family']} | {r['cuda_mode']} | {r['cuda_candidate_policy']} | {r['chains']} | "
            f"{r['cuda_block_size']} | {r['cuda_candidates_per_iter']} | {r['runs']} | "
            f"{r['best_length_min']} | {r['gap_min']} | {r['elapsed_ms_mean']} | {r['speedup_vs_cuda_chain']} |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "- SA CUDA candidate mode 已验证为可运行路径，并保留 chain mode 作为默认兼容模式。",
        "- `best` policy 在每轮候选中选最小 delta，`random` policy 从候选批中按可复现随机方式选一个候选，`hybrid` policy 在 best/random 之间交替，用于比较批量择优、随机提案和组合策略。",
        "- 若 speedup_vs_cuda_chain 小于 1，说明 batch candidate evaluation 在该实例/预算下没有抵消 reduction、shared memory 同步和 thread 0 reversal 的开销。",
        "- SA/QLSA candidate mode 均已接入主线 CUDA 后端，但仍应标记为 batch proposal 变体。",
        "- 本实验不改变 OpenMP 作为主性能结论的定位。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sa_rows = [r for r in rows if r["algorithm_family"] == "sa"]
    if not sa_rows:
        path.with_suffix(".MISSING.md").write_text("缺少 SA CUDA candidate 数据，无法生成图。\n", encoding="utf-8")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        instances = sorted({r["instance"] for r in sa_rows})
        series = [
            ("链模式", "#2ca02c", lambda r: r["cuda_mode"] == "chain"),
            ("best 候选", "#d62728", lambda r: r["cuda_mode"] == "candidate" and r.get("cuda_candidate_policy") == "best"),
            ("random 候选", "#ff7f0e", lambda r: r["cuda_mode"] == "candidate" and r.get("cuda_candidate_policy") == "random"),
            ("hybrid 候选", "#9467bd", lambda r: r["cuda_mode"] == "candidate" and r.get("cuda_candidate_policy") == "hybrid"),
        ]
        x = list(range(len(instances)))
        width = 0.20
        fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=220)
        for offset, (label, color, pred) in enumerate(series):
            values = []
            for inst in instances:
                hit = [r for r in sa_rows if r["instance"] == inst and pred(r)]
                values.append(float(hit[0]["elapsed_ms_mean"]) if hit else math.nan)
            positions = [i + (offset - 1.5) * width for i in x]
            ax.bar(positions, values, width, color=color, label=label)
            for pos, value in zip(positions, values):
                if not math.isnan(value):
                    ax.text(pos, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_title("CUDA SA 候选策略用时对比", fontsize=12)
        ax.set_ylabel("平均用时 (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(instances)
        ax.grid(axis="y", color="#d6e8ff", linewidth=0.6)
        ax.legend(frameon=False, ncol=4)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - fallback for missing matplotlib
        svg = ["<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420'>",
               "<rect width='100%' height='100%' fill='white'/>",
               "<text x='30' y='30' font-size='18'>CUDA chain 与 candidate mode 时间对比</text>"]
        max_v = max(float(r["elapsed_ms_mean"]) for r in sa_rows)
        for idx, r in enumerate(sa_rows):
            value = float(r["elapsed_ms_mean"])
            x = 40 + idx * 80
            h = 300 * value / max_v if max_v else 0
            y = 360 - h
            if r["cuda_mode"] == "chain":
                color = "#2ca02c"
            elif r.get("cuda_candidate_policy") == "random":
                color = "#ff7f0e"
            elif r.get("cuda_candidate_policy") == "hybrid":
                color = "#9467bd"
            else:
                color = "#d62728"
            svg.append(f"<rect x='{x}' y='{y:.1f}' width='45' height='{h:.1f}' fill='{color}'/>")
            svg.append(f"<text x='{x}' y='385' font-size='10'>{r['instance']}</text>")
            svg.append(f"<text x='{x}' y='400' font-size='10'>{r['cuda_mode']}</text>")
        svg.append(f"<text x='30' y='410' font-size='10'>matplotlib 不可用，SVG fallback: {exc}</text>")
        svg.append("</svg>")
        path.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    rows = read_rows(input_path)
    summary = summarize(rows)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    markdown = Path(args.markdown)
    if not markdown.is_absolute():
        markdown = ROOT / markdown
    figure = Path(args.figure)
    if not figure.is_absolute():
        figure = ROOT / figure

    write_summary(output, summary)
    write_markdown(markdown, summary)
    write_figure(figure, summary)
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    print(f"[ok] wrote {markdown.relative_to(ROOT)}")
    print(f"[ok] wrote {figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
