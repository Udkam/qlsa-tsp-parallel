#!/usr/bin/env python3
"""Generate figures and compact comparison tables for the final report.

The script intentionally depends only on the Python standard library plus
matplotlib. If matplotlib is unavailable, it writes simple SVG placeholder
figures instead of failing silently.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INSTANCES = ["berlin52", "eil51", "st70", "eil76", "rat99", "eil101"]
HARD_INSTANCES = ["eil76", "rat99", "eil101"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def write_svg_placeholder(path: Path, title: str, lines: Iterable[str]) -> None:
    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;")
    body = []
    y = 70
    for line in lines:
        text = line.replace("&", "&amp;").replace("<", "&lt;")
        body.append(f'<text x="40" y="{y}" font-size="22">{text}</text>')
        y += 36
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720">
  <rect width="100%" height="100%" fill="white"/>
  <text x="40" y="40" font-size="30" font-weight="bold">{escaped_title}</text>
  {''.join(body)}
</svg>
"""
    path.with_suffix(".svg").write_text(svg, encoding="utf-8")


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        return plt, FancyArrowPatch, FancyBboxPatch
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[warning] matplotlib unavailable: {exc}")
        return None, None, None


def default_rows() -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = read_csv(RESULTS / "final_key_results.csv")
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        if row["category"] == "default_openmp_speedup":
            out[(row["instance"], row["family"])] = row
    return out


def final_key_rows(category: str) -> List[Dict[str, str]]:
    return [r for r in read_csv(RESULTS / "final_key_results.csv") if r["category"] == category]


def best_quality_row(instance: str, family: str) -> Optional[Dict[str, str]]:
    candidates = [
        r
        for r in final_key_rows("targeted_high_budget_quality")
        if r["instance"] == instance and r["family"] == family and r["variant"] == "best-quality"
    ]
    if not candidates:
        candidates = [
            r
            for r in final_key_rows("tuned_validation_quality")
            if r["instance"] == instance and r["family"] == family
        ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (f(r, "gap_min_percent"), f(r, "gap_mean_percent"), f(r, "elapsed_ms_mean")))


def tuned_row(instance: str, family: str) -> Optional[Dict[str, str]]:
    rows = [
        r
        for r in final_key_rows("tuned_validation_quality")
        if r["instance"] == instance and r["family"] == family
    ]
    if not rows:
        return None
    return min(rows, key=lambda r: (f(r, "gap_min_percent"), f(r, "gap_mean_percent")))


def targeted_row(instance: str, family: str) -> Optional[Dict[str, str]]:
    rows = [
        r
        for r in final_key_rows("targeted_high_budget_quality")
        if r["instance"] == instance and r["family"] == family and r["variant"] == "best-quality"
    ]
    if not rows:
        return None
    return min(rows, key=lambda r: (f(r, "gap_min_percent"), f(r, "gap_mean_percent")))


def add_bar_labels(ax, bars, fmt="{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isfinite(height):
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"[figure] {path.relative_to(ROOT)}")


def fig_architecture(plt, FancyArrowPatch, FancyBboxPatch) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_axis_off()
    labels = [
        "TSPLIB95 .tsp",
        "Parser\n解析器",
        "DistanceMatrix\n一维连续距离矩阵",
        "SA / QLSA Core\n2-opt + Q-table",
        "Serial / OpenMP / CUDA\n多后端执行",
        "CSV Results\n实验结果",
        "Analysis / Report\n统计与报告",
    ]
    x0, y0, w, h, gap = 0.02, 0.38, 0.125, 0.32, 0.018
    for idx, label in enumerate(labels):
        x = x0 + idx * (w + gap)
        box = FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.012",
            linewidth=1.4,
            edgecolor="#1f2937",
            facecolor="#e8f1ff" if idx < 4 else "#e9f7ef",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=10)
        if idx < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.004, y0 + h / 2),
                    (x + w + gap - 0.004, y0 + h / 2),
                    arrowstyle="->",
                    mutation_scale=14,
                    linewidth=1.2,
                    color="#334155",
                )
            )
    ax.set_title("图 1：项目整体架构与实验流水线", fontsize=14, pad=12)
    save(fig, FIGURES / "fig_architecture_pipeline.png")
    plt.close(fig)


def fig_openmp_speedup(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [f(rows[(i, "sa")], "speedup") for i in INSTANCES]
    qlsa = [f(rows[(i, "qlsa")], "speedup") for i in INSTANCES]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.36
    bars1 = ax.bar([v - width / 2 for v in x], sa, width, label="SA OpenMP", color="#2563eb")
    bars2 = ax.bar([v + width / 2 for v in x], qlsa, width, label="QLSA OpenMP", color="#f97316")
    ax.set_title("图 2：默认参数多实例 OpenMP speedup")
    ax.set_ylabel("speedup (T_serial / T_OpenMP)")
    ax.set_xticks(x, INSTANCES)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    save(fig, FIGURES / "fig_openmp_speedup.png")
    plt.close(fig)


def fig_openmp_efficiency(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [f(rows[(i, "sa")], "efficiency_percent") for i in INSTANCES]
    qlsa = [f(rows[(i, "qlsa")], "efficiency_percent") for i in INSTANCES]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.36
    bars1 = ax.bar([v - width / 2 for v in x], sa, width, label="SA", color="#2563eb")
    bars2 = ax.bar([v + width / 2 for v in x], qlsa, width, label="QLSA", color="#f97316")
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1, label="ideal 100%")
    ax.set_title("图 3：默认参数 OpenMP parallel efficiency")
    ax.set_ylabel("parallel efficiency (%)")
    ax.set_xticks(x, INSTANCES)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    save(fig, FIGURES / "fig_openmp_efficiency.png")
    plt.close(fig)


def fig_default_gap(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [f(rows[(i, "sa")], "gap_percent") for i in INSTANCES]
    qlsa = [f(rows[(i, "qlsa")], "gap_percent") for i in INSTANCES]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.36
    bars1 = ax.bar([v - width / 2 for v in x], sa, width, label="SA default", color="#2563eb")
    bars2 = ax.bar([v + width / 2 for v in x], qlsa, width, label="QLSA default", color="#f97316")
    for idx, inst in enumerate(INSTANCES):
        if inst in HARD_INSTANCES:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#fef3c7", alpha=0.23)
    ax.set_title("图 4：默认参数 best length Gap 对比")
    ax.set_ylabel("Gap (%)")
    ax.set_xticks(x, INSTANCES)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    add_bar_labels(ax, bars1, "{:.3f}")
    add_bar_labels(ax, bars2, "{:.3f}")
    save(fig, FIGURES / "fig_default_gap.png")
    plt.close(fig)


def fig_tuned_quality(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    labels = []
    default_gap = []
    tuned_gap = []
    targeted_gap = []
    for inst in HARD_INSTANCES:
        for fam in ["sa", "qlsa"]:
            labels.append(f"{inst}\n{fam.upper()}")
            default_gap.append(f(rows[(inst, fam)], "gap_percent"))
            trow = tuned_row(inst, fam)
            tuned_gap.append(f(trow, "gap_min_percent") if trow else float("nan"))
            grow = targeted_row(inst, fam)
            targeted_gap.append(f(grow, "gap_min_percent") if grow else float("nan"))

    x = list(range(len(labels)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bars1 = ax.bar([v - width for v in x], default_gap, width, label="Step 5B default", color="#94a3b8")
    bars2 = ax.bar(x, tuned_gap, width, label="Step 6B tuned min", color="#2563eb")
    bars3 = ax.bar([v + width for v in x], targeted_gap, width, label="Step 6C targeted min", color="#16a34a")
    ax.set_title("图 5：调参和定向增强后的 Gap 改善")
    ax.set_ylabel("Gap (%)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    add_bar_labels(ax, bars1, "{:.3f}")
    add_bar_labels(ax, bars2, "{:.3f}")
    add_bar_labels(ax, bars3, "{:.3f}")
    save(fig, FIGURES / "fig_tuned_quality_improvement.png")
    plt.close(fig)


def fig_paper_runtime(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    paper = {r["instance"]: r for r in read_csv(RESULTS / "paper_table8_runtime.csv")}
    x = list(range(len(INSTANCES)))
    series = [
        ("Paper-SA", [f(paper[i], "paper_sa_s") for i in INSTANCES], "#64748b"),
        ("Paper-QLSA-epsilon", [f(paper[i], "paper_qlsa_epsilon_s") for i in INSTANCES], "#f97316"),
        ("Our-SA-OpenMP", [f(rows[(i, "sa")], "omp_ms") / 1000.0 for i in INSTANCES], "#2563eb"),
        ("Our-QLSA-OpenMP", [f(rows[(i, "qlsa")], "omp_ms") / 1000.0 for i in INSTANCES], "#16a34a"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    width = 0.18
    for idx, (name, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        ax.bar([v + offset for v in x], values, width, label=name, color=color)
    ax.set_yscale("log")
    ax.set_title("图 6：论文 Table 8 与本项目 OpenMP elapsed time 参考对比")
    ax.set_ylabel("elapsed time (s, log scale)")
    ax.set_xticks(x, INSTANCES)
    ax.grid(axis="y", linestyle="--", alpha=0.35, which="both")
    ax.legend(ncols=2)
    save(fig, FIGURES / "fig_paper_runtime_comparison_log.png")
    plt.close(fig)


def fig_paper_quality(plt) -> None:
    paper_rows = read_csv(RESULTS / "paper_hard_instance_quality.csv")
    paper_sa = {}
    paper_best_qlsa = {}
    for inst in HARD_INSTANCES:
        rows = [r for r in paper_rows if r["instance"] == inst]
        paper_sa[inst] = next(f(r, "gap_mean_percent") for r in rows if r["algorithm"] == "Paper-SA")
        qlsa_rows = [r for r in rows if r["algorithm"] != "Paper-SA"]
        best = min(qlsa_rows, key=lambda r: f(r, "gap_mean_percent"))
        paper_best_qlsa[inst] = f(best, "gap_mean_percent")

    our_sa = []
    our_qlsa = []
    for inst in HARD_INSTANCES:
        sa_row = best_quality_row(inst, "sa")
        qlsa_row = best_quality_row(inst, "qlsa")
        our_sa.append(f(sa_row, "gap_mean_percent") if sa_row else 0.0)
        our_qlsa.append(f(qlsa_row, "gap_mean_percent") if qlsa_row else 0.0)

    x = list(range(len(HARD_INSTANCES)))
    series = [
        ("Paper-SA", [paper_sa[i] for i in HARD_INSTANCES], "#64748b"),
        ("Paper-best QLSA family", [paper_best_qlsa[i] for i in HARD_INSTANCES], "#f97316"),
        ("Our best SA", our_sa, "#2563eb"),
        ("Our best QLSA", our_qlsa, "#16a34a"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    width = 0.19
    for idx, (name, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        bars = ax.bar([v + offset for v in x], values, width, label=name, color=color)
        add_bar_labels(ax, bars, "{:.2f}")
    ax.set_title("图 7：harder instances mean Gap 与论文质量结果对比")
    ax.set_ylabel("Mean Gap (%)")
    ax.set_xticks(x, HARD_INSTANCES)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncols=2)
    save(fig, FIGURES / "fig_paper_quality_hard_instances.png")
    plt.close(fig)


def fig_cuda_positioning(plt) -> None:
    rows = read_csv(RESULTS / "step5_berlin52_summary.csv")
    row_map = {r["algorithm"]: r for r in rows}
    labels = [
        "SA serial\nmulti-chain",
        "SA OpenMP",
        "SA CUDA",
        "QLSA serial\nmulti-chain",
        "QLSA OpenMP",
        "QLSA CUDA",
    ]
    algs = ["sa-multichain", "sa-omp", "sa-cuda", "qlsa-multichain", "qlsa-omp", "qlsa-cuda"]
    values = [f(row_map[a], "elapsed_ms_mean") / 1000.0 for a in algs]
    colors = ["#94a3b8", "#2563eb", "#7c3aed", "#cbd5e1", "#f97316", "#dc2626"]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_title("图 8：berlin52 上 Serial / OpenMP / CUDA elapsed time 对比")
    ax.set_ylabel("elapsed time (s)")
    ax.set_xticks(range(len(labels)), labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    add_bar_labels(ax, bars, "{:.2f}")
    save(fig, FIGURES / "fig_cuda_positioning.png")
    plt.close(fig)


def write_report_comparison_summary(rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    paper_runtime = {r["instance"]: r for r in read_csv(RESULTS / "paper_table8_runtime.csv")}
    paper_quality = read_csv(RESULTS / "paper_hard_instance_quality.csv")
    out_path = RESULTS / "report_comparison_summary.csv"
    fields = [
        "category",
        "instance",
        "metric",
        "paper_sa",
        "paper_qlsa_reference",
        "our_sa",
        "our_qlsa",
        "note",
    ]
    out = []
    for inst in INSTANCES:
        out.append(
            {
                "category": "runtime_reference",
                "instance": inst,
                "metric": "elapsed_time_s",
                "paper_sa": f"{f(paper_runtime[inst], 'paper_sa_s'):.3f}",
                "paper_qlsa_reference": f"{f(paper_runtime[inst], 'paper_qlsa_epsilon_s'):.3f}",
                "our_sa": f"{f(rows[(inst, 'sa')], 'omp_ms') / 1000.0:.3f}",
                "our_qlsa": f"{f(rows[(inst, 'qlsa')], 'omp_ms') / 1000.0:.3f}",
                "note": "Reference only: language and hardware differ from the paper.",
            }
        )
    for inst in HARD_INSTANCES:
        inst_rows = [r for r in paper_quality if r["instance"] == inst]
        paper_sa_gap = next(f(r, "gap_mean_percent") for r in inst_rows if r["algorithm"] == "Paper-SA")
        paper_best = min([r for r in inst_rows if r["algorithm"] != "Paper-SA"], key=lambda r: f(r, "gap_mean_percent"))
        sa_row = best_quality_row(inst, "sa")
        qlsa_row = best_quality_row(inst, "qlsa")
        out.append(
            {
                "category": "quality_reference",
                "instance": inst,
                "metric": "mean_gap_percent",
                "paper_sa": f"{paper_sa_gap:.4f}",
                "paper_qlsa_reference": f"{f(paper_best, 'gap_mean_percent'):.4f}",
                "our_sa": f"{f(sa_row, 'gap_mean_percent'):.4f}" if sa_row else "",
                "our_qlsa": f"{f(qlsa_row, 'gap_mean_percent'):.4f}" if qlsa_row else "",
                "note": "Our value uses tuned validation or targeted high-budget best-quality row.",
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f"[csv] {out_path.relative_to(ROOT)}")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = default_rows()
    plt, FancyArrowPatch, FancyBboxPatch = try_import_matplotlib()
    write_report_comparison_summary(rows)

    if plt is None:  # pragma: no cover - depends on environment
        placeholders = {
            "fig_architecture_pipeline": "Architecture pipeline",
            "fig_openmp_speedup": "OpenMP speedup",
            "fig_openmp_efficiency": "OpenMP efficiency",
            "fig_default_gap": "Default Gap",
            "fig_tuned_quality_improvement": "Tuned quality improvement",
            "fig_paper_runtime_comparison_log": "Paper runtime comparison",
            "fig_paper_quality_hard_instances": "Paper quality comparison",
            "fig_cuda_positioning": "CUDA positioning",
        }
        for stem, title in placeholders.items():
            write_svg_placeholder(FIGURES / f"{stem}.svg", title, ["matplotlib is unavailable", "See CSV files under results/"])
        return 0

    fig_architecture(plt, FancyArrowPatch, FancyBboxPatch)
    fig_openmp_speedup(plt, rows)
    fig_openmp_efficiency(plt, rows)
    fig_default_gap(plt, rows)
    fig_tuned_quality(plt, rows)
    fig_paper_runtime(plt, rows)
    fig_paper_quality(plt)
    fig_cuda_positioning(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
