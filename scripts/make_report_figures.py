#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate final report figures from existing CSV data.

The script never fabricates missing data. If an input CSV is unavailable, the
corresponding figure is skipped and recorded in figures/final/MISSING_FIGURES.md.

Figure text is rendered in Chinese where possible; English is kept only for
proper nouns, algorithm/library names and metric abbreviations (SA, QLSA,
OpenMP, CUDA, TSPLIB95, BKS, Gap, Speedup, Efficiency, CSV, Parser, 2-opt,
Softmax, epsilon-greedy, Serial, CPU, GPU, Paper, Python, C++20). A CJK font is
auto-detected; if none is available the script still renders but records the
problem in figures/final/MISSING_FONTS.md.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FINAL = ROOT / "results" / "final"
RESULTS_SUMMARY = ROOT / "results" / "summary"
RESULTS_REFERENCE = ROOT / "results" / "reference"
FIGURES = ROOT / "figures" / "final"

INSTANCES = ["berlin52", "eil51", "st70", "eil76", "rat99", "eil101"]
HARD_INSTANCES = ["eil76", "rat99", "eil101"]

COLORS = {
    "sa": "#1f77b4",
    "qlsa": "#ff7f0e",
    "openmp": "#2ca02c",
    "cuda": "#d62728",
    "paper": "#7f7f7f",
    "muted": "#9ca3af",
    "light": "#e5e7eb",
}

MISSING: list[str] = []

# CJK font fallback candidates, ordered by platform preference.
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",       # Windows
    "SimHei",                # Windows
    "SimSun",                # Windows
    "PingFang SC",           # macOS
    "Heiti SC",              # macOS
    "Noto Sans CJK SC",      # Linux
    "WenQuanYi Micro Hei",   # Linux
]


def detect_cjk_fonts() -> List[str]:
    """Return the available CJK fonts from the candidate list, in order."""
    try:
        import matplotlib.font_manager as fm

        available = {f.name for f in fm.fontManager.ttflist}
    except Exception as exc:  # pragma: no cover
        MISSING.append(f"font detection failed: {exc}")
        return []
    return [name for name in CJK_FONT_CANDIDATES if name in available]


def write_font_note(found: List[str]) -> None:
    """Record a clear note if no CJK font is available; clean it up otherwise."""
    note = FIGURES / "MISSING_FONTS.md"
    if found:
        if note.exists():
            note.unlink()
        return
    message = (
        "# Missing CJK Fonts\n\n"
        "No Chinese-capable font was found among the candidates:\n\n"
        + "\n".join(f"- {name}" for name in CJK_FONT_CANDIDATES)
        + "\n\nFigures were still rendered, but Chinese labels may show as "
        "missing-glyph boxes. Install one of the fonts above and re-run "
        "`scripts/make_report_figures.py`.\n"
    )
    note.write_text(message, encoding="utf-8")
    print("[warning] no CJK font found; Chinese figure text may not render. "
          f"See {note.relative_to(ROOT)}")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        MISSING.append(f"`{path.relative_to(ROOT)}` is missing.")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fobj:
        return list(csv.DictReader(fobj))


def num(row: Optional[Dict[str, str]], key: str, default: float = 0.0) -> float:
    if not row:
        return default
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[figure] {path.relative_to(ROOT)}")


def configure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        found = detect_cjk_fonts()
        write_font_note(found)
        if found:
            print(f"[font] using CJK font: {found[0]}")
        plt.rcParams.update(
            {
                # Detected CJK font first, then a Latin fallback for ASCII glyphs.
                "font.sans-serif": found + ["DejaVu Sans"],
                "axes.unicode_minus": False,
                "font.size": 11,
                "axes.titlesize": 11,
                "axes.labelsize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#111827",
                "axes.linewidth": 0.8,
            }
        )
        return plt, FancyArrowPatch, FancyBboxPatch
    except Exception as exc:  # pragma: no cover
        MISSING.append(f"matplotlib is unavailable: {exc}")
        return None, None, None


def add_light_grid(ax) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.22, color="#6b7280")
    ax.set_axisbelow(True)


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
                fontsize=9,
            )


def default_openmp_rows() -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = read_csv(RESULTS_FINAL / "final_key_results.csv")
    return {
        (row["instance"], row["family"]): row
        for row in rows
        if row.get("category") == "default_openmp_speedup"
    }


def final_rows(category: str) -> List[Dict[str, str]]:
    return [row for row in read_csv(RESULTS_FINAL / "final_key_results.csv") if row.get("category") == category]


def best_quality_row(instance: str, family: str) -> Optional[Dict[str, str]]:
    candidates = [
        row
        for row in final_rows("targeted_high_budget_quality")
        if row.get("instance") == instance and row.get("family") == family and row.get("variant") == "best-quality"
    ]
    if not candidates:
        candidates = [
            row
            for row in final_rows("tuned_validation_quality")
            if row.get("instance") == instance and row.get("family") == family
        ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (num(row, "gap_min_percent"), num(row, "gap_mean_percent"), num(row, "elapsed_ms_mean")))


def exact_row(category: str, instance: str, family: str, variant: str | None = None) -> Optional[Dict[str, str]]:
    rows = [
        row
        for row in final_rows(category)
        if row.get("instance") == instance and row.get("family") == family
    ]
    if variant is not None:
        rows = [row for row in rows if row.get("variant") == variant]
    if not rows:
        return None
    return min(rows, key=lambda row: (num(row, "gap_min_percent"), num(row, "gap_mean_percent"), num(row, "elapsed_ms_mean")))


def fig01_architecture(plt, FancyArrowPatch, FancyBboxPatch) -> None:
    """Paper-style vertical architecture and data-flow diagram.

    Renders a top-down pipeline with rounded boxes, a unified palette and
    uniform arrows. Outputs both PNG (referenced by the report) and SVG.
    """
    fig_w, fig_h = 6.6, 10.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.set_axis_off()

    cx = fig_w / 2.0
    text_color = "#111827"
    arrow_color = "#475569"

    # Unified palette: layer -> (facecolor, edgecolor).
    pal = {
        "input": ("#eef2f7", "#64748b"),   # blue-gray input layer
        "core": ("#dbeafe", "#3b82f6"),    # light blue core data layer
        "algo": ("#ffedd5", "#f97316"),    # light orange algorithm layer
        "serial": ("#e5e7eb", "#6b7280"),  # gray serial backend
        "openmp": ("#dcfce7", "#22c55e"),  # green OpenMP backend
        "cuda": ("#fee2e2", "#ef4444"),    # red CUDA backend
        "output": ("#ede9fe", "#8b5cf6"),  # light purple output layer
        "band": ("#f8fafc", "#cbd5e1"),    # backend container band
    }

    def box(x, y, w, h, kind, edge_lw=1.2):
        fc, ec = pal[kind]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.09",
                linewidth=edge_lw,
                edgecolor=ec,
                facecolor=fc,
                mutation_aspect=1.0,
            )
        )

    def label(x, y, s, size=10.5, color=text_color):
        ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color)

    def arrow(y_from, y_to, x=None):
        xx = cx if x is None else x
        ax.add_patch(
            FancyArrowPatch(
                (xx, y_from),
                (xx, y_to),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=arrow_color,
                shrinkA=0,
                shrinkB=0,
            )
        )

    gap = 0.33
    single_h, layered_h, band_h = 0.62, 0.98, 1.15
    single_w, wide_w, band_w = 2.8, 5.4, 5.8
    y = fig_h - 0.35

    def single(text, kind):
        nonlocal y
        box(cx - single_w / 2, y - single_h, single_w, single_h, kind)
        label(cx, y - single_h / 2, text)
        y -= single_h

    def gap_arrow():
        nonlocal y
        arrow(y, y - gap)
        y -= gap

    def layered(title, cells, kind):
        nonlocal y
        box(cx - wide_w / 2, y - layered_h, wide_w, layered_h, kind)
        label(cx, y - 0.24, title, size=9.0, color="#475569")
        band = wide_w - 0.5
        x0 = cx - band / 2
        cw = band / len(cells)
        cy = y - layered_h + 0.36
        _, ec = pal[kind]
        for i, c in enumerate(cells):
            label(x0 + cw * (i + 0.5), cy, c, size=9.5)
            if i > 0:
                xdiv = x0 + cw * i
                ax.plot([xdiv, xdiv], [y - layered_h + 0.16, cy + 0.18], color=ec, lw=0.8, alpha=0.7)
        y -= layered_h

    def backend_band():
        nonlocal y
        box(cx - band_w / 2, y - band_h, band_w, band_h, "band", edge_lw=1.0)
        label(cx, y - 0.22, "并行后端", size=9.0, color="#475569")
        inner_w, inner_gap, inner_h = 1.6, 0.3, 0.6
        group_w = inner_w * 3 + inner_gap * 2
        gx0 = cx - group_w / 2
        iy = y - band_h + 0.18
        for i, (t, k) in enumerate([("Serial 串行", "serial"), ("OpenMP 多链", "openmp"), ("CUDA 后端", "cuda")]):
            ix = gx0 + i * (inner_w + inner_gap)
            box(ix, iy, inner_w, inner_h, k, edge_lw=1.1)
            label(ix + inner_w / 2, iy + inner_h / 2, t, size=9.5)
        y -= band_h

    single("TSPLIB95 .tsp 文件", "input")
    gap_arrow()
    single("解析器（Parser）", "input")
    gap_arrow()
    single("实例（Instance）", "input")
    gap_arrow()
    layered("核心数据层", ["距离矩阵", "路径（Tour）", "RNG"], "core")
    gap_arrow()
    layered("搜索核心", ["SA", "QLSA", "2-opt 增量"], "algo")
    gap_arrow()
    backend_band()
    gap_arrow()
    single("CSV 结果", "output")
    gap_arrow()
    single("分析脚本", "output")
    gap_arrow()
    single("图表 + 报告", "output")

    save(fig, FIGURES / "fig01_architecture_pipeline.png")
    svg_path = FIGURES / "fig01_architecture_pipeline.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"[figure] {svg_path.relative_to(ROOT)}")
    plt.close(fig)


def fig02_openmp_speedup(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [num(rows.get((inst, "sa")), "speedup") for inst in INSTANCES]
    qlsa = [num(rows.get((inst, "qlsa")), "speedup") for inst in INSTANCES]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], sa, width, label="SA", color=COLORS["sa"])
    bars2 = ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA", color=COLORS["qlsa"])
    ax.set_title("OpenMP 多实例加速比")
    ax.set_ylabel("加速比（Speedup）")
    ax.set_xticks(x, INSTANCES)
    add_light_grid(ax)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    save(fig, FIGURES / "fig02_openmp_speedup.png")
    plt.close(fig)


def fig03_openmp_efficiency(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [num(rows.get((inst, "sa")), "efficiency_percent") for inst in INSTANCES]
    qlsa = [num(rows.get((inst, "qlsa")), "efficiency_percent") for inst in INSTANCES]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], sa, width, label="SA", color=COLORS["sa"])
    bars2 = ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA", color=COLORS["qlsa"])
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1.0, label="理想 100%")
    ax.set_title("OpenMP 多实例并行效率")
    ax.set_ylabel("并行效率 Efficiency（%）")
    ax.set_xticks(x, INSTANCES)
    ax.set_ylim(0, 110)
    add_light_grid(ax)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    save(fig, FIGURES / "fig03_openmp_efficiency.png")
    plt.close(fig)


def fig04_default_gap(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [num(rows.get((inst, "sa")), "gap_percent") for inst in INSTANCES]
    qlsa = [num(rows.get((inst, "qlsa")), "gap_percent") for inst in INSTANCES]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], sa, width, label="SA 默认", color=COLORS["sa"])
    bars2 = ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA 默认", color=COLORS["qlsa"])
    for idx, inst in enumerate(INSTANCES):
        if inst in HARD_INSTANCES:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#fef3c7", alpha=0.23)
    ax.set_title("默认参数 Gap 对比")
    ax.set_ylabel("Gap（%）")
    ax.set_xticks(x, INSTANCES)
    add_light_grid(ax)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1, "{:.2f}")
    add_bar_labels(ax, bars2, "{:.2f}")
    save(fig, FIGURES / "fig04_default_gap.png")
    plt.close(fig)


def fig05_tuning_curve(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    labels: list[str] = []
    default_gap: list[float] = []
    tuned_gap: list[float] = []
    targeted_gap: list[float] = []
    for inst in HARD_INSTANCES:
        for family in ["sa", "qlsa"]:
            labels.append(f"{inst}\n{family.upper()}")
            default_gap.append(num(rows.get((inst, family)), "gap_percent"))
            tuned_gap.append(num(exact_row("tuned_validation_quality", inst, family), "gap_min_percent", float("nan")))
            targeted_gap.append(num(exact_row("targeted_high_budget_quality", inst, family, "best-quality"), "gap_min_percent", float("nan")))

    x = list(range(len(labels)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    bars1 = ax.bar([i - width for i in x], default_gap, width, label="默认参数", color=COLORS["muted"])
    bars2 = ax.bar(x, tuned_gap, width, label="调参验证", color=COLORS["sa"])
    bars3 = ax.bar([i + width for i in x], targeted_gap, width, label="定向高预算", color=COLORS["openmp"])
    ax.set_title("调参与定向增强后的 Gap 改善")
    ax.set_ylabel("最小 Gap（%）")
    ax.set_xticks(x, labels)
    add_light_grid(ax)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1, "{:.2f}")
    add_bar_labels(ax, bars2, "{:.2f}")
    add_bar_labels(ax, bars3, "{:.2f}")
    save(fig, FIGURES / "fig05_tuning_curve.png")
    plt.close(fig)


def fig06_policy_comparison(plt) -> None:
    rows = read_csv(RESULTS_SUMMARY / "policy_comparison_summary.csv")
    if not rows:
        return
    policies = ["epsilon-greedy", "softmax"]
    x = list(range(len(HARD_INSTANCES)))
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    width = 0.36
    for idx, policy in enumerate(policies):
        values = [
            num(next((row for row in rows if row["instance"] == inst and row["policy"] == policy), None), "gap_mean_percent")
            for inst in HARD_INSTANCES
        ]
        offset = (idx - 0.5) * width
        bars = ax.bar([i + offset for i in x], values, width, label=policy, color=COLORS["qlsa"] if idx == 0 else "#ffbb78")
        add_bar_labels(ax, bars, "{:.2f}")
    ax.set_title("QLSA 策略对比")
    ax.set_ylabel("平均 Gap（%）")
    ax.set_xticks(x, HARD_INSTANCES)
    add_light_grid(ax)
    ax.legend(frameon=False)
    save(fig, FIGURES / "fig06_policy_comparison.png")
    plt.close(fig)


def fig07_cuda_positioning(plt) -> None:
    rows = read_csv(RESULTS_SUMMARY / "step5_berlin52_summary.csv")
    if not rows:
        return
    row_map = {row["algorithm"]: row for row in rows}
    labels = ["SA 串行", "SA OpenMP", "SA CUDA", "QLSA 串行", "QLSA OpenMP", "QLSA CUDA"]
    algs = ["sa-multichain", "sa-omp", "sa-cuda", "qlsa-multichain", "qlsa-omp", "qlsa-cuda"]
    values = [num(row_map.get(alg), "elapsed_ms_mean") / 1000.0 for alg in algs]
    colors = [COLORS["muted"], COLORS["openmp"], COLORS["cuda"], "#cbd5e1", COLORS["qlsa"], COLORS["cuda"]]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_title("CUDA 在 berlin52 上的定位实验")
    ax.set_ylabel("运行时间（秒）")
    ax.set_xticks(range(len(labels)), labels, rotation=12)
    add_light_grid(ax)
    add_bar_labels(ax, bars, "{:.2f}")
    save(fig, FIGURES / "fig07_cuda_positioning.png")
    plt.close(fig)


def fig08_paper_runtime(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    paper = {row["instance"]: row for row in read_csv(RESULTS_REFERENCE / "paper_table8_runtime.csv")}
    if not paper:
        return
    x = list(range(len(INSTANCES)))
    series = [
        ("Paper-SA", [num(paper.get(inst), "paper_sa_s") for inst in INSTANCES], COLORS["paper"]),
        ("Paper-QLSA-epsilon", [num(paper.get(inst), "paper_qlsa_epsilon_s") for inst in INSTANCES], "#bdbdbd"),
        ("本文 SA-OpenMP", [num(rows.get((inst, "sa")), "omp_ms") / 1000.0 for inst in INSTANCES], COLORS["sa"]),
        ("本文 QLSA-OpenMP", [num(rows.get((inst, "qlsa")), "omp_ms") / 1000.0 for inst in INSTANCES], COLORS["qlsa"]),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    width = 0.18
    for idx, (label, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        ax.bar([i + offset for i in x], values, width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_title("与论文运行时间参考对比")
    ax.set_ylabel("运行时间（秒，对数轴）")
    ax.set_xticks(x, INSTANCES)
    add_light_grid(ax)
    ax.legend(frameon=False, ncols=2)
    save(fig, FIGURES / "fig08_paper_runtime_comparison.png")
    plt.close(fig)


def fig09_paper_quality(plt) -> None:
    paper_rows = read_csv(RESULTS_REFERENCE / "paper_hard_instance_quality.csv")
    if not paper_rows:
        return
    paper_sa: dict[str, float] = {}
    paper_best: dict[str, float] = {}
    for inst in HARD_INSTANCES:
        rows = [row for row in paper_rows if row["instance"] == inst]
        paper_sa[inst] = num(next((row for row in rows if row["algorithm"] == "Paper-SA"), None), "gap_mean_percent")
        qlsa_rows = [row for row in rows if row["algorithm"] != "Paper-SA"]
        paper_best[inst] = num(min(qlsa_rows, key=lambda row: num(row, "gap_mean_percent")), "gap_mean_percent") if qlsa_rows else 0.0

    our_sa = [num(best_quality_row(inst, "sa"), "gap_mean_percent") for inst in HARD_INSTANCES]
    our_qlsa = [num(best_quality_row(inst, "qlsa"), "gap_mean_percent") for inst in HARD_INSTANCES]
    x = list(range(len(HARD_INSTANCES)))
    series = [
        ("Paper-SA", [paper_sa[inst] for inst in HARD_INSTANCES], COLORS["paper"]),
        ("Paper 最优 QLSA", [paper_best[inst] for inst in HARD_INSTANCES], "#bdbdbd"),
        ("本文最优 SA", our_sa, COLORS["sa"]),
        ("本文最优 QLSA", our_qlsa, COLORS["qlsa"]),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    width = 0.19
    for idx, (label, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        bars = ax.bar([i + offset for i in x], values, width, label=label, color=color)
        add_bar_labels(ax, bars, "{:.2f}")
    ax.set_title("困难实例平均 Gap 对比")
    ax.set_ylabel("平均 Gap（%）")
    ax.set_xticks(x, HARD_INSTANCES)
    add_light_grid(ax)
    ax.legend(frameon=False, ncols=2)
    save(fig, FIGURES / "fig09_paper_quality_comparison.png")
    plt.close(fig)


def fig10_openmp_scaling(plt) -> None:
    rows = read_csv(RESULTS_SUMMARY / "openmp_scaling_final_summary.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    for inst, style in [("berlin52", "-"), ("eil101", "--")]:
        for alg, color in [("sa-omp", COLORS["sa"]), ("qlsa-omp", COLORS["qlsa"])]:
            pts = [
                row
                for row in rows
                if row["instance"] == inst and row["algorithm"] == alg and int(row["chains"]) == 32
            ]
            pts.sort(key=lambda row: int(row["threads"]))
            xs = [int(row["threads"]) for row in pts]
            ys = [num(row, "speedup") for row in pts]
            label = f"{inst} {alg.replace('-omp', '').upper()}"
            ax.plot(xs, ys, linestyle=style, marker="o", linewidth=1.8, color=color, label=label)
    xs = [1, 2, 4, 8, 12, 16]
    ax.plot(xs, xs, linestyle=":", color=COLORS["muted"], linewidth=1.3, label="理想")
    ax.set_title("OpenMP 线程扩展性（补充）")
    ax.set_xlabel("线程数")
    ax.set_ylabel("加速比（Speedup）")
    add_light_grid(ax)
    ax.legend(frameon=False, ncols=2)
    save(fig, FIGURES / "fig10_openmp_scaling_threads.png")
    plt.close(fig)


def write_comparison_summary(rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    paper_runtime_rows = read_csv(RESULTS_REFERENCE / "paper_table8_runtime.csv")
    paper_quality_rows = read_csv(RESULTS_REFERENCE / "paper_hard_instance_quality.csv")
    if not paper_runtime_rows or not paper_quality_rows:
        return
    paper_runtime = {row["instance"]: row for row in paper_runtime_rows}
    output = RESULTS_FINAL / "report_comparison_summary.csv"
    fields = ["category", "instance", "metric", "paper_sa", "paper_qlsa_reference", "our_sa", "our_qlsa", "note"]
    out: list[dict[str, str]] = []
    for inst in INSTANCES:
        out.append(
            {
                "category": "runtime_reference",
                "instance": inst,
                "metric": "elapsed_time_s",
                "paper_sa": f"{num(paper_runtime.get(inst), 'paper_sa_s'):.3f}",
                "paper_qlsa_reference": f"{num(paper_runtime.get(inst), 'paper_qlsa_epsilon_s'):.3f}",
                "our_sa": f"{num(rows.get((inst, 'sa')), 'omp_ms') / 1000.0:.3f}",
                "our_qlsa": f"{num(rows.get((inst, 'qlsa')), 'omp_ms') / 1000.0:.3f}",
                "note": "Reference only; hardware, language and implementation differ.",
            }
        )
    for inst in HARD_INSTANCES:
        inst_rows = [row for row in paper_quality_rows if row["instance"] == inst]
        paper_sa = next((row for row in inst_rows if row["algorithm"] == "Paper-SA"), None)
        paper_qlsa = min([row for row in inst_rows if row["algorithm"] != "Paper-SA"], key=lambda row: num(row, "gap_mean_percent"))
        out.append(
            {
                "category": "quality_reference",
                "instance": inst,
                "metric": "mean_gap_percent",
                "paper_sa": f"{num(paper_sa, 'gap_mean_percent'):.4f}",
                "paper_qlsa_reference": f"{num(paper_qlsa, 'gap_mean_percent'):.4f}",
                "our_sa": f"{num(best_quality_row(inst, 'sa'), 'gap_mean_percent'):.4f}",
                "our_qlsa": f"{num(best_quality_row(inst, 'qlsa'), 'gap_mean_percent'):.4f}",
                "note": "Our value uses tuned validation or targeted high-budget best-quality rows.",
            }
        )
    with output.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    print(f"[csv] {output.relative_to(ROOT)}")


def write_missing_note() -> None:
    note = FIGURES / "MISSING_FIGURES.md"
    if MISSING:
        note.write_text("# Missing Figures or Data\n\n" + "\n".join(f"- {item}" for item in MISSING) + "\n", encoding="utf-8")
        print(f"[warning] wrote {note.relative_to(ROOT)}")
    elif note.exists():
        note.unlink()


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt, FancyArrowPatch, FancyBboxPatch = configure_matplotlib()
    rows = default_openmp_rows()
    write_comparison_summary(rows)

    if plt is None:
        write_missing_note()
        return 0

    if rows:
        fig01_architecture(plt, FancyArrowPatch, FancyBboxPatch)
        fig02_openmp_speedup(plt, rows)
        fig03_openmp_efficiency(plt, rows)
        fig04_default_gap(plt, rows)
        fig05_tuning_curve(plt, rows)
        fig08_paper_runtime(plt, rows)
    else:
        MISSING.append("`results/final/final_key_results.csv` has no default OpenMP rows.")

    fig06_policy_comparison(plt)
    fig07_cuda_positioning(plt)
    fig09_paper_quality(plt)
    fig10_openmp_scaling(plt)
    write_missing_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
