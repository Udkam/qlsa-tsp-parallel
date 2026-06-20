#!/usr/bin/env python3
"""Generate final report figures from existing CSV data.

The script never fabricates missing data. If an input CSV is unavailable, the
corresponding figure is skipped and recorded in figures/final/MISSING_FIGURES.md.
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

        plt.rcParams.update(
            {
                "font.sans-serif": [
                    "Microsoft YaHei",
                    "SimHei",
                    "Noto Sans CJK SC",
                    "Arial Unicode MS",
                    "DejaVu Sans",
                ],
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
    fig, ax = plt.subplots(figsize=(12.8, 3.4))
    ax.set_axis_off()
    labels = [
        "TSPLIB95 .tsp",
        "Parser",
        "DistanceMatrix",
        "SA / QLSA Core",
        "Serial / OpenMP / CUDA",
        "CSV Results",
        "Analysis & Report",
    ]
    colors = ["#e8f1ff", "#e8f1ff", "#e8f1ff", "#fff7ed", "#ecfdf5", "#f8fafc", "#f8fafc"]
    x0, y0, w, h, gap = 0.02, 0.38, 0.125, 0.32, 0.018
    for idx, label in enumerate(labels):
        x = x0 + idx * (w + gap)
        box = FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.012",
            linewidth=1.25,
            edgecolor="#111827",
            facecolor=colors[idx],
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=10)
        if idx < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.004, y0 + h / 2),
                    (x + w + gap - 0.004, y0 + h / 2),
                    arrowstyle="->",
                    mutation_scale=13,
                    linewidth=1.2,
                    color="#334155",
                )
            )
    ax.set_title("System Architecture and Data Flow", pad=10)
    save(fig, FIGURES / "fig01_architecture_pipeline.png")
    plt.close(fig)


def fig02_openmp_speedup(plt, rows: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    x = list(range(len(INSTANCES)))
    sa = [num(rows.get((inst, "sa")), "speedup") for inst in INSTANCES]
    qlsa = [num(rows.get((inst, "qlsa")), "speedup") for inst in INSTANCES]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], sa, width, label="SA", color=COLORS["sa"])
    bars2 = ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA", color=COLORS["qlsa"])
    ax.set_title("OpenMP Speedup across TSPLIB95 Instances")
    ax.set_ylabel("Speedup")
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
    ax.axhline(100, color="#111827", linestyle="--", linewidth=1.0, label="Ideal 100%")
    ax.set_title("OpenMP Parallel Efficiency")
    ax.set_ylabel("Efficiency (%)")
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
    bars1 = ax.bar([i - width / 2 for i in x], sa, width, label="SA default", color=COLORS["sa"])
    bars2 = ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA default", color=COLORS["qlsa"])
    for idx, inst in enumerate(INSTANCES):
        if inst in HARD_INSTANCES:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#fef3c7", alpha=0.23)
    ax.set_title("Default-Parameter Gap")
    ax.set_ylabel("Gap (%)")
    ax.set_xticks(x, INSTANCES)
    add_light_grid(ax)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1, "{:.3f}")
    add_bar_labels(ax, bars2, "{:.3f}")
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
    bars1 = ax.bar([i - width for i in x], default_gap, width, label="Default", color=COLORS["muted"])
    bars2 = ax.bar(x, tuned_gap, width, label="Tuned validation", color=COLORS["sa"])
    bars3 = ax.bar([i + width for i in x], targeted_gap, width, label="Targeted high-budget", color=COLORS["openmp"])
    ax.set_title("Gap Reduction after Tuning and Targeted Enhancement")
    ax.set_ylabel("Minimum Gap (%)")
    ax.set_xticks(x, labels)
    add_light_grid(ax)
    ax.legend(frameon=False)
    add_bar_labels(ax, bars1, "{:.3f}")
    add_bar_labels(ax, bars2, "{:.3f}")
    add_bar_labels(ax, bars3, "{:.3f}")
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
    ax.set_title("QLSA Policy Comparison")
    ax.set_ylabel("Mean Gap (%)")
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
    labels = ["SA serial", "SA OpenMP", "SA CUDA", "QLSA serial", "QLSA OpenMP", "QLSA CUDA"]
    algs = ["sa-multichain", "sa-omp", "sa-cuda", "qlsa-multichain", "qlsa-omp", "qlsa-cuda"]
    values = [num(row_map.get(alg), "elapsed_ms_mean") / 1000.0 for alg in algs]
    colors = [COLORS["muted"], COLORS["openmp"], COLORS["cuda"], "#cbd5e1", COLORS["qlsa"], COLORS["cuda"]]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_title("berlin52 Serial, OpenMP and CUDA Positioning")
    ax.set_ylabel("Elapsed time (s)")
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
        ("Our-SA-OpenMP", [num(rows.get((inst, "sa")), "omp_ms") / 1000.0 for inst in INSTANCES], COLORS["sa"]),
        ("Our-QLSA-OpenMP", [num(rows.get((inst, "qlsa")), "omp_ms") / 1000.0 for inst in INSTANCES], COLORS["qlsa"]),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    width = 0.18
    for idx, (label, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        ax.bar([i + offset for i in x], values, width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_title("Runtime Reference Comparison with the Paper")
    ax.set_ylabel("Elapsed time (s, log scale)")
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
        ("Paper-best QLSA", [paper_best[inst] for inst in HARD_INSTANCES], "#bdbdbd"),
        ("Our best SA", our_sa, COLORS["sa"]),
        ("Our best QLSA", our_qlsa, COLORS["qlsa"]),
    ]
    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    width = 0.19
    for idx, (label, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        bars = ax.bar([i + offset for i in x], values, width, label=label, color=color)
        add_bar_labels(ax, bars, "{:.2f}")
    ax.set_title("Hard-Instance Mean Gap Comparison")
    ax.set_ylabel("Mean Gap (%)")
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
    ax.plot(xs, xs, linestyle=":", color=COLORS["muted"], linewidth=1.3, label="Ideal")
    ax.set_title("Supplementary OpenMP Thread Scaling")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
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
