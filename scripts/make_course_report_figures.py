#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate data-oriented Chinese figures for the course report.

Unlike the earlier architecture sketches, these figures are all based on
experiment CSV files and are intended for direct use in the course report.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "sa": "#1f77b4",
    "qlsa": "#ff7f0e",
    "openmp": "#2ca02c",
    "cuda": "#d62728",
    "mpi": "#9467bd",
    "paper": "#4e79a7",
    "soft": "#fdae6b",
    "chain": "#2ca02c",
    "reference": "#4e79a7",
}


def configure_font() -> None:
    for path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300


def read_csv(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def val(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def finish(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def openmp_speedup() -> None:
    rows = read_csv("results/summary/step5_multi_cpu_summary.csv")
    instances = ["berlin52", "eil51", "st70", "eil76", "rat99", "eil101"]
    sa = [next(val(r, "speedup") for r in rows if r["instance"] == i and r["algorithm"] == "sa-omp") for i in instances]
    qlsa = [next(val(r, "speedup") for r in rows if r["instance"] == i and r["algorithm"] == "qlsa-omp") for i in instances]
    x = list(range(len(instances)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar([i - width / 2 for i in x], sa, width, label="SA", color=COLORS["sa"])
    ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA", color=COLORS["qlsa"])
    ax.axhline(1, color=COLORS["reference"], linewidth=0.8)
    ax.set_title("OpenMP 多链并行加速比")
    ax.set_ylabel("加速比")
    ax.set_xticks(x, instances)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend()
    finish(fig, "fig_course_01_openmp_speedup.png")


def openmp_efficiency() -> None:
    rows = read_csv("results/summary/step5_multi_cpu_summary.csv")
    instances = ["berlin52", "eil51", "st70", "eil76", "rat99", "eil101"]
    sa = [next(val(r, "parallel_efficiency_percent") for r in rows if r["instance"] == i and r["algorithm"] == "sa-omp") for i in instances]
    qlsa = [next(val(r, "parallel_efficiency_percent") for r in rows if r["instance"] == i and r["algorithm"] == "qlsa-omp") for i in instances]
    x = list(range(len(instances)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar([i - width / 2 for i in x], sa, width, label="SA", color=COLORS["sa"])
    ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA", color=COLORS["qlsa"])
    ax.axhline(100, color=COLORS["reference"], linestyle="--", linewidth=0.8, label="理想值")
    ax.set_title("OpenMP 多链并行效率")
    ax.set_ylabel("并行效率（%）")
    ax.set_xticks(x, instances)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend()
    finish(fig, "fig_course_02_openmp_efficiency.png")


def default_gap() -> None:
    rows = read_csv("results/summary/step5_multi_cpu_summary.csv")
    instances = ["berlin52", "eil51", "st70", "eil76", "rat99", "eil101"]
    sa = [next(val(r, "gap_percent") for r in rows if r["instance"] == i and r["algorithm"] == "sa-omp") for i in instances]
    qlsa = [next(val(r, "gap_percent") for r in rows if r["instance"] == i and r["algorithm"] == "qlsa-omp") for i in instances]
    x = list(range(len(instances)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar([i - width / 2 for i in x], sa, width, label="SA", color=COLORS["sa"])
    ax.bar([i + width / 2 for i in x], qlsa, width, label="QLSA", color=COLORS["qlsa"])
    ax.set_title("默认参数下相对最优偏差")
    ax.set_ylabel("Gap（%）")
    ax.set_xticks(x, instances)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend()
    finish(fig, "fig_course_03_default_gap.png")


def targeted_quality() -> None:
    rows = read_csv("results/summary/targeted_quality_summary.csv")
    wanted = [
        ("eil101", "sa", "2000000", "128"),
        ("eil101", "qlsa", "2000000", "128"),
        ("rat99", "sa", "2000000", "128"),
        ("rat99", "qlsa", "2000000", "128"),
    ]
    chosen = []
    for inst, family, iterations, chains in wanted:
        hit = [
            r for r in rows
            if r["instance"] == inst
            and r["family"] == family
            and r["iterations"] == iterations
            and r["chains"] == chains
        ]
        if hit:
            chosen.append(hit[0])
    labels = [f"{r['instance']}\n{r['family'].upper()}" for r in chosen]
    min_gap = [val(r, "gap_percent_min") for r in chosen]
    mean_gap = [val(r, "gap_percent_mean") for r in chosen]
    x = list(range(len(chosen)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar([i - width / 2 for i in x], min_gap, width, label="最小偏差", color="#6baed6")
    ax.bar([i + width / 2 for i in x], mean_gap, width, label="平均偏差", color="#fd8d3c")
    ax.set_title("定向增强后的解质量")
    ax.set_ylabel("Gap（%）")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend()
    finish(fig, "fig_course_04_targeted_quality.png")


def policy_comparison() -> None:
    rows = read_csv("results/summary/policy_comparison_summary.csv")
    instances = ["eil76", "rat99", "eil101"]
    eps = [next(val(r, "gap_mean_percent") for r in rows if r["instance"] == i and r["policy"] == "epsilon-greedy") for i in instances]
    soft = [next(val(r, "gap_mean_percent") for r in rows if r["instance"] == i and r["policy"] == "softmax") for i in instances]
    x = list(range(len(instances)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar([i - width / 2 for i in x], eps, width, label="epsilon-greedy", color=COLORS["qlsa"])
    ax.bar([i + width / 2 for i in x], soft, width, label="softmax", color=COLORS["soft"])
    ax.set_title("QLSA 动作选择策略对比")
    ax.set_ylabel("平均偏差（%）")
    ax.set_xticks(x, instances)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend()
    finish(fig, "fig_course_05_policy_comparison.png")


def cuda_boundary() -> None:
    rows = read_csv("results/summary/large_cuda_formal_summary.csv")
    instances = ["ch130", "a280", "lin318", "rat575"]
    chain_time, cand_time, chain_gap, cand_gap = [], [], [], []
    for inst in instances:
        chain = next(r for r in rows if r["instance"] == inst and r["cuda_mode"] == "chain")
        cand = next(r for r in rows if r["instance"] == inst and r["cuda_mode"] == "candidate")
        chain_time.append(val(chain, "elapsed_ms_mean") / 1000.0)
        cand_time.append(val(cand, "elapsed_ms_mean") / 1000.0)
        chain_gap.append(val(chain, "gap_min"))
        cand_gap.append(val(cand, "gap_min"))
    x = list(range(len(instances)))
    width = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    ax1.bar([i - width / 2 for i in x], chain_time, width, label="多链模式", color=COLORS["chain"])
    ax1.bar([i + width / 2 for i in x], cand_time, width, label="候选批量评价", color=COLORS["cuda"])
    ax1.set_title("CUDA 运行时间")
    ax1.set_ylabel("秒")
    ax1.set_xticks(x, instances)
    ax1.grid(axis="y", color="#d6e8ff")
    ax1.legend()
    ax2.bar([i - width / 2 for i in x], chain_gap, width, label="多链模式", color=COLORS["chain"])
    ax2.bar([i + width / 2 for i in x], cand_gap, width, label="候选批量评价", color=COLORS["cuda"])
    ax2.set_title("CUDA 最小偏差")
    ax2.set_ylabel("Gap（%）")
    ax2.set_xticks(x, instances)
    ax2.grid(axis="y", color="#d6e8ff")
    ax2.legend()
    finish(fig, "fig_course_06_cuda_boundary.png")


def mpi_scaling() -> None:
    rows = read_csv("results/summary/mpi_vm_scaling_formal_summary.csv")
    rows = [r for r in rows if r["np"] == "2"]
    labels = [f"{'SA' if r['algorithm'].startswith('sa') else 'QLSA'}\n{r['threads']}线程/进程" for r in rows]
    speed = [val(r, "speedup_vs_np1") for r in rows]
    comm = [val(r, "communication_ms_mean") for r in rows]
    x = list(range(len(rows)))
    fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
    ax1.bar(x, speed, color=COLORS["mpi"], alpha=0.85, label="加速比")
    ax1.axhline(2, color=COLORS["reference"], linestyle="--", linewidth=0.8, label="理想 2 倍")
    ax1.set_title("双虚拟机 MPI + OpenMP 扩展性实验")
    ax1.set_ylabel("np=2 相对 np=1 的加速比")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", color="#d6e8ff")
    ax2 = ax1.twinx()
    ax2.plot(x, comm, color=COLORS["sa"], marker="o", label="通信时间")
    ax2.set_ylabel("通信时间（毫秒）")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, loc="lower right")
    finish(fig, "fig_course_07_mpi_scaling.png")


def large_openmp() -> None:
    rows = read_csv("results/summary/large_openmp_l1_summary.csv")
    instances = ["ch130", "ch150", "d198", "a280"]
    sa_gap, qlsa_gap, sa_time, qlsa_time = [], [], [], []
    for inst in instances:
        sa = next(r for r in rows if r["instance"] == inst and r["algorithm"] == "sa")
        qlsa = next(r for r in rows if r["instance"] == inst and r["algorithm"] == "qlsa")
        sa_gap.append(val(sa, "gap_min"))
        qlsa_gap.append(val(qlsa, "gap_min"))
        sa_time.append(val(sa, "elapsed_ms_mean") / 1000.0)
        qlsa_time.append(val(qlsa, "elapsed_ms_mean") / 1000.0)
    x = list(range(len(instances)))
    width = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    ax1.bar([i - width / 2 for i in x], sa_gap, width, label="SA", color=COLORS["sa"])
    ax1.bar([i + width / 2 for i in x], qlsa_gap, width, label="QLSA", color=COLORS["qlsa"])
    ax1.set_title("L1 实例最小偏差")
    ax1.set_ylabel("Gap（%）")
    ax1.set_xticks(x, instances)
    ax1.grid(axis="y", color="#d6e8ff")
    ax1.legend()
    ax2.bar([i - width / 2 for i in x], sa_time, width, label="SA", color=COLORS["sa"])
    ax2.bar([i + width / 2 for i in x], qlsa_time, width, label="QLSA", color=COLORS["qlsa"])
    ax2.set_title("L1 实例运行时间")
    ax2.set_ylabel("秒")
    ax2.set_xticks(x, instances)
    ax2.grid(axis="y", color="#d6e8ff")
    ax2.legend()
    finish(fig, "fig_course_08_large_openmp.png")


def paper_quality() -> None:
    paper = read_csv("results/reference/paper_hard_instance_quality.csv")
    ours = read_csv("results/summary/targeted_quality_summary.csv")
    instances = ["eil76", "rat99", "eil101"]
    paper_sa, paper_best, our_best = [], [], []
    for inst in instances:
        ps = [r for r in paper if r["instance"] == inst and r["algorithm"] == "Paper-SA"]
        pq = [r for r in paper if r["instance"] == inst and r["algorithm"] != "Paper-SA"]
        os = [r for r in ours if r["instance"] == inst]
        paper_sa.append(val(ps[0], "gap_mean_percent") if ps else math.nan)
        paper_best.append(min(val(r, "gap_mean_percent") for r in pq) if pq else math.nan)
        our_best.append(min(val(r, "gap_percent_mean") for r in os) if os else math.nan)
    x = list(range(len(instances)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar([i - width for i in x], paper_sa, width, label="论文 SA", color="#4e79a7")
    ax.bar(x, paper_best, width, label="论文 QLSA 系列最好", color="#f28e2b")
    ax.bar([i + width for i in x], our_best, width, label="本项目调优后最好", color="#59a14f")
    ax.set_title("困难实例平均偏差参考对比")
    ax.set_ylabel("平均偏差（%）")
    ax.set_xticks(x, instances)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend()
    finish(fig, "fig_course_09_paper_quality.png")


def openmp_thread_scaling() -> None:
    rows = read_csv("results/summary/openmp_scaling_final_summary.csv")
    threads = [1, 2, 4, 8, 12, 16]
    series = [
        ("berlin52", "sa-omp", "berlin52 SA", COLORS["sa"], "o"),
        ("berlin52", "qlsa-omp", "berlin52 QLSA", COLORS["qlsa"], "o"),
        ("eil101", "sa-omp", "eil101 SA", "#6baed6", "s"),
        ("eil101", "qlsa-omp", "eil101 QLSA", "#fd8d3c", "s"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for inst, alg, label, color, marker in series:
        vals = []
        for th in threads:
            hit = [
                r for r in rows
                if r.get("instance") == inst
                and r.get("algorithm") == alg
                and r.get("chains") == "32"
                and r.get("threads") == str(th)
            ]
            vals.append(val(hit[0], "speedup") if hit else math.nan)
        ax.plot(threads, vals, marker=marker, linewidth=1.8, color=color, label=label)
    ax.plot(threads, threads, linestyle="--", linewidth=1.0, color=COLORS["reference"], label="理想线")
    ax.set_title("OpenMP 线程扩展曲线")
    ax.set_xlabel("线程数")
    ax.set_ylabel("加速比")
    ax.set_xticks(threads)
    ax.grid(axis="y", color="#d6e8ff")
    ax.legend(ncol=2)
    finish(fig, "fig_course_10_openmp_thread_scaling.png")


def main() -> None:
    configure_font()
    openmp_speedup()
    openmp_efficiency()
    default_gap()
    targeted_quality()
    policy_comparison()
    cuda_boundary()
    mpi_scaling()
    large_openmp()
    paper_quality()
    openmp_thread_scaling()
    print("generated course report figures")


if __name__ == "__main__":
    main()
