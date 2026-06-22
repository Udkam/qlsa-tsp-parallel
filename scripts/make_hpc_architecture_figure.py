#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the final MPI + OpenMP + CUDA hybrid architecture figure."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig14_hpc_hybrid_architecture.png"

COLORS = {
    "mpi": "#2ca02c",
    "openmp": "#1f77b4",
    "cuda": "#d62728",
    "cpu": "#ff7f0e",
    "paper": "#4e79a7",
    "line": "#374151",
    "light": "#f8fafc",
}


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams.update({
        "font.size": 11,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.unicode_minus": False,
    })
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        try:
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    return plt, FancyArrowPatch, FancyBboxPatch


def add_box(ax, patch_cls, x, y, w, h, text, color, fontsize=11):
    box = patch_cls(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=color,
        facecolor=COLORS["light"],
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#111827", linespacing=1.25)


def arrow(ax, arrow_cls, x1, y1, x2, y2, label=None):
    arr = arrow_cls((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                    linewidth=1.4, color=COLORS["line"])
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label,
                ha="center", va="bottom", fontsize=9, color="#374151")


def main() -> None:
    plt, Arrow, Box = configure_matplotlib()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "MPI + OpenMP + CUDA 混合 HPC 架构",
            ha="center", va="center", fontsize=16, weight="bold", color="#111827")

    add_box(ax, Box, 0.08, 0.76, 0.25, 0.10, "MPI Rank 0\n搜索链分片 A", COLORS["mpi"])
    add_box(ax, Box, 0.38, 0.76, 0.25, 0.10, "MPI Rank 1\n搜索链分片 B", COLORS["mpi"])
    add_box(ax, Box, 0.68, 0.76, 0.25, 0.10, "更多 Rank\n同构扩展", COLORS["mpi"])

    add_box(ax, Box, 0.08, 0.55, 0.25, 0.11, "OpenMP 线程\n私有 chain / RNG / Q table", COLORS["openmp"], 10)
    add_box(ax, Box, 0.38, 0.55, 0.25, 0.11, "OpenMP 线程\n私有 chain / RNG / Q table", COLORS["openmp"], 10)
    add_box(ax, Box, 0.68, 0.55, 0.25, 0.11, "OpenMP 线程\n私有 chain / RNG / Q table", COLORS["openmp"], 10)

    add_box(ax, Box, 0.13, 0.34, 0.22, 0.10, "CPU 后端\nSA / QLSA 2-opt", COLORS["cpu"])
    add_box(ax, Box, 0.39, 0.34, 0.22, 0.10, "CUDA 后端\n可选 GPU chains", COLORS["cuda"])
    add_box(ax, Box, 0.65, 0.34, 0.22, 0.10, "DistanceMatrix\nRank 内只读", COLORS["paper"])

    add_box(ax, Box, 0.19, 0.14, 0.24, 0.10, "Rank local best\n局部最优", COLORS["mpi"])
    add_box(ax, Box, 0.57, 0.14, 0.28, 0.10, "MPI reduction / gather\n全局最优 tour", COLORS["mpi"])

    for x in [0.205, 0.505, 0.805]:
        arrow(ax, Arrow, x, 0.76, x, 0.66)
    arrow(ax, Arrow, 0.205, 0.55, 0.23, 0.44)
    arrow(ax, Arrow, 0.505, 0.55, 0.50, 0.44)
    arrow(ax, Arrow, 0.805, 0.55, 0.76, 0.44)
    arrow(ax, Arrow, 0.24, 0.34, 0.31, 0.24)
    arrow(ax, Arrow, 0.50, 0.34, 0.31, 0.24)
    arrow(ax, Arrow, 0.76, 0.34, 0.71, 0.24)
    arrow(ax, Arrow, 0.43, 0.19, 0.57, 0.19, "MINLOC + 广播 + 汇总")

    ax.text(0.5, 0.04,
            "设计意图：分布式内存 Rank 切分，共享内存多链执行，可选 GPU 加速后端。",
            ha="center", va="center", fontsize=10, color="#4b5563")

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[figure] {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
