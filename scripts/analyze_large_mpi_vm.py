#!/usr/bin/env python3
"""Analyze large-instance MPI + OpenMP VM runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
SUMMARY_HEADER = [
    "instance",
    "tier",
    "algorithm",
    "np",
    "threads_per_rank",
    "chains",
    "iterations",
    "repeat",
    "bks",
    "best_length_min",
    "gap_min",
    "elapsed_ms_mean",
    "speedup_vs_np1",
    "efficiency",
    "communication_ms_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results" / "raw" / "large_mpi_vm_raw.csv"))
    parser.add_argument("--output", default=str(ROOT / "results" / "summary" / "large_mpi_vm_summary.csv"))
    parser.add_argument("--markdown", default=str(ROOT / "docs" / "dev" / "large_mpi_vm_analysis.md"))
    parser.add_argument("--figure", default=str(ROOT / "figures" / "final" / "fig18_large_mpi_vm_scaling.png"))
    return parser.parse_args()


def load_config() -> tuple[dict[str, str], dict[str, int]]:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    tier_by_instance: dict[str, str] = {}
    for tier, names in data.get("tiers", {}).items():
        for name in names:
            tier_by_instance[name] = tier
    bks = {k: int(v) for k, v in data.get("bks", {}).items()}
    return tier_by_instance, bks


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def family(algorithm: str) -> str:
    return "qlsa" if algorithm.startswith("qlsa") else "sa"


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    tier_by_instance, bks = load_config()
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok" and r.get("best_length")]
    skipped = [r for r in rows if r.get("status", "ok") != "ok"]

    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok_rows:
        instance = row["instance"]
        tier = row.get("tier") or tier_by_instance.get(instance, "")
        key = (
            instance,
            tier,
            family(row["algorithm"]),
            row.get("np", ""),
            row.get("threads_per_rank", row.get("threads", "")),
            row["chains"],
            row["iterations"],
        )
        groups[key].append(row)

    np1_times: dict[tuple[str, str, str, str, str, str], float] = {}
    for key, group in groups.items():
        instance, tier, alg, np_value, threads, chains, iterations = key
        if np_value == "1":
            np1_times[(instance, tier, alg, threads, chains, iterations)] = mean(
                [float(r["elapsed_ms"]) for r in group]
            )

    summary: list[dict[str, str]] = []
    for key, group in sorted(groups.items()):
        instance, tier, alg, np_value, threads, chains, iterations = key
        bks_value = bks.get(instance, 0)
        bests = [float(r["best_length"]) for r in group]
        elapsed_mean = mean([float(r["elapsed_ms"]) for r in group])
        baseline = np1_times.get((instance, tier, alg, threads, chains, iterations), 0.0)
        speedup = baseline / elapsed_mean if baseline > 0 and elapsed_mean > 0 else (1.0 if np_value == "1" else 0.0)
        efficiency = speedup / float(np_value) if np_value and float(np_value) > 0 else 0.0
        comm_values = [float(r["communication_ms"]) for r in group if r.get("communication_ms")]
        gap_min = ((min(bests) - bks_value) / bks_value * 100.0) if bks_value else 0.0
        summary.append(
            {
                "instance": instance,
                "tier": tier,
                "algorithm": alg,
                "np": np_value,
                "threads_per_rank": threads,
                "chains": chains,
                "iterations": iterations,
                "repeat": str(len(group)),
                "bks": str(bks_value) if bks_value else "",
                "best_length_min": f"{min(bests):.0f}",
                "gap_min": f"{gap_min:.4f}",
                "elapsed_ms_mean": f"{elapsed_mean:.4f}",
                "speedup_vs_np1": f"{speedup:.4f}",
                "efficiency": f"{efficiency:.4f}",
                "communication_ms_mean": f"{mean(comm_values):.4f}" if comm_values else "",
            }
        )
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
        path.with_suffix(".svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'><text x='20' y='40'>matplotlib unavailable</text></svg>",
            encoding="utf-8",
        )
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.5), dpi=300)
    if not summary:
        ax.axis("off")
        ax.text(0.5, 0.5, "未获得大实例 MPI 数据", ha="center", va="center", fontsize=12)
        fig.suptitle("大实例 MPI + OpenMP 状态")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    rows = [r for r in summary if r["np"] != "1"]
    labels = [f"{r['instance']}\n{r['algorithm']} t{r['threads_per_rank']}" for r in rows]
    values = [float(r["speedup_vs_np1"]) for r in rows]
    colors = ["#2ca02c" if r["algorithm"] == "sa" else "#ff7f0e" for r in rows]

    ax.bar(labels, values, color=colors, width=0.72)
    ax.axhline(2.0, color="#1f77b4", linestyle="--", linewidth=1.2, label="理想 np=2")
    ax.set_title("大实例 MPI + OpenMP 加速比")
    ax.set_ylabel("相对 np=1 的 speedup")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(
    path: Path,
    raw_path: Path,
    summary: list[dict[str, str]],
    skipped: list[dict[str, str]],
    figure: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel_figure = figure.relative_to(ROOT)
    except ValueError:
        rel_figure = figure
    lines = [
        "# 大实例 MPI + OpenMP VM 实验分析",
        "",
        f"- 输入 raw CSV：`{raw_path}`",
        f"- 输出图：`{rel_figure}`",
        "",
        "本实验在两台 Ubuntu VM 上通过真实 `mpirun` 执行。`np=1` 作为单 VM MPI baseline，`np=2` 作为双 VM 分布式运行。该结果用于证明 rank-level chain decomposition 可以跨分布式内存运行；由于环境是 VMware NAT 和虚拟化 CPU，不能解释为生产 HPC 集群 benchmark。",
        "",
    ]
    if not summary:
        lines.extend(["本次没有获得 `status=ok` 的大实例 MPI 数据。", ""])
    else:
        lines.extend(
            [
                "| instance | algorithm | np | threads/rank | chains | best | min Gap | mean ms | speedup | efficiency | comm ms |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary:
            lines.append(
                f"| {row['instance']} | {row['algorithm']} | {row['np']} | {row['threads_per_rank']} | {row['chains']} | "
                f"{row['best_length_min']} | {row['gap_min']} | {row['elapsed_ms_mean']} | "
                f"{row['speedup_vs_np1']} | {row['efficiency']} | {row['communication_ms_mean']} |"
            )
    if skipped:
        lines.extend(["", "## 跳过或异常记录", "", "| instance | status | error |", "|---|---|---|"])
        for row in skipped:
            lines.append(f"| {row.get('instance','')} | {row.get('status','')} | {row.get('error','')} |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 公开报告中只使用 VM1/VM2 代称，不暴露 IP、用户名、密码或私钥路径。",
            "- speedup 以相同 instance、algorithm、threads/rank、chains 和 iterations 的 np=1 均值为基准。",
            "- communication_ms 来自程序输出，仅用于说明最终归约通信开销量级。",
            "- 该实验支持“分布式 MPI + OpenMP 工程链路可运行”的结论，不支持“生产 HPC 性能上限”的结论。",
        ]
    )
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

    rows = read_csv(raw_path)
    summary, skipped = summarize(rows)
    write_summary(output, summary)
    make_figure(summary, figure)
    write_markdown(markdown, raw_path, summary, skipped, figure)
    print(f"[ok] wrote {output}")
    print(f"[ok] wrote {markdown}")
    print(f"[ok] wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
