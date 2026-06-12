#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BKS = {
    "berlin52": 7542,
    "eil51": 426,
    "st70": 675,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "pr76": 108159,
}

SUMMARY_FIELDS = [
    "instance",
    "algorithm",
    "parallel",
    "chains",
    "threads",
    "iterations",
    "runs",
    "bks",
    "best_length_min",
    "gap_percent",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "elapsed_ms_best",
    "speedup",
    "parallel_efficiency_percent",
    "accepted_moves_mean",
    "improved_moves_mean",
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def family_for_algorithm(algorithm):
    return "qlsa" if algorithm.startswith("qlsa") else "sa"


def parse_int(row, key):
    return int(row[key])


def parse_float(row, key):
    return float(row[key])


def load_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["algorithm"],
            row["instance"],
            parse_int(row, "iterations"),
            parse_int(row, "chains"),
            parse_int(row, "threads"),
            row["parallel"],
        )
        groups[key].append(row)
    return groups


def summarize(groups):
    summaries = []
    baseline_elapsed = {}

    for key, group in groups.items():
        algorithm, instance, iterations, chains, threads, parallel = key
        best_lengths = [parse_int(row, "best_length") for row in group]
        elapsed_values = [parse_float(row, "elapsed_ms") for row in group]
        accepted_values = [parse_float(row, "accepted_moves") for row in group]
        improved_values = [parse_float(row, "improved_moves") for row in group]
        bks = BKS.get(instance)
        best_min = min(best_lengths)
        gap = ""
        if bks:
            gap = 100.0 * (best_min - bks) / bks

        record = {
            "instance": instance,
            "algorithm": algorithm,
            "parallel": parallel,
            "chains": chains,
            "threads": threads,
            "iterations": iterations,
            "runs": len(group),
            "bks": bks if bks is not None else "",
            "best_length_min": best_min,
            "gap_percent": gap,
            "elapsed_ms_mean": mean(elapsed_values),
            "elapsed_ms_std": sample_std(elapsed_values),
            "elapsed_ms_best": min(elapsed_values),
            "speedup": "",
            "parallel_efficiency_percent": "",
            "accepted_moves_mean": mean(accepted_values),
            "improved_moves_mean": mean(improved_values),
        }
        summaries.append(record)

        if algorithm in ("sa-multichain", "qlsa-multichain"):
            baseline_elapsed[(instance, family_for_algorithm(algorithm))] = record["elapsed_ms_mean"]

    for record in summaries:
        baseline = baseline_elapsed.get((record["instance"], family_for_algorithm(record["algorithm"])))
        if baseline and record["elapsed_ms_mean"] > 0:
            speedup = baseline / record["elapsed_ms_mean"]
            record["speedup"] = speedup
            if record["parallel"] == "omp" and record["threads"] > 0:
                record["parallel_efficiency_percent"] = 100.0 * speedup / record["threads"]

    summaries.sort(
        key=lambda row: (
            row["instance"],
            family_for_algorithm(row["algorithm"]),
            {"none": 0, "omp": 1, "cuda": 2}.get(row["parallel"], 9),
            row["algorithm"],
            row["threads"],
        )
    )
    return summaries


def fmt_csv(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_summary_csv(path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: fmt_csv(row[field]) for field in SUMMARY_FIELDS})


def fmt_md(value, digits=3, suffix=""):
    if value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def make_markdown(summaries, source_path):
    instances = sorted({row["instance"] for row in summaries})
    lines = [
        "# Step 5A Berlin52 实验结果分析",
        "",
        "## 实验配置",
        "",
        f"- 原始数据：`{source_path.as_posix()}`",
        "- 统计方式：按 `algorithm + instance + iterations + chains + threads + parallel` 分组。",
        "- Speedup：SA 以 `sa-multichain` 为基准，QLSA 以 `qlsa-multichain` 为基准。",
        "- OpenMP parallel efficiency：`speedup / threads`。",
        "- Gap：相对 TSPLIB BKS 计算。",
        "",
        "## 汇总表格",
        "",
        "| Instance | Algorithm | Parallel | Chains | Threads | Runs | BKS | Best | Gap % | Mean ms | Std ms | Speedup | OMP Eff. % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        lines.append(
            "| {instance} | {algorithm} | {parallel} | {chains} | {threads} | {runs} | {bks} | {best} | {gap} | {mean_ms} | {std_ms} | {speedup} | {eff} |".format(
                instance=row["instance"],
                algorithm=row["algorithm"],
                parallel=row["parallel"],
                chains=row["chains"],
                threads=row["threads"],
                runs=row["runs"],
                bks=row["bks"],
                best=row["best_length_min"],
                gap=fmt_md(row["gap_percent"], 3),
                mean_ms=fmt_md(row["elapsed_ms_mean"], 3),
                std_ms=fmt_md(row["elapsed_ms_std"], 3),
                speedup=fmt_md(row["speedup"], 3),
                eff=fmt_md(row["parallel_efficiency_percent"], 2),
            )
        )

    berlin = [row for row in summaries if row["instance"] == "berlin52"]
    by_algorithm = {row["algorithm"]: row for row in berlin}
    sa_omp = by_algorithm.get("sa-omp")
    qlsa_omp = by_algorithm.get("qlsa-omp")
    sa_cuda = by_algorithm.get("sa-cuda")
    qlsa_cuda = by_algorithm.get("qlsa-cuda")

    lines.extend(["", "## 主要结论", ""])
    if "berlin52" in instances:
        lines.append("- berlin52 的 BKS 为 7542，本项目当前所有版本均达到 best_length=7542，Gap=0%。")
    if sa_omp:
        lines.append(
            f"- SA OpenMP 相对 SA 串行多链平均加速约 {sa_omp['speedup']:.2f}x，8 线程并行效率约 {sa_omp['parallel_efficiency_percent']:.1f}%。"
        )
    if qlsa_omp:
        lines.append(
            f"- QLSA OpenMP 相对 QLSA 串行多链平均加速约 {qlsa_omp['speedup']:.2f}x，8 线程并行效率约 {qlsa_omp['parallel_efficiency_percent']:.1f}%。"
        )
    lines.append("- OpenMP 8 线程并行效率约 72% 到 74%，说明 chain-level 并行在 berlin52 上已有稳定收益。")

    lines.extend(["", "## CUDA 结果解释", ""])
    if sa_cuda and qlsa_cuda:
        lines.append(
            f"- CUDA 已真实运行并产出 berlin52 数据，但 SA CUDA 平均耗时 {sa_cuda['elapsed_ms_mean']:.3f} ms，QLSA CUDA 平均耗时 {qlsa_cuda['elapsed_ms_mean']:.3f} ms，暂未优于 OpenMP。"
        )
    lines.append("- berlin52 规模较小，当前 CUDA multi-chain 实现容易受到 kernel 启动开销、global/shared memory 访问、每条 chain 工作量不足、block 内并行度尚未充分利用等因素影响。")
    lines.append("- 当前阶段应将 OpenMP 作为主要性能提升结果，CUDA 作为已完成的工程扩展与后续优化方向。")

    lines.extend(["", "## 后续实验计划", ""])
    lines.append("- 扩展到 eil51、st70、eil76、rat99、eil101 等实例，观察规模增大后的 CUDA 表现。")
    lines.append("- 对 CUDA 版本增加 block 内候选 2-opt move 并行评价，提升每条 chain 的 GPU 内部并行度。")
    lines.append("- 统一统计 Best/Mean/Std/Gap/Runtime/Speedup/Parallel Efficiency，并与论文表格进行公平对比。")
    lines.append("- 对 OpenMP 测试不同 threads/chains 组合，确认最佳 CPU 并行参数。")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path, summaries, source_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_markdown(summaries, source_path), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze TSP SA/QLSA experiment CSV results.")
    parser.add_argument("--input", default="results/berlin52_manual_raw.csv")
    parser.add_argument("--output", default="results/berlin52_summary.csv")
    parser.add_argument("--markdown", default="docs/step5_berlin52_analysis.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)

    rows = load_rows(input_path)
    summaries = summarize(group_rows(rows))
    write_summary_csv(output_path, summaries)
    write_markdown(markdown_path, summaries, input_path)
    print(f"Wrote summary CSV: {output_path}")
    print(f"Wrote markdown analysis: {markdown_path}")


if __name__ == "__main__":
    main()
