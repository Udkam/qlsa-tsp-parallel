#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


CSV_HEADER = [
    "algorithm",
    "instance",
    "dimension",
    "iterations",
    "seed",
    "init",
    "chains",
    "threads",
    "parallel",
    "best_length",
    "final_length",
    "elapsed_ms",
    "accepted_moves",
    "improved_moves",
]

ALGORITHMS = {"sa-omp", "qlsa-omp"}
BKS = {
    "berlin52": 7542,
    "eil51": 426,
    "st70": 675,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "pr76": 108159,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenMP threads/chains scaling grid.")
    parser.add_argument("--instances", nargs="+", default=["berlin52", "eil101"])
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--raw-output", default="results/openmp_scaling_grid_raw.csv")
    parser.add_argument("--summary-output", default="results/openmp_scaling_grid_summary.csv")
    parser.add_argument("--markdown", default="docs/step6A_openmp_scaling_analysis.md")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_executable(root):
    candidates = [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("Could not find tsp_sa executable.")


def safe_name(text):
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")[:180]


def extract_csv_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(parsed) == len(CSV_HEADER) and parsed[0] in ALGORITHMS:
            rows.append(dict(zip(CSV_HEADER, parsed)))
    return rows


def write_log(log_dir, label, command, completed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"{timestamp}_{safe_name(label)}.log"
    path.write_text(
        "\n".join(
            [
                f"command: {' '.join(command)}",
                f"returncode: {completed.returncode}",
                "",
                "===== STDOUT =====",
                completed.stdout,
                "",
                "===== STDERR =====",
                completed.stderr,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run_command(command, log_dir, label):
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path = write_log(log_dir, label, command, completed)
    if "falling back to serial multi-chain execution" in f"{completed.stdout}\n{completed.stderr}":
        print(f"[warning] fallback detected for {label}; see {log_path}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}; see {log_path}")
    rows = extract_csv_rows(completed.stdout)
    if not rows:
        raise SystemExit(f"No CSV rows found in command output; see {log_path}")
    return rows


def command_for(exe, input_path, args, instance, chains, threads, qlsa):
    command = [
        str(exe),
        "--input",
        str(input_path),
        "--parallel",
        "omp",
        "--chains",
        str(chains),
        "--threads",
        str(threads),
        "--iterations",
        str(args.iterations),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--init",
        "nn",
        "--csv-only",
    ]
    if qlsa:
        command.extend(
            [
                "--qlsa",
                "--alpha",
                "0.1",
                "--gamma",
                "0.9",
                "--epsilon",
                "0.1",
                "--policy",
                "epsilon-greedy",
            ]
        )
    return command


def write_raw(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["instance"],
            row["algorithm"],
            int(row["chains"]),
            int(row["threads"]),
            int(row["iterations"]),
        )
        grouped[key].append(row)

    summaries = []
    baselines = {}
    for key, group in grouped.items():
        instance, algorithm, chains, threads, iterations = key
        best_values = [int(row["best_length"]) for row in group]
        elapsed_values = [float(row["elapsed_ms"]) for row in group]
        accepted = [float(row["accepted_moves"]) for row in group]
        improved = [float(row["improved_moves"]) for row in group]
        bks = BKS.get(instance)
        best_min = min(best_values)
        gap = 100.0 * (best_min - bks) / bks if bks else 0.0
        record = {
            "instance": instance,
            "algorithm": algorithm,
            "chains": chains,
            "threads": threads,
            "iterations": iterations,
            "runs": len(group),
            "bks": bks if bks is not None else "",
            "best_length_min": best_min,
            "gap_percent": gap,
            "elapsed_ms_mean": mean(elapsed_values),
            "elapsed_ms_std": sample_std(elapsed_values),
            "speedup": "",
            "parallel_efficiency_percent": "",
            "accepted_moves_mean": mean(accepted),
            "improved_moves_mean": mean(improved),
        }
        summaries.append(record)
        if threads == 1:
            baselines[(instance, algorithm, chains)] = record["elapsed_ms_mean"]

    for record in summaries:
        baseline = baselines.get((record["instance"], record["algorithm"], record["chains"]))
        if baseline and record["elapsed_ms_mean"] > 0:
            speedup = baseline / record["elapsed_ms_mean"]
            record["speedup"] = speedup
            record["parallel_efficiency_percent"] = 100.0 * speedup / record["threads"]

    summaries.sort(key=lambda row: (row["instance"], row["algorithm"], row["chains"], row["threads"]))
    return summaries


def write_summary(path, summaries):
    fields = [
        "instance",
        "algorithm",
        "chains",
        "threads",
        "iterations",
        "runs",
        "bks",
        "best_length_min",
        "gap_percent",
        "elapsed_ms_mean",
        "elapsed_ms_std",
        "speedup",
        "parallel_efficiency_percent",
        "accepted_moves_mean",
        "improved_moves_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    field: f"{row[field]:.6f}" if isinstance(row[field], float) else row[field]
                    for field in fields
                }
            )


def md_float(value, digits=3):
    if value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def write_markdown(path, summaries):
    best = min(summaries, key=lambda row: (row["gap_percent"], row["elapsed_ms_mean"])) if summaries else None
    lines = [
        "# Step 6A OpenMP Scaling Grid Analysis",
        "",
        "## Purpose",
        "",
        "This experiment measures the relationship between OpenMP threads and independent search chains.",
        "",
        "## Summary Table",
        "",
        "| Instance | Algorithm | Chains | Threads | Runs | Best | Gap % | Mean ms | Speedup | Efficiency % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {instance} | {algorithm} | {chains} | {threads} | {runs} | {best} | {gap} | {ms} | {speedup} | {eff} |".format(
                instance=row["instance"],
                algorithm=row["algorithm"],
                chains=row["chains"],
                threads=row["threads"],
                runs=row["runs"],
                best=row["best_length_min"],
                gap=md_float(row["gap_percent"]),
                ms=md_float(row["elapsed_ms_mean"]),
                speedup=md_float(row["speedup"]),
                eff=md_float(row["parallel_efficiency_percent"], 2),
            )
        )
    lines.extend(["", "## Best Combination", ""])
    if best:
        lines.append(
            f"- Best score by quality then time: {best['instance']} {best['algorithm']} chains={best['chains']} threads={best['threads']} best={best['best_length_min']} gap={best['gap_percent']:.3f}% mean_ms={best['elapsed_ms_mean']:.3f}."
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- Speedup is computed against the same instance, algorithm, and chains with threads=1.")
    lines.append("- If the machine has fewer physical cores than requested threads, results still record the actual measured runtime.")
    lines.append("- Full conclusions should use the full grid, not quick mode.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.quick:
        args.instances = ["berlin52"]
        args.iterations = 100_000
        args.repeat = 1
        args.chains = [8]
        args.threads = [1, 2]

    root = Path.cwd()
    exe = find_executable(root)
    input_dir = Path(args.input_dir)
    log_dir = Path("results") / "logs" / "openmp_scaling_grid"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for instance in args.instances:
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            print(f"[warning] missing {input_path}; skipping {instance}", flush=True)
            continue
        for chains in args.chains:
            for threads in args.threads:
                for qlsa in (False, True):
                    command = command_for(exe, input_path, args, instance, chains, threads, qlsa)
                    label = f"{instance}_{'qlsa' if qlsa else 'sa'}_c{chains}_t{threads}"
                    rows.extend(run_command(command, log_dir, label))

    raw_path = Path(args.raw_output)
    summary_path = Path(args.summary_output)
    markdown_path = Path(args.markdown)
    write_raw(raw_path, rows)
    summaries = summarize(rows)
    write_summary(summary_path, summaries)
    write_markdown(markdown_path, summaries)
    print(f"Wrote raw CSV: {raw_path}", flush=True)
    print(f"Wrote summary CSV: {summary_path}", flush=True)
    print(f"Wrote markdown analysis: {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
