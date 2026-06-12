#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
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

ALGORITHMS = {
    "sa",
    "qlsa",
    "sa-multichain",
    "qlsa-multichain",
    "sa-omp",
    "qlsa-omp",
    "sa-cuda",
    "qlsa-cuda",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Step 5 TSP experiments and analyze results.")
    parser.add_argument("--instances", nargs="+", default=["berlin52"])
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cuda-block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output", default="results/step5_raw.csv")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-qlsa", action="store_true")
    return parser.parse_args()


def find_executable(root):
    candidates = [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build-cuda-real" / "Release" / "tsp_sa.exe",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    lines = ["Could not find tsp_sa executable. Tried:"]
    lines.extend(f"  - {candidate}" for candidate in candidates)
    raise SystemExit("\n".join(lines))


def command_matrix(exe, input_path, args):
    common = [
        str(exe),
        "--input",
        str(input_path),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--init",
        "nn",
        "--repeat",
        str(args.repeat),
        "--csv-only",
    ]

    commands = []
    commands.append(common + ["--parallel", "none", "--chains", str(args.chains)])
    commands.append(
        common
        + [
            "--parallel",
            "omp",
            "--chains",
            str(args.chains),
            "--threads",
            str(args.threads),
        ]
    )
    if not args.no_cuda:
        commands.append(
            common
            + [
                "--parallel",
                "cuda",
                "--chains",
                str(args.chains),
                "--cuda_block_size",
                str(args.cuda_block_size),
            ]
        )

    if not args.no_qlsa:
        qlsa_common = common + [
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
        commands.append(qlsa_common + ["--parallel", "none", "--chains", str(args.chains)])
        commands.append(
            qlsa_common
            + [
                "--parallel",
                "omp",
                "--chains",
                str(args.chains),
                "--threads",
                str(args.threads),
            ]
        )
        if not args.no_cuda:
            commands.append(
                qlsa_common
                + [
                    "--parallel",
                    "cuda",
                    "--chains",
                    str(args.chains),
                    "--cuda_block_size",
                    str(args.cuda_block_size),
                ]
            )
    return commands


def safe_name(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:160]


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
            rows.append(parsed)
    return rows


def write_log(log_dir, name, command, completed):
    log_path = log_dir / f"{name}.log"
    content = [
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
    log_path.write_text("\n".join(content), encoding="utf-8")
    return log_path


def run_command(command, log_dir, label):
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = write_log(log_dir, f"{timestamp}_{safe_name(label)}", command, completed)
    combined = f"{completed.stdout}\n{completed.stderr}"
    if "falling back to serial multi-chain execution" in combined:
        print(f"[warning] fallback detected for {label}; see {log_path}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}; see {log_path}")
    rows = extract_csv_rows(completed.stdout)
    if not rows:
        raise SystemExit(f"No CSV rows found in command output; see {log_path}")
    return rows


def derived_paths(output_path):
    stem = output_path.stem
    if stem.endswith("_raw"):
        summary_stem = stem[: -len("_raw")] + "_summary"
    else:
        summary_stem = stem + "_summary"
    summary_path = output_path.with_name(summary_stem + ".csv")
    markdown_path = Path("docs") / (summary_stem.replace("_summary", "_analysis") + ".md")
    return summary_path, markdown_path


def call_analyzer(root, raw_path, summary_path, markdown_path):
    analyzer = root / "scripts" / "analyze_results.py"
    command = [
        sys.executable,
        str(analyzer),
        "--input",
        str(raw_path),
        "--output",
        str(summary_path),
        "--markdown",
        str(markdown_path),
    ]
    print("[analyze]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    root = Path.cwd()
    if args.quick:
        args.instances = ["berlin52"]
        args.iterations = 100_000
        args.repeat = 1

    exe = find_executable(root)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("results") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for instance in args.instances:
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            print(f"[warning] missing {input_path}; skipping {instance}", flush=True)
            continue
        for command in command_matrix(exe, input_path, args):
            label = f"{instance}_{command[command.index('--parallel') + 1]}_{'qlsa' if '--qlsa' in command else 'sa'}"
            all_rows.extend(run_command(command, log_dir, label))

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    print(f"Wrote raw CSV: {output_path}", flush=True)
    if not all_rows:
        print("No experiment rows were produced; skipping analysis.", flush=True)
        return

    summary_path, markdown_path = derived_paths(output_path)
    call_analyzer(root, output_path, summary_path, markdown_path)


if __name__ == "__main__":
    main()
