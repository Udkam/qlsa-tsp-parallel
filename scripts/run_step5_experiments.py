#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from scripts.experiment_csv import (
        CURRENT_HEADER,
        ExperimentCsvError,
        parse_program_rows,
        resolve_executable,
        row_for_output,
        validate_command_output,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from experiment_csv import (  # type: ignore[no-redef]
        CURRENT_HEADER,
        ExperimentCsvError,
        parse_program_rows,
        resolve_executable,
        row_for_output,
        validate_command_output,
    )

CSV_HEADER = CURRENT_HEADER

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
    parser.add_argument("--executable", type=Path, help="Explicit tsp_sa executable path.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-qlsa", action="store_true")
    return parser.parse_args()


def find_executable(root, explicit=None):
    candidates = [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build-cuda-real" / "Release" / "tsp_sa.exe",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]
    return resolve_executable(explicit, candidates, root=root, description="tsp_sa executable")


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


def extract_csv_rows(stdout, qlsa=None):
    if qlsa is None:
        predicate = lambda algorithm: algorithm.startswith(("sa", "qlsa"))
    elif qlsa:
        predicate = lambda algorithm: algorithm.startswith("qlsa")
    else:
        predicate = lambda algorithm: algorithm.startswith("sa")
    return parse_program_rows(stdout, algorithm_predicate=predicate)


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
    rows = extract_csv_rows(completed.stdout, qlsa="--qlsa" in command)
    try:
        validate_command_output(rows, command, source=label)
    except ExperimentCsvError as exc:
        raise SystemExit(f"{exc}; see {log_path}") from exc
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

    exe = find_executable(root, args.executable)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("results") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for instance in args.instances:
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            raise FileNotFoundError(f"missing requested instance: {input_path}")
        for command in command_matrix(exe, input_path, args):
            label = f"{instance}_{command[command.index('--parallel') + 1]}_{'qlsa' if '--qlsa' in command else 'sa'}"
            all_rows.extend(run_command(command, log_dir, label))

    if not all_rows:
        raise ExperimentCsvError("no experiment rows were produced")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(row_for_output(row, CSV_HEADER) for row in all_rows)

    print(f"Wrote raw CSV: {output_path}", flush=True)
    summary_path, markdown_path = derived_paths(output_path)
    call_analyzer(root, output_path, summary_path, markdown_path)


if __name__ == "__main__":
    main()
