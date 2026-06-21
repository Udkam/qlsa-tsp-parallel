#!/usr/bin/env python3
"""Run a CUDA candidate-mode parameter sweep.

This script is a thin experiment runner. It does not change algorithm behavior;
it only calls the existing `tsp_sa` CLI with explicit CUDA mode parameters and
records stdout/stderr logs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_HEADER = [
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
EXTRA_HEADER = ["cuda_mode", "cuda_block_size", "cuda_candidates_per_iter", "cuda_reversal_mode", "log_file"]


def find_executable() -> Path:
    candidates = [
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build-cuda-real" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "tsp_sa",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find tsp_sa executable; build the CUDA target first.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["a280", "lin318"])
    parser.add_argument("--algorithms", nargs="+", choices=["sa", "qlsa"], default=["sa"])
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--candidates-per-iter", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--reversal-modes", nargs="+", choices=["serial", "parallel"], default=["serial"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "cuda_candidate_sweep_raw.csv"))
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def rows_from_stdout(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("CSV:"):
            continue
        parts = next(csv.reader([line]))
        if len(parts) == len(PROGRAM_HEADER) and (parts[0].startswith("sa-cuda-") or parts[0].startswith("qlsa-cuda-")):
            rows.append(dict(zip(PROGRAM_HEADER, parts)))
    return rows


def run_candidate(
    exe: Path,
    input_path: Path,
    block_size: int,
    candidates: int,
    reversal_mode: str,
    algorithm: str,
    args: argparse.Namespace,
    log_dir: Path,
) -> list[dict[str, str]]:
    command = [
        str(exe),
        "--input",
        str(input_path),
        "--parallel",
        "cuda",
        "--cuda_mode",
        "candidate",
        "--cuda_block_size",
        str(block_size),
        "--cuda_candidates_per_iter",
        str(candidates),
        "--cuda_reversal_mode",
        reversal_mode,
        "--chains",
        str(args.chains),
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
    if algorithm == "qlsa":
        command[1:1] = [
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
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"{input_path.stem}_{algorithm}_candidate_{reversal_mode}_b{block_size}_c{candidates}_{stamp}.log"
    print("[run]", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=None if args.timeout <= 0 else args.timeout,
    )
    log_file.write_text(
        "$ " + " ".join(command) + "\n\n[stdout]\n" + completed.stdout + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed with exit code {completed.returncode}; see {log_file}")
    rows = rows_from_stdout(completed.stdout)
    for row in rows:
        row["cuda_mode"] = "candidate"
        row["cuda_block_size"] = str(block_size)
        row["cuda_candidates_per_iter"] = str(candidates)
        row["cuda_reversal_mode"] = reversal_mode
        row["log_file"] = str(log_file.relative_to(ROOT))
    return rows


def main() -> int:
    args = parse_args()
    exe = find_executable()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results" / "logs" / "cuda_candidate_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for instance in args.instances:
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            print(f"[warning] missing instance, skipped: {input_path}")
            continue
        for block_size in args.block_sizes:
            for candidates in args.candidates_per_iter:
                if candidates <= 0 or block_size <= 0:
                    print(f"[warning] skipped invalid block={block_size} candidates={candidates}")
                    continue
                if candidates > block_size:
                    print(f"[warning] skipped candidates_per_iter={candidates} > block_size={block_size}")
                    continue
                for algorithm in args.algorithms:
                    for reversal_mode in args.reversal_modes:
                        rows.extend(run_candidate(
                            exe, input_path, block_size, candidates,
                            reversal_mode, algorithm, args, log_dir
                        ))

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
