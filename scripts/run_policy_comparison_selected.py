#!/usr/bin/env python3
"""Run selected QLSA policy comparison experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
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
    "policy",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["eil76", "rat99", "eil101"])
    p.add_argument("--iterations", type=int, default=1_000_000)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--chains", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=301)
    p.add_argument("--input-dir", default="data")
    p.add_argument("--output", default="results/policy_comparison_raw.csv")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def find_executable(root: Path) -> Path:
    candidates = [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("Could not find tsp_sa executable")


def extract_rows(stdout: str):
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("algorithm,"):
            continue
        if line.startswith("qlsa-") and line.count(",") >= 13:
            rows.append(next(csv.reader([line])))
    return rows


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:180]


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = find_executable(root)
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = root / "results" / "logs" / "policy_comparison"
    log_dir.mkdir(parents=True, exist_ok=True)

    instances = args.instances
    iterations = args.iterations
    repeat = args.repeat
    if args.quick:
        instances = ["eil76"]
        iterations = 100_000
        repeat = 1

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for inst in instances:
            tsp_path = root / args.input_dir / f"{inst}.tsp"
            if not tsp_path.exists():
                print(f"[warning] missing instance, skipped: {tsp_path}")
                continue
            for policy in ["epsilon-greedy", "softmax"]:
                cmd = [
                    str(exe),
                    "--input",
                    str(tsp_path),
                    "--qlsa",
                    "--parallel",
                    "omp",
                    "--chains",
                    str(args.chains),
                    "--threads",
                    str(args.threads),
                    "--iterations",
                    str(iterations),
                    "--seed",
                    str(args.seed),
                    "--repeat",
                    str(repeat),
                    "--init",
                    "nn",
                    "--alpha",
                    "0.1",
                    "--gamma",
                    "0.9",
                    "--epsilon",
                    "0.1",
                    "--policy",
                    policy,
                    "--csv-only",
                ]
                print("[run]", " ".join(cmd))
                proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_name = safe_name(f"{stamp}_{inst}_{policy}.log")
                (log_dir / log_name).write_text(
                    "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
                    encoding="utf-8",
                )
                if proc.returncode != 0:
                    print(f"[error] command failed for {inst} {policy}, see {log_name}")
                    continue
                for row in extract_rows(proc.stdout):
                    writer.writerow(row + [policy])
    print(f"[done] wrote {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
