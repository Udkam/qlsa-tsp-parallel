#!/usr/bin/env python3
"""Generate convergence proxy data via increasing iteration budgets.

This avoids changing the core CLI for trace logging. Each point is an independent
run with the same seed and a larger iteration budget, so the output should be
read as a budget-sweep proxy rather than a true within-run trace.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


FIELDS = [
    "instance",
    "algorithm",
    "iteration_budget",
    "seed",
    "best_length",
    "final_length",
    "elapsed_ms",
    "accepted_moves",
    "improved_moves",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["berlin52", "rat99"])
    p.add_argument("--budgets", nargs="+", type=int, default=[10_000, 30_000, 100_000, 300_000, 1_000_000])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--input-dir", default="data")
    p.add_argument("--output-dir", default="results/traces")
    return p.parse_args()


def find_executable(root: Path) -> Path:
    for path in [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]:
        if path.exists():
            return path
    raise SystemExit("Could not find tsp_sa executable")


def extract_row(stdout: str):
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(("sa,", "qlsa,")):
            return next(csv.DictReader(["algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,best_length,final_length,elapsed_ms,accepted_moves,improved_moves", line]))
    raise RuntimeError("no CSV row found")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = find_executable(root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = root / "results" / "logs" / "convergence"
    log_dir.mkdir(parents=True, exist_ok=True)

    for inst in args.instances:
        tsp_path = root / args.input_dir / f"{inst}.tsp"
        if not tsp_path.exists():
            print(f"[warning] missing instance, skipped: {tsp_path}")
            continue
        rows = []
        for algorithm in ["sa", "qlsa"]:
            for budget in args.budgets:
                cmd = [
                    str(exe),
                    "--input",
                    str(tsp_path),
                    "--iterations",
                    str(budget),
                    "--seed",
                    str(args.seed),
                    "--init",
                    "nn",
                    "--csv-only",
                ]
                if algorithm == "qlsa":
                    cmd.insert(1, "--qlsa")
                    cmd.extend(["--alpha", "0.1", "--gamma", "0.9", "--epsilon", "0.1", "--policy", "epsilon-greedy"])
                print("[run]", " ".join(cmd))
                proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
                log_path = log_dir / f"{inst}_{algorithm}_{budget}.log"
                log_path.write_text(
                    "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
                    encoding="utf-8",
                )
                if proc.returncode != 0:
                    print(f"[error] failed {inst} {algorithm} budget={budget}; see {log_path}")
                    continue
                row = extract_row(proc.stdout)
                rows.append(
                    {
                        "instance": inst,
                        "algorithm": algorithm,
                        "iteration_budget": budget,
                        "seed": args.seed,
                        "best_length": row["best_length"],
                        "final_length": row["final_length"],
                        "elapsed_ms": row["elapsed_ms"],
                        "accepted_moves": row["accepted_moves"],
                        "improved_moves": row["improved_moves"],
                    }
                )
        out_path = out_dir / f"{inst}_budget_sweep.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] wrote {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
