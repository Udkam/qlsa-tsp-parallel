#!/usr/bin/env python3
"""Summarize real MPI VM smoke raw CSV.

The script only summarizes rows already produced by `run_mpi_smoke.py`; it does
not run experiments and does not fabricate fallback rows.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "algorithm",
    "instance",
    "chains",
    "threads",
    "mpi_ranks",
    "runs",
    "fallback_rows",
    "best_length_min",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "communication_ms_mean",
    "accepted_moves_mean",
    "improved_moves_mean",
]


def mean(values):
    return sum(values) / len(values)


def stddev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["algorithm"],
            row["instance"],
            row["chains"],
            row["threads"],
            row["mpi_ranks"],
        )
        groups[key].append(row)

    out = []
    for (algorithm, instance, chains, threads, ranks), items in sorted(groups.items()):
        elapsed = [float(r["elapsed_ms"]) for r in items]
        comm = [float(r["communication_ms"]) for r in items]
        accepted = [float(r["accepted_moves"]) for r in items]
        improved = [float(r["improved_moves"]) for r in items]
        fallback_rows = sum(1 for r in items if r.get("fallback", "").lower() == "true")
        out.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "chains": chains,
                "threads": threads,
                "mpi_ranks": ranks,
                "runs": str(len(items)),
                "fallback_rows": str(fallback_rows),
                "best_length_min": str(min(int(r["best_length"]) for r in items)),
                "elapsed_ms_mean": f"{mean(elapsed):.3f}",
                "elapsed_ms_std": f"{stddev(elapsed):.3f}",
                "communication_ms_mean": f"{mean(comm):.3f}",
                "accepted_moves_mean": f"{mean(accepted):.3f}",
                "improved_moves_mean": f"{mean(improved):.3f}",
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize MPI VM smoke raw CSV.")
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "raw" / "mpi_vm_smoke_raw.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "summary" / "mpi_vm_smoke_summary.csv")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
