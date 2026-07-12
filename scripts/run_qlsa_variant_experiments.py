#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run QLSA current/paper/paper-sb variant comparison experiments."""

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

# The CLI has kept this 14-column prefix stable while extending its accounting
# and migration fields.  This runner persists only the comparison prefix, but
# must accept every supported CLI schema so current builds do not silently
# produce an empty experiment CSV.
PROGRAM_EXTENDED_HEADER = PROGRAM_HEADER + [
    "total_elapsed_ms",
    "cuda_kernel_elapsed_ms",
    "requested_backend",
    "actual_backend",
    "backend_fallback",
    "backend_fallback_reason",
    "iterations_completed",
    "deadline_reached",
]

PROGRAM_MIGRATION_HEADER = PROGRAM_EXTENDED_HEADER + [
    "migration_topology",
    "migration_interval",
    "migration_rounds",
    "migration_attempts",
    "migrations_adopted",
]

PROGRAM_HEADERS_BY_WIDTH = {
    len(PROGRAM_HEADER): PROGRAM_HEADER,
    len(PROGRAM_EXTENDED_HEADER): PROGRAM_EXTENDED_HEADER,
    len(PROGRAM_MIGRATION_HEADER): PROGRAM_MIGRATION_HEADER,
    len(PROGRAM_MIGRATION_HEADER) + 1: PROGRAM_MIGRATION_HEADER + ["actual_threads"],
}

EXTRA_HEADER = [
    "qlsa_variant",
    "policy",
    "diversity_threshold",
    "diversity_metric",
    "log_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["berlin52", "eil76", "rat99", "eil101"])
    parser.add_argument("--variants", nargs="+", choices=["current", "paper", "paper-sb"], default=["current", "paper", "paper-sb"])
    parser.add_argument("--policies", nargs="+", choices=["epsilon-greedy", "softmax"], default=["epsilon-greedy", "softmax"])
    parser.add_argument("--diversity-thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument(
        "--diversity-metric",
        choices=["edge", "hamming"],
        default="hamming",
        help="paper-sb metric; hamming preserves the published comparison semantics",
    )
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--input-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "qlsa_variant_alignment_raw.csv"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_executable() -> Path:
    candidates = [
        ROOT / "build" / "ninja-cuda-release" / "tsp_sa.exe",
        ROOT / "build" / "ninja-cuda-release" / "tsp_sa",
        ROOT / "build" / "ninja-cpu-release" / "tsp_sa.exe",
        ROOT / "build" / "ninja-cpu-release" / "tsp_sa",
        ROOT / "build" / "vs2022-cuda" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "vs2022-cpu" / "Release" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "tsp_sa",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find tsp_sa executable; build the project first "
        "(for example: cmake --preset ninja-cuda-release)"
    )


def rows_from_stdout(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("CSV:"):
            continue
        try:
            parts = next(csv.reader([line]))
        except csv.Error:
            continue
        if parts[0].startswith("qlsa") and len(parts) in PROGRAM_HEADERS_BY_WIDTH:
            rows.append(dict(zip(PROGRAM_HEADER, parts[: len(PROGRAM_HEADER)])))
    return rows


def run_one(
    exe: Path,
    input_path: Path,
    variant: str,
    policy: str,
    threshold: float | None,
    args: argparse.Namespace,
    log_dir: Path,
) -> list[dict[str, str]]:
    command = [
        str(exe),
        "--qlsa",
        "--qlsa_variant",
        variant,
        "--input",
        str(input_path),
        "--parallel",
        "omp",
        "--chains",
        str(args.chains),
        "--threads",
        str(args.threads),
        "--iterations",
        str(args.iterations),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
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
    threshold_value = ""
    if variant == "paper-sb" and threshold is not None:
        threshold_value = f"{threshold:.2f}"
        command.extend(
            [
                "--diversity_threshold",
                threshold_value,
                "--diversity_metric",
                args.diversity_metric,
            ]
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_threshold = threshold_value or "na"
    log_file = log_dir / f"{input_path.stem}_{variant}_{policy}_{safe_threshold}_{timestamp}.log"
    print("[run]", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
        row["qlsa_variant"] = variant
        row["policy"] = policy
        row["diversity_threshold"] = threshold_value
        row["diversity_metric"] = args.diversity_metric if variant == "paper-sb" else ""
        row["log_file"] = str(log_file.relative_to(ROOT))
    return rows


def main() -> int:
    args = parse_args()
    if args.quick:
        args.instances = ["berlin52", "rat99"]
        args.variants = ["current", "paper", "paper-sb"]
        args.policies = ["epsilon-greedy"]
        args.diversity_thresholds = [0.5]
        args.iterations = 100000
        args.repeat = 1
        args.chains = 16
        args.threads = 8

    exe = find_executable()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results" / "logs" / "qlsa_variant_alignment"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    for instance in args.instances:
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            print(f"[warning] missing instance, skipped: {input_path}")
            continue
        for variant in args.variants:
            thresholds: list[float | None] = [None]
            if variant == "paper-sb":
                thresholds = list(args.diversity_thresholds)
            for policy in args.policies:
                for threshold in thresholds:
                    all_rows.extend(run_one(exe, input_path, variant, policy, threshold, args, log_dir))

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[ok] wrote {len(all_rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
