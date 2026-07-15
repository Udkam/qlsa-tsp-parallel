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

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_HEADER = CURRENT_HEADER
EXTRA_HEADER = [
    "cuda_mode",
    "cuda_block_size",
    "cuda_candidates_per_iter",
    "cuda_reversal_mode",
    "cuda_candidate_policy",
    "log_file",
]


def find_executable(explicit: Path | None = None) -> Path:
    candidates = [
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build-cuda-real" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "tsp_sa",
    ]
    return resolve_executable(
        explicit,
        candidates,
        root=ROOT,
        description="CUDA-enabled tsp_sa executable",
    )


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
    parser.add_argument("--candidate-policies", nargs="+", choices=["best", "random", "hybrid"], default=["best"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "cuda_candidate_sweep_raw.csv"))
    parser.add_argument("--executable", type=Path, help="Explicit tsp_sa executable path.")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def rows_from_stdout(stdout: str) -> list[dict[str, str]]:
    return parse_program_rows(
        stdout,
        algorithm_predicate=lambda algorithm: algorithm.startswith(("sa-cuda-", "qlsa-cuda-")),
    )


def run_candidate(
    exe: Path,
    input_path: Path,
    block_size: int,
    candidates: int,
    reversal_mode: str,
    candidate_policy: str,
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
        "--cuda_candidate_policy",
        candidate_policy,
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
    log_file = log_dir / f"{input_path.stem}_{algorithm}_candidate_{reversal_mode}_{candidate_policy}_b{block_size}_c{candidates}_{stamp}.log"
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
    try:
        validate_command_output(rows, command, source=f"{input_path.name} {algorithm} candidate")
    except ExperimentCsvError as exc:
        raise ExperimentCsvError(f"{exc}; see {log_file}") from exc
    for row in rows:
        row["cuda_mode"] = "candidate"
        row["cuda_block_size"] = str(block_size)
        row["cuda_candidates_per_iter"] = str(candidates)
        row["cuda_reversal_mode"] = reversal_mode
        row["cuda_candidate_policy"] = candidate_policy
        row["log_file"] = str(log_file.relative_to(ROOT))
    return rows


def main() -> int:
    args = parse_args()
    exe = find_executable(args.executable)
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
            raise FileNotFoundError(f"missing requested instance: {input_path}")
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
                        for candidate_policy in args.candidate_policies:
                            rows.extend(run_candidate(
                                exe, input_path, block_size, candidates,
                                reversal_mode, candidate_policy, algorithm, args, log_dir
                            ))

    if not rows:
        raise ExperimentCsvError("no experiment rows were produced")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows(row_for_output(row, PROGRAM_HEADER + EXTRA_HEADER) for row in rows)
    print(f"[ok] wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
