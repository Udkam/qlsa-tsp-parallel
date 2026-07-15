#!/usr/bin/env python3
"""Run large-instance OpenMP SA/QLSA experiments with missing/timeout handling."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
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
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
PROGRAM_HEADER = CURRENT_HEADER
EXTRA_HEADER = ["tier", "status", "log_file", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", nargs="+", choices=["L1", "L2", "L3"])
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--algorithm", choices=["sa", "qlsa", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "large_openmp_raw.csv"))
    parser.add_argument("--executable", type=Path, help="Explicit tsp_sa executable path.")
    parser.add_argument("--timeout", type=int, default=0, help="Per-command timeout seconds; 0 disables timeout.")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_executable(explicit: Path | None = None) -> Path:
    return resolve_executable(
        explicit,
        [
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "tsp_sa",
        ],
        root=ROOT,
        description="tsp_sa executable",
    )


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def tier_map(config: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for tier, instances in config["tiers"].items():
        for instance in instances:
            out[instance] = tier
    return out


def selected_instances(args: argparse.Namespace, config: dict) -> list[str]:
    if args.quick:
        args.iterations = 300_000
        args.repeat = 1
        args.chains = 64
        return ["ch130", "d198", "a280"]
    if args.instances:
        return args.instances
    tiers = args.tier or ["L1"]
    selected: list[str] = []
    for tier in tiers:
        selected.extend(config["tiers"].get(tier, []))
    return selected


def parse_rows(stdout: str) -> list[dict[str, str]]:
    return parse_program_rows(stdout, algorithm_predicate=lambda algorithm: algorithm in {"sa-omp", "qlsa-omp"})


def command_for(exe: Path, input_path: Path, algorithm: str, args: argparse.Namespace) -> list[str]:
    command = [
        str(exe), "--input", str(input_path), "--parallel", "omp",
        "--chains", str(args.chains), "--threads", str(args.threads),
        "--iterations", str(args.iterations), "--repeat", str(args.repeat),
        "--seed", str(args.seed), "--init", "nn", "--csv-only",
    ]
    if algorithm == "qlsa":
        command.extend(["--qlsa", "--alpha", "0.1", "--gamma", "0.9", "--epsilon", "0.1", "--policy", "epsilon-greedy"])
    return command


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def run_one(exe: Path, input_path: Path, tier: str, algorithm: str, args: argparse.Namespace, log_dir: Path) -> list[dict[str, str]]:
    command = command_for(exe, input_path, algorithm, args)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"{input_path.stem}_{algorithm}_{timestamp}.log"
    print("[run]", " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout if args.timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        log_file.write_text("$ " + " ".join(command) + "\n\n[timeout]\n" + str(exc), encoding="utf-8")
        raise ExperimentCsvError(
            f"{input_path.name} {algorithm}: command timed out; see {log_file.relative_to(ROOT)}"
        ) from exc

    log_file.write_text(
        "$ " + " ".join(command) + f"\n\nreturncode={completed.returncode}\n\n[stdout]\n"
        + completed.stdout + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ExperimentCsvError(
            f"{input_path.name} {algorithm}: command failed with return code {completed.returncode}; "
            f"see {log_file.relative_to(ROOT)}"
        )
    rows = parse_rows(completed.stdout)
    try:
        validate_command_output(rows, command, source=f"{input_path.name} {algorithm}")
    except ExperimentCsvError as exc:
        raise ExperimentCsvError(f"{exc}; see {log_file.relative_to(ROOT)}") from exc
    for row in rows:
        row["tier"] = tier
        row["status"] = "ok"
        row["log_file"] = str(log_file.relative_to(ROOT))
        row["error"] = ""
    return rows


def main() -> int:
    args = parse_args()
    config = load_config()
    tmap = tier_map(config)
    instances = selected_instances(args, config)
    exe = find_executable(args.executable)
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output = Path(args.output)
    if args.quick and output.name == "large_openmp_raw.csv":
        output = ROOT / "results" / "raw" / "large_openmp_quick_raw.csv"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results" / "logs" / "large_openmp"
    log_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["sa", "qlsa"] if args.algorithm == "both" else [args.algorithm]
    rows: list[dict[str, str]] = []
    for instance in instances:
        tier = tmap.get(instance, "custom")
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            raise FileNotFoundError(f"missing requested instance: {display_path(input_path)}")
        for algorithm in algorithms:
            rows.extend(run_one(exe, input_path, tier, algorithm, args, log_dir))

    if not rows:
        raise ExperimentCsvError("no experiment rows were produced")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows(row_for_output(row, PROGRAM_HEADER + EXTRA_HEADER) for row in rows)
    print(f"[ok] wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        raise
