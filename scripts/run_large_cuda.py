#!/usr/bin/env python3
"""Run large-instance CUDA chain vs SA candidate mode experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
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
EXTRA_HEADER = ["tier", "cuda_mode", "cuda_block_size", "cuda_candidates_per_iter", "status", "log_file", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--tier", nargs="+", choices=["L1", "L2", "L3"])
    parser.add_argument("--iterations", type=int, default=500_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--candidates-per-iter", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "large_cuda_raw.csv"))
    parser.add_argument("--executable", type=Path, help="Explicit CUDA-enabled tsp_sa executable path.")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_executable(explicit: Path | None = None) -> Path:
    return resolve_executable(
        explicit,
        [
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build-cuda-real" / "Release" / "tsp_sa.exe",
        ],
        root=ROOT,
        description="CUDA-enabled tsp_sa executable",
    )


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def tier_map(config: dict) -> dict[str, str]:
    return {instance: tier for tier, instances in config["tiers"].items() for instance in instances}


def selected_instances(args: argparse.Namespace, config: dict) -> list[str]:
    if args.quick:
        args.iterations = 200_000
        args.repeat = 1
        args.chains = 64
        args.block_size = 128
        args.candidates_per_iter = 128
        return ["ch130", "d198"]
    if args.instances:
        return args.instances
    tiers = args.tier or ["L1"]
    selected: list[str] = []
    for tier in tiers:
        selected.extend(config["tiers"].get(tier, []))
    return selected


def parse_rows(stdout: str) -> list[dict[str, str]]:
    return parse_program_rows(stdout, algorithm_predicate=lambda algorithm: algorithm.startswith("sa-cuda-"))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def run_one(exe: Path, input_path: Path, tier: str, mode: str, args: argparse.Namespace, log_dir: Path) -> list[dict[str, str]]:
    command = [
        str(exe), "--input", str(input_path), "--parallel", "cuda",
        "--cuda_mode", mode, "--cuda_block_size", str(args.block_size),
        "--cuda_candidates_per_iter", str(args.candidates_per_iter),
        "--chains", str(args.chains), "--iterations", str(args.iterations),
        "--repeat", str(args.repeat), "--seed", str(args.seed), "--init", "nn", "--csv-only",
    ]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"{input_path.stem}_{mode}_{timestamp}.log"
    print("[run]", " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=args.timeout if args.timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        log_file.write_text("$ " + " ".join(command) + "\n\n[timeout]\n" + str(exc), encoding="utf-8")
        raise ExperimentCsvError(
            f"{input_path.name} {mode}: command timed out; see {log_file.relative_to(ROOT)}"
        ) from exc
    log_file.write_text(
        "$ " + " ".join(command) + f"\n\nreturncode={completed.returncode}\n\n[stdout]\n"
        + completed.stdout + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ExperimentCsvError(
            f"{input_path.name} {mode}: command failed with return code {completed.returncode}; "
            f"see {log_file.relative_to(ROOT)}"
        )
    rows = parse_rows(completed.stdout)
    try:
        validate_command_output(rows, command, source=f"{input_path.name} {mode}")
    except ExperimentCsvError as exc:
        raise ExperimentCsvError(f"{exc}; see {log_file.relative_to(ROOT)}") from exc
    for row in rows:
        row["tier"] = tier
        row["cuda_mode"] = mode
        row["cuda_block_size"] = str(args.block_size)
        row["cuda_candidates_per_iter"] = str(args.candidates_per_iter)
        row["status"] = "ok"
        row["log_file"] = str(log_file.relative_to(ROOT))
        row["error"] = ""
    return rows


def main() -> int:
    args = parse_args()
    if args.candidates_per_iter <= 0 or args.candidates_per_iter > args.block_size:
        raise SystemExit("--candidates-per-iter must be positive and <= --block-size")
    config = load_config()
    tmap = tier_map(config)
    instances = selected_instances(args, config)
    exe = find_executable(args.executable)
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output = Path(args.output)
    if args.quick and output.name == "large_cuda_raw.csv":
        output = ROOT / "results" / "raw" / "large_cuda_quick_raw.csv"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results" / "logs" / "large_cuda"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for instance in instances:
        tier = tmap.get(instance, "custom")
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            raise FileNotFoundError(f"missing requested instance: {display_path(input_path)}")
        rows.extend(run_one(exe, input_path, tier, "chain", args, log_dir))
        rows.extend(run_one(exe, input_path, tier, "candidate", args, log_dir))

    if not rows:
        raise ExperimentCsvError("no experiment rows were produced")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows(row_for_output(row, PROGRAM_HEADER + EXTRA_HEADER) for row in rows)
    print(f"[ok] wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
