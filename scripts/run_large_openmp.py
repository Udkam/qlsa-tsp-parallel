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


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
PROGRAM_HEADER = [
    "algorithm", "instance", "dimension", "iterations", "seed", "init", "chains", "threads",
    "parallel", "best_length", "final_length", "elapsed_ms", "accepted_moves", "improved_moves",
]
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
    parser.add_argument("--timeout", type=int, default=0, help="Per-command timeout seconds; 0 disables timeout.")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_executable() -> Path:
    for candidate in [
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "tsp_sa",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find tsp_sa executable")


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
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("CSV:"):
            continue
        try:
            parsed = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(parsed) == len(PROGRAM_HEADER) and parsed[0] in {"sa-omp", "qlsa-omp"}:
            rows.append(dict(zip(PROGRAM_HEADER, parsed)))
    return rows


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


def error_row(instance: str, tier: str, args: argparse.Namespace, status: str, error: str, log_file: Path | None) -> dict[str, str]:
    row = {key: "" for key in PROGRAM_HEADER + EXTRA_HEADER}
    row.update({
        "instance": instance,
        "iterations": str(args.iterations),
        "chains": str(args.chains),
        "threads": str(args.threads),
        "parallel": "omp",
        "tier": tier,
        "status": status,
        "error": error,
        "log_file": str(log_file.relative_to(ROOT)) if log_file else "",
    })
    return row


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
        print(f"[timeout] {input_path.name} {algorithm}; see {log_file.relative_to(ROOT)}", flush=True)
        return [error_row(input_path.stem, tier, args, "timeout", str(exc), log_file)]

    log_file.write_text(
        "$ " + " ".join(command) + f"\n\nreturncode={completed.returncode}\n\n[stdout]\n"
        + completed.stdout + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        print(f"[error] command failed; see {log_file.relative_to(ROOT)}", flush=True)
        return [error_row(input_path.stem, tier, args, "failed", f"returncode={completed.returncode}", log_file)]
    rows = parse_rows(completed.stdout)
    for row in rows:
        row["tier"] = tier
        row["status"] = "ok"
        row["log_file"] = str(log_file.relative_to(ROOT))
        row["error"] = ""
    if not rows:
        return [error_row(input_path.stem, tier, args, "no_csv", "no CSV rows in stdout", log_file)]
    return rows


def main() -> int:
    args = parse_args()
    config = load_config()
    tmap = tier_map(config)
    instances = selected_instances(args, config)
    exe = find_executable()
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
            missing = display_path(input_path)
            print(f"[warning] missing {missing}; skipped", flush=True)
            rows.append(error_row(instance, tier, args, "missing", f"missing {missing}", None))
            continue
        for algorithm in algorithms:
            rows.extend(run_one(exe, input_path, tier, algorithm, args, log_dir))

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        raise
