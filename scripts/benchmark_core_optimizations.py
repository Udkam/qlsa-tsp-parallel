#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired before/after benchmark for allocation and OpenMP-team optimizations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.experiment_csv import parse_program_rows, validate_command_output
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from experiment_csv import parse_program_rows, validate_command_output  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Workload:
    name: str
    seed_base: int
    arguments: tuple[str, ...]


WORKLOADS = {
    "qlsa-softmax": Workload(
        "qlsa-softmax",
        1601,
        (
            "--qlsa",
            "--qlsa_variant",
            "current",
            "--policy",
            "softmax",
            "--input",
            "data/a280.tsp",
            "--parallel",
            "omp",
            "--chains",
            "32",
            "--threads",
            "8",
            "--iterations",
            "300000",
            "--repeat",
            "1",
            "--init",
            "nn",
            "--csv-only",
        ),
    ),
    "island-ring": Workload(
        "island-ring",
        1701,
        (
            "--qlsa",
            "--qlsa_variant",
            "current",
            "--policy",
            "softmax",
            "--input",
            "data/eil76.tsp",
            "--parallel",
            "omp",
            "--chains",
            "32",
            "--threads",
            "8",
            "--iterations",
            "1000000",
            "--repeat",
            "1",
            "--init",
            "nn",
            "--migration-topology",
            "ring",
            "--migration-interval",
            "10000",
            "--csv-only",
        ),
    ),
}

SEMANTIC_FIELDS = [
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
    "accepted_moves",
    "improved_moves",
    "requested_backend",
    "actual_backend",
    "backend_fallback",
    "backend_fallback_reason",
    "iterations_completed",
    "deadline_reached",
    "migration_topology",
    "migration_interval",
    "migration_rounds",
    "migration_attempts",
    "migrations_adopted",
    "actual_threads",
]

RAW_FIELDS = [
    "workload",
    "pair",
    "order",
    "implementation",
    "revision",
    "binary_sha256",
    "build_contract_sha256",
    "input_sha256",
    "benchmark_script_sha256",
    *SEMANTIC_FIELDS,
    "elapsed_ms",
    "total_elapsed_ms",
]

SUMMARY_FIELDS = [
    "workload",
    "instance",
    "iterations",
    "chains",
    "threads",
    "migration_topology",
    "migration_interval",
    "pairs",
    "before_revision",
    "after_revision",
    "before_binary_sha256",
    "after_binary_sha256",
    "build_contract_sha256",
    "input_sha256",
    "benchmark_script_sha256",
    "before_mean_ms",
    "before_median_ms",
    "after_mean_ms",
    "after_median_ms",
    "paired_speedup_mean",
    "paired_speedup_median",
    "paired_speedup_q1",
    "paired_speedup_q3",
    "paired_difference_median_ms",
    "semantic_matches",
]

BUILD_CONTRACT_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_GENERATOR",
    "OpenMP_CXX_FLAGS",
    "TSP_ENABLE_CUDA",
    "TSP_ENABLE_MPI",
    "TSP_ENABLE_OPENMP",
    "TSP_REQUIRE_CUDA",
    "TSP_REQUIRE_MPI",
    "TSP_REQUIRE_OPENMP",
)
BUILD_CONTRACT_DEFAULTS = {
    # The baseline revision predates this strict-MPI configuration switch. Its
    # behavior is equivalent to the current default when MPI itself is OFF.
    "TSP_REQUIRE_MPI": "OFF",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolved_executable(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    if not path.is_file():
        raise FileNotFoundError(f"benchmark executable does not exist: {path}")
    return path.resolve()


def resolved_source(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    if not path.is_dir():
        raise FileNotFoundError(f"benchmark source tree does not exist: {path}")
    return path.resolve()


def source_revision(source: Path, executable: Path) -> str:
    try:
        executable.relative_to(source)
    except ValueError as exc:
        raise RuntimeError(
            f"benchmark executable {executable} is outside declared source tree {source}"
        ) from exc
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if head.returncode != 0:
        raise RuntimeError(f"could not read Git revision for {source}: {head.stderr.strip()}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if dirty.returncode != 0:
        raise RuntimeError(f"could not inspect Git state for {source}: {dirty.stderr.strip()}")
    if dirty.stdout.strip():
        raise RuntimeError(f"benchmark source tree has tracked changes: {source}")
    return head.stdout.strip()


def cmake_build_contract(executable: Path) -> tuple[dict[str, str], str]:
    cache_path = executable.parent / "CMakeCache.txt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"benchmark build is missing CMakeCache.txt: {cache_path}")
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        declaration, value = line.split("=", 1)
        key = declaration.split(":", 1)[0]
        if key in BUILD_CONTRACT_KEYS:
            values[key] = value
    for key, value in BUILD_CONTRACT_DEFAULTS.items():
        values.setdefault(key, value)
    missing = sorted(set(BUILD_CONTRACT_KEYS) - values.keys())
    if missing:
        raise RuntimeError(
            f"CMake cache {cache_path} is missing build-contract fields: {', '.join(missing)}"
        )
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return values, sha256_bytes(encoded)


def workload_input(workload: Workload) -> Path:
    arguments = list(workload.arguments)
    position = arguments.index("--input")
    return (ROOT / arguments[position + 1]).resolve()


def command_for(executable: Path, workload: Workload, seed: int) -> list[str]:
    arguments = list(workload.arguments)
    seed_position = arguments.index("--repeat")
    arguments[seed_position:seed_position] = ["--seed", str(seed)]
    return [str(executable), *arguments]


def run_once(executable: Path, workload: Workload, seed: int) -> dict[str, str]:
    command = command_for(executable, workload, seed)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{completed.stderr}"
        )
    rows = parse_program_rows(completed.stdout)
    validate_command_output(rows, command, source=f"{workload.name} seed {seed}")
    return rows[0]


def assert_semantic_match(
    before: dict[str, str], after: dict[str, str], workload: str, seed: int
) -> None:
    differences = [
        f"{field}: {before.get(field)!r} != {after.get(field)!r}"
        for field in SEMANTIC_FIELDS
        if before.get(field, "") != after.get(field, "")
    ]
    if differences:
        raise RuntimeError(
            f"semantic mismatch for {workload} seed {seed}: " + "; ".join(differences)
        )


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for workload_name in sorted({row["workload"] for row in rows}):
        items = [row for row in rows if row["workload"] == workload_name]
        before_items = [row for row in items if row["implementation"] == "before"]
        after_items = [row for row in items if row["implementation"] == "after"]
        if not before_items or len(before_items) != len(after_items):
            raise RuntimeError(f"unbalanced benchmark rows for {workload_name}")
        before_times = [float(row["elapsed_ms"]) for row in before_items]
        after_times = [float(row["elapsed_ms"]) for row in after_items]
        paired_differences = [
            float(before["elapsed_ms"]) - float(after["elapsed_ms"])
            for before, after in zip(
                sorted(before_items, key=lambda row: int(row["pair"])),
                sorted(after_items, key=lambda row: int(row["pair"])),
            )
        ]
        paired_speedups = [
            float(before["elapsed_ms"]) / float(after["elapsed_ms"])
            for before, after in zip(
                sorted(before_items, key=lambda row: int(row["pair"])),
                sorted(after_items, key=lambda row: int(row["pair"])),
            )
        ]
        speedup_quartiles = (
            statistics.quantiles(paired_speedups, n=4, method="inclusive")
            if len(paired_speedups) > 1
            else [paired_speedups[0], paired_speedups[0], paired_speedups[0]]
        )
        first = before_items[0]
        before_mean = statistics.fmean(before_times)
        after_mean = statistics.fmean(after_times)
        before_median = statistics.median(before_times)
        after_median = statistics.median(after_times)
        summaries.append(
            {
                "workload": workload_name,
                "instance": first["instance"],
                "iterations": first["iterations"],
                "chains": first["chains"],
                "threads": first["threads"],
                "migration_topology": first["migration_topology"],
                "migration_interval": first["migration_interval"],
                "pairs": str(len(before_items)),
                "before_revision": first["revision"],
                "after_revision": after_items[0]["revision"],
                "before_binary_sha256": first["binary_sha256"],
                "after_binary_sha256": after_items[0]["binary_sha256"],
                "build_contract_sha256": first["build_contract_sha256"],
                "input_sha256": first["input_sha256"],
                "benchmark_script_sha256": first["benchmark_script_sha256"],
                "before_mean_ms": f"{before_mean:.3f}",
                "before_median_ms": f"{before_median:.3f}",
                "after_mean_ms": f"{after_mean:.3f}",
                "after_median_ms": f"{after_median:.3f}",
                "paired_speedup_mean": f"{statistics.fmean(paired_speedups):.4f}",
                "paired_speedup_median": f"{statistics.median(paired_speedups):.4f}",
                "paired_speedup_q1": f"{speedup_quartiles[0]:.4f}",
                "paired_speedup_q3": f"{speedup_quartiles[2]:.4f}",
                "paired_difference_median_ms": f"{statistics.median(paired_differences):.3f}",
                "semantic_matches": "true",
            }
        )
    return summaries


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument("--after-source", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=sorted(WORKLOADS),
        default=list(WORKLOADS),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=ROOT / "results" / "raw" / "core_optimization_benchmark_raw.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "summary" / "core_optimization_benchmark_summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    before_executable = resolved_executable(args.before)
    after_executable = resolved_executable(args.after)
    before_source = resolved_source(args.before_source)
    after_source = resolved_source(args.after_source)
    before_revision = source_revision(before_source, before_executable)
    after_revision = source_revision(after_source, after_executable)
    before_contract, before_contract_hash = cmake_build_contract(before_executable)
    after_contract, after_contract_hash = cmake_build_contract(after_executable)
    if before_contract != after_contract:
        differences = sorted(
            key
            for key in BUILD_CONTRACT_KEYS
            if before_contract.get(key) != after_contract.get(key)
        )
        raise RuntimeError(
            "before/after CMake build contracts differ: " + ", ".join(differences)
        )
    script_hash = sha256(Path(__file__).resolve())
    executable_info = {
        "before": (before_executable, before_revision, sha256(before_executable)),
        "after": (after_executable, after_revision, sha256(after_executable)),
    }

    raw_rows: list[dict[str, str]] = []
    for workload_name in args.workloads:
        workload = WORKLOADS[workload_name]
        input_hash = sha256(workload_input(workload))
        # Each solver invocation is a new process. One unrecorded launch per
        # executable warms shared OS file/code caches and lets the processor
        # leave its idle state before the paired samples begin.
        for implementation in ("before", "after"):
            run_once(executable_info[implementation][0], workload, workload.seed_base - 1)

        for pair in range(args.pairs):
            seed = workload.seed_base + pair
            order = ("before", "after") if pair % 2 == 0 else ("after", "before")
            pair_rows: dict[str, dict[str, str]] = {}
            for position, implementation in enumerate(order, start=1):
                executable, revision, binary_hash = executable_info[implementation]
                program_row = run_once(executable, workload, seed)
                row = {
                    "workload": workload_name,
                    "pair": str(pair + 1),
                    "order": str(position),
                    "implementation": implementation,
                    "revision": revision,
                    "binary_sha256": binary_hash,
                    "build_contract_sha256": before_contract_hash,
                    "input_sha256": input_hash,
                    "benchmark_script_sha256": script_hash,
                    **program_row,
                }
                raw_rows.append(row)
                pair_rows[implementation] = row
            assert_semantic_match(pair_rows["before"], pair_rows["after"], workload_name, seed)
            print(
                f"[pair] {workload_name} {pair + 1}/{args.pairs}: "
                f"before={pair_rows['before']['elapsed_ms']} ms, "
                f"after={pair_rows['after']['elapsed_ms']} ms"
            )

    raw_output = args.raw_output if args.raw_output.is_absolute() else ROOT / args.raw_output
    summary_output = (
        args.summary_output if args.summary_output.is_absolute() else ROOT / args.summary_output
    )
    write_csv(raw_output, RAW_FIELDS, raw_rows)
    summaries = summarize_rows(raw_rows)
    write_csv(summary_output, SUMMARY_FIELDS, summaries)
    print(f"[ok] wrote {raw_output.relative_to(ROOT)}")
    print(f"[ok] wrote {summary_output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
