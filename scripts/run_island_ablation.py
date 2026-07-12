#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a reproducible paired-seed island-migration ablation matrix.

The executable is invoked once per matrix cell.  Each run directory contains
the normalized raw observations, an incrementally updated manifest, one log per
command, a normalized configuration snapshot, and SHA-256 checksums.  The
runner intentionally requires the current 28-column CLI schema so that a CPU or
CUDA fallback cannot silently enter an OpenMP comparison.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "island_ablation_matrix.json"
SUPPORTED_ALGORITHMS = ("sa", "current", "paper", "paper-sb")
SUPPORTED_TOPOLOGIES = ("independent", "ring", "global")
SAFE_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

# The first 22 columns are the existing result and backend/accounting schema,
# followed by five island-migration columns and the measured worker count.
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
    "total_elapsed_ms",
    "cuda_kernel_elapsed_ms",
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

RAW_HEADER = [
    "job_id",
    "status",
    "error",
    "experiment_name",
    "algorithm_key",
    "algorithm_display",
    "qlsa_variant",
    "instance",
    "dimension",
    "bks",
    "seed",
    "paired_seed_index",
    "execution_order",
    "topology",
    "migration_interval",
    "command_migration_interval",
    "requested_backend",
    "reported_requested_backend",
    "actual_backend",
    "require_backend_match",
    "backend_fallback",
    "fallback_reason",
    "iterations_requested",
    "iterations_reported",
    "iterations_completed",
    "deadline_reached",
    "init",
    "chains",
    "threads",
    "actual_threads",
    "reported_parallel",
    "program_algorithm",
    "best_length",
    "final_length",
    "elapsed_ms",
    "total_elapsed_ms",
    "kernel_elapsed_ms",
    "wall_elapsed_ms",
    "accepted_moves",
    "improved_moves",
    "migration_rounds",
    "migration_attempts",
    "migrations_adopted",
    "migration_adoption_rate",
    "command_json",
    "log_file",
    "return_code",
    "started_at",
    "finished_at",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "input_sha256",
    "executable_sha256",
    "environment_sha256",
]


@dataclass(frozen=True)
class Job:
    job_id: str
    algorithm: dict[str, Any]
    instance: dict[str, Any]
    seed: int
    seed_index: int
    topology: str
    migration_interval: int
    command_migration_interval: int
    iterations: int
    chains: int
    threads: int
    requested_backend: str
    require_backend_match: bool
    command: list[str]
    execution_order: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instances", nargs="+", help="Configured instance-name subset.")
    parser.add_argument("--algorithms", nargs="+", choices=SUPPORTED_ALGORITHMS)
    parser.add_argument("--topologies", nargs="+", choices=SUPPORTED_TOPOLOGIES)
    parser.add_argument("--migration-intervals", nargs="+", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--seed-stride", type=int)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", help="Output directory name; it must not already exist.")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without requiring an executable or writing artifacts.",
    )
    return parser.parse_args(argv)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def normalized_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(normalized_json_bytes(value))
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if not str(config.get("experiment_name", "")).strip():
        raise ValueError("config experiment_name is required")

    instances = config.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("config instances must be a non-empty list")
    instance_names: list[str] = []
    for instance in instances:
        name = str(instance.get("name", "")).strip()
        if not name or not str(instance.get("path", "")).strip():
            raise ValueError("each instance requires name and path")
        if name in {".", ".."} or SAFE_PATH_COMPONENT_RE.fullmatch(name) is None:
            raise ValueError(
                f"instance name must be a safe ASCII path component, got {name!r}"
            )
        if "dimension" in instance:
            require_positive_int(instance["dimension"], f"instance {name} dimension")
        require_positive_int(instance.get("bks"), f"instance {name} bks")
        instance_names.append(name)
    if len(instance_names) != len(set(instance_names)):
        raise ValueError("instance names must be unique")

    algorithms = config.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("config algorithms must be a non-empty list")
    algorithm_keys: list[str] = []
    for algorithm in algorithms:
        key = str(algorithm.get("key", ""))
        if key not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported algorithm key: {key!r}")
        if key != "sa" and algorithm.get("qlsa_variant") != key:
            raise ValueError(f"algorithm {key} must set qlsa_variant={key!r}")
        if key == "sa" and bool(algorithm.get("qlsa")):
            raise ValueError("algorithm sa cannot set qlsa=true")
        if key != "sa" and algorithm.get("qlsa") is not True:
            raise ValueError(f"algorithm {key} must set qlsa=true")
        algorithm_keys.append(key)
    if len(algorithm_keys) != len(set(algorithm_keys)):
        raise ValueError("algorithm keys must be unique")
    if not {"sa", "paper-sb"}.issubset(algorithm_keys):
        raise ValueError("config must include selectable sa and paper-sb algorithms")

    paired = config.get("paired_seeds", {})
    if not isinstance(paired.get("start"), int) or paired["start"] < 0:
        raise ValueError("paired_seeds.start must be a non-negative integer")
    require_positive_int(paired.get("count"), "paired_seeds.count")
    require_positive_int(paired.get("stride"), "paired_seeds.stride")

    execution = config.get("execution", {})
    if execution.get("parallel") not in {"none", "omp"}:
        raise ValueError("execution.parallel must be none or omp for island experiments")
    require_positive_int(execution.get("chains"), "execution.chains")
    require_positive_int(execution.get("threads"), "execution.threads")
    require_positive_int(execution.get("iterations_per_island"), "execution.iterations_per_island")
    diversity_metric = str(execution.get("qlsa", {}).get("diversity_metric", "hamming"))
    if diversity_metric not in {"edge", "hamming"}:
        raise ValueError("execution.qlsa.diversity_metric must be edge or hamming")
    if execution.get("parallel") == "none" and execution.get("threads") != 1:
        raise ValueError("execution.threads must be 1 when execution.parallel is none")
    if execution.get("chains", 0) < 2:
        raise ValueError("execution.chains must be at least 2 for ring/global migration")
    if execution.get("require_backend_match") is not True:
        raise ValueError("execution.require_backend_match must be true")
    timeout = execution.get("command_timeout_seconds")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError("execution.command_timeout_seconds must be positive")

    migration = config.get("migration", {})
    topologies = migration.get("topologies")
    if not isinstance(topologies, list) or set(topologies) != set(SUPPORTED_TOPOLOGIES):
        raise ValueError(f"migration.topologies must contain exactly {list(SUPPORTED_TOPOLOGIES)}")
    if len(topologies) != len(set(topologies)):
        raise ValueError("migration.topologies must not contain duplicates")
    intervals = migration.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("migration.intervals must be a non-empty list")
    for interval in intervals:
        require_positive_int(interval, "migration interval")
    if len(intervals) != len(set(intervals)):
        raise ValueError("migration.intervals must be unique")


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    resolved = resolve_root_path(path)
    raw = resolved.read_bytes()
    try:
        config = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON config {resolved}: {exc}") from exc
    validate_config(config)
    normalized = normalized_json_bytes(config)
    return config, normalized


def select_named(
    items: list[dict[str, Any]], names: list[str] | None, key: str, label: str
) -> list[dict[str, Any]]:
    if not names:
        return items
    available = {str(item[key]): item for item in items}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; configured: {sorted(available)}")
    return [available[name] for name in names]


def make_seeds(config: dict[str, Any], args: argparse.Namespace) -> list[int]:
    paired = config["paired_seeds"]
    start = args.seed_start if args.seed_start is not None else int(paired["start"])
    count = args.seed_count if args.seed_count is not None else int(paired["count"])
    stride = args.seed_stride if args.seed_stride is not None else int(paired["stride"])
    if start < 0 or count <= 0 or stride <= 0:
        raise ValueError("seed start must be non-negative; count and stride must be positive")
    return [start + index * stride for index in range(count)]


def find_executable(config: dict[str, Any], override: Path | None, dry_run: bool) -> Path:
    if override is not None:
        candidate = resolve_root_path(override).resolve()
        if not dry_run and not candidate.is_file():
            raise FileNotFoundError(f"executable not found: {candidate}")
        return candidate
    if not dry_run:
        raise FileNotFoundError(
            "--executable is required for a formal island-ablation run; "
            "automatic executable discovery is intentionally disabled"
        )
    candidates = config.get("executable_candidates", [])
    if candidates:
        return resolve_root_path(candidates[0]).resolve()
    return (ROOT / "tsp_sa").resolve()


def build_command(
    executable: Path,
    execution: dict[str, Any],
    algorithm: dict[str, Any],
    instance_path: Path,
    seed: int,
    topology: str,
    migration_interval: int,
) -> list[str]:
    command = [
        str(executable),
        "--input",
        str(instance_path),
        "--parallel",
        str(execution["parallel"]),
        "--chains",
        str(execution["chains"]),
        "--threads",
        str(execution["threads"]),
        "--iterations",
        str(execution["iterations_per_island"]),
        "--seed",
        str(seed),
        "--repeat",
        "1",
        "--init",
        str(execution.get("init", "nn")),
        "--t0",
        str(execution.get("t0", 1000.0)),
        "--tf",
        str(execution.get("tf", 0.001)),
        "--migration-topology",
        topology,
        "--migration-interval",
        str(migration_interval),
    ]
    if algorithm.get("qlsa"):
        qlsa = execution.get("qlsa", {})
        command.extend(
            [
                "--qlsa",
                "--qlsa_variant",
                str(algorithm["qlsa_variant"]),
                "--alpha",
                str(qlsa.get("alpha", 0.1)),
                "--gamma",
                str(qlsa.get("gamma", 0.9)),
                "--epsilon",
                str(qlsa.get("epsilon", 0.1)),
                "--policy",
                str(qlsa.get("policy", "epsilon-greedy")),
                "--diversity_threshold",
                str(qlsa.get("diversity_threshold", 0.5)),
                "--diversity_metric",
                str(qlsa.get("diversity_metric", "hamming")),
            ]
        )
    command.append("--csv-only")
    return command


def build_jobs(config: dict[str, Any], args: argparse.Namespace, executable: Path) -> list[Job]:
    instances = select_named(config["instances"], args.instances, "name", "instances")
    algorithms = select_named(config["algorithms"], args.algorithms, "key", "algorithms")
    seeds = make_seeds(config, args)
    configured_topologies = list(config["migration"]["topologies"])
    topologies = list(args.topologies or configured_topologies)
    if any(topology not in configured_topologies for topology in topologies):
        raise ValueError("requested topology is not configured")
    if len(topologies) != len(set(topologies)):
        raise ValueError("requested topologies must not contain duplicates")
    configured_intervals = [int(value) for value in config["migration"]["intervals"]]
    intervals = list(args.migration_intervals or configured_intervals)
    if any(value <= 0 for value in intervals):
        raise ValueError("migration intervals must be positive")
    if len(intervals) != len(set(intervals)):
        raise ValueError("requested migration intervals must not contain duplicates")
    unknown_intervals = sorted(set(intervals) - set(configured_intervals))
    if unknown_intervals:
        raise ValueError(f"unconfigured migration intervals: {unknown_intervals}")

    execution = config["execution"]
    require_backend_match = bool(execution["require_backend_match"])
    # A condition is one topology/interval treatment.  Independent is repeated
    # at every interval so its resumable-chunk/barrier overhead is matched to
    # the corresponding ring/global treatment instead of biasing runtimes.
    conditions: list[tuple[str, int, int]] = []
    for topology in topologies:
        for interval in intervals:
            conditions.append((topology, interval, interval))

    jobs: list[Job] = []
    sequence = 0
    for instance in instances:
        instance_path = resolve_root_path(instance["path"]).resolve()
        for seed_index, seed in enumerate(seeds):
            # Cyclic condition rotations prevent every paired seed from
            # observing the same topology/interval process position. Algorithm
            # order rotates independently by seed so a condition cannot get a
            # fixed first algorithm through cancellation with condition order.
            condition_rotation = seed_index % len(conditions)
            ordered_conditions = (
                conditions[condition_rotation:] + conditions[:condition_rotation]
            )
            execution_order = 0
            for (
                topology,
                semantic_interval,
                command_interval,
            ) in ordered_conditions:
                algorithm_rotation = seed_index % len(algorithms)
                ordered_algorithms = (
                    algorithms[algorithm_rotation:] + algorithms[:algorithm_rotation]
                )
                for algorithm in ordered_algorithms:
                    sequence += 1
                    execution_order += 1
                    command = build_command(
                        executable,
                        execution,
                        algorithm,
                        instance_path,
                        seed,
                        topology,
                        command_interval,
                    )
                    safe_algorithm = str(algorithm["key"]).replace("-", "_")
                    job_id = (
                        f"{sequence:04d}_{instance['name']}_{safe_algorithm}_seed{seed}_"
                        f"{topology}_mi{semantic_interval}"
                    )
                    jobs.append(
                        Job(
                            job_id=job_id,
                            algorithm=algorithm,
                            instance=instance,
                            seed=seed,
                            seed_index=seed_index,
                            topology=topology,
                            migration_interval=semantic_interval,
                            command_migration_interval=command_interval,
                            iterations=int(execution["iterations_per_island"]),
                            chains=int(execution["chains"]),
                            threads=int(execution["threads"]),
                            requested_backend=str(execution["parallel"]),
                            require_backend_match=require_backend_match,
                            command=command,
                            execution_order=execution_order,
                        )
                    )
    return jobs


def command_display(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def parse_program_int(value: str, field: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {parsed}")
    return parsed


def parse_program_float(value: str, field: str, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}, got {value!r}")
    return parsed


def parse_program_bool(value: str, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def program_row_type_errors(row: dict[str, str]) -> list[str]:
    """Return every independent type/schema error in one pass."""

    errors: list[str] = []
    parsed_ints: dict[str, int] = {}
    parsed_floats: dict[str, float] = {}
    parsed_bools: dict[str, bool] = {}

    for field in (
        "algorithm",
        "instance",
        "init",
        "parallel",
        "requested_backend",
        "actual_backend",
        "migration_topology",
    ):
        if not row[field].strip():
            errors.append(f"{field} must not be empty")

    if row["init"].strip() not in {"nn", "random"}:
        errors.append(f"init has unsupported value {row['init']!r}")
    if row["algorithm"].strip() and not row["algorithm"].strip().lower().startswith(
        ("sa", "qlsa")
    ):
        errors.append(f"algorithm has unsupported value {row['algorithm']!r}")

    for field, minimum in (
        ("dimension", 1),
        ("iterations", 1),
        ("seed", 0),
        ("chains", 1),
        ("threads", 1),
        ("best_length", 0),
        ("final_length", 0),
        ("accepted_moves", 0),
        ("improved_moves", 0),
        ("iterations_completed", 0),
        ("migration_interval", 0),
        ("migration_rounds", 0),
        ("migration_attempts", 0),
        ("migrations_adopted", 0),
        ("actual_threads", 1),
    ):
        try:
            parsed_ints[field] = parse_program_int(row[field], field, minimum)
        except ValueError as exc:
            errors.append(str(exc))

    for field in ("elapsed_ms", "total_elapsed_ms", "cuda_kernel_elapsed_ms"):
        try:
            parsed_floats[field] = parse_program_float(row[field], field)
        except ValueError as exc:
            errors.append(str(exc))

    for field in ("backend_fallback", "deadline_reached"):
        try:
            parsed_bools[field] = parse_program_bool(row[field], field)
        except ValueError as exc:
            errors.append(str(exc))

    for field in ("parallel", "requested_backend", "actual_backend"):
        value = normalize_backend(row[field])
        if row[field].strip() and value not in {"serial", "omp", "cuda"}:
            errors.append(f"{field} has unsupported value {row[field]!r}")

    topology = row["migration_topology"].strip().lower()
    if topology and topology not in SUPPORTED_TOPOLOGIES:
        errors.append(f"migration_topology has unsupported value {topology!r}")

    if (
        "improved_moves" in parsed_ints
        and "accepted_moves" in parsed_ints
        and parsed_ints["improved_moves"] > parsed_ints["accepted_moves"]
    ):
        errors.append("improved_moves must not exceed accepted_moves")
    if (
        "best_length" in parsed_ints
        and "final_length" in parsed_ints
        and parsed_ints["best_length"] > parsed_ints["final_length"]
    ):
        errors.append("best_length must not exceed final_length")
    if (
        "cuda_kernel_elapsed_ms" in parsed_floats
        and "total_elapsed_ms" in parsed_floats
        and parsed_floats["cuda_kernel_elapsed_ms"]
        > parsed_floats["total_elapsed_ms"] + 1e-9
    ):
        errors.append("cuda_kernel_elapsed_ms must not exceed total_elapsed_ms")
    if (
        "migrations_adopted" in parsed_ints
        and "migration_attempts" in parsed_ints
        and parsed_ints["migrations_adopted"] > parsed_ints["migration_attempts"]
    ):
        errors.append("migrations_adopted must not exceed migration_attempts")
    if parsed_bools.get("backend_fallback") is True and not row[
        "backend_fallback_reason"
    ].strip():
        errors.append("backend_fallback_reason is required when backend_fallback=true")
    if parsed_bools.get("backend_fallback") is False and row[
        "backend_fallback_reason"
    ].strip():
        errors.append("backend_fallback_reason must be empty when backend_fallback=false")
    return errors


def parse_program_row(stdout: str) -> dict[str, str]:
    candidates: list[dict[str, str]] = []
    invalid_candidates: list[str] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("CSV:"):
            continue
        try:
            parts = next(csv.reader([line]))
        except csv.Error as exc:
            if line.lower().startswith(("sa", "qlsa")):
                invalid_candidates.append(
                    f"line {line_number}: malformed CSV ({exc})"
                )
            continue
        if not parts:
            continue
        looks_like_program_row = len(parts) == len(PROGRAM_HEADER) or parts[
            0
        ].strip().lower().startswith(("sa", "qlsa"))
        if not looks_like_program_row:
            continue
        if len(parts) != len(PROGRAM_HEADER):
            invalid_candidates.append(
                f"line {line_number}: expected strict 28-column schema, got {len(parts)}"
            )
            continue
        row = dict(zip(PROGRAM_HEADER, parts))
        type_errors = program_row_type_errors(row)
        if type_errors:
            invalid_candidates.append(
                f"line {line_number}: " + "; ".join(type_errors)
            )
            continue
        candidates.append(row)
    if invalid_candidates:
        raise ValueError("invalid program CSV row(s): " + "; ".join(invalid_candidates))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one strict 28-column program CSV row, found {len(candidates)}"
        )
    return candidates[0]


def normalize_backend(value: str) -> str:
    lowered = value.strip().lower()
    aliases = {
        "none": "serial",
        "cpu_serial": "serial",
        "serial-multichain": "serial",
        "openmp": "omp",
        "gpu": "cuda",
    }
    return aliases.get(lowered, lowered)


def base_raw_row(
    job: Job,
    config: dict[str, Any],
    git_state: dict[str, Any],
    config_hash: str,
    input_hash: str,
    executable_hash: str,
    log_relative: Path,
    environment_hash: str = "",
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": "error",
        "error": "",
        "experiment_name": config["experiment_name"],
        "algorithm_key": job.algorithm["key"],
        "algorithm_display": job.algorithm.get("display_name", job.algorithm["key"]),
        "qlsa_variant": job.algorithm.get("qlsa_variant", ""),
        "instance": job.instance["name"],
        "bks": job.instance["bks"],
        "seed": job.seed,
        "paired_seed_index": job.seed_index,
        "execution_order": job.execution_order,
        "topology": job.topology,
        "migration_interval": job.migration_interval,
        "command_migration_interval": job.command_migration_interval,
        "requested_backend": normalize_backend(job.requested_backend),
        "actual_backend": "unknown",
        "require_backend_match": str(job.require_backend_match).lower(),
        "iterations_requested": job.iterations,
        "chains": job.chains,
        "threads": job.threads,
        "command_json": json.dumps(job.command, ensure_ascii=False, separators=(",", ":")),
        "log_file": log_relative.as_posix(),
        "git_commit": git_state.get("commit", ""),
        "git_dirty": str(git_state.get("dirty", "")).lower(),
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "executable_sha256": executable_hash,
        "environment_sha256": environment_hash,
    }


def command_option(command: list[str], flag: str, default: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(command):
        return default
    return command[index + 1]


def expected_algorithm_label(job: Job) -> str:
    key = str(job.algorithm["key"])
    if key == "sa":
        base = "sa"
    else:
        variant = str(job.algorithm.get("qlsa_variant", key))
        base = "qlsa" if variant == "current" else f"qlsa-{variant}"
    label = f"{base}-island-{job.topology}"
    return label + ("-omp" if normalize_backend(job.requested_backend) == "omp" else "")


def expected_migration_counters(job: Job) -> tuple[int, int]:
    if job.topology == "independent":
        return 0, 0
    rounds = (job.iterations - 1) // job.command_migration_interval
    attempts_per_round = job.chains if job.topology == "ring" else job.chains - 1
    return rounds, rounds * attempts_per_round


def validate_and_copy_program_row(
    job: Job, program: dict[str, str], raw: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    requested = normalize_backend(job.requested_backend)
    reported_requested = normalize_backend(program["requested_backend"])
    actual = normalize_backend(program["actual_backend"])
    fallback = parse_program_bool(program["backend_fallback"], "backend_fallback")
    deadline_reached = parse_program_bool(program["deadline_reached"], "deadline_reached")
    attempts = int(program["migration_attempts"])
    adopted = int(program["migrations_adopted"])
    rounds = int(program["migration_rounds"])

    raw.update(
        {
            "dimension": program["dimension"],
            "iterations_reported": program["iterations"],
            "iterations_completed": program["iterations_completed"],
            "deadline_reached": str(deadline_reached).lower(),
            "init": program["init"],
            "chains": program["chains"],
            "threads": program["threads"],
            "actual_threads": program["actual_threads"],
            "reported_parallel": program["parallel"],
            "program_algorithm": program["algorithm"],
            "best_length": program["best_length"],
            "final_length": program["final_length"],
            "elapsed_ms": program["elapsed_ms"],
            "total_elapsed_ms": program["total_elapsed_ms"],
            "kernel_elapsed_ms": program["cuda_kernel_elapsed_ms"],
            "accepted_moves": program["accepted_moves"],
            "improved_moves": program["improved_moves"],
            "reported_requested_backend": reported_requested,
            "actual_backend": actual,
            "backend_fallback": str(fallback).lower(),
            "fallback_reason": program["backend_fallback_reason"],
            "migration_rounds": rounds,
            "migration_attempts": attempts,
            "migrations_adopted": adopted,
            "migration_adoption_rate": f"{(adopted / attempts if attempts else 0.0):.12g}",
        }
    )

    expected_algorithm = expected_algorithm_label(job)
    if program["algorithm"] != expected_algorithm:
        errors.append(
            f"reported algorithm {program['algorithm']!r} != expected {expected_algorithm!r} "
            f"for variant {job.algorithm.get('qlsa_variant', 'sa')!r}"
        )
    if int(program["seed"]) != job.seed:
        errors.append(f"reported seed {program['seed']} != requested {job.seed}")
    if program["instance"].strip().lower() != str(job.instance["name"]).lower():
        errors.append(f"reported instance {program['instance']!r} != {job.instance['name']!r}")
    expected_dimension = job.instance.get("dimension")
    if expected_dimension is not None and int(program["dimension"]) != int(expected_dimension):
        errors.append(
            f"reported dimension {program['dimension']} != expected {expected_dimension}"
        )
    if int(program["iterations"]) != job.iterations:
        errors.append(f"reported iterations {program['iterations']} != requested {job.iterations}")
    expected_completed = job.iterations * job.chains
    if int(program["iterations_completed"]) != expected_completed:
        errors.append(
            f"iterations_completed {program['iterations_completed']} != exact expected "
            f"iterations*chains {expected_completed}"
        )
    if deadline_reached:
        errors.append("fixed-iteration island run unexpectedly reported deadline_reached=true")
    expected_init = command_option(
        job.command,
        "--init",
        str((config or {}).get("execution", {}).get("init", "nn")),
    )
    if program["init"] != expected_init:
        errors.append(f"reported init {program['init']!r} != requested {expected_init!r}")
    if int(program["chains"]) != job.chains:
        errors.append(f"reported chains {program['chains']} != requested {job.chains}")
    if int(program["threads"]) != job.threads:
        errors.append(
            f"reported requested threads {program['threads']} != requested {job.threads}"
        )
    if int(program["actual_threads"]) != job.threads:
        errors.append(
            f"actual threads {program['actual_threads']} != requested {job.threads}"
        )
    if normalize_backend(program["parallel"]) != requested:
        errors.append(f"reported parallel {program['parallel']!r} != requested backend {requested!r}")
    if reported_requested != requested:
        errors.append(f"program requested backend {reported_requested!r} != runner request {requested!r}")
    if job.require_backend_match and actual != requested:
        errors.append(f"requested backend {requested!r}, actual backend {actual!r}")
    if fallback:
        reason = program["backend_fallback_reason"].strip()
        errors.append("backend fallback was reported" + (f": {reason}" if reason else ""))
    if program["migration_topology"].strip().lower() != job.topology:
        errors.append(
            f"reported topology {program['migration_topology']!r} != requested {job.topology!r}"
        )
    if int(program["migration_interval"]) != job.command_migration_interval:
        errors.append(
            f"reported migration interval {program['migration_interval']} != command interval "
            f"{job.command_migration_interval}"
        )
    expected_rounds, expected_attempts = expected_migration_counters(job)
    if rounds != expected_rounds:
        errors.append(
            f"migration_rounds {rounds} != expected {expected_rounds} for the iteration budget"
        )
    if attempts != expected_attempts:
        errors.append(
            f"migration_attempts {attempts} != expected {expected_attempts} for "
            f"{job.topology} topology"
        )
    if job.topology == "independent" and adopted != 0:
        errors.append("independent topology reported non-zero migrations_adopted")
    return errors


def process_timeout(config: dict[str, Any], override: float | None) -> float | None:
    timeout = override if override is not None else config["execution"].get("command_timeout_seconds")
    if timeout is None:
        return None
    timeout_value = float(timeout)
    if timeout_value <= 0:
        raise ValueError("timeout must be positive")
    return timeout_value


def run_job(
    job: Job,
    config: dict[str, Any],
    run_dir: Path,
    git_state: dict[str, Any],
    config_hash: str,
    executable_hash: str,
    input_hash: str,
    timeout_override: float | None,
    environment_hash: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    log_relative = Path("logs") / f"{job.job_id}.log"
    log_absolute = run_dir / log_relative
    raw = base_raw_row(
        job,
        config,
        git_state,
        config_hash,
        input_hash,
        executable_hash,
        log_relative,
        environment_hash,
    )
    started_at = now_iso()
    wall_start = time.perf_counter()
    return_code: int | None = None
    stdout = ""
    stderr = ""
    error = ""
    timeout = process_timeout(config, timeout_override)
    try:
        completed = subprocess.run(
            job.command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        if return_code != 0:
            error = f"command exited with code {return_code}"
        else:
            program = parse_program_row(stdout)
            validation_errors = validate_and_copy_program_row(job, program, raw, config)
            if validation_errors:
                error = "; ".join(validation_errors)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        error = f"command timed out after {timeout:.3f} seconds"
    except (OSError, ValueError) as exc:
        error = str(exc)

    wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
    finished_at = now_iso()
    raw.update(
        {
            "status": "ok" if not error else "error",
            "error": error,
            "wall_elapsed_ms": f"{wall_elapsed_ms:.3f}",
            "return_code": "" if return_code is None else return_code,
            "started_at": started_at,
            "finished_at": finished_at,
        }
    )

    log_text = (
        f"$ {command_display(job.command)}\n\n"
        f"[metadata]\nstarted_at={started_at}\nfinished_at={finished_at}\n"
        f"wall_elapsed_ms={wall_elapsed_ms:.3f}\nreturn_code={return_code}\n"
        f"timeout_seconds={timeout}\nerror={error}\n\n"
        f"[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n"
    )
    log_absolute.write_text(log_text, encoding="utf-8")
    manifest_job = {
        "job_id": job.job_id,
        "status": raw["status"],
        "error": error,
        "algorithm_key": job.algorithm["key"],
        "instance": job.instance["name"],
        "seed": job.seed,
        "execution_order": job.execution_order,
        "topology": job.topology,
        "migration_interval": job.migration_interval,
        "command_migration_interval": job.command_migration_interval,
        "requested_backend": normalize_backend(job.requested_backend),
        "actual_backend": raw["actual_backend"],
        "actual_threads": raw.get("actual_threads", ""),
        "command": job.command,
        "command_display": command_display(job.command),
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_elapsed_ms": wall_elapsed_ms,
        "log_file": log_relative.as_posix(),
        "log_sha256": sha256_file(log_absolute),
        "result": {
            key: raw.get(key, "") for key in RAW_HEADER if key != "command_json"
        },
    }
    return raw, manifest_job


def run_capture(command: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip()[:8000],
        "stderr": completed.stderr.strip()[:8000],
    }


def collect_git_state() -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--porcelain=v1"])
    commit_value = commit.get("stdout", "").splitlines()
    status_value = status.get("stdout", "")
    return {
        "commit": commit_value[0] if commit.get("return_code") == 0 and commit_value else "",
        "dirty": bool(status_value) if status.get("return_code") == 0 else None,
        "status_porcelain": status_value if status.get("return_code") == 0 else "",
    }


def collect_environment() -> dict[str, Any]:
    return {
        "captured_at": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
    }


def safe_run_id(value: str | None, experiment_name: str) -> str:
    if value:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ValueError("--run-id must contain only letters, digits, dot, underscore, or hyphen")
        return value
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name).strip("._-")
    return f"{stamp}_{safe_name}"


def write_checksum_sidecar(run_dir: Path, paths: Iterable[Path]) -> Path:
    entries: list[str] = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    checksum_path = run_dir / "checksums.sha256"
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return checksum_path


def print_dry_run(config: dict[str, Any], executable: Path, jobs: list[Job]) -> None:
    summary = {
        "dry_run": True,
        "experiment_name": config["experiment_name"],
        "executable": str(executable),
        "job_count": len(jobs),
        "instances": sorted({str(job.instance["name"]) for job in jobs}),
        "algorithms": sorted({str(job.algorithm["key"]) for job in jobs}),
        "seeds": sorted({job.seed for job in jobs}),
        "topologies": sorted({job.topology for job in jobs}),
        "migration_intervals": sorted(
            {job.migration_interval for job in jobs if job.topology != "independent"}
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    for job in jobs:
        print(f"[dry-run] {job.job_id}: {command_display(job.command)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = resolve_root_path(args.config)
        config, config_bytes = load_config(config_path)
        executable = find_executable(config, args.executable, args.dry_run)
        jobs = build_jobs(config, args, executable)
        if not jobs:
            raise ValueError("selected island-ablation matrix is empty")
        if args.dry_run:
            print_dry_run(config, executable, jobs)
            return 0

        output_root = resolve_root_path(args.output_root or config.get("output_root", "results/island_ablation"))
        run_id = safe_run_id(args.run_id, str(config["experiment_name"]))
        run_dir = output_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        (run_dir / "logs").mkdir(parents=True)

        snapshot_path = run_dir / "config.snapshot.json"
        snapshot_path.write_bytes(config_bytes)
        config_hash = sha256_bytes(config_bytes)
        executable_hash = sha256_file(executable)
        selected_instance_names = {str(job.instance["name"]) for job in jobs}
        input_hashes: dict[str, str] = {}
        input_paths: dict[str, Path] = {}
        for instance in config["instances"]:
            name = str(instance["name"])
            if name not in selected_instance_names:
                continue
            path = resolve_root_path(instance["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"instance file not found: {path}")
            input_paths[name] = path
            input_hashes[name] = sha256_file(path)

        git_state = collect_git_state()
        environment = collect_environment()
        environment_identity = {
            key: value for key, value in environment.items() if key != "captured_at"
        }
        environment_hash = sha256_bytes(normalized_json_bytes(environment_identity))
        raw_path = run_dir / "raw.csv"
        manifest_path = run_dir / "manifest.json"
        rows: list[dict[str, Any]] = []
        started_at = now_iso()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "experiment_name": config["experiment_name"],
            "started_at": started_at,
            "finished_at": None,
            "config_source": str(config_path.resolve()),
            "config_snapshot": snapshot_path.name,
            "config_sha256": config_hash,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "git": git_state,
            "environment": environment,
            "environment_sha256": environment_hash,
            "executable": {"path": str(executable), "sha256": executable_hash},
            "inputs": {
                name: {"path": str(input_paths[name]), "sha256": digest}
                for name, digest in input_hashes.items()
            },
            "matrix": {
                "job_count": len(jobs),
                "instances": sorted(selected_instance_names),
                "algorithms": sorted({str(job.algorithm["key"]) for job in jobs}),
                "seeds": sorted({job.seed for job in jobs}),
                "topologies": sorted({job.topology for job in jobs}),
                "migration_intervals": sorted(
                    {job.migration_interval for job in jobs if job.topology != "independent"}
                ),
                "independent_job_count": sum(job.topology == "independent" for job in jobs),
                "execution_order_design": (
                    "paired-seed cyclic Latin rotation over topology/interval conditions; "
                    "algorithm order rotates within each condition"
                ),
            },
            "jobs": [],
            "artifacts": {},
        }
        atomic_write_csv(raw_path, rows)
        atomic_write_json(manifest_path, manifest)

        failed = False
        for index, job in enumerate(jobs, start=1):
            print(f"[run {index}/{len(jobs)}] {job.job_id}", flush=True)
            raw, manifest_job = run_job(
                job,
                config,
                run_dir,
                git_state,
                config_hash,
                executable_hash,
                input_hashes[str(job.instance["name"])],
                args.timeout_seconds,
                environment_hash,
            )
            rows.append(raw)
            manifest["jobs"].append(manifest_job)
            atomic_write_csv(raw_path, rows)
            atomic_write_json(manifest_path, manifest)
            if raw["status"] != "ok":
                failed = True
                print(f"[error] {job.job_id}: {raw['error']}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    break

        manifest["finished_at"] = now_iso()
        manifest["status"] = "failed" if failed else "complete"
        manifest["completed_job_count"] = len(rows)
        manifest["successful_job_count"] = sum(row["status"] == "ok" for row in rows)
        manifest["failed_job_count"] = sum(row["status"] != "ok" for row in rows)
        log_artifacts = [
            {"path": job["log_file"], "sha256": job["log_sha256"]}
            for job in manifest["jobs"]
        ]
        checksum_scope = [
            snapshot_path.name,
            raw_path.name,
            manifest_path.name,
            *(item["path"] for item in log_artifacts),
        ]
        manifest["artifacts"] = {
            "raw_csv": {"path": raw_path.name, "sha256": sha256_file(raw_path)},
            "config_snapshot": {
                "path": snapshot_path.name,
                "sha256": sha256_file(snapshot_path),
            },
            "logs": log_artifacts,
            "checksums": {
                "path": "checksums.sha256",
                "covers": checksum_scope,
                "note": "The checksum sidecar cannot recursively checksum itself.",
            },
        }
        atomic_write_json(manifest_path, manifest)
        checksum_inputs = [
            snapshot_path,
            raw_path,
            manifest_path,
            *(run_dir / item["path"] for item in log_artifacts),
        ]
        checksum_path = write_checksum_sidecar(run_dir, checksum_inputs)
        print(f"[ok] raw CSV: {raw_path}")
        print(f"[ok] manifest: {manifest_path}")
        print(f"[ok] checksums: {checksum_path}")
        return 1 if failed else 0
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
