#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a paired-seed, equal-budget SA/QLSA experiment matrix.

Each algorithm/instance/seed combination is launched as a separate process so
that paired observations remain explicit.  A run directory contains the raw
CSV, a machine-readable manifest, per-command logs, a normalized configuration
snapshot, and SHA-256 checksums.
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
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "fair_experiment_matrix.json"
BUDGET_SCHEMES = ("equal-iterations", "fixed-time")
ALGORITHM_KEYS = ("sa", "current", "paper", "paper-sb")
SAFE_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

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

# Newer executables append these fields while preserving the original 14-column
# prefix.  Keeping the prefix parser allows the runner to audit older builds too.
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

PROGRAM_ACTUAL_THREADS_HEADER = PROGRAM_MIGRATION_HEADER + ["actual_threads"]

PROGRAM_HEADERS_BY_WIDTH = {
    len(PROGRAM_HEADER): PROGRAM_HEADER,
    len(PROGRAM_EXTENDED_HEADER): PROGRAM_EXTENDED_HEADER,
    len(PROGRAM_MIGRATION_HEADER): PROGRAM_MIGRATION_HEADER,
    len(PROGRAM_ACTUAL_THREADS_HEADER): PROGRAM_ACTUAL_THREADS_HEADER,
}

RAW_HEADER = [
    "job_id",
    "status",
    "error",
    "experiment_name",
    "budget_scheme",
    "budget_target",
    "budget_unit",
    "algorithm_key",
    "algorithm_display",
    "qlsa_variant",
    "instance",
    "dimension",
    "bks",
    "seed",
    "paired_seed_index",
    "execution_order",
    "requested_backend",
    "actual_backend",
    "require_backend_match",
    "iterations_requested",
    "iterations_reported",
    "time_limit_ms",
    "proposal_cost_per_iteration",
    "proposal_evaluations_per_chain",
    "proposal_evaluations_total",
    "proposal_evaluations_actual_total",
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
    "iterations_completed",
    "deadline_reached",
    "fixed_time_tolerance_ms",
    "fixed_time_elapsed_delta_ms",
    "backend_fallback",
    "fallback_reason",
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
    budget_scheme: str
    budget_target: int
    budget_unit: str
    algorithm: dict[str, Any]
    instance: dict[str, Any]
    seed: int
    seed_index: int
    iterations: int
    time_limit_ms: int | None
    proposals_per_chain: int
    proposals_total: int
    chains: int
    threads: int
    requested_backend: str
    require_backend_match: bool
    command: list[str]
    execution_order: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--budget",
        choices=[*BUDGET_SCHEMES, "all"],
        help="Budget scheme to run (default: config default_budget).",
    )
    parser.add_argument("--instances", nargs="+", help="Subset of configured instance names.")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_KEYS)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--seed-stride", type=int)
    parser.add_argument("--executable", type=Path)
    parser.add_argument(
        "--allow-auto-executable",
        action="store_true",
        help="Allow configured executable candidates when --executable is omitted.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", help="Stable output directory name; must not already exist.")
    parser.add_argument("--timeout-seconds", type=float, help="Per-process timeout override.")
    parser.add_argument("--allow-backend-fallback", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed job.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the full matrix without writing files.")
    return parser.parse_args(argv)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(normalized_json_bytes(value))
    os.replace(temporary, path)


def run_capture(command: list[str], cwd: Path = ROOT, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return {
        "command": command,
        "return_code": completed.returncode,
        "output": output[:8000],
    }


def collect_git_state() -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--porcelain=v1"])
    return {
        "commit": commit.get("output", "").splitlines()[0] if commit.get("return_code") == 0 else "",
        "dirty": bool(status.get("output", "")) if status.get("return_code") == 0 else None,
        "status_porcelain": status.get("output", "") if status.get("return_code") == 0 else "",
    }


def total_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.total_phys)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def find_cmake_cache(executable: Path) -> Path | None:
    for parent in [executable.parent, *executable.parents]:
        candidate = parent / "CMakeCache.txt"
        if candidate.exists():
            return candidate
        if parent == ROOT:
            break
    return None


def read_compiler_metadata(executable: Path) -> dict[str, Any]:
    cache_path = find_cmake_cache(executable)
    metadata: dict[str, Any] = {"cmake_cache": str(cache_path) if cache_path else ""}
    if cache_path:
        interesting = {
            "CMAKE_CXX_COMPILER",
            "CMAKE_CXX_COMPILER_ID",
            "CMAKE_CXX_COMPILER_VERSION",
            "CMAKE_CUDA_COMPILER",
            "CMAKE_CUDA_COMPILER_ID",
            "CMAKE_CUDA_COMPILER_VERSION",
            "CMAKE_BUILD_TYPE",
            "CMAKE_GENERATOR",
        }
        for raw_line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line or raw_line.startswith(("#", "//")) or "=" not in raw_line:
                continue
            lhs, value = raw_line.split("=", 1)
            key = lhs.split(":", 1)[0]
            if key in interesting:
                metadata[key] = value

        compiler = metadata.get("CMAKE_CXX_COMPILER")
        if compiler:
            compiler_path = Path(str(compiler))
            command = [str(compiler_path)] if compiler_path.name.lower() in {"cl", "cl.exe"} else [str(compiler_path), "--version"]
            metadata["cxx_compiler_version_output"] = run_capture(command, cwd=cache_path.parent)

    metadata["cmake_version"] = run_capture(["cmake", "--version"])
    if shutil.which("nvcc"):
        metadata["nvcc_version"] = run_capture(["nvcc", "--version"])
    return metadata


def collect_environment(executable: Path) -> dict[str, Any]:
    gpu: dict[str, Any]
    if shutil.which("nvidia-smi"):
        gpu = run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    else:
        gpu = {"available": False}
    return {
        "captured_at": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes(),
        "python": sys.version,
        "python_executable": sys.executable,
        "gpu": gpu,
        "compiler": read_compiler_metadata(executable),
    }


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    resolved = path if path.is_absolute() else ROOT / path
    raw = resolved.read_bytes()
    try:
        config = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON config {resolved}: {exc}") from exc
    validate_config(config)
    return config, normalized_json_bytes(config)


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
    algorithms = config.get("algorithms")
    if not isinstance(instances, list) or not instances:
        raise ValueError("config instances must be a non-empty list")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("config algorithms must be a non-empty list")

    instance_names: list[str] = []
    for item in instances:
        name = str(item.get("name", "")).strip()
        if not name or not item.get("path"):
            raise ValueError("each instance requires name and path")
        if name in {".", ".."} or SAFE_PATH_COMPONENT_RE.fullmatch(name) is None:
            raise ValueError(
                f"instance name must be a safe ASCII path component, got {name!r}"
            )
        require_positive_int(item.get("bks"), f"instance {name} bks")
        instance_names.append(name)
    if len(instance_names) != len(set(instance_names)):
        raise ValueError("instance names must be unique")

    algorithm_names: list[str] = []
    for item in algorithms:
        key = str(item.get("key", ""))
        if key not in ALGORITHM_KEYS:
            raise ValueError(f"unsupported algorithm key: {key!r}")
        require_positive_int(item.get("proposal_cost_per_iteration"), f"algorithm {key} proposal cost")
        if key != "sa" and item.get("qlsa_variant") != key:
            raise ValueError(f"algorithm {key} must use qlsa_variant={key!r}")
        algorithm_names.append(key)
    if len(algorithm_names) != len(set(algorithm_names)):
        raise ValueError("algorithm keys must be unique")
    missing_algorithms = set(ALGORITHM_KEYS) - set(algorithm_names)
    if missing_algorithms:
        raise ValueError(f"config must include all comparison algorithms: {sorted(missing_algorithms)}")

    seed_config = config.get("paired_seeds", {})
    require_positive_int(seed_config.get("count"), "paired_seeds.count")
    require_positive_int(seed_config.get("stride"), "paired_seeds.stride")
    if not isinstance(seed_config.get("start"), int) or seed_config["start"] < 0:
        raise ValueError("paired_seeds.start must be a non-negative integer")

    execution = config.get("execution", {})
    if execution.get("parallel") not in {"none", "omp", "cuda"}:
        raise ValueError("execution.parallel must be none, omp, or cuda")
    require_positive_int(execution.get("chains"), "execution.chains")
    require_positive_int(execution.get("threads"), "execution.threads")
    diversity_metric = str(execution.get("qlsa", {}).get("diversity_metric", "hamming"))
    if diversity_metric not in {"edge", "hamming"}:
        raise ValueError("execution.qlsa.diversity_metric must be edge or hamming")

    budgets = config.get("budgets", {})
    if "equal-proposals" in budgets:
        raise ValueError(
            "equal-proposals is deprecated: proposal_cost_per_iteration=1 makes it "
            "identical to equal-iterations rather than an independent equal-work budget"
        )
    require_positive_int(budgets.get("equal-iterations", {}).get("iterations_per_chain"), "equal-iterations.iterations_per_chain")
    require_positive_int(budgets.get("fixed-time", {}).get("time_limit_ms"), "fixed-time.time_limit_ms")
    require_positive_int(
        budgets.get("fixed-time", {}).get("iterations_ceiling_per_chain"),
        "fixed-time.iterations_ceiling_per_chain",
    )
    for budget_name, budget in budgets.items():
        if budget_name not in BUDGET_SCHEMES or not isinstance(budget, dict):
            continue
        if "chains" in budget:
            require_positive_int(budget["chains"], f"{budget_name}.chains")
        if "threads" in budget:
            require_positive_int(budget["threads"], f"{budget_name}.threads")
    if config.get("default_budget") not in BUDGET_SCHEMES:
        raise ValueError(f"default_budget must be one of {BUDGET_SCHEMES}")


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_named(items: list[dict[str, Any]], names: list[str] | None, key: str, label: str) -> list[dict[str, Any]]:
    if not names:
        return items
    available = {str(item[key]): item for item in items}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; configured: {sorted(available)}")
    return [available[name] for name in names]


def find_executable(
    config: dict[str, Any], override: Path | None, allow_auto_executable: bool = False
) -> Path:
    if override:
        candidate = resolve_root_path(override)
        if not candidate.is_file():
            raise FileNotFoundError(f"executable not found: {candidate}")
        return candidate.resolve()
    if not allow_auto_executable:
        raise FileNotFoundError(
            "--executable is required for a reproducible experiment; "
            "pass --allow-auto-executable to opt into configured candidates"
        )
    for value in config.get("executable_candidates", []):
        candidate = resolve_root_path(value)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not find tsp_sa executable; pass --executable after building the project")


def selected_budgets(config: dict[str, Any], requested: str | None) -> list[str]:
    budget = requested or str(config["default_budget"])
    return list(BUDGET_SCHEMES) if budget == "all" else [budget]


def make_seeds(config: dict[str, Any], args: argparse.Namespace) -> list[int]:
    seed_config = config["paired_seeds"]
    start = args.seed_start if args.seed_start is not None else int(seed_config["start"])
    count = args.seed_count if args.seed_count is not None else int(seed_config["count"])
    stride = args.seed_stride if args.seed_stride is not None else int(seed_config["stride"])
    if start < 0 or count <= 0 or stride <= 0:
        raise ValueError("seed start must be non-negative; count and stride must be positive")
    return [start + index * stride for index in range(count)]


def derive_budget(
    config: dict[str, Any], budget_scheme: str, algorithm: dict[str, Any], chains: int
) -> tuple[int, int | None, int, int, int, str]:
    budget = config["budgets"][budget_scheme]
    proposal_cost = int(algorithm["proposal_cost_per_iteration"])
    if budget_scheme == "equal-iterations":
        iterations = int(budget["iterations_per_chain"])
        target = iterations
        unit = "iterations_per_chain"
    elif budget_scheme == "fixed-time":
        iterations = int(budget["iterations_ceiling_per_chain"])
        target = int(budget["time_limit_ms"])
        unit = "solver_wall_time_ms"
    else:
        raise ValueError(f"unsupported budget scheme: {budget_scheme}")
    time_limit_ms = int(budget["time_limit_ms"]) if budget_scheme == "fixed-time" else None
    proposals_per_chain = iterations * proposal_cost
    return iterations, time_limit_ms, proposals_per_chain, proposals_per_chain * chains, target, unit


def build_command(
    executable: Path,
    execution: dict[str, Any],
    algorithm: dict[str, Any],
    instance_path: Path,
    iterations: int,
    time_limit_ms: int | None,
    seed: int,
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
        str(iterations),
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
    ]
    if time_limit_ms is not None:
        command.extend(["--time-limit-ms", str(time_limit_ms)])
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
    budgets = selected_budgets(config, args.budget)
    seeds = make_seeds(config, args)
    base_execution = dict(config["execution"])
    require_backend_match = bool(base_execution.get("require_backend_match", True)) and not args.allow_backend_fallback

    for instance in instances:
        path = resolve_root_path(instance["path"])
        if not path.is_file():
            raise FileNotFoundError(f"instance file not found: {path}")

    jobs: list[Job] = []
    sequence = 0
    for budget_scheme in budgets:
        execution = dict(base_execution)
        budget_execution = config["budgets"][budget_scheme]
        execution["chains"] = int(budget_execution.get("chains", base_execution["chains"]))
        execution["threads"] = int(budget_execution.get("threads", base_execution["threads"]))
        chains = int(execution["chains"])
        for instance in instances:
            instance_path = resolve_root_path(instance["path"]).resolve()
            for seed_index, seed in enumerate(seeds):
                # A cyclic Latin square balances process-order effects while
                # preserving paired seeds.  The order is deterministic and is
                # recorded in raw.csv for every observation.
                rotation = seed_index % len(algorithms)
                ordered_algorithms = algorithms[rotation:] + algorithms[:rotation]
                for execution_order, algorithm in enumerate(ordered_algorithms, start=1):
                    sequence += 1
                    iterations, time_limit_ms, proposals, proposals_total, target, unit = derive_budget(
                        config, budget_scheme, algorithm, chains
                    )
                    command = build_command(
                        executable,
                        execution,
                        algorithm,
                        instance_path,
                        iterations,
                        time_limit_ms,
                        seed,
                    )
                    safe_budget = budget_scheme.replace("-", "_")
                    job_id = f"{sequence:04d}_{safe_budget}_{instance['name']}_{algorithm['key']}_seed{seed}"
                    jobs.append(
                        Job(
                            job_id=job_id,
                            budget_scheme=budget_scheme,
                            budget_target=target,
                            budget_unit=unit,
                            algorithm=algorithm,
                            instance=instance,
                            seed=seed,
                            seed_index=seed_index,
                            iterations=iterations,
                            time_limit_ms=time_limit_ms,
                            proposals_per_chain=proposals,
                            proposals_total=proposals_total,
                            chains=chains,
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


def parse_program_bool(value: str, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


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


def validate_program_row_types(row: dict[str, str]) -> None:
    for field in ("algorithm", "instance", "init", "parallel"):
        if not row[field].strip():
            raise ValueError(f"{field} must not be empty")
    if row["init"] not in {"nn", "random"}:
        raise ValueError(f"init has unsupported value {row['init']!r}")

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
    ):
        parse_program_int(row[field], field, minimum)
    parse_program_float(row["elapsed_ms"], "elapsed_ms")

    reported_parallel = normalize_backend(row["parallel"])
    if reported_parallel not in {"serial", "omp", "cuda"}:
        raise ValueError(f"parallel has unsupported value {row['parallel']!r}")

    if "actual_backend" in row:
        for field in ("requested_backend", "actual_backend"):
            if not row[field].strip():
                raise ValueError(f"{field} must not be empty")
            if normalize_backend(row[field]) not in {"serial", "omp", "cuda"}:
                raise ValueError(f"{field} has unsupported value {row[field]!r}")
        total_elapsed = parse_program_float(row["total_elapsed_ms"], "total_elapsed_ms")
        kernel_elapsed = parse_program_float(
            row["cuda_kernel_elapsed_ms"], "cuda_kernel_elapsed_ms"
        )
        if kernel_elapsed > total_elapsed + 1e-9:
            raise ValueError("cuda_kernel_elapsed_ms must not exceed total_elapsed_ms")
        fallback = parse_program_bool(row["backend_fallback"], "backend_fallback")
        if fallback and not row["backend_fallback_reason"].strip():
            raise ValueError("backend_fallback_reason is required when backend_fallback=true")
        parse_program_int(row["iterations_completed"], "iterations_completed", 0)
        parse_program_bool(row["deadline_reached"], "deadline_reached")

    if "migration_topology" in row:
        topology = row["migration_topology"].strip().lower()
        if topology not in {"disabled", "independent", "ring", "global"}:
            raise ValueError(f"migration_topology has unsupported value {topology!r}")
        migration_counts: dict[str, int] = {}
        for field in (
            "migration_interval",
            "migration_rounds",
            "migration_attempts",
            "migrations_adopted",
        ):
            migration_counts[field] = parse_program_int(row[field], field, 0)
        if migration_counts["migrations_adopted"] > migration_counts["migration_attempts"]:
            raise ValueError("migrations_adopted must not exceed migration_attempts")

    if "actual_threads" in row:
        parse_program_int(row["actual_threads"], "actual_threads", 1)


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
                invalid_candidates.append(f"line {line_number}: malformed CSV ({exc})")
            continue
        if not parts or not parts[0].strip().lower().startswith(("sa", "qlsa")):
            continue
        header = PROGRAM_HEADERS_BY_WIDTH.get(len(parts))
        if header is None:
            invalid_candidates.append(
                f"line {line_number}: unsupported CSV column count {len(parts)}; "
                f"expected one of {sorted(PROGRAM_HEADERS_BY_WIDTH)}"
            )
            continue
        row = dict(zip(header, parts))
        try:
            validate_program_row_types(row)
        except ValueError as exc:
            invalid_candidates.append(f"line {line_number}: {exc}")
            continue
        candidates.append(row)
    if invalid_candidates:
        raise ValueError("invalid program CSV row(s): " + "; ".join(invalid_candidates))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one program CSV row, found {len(candidates)}")
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


def detect_actual_backend(requested: str, stdout: str, stderr: str) -> str:
    combined = stdout + "\n" + stderr
    explicit = re.search(r"actual[_ -]?backend\s*[=:]\s*([A-Za-z0-9_-]+)", combined, re.IGNORECASE)
    if explicit:
        return normalize_backend(explicit.group(1))
    lowered = combined.lower()
    if "falling back to serial" in lowered or "fallback to serial" in lowered:
        return "serial"
    return normalize_backend(requested)


def base_raw_row(
    job: Job,
    config: dict[str, Any],
    git_state: dict[str, Any],
    config_hash: str,
    input_hash: str,
    executable_hash: str,
    log_file: Path,
    environment_hash: str = "",
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": "error",
        "error": "",
        "experiment_name": config["experiment_name"],
        "budget_scheme": job.budget_scheme,
        "budget_target": job.budget_target,
        "budget_unit": job.budget_unit,
        "algorithm_key": job.algorithm["key"],
        "algorithm_display": job.algorithm.get("display_name", job.algorithm["key"]),
        "qlsa_variant": job.algorithm.get("qlsa_variant", ""),
        "instance": job.instance["name"],
        "bks": job.instance["bks"],
        "seed": job.seed,
        "paired_seed_index": job.seed_index,
        "execution_order": job.execution_order,
        "requested_backend": normalize_backend(job.requested_backend),
        "actual_backend": "unknown",
        "require_backend_match": str(job.require_backend_match).lower(),
        "iterations_requested": job.iterations,
        "time_limit_ms": job.time_limit_ms if job.time_limit_ms is not None else "",
        "proposal_cost_per_iteration": job.algorithm["proposal_cost_per_iteration"],
        "proposal_evaluations_per_chain": job.proposals_per_chain,
        "proposal_evaluations_total": job.proposals_total,
        "proposal_evaluations_actual_total": "",
        "chains": job.chains,
        "threads": job.threads,
        "command_json": json.dumps(job.command, ensure_ascii=False, separators=(",", ":")),
        "log_file": str(log_file),
        "git_commit": git_state.get("commit", ""),
        "git_dirty": str(git_state.get("dirty", "")).lower(),
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "executable_sha256": executable_hash,
        "environment_sha256": environment_hash,
    }


def write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def process_timeout(job: Job, config: dict[str, Any], override: float | None) -> float | None:
    if override is not None:
        if override <= 0:
            raise ValueError("--timeout-seconds must be positive")
        return override
    execution = config["execution"]
    configured = execution.get("command_timeout_seconds")
    if configured is not None:
        return float(configured)
    if job.time_limit_ms is not None:
        grace = float(execution.get("timeout_grace_seconds", 120))
        return job.time_limit_ms / 1000.0 + grace
    return None


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

    requested = normalize_backend(job.requested_backend)
    migration = command_option(job.command, "--migration-topology", "disabled")
    if migration != "disabled":
        label = f"{base}-island-{migration}"
        return label + ("-omp" if requested == "omp" else "")
    if requested == "cuda":
        mode = command_option(job.command, "--cuda_mode", "chain")
        policy = command_option(job.command, "--cuda_candidate_policy", "best")
        if mode == "candidate" and policy != "best":
            return f"{base}-cuda-candidate-{policy}"
        return f"{base}-cuda-{mode}"
    if requested == "omp":
        return f"{base}-omp"
    if job.chains > 1:
        return f"{base}-multichain"
    return base


def expected_effective_threads(job: Job) -> int:
    requested = normalize_backend(job.requested_backend)
    if requested == "omp":
        return job.threads
    if requested == "cuda":
        return int(command_option(job.command, "--cuda_block_size", "128"))
    return 1


def validate_program_against_job(
    program: dict[str, str], job: Job, config: dict[str, Any], raw: dict[str, Any]
) -> list[str]:
    errors: list[str] = []

    expected_algorithm = expected_algorithm_label(job)
    if program["algorithm"] != expected_algorithm:
        errors.append(
            f"reported algorithm {program['algorithm']!r} does not match expected "
            f"{expected_algorithm!r} for variant {job.algorithm.get('qlsa_variant', 'sa')!r}"
        )
    if int(program["seed"]) != job.seed:
        errors.append(f"reported seed {program['seed']} does not match requested seed {job.seed}")
    if program["instance"].lower() != str(job.instance["name"]).lower():
        errors.append(
            f"reported instance {program['instance']!r} does not match {job.instance['name']!r}"
        )
    if int(program["iterations"]) != job.iterations:
        errors.append(
            f"reported iterations {program['iterations']} does not match requested {job.iterations}"
        )
    if int(program["chains"]) != job.chains:
        errors.append(f"reported chains {program['chains']} does not match requested {job.chains}")
    effective_threads = expected_effective_threads(job)
    if int(program["threads"]) != effective_threads:
        errors.append(
            f"reported effective threads {program['threads']} does not match expected {effective_threads}"
        )
    if "actual_threads" in program and int(program["actual_threads"]) != effective_threads:
        errors.append(
            f"observed actual_threads {program['actual_threads']} does not match requested "
            f"effective threads {effective_threads}"
        )
    expected_init = command_option(
        job.command, "--init", str(config.get("execution", {}).get("init", "nn"))
    )
    if program["init"] != expected_init:
        errors.append(f"reported init {program['init']!r} does not match requested {expected_init!r}")

    requested_backend = normalize_backend(job.requested_backend)
    reported_parallel = normalize_backend(program["parallel"])
    if reported_parallel != requested_backend:
        errors.append(
            f"reported parallel mode {reported_parallel!r} does not match requested "
            f"{requested_backend!r}"
        )
    if "requested_backend" in program:
        reported_requested = normalize_backend(program["requested_backend"])
        if reported_requested != requested_backend:
            errors.append(
                f"program reported requested backend {reported_requested!r}, "
                f"runner requested {requested_backend!r}"
            )

    ceiling = job.iterations * job.chains
    completed_text = program.get("iterations_completed", "")
    deadline_text = program.get("deadline_reached", "")
    if not completed_text or not deadline_text:
        errors.append(
            "explicit iterations_completed and deadline_reached metadata are required "
            "to validate the experiment budget"
        )
        return errors

    completed = int(completed_text)
    deadline_reached = parse_program_bool(deadline_text, "deadline_reached")
    raw["proposal_evaluations_actual_total"] = (
        completed * int(job.algorithm["proposal_cost_per_iteration"])
    )
    if completed > ceiling:
        errors.append(f"iterations_completed {completed} exceeds requested ceiling {ceiling}")

    if job.budget_scheme == "equal-iterations":
        if completed != ceiling:
            errors.append(
                f"equal-iterations completed {completed} iterations, expected exactly {ceiling}"
            )
        if deadline_reached:
            errors.append("equal-iterations run unexpectedly reported deadline_reached=true")
    elif job.budget_scheme == "fixed-time":
        if completed < ceiling and not deadline_reached:
            errors.append(
                "fixed-time run stopped before its iteration ceiling without "
                "deadline_reached=true"
            )
        if "total_elapsed_ms" not in program:
            errors.append("fixed-time validation requires explicit total_elapsed_ms")
        else:
            target = float(job.time_limit_ms or job.budget_target)
            configured_tolerance = config.get("execution", {}).get("fixed_time_tolerance_ms")
            tolerance = (
                float(configured_tolerance)
                if configured_tolerance is not None
                else max(250.0, target * 0.05)
            )
            if not math.isfinite(tolerance) or tolerance < 0:
                errors.append(f"fixed_time_tolerance_ms must be finite and non-negative, got {tolerance!r}")
            else:
                delta = abs(float(program["total_elapsed_ms"]) - target)
                raw["fixed_time_tolerance_ms"] = f"{tolerance:.3f}"
                raw["fixed_time_elapsed_delta_ms"] = f"{delta:.3f}"
                if delta > tolerance:
                    errors.append(
                        f"fixed-time total_elapsed_ms {program['total_elapsed_ms']} differs from "
                        f"target {target:.3f} ms by {delta:.3f} ms (tolerance {tolerance:.3f} ms)"
                    )
    else:
        errors.append(f"unsupported formal budget scheme {job.budget_scheme!r}")
    return errors


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
    errors: list[str] = []
    timeout = process_timeout(job, config, timeout_override)
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
            errors.append(f"command exited with code {return_code}")
        else:
            program = parse_program_row(stdout)
            raw.update(
                {
                    "dimension": program["dimension"],
                    "iterations_reported": program["iterations"],
                    "init": program["init"],
                    "chains": program["chains"],
                    "threads": program["threads"],
                    "actual_threads": program.get("actual_threads", ""),
                    "reported_parallel": program["parallel"],
                    "program_algorithm": program["algorithm"],
                    "best_length": program["best_length"],
                    "final_length": program["final_length"],
                    "elapsed_ms": program["elapsed_ms"],
                    "total_elapsed_ms": program.get("total_elapsed_ms", ""),
                    "kernel_elapsed_ms": program.get("cuda_kernel_elapsed_ms", ""),
                    "accepted_moves": program["accepted_moves"],
                    "improved_moves": program["improved_moves"],
                    "iterations_completed": program.get("iterations_completed", ""),
                    "deadline_reached": program.get("deadline_reached", ""),
                    "backend_fallback": program.get("backend_fallback", ""),
                    "fallback_reason": program.get("backend_fallback_reason", ""),
                }
            )
            errors.extend(validate_program_against_job(program, job, config, raw))

            has_explicit_actual = bool(program.get("actual_backend", "").strip())
            has_explicit_fallback = "backend_fallback" in program
            has_explicit_actual_threads = bool(program.get("actual_threads", "").strip())
            if job.require_backend_match and not (
                has_explicit_actual and has_explicit_fallback and has_explicit_actual_threads
            ):
                errors.append(
                    "strict backend validation requires explicit actual_backend and "
                    "backend_fallback fields plus actual_threads; legacy/transitional CSV "
                    "is not accepted"
                )

            actual_backend = (
                normalize_backend(program["actual_backend"])
                if has_explicit_actual
                else detect_actual_backend(job.requested_backend, stdout, stderr)
            )
            raw["actual_backend"] = actual_backend
            requested_backend = normalize_backend(job.requested_backend)
            if job.require_backend_match and actual_backend != requested_backend:
                errors.append(
                    f"requested backend {requested_backend!r}, actual backend {actual_backend!r}"
                )
            fallback_reported = (
                parse_program_bool(program["backend_fallback"], "backend_fallback")
                if has_explicit_fallback
                else actual_backend != requested_backend
            )
            if job.require_backend_match and fallback_reported:
                reason = program.get("backend_fallback_reason", "").strip()
                errors.append("backend fallback was reported" + (f": {reason}" if reason else ""))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timeout_description = f"{timeout:.3f}" if timeout is not None else "unknown"
        errors.append(f"command timed out after {timeout_description} seconds")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
    finished_at = now_iso()
    error = "; ".join(errors)
    raw.update(
        {
            "status": "ok" if not error else "error",
            "error": error,
            "return_code": "" if return_code is None else return_code,
            "wall_elapsed_ms": f"{wall_elapsed_ms:.3f}",
            "started_at": started_at,
            "finished_at": finished_at,
        }
    )

    log_text = (
        f"$ {command_display(job.command)}\n\n"
        f"[metadata]\nstarted_at={started_at}\nfinished_at={finished_at}\n"
        f"wall_elapsed_ms={wall_elapsed_ms:.3f}\nreturn_code={return_code}\nerror={error}\n\n"
        f"[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n"
    )
    log_absolute.write_text(log_text, encoding="utf-8")
    manifest_job = {
        "job_id": job.job_id,
        "status": raw["status"],
        "error": error,
        "command": job.command,
        "command_display": command_display(job.command),
        "budget_scheme": job.budget_scheme,
        "budget_target": job.budget_target,
        "budget_unit": job.budget_unit,
        "algorithm_key": job.algorithm["key"],
        "instance": job.instance["name"],
        "seed": job.seed,
        "execution_order": job.execution_order,
        "chains": job.chains,
        "threads": job.threads,
        "requested_backend": normalize_backend(job.requested_backend),
        "actual_backend": raw["actual_backend"],
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_elapsed_ms": wall_elapsed_ms,
        "return_code": return_code,
        "log_file": log_relative.as_posix(),
        "log_sha256": sha256_file(log_absolute),
        "result": {key: raw.get(key, "") for key in RAW_HEADER if key not in {"command_json"}},
    }
    return raw, manifest_job


def write_checksum_sidecar(run_dir: Path, paths: Iterable[Path]) -> Path:
    entries: list[str] = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    output = run_dir / "checksums.sha256"
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return output


def safe_run_id(value: str | None, experiment_name: str) -> str:
    if value:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ValueError("--run-id must contain only letters, digits, dot, underscore, or hyphen")
        return value
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name).strip("._-")
    return f"{stamp}_{safe_name}"


def print_dry_run(config: dict[str, Any], executable: Path, jobs: list[Job]) -> None:
    summary = {
        "dry_run": True,
        "experiment_name": config["experiment_name"],
        "executable": str(executable),
        "job_count": len(jobs),
        "budgets": sorted({job.budget_scheme for job in jobs}),
        "instances": sorted({str(job.instance["name"]) for job in jobs}),
        "algorithms": sorted({str(job.algorithm["key"]) for job in jobs}),
        "seeds": sorted({job.seed for job in jobs}),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    for job in jobs:
        print(f"[dry-run] {job.job_id}: {command_display(job.command)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    try:
        config, config_bytes = load_config(config_path)
        executable = find_executable(config, args.executable, args.allow_auto_executable)
        jobs = build_jobs(config, args, executable)
        if not jobs:
            raise ValueError("selected experiment matrix is empty")
        if args.dry_run:
            print_dry_run(config, executable, jobs)
            return 0

        output_root = resolve_root_path(args.output_root or config.get("output_root", "results/fair_experiments"))
        run_id = safe_run_id(args.run_id, str(config["experiment_name"]))
        run_dir = output_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        (run_dir / "logs").mkdir(parents=True)

        snapshot_path = run_dir / "config.snapshot.json"
        snapshot_path.write_bytes(config_bytes)
        config_hash = sha256_bytes(config_bytes)
        executable_hash = sha256_file(executable)
        input_hashes = {
            str(instance["name"]): sha256_file(resolve_root_path(instance["path"]))
            for instance in config["instances"]
            if any(job.instance["name"] == instance["name"] for job in jobs)
        }
        git_state = collect_git_state()
        raw_path = run_dir / "raw.csv"
        manifest_path = run_dir / "manifest.json"
        started_at = now_iso()
        environment = collect_environment(executable)
        environment_identity = {
            key: value for key, value in environment.items() if key != "captured_at"
        }
        environment_hash = sha256_bytes(normalized_json_bytes(environment_identity))
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
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "git": git_state,
            "environment": environment,
            "environment_sha256": environment_hash,
            "executable": {"path": str(executable), "sha256": executable_hash},
            "inputs": {
                name: {
                    "path": str(resolve_root_path(next(i["path"] for i in config["instances"] if i["name"] == name))),
                    "sha256": digest,
                }
                for name, digest in input_hashes.items()
            },
            "matrix": {
                "job_count": len(jobs),
                "budgets": sorted({job.budget_scheme for job in jobs}),
                "instances": sorted({str(job.instance["name"]) for job in jobs}),
                "algorithms": sorted({str(job.algorithm["key"]) for job in jobs}),
                "seeds": sorted({job.seed for job in jobs}),
            },
            "jobs": [],
            "artifacts": {},
        }
        rows: list[dict[str, Any]] = []
        write_raw_csv(raw_path, rows)
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
            write_raw_csv(raw_path, rows)
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
            "config_snapshot": {"path": snapshot_path.name, "sha256": sha256_file(snapshot_path)},
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
