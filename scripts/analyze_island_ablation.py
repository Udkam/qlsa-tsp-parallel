#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze paired island-migration ablations with fail-closed comparability checks.

Formal analysis requires one independent, ring, and global observation for the
same seeds at every migration interval.  Rows are partitioned by a condition
fingerprint that includes the build, input, machine, budget, worker layout, and
backend; observations from different conditions can therefore never be pooled.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOPOLOGIES = ("independent", "ring", "global")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
INTEGER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)")

SUMMARY_HEADER = [
    "condition_id",
    "algorithm_key",
    "instance",
    "topology",
    "migration_interval",
    "bks",
    "n",
    "seeds",
    "mean_best_length",
    "median_best_length",
    "std_best_length",
    "min_best_length",
    "max_best_length",
    "mean_bks_gap_pct",
    "median_bks_gap_pct",
    "std_bks_gap_pct",
    "mean_runtime_ms",
    "median_runtime_ms",
    "std_runtime_ms",
    "mean_iterations_completed",
    "migration_attempts_total",
    "migrations_adopted_total",
    "migration_adoption_rate",
]

PAIRED_SEED_HEADER = [
    "condition_id",
    "algorithm_key",
    "instance",
    "topology",
    "migration_interval",
    "seed",
    "bks",
    "independent_best_length",
    "migration_best_length",
    "best_length_difference",
    "independent_bks_gap_pct",
    "migration_bks_gap_pct",
    "bks_gap_difference_pct",
    "independent_runtime_ms",
    "migration_runtime_ms",
    "runtime_difference_ms",
    "migration_attempts",
    "migrations_adopted",
    "migration_adoption_rate",
]

PAIRED_SUMMARY_HEADER = [
    "condition_id",
    "algorithm_key",
    "instance",
    "topology",
    "migration_interval",
    "bks",
    "n_migration_rows",
    "n_pairs",
    "missing_independent_pairs",
    "migration_wins",
    "ties",
    "migration_losses",
    "exact_sign_test_p_two_sided",
    "exact_sign_test_p_holm",
    "sign_reject_holm_0_05",
    "mean_best_length_difference",
    "median_best_length_difference",
    "std_best_length_difference",
    "mean_bks_gap_difference_pct",
    "mean_runtime_difference_ms",
    "migration_attempts_total",
    "migrations_adopted_total",
    "migration_adoption_rate",
]

REQUIRED_RAW_FIELDS = {
    "status",
    "algorithm_key",
    "instance",
    "bks",
    "seed",
    "topology",
    "migration_interval",
    "requested_backend",
    "reported_requested_backend",
    "actual_backend",
    "require_backend_match",
    "backend_fallback",
    "threads",
    "actual_threads",
    "best_length",
    "elapsed_ms",
    "iterations_reported",
    "iterations_completed",
    "migration_attempts",
    "migrations_adopted",
}

PROVENANCE_FIELDS = (
    "config_sha256",
    "input_sha256",
    "executable_sha256",
    "environment_sha256",
)

# Topology, interval, seed, and algorithm are deliberately excluded: they are
# treatments/blocks compared within one condition, not condition identity.
CONDITION_FIELDS = (
    "config_sha256",
    "input_sha256",
    "executable_sha256",
    "environment_sha256",
    "iterations",
    "chains",
    "threads",
    "actual_threads",
    "init",
    "requested_backend",
    "actual_backend",
    "instance",
    "bks",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="+",
        type=Path,
        help="Run directory/directories or raw CSV file(s).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-incomplete-pairs",
        action="store_true",
        help=(
            "Exploratory only: use the seed intersection shared by independent, ring, "
            "and global instead of rejecting incomplete blocks."
        ),
    )
    return parser.parse_args(argv)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


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


def parse_bool(value: str, label: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"{label} must be exactly true or false, got {value!r}")


def as_int(row: dict[str, str], field: str) -> int:
    raw = row.get(field, "")
    value = raw.strip()
    if not INTEGER_RE.fullmatch(value):
        raise ValueError(f"invalid integer {field}={raw!r}")
    return int(value)


def as_float(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"invalid float {field}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite float {field}={raw!r}")
    return value


def require_nonempty(row: dict[str, str], field: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def runtime_ms(row: dict[str, str]) -> float:
    for field in ("total_elapsed_ms", "elapsed_ms", "wall_elapsed_ms"):
        if row.get(field, "").strip():
            value = as_float(row, field)
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            return value
    raise ValueError("row has no runtime field")


def normalized_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_condition(row: dict[str, str]) -> dict[str, str]:
    iterations = row.get("_iterations", "").strip()
    if not iterations:
        iterations = row.get("iterations_requested", "").strip() or row.get(
            "iterations_reported", ""
        ).strip()
    chains = row.get("_chains", "").strip() or row.get("chains", "").strip()
    init = row.get("_init", "").strip() or row.get("init", "").strip()
    values = {
        "config_sha256": row.get("config_sha256", "").strip().lower(),
        "input_sha256": row.get("input_sha256", "").strip().lower(),
        "executable_sha256": row.get("executable_sha256", "").strip().lower(),
        "environment_sha256": row.get("environment_sha256", "").strip().lower(),
        "iterations": iterations,
        "chains": chains,
        "threads": row.get("threads", "").strip(),
        "actual_threads": row.get("actual_threads", "").strip(),
        "init": init,
        "requested_backend": normalize_backend(row.get("requested_backend", "")),
        "actual_backend": normalize_backend(row.get("actual_backend", "")),
        "instance": row.get("instance", "").strip(),
        "bks": row.get("bks", "").strip(),
    }
    return {field: values[field] for field in CONDITION_FIELDS}


def condition_id_for_row(row: dict[str, str]) -> str:
    encoded = json.dumps(
        normalized_condition(row),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def row_condition_id(row: dict[str, str]) -> str:
    return row.get("_condition_id", "") or condition_id_for_row(row)


def resolve_input(path: Path) -> tuple[Path, Path, bool]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    is_run_directory = resolved.is_dir()
    raw_path = resolved / "raw.csv" if is_run_directory else resolved
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw CSV not found: {raw_path}")
    return raw_path, raw_path.parent, is_run_directory


def _normalize_bundle_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    raw = value.strip()
    if "\\" in raw:
        raise ValueError(f"{label} must use canonical forward slashes: {raw!r}")
    relative_path = PurePosixPath(raw)
    if (
        relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or ":" in raw
        or relative_path.as_posix() != raw
    ):
        raise ValueError(f"{label} contains unsafe path {raw!r}")
    return raw


def _manifest_log_digests(manifest: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, str]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("manifest jobs must be an array")
    if not jobs:
        raise ValueError("complete run manifest must contain at least one job log")
    declared_logs = artifacts.get("logs")
    if not isinstance(declared_logs, list):
        raise ValueError("manifest artifacts.logs must be an array")

    def collect(
        entries: list[Any], path_key: str, hash_key: str, label: str
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{label}[{index}] must be an object")
            relative = _normalize_bundle_relative(
                entry.get(path_key), f"{label}[{index}].{path_key}"
            )
            if not relative.startswith("logs/"):
                raise ValueError(f"{label}[{index}].{path_key} must be inside logs/")
            digest = str(entry.get(hash_key, "")).strip().lower()
            if SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label}[{index}].{hash_key} is not a SHA-256 digest")
            if relative in result:
                raise ValueError(f"{label} contains duplicate log path {relative!r}")
            result[relative] = digest
        return result

    job_logs = collect(jobs, "log_file", "log_sha256", "manifest jobs")
    artifact_logs = collect(
        declared_logs, "path", "sha256", "manifest artifacts.logs"
    )
    if job_logs != artifact_logs:
        raise ValueError(
            "manifest jobs log declarations do not match manifest artifacts.logs"
        )
    return job_logs


def _read_checksum_sidecar(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"{path} line {line_number} is not a SHA-256 sidecar entry")
        relative = _normalize_bundle_relative(
            match.group(2), f"{path} line {line_number} checksum path"
        )
        if relative in entries:
            raise ValueError(f"{path} contains duplicate checksum entry {relative!r}")
        entries[relative] = match.group(1).lower()
    return entries


def validate_run_directory(raw_path: Path, run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    checksum_path = run_dir / "checksums.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run directory is missing manifest.json: {run_dir}")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"run directory is missing checksums.sha256: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    if manifest.get("status") != "complete":
        raise ValueError(
            f"run manifest status must be complete, got {manifest.get('status')!r}: {manifest_path}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be an object")

    raw_hash = sha256_file(raw_path)
    manifest_hash = sha256_file(manifest_path)
    raw_artifact = artifacts.get("raw_csv", {})
    if not isinstance(raw_artifact, dict):
        raise ValueError("manifest artifacts.raw_csv must be an object")
    if raw_artifact.get("path") != "raw.csv":
        raise ValueError("manifest artifacts.raw_csv.path must be raw.csv")
    if str(raw_artifact.get("sha256", "")).lower() != raw_hash:
        raise ValueError("raw.csv SHA-256 does not match manifest artifacts.raw_csv")

    sidecar = _read_checksum_sidecar(checksum_path)
    resolved_run_dir = run_dir.resolve()
    for relative, expected_hash in sidecar.items():
        candidate = (run_dir / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            candidate.relative_to(resolved_run_dir)
        except ValueError as exc:
            raise ValueError(f"checksum path escapes run directory: {relative!r}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"checksummed run artifact is missing: {candidate}")
        if sha256_file(candidate) != expected_hash:
            raise ValueError(f"checksums.sha256 mismatch for {relative}")

    config_artifact = artifacts.get("config_snapshot", {})
    if not isinstance(config_artifact, dict):
        raise ValueError("manifest artifacts.config_snapshot must be an object")
    config_relative = str(config_artifact.get("path", ""))
    config_hash = str(config_artifact.get("sha256", "")).lower()
    if config_relative != "config.snapshot.json":
        raise ValueError("manifest artifacts.config_snapshot.path must be config.snapshot.json")
    if sidecar.get(config_relative) != config_hash:
        raise ValueError("config snapshot SHA-256 is not covered consistently by checksums.sha256")
    if config_hash != str(manifest.get("config_sha256", "")).strip().lower():
        raise ValueError("manifest config snapshot SHA-256 disagrees with config_sha256")

    log_digests = _manifest_log_digests(manifest, artifacts)
    expected_digests = {
        "config.snapshot.json": config_hash,
        "raw.csv": raw_hash,
        "manifest.json": manifest_hash,
        **log_digests,
    }
    if set(sidecar) != set(expected_digests):
        raise ValueError(
            "checksums.sha256 entries do not match the required config/raw/manifest/log set"
        )
    for relative, expected_hash in expected_digests.items():
        if sidecar[relative] != expected_hash:
            raise ValueError(f"checksums.sha256 mismatch for {relative}")

    checksums_artifact = artifacts.get("checksums")
    if not isinstance(checksums_artifact, dict):
        raise ValueError("manifest artifacts.checksums must be an object")
    if checksums_artifact.get("path") != "checksums.sha256":
        raise ValueError("manifest artifacts.checksums.path must be checksums.sha256")
    declared_covers = checksums_artifact.get("covers")
    if not isinstance(declared_covers, list):
        raise ValueError("manifest artifacts.checksums.covers must be an array")
    normalized_covers = [
        _normalize_bundle_relative(value, f"manifest checksum coverage[{index}]")
        for index, value in enumerate(declared_covers)
    ]
    if len(normalized_covers) != len(set(normalized_covers)):
        raise ValueError("manifest checksum coverage contains duplicate paths")
    if set(normalized_covers) != set(expected_digests):
        raise ValueError(
            "manifest checksum coverage does not match the required config/raw/manifest/log set"
        )

    environment_hash = str(manifest.get("environment_sha256", "")).strip().lower()
    environment = manifest.get("environment")
    derived_environment_hash = ""
    if isinstance(environment, dict):
        identity = {key: value for key, value in environment.items() if key != "captured_at"}
        derived_environment_hash = sha256_bytes(normalized_json_bytes(identity))
    if not environment_hash:
        if not derived_environment_hash:
            raise ValueError("manifest requires environment or environment_sha256")
        environment_hash = derived_environment_hash
    elif derived_environment_hash and environment_hash != derived_environment_hash:
        raise ValueError("manifest environment_sha256 does not match its environment identity")
    if not SHA256_RE.fullmatch(environment_hash):
        raise ValueError("manifest environment_sha256 is not a SHA-256 digest")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "checksums_path": checksum_path,
        "environment_sha256": environment_hash,
        "provenance_validation": "validated",
    }


def _validate_hash(value: str, field: str, required: bool) -> str:
    normalized = value.strip().lower()
    if not normalized:
        if required:
            raise ValueError(f"missing required provenance field {field}")
        return ""
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} is not a SHA-256 digest: {value!r}")
    return normalized


def _validate_manifest_row(row: dict[str, str], manifest: dict[str, Any]) -> None:
    expected_config = str(manifest.get("config_sha256", "")).strip().lower()
    expected_executable = str(manifest.get("executable", {}).get("sha256", "")).strip().lower()
    expected_inputs = manifest.get("inputs", {})
    instance = row.get("instance", "").strip()
    expected_input = ""
    if isinstance(expected_inputs, dict) and isinstance(expected_inputs.get(instance), dict):
        expected_input = str(expected_inputs[instance].get("sha256", "")).strip().lower()
    comparisons = (
        ("config_sha256", expected_config),
        ("executable_sha256", expected_executable),
        ("input_sha256", expected_input),
    )
    for field, expected in comparisons:
        if not expected:
            raise ValueError(f"manifest has no {field} provenance for instance {instance!r}")
        if row.get(field, "").strip().lower() != expected:
            raise ValueError(f"row {field} does not match validated manifest provenance")


def _modern_accounting(row: dict[str, str], provenance_required: bool) -> bool:
    return provenance_required or any(row.get(field, "").strip() for field in PROVENANCE_FIELDS)


def validate_successful_row(
    row: dict[str, str], *, provenance_required: bool, manifest: dict[str, Any] | None
) -> None:
    algorithm = require_nonempty(row, "algorithm_key")
    instance = require_nonempty(row, "instance")
    if any(character in algorithm + instance for character in "\r\n"):
        raise ValueError("algorithm_key and instance must be single-line values")

    topology = require_nonempty(row, "topology").lower()
    if topology not in EXPECTED_TOPOLOGIES:
        raise ValueError(f"unsupported topology {topology!r}")
    row["topology"] = topology
    interval = as_int(row, "migration_interval")
    if interval <= 0:
        raise ValueError("every topology, including its matched independent baseline, needs a positive interval")
    if row.get("command_migration_interval", "").strip():
        command_interval = as_int(row, "command_migration_interval")
        if command_interval != interval:
            raise ValueError("command_migration_interval must equal semantic migration_interval")

    requested = normalize_backend(require_nonempty(row, "requested_backend"))
    reported_requested = normalize_backend(require_nonempty(row, "reported_requested_backend"))
    actual = normalize_backend(require_nonempty(row, "actual_backend"))
    if requested not in {"serial", "omp", "cuda"}:
        raise ValueError(f"unsupported requested backend {requested!r}")
    if not parse_bool(row.get("require_backend_match", ""), "require_backend_match"):
        raise ValueError("successful row does not require backend matching")
    if reported_requested != requested:
        raise ValueError(
            f"reported requested backend {reported_requested!r} != runner request {requested!r}"
        )
    if requested != actual:
        raise ValueError(f"backend mismatch: requested {requested!r}, actual {actual!r}")
    if parse_bool(row.get("backend_fallback", ""), "backend_fallback"):
        raise ValueError("successful row reports backend fallback")
    if row.get("reported_parallel", "").strip():
        reported_parallel = normalize_backend(row["reported_parallel"])
        if reported_parallel != requested:
            raise ValueError(
                f"reported parallel backend {reported_parallel!r} != request {requested!r}"
            )

    requested_threads = as_int(row, "threads")
    actual_threads = as_int(row, "actual_threads")
    if requested_threads <= 0 or actual_threads != requested_threads:
        raise ValueError(
            f"thread mismatch: requested {requested_threads}, actual {actual_threads}"
        )

    bks = as_int(row, "bks")
    best = as_int(row, "best_length")
    seed = as_int(row, "seed")
    if bks <= 0 or best <= 0 or seed < 0:
        raise ValueError("BKS/best length must be positive and seed must be non-negative")
    if row.get("dimension", "").strip() and as_int(row, "dimension") <= 0:
        raise ValueError("dimension must be positive")
    if row.get("final_length", "").strip() and as_int(row, "final_length") <= 0:
        raise ValueError("final_length must be positive")

    modern = _modern_accounting(row, provenance_required)
    reported_iterations = as_int(row, "iterations_reported")
    completed_iterations = as_int(row, "iterations_completed")
    if reported_iterations <= 0 or completed_iterations <= 0:
        raise ValueError("reported/completed iterations must be positive")
    if row.get("iterations_requested", "").strip():
        requested_iterations = as_int(row, "iterations_requested")
    elif modern:
        raise ValueError("iterations_requested is required for a provenance-bearing row")
    else:
        requested_iterations = reported_iterations
    if requested_iterations <= 0 or reported_iterations != requested_iterations:
        raise ValueError(
            f"iterations mismatch: requested {requested_iterations}, reported {reported_iterations}"
        )

    if row.get("chains", "").strip():
        chains = as_int(row, "chains")
    elif not modern and completed_iterations % requested_iterations == 0:
        chains = completed_iterations // requested_iterations
    else:
        raise ValueError("chains is required to verify complete iterations")
    if chains <= 0:
        raise ValueError("chains must be positive")
    expected_completed = requested_iterations * chains
    if completed_iterations != expected_completed:
        raise ValueError(
            f"incomplete iterations: completed {completed_iterations}, expected {expected_completed}"
        )

    if row.get("deadline_reached", "").strip():
        deadline_reached = parse_bool(row["deadline_reached"], "deadline_reached")
    elif modern:
        raise ValueError("deadline_reached is required for a provenance-bearing row")
    else:
        deadline_reached = False
    if deadline_reached:
        raise ValueError("deadline_reached=true is not a complete fixed-work observation")

    if row.get("init", "").strip():
        init = require_nonempty(row, "init")
    elif modern:
        raise ValueError("init is required for a provenance-bearing row")
    else:
        init = "unknown"

    for field in ("elapsed_ms", "total_elapsed_ms", "kernel_elapsed_ms", "wall_elapsed_ms"):
        if row.get(field, "").strip() and as_float(row, field) < 0.0:
            raise ValueError(f"{field} must be non-negative")
    _ = runtime_ms(row)
    if row.get("accepted_moves", "").strip():
        accepted = as_int(row, "accepted_moves")
        if accepted < 0:
            raise ValueError("accepted_moves must be non-negative")
        if row.get("improved_moves", "").strip():
            improved = as_int(row, "improved_moves")
            if improved < 0 or improved > accepted:
                raise ValueError("improved_moves must be between zero and accepted_moves")

    attempts = as_int(row, "migration_attempts")
    adopted = as_int(row, "migrations_adopted")
    if attempts < 0 or adopted < 0 or adopted > attempts:
        raise ValueError("invalid migration counters")
    if row.get("migration_rounds", "").strip():
        rounds = as_int(row, "migration_rounds")
        if rounds < 0:
            raise ValueError("migration_rounds must be non-negative")
    elif modern:
        raise ValueError("migration_rounds is required for a provenance-bearing row")
    if topology == "independent" and adopted != 0:
        raise ValueError("independent row must not adopt migrations")
    if row.get("migration_adoption_rate", "").strip():
        reported_rate = as_float(row, "migration_adoption_rate")
        expected_rate = adopted / attempts if attempts else 0.0
        if not 0.0 <= reported_rate <= 1.0 or not math.isclose(
            reported_rate, expected_rate, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("migration_adoption_rate does not match migration counters")

    if row.get("return_code", "").strip() and as_int(row, "return_code") != 0:
        raise ValueError("status=ok row must have return_code=0")
    if row.get("error", "").strip():
        raise ValueError("status=ok row must not contain an error")

    for field in PROVENANCE_FIELDS:
        row[field] = _validate_hash(row.get(field, ""), field, provenance_required)
    if manifest is not None:
        _validate_manifest_row(row, manifest)

    row["_iterations"] = str(requested_iterations)
    row["_chains"] = str(chains)
    row["_init"] = init
    row["_condition_id"] = condition_id_for_row(row)


def read_raw_rows(
    path: Path,
    *,
    provenance_required: bool = False,
    environment_sha256: str = "",
    manifest: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("raw CSV has no header")
        missing = REQUIRED_RAW_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"raw CSV is missing fields: {sorted(missing)}")
        all_rows = list(reader)

    successful: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, int, str, int]] = set()
    for row_number, row in enumerate(all_rows, start=2):
        if row.get("status", "").strip().lower() != "ok":
            continue
        row["_source"] = str(path)
        row["_line"] = str(row_number)
        if environment_sha256:
            existing = row.get("environment_sha256", "").strip().lower()
            if existing and existing != environment_sha256:
                raise ValueError(
                    f"raw CSV row {row_number}: environment_sha256 disagrees with manifest"
                )
            row["environment_sha256"] = environment_sha256
        try:
            validate_successful_row(
                row,
                provenance_required=provenance_required,
                manifest=manifest,
            )
            key = (
                row["_condition_id"],
                row["algorithm_key"],
                row["instance"],
                as_int(row, "seed"),
                row["topology"],
                as_int(row, "migration_interval"),
            )
            if key in seen:
                raise ValueError(f"duplicate successful matrix cell {key}")
            seen.add(key)
        except ValueError as exc:
            raise ValueError(f"raw CSV row {row_number}: {exc}") from exc
        successful.append(row)
    return all_rows, successful


def read_inputs(
    inputs: Sequence[Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    resolved = [resolve_input(path) for path in inputs]
    raw_paths = [item[0] for item in resolved]
    if len(set(raw_paths)) != len(raw_paths):
        raise ValueError("the same raw CSV was supplied more than once")
    resolved.sort(key=lambda item: str(item[0]).casefold())
    provenance_required = len(resolved) > 1
    all_rows: list[dict[str, str]] = []
    successful: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []

    for raw_path, run_dir, is_run_directory in resolved:
        context: dict[str, Any] | None = None
        if is_run_directory:
            context = validate_run_directory(raw_path, run_dir)
        environment_hash = context["environment_sha256"] if context else ""
        manifest = context["manifest"] if context else None
        source_all, source_successful = read_raw_rows(
            raw_path,
            provenance_required=provenance_required or is_run_directory,
            environment_sha256=environment_hash,
            manifest=manifest,
        )
        all_rows.extend(source_all)
        successful.extend(source_successful)
        metadata.append(
            {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "input_kind": "run-directory" if is_run_directory else "direct-csv",
                "provenance_validation": "validated" if context else "unavailable",
                **(
                    {
                        "manifest": str(context["manifest_path"]),
                        "manifest_sha256": sha256_file(context["manifest_path"]),
                        "checksums": str(context["checksums_path"]),
                    }
                    if context
                    else {}
                ),
            }
        )

    global_seen: dict[tuple[str, str, str, int, str, int], dict[str, str]] = {}
    for row in successful:
        key = (
            row["_condition_id"],
            row["algorithm_key"],
            row["instance"],
            as_int(row, "seed"),
            row["topology"],
            as_int(row, "migration_interval"),
        )
        if key in global_seen:
            previous = global_seen[key]
            raise ValueError(
                f"duplicate matrix cell across inputs {key}: "
                f"{previous['_source']}:{previous['_line']} and {row['_source']}:{row['_line']}"
            )
        global_seen[key] = row

    successful.sort(
        key=lambda row: (
            row["_condition_id"],
            row["algorithm_key"],
            row["instance"],
            as_int(row, "migration_interval"),
            row["topology"],
            as_int(row, "seed"),
        )
    )
    return all_rows, successful, metadata


def finite_mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def format_number(value: float) -> str:
    return f"{value:.12g}"


def bks_gap_pct(best_length: int, bks: int) -> float:
    return (best_length - bks) * 100.0 / bks


def group_summary(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row_condition_id(row),
                row["algorithm_key"],
                row["instance"],
                row["topology"],
                as_int(row, "migration_interval"),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        condition_id, algorithm, instance, topology, interval = key
        group = sorted(groups[key], key=lambda row: as_int(row, "seed"))
        bks_values = {as_int(row, "bks") for row in group}
        if len(bks_values) != 1:
            raise ValueError(f"inconsistent BKS values for group {key}")
        bks = next(iter(bks_values))
        best = [float(as_int(row, "best_length")) for row in group]
        gaps = [bks_gap_pct(as_int(row, "best_length"), bks) for row in group]
        runtimes = [runtime_ms(row) for row in group]
        iterations = [float(as_int(row, "iterations_completed")) for row in group]
        attempts = sum(as_int(row, "migration_attempts") for row in group)
        adopted = sum(as_int(row, "migrations_adopted") for row in group)
        output.append(
            {
                "condition_id": condition_id,
                "algorithm_key": algorithm,
                "instance": instance,
                "topology": topology,
                "migration_interval": interval,
                "bks": bks,
                "n": len(group),
                "seeds": json.dumps(
                    [as_int(row, "seed") for row in group], separators=(",", ":")
                ),
                "mean_best_length": format_number(finite_mean(best)),
                "median_best_length": format_number(statistics.median(best)),
                "std_best_length": format_number(sample_std(best)),
                "min_best_length": format_number(min(best)),
                "max_best_length": format_number(max(best)),
                "mean_bks_gap_pct": format_number(finite_mean(gaps)),
                "median_bks_gap_pct": format_number(statistics.median(gaps)),
                "std_bks_gap_pct": format_number(sample_std(gaps)),
                "mean_runtime_ms": format_number(finite_mean(runtimes)),
                "median_runtime_ms": format_number(statistics.median(runtimes)),
                "std_runtime_ms": format_number(sample_std(runtimes)),
                "mean_iterations_completed": format_number(finite_mean(iterations)),
                "migration_attempts_total": attempts,
                "migrations_adopted_total": adopted,
                "migration_adoption_rate": format_number(adopted / attempts if attempts else 0.0),
            }
        )
    return output


def exact_sign_test_two_sided(wins: int, losses: int) -> float:
    """Exact two-sided Binomial(0.5) sign test, excluding ties."""
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, probability)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("Holm correction requires finite p-values in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


BlockKey = tuple[str, str, str, int]
SettingMaps = dict[str, dict[int, dict[str, str]]]


def rows_by_complete_block(rows: Sequence[dict[str, str]]) -> dict[BlockKey, SettingMaps]:
    grouped: dict[BlockKey, SettingMaps] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (
            row_condition_id(row),
            row["algorithm_key"],
            row["instance"],
            as_int(row, "migration_interval"),
        )
        seed = as_int(row, "seed")
        topology = row["topology"]
        if seed in grouped[key][topology]:
            raise ValueError(f"duplicate topology/seed observation in block {key}/{topology}/{seed}")
        grouped[key][topology][seed] = row
    return grouped


def pairing_issues(rows: Sequence[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    for key, by_topology in sorted(rows_by_complete_block(rows).items()):
        missing = [topology for topology in EXPECTED_TOPOLOGIES if topology not in by_topology]
        if missing:
            issues.append(f"{key}: missing topologies {missing}")
            continue
        seed_sets = {topology: set(by_topology[topology]) for topology in EXPECTED_TOPOLOGIES}
        reference = seed_sets["independent"]
        mismatched = {
            topology: sorted(seeds ^ reference)
            for topology, seeds in seed_sets.items()
            if seeds != reference
        }
        if mismatched:
            issues.append(f"{key}: paired seed mismatch {mismatched}")
    return issues


def paired_analysis(
    rows: Sequence[dict[str, str]], *, allow_incomplete_pairs: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = rows_by_complete_block(rows)
    issues = pairing_issues(rows)
    if issues and not allow_incomplete_pairs:
        raise ValueError(
            "complete independent/ring/global seed blocks are required by default: "
            + "; ".join(issues)
            + "; pass --allow-incomplete-pairs only for exploratory analysis"
        )

    paired_seed_rows: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    for block_key in sorted(grouped):
        condition_id, algorithm, instance, interval = block_key
        by_topology = grouped[block_key]
        if "independent" not in by_topology:
            continue
        available_topologies = [
            topology for topology in ("ring", "global") if topology in by_topology
        ]
        if not available_topologies:
            continue
        participating = ["independent", *available_topologies]
        complete_seeds = set.intersection(*(set(by_topology[item]) for item in participating))
        common_seeds = sorted(complete_seeds)
        independent_rows = by_topology["independent"]

        for topology in available_topologies:
            migrated_rows = by_topology[topology]
            differences: list[float] = []
            gap_differences: list[float] = []
            runtime_differences: list[float] = []
            wins = ties = losses = 0
            attempts_total = 0
            adopted_total = 0
            bks_values = {
                as_int(row, "bks") for row in [*independent_rows.values(), *migrated_rows.values()]
            }
            if len(bks_values) != 1:
                raise ValueError(f"inconsistent BKS values for paired block {block_key}")
            bks = next(iter(bks_values))

            for seed in common_seeds:
                independent = independent_rows[seed]
                migration_row = migrated_rows[seed]
                independent_best = as_int(independent, "best_length")
                migration_best = as_int(migration_row, "best_length")
                difference = float(migration_best - independent_best)
                independent_gap = bks_gap_pct(independent_best, bks)
                migration_gap = bks_gap_pct(migration_best, bks)
                gap_difference = migration_gap - independent_gap
                independent_runtime = runtime_ms(independent)
                migration_runtime = runtime_ms(migration_row)
                runtime_difference = migration_runtime - independent_runtime
                attempts = as_int(migration_row, "migration_attempts")
                adopted = as_int(migration_row, "migrations_adopted")
                differences.append(difference)
                gap_differences.append(gap_difference)
                runtime_differences.append(runtime_difference)
                attempts_total += attempts
                adopted_total += adopted
                if difference < 0:
                    wins += 1
                elif difference > 0:
                    losses += 1
                else:
                    ties += 1
                paired_seed_rows.append(
                    {
                        "condition_id": condition_id,
                        "algorithm_key": algorithm,
                        "instance": instance,
                        "topology": topology,
                        "migration_interval": interval,
                        "seed": seed,
                        "bks": bks,
                        "independent_best_length": independent_best,
                        "migration_best_length": migration_best,
                        "best_length_difference": format_number(difference),
                        "independent_bks_gap_pct": format_number(independent_gap),
                        "migration_bks_gap_pct": format_number(migration_gap),
                        "bks_gap_difference_pct": format_number(gap_difference),
                        "independent_runtime_ms": format_number(independent_runtime),
                        "migration_runtime_ms": format_number(migration_runtime),
                        "runtime_difference_ms": format_number(runtime_difference),
                        "migration_attempts": attempts,
                        "migrations_adopted": adopted,
                        "migration_adoption_rate": format_number(
                            adopted / attempts if attempts else 0.0
                        ),
                    }
                )

            paired_summaries.append(
                {
                    "condition_id": condition_id,
                    "algorithm_key": algorithm,
                    "instance": instance,
                    "topology": topology,
                    "migration_interval": interval,
                    "bks": bks,
                    "n_migration_rows": len(migrated_rows),
                    "n_pairs": len(differences),
                    "missing_independent_pairs": len(set(migrated_rows) - set(independent_rows)),
                    "migration_wins": wins,
                    "ties": ties,
                    "migration_losses": losses,
                    "exact_sign_test_p_two_sided": format_number(
                        exact_sign_test_two_sided(wins, losses)
                    ),
                    "mean_best_length_difference": (
                        format_number(finite_mean(differences)) if differences else ""
                    ),
                    "median_best_length_difference": (
                        format_number(statistics.median(differences)) if differences else ""
                    ),
                    "std_best_length_difference": (
                        format_number(sample_std(differences)) if differences else ""
                    ),
                    "mean_bks_gap_difference_pct": (
                        format_number(finite_mean(gap_differences)) if differences else ""
                    ),
                    "mean_runtime_difference_ms": (
                        format_number(finite_mean(runtime_differences)) if differences else ""
                    ),
                    "migration_attempts_total": attempts_total,
                    "migrations_adopted_total": adopted_total,
                    "migration_adoption_rate": format_number(
                        adopted_total / attempts_total if attempts_total else 0.0
                    ),
                }
            )

    adjustment_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, summary in enumerate(paired_summaries):
        adjustment_groups[
            (summary["condition_id"], summary["algorithm_key"], summary["instance"])
        ].append(index)
    for indices in adjustment_groups.values():
        adjusted = holm_adjust(
            [float(paired_summaries[index]["exact_sign_test_p_two_sided"]) for index in indices]
        )
        for index, value in zip(indices, adjusted):
            paired_summaries[index]["exact_sign_test_p_holm"] = format_number(value)
            paired_summaries[index]["sign_reject_holm_0_05"] = str(value <= 0.05).lower()

    paired_seed_rows.sort(
        key=lambda row: (
            row["condition_id"],
            row["algorithm_key"],
            row["instance"],
            row["migration_interval"],
            row["topology"],
            row["seed"],
        )
    )
    paired_summaries.sort(
        key=lambda row: (
            row["condition_id"],
            row["algorithm_key"],
            row["instance"],
            row["migration_interval"],
            row["topology"],
        )
    )
    return paired_seed_rows, paired_summaries


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(normalized_json_bytes(value))
    os.replace(temporary, path)


def write_checksums(output_dir: Path, paths: Iterable[Path]) -> Path:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    output = output_dir / "analysis_checksums.sha256"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def condition_manifest(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    conditions: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row_condition_id(row)
        components = normalized_condition(row)
        previous = conditions.get(identifier)
        if previous is not None and previous != components:
            raise ValueError(f"condition ID collision for {identifier}")
        conditions[identifier] = components
    return [
        {"condition_id": identifier, "components": conditions[identifier]}
        for identifier in sorted(conditions)
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        all_rows, successful, input_metadata = read_inputs(args.input)
        if not successful:
            raise ValueError("raw CSV contains no successful rows")
        issues = pairing_issues(successful)
        if issues and not args.allow_incomplete_pairs:
            raise ValueError(
                "complete independent/ring/global seed blocks are required by default: "
                + "; ".join(issues)
                + "; pass --allow-incomplete-pairs only for exploratory analysis"
            )

        if args.output_dir:
            output_dir = args.output_dir.resolve()
        elif len(args.input) == 1:
            _, run_dir, _ = resolve_input(args.input[0])
            output_dir = run_dir / "analysis"
        else:
            stamp = dt.datetime.now().astimezone().strftime("analysis_%Y%m%d_%H%M%S_%z")
            output_dir = ROOT / "results" / "island_ablation" / stamp
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = group_summary(successful)
        paired_seed_rows, paired_summaries = paired_analysis(
            successful, allow_incomplete_pairs=args.allow_incomplete_pairs
        )
        summary_path = output_dir / "summary.csv"
        paired_seed_path = output_dir / "paired_seed_differences.csv"
        paired_summary_path = output_dir / "paired_comparisons.csv"
        metadata_path = output_dir / "analysis.json"
        atomic_write_csv(summary_path, SUMMARY_HEADER, summaries)
        atomic_write_csv(paired_seed_path, PAIRED_SEED_HEADER, paired_seed_rows)
        atomic_write_csv(paired_summary_path, PAIRED_SUMMARY_HEADER, paired_summaries)
        metadata = {
            "schema_version": 2,
            "created_at": now_iso(),
            "inputs": input_metadata,
            "row_counts": {
                "raw": len(all_rows),
                "successful": len(successful),
                "excluded_non_ok": len(all_rows) - len(successful),
                "summary_groups": len(summaries),
                "paired_seed_differences": len(paired_seed_rows),
                "paired_comparisons": len(paired_summaries),
            },
            "condition_fields": list(CONDITION_FIELDS),
            "conditions": condition_manifest(successful),
            "pairing_mode": (
                "complete-common-block-intersection-exploratory"
                if args.allow_incomplete_pairs
                else "complete-independent-ring-global-blocks-required"
            ),
            "pairing_issues": issues,
            "sign_convention": (
                "paired differences are migration minus matched independent at the same interval; "
                "negative best-length differences are migration wins; ties are excluded from "
                "the exact two-sided binomial sign test"
            ),
            "multiple_comparisons": (
                "Holm correction across all topology/interval sign tests within each "
                "condition_id, algorithm_key, and instance"
            ),
            "runtime_field_priority": ["total_elapsed_ms", "elapsed_ms", "wall_elapsed_ms"],
            "artifacts": {
                "summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)},
                "paired_seed_differences": {
                    "path": paired_seed_path.name,
                    "sha256": sha256_file(paired_seed_path),
                },
                "paired_comparisons": {
                    "path": paired_summary_path.name,
                    "sha256": sha256_file(paired_summary_path),
                },
            },
        }
        atomic_write_json(metadata_path, metadata)
        checksums = write_checksums(
            output_dir, [summary_path, paired_seed_path, paired_summary_path, metadata_path]
        )
        for issue in issues:
            print(f"[warning] {issue}", file=sys.stderr)
        print(f"[ok] summary: {summary_path}")
        print(f"[ok] paired seed differences: {paired_seed_path}")
        print(f"[ok] paired comparisons: {paired_summary_path}")
        print(f"[ok] analysis metadata: {metadata_path}")
        print(f"[ok] checksums: {checksums}")
        return 0
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
