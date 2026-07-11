#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze paired-seed fair experiments using only the Python standard library.

Outputs:
  * summary.csv: per-algorithm quality/runtime summaries and bootstrap CIs
  * pairwise.csv: paired differences, Wilcoxon, sign tests, and Holm correction
  * friedman.csv: four-algorithm Friedman tests over complete seed blocks
  * analysis_manifest.json and checksums.sha256 for auditability

Lower best_length is always treated as better.  Pairwise differences use
``algorithm_a - algorithm_b``; a negative value therefore favors algorithm A.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ORDER = ("sa", "current", "paper", "paper-sb")
REQUIRED_COLUMNS = {
    "status",
    "budget_scheme",
    "algorithm_key",
    "instance",
    "bks",
    "seed",
    "best_length",
    "elapsed_ms",
    "requested_backend",
    "actual_backend",
}

SUMMARY_HEADER = [
    "condition_id",
    "budget_scheme",
    "instance",
    "requested_backend",
    "actual_backend",
    "algorithm_key",
    "runs",
    "unique_seeds",
    "bks",
    "best_length_min",
    "best_length_mean",
    "best_length_median",
    "best_length_std",
    "best_length_mean_ci_low",
    "best_length_mean_ci_high",
    "gap_percent_mean",
    "gap_percent_median",
    "gap_percent_mean_ci_low",
    "gap_percent_mean_ci_high",
    "bks_hit_count",
    "bks_hit_rate",
    "timing_source",
    "elapsed_ms_mean",
    "elapsed_ms_median",
    "elapsed_ms_std",
    "elapsed_ms_mean_ci_low",
    "elapsed_ms_mean_ci_high",
    "iterations_completed_samples",
    "iterations_completed_mean",
    "iterations_completed_median",
    "iterations_completed_std",
    "deadline_reached_samples",
    "deadline_reached_count",
    "deadline_reached_rate",
    "bootstrap_samples",
    "confidence_level",
]

PAIRWISE_HEADER = [
    "condition_id",
    "budget_scheme",
    "instance",
    "requested_backend",
    "actual_backend",
    "algorithm_a",
    "algorithm_b",
    "seeds_a",
    "seeds_b",
    "paired_seeds",
    "paired_seed_values_json",
    "mean_difference_a_minus_b",
    "median_difference_a_minus_b",
    "std_difference_a_minus_b",
    "mean_difference_ci_low",
    "mean_difference_ci_high",
    "wins_a",
    "wins_b",
    "ties",
    "wilcoxon_nonzero_pairs",
    "wilcoxon_w_plus",
    "wilcoxon_w_minus",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
    "wilcoxon_method",
    "wilcoxon_p_holm",
    "wilcoxon_reject_holm_0_05",
    "sign_non_tied_pairs",
    "sign_p_value",
    "sign_p_holm",
    "sign_reject_holm_0_05",
    "bootstrap_samples",
    "confidence_level",
]

FRIEDMAN_HEADER = [
    "condition_id",
    "budget_scheme",
    "instance",
    "requested_backend",
    "actual_backend",
    "algorithms_json",
    "union_seed_count",
    "complete_seed_blocks",
    "complete_seed_values_json",
    "statistic",
    "degrees_of_freedom",
    "p_value",
    "tie_correction",
    "mean_ranks_json",
    "status",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Raw CSV file(s) or run directories.")
    parser.add_argument("--output-dir", type=Path, help="New output directory (default: timestamped directory).")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHM_ORDER))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=20260711)
    pairing = parser.add_mutually_exclusive_group()
    pairing.add_argument(
        "--allow-incomplete-pairs",
        action="store_true",
        help="Allow degraded pairwise analysis on seed intersections (not recommended for formal results).",
    )
    pairing.add_argument(
        "--strict-pairing",
        action="store_true",
        help="Deprecated compatibility flag; complete paired blocks are now required by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return digest


def resolve_bundle_member(run_dir: Path, member: Any, label: str) -> Path:
    raw_member = str(member or "").strip()
    if not raw_member:
        raise ValueError(f"{label} path is missing")
    relative = Path(raw_member)
    if relative.is_absolute():
        raise ValueError(f"{label} path must be relative to the run directory: {raw_member}")
    resolved = (run_dir / relative).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the run directory: {raw_member}") from exc
    return resolved


def verify_declared_file(path: Path, expected_sha256: Any, label: str) -> dict[str, Any]:
    expected = require_sha256(expected_sha256, f"{label} sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "status": "verified",
    }


def collect_declared_logs(
    run_dir: Path,
    declarations: Any,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(declarations, list):
        raise ValueError(f"run manifest {label} must be a list")

    logs: dict[str, dict[str, Any]] = {}
    for index, declaration in enumerate(declarations):
        item_label = f"{label}[{index}]"
        if not isinstance(declaration, dict):
            raise ValueError(f"run manifest {item_label} must be an object")
        path = resolve_bundle_member(run_dir, declaration.get("path"), item_label)
        relative_path = path.relative_to(run_dir).as_posix()
        digest = require_sha256(declaration.get("sha256"), f"{item_label} sha256")
        if relative_path in logs:
            raise ValueError(f"run manifest {label} contains duplicate log {relative_path}")
        logs[relative_path] = {"path": path, "sha256": digest}
    return logs


def collect_job_logs(run_dir: Path, jobs: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(jobs, list):
        raise ValueError("run manifest jobs must be a list")
    if not jobs:
        raise ValueError("complete run manifest must contain at least one job log")

    declarations: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"run manifest jobs[{index}] must be an object")
        declarations.append(
            {
                "path": job.get("log_file"),
                "sha256": job.get("log_sha256"),
            }
        )
    return collect_declared_logs(run_dir, declarations, "jobs log declarations")


def collect_declared_coverage(run_dir: Path, covers: Any) -> set[str]:
    if not isinstance(covers, list):
        raise ValueError("run manifest artifacts.checksums.covers must be a list")

    names: set[str] = set()
    for index, member in enumerate(covers):
        path = resolve_bundle_member(
            run_dir,
            member,
            f"artifacts.checksums.covers[{index}]",
        )
        relative_path = path.relative_to(run_dir).as_posix()
        if relative_path in names:
            raise ValueError(
                "run manifest artifacts.checksums.covers contains duplicate file "
                f"{relative_path}"
            )
        names.add(relative_path)
    return names


def read_checksum_sidecar(run_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    checksum_path = run_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"run checksum sidecar not found: {checksum_path}")

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9A-Fa-f]{64})[ \t]+\*?(.+)", line)
        if match is None:
            raise ValueError(
                f"invalid checksum entry at {checksum_path}:{line_number}: {raw_line!r}"
            )
        expected = match.group(1).lower()
        declared_name = match.group(2).strip()
        member_path = resolve_bundle_member(
            run_dir,
            declared_name,
            f"checksum entry at line {line_number}",
        )
        relative_name = member_path.relative_to(run_dir).as_posix()
        if relative_name in seen:
            raise ValueError(f"duplicate checksum entry for {relative_name} in {checksum_path}")
        seen.add(relative_name)
        if not member_path.is_file():
            raise FileNotFoundError(
                f"checksums.sha256 lists a missing file: {member_path}"
            )
        actual = sha256_file(member_path)
        if actual != expected:
            raise ValueError(
                f"checksums.sha256 mismatch for {member_path}: expected {expected}, got {actual}"
            )
        entries.append(
            {
                "path": str(member_path),
                "relative_path": relative_name,
                "sha256": actual,
            }
        )
    if not entries:
        raise ValueError(f"run checksum sidecar has no entries: {checksum_path}")
    return checksum_path, entries


def verify_run_directory(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid run manifest JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"run manifest must contain a JSON object: {manifest_path}")
    manifest_status = str(manifest.get("status", "")).strip()
    if manifest_status != "complete":
        raise ValueError(
            f"run manifest status must be 'complete', got {manifest_status!r}: {manifest_path}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"run manifest artifacts must be an object: {manifest_path}")
    raw_declaration = artifacts.get("raw_csv")
    if not isinstance(raw_declaration, dict):
        raise ValueError(f"run manifest artifacts.raw_csv must be an object: {manifest_path}")
    raw_path = resolve_bundle_member(
        run_dir,
        raw_declaration.get("path"),
        "artifacts.raw_csv",
    )
    raw_report = verify_declared_file(
        raw_path,
        raw_declaration.get("sha256"),
        "artifacts.raw_csv",
    )

    top_snapshot = manifest.get("config_snapshot")
    top_config_hash = manifest.get("config_sha256")
    artifact_snapshot = artifacts.get("config_snapshot")
    if top_config_hash is not None and top_snapshot is None:
        raise ValueError(
            f"run manifest declares config_sha256 without config_snapshot: {manifest_path}"
        )
    if artifact_snapshot is not None and not isinstance(artifact_snapshot, dict):
        raise ValueError(
            f"run manifest artifacts.config_snapshot must be an object: {manifest_path}"
        )

    snapshot_declarations: list[tuple[Path, Any, str]] = []
    if top_snapshot is not None:
        if top_config_hash is None:
            raise ValueError(
                f"run manifest config_snapshot is missing config_sha256: {manifest_path}"
            )
        snapshot_declarations.append(
            (
                resolve_bundle_member(run_dir, top_snapshot, "config_snapshot"),
                top_config_hash,
                "config_snapshot",
            )
        )
    if isinstance(artifact_snapshot, dict):
        snapshot_declarations.append(
            (
                resolve_bundle_member(
                    run_dir,
                    artifact_snapshot.get("path"),
                    "artifacts.config_snapshot",
                ),
                artifact_snapshot.get("sha256"),
                "artifacts.config_snapshot",
            )
        )

    if not snapshot_declarations:
        raise ValueError(f"run manifest does not declare a config snapshot: {manifest_path}")
    canonical_path, canonical_hash, _ = snapshot_declarations[0]
    canonical_hash = require_sha256(canonical_hash, "config snapshot sha256")
    for declared_path, declared_hash, label in snapshot_declarations:
        normalized_hash = require_sha256(declared_hash, f"{label} sha256")
        if declared_path != canonical_path or normalized_hash != canonical_hash:
            raise ValueError(
                "run manifest has inconsistent config snapshot declarations: "
                f"{manifest_path}"
            )
    config_report = verify_declared_file(
        canonical_path,
        canonical_hash,
        "config snapshot",
    )

    job_logs = collect_job_logs(run_dir, manifest.get("jobs"))
    artifact_logs = collect_declared_logs(
        run_dir,
        artifacts.get("logs"),
        "artifacts.logs",
    )
    job_log_hashes = {name: item["sha256"] for name, item in job_logs.items()}
    artifact_log_hashes = {
        name: item["sha256"] for name, item in artifact_logs.items()
    }
    if job_log_hashes != artifact_log_hashes:
        raise ValueError(
            "run manifest jobs log declarations do not match artifacts.logs"
        )
    log_reports = [
        verify_declared_file(item["path"], item["sha256"], f"log {name}")
        for name, item in sorted(job_logs.items())
    ]

    checksum_path, checksum_entries = read_checksum_sidecar(run_dir)
    checksum_by_name = {
        entry["relative_path"]: entry for entry in checksum_entries
    }
    checksum_names = set(checksum_by_name)
    required_coverage = {
        manifest_path.relative_to(run_dir).as_posix(),
        raw_path.relative_to(run_dir).as_posix(),
        canonical_path.relative_to(run_dir).as_posix(),
        *job_logs,
    }

    checksum_declaration = artifacts.get("checksums")
    if not isinstance(checksum_declaration, dict):
        raise ValueError("run manifest artifacts.checksums must be an object")
    declared_checksum_path = resolve_bundle_member(
        run_dir,
        checksum_declaration.get("path"),
        "artifacts.checksums",
    )
    if declared_checksum_path != checksum_path.resolve():
        raise ValueError(
            "run manifest artifacts.checksums.path does not identify checksums.sha256"
        )
    declared_coverage = collect_declared_coverage(
        run_dir,
        checksum_declaration.get("covers"),
    )

    coverage_errors: list[str] = []
    for source, names in (
        ("manifest artifacts.checksums.covers", declared_coverage),
        ("checksums.sha256", checksum_names),
    ):
        missing = sorted(required_coverage - names)
        unexpected = sorted(names - required_coverage)
        if missing:
            coverage_errors.append(f"{source} is missing {missing}")
        if unexpected:
            coverage_errors.append(f"{source} has unexpected files {unexpected}")
    if coverage_errors:
        raise ValueError(
            "run artifact checksum coverage is not exact: " + "; ".join(coverage_errors)
        )

    for relative_path, expected_hash in sorted(job_log_hashes.items()):
        sidecar_hash = checksum_by_name[relative_path]["sha256"]
        if sidecar_hash != expected_hash:
            raise ValueError(
                "log SHA-256 is inconsistent between manifest and checksums.sha256 "
                f"for {relative_path}"
            )

    report = {
        "input_kind": "run_directory",
        "status": "verified",
        "run_directory": str(run_dir),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "status": manifest_status,
        },
        "raw_csv": raw_report,
        "config_snapshot": config_report,
        "logs": {
            "verified_file_count": len(log_reports),
            "verified_files": log_reports,
            "status": "verified",
        },
        "checksums": {
            "path": str(checksum_path),
            "sha256": sha256_file(checksum_path),
            "verified_file_count": len(checksum_entries),
            "verified_files": checksum_entries,
            "status": "verified",
        },
    }
    return raw_path, report


def resolve_input_with_integrity(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if resolved.is_dir():
        return verify_run_directory(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"input CSV or run directory not found: {resolved}")
    return resolved, {
        "input_kind": "direct_csv",
        "status": "unavailable",
        "reason": "direct CSV input has no run manifest or checksum bundle to verify",
        "raw_csv": {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        },
    }


def resolve_input(path: Path) -> Path:
    """Resolve one input, retaining the legacy path-only helper API."""
    return resolve_input_with_integrity(path)[0]


# These fields define one comparable experimental condition. Algorithm-specific
# fields (for example qlsa_variant) are intentionally excluded so that the four
# algorithms can form paired blocks, while budget, executable, input, build
# configuration, and machine identity remain part of the condition.
CONDITION_FIELDS = (
    "experiment_name",
    "budget_scheme",
    "budget_target",
    "budget_unit",
    "instance",
    "bks",
    "requested_backend",
    "actual_backend",
    "require_backend_match",
    "time_limit_ms",
    "init",
    "chains",
    "threads",
    "actual_threads",
    "reported_parallel",
    "config_sha256",
    "input_sha256",
    "executable_sha256",
    "environment_sha256",
)

PROVENANCE_FIELDS = (
    "config_sha256",
    "input_sha256",
    "executable_sha256",
    "environment_sha256",
)


def normalized_condition(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "").strip() for field in CONDITION_FIELDS}


def condition_id_for_row(row: dict[str, str]) -> str:
    encoded = json.dumps(
        normalized_condition(row),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def validate_multi_input_provenance(rows: Sequence[dict[str, str]], input_count: int) -> None:
    if input_count <= 1:
        return
    missing: list[str] = []
    for row in rows:
        absent = [field for field in PROVENANCE_FIELDS if not row.get(field, "").strip()]
        if absent:
            missing.append(
                f"{row.get('_source')}:{row.get('_line')} missing {','.join(absent)}"
            )
            if len(missing) >= 3:
                break
    if missing:
        raise ValueError(
            "multiple input files require complete provenance to prevent cross-condition mixing: "
            + "; ".join(missing)
        )


def read_rows(paths: Sequence[Path]) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    skipped = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                if row.get("status", "").strip().lower() != "ok":
                    skipped += 1
                    continue
                row["_source"] = str(path)
                row["_line"] = str(line_number)
                row["_condition_id"] = condition_id_for_row(row)
                rows.append(row)
    return rows, skipped


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer {key!r} at {row.get('_source')}:{row.get('_line')}") from exc


def as_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid number {key!r} at {row.get('_source')}:{row.get('_line')}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key!r} at {row.get('_source')}:{row.get('_line')}")
    return value


def runtime_ms(row: dict[str, str]) -> tuple[float, str]:
    """Prefer end-to-end timing from extended executables, with legacy fallback."""
    if row.get("total_elapsed_ms", "").strip():
        return as_float(row, "total_elapsed_ms"), "total_elapsed_ms"
    return as_float(row, "elapsed_ms"), "elapsed_ms_legacy"


def optional_integer_values(rows: Sequence[dict[str, str]], key: str) -> list[int]:
    return [as_int(row, key) for row in rows if row.get(key, "").strip()]


def optional_boolean_values(rows: Sequence[dict[str, str]], key: str) -> list[bool]:
    values: list[bool] = []
    for row in rows:
        raw = row.get(key, "").strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "on"}:
            values.append(True)
        elif raw in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"invalid boolean {key!r} at {row.get('_source')}:{row.get('_line')}")
    return values


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def stable_seed(base_seed: int, *parts: Any) -> int:
    material = json.dumps([base_seed, *parts], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def bootstrap_ci(
    values: Sequence[float],
    samples: int,
    confidence: float,
    rng_seed: int,
    statistic: Callable[[Sequence[float]], float] = mean,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap CI requires at least one value")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if len(values) == 1:
        value = statistic(values)
        return value, value
    rng = random.Random(rng_seed)
    n = len(values)
    estimates = [
        statistic([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(samples)
    ]
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return percentile(estimates, alpha), percentile(estimates, 1.0 - alpha)


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict[str, Any]:
    nonzero = [difference for difference in differences if difference != 0.0]
    if not nonzero:
        return {
            "n": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "exact-enumeration",
        }
    ranks = average_ranks([abs(value) for value in nonzero])
    doubled_ranks = [int(round(rank * 2.0)) for rank in ranks]
    w_plus_twice = sum(rank for rank, difference in zip(doubled_ranks, nonzero) if difference > 0)
    total_twice = sum(doubled_ranks)

    counts = [0] * (total_twice + 1)
    counts[0] = 1
    reachable = 0
    for rank in doubled_ranks:
        for value in range(reachable, -1, -1):
            if counts[value]:
                counts[value + rank] += counts[value]
        reachable += rank
    observed_distance = abs(2 * w_plus_twice - total_twice)
    extreme = sum(count for value, count in enumerate(counts) if abs(2 * value - total_twice) >= observed_distance)
    p_value = min(1.0, extreme / float(2 ** len(nonzero)))
    w_plus = w_plus_twice / 2.0
    w_minus = (total_twice - w_plus_twice) / 2.0
    return {
        "n": len(nonzero),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": min(w_plus, w_minus),
        "p_value": p_value,
        "method": "exact-enumeration",
    }


def exact_sign_test(differences: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    ties = len(differences) - positive - negative
    non_tied = positive + negative
    if non_tied == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(non_tied, k) for k in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / float(2 ** non_tied))
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "n": non_tied,
        "p_value": p_value,
    }


def regularized_gamma_q(shape: float, value: float) -> float:
    """Regularized upper incomplete gamma Q(shape, value)."""
    if shape <= 0.0 or value < 0.0:
        raise ValueError("gamma arguments out of range")
    if value == 0.0:
        return 1.0
    epsilon = 3.0e-14
    tiny = 1.0e-300
    maximum_iterations = 10000
    log_term = -value + shape * math.log(value) - math.lgamma(shape)

    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        incremented_shape = shape
        for _ in range(maximum_iterations):
            incremented_shape += 1.0
            term *= value / incremented_shape
            total += term
            if abs(term) < abs(total) * epsilon:
                lower = total * math.exp(log_term)
                return min(1.0, max(0.0, 1.0 - lower))
        raise ArithmeticError("gamma series failed to converge")

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for index in range(1, maximum_iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) < epsilon:
            upper = math.exp(log_term) * fraction
            return min(1.0, max(0.0, upper))
    raise ArithmeticError("gamma continued fraction failed to converge")


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    if statistic < 0.0 or degrees_of_freedom <= 0:
        raise ValueError("invalid chi-square arguments")
    return regularized_gamma_q(degrees_of_freedom / 2.0, statistic / 2.0)


def friedman_test(blocks: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("Friedman test requires at least one complete block")
    algorithm_count = len(blocks[0])
    if algorithm_count < 3 or any(len(block) != algorithm_count for block in blocks):
        raise ValueError("Friedman test requires rectangular blocks with at least three algorithms")
    block_count = len(blocks)
    rank_sums = [0.0] * algorithm_count
    tie_sum = 0.0
    for block in blocks:
        ranks = average_ranks(block)
        for index, rank in enumerate(ranks):
            rank_sums[index] += rank
        frequencies: dict[float, int] = defaultdict(int)
        for value in block:
            frequencies[value] += 1
        tie_sum += sum(count ** 3 - count for count in frequencies.values() if count > 1)

    statistic = (
        12.0 / (block_count * algorithm_count * (algorithm_count + 1.0))
        * sum(rank_sum * rank_sum for rank_sum in rank_sums)
        - 3.0 * block_count * (algorithm_count + 1.0)
    )
    correction = 1.0 - tie_sum / (block_count * (algorithm_count ** 3 - algorithm_count))
    if correction <= 0.0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic /= correction
        p_value = chi_square_survival(max(0.0, statistic), algorithm_count - 1)
    return {
        "statistic": max(0.0, statistic),
        "degrees_of_freedom": algorithm_count - 1,
        "p_value": p_value,
        "tie_correction": correction,
        "mean_ranks": [rank_sum / block_count for rank_sum in rank_sums],
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def validate_unique_rows(rows: Sequence[dict[str, str]]) -> None:
    seen: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (
            row["_condition_id"],
            row["algorithm_key"],
            as_int(row, "seed"),
        )
        if key in seen:
            previous = seen[key]
            raise ValueError(
                "duplicate paired observation for "
                f"{key}: {previous.get('_source')}:{previous.get('_line')} and "
                f"{row.get('_source')}:{row.get('_line')}"
            )
        seen[key] = row


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["_condition_id"],
        row["budget_scheme"],
        row["instance"],
        row["requested_backend"],
        row["actual_backend"],
    )


def format_float(value: float) -> str:
    return f"{value:.12g}"


def summarize_algorithms(
    rows: Sequence[dict[str, str]], samples: int, confidence: float, random_seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(*group_key(row), row["algorithm_key"])].append(row)

    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        condition_id, budget, instance, requested, actual, algorithm = key
        items = sorted(items, key=lambda item: as_int(item, "seed"))
        bks_values = {as_int(item, "bks") for item in items}
        if len(bks_values) != 1:
            raise ValueError(f"inconsistent BKS values for {key}: {sorted(bks_values)}")
        bks = next(iter(bks_values))
        if bks <= 0:
            raise ValueError(f"BKS must be positive for {key}")
        best = [float(as_int(item, "best_length")) for item in items]
        elapsed_with_sources = [runtime_ms(item) for item in items]
        elapsed = [value for value, _ in elapsed_with_sources]
        timing_sources = {source for _, source in elapsed_with_sources}
        if len(timing_sources) != 1:
            raise ValueError(
                f"mixed timing sources are not comparable for condition {condition_id}, "
                f"algorithm {algorithm}: {sorted(timing_sources)}"
            )
        iterations_completed = optional_integer_values(items, "iterations_completed")
        deadline_reached = optional_boolean_values(items, "deadline_reached")
        gaps = [(value - bks) / bks * 100.0 for value in best]
        best_ci = bootstrap_ci(best, samples, confidence, stable_seed(random_seed, "best", *key))
        gap_ci = bootstrap_ci(gaps, samples, confidence, stable_seed(random_seed, "gap", *key))
        elapsed_ci = bootstrap_ci(elapsed, samples, confidence, stable_seed(random_seed, "elapsed", *key))
        output.append(
            {
                "condition_id": condition_id,
                "budget_scheme": budget,
                "instance": instance,
                "requested_backend": requested,
                "actual_backend": actual,
                "algorithm_key": algorithm,
                "runs": len(items),
                "unique_seeds": len({as_int(item, "seed") for item in items}),
                "bks": bks,
                "best_length_min": int(min(best)),
                "best_length_mean": format_float(mean(best)),
                "best_length_median": format_float(statistics.median(best)),
                "best_length_std": format_float(sample_std(best)),
                "best_length_mean_ci_low": format_float(best_ci[0]),
                "best_length_mean_ci_high": format_float(best_ci[1]),
                "gap_percent_mean": format_float(mean(gaps)),
                "gap_percent_median": format_float(statistics.median(gaps)),
                "gap_percent_mean_ci_low": format_float(gap_ci[0]),
                "gap_percent_mean_ci_high": format_float(gap_ci[1]),
                "bks_hit_count": sum(value <= bks for value in best),
                "bks_hit_rate": format_float(sum(value <= bks for value in best) / len(best)),
                "timing_source": next(iter(timing_sources)),
                "elapsed_ms_mean": format_float(mean(elapsed)),
                "elapsed_ms_median": format_float(statistics.median(elapsed)),
                "elapsed_ms_std": format_float(sample_std(elapsed)),
                "elapsed_ms_mean_ci_low": format_float(elapsed_ci[0]),
                "elapsed_ms_mean_ci_high": format_float(elapsed_ci[1]),
                "iterations_completed_samples": len(iterations_completed),
                "iterations_completed_mean": format_float(mean(iterations_completed)) if iterations_completed else "",
                "iterations_completed_median": format_float(statistics.median(iterations_completed)) if iterations_completed else "",
                "iterations_completed_std": format_float(sample_std(iterations_completed)) if iterations_completed else "",
                "deadline_reached_samples": len(deadline_reached),
                "deadline_reached_count": sum(deadline_reached) if deadline_reached else "",
                "deadline_reached_rate": format_float(sum(deadline_reached) / len(deadline_reached)) if deadline_reached else "",
                "bootstrap_samples": samples,
                "confidence_level": confidence,
            }
        )
    return output


def rows_by_group_algorithm_seed(
    rows: Sequence[dict[str, str]], algorithms: Sequence[str]
) -> dict[tuple[str, str, str, str, str], dict[str, dict[int, dict[str, str]]]]:
    selected = set(algorithms)
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["algorithm_key"] in selected:
            grouped[group_key(row)][row["algorithm_key"]][as_int(row, "seed")] = row
    return grouped


def analyze_pairwise(
    rows: Sequence[dict[str, str]],
    algorithms: Sequence[str],
    samples: int,
    confidence: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    grouped = rows_by_group_algorithm_seed(rows, algorithms)
    output: list[dict[str, Any]] = []
    for key, by_algorithm in sorted(grouped.items()):
        available = [algorithm for algorithm in algorithms if algorithm in by_algorithm]
        group_rows: list[dict[str, Any]] = []
        for algorithm_a, algorithm_b in itertools.combinations(available, 2):
            seeds_a = set(by_algorithm[algorithm_a])
            seeds_b = set(by_algorithm[algorithm_b])
            paired = sorted(seeds_a & seeds_b)
            if not paired:
                continue
            differences = [
                float(as_int(by_algorithm[algorithm_a][seed], "best_length"))
                - float(as_int(by_algorithm[algorithm_b][seed], "best_length"))
                for seed in paired
            ]
            interval = bootstrap_ci(
                differences,
                samples,
                confidence,
                stable_seed(random_seed, "pair", *key, algorithm_a, algorithm_b),
            )
            wilcoxon = wilcoxon_signed_rank(differences)
            sign = exact_sign_test(differences)
            group_rows.append(
                {
                    "condition_id": key[0],
                    "budget_scheme": key[1],
                    "instance": key[2],
                    "requested_backend": key[3],
                    "actual_backend": key[4],
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "seeds_a": len(seeds_a),
                    "seeds_b": len(seeds_b),
                    "paired_seeds": len(paired),
                    "paired_seed_values_json": json.dumps(paired, separators=(",", ":")),
                    "mean_difference_a_minus_b": format_float(mean(differences)),
                    "median_difference_a_minus_b": format_float(statistics.median(differences)),
                    "std_difference_a_minus_b": format_float(sample_std(differences)),
                    "mean_difference_ci_low": format_float(interval[0]),
                    "mean_difference_ci_high": format_float(interval[1]),
                    "wins_a": sign["negative"],
                    "wins_b": sign["positive"],
                    "ties": sign["ties"],
                    "wilcoxon_nonzero_pairs": wilcoxon["n"],
                    "wilcoxon_w_plus": format_float(wilcoxon["w_plus"]),
                    "wilcoxon_w_minus": format_float(wilcoxon["w_minus"]),
                    "wilcoxon_statistic": format_float(wilcoxon["statistic"]),
                    "wilcoxon_p_value": format_float(wilcoxon["p_value"]),
                    "wilcoxon_method": wilcoxon["method"],
                    "sign_non_tied_pairs": sign["n"],
                    "sign_p_value": format_float(sign["p_value"]),
                    "bootstrap_samples": samples,
                    "confidence_level": confidence,
                }
            )
        if group_rows:
            wilcoxon_adjusted = holm_adjust([float(item["wilcoxon_p_value"]) for item in group_rows])
            sign_adjusted = holm_adjust([float(item["sign_p_value"]) for item in group_rows])
            for item, wilcoxon_p, sign_p in zip(group_rows, wilcoxon_adjusted, sign_adjusted):
                item["wilcoxon_p_holm"] = format_float(wilcoxon_p)
                item["wilcoxon_reject_holm_0_05"] = str(wilcoxon_p <= 0.05).lower()
                item["sign_p_holm"] = format_float(sign_p)
                item["sign_reject_holm_0_05"] = str(sign_p <= 0.05).lower()
            output.extend(group_rows)
    return output


def analyze_friedman(
    rows: Sequence[dict[str, str]], algorithms: Sequence[str]
) -> list[dict[str, Any]]:
    grouped = rows_by_group_algorithm_seed(rows, algorithms)
    output: list[dict[str, Any]] = []
    for key, by_algorithm in sorted(grouped.items()):
        union_seeds = set().union(*(set(seed_map) for seed_map in by_algorithm.values())) if by_algorithm else set()
        complete = sorted(
            seed
            for seed in union_seeds
            if all(algorithm in by_algorithm and seed in by_algorithm[algorithm] for algorithm in algorithms)
        )
        base: dict[str, Any] = {
            "condition_id": key[0],
            "budget_scheme": key[1],
            "instance": key[2],
            "requested_backend": key[3],
            "actual_backend": key[4],
            "algorithms_json": json.dumps(list(algorithms), separators=(",", ":")),
            "union_seed_count": len(union_seeds),
            "complete_seed_blocks": len(complete),
            "complete_seed_values_json": json.dumps(complete, separators=(",", ":")),
        }
        if len(algorithms) < 3 or not complete:
            base.update(
                {
                    "statistic": "",
                    "degrees_of_freedom": max(0, len(algorithms) - 1),
                    "p_value": "",
                    "tie_correction": "",
                    "mean_ranks_json": "{}",
                    "status": "insufficient-complete-blocks",
                }
            )
        else:
            blocks = [
                [float(as_int(by_algorithm[algorithm][seed], "best_length")) for algorithm in algorithms]
                for seed in complete
            ]
            result = friedman_test(blocks)
            base.update(
                {
                    "statistic": format_float(result["statistic"]),
                    "degrees_of_freedom": result["degrees_of_freedom"],
                    "p_value": format_float(result["p_value"]),
                    "tie_correction": format_float(result["tie_correction"]),
                    "mean_ranks_json": json.dumps(
                        dict(zip(algorithms, result["mean_ranks"])),
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "status": "ok",
                }
            )
        output.append(base)
    return output


def pairing_warnings(
    rows: Sequence[dict[str, str]], algorithms: Sequence[str]
) -> list[str]:
    grouped = rows_by_group_algorithm_seed(rows, algorithms)
    warnings: list[str] = []
    for key, by_algorithm in sorted(grouped.items()):
        missing = [algorithm for algorithm in algorithms if algorithm not in by_algorithm]
        if missing:
            warnings.append(f"{key}: missing algorithms {missing}")
            continue
        seed_sets = {algorithm: set(by_algorithm[algorithm]) for algorithm in algorithms}
        reference = seed_sets[algorithms[0]]
        mismatched = {algorithm: sorted(seeds ^ reference) for algorithm, seeds in seed_sets.items() if seeds != reference}
        if mismatched:
            warnings.append(f"{key}: paired seed mismatch {mismatched}")
    return warnings


def write_csv(path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_checksums(output_dir: Path, paths: Iterable[Path]) -> Path:
    entries = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return checksum_path


def default_output_dir() -> Path:
    stamp = dt.datetime.now().astimezone().strftime("analysis_%Y%m%d_%H%M%S_%z")
    return ROOT / "results" / "fair_experiments" / stamp


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.bootstrap_samples <= 0:
            raise ValueError("--bootstrap-samples must be positive")
        if not 0.0 < args.confidence < 1.0:
            raise ValueError("--confidence must be in (0, 1)")
        if len(args.algorithms) != len(set(args.algorithms)):
            raise ValueError("--algorithms must be unique")

        resolved_inputs = [resolve_input_with_integrity(path) for path in args.input]
        inputs = [path for path, _ in resolved_inputs]
        input_integrity = [report for _, report in resolved_inputs]
        rows, skipped = read_rows(inputs)
        if not rows:
            raise ValueError("no successful rows are available for analysis")
        validate_multi_input_provenance(rows, len(inputs))
        available_algorithms = {row["algorithm_key"] for row in rows}
        missing_algorithms = set(args.algorithms) - available_algorithms
        if missing_algorithms:
            raise ValueError(
                f"requested algorithms are absent from successful rows: {sorted(missing_algorithms)}"
            )
        selected_algorithms = set(args.algorithms)
        rows = [row for row in rows if row["algorithm_key"] in selected_algorithms]
        validate_unique_rows(rows)
        warnings = pairing_warnings(rows, args.algorithms)
        if warnings and not args.allow_incomplete_pairs:
            raise ValueError(
                "complete paired blocks are required by default: "
                + "; ".join(warnings)
                + "; pass --allow-incomplete-pairs only for exploratory analysis"
            )

        output_dir = args.output_dir or default_output_dir()
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = summarize_algorithms(rows, args.bootstrap_samples, args.confidence, args.random_seed)
        pairwise = analyze_pairwise(
            rows,
            args.algorithms,
            args.bootstrap_samples,
            args.confidence,
            args.random_seed,
        )
        friedman = analyze_friedman(rows, args.algorithms)

        summary_path = output_dir / "summary.csv"
        pairwise_path = output_dir / "pairwise.csv"
        friedman_path = output_dir / "friedman.csv"
        manifest_path = output_dir / "analysis_manifest.json"
        write_csv(summary_path, SUMMARY_HEADER, summary)
        write_csv(pairwise_path, PAIRWISE_HEADER, pairwise)
        write_csv(friedman_path, FRIEDMAN_HEADER, friedman)

        manifest = {
            "schema_version": 1,
            "created_at": now_iso(),
            "statistics_engine": "python-standard-library",
            "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "python": sys.version,
            "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
            "input_integrity": input_integrity,
            "successful_input_rows": len(rows),
            "skipped_non_ok_rows": skipped,
            "algorithms": args.algorithms,
            "condition_fields": list(CONDITION_FIELDS),
            "condition_count": len({row["_condition_id"] for row in rows}),
            "pairing_mode": (
                "seed-intersection-exploratory"
                if args.allow_incomplete_pairs
                else "complete-blocks-required"
            ),
            "bootstrap": {
                "samples": args.bootstrap_samples,
                "confidence": args.confidence,
                "base_random_seed": args.random_seed,
                "method": "percentile bootstrap of the arithmetic mean",
            },
            "tests": {
                "wilcoxon": "two-sided exact sign enumeration of average signed ranks; zero differences omitted",
                "sign": "two-sided exact binomial test; ties omitted",
                "friedman": "tie-corrected chi-square approximation with stdlib incomplete-gamma survival function",
                "primary_pairwise_test": "Wilcoxon signed-rank",
                "secondary_pairwise_test": "exact sign test",
                "multiple_comparisons": "Holm adjustment within each condition_id over the six algorithm pairs",
            },
            "pairing_warnings": warnings,
            "artifacts": {
                "summary": summary_path.name,
                "pairwise": pairwise_path.name,
                "friedman": friedman_path.name,
            },
        }
        atomic_write_json(manifest_path, manifest)
        checksum_path = write_checksums(
            output_dir,
            [summary_path, pairwise_path, friedman_path, manifest_path],
        )

        for warning in warnings:
            print(f"[warning] {warning}", file=sys.stderr)
        print(f"[ok] summary: {summary_path}")
        print(f"[ok] pairwise: {pairwise_path}")
        print(f"[ok] Friedman: {friedman_path}")
        print(f"[ok] manifest: {manifest_path}")
        print(f"[ok] checksums: {checksum_path}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, ArithmeticError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
