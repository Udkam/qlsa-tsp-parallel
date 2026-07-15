"""Strict parser and contract checks for the 16-column ``tsp_sa_mpi`` CSV."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    from scripts.experiment_csv import BASE_HEADER, is_tsp_algorithm_label
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from experiment_csv import BASE_HEADER, is_tsp_algorithm_label  # type: ignore[no-redef]


MPI_LEGACY_HEADER = BASE_HEADER + ["mpi_ranks", "communication_ms"]
MPI_HEADER = MPI_LEGACY_HEADER + [
    "actual_threads",
    "iterations_completed",
    "deadline_reached",
]
MPI_HEADERS_BY_WIDTH = {
    len(MPI_LEGACY_HEADER): MPI_LEGACY_HEADER,
    len(MPI_HEADER): MPI_HEADER,
}
MPI_SCHEMA_WIDTH = len(MPI_HEADER)
MPI_ALGORITHMS = {"sa-mpi-omp", "qlsa-mpi-omp"}


class MpiCsvError(RuntimeError):
    """Raised when MPI runner output cannot prove the requested execution."""


def parse_mpi_program_rows(stdout: str, *, algorithm: str | None = None) -> list[dict[str, str]]:
    """Parse MPI rows while retaining their original schema width."""

    expected_algorithm = None if algorithm is None else f"{algorithm}-mpi-omp"
    if algorithm is not None and algorithm not in {"sa", "qlsa"}:
        raise MpiCsvError(f"unsupported MPI algorithm family: {algorithm!r}")

    rows: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("CSV:"):
            continue
        try:
            parts = next(csv.reader([line], strict=True))
        except csv.Error as exc:
            if line.split(",", 1)[0].strip().lower().startswith(("sa", "qlsa")):
                raise MpiCsvError(f"malformed MPI CSV at stdout line {line_number}: {exc}") from exc
            continue
        if not parts or not is_tsp_algorithm_label(parts[0]):
            continue
        header = MPI_HEADERS_BY_WIDTH.get(len(parts))
        if header is None:
            raise MpiCsvError(
                f"stdout line {line_number}: expected an MPI CSV width in "
                f"{sorted(MPI_HEADERS_BY_WIDTH)}, got {len(parts)} columns"
            )
        if parts[0] not in MPI_ALGORITHMS:
            raise MpiCsvError(
                f"stdout line {line_number}: unsupported MPI algorithm label {parts[0]!r}"
            )
        if expected_algorithm is not None and parts[0] != expected_algorithm:
            raise MpiCsvError(
                f"stdout line {line_number}: algorithm={parts[0]!r}, expected {expected_algorithm!r}"
            )
        if parts[8] != "mpi-omp":
            raise MpiCsvError(
                f"stdout line {line_number}: parallel={parts[8]!r}, expected 'mpi-omp'"
            )
        row = {field: "" for field in MPI_HEADER}
        row.update(zip(header, parts))
        row["_csv_schema_width"] = str(len(parts))
        rows.append(row)
    return rows


def _int(row: Mapping[str, str], field: str, source: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MpiCsvError(f"{source}: {field} must be an integer, got {row.get(field)!r}") from exc


def _nonnegative_float(row: Mapping[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MpiCsvError(f"{source}: {field} must be numeric, got {row.get(field)!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise MpiCsvError(f"{source}: {field} must be a finite non-negative value, got {value!r}")
    return value


@dataclass(frozen=True)
class MpiExecutionContract:
    algorithm: str
    iterations: int
    chains: int
    threads: int
    ranks: int
    repeat: int
    seed: int
    init: str = "nn"


def validate_mpi_execution_contract(
    rows: Sequence[Mapping[str, str]],
    contract: MpiExecutionContract,
    *,
    source: str = "tsp_sa_mpi output",
    require_current_schema: bool = True,
) -> None:
    """Require one exact 16-column MPI result for each requested repeat."""

    if contract.repeat <= 0:
        raise MpiCsvError("repeat must be positive")
    if len(rows) != contract.repeat:
        raise MpiCsvError(f"{source}: expected {contract.repeat} CSV rows from --repeat, got {len(rows)}")
    expected_algorithm = f"{contract.algorithm}-mpi-omp"
    for index, row in enumerate(rows, start=1):
        prefix = f"{source}: row {index}"
        width = _int(row, "_csv_schema_width", prefix)
        if width not in MPI_HEADERS_BY_WIDTH:
            raise MpiCsvError(f"{prefix}: unrecognized MPI CSV schema width {width}")
        if require_current_schema and width != MPI_SCHEMA_WIDTH:
            raise MpiCsvError(
                f"{prefix}: legacy {width}-column MPI CSV cannot prove actual OpenMP threads "
                f"or completed work; {MPI_SCHEMA_WIDTH} columns are required"
            )
        if row.get("algorithm") != expected_algorithm:
            raise MpiCsvError(
                f"{prefix}: algorithm={row.get('algorithm')!r}, expected {expected_algorithm!r}"
            )
        if row.get("parallel") != "mpi-omp":
            raise MpiCsvError(f"{prefix}: parallel={row.get('parallel')!r}, expected 'mpi-omp'")
        if row.get("init", "").lower() != contract.init.lower():
            raise MpiCsvError(
                f"{prefix}: init={row.get('init')!r}, expected {contract.init!r}"
            )
        expected_fields = {
            "iterations": contract.iterations,
            "chains": contract.chains,
            "threads": contract.threads,
            "mpi_ranks": contract.ranks,
            "seed": contract.seed + index - 1,
        }
        for field, expected in expected_fields.items():
            actual = _int(row, field, prefix)
            if actual != expected:
                raise MpiCsvError(f"{prefix}: {field}={actual}, expected {expected}")
        _nonnegative_float(row, "elapsed_ms", prefix)
        _nonnegative_float(row, "communication_ms", prefix)
        if width == MPI_SCHEMA_WIDTH:
            actual_threads = _int(row, "actual_threads", prefix)
            if actual_threads != contract.threads:
                raise MpiCsvError(
                    f"{prefix}: actual_threads={actual_threads}, expected {contract.threads}"
                )
            completed = _int(row, "iterations_completed", prefix)
            expected_completed = contract.iterations * contract.chains
            if completed != expected_completed:
                raise MpiCsvError(
                    f"{prefix}: iterations_completed={completed}, expected {expected_completed}"
                )
            deadline = row.get("deadline_reached", "").strip().lower()
            if deadline not in {"false", "0", "no"}:
                if deadline in {"true", "1", "yes"}:
                    raise MpiCsvError(f"{prefix}: deadline_reached=true for fixed work")
                raise MpiCsvError(
                    f"{prefix}: deadline_reached must be a boolean, got {deadline!r}"
                )
