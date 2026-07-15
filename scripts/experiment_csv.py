"""Shared parsing and execution-contract checks for ``tsp_sa`` CSV output.

The command-line program has a stable 14-column result prefix and has added
accounting fields over time.  Experiment runners use this module so a newer
program result cannot be silently discarded by an older runner.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


BASE_HEADER = [
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

EXTENDED_HEADER = BASE_HEADER + [
    "total_elapsed_ms",
    "cuda_kernel_elapsed_ms",
    "requested_backend",
    "actual_backend",
    "backend_fallback",
    "backend_fallback_reason",
    "iterations_completed",
    "deadline_reached",
]

MIGRATION_HEADER = EXTENDED_HEADER + [
    "migration_topology",
    "migration_interval",
    "migration_rounds",
    "migration_attempts",
    "migrations_adopted",
]

CURRENT_HEADER = MIGRATION_HEADER + ["actual_threads"]
PROGRAM_HEADERS_BY_WIDTH = {
    len(BASE_HEADER): BASE_HEADER,
    len(EXTENDED_HEADER): EXTENDED_HEADER,
    len(MIGRATION_HEADER): MIGRATION_HEADER,
    len(CURRENT_HEADER): CURRENT_HEADER,
}
CURRENT_SCHEMA_WIDTH = len(CURRENT_HEADER)


class ExperimentCsvError(RuntimeError):
    """Raised when program output is incomplete or contradicts its command."""


AlgorithmPredicate = Callable[[str], bool]


def is_tsp_algorithm_label(label: str) -> bool:
    """Return whether *label* is a result label emitted by the TSP CLI."""

    normalized = label.strip().lower()
    return normalized == "sa" or normalized == "qlsa" or normalized.startswith(("sa-", "qlsa-"))


def parse_program_rows(
    stdout: str,
    *,
    algorithm_predicate: AlgorithmPredicate | None = None,
) -> list[dict[str, str]]:
    """Parse recognized CSV result lines from CLI stdout.

    All four documented schemas (14, 22, 27, and 28 fields) are accepted.
    Rows are normalized to :data:`CURRENT_HEADER`; ``_csv_schema_width`` keeps
    the original width for the execution-contract validator.  A line that
    looks like a program result but has an unsupported width is an error rather
    than an empty result set, which keeps schema upgrades fail-closed.
    """

    rows: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("CSV:"):
            continue
        try:
            parts = next(csv.reader([line], strict=True))
        except csv.Error as exc:
            if line.split(",", 1)[0].strip().lower().startswith(("sa", "qlsa")):
                raise ExperimentCsvError(f"malformed CSV result at stdout line {line_number}: {exc}") from exc
            continue
        if not parts or not is_tsp_algorithm_label(parts[0]):
            continue
        if algorithm_predicate is not None and not algorithm_predicate(parts[0]):
            raise ExperimentCsvError(
                f"unexpected algorithm label {parts[0]!r} at stdout line {line_number}"
            )
        header = PROGRAM_HEADERS_BY_WIDTH.get(len(parts))
        if header is None:
            raise ExperimentCsvError(
                f"unsupported tsp_sa CSV schema with {len(parts)} fields at stdout line {line_number}; "
                f"supported widths are {sorted(PROGRAM_HEADERS_BY_WIDTH)}"
            )
        row = {field: "" for field in CURRENT_HEADER}
        row.update(zip(header, parts))
        row["_csv_schema_width"] = str(len(parts))
        rows.append(row)
    return rows


def row_for_output(row: Mapping[str, str], fieldnames: Iterable[str]) -> dict[str, str]:
    """Return only writer fields, omitting parser metadata such as schema width."""

    return {field: row.get(field, "") for field in fieldnames}


def _parse_integer(row: Mapping[str, str], field: str, source: str) -> int:
    value = row.get(field, "")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentCsvError(f"{source}: field {field!r} must be an integer, got {value!r}") from exc


def _parse_false(row: Mapping[str, str], field: str, source: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value in {"false", "0", "no"}:
        return False
    if value in {"true", "1", "yes"}:
        return True
    raise ExperimentCsvError(f"{source}: field {field!r} must be a boolean, got {value!r}")


def _option_value(command: Sequence[str | Path], option: str, default: str | None = None) -> str | None:
    arguments = [str(item) for item in command]
    try:
        position = arguments.index(option)
    except ValueError:
        return default
    if position + 1 >= len(arguments):
        raise ExperimentCsvError(f"command option {option!r} has no value")
    return arguments[position + 1]


@dataclass(frozen=True)
class ExecutionContract:
    expected_rows: int
    algorithm_label: str
    parallel: str
    iterations: int
    seed: int
    init: str
    chains: int
    workers: int
    requested_backend: str


def contract_from_command(command: Sequence[str | Path]) -> ExecutionContract:
    """Derive the fixed-iteration result contract from a ``tsp_sa`` command."""

    parallel = (_option_value(command, "--parallel", "none") or "none").lower()
    if parallel not in {"none", "omp", "cuda"}:
        raise ExperimentCsvError(f"unsupported --parallel value in command: {parallel!r}")

    def required_int(option: str, default: int | None = None) -> int:
        value = _option_value(command, option, None if default is None else str(default))
        try:
            return int(value) if value is not None else (_ for _ in ()).throw(ValueError())
        except ValueError as exc:
            raise ExperimentCsvError(f"command option {option!r} must be an integer") from exc

    iterations = required_int("--iterations")
    seed = required_int("--seed", 1)
    chains = required_int("--chains", 1)
    expected_rows = required_int("--repeat", 1)
    if iterations <= 0 or chains <= 0 or expected_rows <= 0:
        raise ExperimentCsvError("--iterations, --chains, and --repeat must all be positive")
    if parallel == "cuda":
        workers = required_int("--cuda_block_size", 128)
        requested_backend = "cuda"
    elif parallel == "omp":
        workers = required_int("--threads", 1)
        # The CLI preserves ``parallel=omp`` in its descriptive field, but
        # deliberately dispatches a one-thread request through the serial
        # backend.  Its accounting fields therefore report cpu_serial.
        requested_backend = "openmp" if workers > 1 else "cpu_serial"
    else:
        workers = 1
        requested_backend = "cpu_serial"
    if workers <= 0:
        raise ExperimentCsvError("requested worker count must be positive")

    arguments = [str(item) for item in command]
    use_qlsa = "--qlsa" in arguments
    variant = (_option_value(command, "--qlsa_variant", "current") or "current").lower()
    base = "qlsa" if use_qlsa else "sa"
    if use_qlsa and variant != "current":
        base += f"-{variant}"
    migration = (_option_value(command, "--migration-topology", "disabled") or "disabled").lower()
    if migration != "disabled":
        algorithm_label = f"{base}-island-{migration}"
        if parallel == "omp":
            algorithm_label += "-omp"
    elif parallel == "cuda":
        cuda_mode = (_option_value(command, "--cuda_mode", "chain") or "chain").lower()
        candidate_policy = (
            _option_value(command, "--cuda_candidate_policy", "best") or "best"
        ).lower()
        if cuda_mode == "candidate" and candidate_policy != "best":
            algorithm_label = f"{base}-cuda-candidate-{candidate_policy}"
        else:
            algorithm_label = f"{base}-cuda-{cuda_mode}"
    elif parallel == "omp":
        algorithm_label = f"{base}-omp"
    elif chains > 1:
        algorithm_label = f"{base}-multichain"
    else:
        algorithm_label = base

    init = (_option_value(command, "--init", "nn") or "nn").lower()
    return ExecutionContract(
        expected_rows,
        algorithm_label,
        parallel,
        iterations,
        seed,
        init,
        chains,
        workers,
        requested_backend,
    )


def validate_execution_contract(
    rows: Sequence[Mapping[str, str]],
    contract: ExecutionContract,
    *,
    source: str = "tsp_sa output",
    require_current_schema: bool = True,
) -> None:
    """Verify that rows fully and truthfully represent one requested command.

    New experiment execution requires the current 28-field schema because only
    it records fallback, actual backend, actual worker count, and completed
    iterations.  Historical rows remain parseable for analysis via
    ``require_current_schema=False``.
    """

    if len(rows) != contract.expected_rows:
        raise ExperimentCsvError(
            f"{source}: expected {contract.expected_rows} CSV rows from --repeat, got {len(rows)}"
        )
    for index, row in enumerate(rows, start=1):
        prefix = f"{source}: row {index}"
        schema_width = _parse_integer(row, "_csv_schema_width", prefix)
        if schema_width not in PROGRAM_HEADERS_BY_WIDTH:
            raise ExperimentCsvError(f"{prefix}: unrecognized schema width {schema_width}")
        if require_current_schema and schema_width != CURRENT_SCHEMA_WIDTH:
            raise ExperimentCsvError(
                f"{prefix}: received legacy {schema_width}-field CSV output; "
                f"the {CURRENT_SCHEMA_WIDTH}-field schema is required to verify backend execution"
            )
        if row.get("parallel", "").lower() != contract.parallel:
            raise ExperimentCsvError(
                f"{prefix}: parallel={row.get('parallel')!r}, expected {contract.parallel!r}"
            )
        if row.get("algorithm", "").lower() != contract.algorithm_label:
            raise ExperimentCsvError(
                f"{prefix}: algorithm={row.get('algorithm')!r}, "
                f"expected {contract.algorithm_label!r}"
            )
        if row.get("init", "").lower() != contract.init:
            raise ExperimentCsvError(
                f"{prefix}: init={row.get('init')!r}, expected {contract.init!r}"
            )
        for field, expected in (
            ("iterations", contract.iterations),
            ("chains", contract.chains),
            ("threads", contract.workers),
            ("seed", contract.seed + index - 1),
        ):
            actual = _parse_integer(row, field, prefix)
            if actual != expected:
                raise ExperimentCsvError(f"{prefix}: {field}={actual}, expected {expected}")

        if schema_width == CURRENT_SCHEMA_WIDTH:
            actual_requested = row.get("requested_backend", "").lower()
            actual_backend = row.get("actual_backend", "").lower()
            if actual_requested != contract.requested_backend:
                raise ExperimentCsvError(
                    f"{prefix}: requested_backend={actual_requested!r}, expected {contract.requested_backend!r}"
                )
            fallback = _parse_false(row, "backend_fallback", prefix)
            if fallback:
                reason = row.get("backend_fallback_reason", "").strip()
                suffix = f" ({reason})" if reason else ""
                raise ExperimentCsvError(f"{prefix}: backend fallback reported{suffix}")
            if row.get("backend_fallback_reason", "").strip():
                raise ExperimentCsvError(
                    f"{prefix}: backend_fallback=false but fallback reason is non-empty"
                )
            if actual_backend != contract.requested_backend:
                raise ExperimentCsvError(
                    f"{prefix}: actual_backend={actual_backend!r}, expected {contract.requested_backend!r}"
                )
            actual_threads = _parse_integer(row, "actual_threads", prefix)
            if actual_threads != contract.workers:
                raise ExperimentCsvError(
                    f"{prefix}: actual_threads={actual_threads}, expected {contract.workers}"
                )
            completed = _parse_integer(row, "iterations_completed", prefix)
            expected_completed = contract.iterations * contract.chains
            if completed != expected_completed:
                raise ExperimentCsvError(
                    f"{prefix}: iterations_completed={completed}, expected {expected_completed}"
                )
            if _parse_false(row, "deadline_reached", prefix):
                raise ExperimentCsvError(f"{prefix}: deadline_reached=true for a fixed-iteration experiment")


def validate_command_output(
    rows: Sequence[Mapping[str, str]],
    command: Sequence[str | Path],
    *,
    source: str = "tsp_sa output",
    require_current_schema: bool = True,
) -> None:
    """Convenience wrapper for the usual runner case."""

    validate_execution_contract(
        rows,
        contract_from_command(command),
        source=source,
        require_current_schema=require_current_schema,
    )


def resolve_executable(
    explicit: str | Path | None,
    candidates: Iterable[Path],
    *,
    root: Path,
    description: str = "tsp_sa executable",
) -> Path:
    """Prefer an explicit executable and otherwise locate the first candidate."""

    if explicit is not None:
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"explicit {description} does not exist: {path}")
        return path
    candidates = list(candidates)
    for path in candidates:
        if path.is_file():
            return path
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n{attempted}")
