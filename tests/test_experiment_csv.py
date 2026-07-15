"""Regression tests for shared CLI CSV parsing and runner schema migration."""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
import unittest
from pathlib import Path

from scripts.experiment_csv import (
    BASE_HEADER,
    CURRENT_HEADER,
    EXTENDED_HEADER,
    MIGRATION_HEADER,
    ExperimentCsvError,
    contract_from_command,
    parse_program_rows,
    validate_execution_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def current_values(
    *,
    algorithm: str = "sa-omp",
    parallel: str = "omp",
    workers: int = 2,
    backend: str = "openmp",
) -> dict[str, str]:
    values = {field: "" for field in CURRENT_HEADER}
    values.update(
        {
            "algorithm": algorithm,
            "instance": "square4",
            "dimension": "4",
            "iterations": "5",
            "seed": "1",
            "init": "nn",
            "chains": "2",
            "threads": str(workers),
            "parallel": parallel,
            "best_length": "40",
            "final_length": "40",
            "elapsed_ms": "1.000",
            "accepted_moves": "4",
            "improved_moves": "2",
            "total_elapsed_ms": "1.000",
            "cuda_kernel_elapsed_ms": "0.000",
            "requested_backend": backend,
            "actual_backend": backend,
            "backend_fallback": "false",
            "backend_fallback_reason": "",
            "iterations_completed": "10",
            "deadline_reached": "false",
            "migration_topology": "disabled",
            "migration_interval": "0",
            "migration_rounds": "0",
            "migration_attempts": "0",
            "migrations_adopted": "0",
            "actual_threads": str(workers),
        }
    )
    return values


def csv_line(values: dict[str, str], header: list[str] = CURRENT_HEADER) -> str:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow([values[field] for field in header])
    return stream.getvalue()


def omp_command() -> list[str]:
    return [
        "tsp_sa",
        "--parallel",
        "omp",
        "--chains",
        "2",
        "--threads",
        "2",
        "--iterations",
        "5",
        "--repeat",
        "1",
        "--csv-only",
    ]


class ExperimentCsvTests(unittest.TestCase):
    def test_parser_accepts_all_documented_widths(self) -> None:
        values = current_values()
        for header in (BASE_HEADER, EXTENDED_HEADER, MIGRATION_HEADER, CURRENT_HEADER):
            with self.subTest(width=len(header)):
                rows = parse_program_rows(csv_line(values, header))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["_csv_schema_width"], str(len(header)))
                self.assertEqual(rows[0]["algorithm"], "sa-omp")
                self.assertEqual(rows[0]["actual_backend"], "openmp" if len(header) >= 22 else "")

    def test_current_execution_contract_accepts_complete_row(self) -> None:
        rows = parse_program_rows(csv_line(current_values()))
        validate_execution_contract(rows, contract_from_command(omp_command()))

    def test_one_thread_openmp_contract_expects_serial_accounting(self) -> None:
        command = omp_command()
        command[command.index("--threads") + 1] = "1"
        values = current_values(workers=1, backend="cpu_serial")
        rows = parse_program_rows(csv_line(values))
        validate_execution_contract(rows, contract_from_command(command))

    def test_execution_contract_rejects_incomplete_and_contradictory_output(self) -> None:
        contract = contract_from_command(omp_command())
        good = parse_program_rows(csv_line(current_values()))[0]
        cases = {
            "no_rows": [],
            "legacy_schema": parse_program_rows(csv_line(current_values(), BASE_HEADER)),
            "backend_fallback": [{**good, "backend_fallback": "true", "backend_fallback_reason": "CUDA unavailable"}],
            "stale_fallback_reason": [{**good, "backend_fallback_reason": "stale reason"}],
            "actual_backend": [{**good, "actual_backend": "cpu_serial"}],
            "actual_threads": [{**good, "actual_threads": "1"}],
            "iterations_completed": [{**good, "iterations_completed": "9"}],
            "algorithm": [{**good, "algorithm": "qlsa-omp"}],
            "seed": [{**good, "seed": "2"}],
            "init": [{**good, "init": "random"}],
            "sample_count": [good, dict(good)],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ExperimentCsvError):
                    validate_execution_contract(rows, contract)

    def test_legacy_rows_can_be_checked_for_base_contract_when_requested(self) -> None:
        rows = parse_program_rows(csv_line(current_values(), BASE_HEADER))
        validate_execution_contract(
            rows,
            contract_from_command(omp_command()),
            require_current_schema=False,
        )

    def test_qlsa_variant_label_is_derived_from_command(self) -> None:
        command = omp_command() + ["--qlsa", "--qlsa_variant", "paper-sb"]
        contract = contract_from_command(command)
        self.assertEqual(contract.algorithm_label, "qlsa-paper-sb-omp")


class RunnerSchemaMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda_candidate = load_script("test_cuda_candidate_runner", "scripts/run_cuda_candidate_experiments.py")
        cls.cuda_sweep = load_script("test_cuda_sweep_runner", "scripts/run_cuda_candidate_sweep.py")
        cls.large_openmp = load_script("test_large_openmp_runner", "scripts/run_large_openmp.py")
        cls.large_cuda = load_script("test_large_cuda_runner", "scripts/run_large_cuda.py")
        cls.openmp_grid = load_script("test_openmp_grid_runner", "scripts/run_openmp_scaling_grid.py")
        cls.step5 = load_script("test_step5_runner", "scripts/run_step5_experiments.py")
        cls.tuned = load_script("test_tuned_runner", "scripts/run_tuned_validation.py")
        cls.targeted = load_script("test_targeted_runner", "scripts/run_targeted_quality.py")
        cls.qlsa = load_script("test_qlsa_variant_runner_shared", "scripts/run_qlsa_variant_experiments.py")

    def assert_current_row_preserved(self, parser, line: str, *args) -> None:
        rows = parser("CSV:\n" + line, *args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_csv_schema_width"], "28")
        self.assertNotEqual(rows[0]["actual_backend"], "")
        self.assertNotEqual(rows[0]["actual_threads"], "")

    def test_all_migrated_runners_accept_current_schema(self) -> None:
        omp_line = csv_line(current_values())
        cuda_line = csv_line(
            current_values(
                algorithm="sa-cuda-chain",
                parallel="cuda",
                workers=128,
                backend="cuda",
            )
        )
        qlsa_line = csv_line(current_values(algorithm="qlsa-omp"))
        cases = [
            (self.cuda_candidate.rows_from_stdout, cuda_line, ()),
            (self.cuda_sweep.rows_from_stdout, cuda_line, ()),
            (self.large_openmp.parse_rows, omp_line, ()),
            (self.large_cuda.parse_rows, cuda_line, ()),
            (self.openmp_grid.extract_csv_rows, omp_line, ()),
            (self.step5.extract_csv_rows, omp_line, ()),
            (self.tuned.extract_csv_rows, omp_line, ("sa",)),
            (self.targeted.extract_csv_rows, omp_line, ("sa",)),
            (self.qlsa.rows_from_stdout, qlsa_line, ()),
        ]
        for parser, line, args in cases:
            with self.subTest(parser=parser.__module__):
                self.assert_current_row_preserved(parser, line, *args)


if __name__ == "__main__":
    unittest.main()
