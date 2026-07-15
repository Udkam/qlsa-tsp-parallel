"""Guard against treating normal CLI accounting fields as MPI fields."""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiment_csv import CURRENT_HEADER
from scripts.mpi_csv import (
    MpiCsvError,
    MpiExecutionContract,
    parse_mpi_program_rows,
    validate_mpi_execution_contract,
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


runner = load_script("test_mpi_smoke_runner", "scripts/run_mpi_smoke.py")
large_runner = load_script("test_large_mpi_vm_runner", "scripts/run_large_mpi_vm.py")
scaling_runner = load_script("test_mpi_vm_scaling_runner", "scripts/run_mpi_vm_scaling.py")


def csv_row(values: list[str]) -> list[str]:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(values)
    return next(csv.reader([stream.getvalue()]))


def csv_text(values: list[str]) -> str:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(values)
    return stream.getvalue()


def openmp_current_row() -> list[str]:
    values = {field: "" for field in CURRENT_HEADER}
    values.update(
        {
            "algorithm": "sa-omp",
            "instance": "square4",
            "dimension": "4",
            "iterations": "5",
            "seed": "1",
            "init": "nn",
            "chains": "2",
            "threads": "2",
            "parallel": "omp",
            "best_length": "40",
            "final_length": "40",
            "elapsed_ms": "1.000",
            "accepted_moves": "4",
            "improved_moves": "2",
            "total_elapsed_ms": "1.000",
            "cuda_kernel_elapsed_ms": "0.000",
            "requested_backend": "openmp",
            "actual_backend": "openmp",
            "backend_fallback": "false",
            "backend_fallback_reason": "",
            "iterations_completed": "10",
            "deadline_reached": "false",
            "migration_topology": "disabled",
            "migration_interval": "0",
            "migration_rounds": "0",
            "migration_attempts": "0",
            "migrations_adopted": "0",
            "actual_threads": "2",
        }
    )
    return csv_row([values[field] for field in CURRENT_HEADER])


def mpi_row(*, seed: int = 1, ranks: int = 2, threads: int = 2, chains: int = 2) -> list[str]:
    values = openmp_current_row()[:14]
    values[0] = "sa-mpi-omp"
    values[3] = "5"
    values[4] = str(seed)
    values[6] = str(chains)
    values[7] = str(threads)
    values[8] = "mpi-omp"
    return values + [
        str(ranks),
        "0.125",
        str(threads),
        str(5 * chains),
        "false",
    ]


class MpiSmokeSchemaTests(unittest.TestCase):
    def test_smoke_discovers_preset_outputs_and_respects_explicit_mpi_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpu = root / "build" / "ninja-cpu-release" / "tsp_sa.exe"
            mpi = root / "build" / "ninja-mpi-release" / "tsp_sa_mpi.exe"
            cpu.parent.mkdir(parents=True)
            mpi.parent.mkdir(parents=True)
            cpu.write_bytes(b"cpu")
            mpi.write_bytes(b"mpi")
            with mock.patch.object(runner, "ROOT", root):
                self.assertEqual(runner.find_tsp_sa(), cpu)
                self.assertEqual(runner.find_tsp_sa_mpi(), mpi)
                self.assertEqual(runner.resolve_mpi_executable(mpi), mpi.resolve())

    def test_openmp_fallback_keeps_caller_supplied_mpi_defaults(self) -> None:
        row = runner.normalize_row(
            openmp_current_row(),
            mode="openmp-fallback",
            fallback=True,
            mpi_ranks=1,
            communication_ms=0.0,
            speedup=None,
            efficiency=None,
        )
        self.assertEqual(row["mpi_ranks"], "1")
        self.assertEqual(row["communication_ms"], "0.000")
        self.assertEqual(row["parallel"], "omp")

    def test_real_mpi_row_uses_extended_accounting_fields(self) -> None:
        mpi = mpi_row()
        mpi[0] = "sa-mpi-omp"
        mpi[8] = "mpi-omp"
        row = runner.normalize_row(
            mpi,
            mode="mpi",
            fallback=False,
            mpi_ranks=99,
            communication_ms=99.0,
            speedup=None,
            efficiency=None,
        )
        self.assertEqual(row["mpi_ranks"], "2")
        self.assertEqual(row["communication_ms"], "0.125")
        self.assertEqual(row["actual_threads"], "2")
        self.assertEqual(row["iterations_completed"], "10")
        self.assertEqual(row["deadline_reached"], "false")

    def test_malformed_mpi_row_fails_closed(self) -> None:
        malformed = openmp_current_row()[:14] + ["2"]
        malformed[0] = "sa-mpi-omp"
        malformed[8] = "mpi-omp"
        with self.assertRaises(RuntimeError):
            runner.normalize_row(
                malformed,
                mode="mpi",
                fallback=False,
                mpi_ranks=2,
                communication_ms=0.0,
                speedup=None,
                efficiency=None,
            )

    def test_large_mpi_runners_reject_normal_cli_schema_and_preserve_mpi_fields(self) -> None:
        stdout = "CSV:\n" + csv_text(mpi_row())
        large_rows = large_runner.parse_rows(stdout, "sa")
        scaling_rows = scaling_runner.csv_rows(stdout, "sa")
        self.assertEqual(large_rows[0]["mpi_ranks"], "2")
        self.assertEqual(large_rows[0]["communication_ms"], "0.125")
        self.assertEqual(scaling_rows[0]["algorithm"], "sa-mpi-omp")
        normalized = scaling_runner.normalize(7, scaling_rows[0], Path("hosts"))
        self.assertEqual(normalized["mpi_ranks"], "2")

        normal_cli_stdout = "CSV:\n" + csv_text(openmp_current_row())
        with self.assertRaises(MpiCsvError):
            large_runner.parse_rows(normal_cli_stdout, "sa")
        with self.assertRaises(MpiCsvError):
            scaling_runner.csv_rows(normal_cli_stdout, "sa")

    def test_mpi_execution_contract_requires_repeat_budget_threads_and_ranks(self) -> None:
        good_rows = parse_mpi_program_rows(
            "CSV:\n" + csv_text(mpi_row(seed=1)) + "\n" + csv_text(mpi_row(seed=2)),
            algorithm="sa",
        )
        contract = MpiExecutionContract(
            algorithm="sa",
            iterations=5,
            chains=2,
            threads=2,
            ranks=2,
            repeat=2,
            seed=1,
        )
        validate_mpi_execution_contract(good_rows, contract)

        for field, value in (("iterations", "4"), ("chains", "3"), ("threads", "1"), ("mpi_ranks", "1")):
            with self.subTest(field=field):
                bad_rows = [dict(row) for row in good_rows]
                bad_rows[0][field] = value
                with self.assertRaises(MpiCsvError):
                    validate_mpi_execution_contract(bad_rows, contract)

        with self.assertRaises(MpiCsvError):
            validate_mpi_execution_contract(good_rows[:1], contract)

        for field, value in (
            ("actual_threads", "1"),
            ("iterations_completed", "9"),
            ("deadline_reached", "true"),
            ("init", "random"),
        ):
            with self.subTest(accounting_field=field):
                bad_rows = [dict(row) for row in good_rows]
                bad_rows[0][field] = value
                with self.assertRaises(MpiCsvError):
                    validate_mpi_execution_contract(bad_rows, contract)

    def test_legacy_mpi_schema_is_readable_but_not_valid_for_new_runs(self) -> None:
        legacy = mpi_row()[:16]
        rows = parse_mpi_program_rows(csv_text(legacy), algorithm="sa")
        contract = MpiExecutionContract("sa", 5, 2, 2, 2, 1, 1)
        with self.assertRaises(MpiCsvError):
            validate_mpi_execution_contract(rows, contract)
        validate_mpi_execution_contract(rows, contract, require_current_schema=False)


if __name__ == "__main__":
    unittest.main()
