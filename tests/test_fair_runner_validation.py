#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed validation tests for the formal fair-experiment runner."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "fair_runner_validation_subject", ROOT / "scripts/run_fair_experiments.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import fair experiment runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def csv_line(values: list[str]) -> str:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(values)
    return stream.getvalue()


def valid_program(overrides: dict[str, str] | None = None, width: int = 28) -> str:
    row = {
        "algorithm": "sa-omp",
        "instance": "square4",
        "dimension": "4",
        "iterations": "100",
        "seed": "17",
        "init": "nn",
        "chains": "8",
        "threads": "8",
        "parallel": "omp",
        "best_length": "4",
        "final_length": "4",
        "elapsed_ms": "1000.0",
        "accepted_moves": "200",
        "improved_moves": "100",
        "total_elapsed_ms": "1000.0",
        "cuda_kernel_elapsed_ms": "0.0",
        "requested_backend": "openmp",
        "actual_backend": "openmp",
        "backend_fallback": "false",
        "backend_fallback_reason": "",
        "iterations_completed": "800",
        "deadline_reached": "false",
        "migration_topology": "disabled",
        "migration_interval": "0",
        "migration_rounds": "0",
        "migration_attempts": "0",
        "migrations_adopted": "0",
        "actual_threads": "8",
    }
    if overrides:
        row.update(overrides)
    header = {
        14: runner.PROGRAM_HEADER,
        22: runner.PROGRAM_EXTENDED_HEADER,
        27: runner.PROGRAM_MIGRATION_HEADER,
        28: runner.PROGRAM_ACTUAL_THREADS_HEADER,
    }[width]
    return csv_line([row[field] for field in header])


def make_job(
    command: list[str],
    *,
    budget: str = "equal-iterations",
    time_limit_ms: int | None = None,
    require_backend_match: bool = True,
) -> object:
    return runner.Job(
        job_id="validation-fixture",
        budget_scheme=budget,
        budget_target=time_limit_ms or 100,
        budget_unit="solver_wall_time_ms" if budget == "fixed-time" else "iterations_per_chain",
        algorithm={"key": "sa", "display_name": "SA", "proposal_cost_per_iteration": 1},
        instance={"name": "square4", "bks": 4},
        seed=17,
        seed_index=0,
        iterations=100,
        time_limit_ms=time_limit_ms,
        proposals_per_chain=100,
        proposals_total=800,
        chains=8,
        threads=8,
        requested_backend="omp",
        require_backend_match=require_backend_match,
        command=command,
        execution_order=1,
    )


def execute_fixture(program_output: str, **job_options):
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        (run_dir / "logs").mkdir()
        command = [sys.executable, "-c", f"print({program_output!r})"]
        job = make_job(command, **job_options)
        return runner.run_job(
            job,
            {
                "experiment_name": "fixture",
                "execution": {"init": "nn", "fixed_time_tolerance_ms": 50},
            },
            run_dir,
            {"commit": "deadbeef", "dirty": False},
            "config-hash",
            "executable-hash",
            "input-hash",
            10.0,
            "environment-hash",
        )


class ProgramCsvSchemaTests(unittest.TestCase):
    def test_only_known_widths_are_accepted(self) -> None:
        for width in (14, 22, 27, 28):
            with self.subTest(width=width):
                parsed = runner.parse_program_row(valid_program(width=width))
                self.assertEqual(len(parsed), width)

        full = next(csv.reader([valid_program(width=28)]))
        for width in (15, 16, 21, 23, 24, 26, 29):
            with self.subTest(width=width):
                values = (full + ["extra"] * width)[:width]
                with self.assertRaisesRegex(ValueError, "unsupported CSV column count"):
                    runner.parse_program_row(csv_line(values))

    def test_every_typed_field_is_validated(self) -> None:
        invalid = (
            ({"threads": "eight"}, "threads must be an integer"),
            ({"elapsed_ms": "nan"}, "elapsed_ms must be finite"),
            ({"backend_fallback": "yes"}, "backend_fallback must be true or false"),
            ({"actual_threads": "0"}, "actual_threads must be >= 1"),
            (
                {"migration_attempts": "1", "migrations_adopted": "2"},
                "migrations_adopted must not exceed migration_attempts",
            ),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    runner.parse_program_row(valid_program(overrides))


class StrictRunValidationTests(unittest.TestCase):
    def test_strict_mode_rejects_legacy_and_transitional_metadata(self) -> None:
        for width in (14, 22, 27):
            with self.subTest(width=width):
                raw, _ = execute_fixture(valid_program(width=width))
                self.assertEqual(raw["status"], "error")
                self.assertIn("strict backend validation requires", raw["error"])

    def test_semantic_mismatches_are_aggregated(self) -> None:
        output = valid_program(
            {
                "algorithm": "qlsa-paper-omp",
                "iterations": "99",
                "init": "random",
                "chains": "7",
                "threads": "9",
                "actual_threads": "9",
                "parallel": "cuda",
                "requested_backend": "cuda",
                "actual_backend": "cuda",
                "iterations_completed": "693",
            }
        )
        raw, _ = execute_fixture(output)
        self.assertEqual(raw["status"], "error")
        for fragment in (
            "reported algorithm",
            "reported iterations",
            "reported chains",
            "reported effective threads",
            "observed actual_threads",
            "reported init",
            "reported parallel mode",
            "program reported requested backend",
            "actual backend",
            "equal-iterations completed",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, raw["error"])
        self.assertGreaterEqual(raw["error"].count(";"), 8)

    def test_equal_budget_requires_exact_completion_without_deadline(self) -> None:
        raw, _ = execute_fixture(
            valid_program({"iterations_completed": "799", "deadline_reached": "true"})
        )
        self.assertEqual(raw["status"], "error")
        self.assertIn("expected exactly 800", raw["error"])
        self.assertIn("unexpectedly reported deadline_reached=true", raw["error"])
        self.assertEqual(raw["proposal_evaluations_actual_total"], 799)

    def test_fixed_time_checks_deadline_and_reported_total_time(self) -> None:
        raw, _ = execute_fixture(
            valid_program(
                {
                    "iterations_completed": "700",
                    "deadline_reached": "false",
                    "total_elapsed_ms": "1500.0",
                }
            ),
            budget="fixed-time",
            time_limit_ms=1000,
        )
        self.assertEqual(raw["status"], "error")
        self.assertIn("without deadline_reached=true", raw["error"])
        self.assertIn("tolerance 50.000 ms", raw["error"])
        self.assertEqual(raw["fixed_time_elapsed_delta_ms"], "500.000")
        self.assertEqual(raw["proposal_evaluations_actual_total"], 700)

    def test_valid_modern_row_records_environment_and_actual_work(self) -> None:
        raw, manifest = execute_fixture(valid_program())
        self.assertEqual(raw["status"], "ok", raw["error"])
        self.assertEqual(raw["environment_sha256"], "environment-hash")
        self.assertEqual(raw["proposal_evaluations_actual_total"], 800)
        self.assertEqual(raw["actual_threads"], "8")
        self.assertEqual(manifest["status"], "ok")


class MatrixConstructionTests(unittest.TestCase):
    def test_instance_name_cannot_escape_log_directory(self) -> None:
        config = json.loads(
            (ROOT / "configs/fair_experiment_matrix.json").read_text(encoding="utf-8")
        )
        config["instances"][0]["name"] = "../escaped"
        with self.assertRaisesRegex(ValueError, "safe ASCII path component"):
            runner.validate_config(config)

    def test_executable_must_be_explicit_unless_auto_discovery_is_opted_in(self) -> None:
        config = {"executable_candidates": [str(Path(sys.executable).resolve())]}
        with self.assertRaisesRegex(FileNotFoundError, "--executable is required"):
            runner.find_executable(config, None)
        self.assertEqual(
            runner.find_executable(config, None, allow_auto_executable=True),
            Path(sys.executable).resolve(),
        )

    def test_algorithms_use_reproducible_cyclic_latin_order(self) -> None:
        algorithms = [
            {"key": "sa", "qlsa": False, "proposal_cost_per_iteration": 1},
            {
                "key": "current",
                "qlsa": True,
                "qlsa_variant": "current",
                "proposal_cost_per_iteration": 1,
            },
            {
                "key": "paper",
                "qlsa": True,
                "qlsa_variant": "paper",
                "proposal_cost_per_iteration": 1,
            },
            {
                "key": "paper-sb",
                "qlsa": True,
                "qlsa_variant": "paper-sb",
                "proposal_cost_per_iteration": 1,
            },
        ]
        config = {
            "instances": [
                {"name": "square4", "path": "tests/fixtures/square4.tsp", "bks": 4}
            ],
            "algorithms": algorithms,
            "paired_seeds": {"start": 17, "count": 4, "stride": 1},
            "execution": {
                "parallel": "omp",
                "chains": 8,
                "threads": 8,
                "init": "nn",
                "require_backend_match": True,
            },
            "default_budget": "equal-iterations",
            "budgets": {
                "equal-iterations": {"iterations_per_chain": 100},
                "fixed-time": {"time_limit_ms": 1000, "iterations_ceiling_per_chain": 1000},
            },
        }
        args = runner.parse_args(
            ["--executable", sys.executable, "--budget", "equal-iterations"]
        )
        jobs = runner.build_jobs(config, args, Path(sys.executable))
        keys = [job.algorithm["key"] for job in jobs]
        self.assertEqual(keys[0:4], ["sa", "current", "paper", "paper-sb"])
        self.assertEqual(keys[4:8], ["current", "paper", "paper-sb", "sa"])
        self.assertEqual(keys[8:12], ["paper", "paper-sb", "sa", "current"])
        self.assertEqual(keys[12:16], ["paper-sb", "sa", "current", "paper"])
        for start in range(0, 16, 4):
            self.assertEqual([job.execution_order for job in jobs[start : start + 4]], [1, 2, 3, 4])

    def test_deprecated_equal_proposals_budget_is_rejected(self) -> None:
        config = {
            "schema_version": 1,
            "experiment_name": "fixture",
            "instances": [{"name": "x", "path": "x", "bks": 1}],
            "algorithms": [
                {"key": key, "qlsa_variant": key, "proposal_cost_per_iteration": 1}
                if key != "sa"
                else {"key": "sa", "proposal_cost_per_iteration": 1}
                for key in runner.ALGORITHM_KEYS
            ],
            "paired_seeds": {"start": 1, "count": 1, "stride": 1},
            "execution": {"parallel": "omp", "chains": 1, "threads": 1},
            "budgets": {
                "equal-iterations": {"iterations_per_chain": 1},
                "equal-proposals": {"proposal_evaluations_per_chain": 1},
                "fixed-time": {"time_limit_ms": 1, "iterations_ceiling_per_chain": 1},
            },
            "default_budget": "equal-iterations",
        }
        with self.assertRaisesRegex(ValueError, "equal-proposals is deprecated"):
            runner.validate_config(config)


if __name__ == "__main__":
    unittest.main()
