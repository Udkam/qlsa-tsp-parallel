#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed contract tests for the island-ablation runner."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "island_runner_hardening_subject", ROOT / "scripts/run_island_ablation.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import island runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "island_runner_bundle_analyzer", ROOT / "scripts/analyze_island_ablation.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import island analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()


def csv_line(values: list[str]) -> str:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(values)
    return stream.getvalue()


def make_job(
    *,
    topology: str = "ring",
    interval: int = 10,
    algorithm: str = "sa",
) -> object:
    algorithm_config: dict[str, object] = {
        "key": algorithm,
        "display_name": algorithm,
        "qlsa": algorithm != "sa",
    }
    if algorithm != "sa":
        algorithm_config["qlsa_variant"] = algorithm
    command = [
        "fixture-tsp-sa",
        "--init",
        "nn",
        "--migration-topology",
        topology,
        "--migration-interval",
        str(interval),
        "--csv-only",
    ]
    return runner.Job(
        job_id="validation-fixture",
        algorithm=algorithm_config,
        instance={"name": "square4", "dimension": 4, "bks": 4},
        seed=17,
        seed_index=0,
        topology=topology,
        migration_interval=interval,
        command_migration_interval=interval,
        iterations=100,
        chains=4,
        threads=4,
        requested_backend="omp",
        require_backend_match=True,
        command=command,
        execution_order=1,
    )


def valid_program(job=None, overrides: dict[str, str] | None = None) -> str:
    job = job or make_job()
    rounds, attempts = runner.expected_migration_counters(job)
    row = {
        "algorithm": runner.expected_algorithm_label(job),
        "instance": "square4",
        "dimension": "4",
        "iterations": "100",
        "seed": "17",
        "init": "nn",
        "chains": "4",
        "threads": "4",
        "parallel": "omp",
        "best_length": "4",
        "final_length": "4",
        "elapsed_ms": "25.0",
        "accepted_moves": "80",
        "improved_moves": "20",
        "total_elapsed_ms": "25.0",
        "cuda_kernel_elapsed_ms": "0.0",
        "requested_backend": "openmp",
        "actual_backend": "openmp",
        "backend_fallback": "false",
        "backend_fallback_reason": "",
        "iterations_completed": "400",
        "deadline_reached": "false",
        "migration_topology": job.topology,
        "migration_interval": str(job.command_migration_interval),
        "migration_rounds": str(rounds),
        "migration_attempts": str(attempts),
        "migrations_adopted": "0",
        "actual_threads": "4",
    }
    if overrides:
        row.update(overrides)
    return csv_line([row[field] for field in runner.PROGRAM_HEADER])


class ExecutablePolicyTests(unittest.TestCase):
    def test_formal_run_requires_explicit_executable_but_dry_run_does_not(self) -> None:
        config = {"executable_candidates": [str(Path(sys.executable).resolve())]}
        with self.assertRaisesRegex(FileNotFoundError, "--executable is required"):
            runner.find_executable(config, None, False)
        self.assertEqual(
            runner.find_executable(config, None, True), Path(sys.executable).resolve()
        )
        self.assertEqual(
            runner.find_executable(config, Path(sys.executable), False),
            Path(sys.executable).resolve(),
        )


class StrictProgramCsvTests(unittest.TestCase):
    def test_every_28_column_field_is_typed_and_errors_are_aggregated(self) -> None:
        invalid = valid_program(
            overrides={
                "algorithm": "unknown",
                "dimension": "0",
                "threads": "four",
                "elapsed_ms": "nan",
                "requested_backend": "mystery",
                "backend_fallback": "yes",
                "deadline_reached": "sometimes",
                "migration_attempts": "1",
                "migrations_adopted": "2",
                "actual_threads": "0",
            }
        )
        with self.assertRaises(ValueError) as caught:
            runner.parse_program_row(invalid)
        message = str(caught.exception)
        for fragment in (
            "algorithm has unsupported value",
            "dimension must be >= 1",
            "threads must be an integer",
            "elapsed_ms must be finite",
            "requested_backend has unsupported value",
            "backend_fallback must be true or false",
            "deadline_reached must be true or false",
            "migrations_adopted must not exceed migration_attempts",
            "actual_threads must be >= 1",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, message)

    def test_only_one_strict_28_column_row_is_accepted(self) -> None:
        values = next(csv.reader([valid_program()]))
        with self.assertRaisesRegex(ValueError, "28-column"):
            runner.parse_program_row(csv_line(values[:-1]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.parse_program_row(valid_program() + "\n" + valid_program())


class SemanticValidationTests(unittest.TestCase):
    def test_all_job_mismatches_are_reported_together(self) -> None:
        job = make_job(topology="ring", interval=10)
        program = runner.parse_program_row(
            valid_program(
                job,
                {
                    "algorithm": "qlsa-paper-sb-island-global-omp",
                    "instance": "other",
                    "dimension": "5",
                    "iterations": "99",
                    "seed": "18",
                    "init": "random",
                    "chains": "3",
                    "threads": "3",
                    "parallel": "cuda",
                    "requested_backend": "cuda",
                    "actual_backend": "cuda",
                    "backend_fallback": "true",
                    "backend_fallback_reason": "fixture fallback",
                    "iterations_completed": "297",
                    "deadline_reached": "true",
                    "migration_topology": "global",
                    "migration_interval": "11",
                    "migration_rounds": "8",
                    "migration_attempts": "16",
                    "migrations_adopted": "1",
                    "actual_threads": "3",
                },
            )
        )
        raw = runner.base_raw_row(
            job,
            {"experiment_name": "fixture"},
            {"commit": "deadbeef", "dirty": False},
            "config-hash",
            "input-hash",
            "executable-hash",
            Path("logs/fixture.log"),
        )
        errors = runner.validate_and_copy_program_row(
            job, program, raw, {"execution": {"init": "nn"}}
        )
        message = "; ".join(errors)
        for fragment in (
            "reported algorithm",
            "reported seed",
            "reported instance",
            "reported dimension",
            "reported iterations",
            "iterations_completed",
            "deadline_reached=true",
            "reported init",
            "reported chains",
            "reported requested threads",
            "actual threads",
            "reported parallel",
            "program requested backend",
            "actual backend",
            "backend fallback",
            "reported topology",
            "reported migration interval",
            "migration_rounds",
            "migration_attempts",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, message)
        self.assertGreaterEqual(message.count(";"), 18)

    def test_valid_row_records_environment_and_exact_work(self) -> None:
        job = make_job()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "logs").mkdir()
            completed = subprocess.CompletedProcess(
                ["fixture"], 0, stdout=valid_program(job), stderr=""
            )
            with mock.patch.object(runner.subprocess, "run", return_value=completed):
                raw, manifest_job = runner.run_job(
                    job,
                    {"experiment_name": "fixture", "execution": {"init": "nn"}},
                    run_dir,
                    {"commit": "deadbeef", "dirty": False},
                    "config-hash",
                    "executable-hash",
                    "input-hash",
                    10.0,
                    "environment-hash",
                )
        self.assertEqual(raw["status"], "ok", raw["error"])
        self.assertEqual(raw["iterations_completed"], "400")
        self.assertEqual(raw["environment_sha256"], "environment-hash")
        self.assertEqual(raw["execution_order"], 1)
        self.assertEqual(manifest_job["result"]["environment_sha256"], "environment-hash")


class MatrixOrderingTests(unittest.TestCase):
    def test_instance_name_cannot_escape_log_directory(self) -> None:
        config = json.loads(
            (ROOT / "configs/island_ablation_matrix.json").read_text(encoding="utf-8")
        )
        config["instances"][0]["name"] = "..\\escaped"
        with self.assertRaisesRegex(ValueError, "safe ASCII path component"):
            runner.validate_config(config)

    def test_conditions_and_algorithms_use_reproducible_cyclic_rotations(self) -> None:
        config, _ = runner.load_config(ROOT / "configs/island_ablation_matrix.json")
        args = argparse.Namespace(
            instances=["eil76"],
            algorithms=["sa", "paper-sb"],
            topologies=None,
            migration_intervals=None,
            seed_start=17,
            seed_count=3,
            seed_stride=1,
        )
        jobs = runner.build_jobs(config, args, Path("fixture-tsp-sa"))
        self.assertEqual(len(jobs), 3 * 6 * 2)

        expected_conditions = [
            ("independent", 10000),
            ("independent", 100000),
            ("ring", 10000),
            ("ring", 100000),
            ("global", 10000),
            ("global", 100000),
        ]
        for seed_index in range(3):
            seed_jobs = [job for job in jobs if job.seed_index == seed_index]
            self.assertEqual(
                [job.execution_order for job in seed_jobs], list(range(1, 13))
            )
            observed_conditions = [
                (seed_jobs[index].topology, seed_jobs[index].migration_interval)
                for index in range(0, len(seed_jobs), 2)
            ]
            rotation = seed_index % len(expected_conditions)
            self.assertEqual(
                observed_conditions,
                expected_conditions[rotation:] + expected_conditions[:rotation],
            )
            for condition_index in range(6):
                pair = seed_jobs[condition_index * 2 : condition_index * 2 + 2]
                self.assertEqual({job.algorithm["key"] for job in pair}, {"sa", "paper-sb"})

        for seed in (17, 18, 19):
            seed_jobs = [job for job in jobs if job.seed == seed]
            keys = {
                (
                    job.algorithm["key"],
                    job.topology,
                    job.migration_interval,
                )
                for job in seed_jobs
            }
            self.assertEqual(len(keys), 12)

        for condition in expected_conditions:
            first_algorithms = []
            for seed_index in range(3):
                seed_jobs = [job for job in jobs if job.seed_index == seed_index]
                matching = [
                    job
                    for job in seed_jobs
                    if (job.topology, job.migration_interval) == condition
                ]
                self.assertEqual(len(matching), 2)
                first_algorithms.append(matching[0].algorithm["key"])
            self.assertEqual(first_algorithms, ["sa", "paper-sb", "sa"])


class ArtifactIntegrityTests(unittest.TestCase):
    def test_manifest_raw_environment_and_checksum_scope_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            output_root = temp / "results"
            config_path = temp / "matrix.json"
            config = {
                "schema_version": 1,
                "experiment_name": "island-fixture",
                "instances": [
                    {
                        "name": "square4",
                        "path": str((ROOT / "tests/fixtures/square4.tsp").resolve()),
                        "dimension": 4,
                        "bks": 4,
                    }
                ],
                "algorithms": [
                    {"key": "sa", "display_name": "SA", "qlsa": False},
                    {
                        "key": "paper-sb",
                        "display_name": "paper-sb",
                        "qlsa": True,
                        "qlsa_variant": "paper-sb",
                    },
                ],
                "paired_seeds": {"start": 17, "count": 1, "stride": 1},
                "execution": {
                    "parallel": "omp",
                    "chains": 4,
                    "threads": 4,
                    "iterations_per_island": 100,
                    "init": "nn",
                    "require_backend_match": True,
                },
                "migration": {
                    "topologies": ["independent", "ring", "global"],
                    "intervals": [10],
                },
                "output_root": str(output_root),
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            fixture_output = valid_program(make_job(topology="independent", interval=10))

            def fake_run(command, **_kwargs):
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(command, 0, "deadbeef\n", "")
                if command[:3] == ["git", "status", "--porcelain=v1"]:
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(command, 0, fixture_output, "")

            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                code = runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--executable",
                        sys.executable,
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "artifact-fixture",
                        "--instances",
                        "square4",
                        "--algorithms",
                        "sa",
                        "--topologies",
                        "independent",
                        "--migration-intervals",
                        "10",
                    ]
                )
            self.assertEqual(code, 0)

            run_dir = output_root / "artifact-fixture"
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            with (run_dir / "raw.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["environment_sha256"], manifest["environment_sha256"])
            self.assertEqual(manifest["jobs"][0]["execution_order"], 1)

            checksum_entries: dict[str, str] = {}
            for line in (run_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                checksum_entries[relative] = digest
            expected = {
                "config.snapshot.json",
                "raw.csv",
                "manifest.json",
                manifest["jobs"][0]["log_file"],
            }
            self.assertEqual(set(checksum_entries), expected)
            self.assertEqual(set(manifest["artifacts"]["checksums"]["covers"]), expected)
            for relative, digest in checksum_entries.items():
                self.assertEqual(runner.sha256_file(run_dir / relative), digest)

            removed_log = manifest["jobs"][0]["log_file"]
            manifest["artifacts"]["checksums"]["covers"].remove(removed_log)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            remaining = [
                run_dir / "config.snapshot.json",
                run_dir / "raw.csv",
                run_dir / "manifest.json",
            ]
            runner.write_checksum_sidecar(run_dir, remaining)
            (run_dir / removed_log).unlink()
            with self.assertRaisesRegex(ValueError, "required config/raw/manifest/log set"):
                analyzer.read_inputs([run_dir])


if __name__ == "__main__":
    unittest.main()
