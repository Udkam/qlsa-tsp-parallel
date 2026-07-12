#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free tests for the island migration ablation pipeline."""

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


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_script("island_ablation_runner", "scripts/run_island_ablation.py")
analyzer = load_script("island_ablation_analyzer", "scripts/analyze_island_ablation.py")


def program_csv(
    *,
    seed: int = 17,
    topology: str = "ring",
    migration_interval: int = 1000,
    requested_backend: str = "openmp",
    actual_backend: str = "openmp",
    fallback: str = "false",
    fallback_reason: str = "",
    best_length: int = 101,
    attempts: int = 24,
    adopted: int = 6,
) -> str:
    values = [
        f"sa-island-{topology}",
        "square4",
        "4",
        "100",
        str(seed),
        "nn",
        "8",
        "8",
        "omp",
        str(best_length),
        str(best_length + 1),
        "2.5",
        "20",
        "10",
        "2.5",
        "0.0",
        requested_backend,
        actual_backend,
        fallback,
        fallback_reason,
        "800",
        "false",
        topology,
        str(migration_interval),
        "3" if topology != "independent" else "0",
        str(attempts if topology != "independent" else 0),
        str(adopted if topology != "independent" else 0),
        "8",
    ]
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(values)
    return stream.getvalue()


class MatrixTests(unittest.TestCase):
    def test_default_matrix_has_360_jobs_and_interval_matched_independent_baselines(self) -> None:
        config, _ = runner.load_config(ROOT / "configs/island_ablation_matrix.json")
        args = argparse.Namespace(
            instances=None,
            algorithms=None,
            topologies=None,
            migration_intervals=None,
            seed_start=None,
            seed_count=None,
            seed_stride=None,
        )
        jobs = runner.build_jobs(config, args, ROOT / "fixture-tsp-sa")
        self.assertEqual(len(jobs), 3 * 2 * 10 * (3 * 2))
        independent = [job for job in jobs if job.topology == "independent"]
        self.assertEqual(len(independent), 3 * 2 * 10 * 2)
        independent_keys = {
            (
                job.algorithm["key"],
                job.instance["name"],
                job.seed,
                job.migration_interval,
            )
            for job in independent
        }
        self.assertEqual(len(independent_keys), len(independent))
        self.assertEqual({job.migration_interval for job in independent}, {10000, 100000})
        self.assertTrue(
            all(job.migration_interval == job.command_migration_interval for job in independent)
        )
        self.assertTrue(all("--migration-topology" in job.command for job in jobs))
        paper_sb_commands = [job.command for job in jobs if job.algorithm["key"] == "paper-sb"]
        self.assertTrue(paper_sb_commands)
        for command in paper_sb_commands:
            metric_index = command.index("--diversity_metric")
            self.assertEqual(command[metric_index + 1], "hamming")

    def test_subset_and_dry_run_need_no_project_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "must-not-exist"
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                code = runner.main(
                    [
                        "--dry-run",
                        "--executable",
                        str(Path(temporary) / "not-built.exe"),
                        "--instances",
                        "eil76",
                        "--algorithms",
                        "sa",
                        "--seed-count",
                        "1",
                        "--topologies",
                        "independent",
                        "ring",
                        "--migration-intervals",
                        "10000",
                        "--output-root",
                        str(output_root),
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue().splitlines()[0])
            self.assertEqual(summary["job_count"], 2)
            self.assertFalse(output_root.exists())


class ProgramRowAndBackendTests(unittest.TestCase):
    def test_parser_requires_the_current_28_column_schema(self) -> None:
        row = runner.parse_program_row(program_csv())
        self.assertEqual(row["migration_topology"], "ring")
        self.assertEqual(row["migrations_adopted"], "6")
        short = ",".join(program_csv().split(",")[:-1])
        with self.assertRaisesRegex(ValueError, "28-column"):
            runner.parse_program_row(short)

    def test_fallback_is_rejected_and_logged_without_a_real_project_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "logs").mkdir()
            stdout = program_csv(
                actual_backend="cpu_serial",
                fallback="true",
                fallback_reason="OpenMP unavailable",
            )
            completed = subprocess.CompletedProcess(["fixture"], 0, stdout=stdout, stderr="")
            job = runner.Job(
                job_id="fallback-fixture",
                algorithm={"key": "sa", "display_name": "SA", "qlsa": False},
                instance={"name": "square4", "bks": 4},
                seed=17,
                seed_index=0,
                topology="ring",
                migration_interval=1000,
                command_migration_interval=1000,
                iterations=100,
                chains=8,
                threads=8,
                requested_backend="omp",
                require_backend_match=True,
                command=["fixture-tsp-sa", "--csv-only"],
            )
            with mock.patch.object(runner.subprocess, "run", return_value=completed):
                raw, manifest = runner.run_job(
                    job,
                    {"experiment_name": "fixture", "execution": {}},
                    run_dir,
                    {"commit": "deadbeef", "dirty": False},
                    "config-hash",
                    "executable-hash",
                    "input-hash",
                    10.0,
                )
            self.assertEqual(raw["status"], "error")
            self.assertEqual(raw["actual_backend"], "serial")
            self.assertIn("backend fallback was reported", raw["error"])
            self.assertIn("requested backend 'omp', actual backend 'serial'", raw["error"])
            self.assertEqual(manifest["status"], "error")
            self.assertTrue((run_dir / "logs/fallback-fixture.log").is_file())


class AnalyzerTests(unittest.TestCase):
    @staticmethod
    def make_raw_row(
        *,
        seed: int,
        topology: str,
        interval: int,
        best: int,
        runtime: float,
        attempts: int = 0,
        adopted: int = 0,
        fallback: str = "false",
    ) -> dict[str, str]:
        row = {field: "" for field in runner.RAW_HEADER}
        row.update(
            {
                "job_id": f"{topology}-{interval}-{seed}",
                "status": "ok",
                "experiment_name": "fixture",
                "algorithm_key": "sa",
                "instance": "fixture100",
                "bks": "100",
                "seed": str(seed),
                "topology": topology,
                "migration_interval": str(interval),
                "requested_backend": "omp",
                "reported_requested_backend": "omp",
                "actual_backend": "omp",
                "require_backend_match": "true",
                "backend_fallback": fallback,
                "threads": "8",
                "actual_threads": "8",
                "best_length": str(best),
                "elapsed_ms": str(runtime + 1),
                "total_elapsed_ms": str(runtime),
                "iterations_completed": "800",
                "iterations_reported": "100",
                "migration_attempts": str(attempts),
                "migrations_adopted": str(adopted),
            }
        )
        return row

    @staticmethod
    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.RAW_HEADER)
            writer.writeheader()
            writer.writerows(rows)

    def test_summary_pairing_and_exact_sign_test(self) -> None:
        rows: list[dict[str, str]] = []
        for seed, independent_best, migration_best in [
            (1, 110, 109),
            (2, 108, 106),
            (3, 107, 106),
        ]:
            rows.append(
                self.make_raw_row(
                    seed=seed,
                    topology="independent",
                    interval=1000,
                    best=independent_best,
                    runtime=10 + seed,
                )
            )
            rows.append(
                self.make_raw_row(
                    seed=seed,
                    topology="ring",
                    interval=1000,
                    best=migration_best,
                    runtime=12 + seed,
                    attempts=24,
                    adopted=6,
                )
            )
            rows.append(
                self.make_raw_row(
                    seed=seed,
                    topology="global",
                    interval=1000,
                    best=independent_best,
                    runtime=13 + seed,
                    attempts=24,
                    adopted=3,
                )
            )
        summaries = analyzer.group_summary(rows)
        seed_pairs, paired = analyzer.paired_analysis(rows)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(len(seed_pairs), 6)
        self.assertEqual(len(paired), 2)
        ring = next(item for item in paired if item["topology"] == "ring")
        self.assertEqual(ring["migration_wins"], 3)
        self.assertEqual(ring["migration_losses"], 0)
        self.assertAlmostEqual(float(ring["exact_sign_test_p_two_sided"]), 0.25)
        self.assertAlmostEqual(float(ring["migration_adoption_rate"]), 0.25)

    def test_analyzer_writes_all_artifacts_and_rejects_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            raw_path = temp / "raw.csv"
            output_dir = temp / "analysis"
            rows = [
                self.make_raw_row(
                    seed=1, topology="independent", interval=10000, best=105, runtime=10
                ),
                self.make_raw_row(
                    seed=1,
                    topology="ring",
                    interval=10000,
                    best=104,
                    runtime=10.5,
                    attempts=8,
                    adopted=1,
                ),
                self.make_raw_row(
                    seed=1,
                    topology="global",
                    interval=10000,
                    best=103,
                    runtime=11,
                    attempts=8,
                    adopted=2,
                ),
            ]
            self.write_rows(raw_path, rows)
            code = analyzer.main([str(raw_path), "--output-dir", str(output_dir)])
            self.assertEqual(code, 0)
            for filename in (
                "summary.csv",
                "paired_seed_differences.csv",
                "paired_comparisons.csv",
                "analysis.json",
                "analysis_checksums.sha256",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)

            rows[0]["backend_fallback"] = "true"
            self.write_rows(raw_path, rows)
            with self.assertRaisesRegex(ValueError, "backend fallback"):
                analyzer.read_raw_rows(raw_path)


if __name__ == "__main__":
    unittest.main()
