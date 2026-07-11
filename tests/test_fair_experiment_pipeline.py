#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, dependency-free fixture tests for the fair experiment pipeline."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_script("fair_experiment_runner", "scripts/run_fair_experiments.py")
analyzer = load_script("paired_experiment_analyzer", "scripts/analyze_paired_experiments.py")


class StatisticsTests(unittest.TestCase):
    def test_exact_small_sample_tests(self) -> None:
        wilcoxon = analyzer.wilcoxon_signed_rank([1.0, 2.0, 3.0])
        self.assertEqual(wilcoxon["n"], 3)
        self.assertAlmostEqual(wilcoxon["statistic"], 0.0)
        self.assertAlmostEqual(wilcoxon["p_value"], 0.25)

        sign = analyzer.exact_sign_test([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(sign["n"], 5)
        self.assertAlmostEqual(sign["p_value"], 0.0625)

        adjusted = analyzer.holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)

    def test_friedman_known_ordering(self) -> None:
        result = analyzer.friedman_test(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        self.assertAlmostEqual(result["statistic"], 9.0)
        self.assertEqual(result["degrees_of_freedom"], 3)
        self.assertAlmostEqual(result["p_value"], 0.0292908865348887)
        self.assertEqual(result["mean_ranks"], [1.0, 2.0, 3.0, 4.0])


class RunnerTests(unittest.TestCase):
    def test_program_parser_accepts_legacy_and_extended_rows(self) -> None:
        legacy_values = [
            "sa-omp",
            "square4",
            "4",
            "100",
            "17",
            "nn",
            "8",
            "8",
            "omp",
            "4",
            "4",
            "1.5",
            "20",
            "10",
        ]
        legacy = runner.parse_program_row(",".join(legacy_values))
        self.assertEqual(legacy["best_length"], "4")
        self.assertNotIn("actual_backend", legacy)

        extended_values = legacy_values + [
            "2.5",
            "1.25",
            "openmp",
            "cpu_serial",
            "true",
            "OpenMP unavailable",
            "87",
            "true",
        ]
        stream = io.StringIO()
        csv.writer(stream).writerow(extended_values)
        extended = runner.parse_program_row(stream.getvalue())
        self.assertEqual(extended["total_elapsed_ms"], "2.5")
        self.assertEqual(extended["cuda_kernel_elapsed_ms"], "1.25")
        self.assertEqual(runner.normalize_backend(extended["requested_backend"]), "omp")
        self.assertEqual(runner.normalize_backend(extended["actual_backend"]), "serial")
        self.assertEqual(extended["iterations_completed"], "87")
        self.assertEqual(extended["deadline_reached"], "true")

    def test_reported_backend_fallback_is_a_failed_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "logs").mkdir()
            values = [
                "sa-omp",
                "square4",
                "4",
                "100",
                "17",
                "nn",
                "8",
                "8",
                "omp",
                "4",
                "4",
                "1.5",
                "20",
                "10",
                "2.5",
                "0.0",
                "openmp",
                "cpu_serial",
                "true",
                "OpenMP unavailable",
                "87",
                "false",
            ]
            stream = io.StringIO()
            csv.writer(stream, lineterminator="").writerow(values)
            job = runner.Job(
                job_id="fallback-fixture",
                budget_scheme="equal-iterations",
                budget_target=100,
                budget_unit="iterations_per_chain",
                algorithm={
                    "key": "sa",
                    "display_name": "SA",
                    "proposal_cost_per_iteration": 1,
                },
                instance={"name": "square4", "bks": 4},
                seed=17,
                seed_index=0,
                iterations=100,
                time_limit_ms=None,
                proposals_per_chain=100,
                proposals_total=800,
                chains=8,
                threads=8,
                requested_backend="omp",
                require_backend_match=True,
                command=[sys.executable, "-c", f"print({stream.getvalue()!r})"],
            )
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
            self.assertEqual(manifest["status"], "error")
            self.assertTrue((run_dir / "logs/fallback-fixture.log").is_file())

    def test_dry_run_expands_all_budgets_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            config = json.loads((ROOT / "configs/fair_experiment_matrix.json").read_text(encoding="utf-8"))
            config["instances"] = [
                {
                    "name": "square4",
                    "path": "tests/fixtures/square4.tsp",
                    "bks": 4,
                }
            ]
            config["paired_seeds"] = {"start": 17, "count": 1, "stride": 1}
            config_path = temp / "matrix.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output_root = temp / "must-not-exist"
            command = [
                sys.executable,
                str(ROOT / "scripts/run_fair_experiments.py"),
                "--config",
                str(config_path),
                "--executable",
                sys.executable,
                "--output-root",
                str(output_root),
                "--budget",
                "all",
                "--algorithms",
                "sa",
                "--dry-run",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = completed.stdout.splitlines()
            summary = json.loads(lines[0])
            self.assertEqual(summary["job_count"], 2)
            self.assertEqual(summary["budgets"], list(runner.BUDGET_SCHEMES))
            fixed_line = next(line for line in lines if "fixed_time" in line)
            equal_line = next(line for line in lines if "equal_iterations" in line)
            self.assertIn("--time-limit-ms 30000", fixed_line)
            self.assertIn("--chains 8 --threads 8", fixed_line)
            self.assertIn("--chains 64 --threads 8", equal_line)
            self.assertFalse(output_root.exists())


class AnalyzerIntegrityTests(unittest.TestCase):
    @staticmethod
    def make_row(
        algorithm: str,
        seed: int,
        best_length: int,
        *,
        total_elapsed_ms: str = "10.0",
        elapsed_ms: str = "11.0",
        config_hash: str = "config-a",
    ) -> dict[str, str]:
        row = {field: "" for field in runner.RAW_HEADER}
        row.update(
            {
                "status": "ok",
                "experiment_name": "fixture",
                "budget_scheme": "equal-iterations",
                "budget_target": "100",
                "budget_unit": "iterations_per_chain",
                "algorithm_key": algorithm,
                "instance": "fixture100",
                "bks": "100",
                "seed": str(seed),
                "requested_backend": "omp",
                "actual_backend": "omp",
                "require_backend_match": "true",
                "init": "nn",
                "chains": "8",
                "threads": "8",
                "actual_threads": "8",
                "reported_parallel": "omp",
                "best_length": str(best_length),
                "elapsed_ms": elapsed_ms,
                "total_elapsed_ms": total_elapsed_ms,
                "config_sha256": config_hash,
                "input_sha256": "input-a",
                "executable_sha256": "exe-a",
                "environment_sha256": "env-a",
                "_source": "fixture.csv",
                "_line": str(seed),
            }
        )
        row["_condition_id"] = analyzer.condition_id_for_row(row)
        return row

    def test_condition_id_separates_configurations(self) -> None:
        first = self.make_row("sa", 1, 110, config_hash="config-a")
        second = self.make_row("sa", 1, 110, config_hash="config-b")
        self.assertNotEqual(first["_condition_id"], second["_condition_id"])
        analyzer.validate_unique_rows([first, second])

    def test_multiple_inputs_require_complete_provenance(self) -> None:
        row = self.make_row("sa", 1, 110)
        row["environment_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "complete provenance"):
            analyzer.validate_multi_input_provenance([row], 2)

    def test_summary_bootstrap_is_input_order_invariant(self) -> None:
        rows = [
            self.make_row("sa", 3, 108),
            self.make_row("sa", 1, 110),
            self.make_row("sa", 2, 109),
        ]
        forward = analyzer.summarize_algorithms(rows, 300, 0.95, 77)
        reverse = analyzer.summarize_algorithms(list(reversed(rows)), 300, 0.95, 77)
        self.assertEqual(forward, reverse)

    def test_mixed_timing_sources_are_rejected(self) -> None:
        modern = self.make_row("sa", 1, 110, total_elapsed_ms="10.0")
        legacy = self.make_row("sa", 2, 109, total_elapsed_ms="")
        # Timing schema is deliberately not a condition field: mixing is
        # detected explicitly instead of silently creating two groups.
        self.assertEqual(modern["_condition_id"], legacy["_condition_id"])
        with self.assertRaisesRegex(ValueError, "mixed timing sources"):
            analyzer.summarize_algorithms([modern, legacy], 100, 0.95, 77)

    def test_pairing_warnings_detect_algorithm_seed_loss(self) -> None:
        rows = [
            self.make_row("sa", 1, 110),
            self.make_row("sa", 2, 109),
            self.make_row("current", 1, 108),
        ]
        warnings = analyzer.pairing_warnings(rows, ["sa", "current"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("paired seed mismatch", warnings[0])


class EndToEndFixtureTests(unittest.TestCase):
    @staticmethod
    def write_fixture(path: Path) -> None:
        values = {
            "sa": [110, 109, 108, 107, 106, 105],
            "current": [108, 107, 106, 105, 104, 103],
            "paper": [107, 106, 105, 104, 103, 102],
            "paper-sb": [100, 100, 101, 100, 101, 100],
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.RAW_HEADER)
            writer.writeheader()
            for algorithm, best_lengths in values.items():
                for index, best_length in enumerate(best_lengths):
                    row = {field: "" for field in runner.RAW_HEADER}
                    row.update(
                        {
                            "job_id": f"{algorithm}-{index}",
                            "status": "ok",
                            "budget_scheme": "equal-iterations",
                            "algorithm_key": algorithm,
                            "instance": "fixture100",
                            "bks": "100",
                            "seed": str(1001 + index),
                            "requested_backend": "omp",
                            "actual_backend": "omp",
                            "best_length": str(best_length),
                            "elapsed_ms": str(900 + index),
                            "total_elapsed_ms": str(10 + index + len(algorithm) / 10.0),
                        }
                    )
                    writer.writerow(row)

    def test_stdlib_only_analyzer_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            raw_path = temp / "raw.csv"
            output_dir = temp / "analysis"
            self.write_fixture(raw_path)
            command = [
                sys.executable,
                "-S",
                str(ROOT / "scripts/analyze_paired_experiments.py"),
                "--input",
                str(raw_path),
                "--output-dir",
                str(output_dir),
                "--bootstrap-samples",
                "300",
                "--strict-pairing",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            with (output_dir / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
                summary = list(csv.DictReader(handle))
            with (output_dir / "pairwise.csv").open("r", encoding="utf-8", newline="") as handle:
                pairwise = list(csv.DictReader(handle))
            with (output_dir / "friedman.csv").open("r", encoding="utf-8", newline="") as handle:
                friedman = list(csv.DictReader(handle))

            self.assertEqual(len(summary), 4)
            self.assertEqual(len(pairwise), 6)
            self.assertEqual(len(friedman), 1)
            self.assertTrue(all(row["timing_source"] == "total_elapsed_ms" for row in summary))
            paper_sb = next(row for row in summary if row["algorithm_key"] == "paper-sb")
            self.assertEqual(paper_sb["bks_hit_count"], "4")
            self.assertAlmostEqual(float(paper_sb["bks_hit_rate"]), 4 / 6)
            self.assertEqual(friedman[0]["complete_seed_blocks"], "6")
            self.assertEqual(friedman[0]["status"], "ok")
            self.assertTrue(all(0.0 <= float(row["wilcoxon_p_holm"]) <= 1.0 for row in pairwise))
            self.assertTrue((output_dir / "analysis_manifest.json").is_file())
            self.assertTrue((output_dir / "checksums.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
