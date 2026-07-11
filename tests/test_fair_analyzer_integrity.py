#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity checks for run-directory inputs to the paired analyzer."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_analyzer():
    name = "fair_analyzer_integrity_target"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "scripts/analyze_paired_experiments.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import paired experiment analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()


FIELDNAMES = [
    "status",
    "job_id",
    "experiment_name",
    "budget_scheme",
    "budget_target",
    "budget_unit",
    "algorithm_key",
    "instance",
    "bks",
    "seed",
    "requested_backend",
    "actual_backend",
    "require_backend_match",
    "time_limit_ms",
    "init",
    "chains",
    "threads",
    "actual_threads",
    "reported_parallel",
    "best_length",
    "elapsed_ms",
    "total_elapsed_ms",
    "config_sha256",
    "input_sha256",
    "executable_sha256",
    "environment_sha256",
    "log_file",
]


class AnalyzerRunDirectoryIntegrityTests(unittest.TestCase):
    def write_raw(self, path: Path, config_sha256: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for index, algorithm in enumerate(analyzer.ALGORITHM_ORDER):
                writer.writerow(
                    {
                        "status": "ok",
                        "job_id": f"{algorithm}-job",
                        "experiment_name": "integrity-fixture",
                        "budget_scheme": "equal-iterations",
                        "budget_target": "100",
                        "budget_unit": "iterations_per_chain",
                        "algorithm_key": algorithm,
                        "instance": "fixture100",
                        "bks": "100",
                        "seed": "2601",
                        "requested_backend": "omp",
                        "actual_backend": "omp",
                        "require_backend_match": "true",
                        "time_limit_ms": "",
                        "init": "nn",
                        "chains": "8",
                        "threads": "8",
                        "actual_threads": "8",
                        "reported_parallel": "omp",
                        "best_length": str(110 - index),
                        "elapsed_ms": str(10 + index),
                        "total_elapsed_ms": str(10 + index),
                        "config_sha256": config_sha256,
                        "input_sha256": "input-fixture",
                        "executable_sha256": "executable-fixture",
                        "environment_sha256": "environment-fixture",
                        "log_file": f"logs/{algorithm}-job.log",
                    }
                )

    def make_bundle(self, parent: Path, *, status: str = "complete") -> Path:
        run_dir = parent / "run"
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True)

        snapshot_path = run_dir / "config.snapshot.json"
        snapshot_path.write_text('{"fixture":true}\n', encoding="utf-8")
        config_hash = analyzer.sha256_file(snapshot_path)
        raw_path = run_dir / "raw.csv"
        self.write_raw(raw_path, config_hash)
        log_paths: list[Path] = []
        jobs: list[dict[str, str]] = []
        log_artifacts: list[dict[str, str]] = []
        for algorithm in analyzer.ALGORITHM_ORDER:
            job_id = f"{algorithm}-job"
            log_path = logs_dir / f"{job_id}.log"
            log_path.write_text(f"fixture log for {algorithm}\n", encoding="utf-8")
            relative_log = log_path.relative_to(run_dir).as_posix()
            log_hash = analyzer.sha256_file(log_path)
            log_paths.append(log_path)
            jobs.append(
                {
                    "job_id": job_id,
                    "log_file": relative_log,
                    "log_sha256": log_hash,
                }
            )
            log_artifacts.append({"path": relative_log, "sha256": log_hash})

        checksum_scope = [
            snapshot_path.name,
            raw_path.name,
            "manifest.json",
            *(item["path"] for item in log_artifacts),
        ]

        manifest = {
            "schema_version": 1,
            "run_id": "integrity-fixture",
            "status": status,
            "config_snapshot": snapshot_path.name,
            "config_sha256": config_hash,
            "jobs": jobs,
            "artifacts": {
                "raw_csv": {
                    "path": raw_path.name,
                    "sha256": analyzer.sha256_file(raw_path),
                },
                "config_snapshot": {
                    "path": snapshot_path.name,
                    "sha256": config_hash,
                },
                "logs": log_artifacts,
                "checksums": {
                    "path": "checksums.sha256",
                    "covers": checksum_scope,
                },
            },
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_paths = [snapshot_path, raw_path, manifest_path, *log_paths]
        (run_dir / "checksums.sha256").write_text(
            "".join(
                f"{analyzer.sha256_file(path)}  {path.relative_to(run_dir).as_posix()}\n"
                for path in checksum_paths
            ),
            encoding="utf-8",
        )
        return run_dir

    def rewrite_manifest_and_sidecar(
        self,
        run_dir: Path,
        manifest: dict[str, object],
        *,
        omit: set[str] | None = None,
    ) -> None:
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries: dict[str, str] = {}
        for line in (run_dir / "checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, relative = line.split("  ", 1)
            entries[relative] = digest
        entries["manifest.json"] = analyzer.sha256_file(manifest_path)
        for relative in omit or set():
            entries.pop(relative, None)
        (run_dir / "checksums.sha256").write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in entries.items()),
            encoding="utf-8",
        )

    def run_analysis(self, input_path: Path, output_dir: Path) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = analyzer.main(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--bootstrap-samples",
                    "20",
                ]
            )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_valid_bundle_is_verified_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            run_dir = self.make_bundle(temp)
            output_dir = temp / "analysis"
            result, _, stderr = self.run_analysis(run_dir, output_dir)
            self.assertEqual(result, 0, stderr)

            analysis_manifest = json.loads(
                (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            integrity = analysis_manifest["input_integrity"]
            self.assertEqual(len(integrity), 1)
            self.assertEqual(integrity[0]["input_kind"], "run_directory")
            self.assertEqual(integrity[0]["status"], "verified")
            self.assertEqual(integrity[0]["manifest"]["status"], "complete")
            self.assertEqual(integrity[0]["raw_csv"]["status"], "verified")
            self.assertEqual(integrity[0]["config_snapshot"]["status"], "verified")
            self.assertEqual(integrity[0]["logs"]["verified_file_count"], 4)
            self.assertEqual(integrity[0]["checksums"]["verified_file_count"], 7)

    def test_damaged_raw_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary))
            with (run_dir / "raw.csv").open("a", encoding="utf-8") as handle:
                handle.write("corrupted\n")
            with self.assertRaisesRegex(ValueError, "artifacts.raw_csv SHA-256 mismatch"):
                analyzer.verify_run_directory(run_dir)

    def test_failed_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary), status="failed")
            with self.assertRaisesRegex(ValueError, "status must be 'complete'"):
                analyzer.verify_run_directory(run_dir)

    def test_declared_log_file_missing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary))
            (run_dir / "logs/sa-job.log").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "log .* file not found"):
                analyzer.verify_run_directory(run_dir)

    def test_declared_log_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary))
            (run_dir / "logs/sa-job.log").write_text(
                "tampered log\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "log .* SHA-256 mismatch"):
                analyzer.verify_run_directory(run_dir)

    def test_log_removed_from_sidecar_and_covers_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary))
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            omitted = manifest["jobs"][0]["log_file"]
            manifest["artifacts"]["checksums"]["covers"].remove(omitted)
            self.rewrite_manifest_and_sidecar(run_dir, manifest, omit={omitted})
            with self.assertRaisesRegex(ValueError, "coverage is not exact"):
                analyzer.verify_run_directory(run_dir)

    def test_job_and_artifact_log_declarations_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary))
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["logs"][0]["sha256"] = "0" * 64
            self.rewrite_manifest_and_sidecar(run_dir, manifest)
            with self.assertRaisesRegex(
                ValueError, "jobs log declarations do not match artifacts.logs"
            ):
                analyzer.verify_run_directory(run_dir)

    def test_declared_config_snapshot_hash_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.make_bundle(Path(temporary))
            (run_dir / "config.snapshot.json").write_text(
                '{"fixture":false}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config snapshot SHA-256 mismatch"):
                analyzer.verify_run_directory(run_dir)

    def test_direct_csv_is_allowed_but_integrity_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            raw_path = temp / "direct.csv"
            self.write_raw(raw_path, "direct-config")
            output_dir = temp / "analysis"
            result, _, stderr = self.run_analysis(raw_path, output_dir)
            self.assertEqual(result, 0, stderr)

            analysis_manifest = json.loads(
                (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            integrity = analysis_manifest["input_integrity"][0]
            self.assertEqual(integrity["input_kind"], "direct_csv")
            self.assertEqual(integrity["status"], "unavailable")
            self.assertIn("no run manifest", integrity["reason"])

    def test_condition_id_includes_actual_threads_and_environment(self) -> None:
        base = {
            "experiment_name": "fixture",
            "budget_scheme": "equal-iterations",
            "budget_target": "100",
            "budget_unit": "iterations_per_chain",
            "instance": "fixture100",
            "bks": "100",
            "requested_backend": "omp",
            "actual_backend": "omp",
            "actual_threads": "8",
            "environment_sha256": "environment-a",
        }
        different_threads = dict(base, actual_threads="4")
        different_environment = dict(base, environment_sha256="environment-b")
        self.assertNotEqual(
            analyzer.condition_id_for_row(base),
            analyzer.condition_id_for_row(different_threads),
        )
        self.assertNotEqual(
            analyzer.condition_id_for_row(base),
            analyzer.condition_id_for_row(different_environment),
        )


if __name__ == "__main__":
    unittest.main()
