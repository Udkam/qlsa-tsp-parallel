#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free hardening tests for the island-ablation analyzer."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_analyzer():
    path = ROOT / "scripts" / "analyze_island_ablation.py"
    spec = importlib.util.spec_from_file_location("island_analyzer_hardening_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()

CONFIG_HASH = "1" * 64
INPUT_HASH = "2" * 64
EXECUTABLE_HASH = "3" * 64
ENVIRONMENT_HASH = "4" * 64


def make_row(
    *,
    seed: int = 1,
    topology: str = "independent",
    interval: int = 1000,
    algorithm: str = "sa",
    best: int = 110,
    config_hash: str = CONFIG_HASH,
    input_hash: str = INPUT_HASH,
    executable_hash: str = EXECUTABLE_HASH,
    environment_hash: str = ENVIRONMENT_HASH,
) -> dict[str, str]:
    attempts = 0 if topology == "independent" else 16
    adopted = 0 if topology == "independent" else 4
    return {
        "job_id": f"{algorithm}-{topology}-{interval}-{seed}",
        "status": "ok",
        "error": "",
        "experiment_name": "hardening-fixture",
        "algorithm_key": algorithm,
        "algorithm_display": algorithm.upper(),
        "qlsa_variant": "",
        "instance": "fixture100",
        "dimension": "100",
        "bks": "100",
        "seed": str(seed),
        "paired_seed_index": str(seed - 1),
        "execution_order": str(seed),
        "topology": topology,
        "migration_interval": str(interval),
        "command_migration_interval": str(interval),
        "requested_backend": "omp",
        "reported_requested_backend": "openmp",
        "actual_backend": "openmp",
        "require_backend_match": "true",
        "backend_fallback": "false",
        "fallback_reason": "",
        "iterations_requested": "100",
        "iterations_reported": "100",
        "iterations_completed": "800",
        "deadline_reached": "false",
        "init": "nn",
        "chains": "8",
        "threads": "8",
        "actual_threads": "8",
        "reported_parallel": "omp",
        "program_algorithm": f"sa-island-{topology}",
        "best_length": str(best),
        "final_length": str(best + 5),
        "elapsed_ms": "12.5",
        "total_elapsed_ms": "12.25",
        "kernel_elapsed_ms": "0",
        "wall_elapsed_ms": "13",
        "accepted_moves": "20",
        "improved_moves": "10",
        "migration_rounds": "4",
        "migration_attempts": str(attempts),
        "migrations_adopted": str(adopted),
        "migration_adoption_rate": str(adopted / attempts if attempts else 0.0),
        "command_json": "[]",
        "log_file": "fixture.log",
        "return_code": "0",
        "started_at": "2026-07-11T00:00:00+08:00",
        "finished_at": "2026-07-11T00:00:01+08:00",
        "git_commit": "deadbeef",
        "git_dirty": "false",
        "config_sha256": config_hash,
        "input_sha256": input_hash,
        "executable_sha256": executable_hash,
        "environment_sha256": environment_hash,
    }


def complete_rows(
    *, interval: int = 1000, seeds: tuple[int, ...] = (1, 2, 3), **kwargs: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in seeds:
        rows.append(make_row(seed=seed, topology="independent", interval=interval, best=110, **kwargs))
        rows.append(make_row(seed=seed, topology="ring", interval=interval, best=109, **kwargs))
        rows.append(make_row(seed=seed, topology="global", interval=interval, best=111, **kwargs))
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_run_directory(path: Path, rows: list[dict[str, str]], status: str = "complete") -> None:
    path.mkdir()
    config_path = path / "config.snapshot.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config_hash = analyzer.sha256_file(config_path)
    log_path = path / "logs" / "fixture.log"
    log_path.parent.mkdir()
    log_path.write_text("fixture output\n", encoding="utf-8")
    log_hash = analyzer.sha256_file(log_path)
    rows_with_provenance = [dict(row, config_sha256=config_hash) for row in rows]
    raw_path = path / "raw.csv"
    write_rows(raw_path, rows_with_provenance)
    manifest = {
        "schema_version": 1,
        "status": status,
        "config_sha256": config_hash,
        "environment_sha256": ENVIRONMENT_HASH,
        "executable": {"sha256": EXECUTABLE_HASH},
        "inputs": {"fixture100": {"sha256": INPUT_HASH}},
        "jobs": [
            {
                "log_file": "logs/fixture.log",
                "log_sha256": log_hash,
            }
        ],
        "artifacts": {
            "config_snapshot": {
                "path": "config.snapshot.json",
                "sha256": config_hash,
            },
            "logs": [
                {
                    "path": "logs/fixture.log",
                    "sha256": log_hash,
                }
            ],
            "raw_csv": {"path": "raw.csv", "sha256": analyzer.sha256_file(raw_path)},
            "checksums": {
                "path": "checksums.sha256",
                "covers": [
                    "config.snapshot.json",
                    "logs/fixture.log",
                    "raw.csv",
                    "manifest.json",
                ],
            },
        },
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "checksums.sha256").write_text(
        f"{config_hash}  config.snapshot.json\n"
        f"{log_hash}  logs/fixture.log\n"
        f"{analyzer.sha256_file(raw_path)}  raw.csv\n"
        f"{analyzer.sha256_file(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )


class StrictSuccessfulRowTests(unittest.TestCase):
    def assert_rejected(self, mutation: dict[str, str], pattern: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            row = make_row()
            row.update(mutation)
            write_rows(path, [row])
            with self.assertRaisesRegex(ValueError, pattern):
                analyzer.read_raw_rows(path)

    def test_revalidates_backend_threads_deadline_and_complete_iterations(self) -> None:
        cases = [
            ({"actual_backend": "cpu_serial"}, "backend mismatch"),
            ({"backend_fallback": "true"}, "backend fallback"),
            ({"actual_threads": "7"}, "thread mismatch"),
            ({"deadline_reached": "true"}, "deadline_reached=true"),
            ({"deadline_reached": ""}, "deadline_reached is required"),
            ({"iterations_completed": "799"}, "incomplete iterations"),
            ({"iterations_requested": ""}, "iterations_requested is required"),
            ({"chains": ""}, "chains is required"),
        ]
        for mutation, pattern in cases:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation, pattern)

    def test_strict_numeric_and_boolean_parsing(self) -> None:
        cases = [
            ({"require_backend_match": "yes"}, "exactly true or false"),
            ({"backend_fallback": "0"}, "exactly true or false"),
            ({"threads": "8.0"}, "invalid integer"),
            ({"seed": "01"}, "invalid integer"),
            ({"total_elapsed_ms": "nan"}, "non-finite"),
        ]
        for mutation, pattern in cases:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation, pattern)

    def test_non_ok_rows_are_excluded_without_being_treated_as_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            failed = make_row()
            failed.update({"status": "error", "actual_backend": "serial", "best_length": "nan"})
            write_rows(path, [failed, make_row(seed=2)])
            all_rows, successful = analyzer.read_raw_rows(path)
            self.assertEqual(len(all_rows), 2)
            self.assertEqual([row["seed"] for row in successful], ["2"])


class ConditionAndPairingTests(unittest.TestCase):
    def test_condition_id_uses_every_required_comparability_field(self) -> None:
        base = make_row()
        identifier = analyzer.condition_id_for_row(base)
        mutations = {
            "config_sha256": "5" * 64,
            "input_sha256": "6" * 64,
            "executable_sha256": "7" * 64,
            "environment_sha256": "8" * 64,
            "iterations_requested": "101",
            "chains": "9",
            "threads": "9",
            "actual_threads": "9",
            "init": "random",
            "requested_backend": "serial",
            "actual_backend": "serial",
            "instance": "fixture101",
            "bks": "101",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dict(base)
                changed[field] = value
                self.assertNotEqual(identifier, analyzer.condition_id_for_row(changed))

    def test_default_pairing_rejects_missing_blocks_and_exploratory_uses_one_common_block(self) -> None:
        rows = [
            *(make_row(seed=seed, topology="independent") for seed in (1, 2, 3)),
            *(make_row(seed=seed, topology="ring") for seed in (1, 2)),
            *(make_row(seed=seed, topology="global") for seed in (1, 3)),
        ]
        with self.assertRaisesRegex(ValueError, "paired seed mismatch"):
            analyzer.paired_analysis(rows)
        seed_rows, summaries = analyzer.paired_analysis(rows, allow_incomplete_pairs=True)
        self.assertEqual({row["seed"] for row in seed_rows}, {1})
        self.assertEqual(len(seed_rows), 2)
        self.assertTrue(all(row["n_pairs"] == 1 for row in summaries))

    def test_interval_matched_baselines_and_holm_adjustment(self) -> None:
        rows = complete_rows(interval=1000) + complete_rows(interval=2000)
        seed_rows, summaries = analyzer.paired_analysis(rows)
        self.assertEqual(len(seed_rows), 12)
        self.assertEqual(len(summaries), 4)
        self.assertEqual({row["migration_interval"] for row in summaries}, {1000, 2000})
        self.assertTrue(all("exact_sign_test_p_holm" in row for row in summaries))
        self.assertTrue(all("sign_reject_holm_0_05" in row for row in summaries))
        self.assertTrue(all(float(row["exact_sign_test_p_holm"]) == 1.0 for row in summaries))

    def test_exact_sign_test_known_values(self) -> None:
        self.assertEqual(analyzer.exact_sign_test_two_sided(0, 0), 1.0)
        self.assertEqual(analyzer.exact_sign_test_two_sided(2, 2), 1.0)
        self.assertAlmostEqual(analyzer.exact_sign_test_two_sided(3, 0), 0.25)
        self.assertAlmostEqual(analyzer.exact_sign_test_two_sided(5, 0), 0.0625)


class ProvenanceAndOrderingTests(unittest.TestCase):
    def test_main_outputs_condition_ids_and_integrity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.csv"
            output_dir = root / "analysis"
            write_rows(raw_path, complete_rows())
            self.assertEqual(
                analyzer.main([str(raw_path), "--output-dir", str(output_dir)]), 0
            )
            with (output_dir / "summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summaries = list(csv.DictReader(handle))
            self.assertTrue(summaries)
            self.assertTrue(all(row["condition_id"] for row in summaries))
            metadata = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["inputs"][0]["provenance_validation"], "unavailable")
            self.assertEqual(len(metadata["conditions"]), 1)

    def test_multi_input_missing_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.csv"
            second = Path(temporary) / "second.csv"
            write_rows(first, complete_rows(interval=1000))
            missing = complete_rows(interval=2000)
            for row in missing:
                row["environment_sha256"] = ""
            write_rows(second, missing)
            with self.assertRaisesRegex(ValueError, "missing required provenance"):
                analyzer.read_inputs([first, second])

    def test_different_conditions_never_mix_and_input_order_is_irrelevant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "z.csv"
            second = Path(temporary) / "a.csv"
            write_rows(first, complete_rows(interval=1000))
            write_rows(
                second,
                complete_rows(interval=1000, executable_hash="9" * 64),
            )
            _, forward, _ = analyzer.read_inputs([first, second])
            _, reverse, _ = analyzer.read_inputs([second, first])
            self.assertEqual(analyzer.group_summary(forward), analyzer.group_summary(reverse))
            self.assertEqual(analyzer.paired_analysis(forward), analyzer.paired_analysis(reverse))
            condition_ids = {row["_condition_id"] for row in forward}
            self.assertEqual(len(condition_ids), 2)
            self.assertEqual(
                {row["condition_id"] for row in analyzer.group_summary(forward)}, condition_ids
            )

    def test_run_directory_validates_manifest_and_sidecar_but_direct_csv_is_marked_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            write_run_directory(run_dir, complete_rows())
            _, rows, metadata = analyzer.read_inputs([run_dir])
            self.assertEqual(len(rows), 9)
            self.assertEqual(metadata[0]["provenance_validation"], "validated")

            _, direct_rows, direct_metadata = analyzer.read_inputs([run_dir / "raw.csv"])
            self.assertEqual(len(direct_rows), 9)
            self.assertEqual(direct_metadata[0]["provenance_validation"], "unavailable")

            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "failed"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "status must be complete"):
                analyzer.read_inputs([run_dir])

    def test_run_directory_rejects_checksum_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            write_run_directory(run_dir, complete_rows())
            sidecar = run_dir / "checksums.sha256"
            sidecar.write_text(
                f"{'0' * 64}  raw.csv\n"
                f"{analyzer.sha256_file(run_dir / 'manifest.json')}  manifest.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mismatch for raw.csv"):
                analyzer.read_inputs([run_dir])

    def test_run_directory_rejects_tampered_config_and_missing_log(self) -> None:
        cases = (
            ("config.snapshot.json", "tamper", "mismatch for config.snapshot.json"),
            ("logs/fixture.log", "delete", "checksummed run artifact is missing"),
        )
        for relative, mutation, pattern in cases:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                write_run_directory(run_dir, complete_rows())
                target = run_dir / relative
                if mutation == "tamper":
                    target.write_text('{"tampered": true}\n', encoding="utf-8")
                else:
                    target.unlink()
                with self.assertRaisesRegex((ValueError, FileNotFoundError), pattern):
                    analyzer.read_inputs([run_dir])

    def test_log_cannot_be_removed_from_sidecar_and_coverage_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            write_run_directory(run_dir, complete_rows())
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["checksums"]["covers"] = [
                "config.snapshot.json",
                "raw.csv",
                "manifest.json",
            ]
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (run_dir / "checksums.sha256").write_text(
                f"{analyzer.sha256_file(run_dir / 'config.snapshot.json')}  config.snapshot.json\n"
                f"{analyzer.sha256_file(run_dir / 'raw.csv')}  raw.csv\n"
                f"{analyzer.sha256_file(manifest_path)}  manifest.json\n",
                encoding="utf-8",
            )
            (run_dir / "logs" / "fixture.log").unlink()
            with self.assertRaisesRegex(ValueError, "required config/raw/manifest/log set"):
                analyzer.read_inputs([run_dir])


if __name__ == "__main__":
    unittest.main()
