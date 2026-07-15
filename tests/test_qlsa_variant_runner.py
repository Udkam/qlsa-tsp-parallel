#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for the standalone QLSA-variant experiment runner."""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_script("qlsa_variant_runner", "scripts/run_qlsa_variant_experiments.py")


class QLSAVariantRunnerTests(unittest.TestCase):
    def test_find_executable_discovers_ninja_preset_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "build" / "ninja-cuda-release" / "tsp_sa.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            with mock.patch.object(runner, "ROOT", root):
                self.assertEqual(runner.find_executable(), executable)

    def test_rows_from_stdout_accepts_current_extended_csv_schema(self) -> None:
        legacy_values = [
            "qlsa-paper-sb-omp",
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
        extended_values = legacy_values + [
            "2.5",
            "0.0",
            "openmp",
            "openmp",
            "false",
            "",
            "100",
            "false",
            "independent",
            "0",
            "0",
            "0",
            "0",
            "8",
        ]
        stream = io.StringIO()
        csv.writer(stream, lineterminator="").writerow(extended_values)

        rows = runner.rows_from_stdout("CSV:\n" + stream.getvalue())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["algorithm"], "qlsa-paper-sb-omp")
        self.assertEqual(rows[0]["actual_backend"], "openmp")
        self.assertEqual(rows[0]["actual_threads"], "8")
        self.assertEqual(rows[0]["migration_topology"], "independent")
        self.assertEqual(rows[0]["_csv_schema_width"], "28")

    def test_quick_mode_uses_a_separate_default_output(self) -> None:
        quick = SimpleNamespace(quick=True, output=str(runner.DEFAULT_OUTPUT))
        explicit = SimpleNamespace(quick=True, output="custom.csv")
        self.assertEqual(runner.output_path_for_args(quick), runner.QUICK_OUTPUT)
        self.assertEqual(runner.output_path_for_args(explicit), runner.ROOT / "custom.csv")

    def test_run_one_rejects_reported_backend_fallback(self) -> None:
        values = [
            "qlsa-paper-sb-omp", "square4", "4", "100", "17", "nn", "8", "8", "omp",
            "4", "4", "1.5", "20", "10", "2.5", "0.0", "openmp", "openmp", "true",
            "CUDA unavailable", "800", "false", "disabled", "0", "0", "0", "0", "8",
        ]
        stream = io.StringIO()
        csv.writer(stream, lineterminator="").writerow(values)
        args = SimpleNamespace(
            chains=8,
            threads=8,
            iterations=100,
            repeat=1,
            seed=17,
            diversity_metric="hamming",
        )
        completed = SimpleNamespace(returncode=0, stdout=stream.getvalue(), stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(runner.subprocess, "run", return_value=completed):
                with self.assertRaises(runner.ExperimentCsvError):
                    runner.run_one(
                        root / "tsp_sa.exe",
                        root / "square4.tsp",
                        "paper-sb",
                        "epsilon-greedy",
                        0.5,
                        args,
                        root,
                    )


if __name__ == "__main__":
    unittest.main()
