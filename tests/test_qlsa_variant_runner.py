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
        self.assertEqual(rows[0], dict(zip(runner.PROGRAM_HEADER, legacy_values)))


if __name__ == "__main__":
    unittest.main()
