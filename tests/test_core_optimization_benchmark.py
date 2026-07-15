#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the paired core-optimization benchmark summary."""

from __future__ import annotations

import importlib.util
import sys
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


benchmark = load_script(
    "core_optimization_benchmark", "scripts/benchmark_core_optimizations.py"
)


def row(implementation: str, pair: int, elapsed_ms: float) -> dict[str, str]:
    return {
        "workload": "paired-test",
        "implementation": implementation,
        "pair": str(pair),
        "elapsed_ms": str(elapsed_ms),
        "instance": "eil76",
        "iterations": "100",
        "chains": "4",
        "threads": "2",
        "migration_topology": "independent",
        "migration_interval": "0",
        "revision": f"{implementation}-revision",
        "binary_sha256": f"{implementation}-binary",
        "build_contract_sha256": "same-contract",
        "input_sha256": "same-input",
        "benchmark_script_sha256": "same-script",
    }


class CoreOptimizationBenchmarkTests(unittest.TestCase):
    def test_speedup_is_summarized_from_paired_ratios(self) -> None:
        rows = [
            row("before", 1, 100.0),
            row("after", 1, 50.0),
            row("before", 2, 300.0),
            row("after", 2, 100.0),
            row("before", 3, 300.0),
            row("after", 3, 200.0),
        ]
        summary = benchmark.summarize_rows(rows)[0]
        # Per-pair speedups are 2.0, 3.0, and 1.5. Their median is 2.0,
        # while median(before) / median(after) would incorrectly be 3.0.
        self.assertEqual(summary["paired_speedup_median"], "2.0000")
        self.assertEqual(summary["paired_speedup_mean"], "2.1667")

    def test_single_pair_has_defined_quartiles(self) -> None:
        summary = benchmark.summarize_rows(
            [row("before", 1, 100.0), row("after", 1, 80.0)]
        )[0]
        self.assertEqual(summary["paired_speedup_q1"], "1.2500")
        self.assertEqual(summary["paired_speedup_q3"], "1.2500")


if __name__ == "__main__":
    unittest.main()
