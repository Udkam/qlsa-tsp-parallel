#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for QLSA variant metric-aware analysis."""

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


analyzer = load_script("qlsa_variant_analyzer", "scripts/analyze_qlsa_variant_experiments.py")


def make_row(metric: str | None, best_length: str) -> dict[str, str]:
    row = {
        "instance": "eil76",
        "qlsa_variant": "paper-sb",
        "policy": "epsilon-greedy",
        "diversity_threshold": "0.50",
        "best_length": best_length,
        "elapsed_ms": "10.0",
        "accepted_moves": "20",
        "improved_moves": "5",
        "iterations": "100",
        "chains": "4",
        "threads": "2",
    }
    if metric is not None:
        row["diversity_metric"] = metric
    return row


class QLSAVariantAnalysisTests(unittest.TestCase):
    def test_legacy_hamming_and_edge_are_separate_summary_conditions(self) -> None:
        summary = analyzer.group_rows(
            [
                make_row(None, "540"),
                make_row("edge", "541"),
            ]
        )

        self.assertEqual(len(summary), 2)
        by_metric = {row["diversity_metric"]: row for row in summary}
        self.assertEqual(set(by_metric), {"hamming", "edge"})
        self.assertEqual(by_metric["hamming"]["best_length_min"], "540")
        self.assertEqual(by_metric["edge"]["best_length_min"], "541")


if __name__ == "__main__":
    unittest.main()
