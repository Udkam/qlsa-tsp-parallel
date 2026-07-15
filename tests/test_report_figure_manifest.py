#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for report figure/source provenance validation."""

from __future__ import annotations

import importlib.util
import json
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


manifest_module = load_script("report_figure_manifest", "scripts/report_figure_manifest.py")


class ReportFigureManifestTests(unittest.TestCase):
    def test_manifest_detects_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "figures").mkdir()
            (root / "results").mkdir()
            figure = root / "figures" / "figure.png"
            source = root / "results" / "source.csv"
            figure.write_bytes(b"png")
            source.write_text("value\n1\n", encoding="utf-8")
            mapping = {
                "figures/figure.png": {
                    "sources": ["results/source.csv"],
                    "command": "generate",
                }
            }
            manifest = root / "figures" / "manifest.json"
            with mock.patch.object(manifest_module, "ROOT", root), mock.patch.object(
                manifest_module, "FIGURE_SOURCES", mapping
            ):
                manifest.write_text(
                    json.dumps(manifest_module.build_manifest()), encoding="utf-8"
                )
                self.assertEqual(manifest_module.validate_manifest(manifest), [])
                source.write_text("value\n2\n", encoding="utf-8")
                errors = manifest_module.validate_manifest(manifest)
            self.assertTrue(any("stale figure source hash" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
