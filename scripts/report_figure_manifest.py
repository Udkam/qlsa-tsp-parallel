#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create and verify the data provenance manifest for report figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "figures" / "report_figure_manifest.json"

# Each report-facing figure is tied to the CSV files that determine its plotted
# values.  The command is documentation for exact regeneration; validation is
# based on SHA-256 so a changed CSV or stale image fails closed.
FIGURE_SOURCES: dict[str, dict[str, object]] = {
    "figures/fig_course_01_openmp_speedup.png": {
        "sources": ["results/summary/step5_multi_cpu_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_02_openmp_efficiency.png": {
        "sources": ["results/summary/step5_multi_cpu_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_03_default_gap.png": {
        "sources": ["results/summary/step5_multi_cpu_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_04_targeted_quality.png": {
        "sources": ["results/summary/targeted_quality_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_05_policy_comparison.png": {
        "sources": ["results/summary/policy_comparison_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_06_cuda_boundary.png": {
        "sources": ["results/summary/large_cuda_formal_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_07_mpi_scaling.png": {
        "sources": ["results/summary/mpi_vm_scaling_formal_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_09_paper_quality.png": {
        "sources": [
            "results/reference/paper_hard_instance_quality.csv",
            "results/summary/targeted_quality_summary.csv",
        ],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_10_openmp_thread_scaling.png": {
        "sources": ["results/summary/openmp_scaling_final_summary.csv"],
        "command": "py scripts/make_course_report_figures.py",
    },
    "figures/fig_course_11_representative_openmp.png": {
        "sources": [
            "results/raw/final_all_data_openmp_raw.csv",
            "results/summary/final_representative_openmp_summary.csv",
        ],
        "command": (
            "py scripts/analyze_all_data_openmp.py --input "
            "results/raw/final_all_data_openmp_raw.csv --output "
            "results/summary/final_representative_openmp_summary.csv --figure "
            "figures/fig_course_11_representative_openmp.png"
        ),
    },
    "figures/fig_cuda_candidate_policy_formal.png": {
        "sources": [
            "results/raw/cuda_candidate_policy_formal_raw.csv",
            "results/summary/cuda_candidate_policy_formal_summary.csv",
        ],
        "command": (
            "py scripts/analyze_cuda_candidate.py --input "
            "results/raw/cuda_candidate_policy_formal_raw.csv --output "
            "results/summary/cuda_candidate_policy_formal_summary.csv --figure "
            "figures/fig_cuda_candidate_policy_formal.png"
        ),
    },
    "figures/fig_qlsa_variant_alignment.png": {
        "sources": [
            "results/raw/qlsa_variant_alignment_raw.csv",
            "results/summary/qlsa_variant_alignment_summary.csv",
        ],
        "command": (
            "py scripts/analyze_qlsa_variant_experiments.py --input "
            "results/raw/qlsa_variant_alignment_raw.csv --output "
            "results/summary/qlsa_variant_alignment_summary.csv --figure "
            "figures/fig_qlsa_variant_alignment.png"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for figure_rel, metadata in FIGURE_SOURCES.items():
        figure = ROOT / figure_rel
        if not figure.is_file():
            raise FileNotFoundError(f"missing report figure: {figure_rel}")
        source_entries: list[dict[str, str]] = []
        for source_rel in metadata["sources"]:
            source = ROOT / str(source_rel)
            if not source.is_file():
                raise FileNotFoundError(f"missing figure source: {source_rel}")
            source_entries.append({"path": str(source_rel), "sha256": sha256(source)})
        entries.append(
            {
                "figure": figure_rel,
                "figure_sha256": sha256(figure),
                "sources": source_entries,
                "command": metadata["command"],
            }
        )
    return {"schema_version": 1, "hash_algorithm": "sha256", "figures": entries}


def write_manifest(path: Path = DEFAULT_MANIFEST) -> None:
    payload = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing report figure manifest: {path.relative_to(ROOT)}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid report figure manifest: {exc}"]
    if payload.get("schema_version") != 1:
        errors.append("report figure manifest has unsupported schema_version")
    entries = payload.get("figures")
    if not isinstance(entries, list):
        return errors + ["report figure manifest must contain a figures list"]

    expected = set(FIGURE_SOURCES)
    actual: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("report figure manifest contains a non-object entry")
            continue
        figure_rel = entry.get("figure")
        if not isinstance(figure_rel, str):
            errors.append("report figure manifest entry is missing figure path")
            continue
        if figure_rel in actual:
            errors.append(f"duplicate report figure manifest entry: {figure_rel}")
            continue
        actual.add(figure_rel)
        figure = ROOT / figure_rel
        if not figure.is_file():
            errors.append(f"missing report figure: {figure_rel}")
        elif sha256(figure) != entry.get("figure_sha256"):
            errors.append(f"stale report figure hash: {figure_rel}")

        configured = FIGURE_SOURCES.get(figure_rel)
        if configured is None:
            errors.append(f"unexpected report figure manifest entry: {figure_rel}")
            continue
        source_entries = entry.get("sources")
        if not isinstance(source_entries, list):
            errors.append(f"missing source list for report figure: {figure_rel}")
            continue
        configured_sources = {str(value) for value in configured["sources"]}
        manifest_sources: set[str] = set()
        for source_entry in source_entries:
            if not isinstance(source_entry, dict) or not isinstance(source_entry.get("path"), str):
                errors.append(f"invalid source entry for report figure: {figure_rel}")
                continue
            source_rel = source_entry["path"]
            manifest_sources.add(source_rel)
            source = ROOT / source_rel
            if not source.is_file():
                errors.append(f"missing figure source: {source_rel}")
            elif sha256(source) != source_entry.get("sha256"):
                errors.append(f"stale figure source hash: {source_rel} -> {figure_rel}")
        if manifest_sources != configured_sources:
            errors.append(f"source mapping mismatch for report figure: {figure_rel}")

    missing_entries = expected - actual
    for figure_rel in sorted(missing_entries):
        errors.append(f"missing report figure manifest entry: {figure_rel}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write hashes for the current figures and sources")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.write:
        write_manifest(manifest)
        print(f"[ok] wrote report figure manifest: {manifest.relative_to(ROOT)}")
        return 0
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1
    print(f"[ok] report figure manifest is current: {manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
