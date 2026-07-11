#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the final course report package kept under docs/final."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_REPORT = ROOT / "docs" / "final" / "report.md"
PERSONAL_REPORT = ROOT / "docs" / "final" / "personal_report.md"
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)

REQUIRED_RESULTS = [
    ROOT / "results" / "summary" / "step5_multi_cpu_summary.csv",
    ROOT / "results" / "summary" / "targeted_quality_summary.csv",
    ROOT / "results" / "summary" / "cuda_qlsa_candidate_summary.csv",
    ROOT / "results" / "summary" / "mpi_vm_scaling_formal_summary.csv",
    ROOT / "results" / "summary" / "fair_paired_eil76_summary.csv",
    ROOT / "results" / "summary" / "fair_paired_eil76_pairwise.csv",
    ROOT / "results" / "summary" / "fair_paired_eil76_friedman.csv",
    ROOT / "results" / "summary" / "island_eil76_summary.csv",
    ROOT / "results" / "summary" / "island_eil76_paired_comparisons.csv",
    ROOT / "results" / "final" / "fair_island_clean_run_provenance.csv",
    ROOT / "results" / "reference" / "paper_hard_instance_quality.csv",
    ROOT / "results" / "final" / "RESULTS_INDEX.md",
]

FORBIDDEN = [
    "TODO",
    "<你的姓名>",
    "<你的学号>",
    "CUDA 全面优于 OpenMP",
    "QLSA 总是优于 SA",
    "完整复刻 SB-QLSA",
    "百万城市级",
    "figures" + "/final",
    "figures" + "\\final",
    "\u951f\u65a4\u62f7",
    "\ufffd",
    "\u951b",
    "\u6b5a",
]


def image_refs(text: str) -> list[str]:
    refs = [raw for _, raw in MD_IMAGE_RE.findall(text)]
    refs.extend(HTML_IMAGE_RE.findall(text))
    return refs


def check_images(report: Path) -> list[str]:
    errors: list[str] = []
    text = report.read_text(encoding="utf-8")
    for raw in image_refs(text):
        candidate = (report.parent / raw.split("#", 1)[0].strip()).resolve()
        if not candidate.exists():
            errors.append(f"{report.relative_to(ROOT)}: missing image {raw}")
    return errors


def check_text(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for phrase in FORBIDDEN:
        if phrase in text:
            errors.append(f"{path.relative_to(ROOT)}: forbidden phrase: {phrase}")
    if "?" * 4 in text or "\u95bf" in text or "\u95c1" in text or "\u95c2" in text:
        errors.append(f"{path.relative_to(ROOT)}: possible mojibake")
    return errors


def main() -> int:
    errors: list[str] = []

    for report in [COURSE_REPORT, PERSONAL_REPORT]:
        if not report.exists():
            errors.append(f"missing document: {report.relative_to(ROOT)}")
            continue
        errors.extend(check_text(report))

    if COURSE_REPORT.exists():
        errors.extend(check_images(COURSE_REPORT))

    for path in REQUIRED_RESULTS:
        if not path.exists():
            errors.append(f"missing result file: {path.relative_to(ROOT)}")

    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    print("[ok] final course documents check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
