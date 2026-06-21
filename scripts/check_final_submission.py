#!/usr/bin/env python3
"""Check the single course submission package."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_REPORT = ROOT / "docs" / "final" / "final_report_course.md"
COURSE_PACKAGE = ROOT / "submission" / "course"
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

KEY_CSV = [
    "final_key_results.csv",
    "step5_multi_cpu_summary.csv",
    "tuned_validation_summary.csv",
    "targeted_quality_summary.csv",
    "paper_table8_runtime.csv",
    "paper_hard_instance_quality.csv",
    "report_comparison_summary.csv",
    "policy_comparison_summary.csv",
    "openmp_scaling_final_summary.csv",
    "cuda_qlsa_candidate_summary.csv",
    "cuda_reversal_summary.csv",
    "large_instance_inventory.csv",
    "large_instance_download_status.csv",
    "large_openmp_l1_summary.csv",
    "large_cuda_formal_summary.csv",
    "large_mpi_vm_formal_aggressive_summary.csv",
]


def check_images(report: Path) -> list[str]:
    errors: list[str] = []
    text = report.read_text(encoding="utf-8")
    for alt, raw in IMAGE_RE.findall(text):
        if re.search(r"图\s*[0-9一二三四五六七八九十]+\s*[:：]", alt):
            errors.append(f"{report.relative_to(ROOT)}: numbered figure alt text: {alt}")
        candidate = (report.parent / raw.split("#", 1)[0].strip()).resolve()
        if not candidate.exists():
            errors.append(f"{report.relative_to(ROOT)}: missing image {raw}")
    return errors


def main() -> int:
    errors: list[str] = []

    if not COURSE_REPORT.exists():
        errors.append(f"missing report: {COURSE_REPORT.relative_to(ROOT)}")
    else:
        errors.extend(check_images(COURSE_REPORT))

    if not COURSE_PACKAGE.exists():
        errors.append(f"missing package: {COURSE_PACKAGE.relative_to(ROOT)}")
    else:
        for rel in ["final_report.md", "reproduction_commands.md", "figures", "results_key"]:
            if not (COURSE_PACKAGE / rel).exists():
                errors.append(f"missing package item: {COURSE_PACKAGE.relative_to(ROOT)}/{rel}")
        for csv_name in KEY_CSV:
            if not (COURSE_PACKAGE / "results_key" / csv_name).exists():
                errors.append(f"missing package key CSV: {COURSE_PACKAGE.relative_to(ROOT)}/results_key/{csv_name}")
        package_report = COURSE_PACKAGE / "final_report.md"
        if package_report.exists():
            package_text = package_report.read_text(encoding="utf-8")
            for _, raw in IMAGE_RE.findall(package_text):
                candidate = (package_report.parent / raw.split("#", 1)[0].strip()).resolve()
                if not candidate.exists():
                    errors.append(f"{package_report.relative_to(ROOT)}: missing package image {raw}")

    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    print("[ok] course submission package check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
