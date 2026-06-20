#!/usr/bin/env python3
"""Final submission integrity checks for the course/public package layout."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_REPORT = ROOT / "docs" / "final" / "final_report_course.md"
PUBLIC_REPORT = ROOT / "docs" / "final" / "final_report_public.md"
COURSE_PACKAGE = ROOT / "submission" / "course"
PUBLIC_PACKAGE = ROOT / "submission" / "public"
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

FORBIDDEN_PUBLIC = [
    "陈乐浚",
    "22361054",
    "CUDA 比 OpenMP 快",
    "QLSA 总是优于 SA",
    "完整复刻 SB-QLSA",
    "同平台公平 benchmark",
    "????",
    "�",
    "锟斤拷",
]

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
]


def check_images(report: Path) -> list[str]:
    errors: list[str] = []
    text = report.read_text(encoding="utf-8")
    for alt, raw in IMAGE_RE.findall(text):
        if re.search(r"图\s*[0-9xX]+\s*[:：]", alt):
            errors.append(f"{report.relative_to(ROOT)}: numbered figure alt text: {alt}")
        candidate = (report.parent / raw.split("#", 1)[0].strip()).resolve()
        if not candidate.exists():
            errors.append(f"{report.relative_to(ROOT)}: missing image {raw}")
    return errors


def main() -> int:
    errors: list[str] = []

    for report in [COURSE_REPORT, PUBLIC_REPORT]:
        if not report.exists():
            errors.append(f"missing report: {report.relative_to(ROOT)}")
        else:
            errors.extend(check_images(report))

    if PUBLIC_REPORT.exists():
        public_text = PUBLIC_REPORT.read_text(encoding="utf-8")
        for phrase in FORBIDDEN_PUBLIC:
            if phrase in public_text:
                errors.append(f"public report contains forbidden phrase: {phrase}")

    for package in [COURSE_PACKAGE, PUBLIC_PACKAGE]:
        if not package.exists():
            errors.append(f"missing package: {package.relative_to(ROOT)}")
            continue
        if not (package / "figures").exists():
            errors.append(f"missing package figures directory: {package.relative_to(ROOT)}")
        if not (package / "results_key").exists():
            errors.append(f"missing package results_key directory: {package.relative_to(ROOT)}")
        for csv_name in KEY_CSV:
            if not (package / "results_key" / csv_name).exists():
                errors.append(f"missing package key CSV: {package.relative_to(ROOT)}/results_key/{csv_name}")

    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    print("[ok] final submission package check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
