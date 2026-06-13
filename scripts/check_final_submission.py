#!/usr/bin/env python3
"""Final submission integrity checks."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "final_report_extreme.md"
SUBMISSION = ROOT / "submission"
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

FORBIDDEN = [
    "TODO",
    "请补充",
    "<你的姓名>",
    "<你的学号>",
    "CUDA 比 OpenMP 快",
    "CUDA 快于 OpenMP",
    "CUDA 优于 OpenMP",
    "QLSA 总是优于 SA",
    "完全复刻 SB-QLSA",
    "完全复刻论文 SB-QLSA",
    "同平台公平 benchmark",
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


def table_columns(line: str) -> int:
    s = line.strip()
    if not s.startswith("|") or not s.endswith("|"):
        return 0
    return len(s.strip("|").split("|"))


def main() -> int:
    failed = False
    warnings = 0

    if not REPORT.exists():
        print(f"[error] missing report: {REPORT}")
        failed = True
        text = ""
    else:
        text = REPORT.read_text(encoding="utf-8")

    if not SUBMISSION.exists():
        print(f"[error] missing submission directory: {SUBMISSION}")
        failed = True

    for phrase in FORBIDDEN:
        if phrase in text:
            print(f"[error] forbidden phrase found: {phrase}")
            failed = True

    if text.count("$$") % 2 != 0:
        print("[error] unbalanced $$ delimiters")
        failed = True

    for raw in IMAGE_RE.findall(text):
        path_text = raw.split("#", 1)[0].strip()
        candidate = (REPORT.parent / path_text).resolve()
        if not candidate.exists():
            print(f"[error] report image missing: {raw}")
            failed = True
        else:
            print(f"[ok] report image: {raw}")

    for line_no, line in enumerate(text.splitlines(), start=1):
        cols = table_columns(line)
        if cols > 7:
            print(f"[warning] wide table row line {line_no}: {cols} columns")
            warnings += 1

    if not (ROOT / "docs" / "personal_report_appendix.md").exists():
        print("[error] missing personal report appendix")
        failed = True

    for csv_name in KEY_CSV:
        root_path = ROOT / "results" / csv_name
        package_path = SUBMISSION / "results_key" / csv_name
        if not root_path.exists():
            print(f"[error] missing key CSV in results: {csv_name}")
            failed = True
        if not package_path.exists():
            print(f"[error] missing key CSV in submission package: {csv_name}")
            failed = True

    for required in [
        SUBMISSION / "final_report_extreme.md",
        SUBMISSION / "personal_report_appendix.md",
        SUBMISSION / "final_submission_readme.md",
        SUBMISSION / "figures",
        SUBMISSION / "results_key",
    ]:
        if not required.exists():
            print(f"[error] missing submission artifact: {required}")
            failed = True

    if failed:
        print(f"[failed] final submission check failed with {warnings} warning(s)")
        return 1
    print(f"[ok] final submission check passed with {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
