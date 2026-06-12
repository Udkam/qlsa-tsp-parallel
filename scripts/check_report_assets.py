#!/usr/bin/env python3
"""Check local image references and basic wording constraints in final reports."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = [
    ROOT / "docs" / "final_report_v3.md",
    ROOT / "docs" / "final_report_v2.md",
]
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

FORBIDDEN = [
    "CUDA 比 OpenMP 快",
    "CUDA 快于 OpenMP",
    "CUDA 优于 OpenMP",
    "QLSA 总是优于 SA",
    "QLSA 全面优于 SA",
    "所有实例都达到 BKS",
    "默认参数下全部实例均达到 BKS",
    "<你的姓名>",
    "<你的学号>",
    "<你的专业>",
    "TODO",
    "请补充",
]


def choose_report() -> Path:
    if len(sys.argv) >= 2:
        p = Path(sys.argv[1])
        return p if p.is_absolute() else (ROOT / p).resolve()
    for report in DEFAULT_REPORTS:
        if report.exists():
            return report
    return DEFAULT_REPORTS[0]


def main() -> int:
    report = choose_report()
    if not report.exists():
        print(f"[error] missing report: {report}")
        return 1

    text = report.read_text(encoding="utf-8")
    failed = False

    image_paths = IMAGE_RE.findall(text)
    if len(image_paths) < 6:
        print(f"[error] expected at least 6 image references, found {len(image_paths)}")
        failed = True

    for raw in image_paths:
        path_part = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_part).resolve()
        if not candidate.exists():
            print(f"[error] missing image asset: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image asset exists: {raw}")

    for phrase in FORBIDDEN:
        if phrase in text:
            print(f"[error] forbidden or placeholder phrase found: {phrase}")
            failed = True

    required_any = [
        ["课程要求与完成情况", "课程要求与项目完成度对应关系"],
        ["与参考论文的对比", "与论文结果对比"],
        ["OpenMP", "CUDA", "QLSA"],
    ]
    for options in required_any:
        if not any(opt in text for opt in options):
            print(f"[error] missing required content, expected one of: {options}")
            failed = True

    if failed:
        return 1
    print(f"[ok] report assets and basic wording checks passed: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
