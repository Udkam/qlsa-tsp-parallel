#!/usr/bin/env python3
"""Check image references and high-risk wording in the final report."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = [
    ROOT / "docs" / "final" / "final_report_master_v2.md",
    ROOT / "docs" / "final" / "final_report_master.md",
    ROOT / "docs" / "final" / "final_report_course.md",
    ROOT / "docs" / "final" / "final_report_public.md",
]
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

FORBIDDEN_PHRASES = [
    "CUDA 比 OpenMP 快",
    "CUDA 快于 OpenMP",
    "CUDA 优于 OpenMP",
    "QLSA 总是优于 SA",
    "QLSA 全面优于 SA",
    "所有实例都达到 BKS",
    "默认参数下全部实例均达到 BKS",
    "完全复刻 SB-QLSA",
    "同平台公平 benchmark",
    "严格公平 benchmark",
    "图x：",
    "图X：",
    "<你的姓名>",
    "<你的学号>",
    "<你的专业>",
    "TODO",
    "请补充",
]

REQUIRED_CONTENT = [
    "问题背景与课程目标映射",
    "论文方法拆解",
    "系统设计",
    "并行设计",
    "实验设计",
    "实验结果",
    "与论文对比",
    "工程难度",
    "局限性",
]


def choose_report() -> Path:
    if len(sys.argv) >= 2:
        candidate = Path(sys.argv[1])
        return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
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

    for alt_text, raw in image_paths:
        if re.search(r"图\s*[0-9一二三四五六七八九十xX]+\s*[:：]", alt_text):
            print(f"[error] image alt text uses placeholder figure numbering: {alt_text}")
            failed = True
        path_part = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_part).resolve()
        if not candidate.exists():
            print(f"[error] missing image asset: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image asset exists: {raw}")

    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            print(f"[error] forbidden or placeholder phrase found: {phrase}")
            failed = True

    for item in REQUIRED_CONTENT:
        if item not in text:
            print(f"[error] missing required content: {item}")
            failed = True

    if failed:
        return 1
    print(f"[ok] report asset check passed: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
