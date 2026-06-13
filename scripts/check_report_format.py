#!/usr/bin/env python3
"""Check formatting constraints for the final report markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

ERROR_PHRASES = [
    "# 0.",
    "TODO",
    "请补充",
    "<你的姓名>",
    "<你的学号>",
    "<你的专业>",
    "CUDA 比 OpenMP 快",
    "CUDA 快于 OpenMP",
    "CUDA 优于 OpenMP",
    "QLSA 总是优于 SA",
    "QLSA 全面优于 SA",
    "所有实例都达到 BKS",
    "默认参数下全部实例均达到 BKS",
    "同平台公平 benchmark",
    "严格公平 benchmark",
    "完全复刻论文 SB-QLSA",
]


def count_markdown_columns(line: str) -> int:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return 0
    cells = [c for c in stripped.strip("|").split("|")]
    return len(cells)


def is_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    body = stripped.strip("|").replace(":", "").replace("-", "").replace("|", "").strip()
    return body == ""


def main() -> int:
    if len(sys.argv) >= 2:
        report = Path(sys.argv[1])
    else:
        report = ROOT / "docs" / "final_report_v3.md"
    if not report.is_absolute():
        report = (ROOT / report).resolve()

    if not report.exists():
        print(f"[error] report does not exist: {report}")
        return 1

    text = report.read_text(encoding="utf-8")
    lines = text.splitlines()
    failed = False
    warnings = 0

    for phrase in ERROR_PHRASES:
        if phrase in text:
            print(f"[error] forbidden phrase found: {phrase}")
            failed = True

    for raw in IMAGE_RE.findall(text):
        path_text = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_text).resolve()
        if not candidate.exists():
            print(f"[error] missing image: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image exists: {raw}")

    for idx, line in enumerate(lines, start=1):
        cols = count_markdown_columns(line)
        if cols > 7 and not is_separator(line):
            print(f"[warning] table row has {cols} columns at line {idx}: {line[:120]}")
            warnings += 1

    required_groups = [
        ["## 摘要"],
        ["## 1. 基本信息"],
        ["## 2. 课程要求与完成情况", "## 2. 课程要求与完成度"],
        ["与参考论文的对比", "与论文结果对比"],
        ["总结与贡献"],
    ]
    for options in required_groups:
        if not any(option in text for option in options):
            print(f"[error] missing required heading/content, expected one of: {options}")
            failed = True

    if "预期目标" not in text or "实际" not in text:
        print("[error] missing expected-vs-actual completion table")
        failed = True

    if failed:
        print(f"[failed] format check failed with {warnings} warning(s)")
        return 1
    print(f"[ok] format check passed with {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
