#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check structural and wording constraints for final-report Markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

FORBIDDEN_PHRASES = [
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
    "完整复刻 SB-QLSA",
    "图x：",
    "图X：",
]

MOJIBAKE_TOKENS = [
    "\ufffd",
    "????",
    "锟斤拷",
    "閿熸枻鎷",
]

REQUIRED_HEADINGS = [
    "# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化",
    "## 摘要",
    "## 1. 基本信息",
    "## 2. 预期目标与实际完成情况",
    "## 3. 参考论文方法与实现差异",
    "## 4. 方案设计",
    "## 5. 并行方案设计",
    "## 6. 实施过程与解决的问题",
    "## 7. 实验设计",
    "## 8. 实验结果与分析",
    "## 9. 与近期论文结果对比",
    "## 10. 工程难度与证据等级",
    "## 11. 局限性",
    "## 12. 总结",
]

REQUIRED_FIGURES = [
    "fig01_architecture_pipeline.png",
    "fig02_openmp_speedup.png",
    "fig03_openmp_efficiency.png",
    "fig04_default_gap.png",
    "fig05_tuning_curve.png",
    "fig06_policy_comparison.png",
    "fig07_cuda_positioning.png",
    "fig08_paper_runtime_comparison.png",
    "fig09_paper_quality_comparison.png",
    "fig13_mpi_vm_scaling_formal.png",
    "fig14_hpc_hybrid_architecture.png",
]


def count_markdown_columns(line: str) -> int:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return 0
    return len(stripped.strip("|").split("|"))


def is_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    body = stripped.strip("|").replace(":", "").replace("-", "").replace("|", "").strip()
    return body == ""


def main() -> int:
    report = Path(sys.argv[1]) if len(sys.argv) >= 2 else ROOT / "docs" / "final" / "final_report_course.md"
    if not report.is_absolute():
        report = (ROOT / report).resolve()
    if not report.exists():
        print(f"[error] report does not exist: {report}")
        return 1

    text = report.read_text(encoding="utf-8")
    lines = text.splitlines()
    failed = False
    warnings = 0

    for token in MOJIBAKE_TOKENS:
        if token in text:
            print(f"[error] possible mojibake token found: {token}")
            failed = True

    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            print(f"[error] forbidden phrase found: {phrase}")
            failed = True

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            print(f"[error] missing required heading: {heading}")
            failed = True

    if "课程评分点" not in text or "支撑材料" not in text:
        print("[error] missing course requirement mapping")
        failed = True

    has_risk_statement = (
        "不同硬件" in text
        and "不同语言" in text
        and ("严格基准" in text or "不是严格 benchmark" in text or "不能作为同硬件同语言下的严格基准" in text)
    )
    if not has_risk_statement:
        print("[error] missing paper-comparison risk statement")
        failed = True

    if text.count("$$") % 2 != 0:
        print("[error] unbalanced $$ math delimiters")
        failed = True

    images = IMAGE_RE.findall(text)
    if len(images) < 8:
        print("[error] report should reference at least 8 figures")
        failed = True

    for alt_text, raw in images:
        if re.search(r"图\s*[0-9一二三四五六七八九十]+", alt_text):
            print(f"[error] image alt text contains figure numbering: {alt_text}")
            failed = True
        path_text = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_text).resolve()
        if not candidate.exists():
            print(f"[error] missing image: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image exists: {raw}")

    for fig_name in REQUIRED_FIGURES:
        if fig_name not in text:
            print(f"[error] missing required figure reference: {fig_name}")
            failed = True

    for idx, line in enumerate(lines, start=1):
        cols = count_markdown_columns(line)
        if cols > 7 and not is_separator(line):
            print(f"[warning] table row has {cols} columns at line {idx}: {line[:120]}")
            warnings += 1

    if failed:
        print(f"[failed] format check failed with {warnings} warning(s)")
        return 1
    print(f"[ok] format check passed with {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
