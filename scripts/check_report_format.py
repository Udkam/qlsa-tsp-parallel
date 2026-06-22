#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check structural and wording constraints for final-report Markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

MOJIBAKE_TOKENS = [
    "\ufffd",
    "?" * 4,
    "\u951f\u65a4\u62f7",
    "\u95bf\u7192\u67bb\u93b7",
    "\u95c1\u8de8\u5590\u93cb",
]

FORBIDDEN_PHRASES = [
    "TODO",
    "<你的姓名>",
    "<你的学号>",
    "<你的专业>",
    "CUDA 全面优于 OpenMP",
    "CUDA 比 OpenMP 快",
    "CUDA 优于 OpenMP",
    "MPI VM 实验证明生产 HPC 性能",
    "QLSA 总是优于 SA",
    "QLSA 全面优于 SA",
    "所有实例都达到 BKS",
    "默认参数下全部实例均达到 BKS",
    "同平台公平 benchmark",
    "严格公平 benchmark",
    "完整复刻 SB-QLSA",
    "百万城市级实验",
    "强证据链",
    "完整闭环",
    "工程闭环",
    "显著证明",
    "{sa_avg_speed",
    "{qlsa_avg_speed",
    "times100",
]

REQUIRED_TITLE_OPTIONS = [
    "# TSP 多搜索链 SA/QLSA 并行优化",
    "# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化",
]

REQUIRED_HEADINGS_COURSE = [
    "## 摘要",
    "## 1 选题背景与目标",
    "## 2 参考论文与本项目定位",
    "## 3 串行搜索内核设计",
    "## 4 多搜索链并行化方案",
    "## 5 工程实现与实验流程",
    "## 6 实验设计",
    "## 7 实验结果与分析",
    "## 8 与参考论文结果对比",
    "## 9 实施过程中遇到的问题",
    "## 10 总结与后续工作",
    "## 参考文献",
]

REQUIRED_FIGURES_COURSE = [
    "fig_course_01_openmp_speedup.png",
    "fig_course_02_openmp_efficiency.png",
    "fig_course_03_default_gap.png",
    "fig_course_04_targeted_quality.png",
    "fig_course_05_policy_comparison.png",
    "fig_course_06_cuda_boundary.png",
    "fig_course_07_mpi_scaling.png",
    "fig_course_09_paper_quality.png",
    "fig_course_10_openmp_thread_scaling.png",
    "fig_course_11_representative_openmp.png",
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


def resolve_report(argv: list[str]) -> Path:
    if len(argv) >= 2:
        report = Path(argv[1])
        return report if report.is_absolute() else (ROOT / report).resolve()
    return ROOT / "docs" / "final" / "final_report_course.md"


def main() -> int:
    report = resolve_report(sys.argv)
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

    if not any(title in text for title in REQUIRED_TITLE_OPTIONS):
        print("[error] missing required report title")
        failed = True

    for heading in REQUIRED_HEADINGS_COURSE:
        if heading not in text:
            print(f"[error] missing required heading: {heading}")
            failed = True

    if not ("课程要求" in text and "并行算法" in text and "加速比" in text):
        print("[error] missing course-goal discussion")
        failed = True

    risk_ok = (
        "不同语言" in text
        and "不同硬件" in text
        and ("不能作为同平台" in text or "不是同平台" in text or "不同平台" in text)
    )
    if not risk_ok:
        print("[error] missing paper-comparison risk statement")
        failed = True

    if text.count("$$") % 2 != 0:
        print("[error] unbalanced $$ math delimiters")
        failed = True

    if re.search(r"(?<!\\)frac", text):
        print("[error] possible broken LaTeX fraction: frac")
        failed = True

    if re.search(r"(?:^|\n)\s*[图表]\s*[0-9一二三四五六七八九十]+[:：]", text):
        print("[error] found standalone numbered figure/table caption style")
        failed = True

    if re.search(r"\b表\s*[0-9一二三四五六七八九十]+\b", text):
        print("[error] found numbered table wording")
        failed = True

    if "## 附录 A" in text or "附录 A 个人工作说明" in text:
        print("[error] main course report should not include personal appendix")
        failed = True

    images = IMAGE_RE.findall(text)
    if len(images) < 8:
        print(f"[error] report should reference at least 8 figures, found {len(images)}")
        failed = True

    for alt_text, raw in images:
        if re.search(r"图\s*[0-9一二三四五六七八九十]+[:：]?", alt_text):
            print(f"[error] image alt text contains figure numbering: {alt_text}")
            failed = True
        path_text = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_text).resolve()
        if not candidate.exists():
            print(f"[error] missing image: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image exists: {raw}")

    for fig_name in REQUIRED_FIGURES_COURSE:
        if fig_name not in text:
            print(f"[error] missing required figure reference: {fig_name}")
            failed = True

    for idx, line in enumerate(lines, start=1):
        cols = count_markdown_columns(line)
        if cols > 6 and not is_separator(line):
            print(f"[warning] table row has {cols} columns at line {idx}: {line[:120]}")
            warnings += 1

    if failed:
        print(f"[failed] format check failed with {warnings} warning(s)")
        return 1

    print(f"[ok] format check passed with {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
