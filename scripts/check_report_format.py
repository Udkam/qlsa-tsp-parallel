#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check structural and wording constraints for the final course report."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)

MOJIBAKE_TOKENS = [
    "\ufffd",
    "?" * 4,
    "\u951f\u65a4\u62f7",
    "\u95bf",
    "\u95c1",
    "\u95c2",
]

FORBIDDEN_PHRASES = [
    "TODO",
    "<你的姓名>",
    "<你的学号>",
    "<你的专业>",
    "CUDA 全面优于 OpenMP",
    "CUDA 比 OpenMP 快",
    "MPI VM 实验证明生产 HPC 性能",
    "QLSA 总是优于 SA",
    "所有实例都达到 BKS",
    "同平台公平 benchmark",
    "完整复刻 SB-QLSA",
    "百万城市级实验",
    "强证据链",
    "完整闭环",
    "工程闭环",
    "显著证明",
    "不能写成",
    "{sa_avg_speed",
    "{qlsa_avg_speed",
    "times100",
    "figures" + "/final",
    "figures" + "\\final",
]

REQUIRED_TITLE_OPTIONS = [
    "# TSP 多搜索链 SA/QLSA 并行优化",
    "# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化",
]

REQUIRED_HEADINGS = [
    "## 摘要",
    "## 1 选题背景与目标",
    "## 2 参考论文与本项目定位",
    "## 3 串行搜索内核设计",
    "## 4 多搜索链并行化方案",
    "## 5 工程实现与实验流程",
    "## 6 实验设计",
    "## 7 实验结果与分析",
    "## 8 与参考论文结果对比",
    "## 9 实验过程中遇到的问题",
    "## 10 总结",
    "## 11 后续可优化方向",
    "## 参考文献",
]

REQUIRED_FIGURES = [
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


def resolve_report(argv: list[str]) -> Path:
    if len(argv) >= 2:
        report = Path(argv[1])
        return report if report.is_absolute() else (ROOT / report).resolve()
    return ROOT / "docs" / "final" / "report.md"


def image_refs(text: str) -> list[str]:
    refs = [raw for _, raw in MD_IMAGE_RE.findall(text)]
    refs.extend(HTML_IMAGE_RE.findall(text))
    return refs


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

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            print(f"[error] missing required heading: {heading}")
            failed = True

    if not ("目标" in text and "并行" in text and "加速比" in text):
        print("[error] missing project-goal and parallel-performance discussion")
        failed = True

    refs = image_refs(text)
    if len(refs) < 8:
        print(f"[error] report should reference at least 8 figures, found {len(refs)}")
        failed = True

    for raw in refs:
        candidate = (report.parent / raw.split("#", 1)[0].strip()).resolve()
        if not candidate.exists():
            print(f"[error] missing image asset: {raw}")
            failed = True
        else:
            print(f"[ok] image exists: {raw}")

    for fig_name in REQUIRED_FIGURES:
        if fig_name not in text:
            print(f"[error] missing required figure: {fig_name}")
            failed = True

    for idx, line in enumerate(lines, start=1):
        columns = count_markdown_columns(line)
        if columns > 7 and not is_separator(line):
            print(f"[warning] wide table at line {idx}: {columns} columns")
            warnings += 1

    if failed:
        print(f"[failed] format check failed with {warnings} warning(s)")
        return 1

    print(f"[ok] format check passed with {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
