#!/usr/bin/env python3
"""Check structural and wording constraints for the final report markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

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
    "完全复刻 SB-QLSA",
    "本项目围绕",
    "通过本文我们实现了",
    "可以看到明显提升",
    "图x：",
    "图X：",
]

REQUIRED_HEADINGS = [
    "# 面向旅行推销员问题的 Q-Learning 辅助模拟退火算法并行化实现与性能优化",
    "## 摘要",
    "## 1. 基本信息",
    "## 2. 预期目标与实际完成情况",
    "## 3. 参考论文方法与本项目定位",
    "## 4. 方案设计",
    "## 5. 并行方案设计",
    "## 6. 实施过程与解决的问题",
    "## 7. 实验设计",
    "## 8. 实验结果与分析",
    "## 9. 与近期论文结果对比",
    "## 10. 工程难度与完成质量说明",
    "## 11. 局限性",
    "## 12. 总结",
]

MASTER_HEADINGS = [
    "# 1. 问题背景与课程目标映射",
    "# 2. 论文方法拆解",
    "# 3. 系统设计",
    "# 4. 并行设计",
    "# 5. 实验设计",
    "# 6. 实验结果",
    "# 7. 与论文对比",
    "# 8. 工程难度",
    "# 9. 局限性",
    "# 10. 总结",
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
    if len(sys.argv) >= 2:
        report = Path(sys.argv[1])
    else:
        report = ROOT / "docs" / "final" / "final_report_course.md"
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

    required_headings = MASTER_HEADINGS if report.name.startswith("final_report_master") else REQUIRED_HEADINGS
    for heading in required_headings:
        if heading not in text:
            print(f"[error] missing required heading: {heading}")
            failed = True

    has_course_mapping = (
        ("预期目标" in text and "完成情况" in text and "课程评分点" in text)
        or ("课程评分点" in text and "课程目标" in text and "报告证据" in text)
    )
    if not has_course_mapping:
        print("[error] missing course requirement mapping")
        failed = True

    has_time_risk = "绝对时间不可直接比较" in text or "绝对时间不能视作同一平台下的严格性能比较" in text
    if "不同硬件" not in text or "不同语言" not in text or not has_time_risk:
        print("[error] missing paper-comparison risk statement")
        failed = True

    if text.count("$$") % 2 != 0:
        print("[error] unbalanced $$ math delimiters")
        failed = True

    for alt_text, raw in IMAGE_RE.findall(text):
        if re.search(r"图\s*[0-9一二三四五六七八九十xX]+\s*[:：]", alt_text):
            print(f"[error] image alt text uses placeholder figure numbering: {alt_text}")
            failed = True
        path_text = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_text).resolve()
        if not candidate.exists():
            print(f"[error] missing image: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image exists: {raw}")

    if len(IMAGE_RE.findall(text)) < 6:
        print("[error] report should reference at least 6 figures")
        failed = True

    required_figures = [
        "fig01_architecture_pipeline.png",
        "fig02_openmp_speedup.png",
        "fig03_openmp_efficiency.png",
        "fig04_default_gap.png",
        "fig05_tuning_curve.png",
        "fig06_policy_comparison.png",
        "fig07_cuda_positioning.png",
        "fig08_paper_runtime_comparison.png",
        "fig09_paper_quality_comparison.png",
    ]
    for fig_name in required_figures:
        if fig_name not in text:
            print(f"[error] missing required final figure reference: {fig_name}")
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
