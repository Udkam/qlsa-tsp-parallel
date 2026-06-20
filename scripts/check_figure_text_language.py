#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that report figure text in make_report_figures.py is Chinese-ized.

This is a source-level guard: it cannot read text out of rendered PNGs, so it
verifies that the figure-producing script uses the expected Chinese titles and
labels, and that the old English titles are gone. English is intentionally kept
only for proper nouns, algorithm/library names and metric abbreviations.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPT = ROOT / "scripts" / "make_report_figures.py"

# Chinese strings that must appear (figure titles, layer titles, axis labels).
REQUIRED_CHINESE = [
    "OpenMP 多实例加速比",
    "OpenMP 多实例并行效率",
    "默认参数 Gap 对比",
    "调参与定向增强后的 Gap 改善",
    "QLSA 策略对比",
    "CUDA 在 berlin52 上的定位实验",
    "与论文运行时间参考对比",
    "困难实例平均 Gap 对比",
    "核心数据层",
    "搜索核心",
    "并行后端",
    "运行时间",
]

# Old English figure/title text that must no longer be present.
FORBIDDEN_ENGLISH = [
    "OpenMP Speedup across TSPLIB95 Instances",
    "OpenMP Parallel Efficiency",
    "Default-Parameter Gap",
    "Gap Reduction after Tuning",
    "QLSA Policy Comparison",
    "and CUDA Positioning",
    "Runtime Reference Comparison with the Paper",
    "Hard-Instance Mean Gap Comparison",
    "System Architecture and Data Flow",
    "Core Data Layer",
    "Search Core",
    "Parallel Backends",
]


def main() -> int:
    if not FIGURE_SCRIPT.exists():
        print(f"[error] missing figure script: {FIGURE_SCRIPT}")
        return 1

    try:
        text = FIGURE_SCRIPT.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        print(f"[error] {FIGURE_SCRIPT.relative_to(ROOT)} is not valid UTF-8: {exc}")
        return 1

    failed = False
    for phrase in REQUIRED_CHINESE:
        if phrase not in text:
            print(f"[error] missing required Chinese figure text: {phrase}")
            failed = True
    for phrase in FORBIDDEN_ENGLISH:
        if phrase in text:
            print(f"[error] leftover English figure text: {phrase}")
            failed = True

    if failed:
        return 1
    print(f"[ok] figure text language check passed: {FIGURE_SCRIPT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
