#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Markdown report image references and unsafe wording."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs" / "final" / "final_report_course.md"
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

MOJIBAKE_TOKENS = [
    "\ufffd",
    "?" * 4,
    "\u951f\u65a4\u62f7",
    "\u95bf",
    "\u95c1",
    "\u95c2",
    "å",
    "ä",
]

FORBIDDEN_PHRASES = [
    "CUDA 全面优于 OpenMP",
    "CUDA 比 OpenMP 快",
    "CUDA 优于 OpenMP",
    "MPI VM 实验证明生产 HPC 性能",
    "QLSA 总是优于 SA",
    "所有实例都达到 BKS",
    "完整复刻 SB-QLSA",
    "同平台公平 benchmark",
    "百万城市级实验",
    "强证据链",
    "完整闭环",
    "工程闭环",
    "显著证明",
    "<你的姓名>",
    "<你的学号>",
    "<你的专业>",
    "{sa_avg_speed",
    "{qlsa_avg_speed",
    "figures" + "/final",
    "figures" + "\\final",
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
    "fig_cuda_candidate_policy_formal.png",
    "fig_qlsa_variant_alignment.png",
]


def choose_report() -> Path:
    if len(sys.argv) >= 2:
        candidate = Path(sys.argv[1])
        return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
    return DEFAULT_REPORT


def main() -> int:
    report = choose_report()
    if not report.exists():
        print(f"[error] missing report: {report}")
        return 1

    text = report.read_text(encoding="utf-8")
    failed = False

    for token in MOJIBAKE_TOKENS:
        if token in text:
            print(f"[error] possible mojibake token found: {token}")
            failed = True

    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            print(f"[error] forbidden or placeholder phrase found: {phrase}")
            failed = True

    image_refs = IMAGE_RE.findall(text)
    if len(image_refs) < 8:
        print(f"[error] expected at least 8 image references, found {len(image_refs)}")
        failed = True

    for alt_text, raw in image_refs:
        if re.search(r"图\s*[0-9一二三四五六七八九十]+\s*[:：]", alt_text):
            print(f"[error] image alt text should not contain numbered caption: {alt_text}")
            failed = True

        path_part = raw.split("#", 1)[0].strip()
        candidate = (report.parent / path_part).resolve()
        if not candidate.exists():
            print(f"[error] missing image asset: {raw} -> {candidate}")
            failed = True
        else:
            print(f"[ok] image asset exists: {raw}")

    for fig_name in REQUIRED_FIGURES:
        if fig_name not in text:
            print(f"[error] missing required figure reference: {fig_name}")
            failed = True

    if failed:
        return 1

    print(f"[ok] report asset check passed: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
