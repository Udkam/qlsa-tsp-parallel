#!/usr/bin/env python3
"""Check public-facing files for private information and encoding problems."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TARGETS = [
    ROOT / "README.md",
    ROOT / "docs" / "final" / "final_report_public.md",
    ROOT / "submission" / "public",
]

PRIVATE_PATTERNS = [
    ("name", re.compile(r"陈乐浚")),
    ("student_id_exact", re.compile(r"22361054")),
    ("qq_email", re.compile(r"[A-Za-z0-9._%+-]+@qq\.com", re.IGNORECASE)),
    ("sysu_email", re.compile(r"[A-Za-z0-9._%+-]+@mail\.sysu\.edu\.cn", re.IGNORECASE)),
    ("student_id_like", re.compile(r"(?<!\d)22[0-9]{6}(?!\d)")),
]

MOJIBAKE_PATTERNS = [
    ("question_marks", "????"),
    ("replacement_char", "�"),
    ("mojibake_cn", "锟斤拷"),
]

TEXT_SUFFIXES = {".md", ".txt", ".csv"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for target in PUBLIC_TARGETS:
        if target.is_file() and target.suffix.lower() in TEXT_SUFFIXES:
            files.append(target)
        elif target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                    files.append(path)
    return files


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"{path.relative_to(ROOT)}: UTF-8 decode failed: {exc}"]

    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        for label, pattern in PRIVATE_PATTERNS:
            if pattern.search(line):
                errors.append(f"{path.relative_to(ROOT)}:{idx}: private info matched {label}")
        for label, token in MOJIBAKE_PATTERNS:
            if token in line:
                errors.append(f"{path.relative_to(ROOT)}:{idx}: mojibake token matched {label}")
        if re.search(r"(?:å|ä|Ã|Â)[A-Za-z]{2,}", line):
            errors.append(f"{path.relative_to(ROOT)}:{idx}: possible mojibake latin fragment")
    return errors


def main() -> int:
    files = iter_files()
    if not files:
        print("[warning] no public-facing text files found to check")
        return 0

    all_errors: list[str] = []
    for path in files:
        all_errors.extend(check_file(path))

    if all_errors:
        for error in all_errors:
            print(f"[error] {error}")
        return 1

    print(f"[ok] privacy and UTF-8 checks passed for {len(files)} public-facing file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
