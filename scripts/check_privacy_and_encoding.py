#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check public-facing project text files for private data and mojibake."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "README.md",
    ROOT / "docs" / "final" / "report.md",
    ROOT / "docs" / "final" / "personal_report.md",
    ROOT / "docs" / "final" / "known_limitations.md",
    ROOT / "docs" / "final" / "reproduction_commands.md",
    ROOT / "results" / "final" / "RESULTS_INDEX.md",
]
TEXT_SUFFIXES = {".md", ".txt", ".csv"}

PRIVATE_PATTERNS = [
    ("qq_email", re.compile(r"[A-Za-z0-9._%+-]+@qq\.com", re.IGNORECASE)),
    ("sysu_email", re.compile(r"[A-Za-z0-9._%+-]+@mail\.sysu\.edu\.cn", re.IGNORECASE)),
    ("vm_private_ip", re.compile(r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}")),
    ("ssh_key_marker", re.compile(r"BEGIN (?:OPENSSH|RSA) PRIVATE KEY")),
    ("local_key_path", re.compile(r"\.ssh_mpi_vm_key|E:\\tmp\\mpi_vm_key", re.IGNORECASE)),
    ("password_word", re.compile(r"密码[:：]|password[:=]", re.IGNORECASE)),
]

MOJIBAKE_TOKENS = [
    "?" * 4,
    "\ufffd",
    "\u951f\u65a4\u62f7",
    "\u951b",
    "\u6b5a",
    "\u95bf",
    "\u95c1",
    "\u95c2",
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS:
        if target.is_file() and target.suffix.lower() in TEXT_SUFFIXES:
            files.append(target)
        elif target.is_dir():
            files.extend(
                path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
            )
    return sorted(files)


def check_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"{path.relative_to(ROOT)}: UTF-8 decode failed: {exc}"]

    errors: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        for label, pattern in PRIVATE_PATTERNS:
            if pattern.search(line):
                errors.append(f"{path.relative_to(ROOT)}:{idx}: private info matched {label}")
        for token in MOJIBAKE_TOKENS:
            if token in line:
                errors.append(f"{path.relative_to(ROOT)}:{idx}: possible mojibake token `{token}`")
    return errors


def main() -> int:
    files = iter_files()
    errors: list[str] = []
    for path in files:
        errors.extend(check_file(path))

    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    print(f"[ok] privacy and UTF-8/mojibake checks passed for {len(files)} public-facing file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
