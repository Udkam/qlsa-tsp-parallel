#!/usr/bin/env python3
"""Download selected large TSPLIB95 symmetric TSP instances.

The downloader is conservative: each downloaded file is validated before being
kept, HTML error pages are rejected, and source/SHA256 metadata are recorded.
It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
DEFAULT_DOC = ROOT / "docs" / "dev" / "large_instance_data_sources.md"
DEFAULT_CSV = ROOT / "results" / "final" / "large_instance_download_status.csv"

HEADER_KEYS = [
    "NAME",
    "TYPE",
    "COMMENT",
    "DIMENSION",
    "CAPACITY",
    "EDGE_WEIGHT_TYPE",
    "EDGE_WEIGHT_FORMAT",
    "DISPLAY_DATA_TYPE",
]
SECTION_KEYS = [
    "NODE_COORD_SECTION",
    "EDGE_WEIGHT_SECTION",
    "DISPLAY_DATA_SECTION",
    "DEPOT_SECTION",
    "EOF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", nargs="+", choices=["L1", "L2", "L3"], default=["L1", "L2", "L3"])
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-doc", default=str(DEFAULT_DOC))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_instances(config: dict, tiers: list[str], explicit: list[str] | None) -> list[tuple[str, str]]:
    tier_for = {instance: tier for tier, instances in config["tiers"].items() for instance in instances}
    if explicit:
        return [(tier_for.get(instance, "custom"), instance) for instance in explicit]
    out: list[tuple[str, str]] = []
    for tier in tiers:
        out.extend((tier, instance) for instance in config["tiers"].get(tier, []))
    return out


def source_urls(instance: str) -> list[str]:
    return [
        f"https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{instance}.tsp.gz",
        f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{instance}.tsp.gz",
        f"https://raw.githubusercontent.com/mastqe/tsplib/master/{instance}.tsp",
        f"https://raw.githubusercontent.com/juandes/tsplib/master/tsp/{instance}.tsp",
        f"https://raw.githubusercontent.com/MLCourseProjects/TSPLIB/master/{instance}.tsp",
    ]


def download(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "parallel-algorithm-tsplib-downloader/1.0"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        data = response.read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    return data


def is_html(data: bytes) -> bool:
    head = data[:256].decode("utf-8", errors="ignore").lower()
    return "<html" in head or "<!doctype" in head or "<body" in head


def tokens(text: str) -> list[str]:
    return text.replace("\r", " ").replace("\n", " ").split()


def clean_key(token: str) -> str:
    return token.rstrip(":").upper()


def normalize_tsplib(text: str) -> str:
    """Normalize one-line TSPLIB mirrors into line-oriented TSPLIB text."""
    if text.count("\n") > 5:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"

    parts = tokens(text)
    i = 0
    lines: list[str] = []
    known = set(HEADER_KEYS + SECTION_KEYS)
    while i < len(parts):
        key = clean_key(parts[i])
        if key in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION"}:
            lines.append(key)
            i += 1
            if key == "NODE_COORD_SECTION" or key == "DISPLAY_DATA_SECTION":
                while i < len(parts) and clean_key(parts[i]) != "EOF":
                    if i + 2 >= len(parts):
                        break
                    lines.append(f"{parts[i]} {parts[i + 1]} {parts[i + 2]}")
                    i += 3
            elif key == "EDGE_WEIGHT_SECTION":
                row: list[str] = []
                while i < len(parts) and clean_key(parts[i]) != "EOF":
                    row.append(parts[i])
                    if len(row) >= 16:
                        lines.append(" ".join(row))
                        row = []
                    i += 1
                if row:
                    lines.append(" ".join(row))
            continue
        if key == "EOF":
            lines.append("EOF")
            i += 1
            break
        if key in HEADER_KEYS:
            i += 1
            if i < len(parts) and parts[i] == ":":
                i += 1
            value: list[str] = []
            while i < len(parts) and clean_key(parts[i]) not in known and parts[i] != ":":
                value.append(parts[i])
                i += 1
            lines.append(f"{key}: {' '.join(value)}".rstrip())
            continue
        i += 1
    if not lines:
        return text.strip() + "\n"
    if lines[-1] != "EOF":
        lines.append("EOF")
    return "\n".join(lines) + "\n"


def header_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:?\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def validate(text: str) -> tuple[bool, str, str, str]:
    if "<html" in text.lower() or "<!doctype" in text.lower():
        return False, "", "", "html response rejected"
    name = header_value(text, "NAME")
    typ = header_value(text, "TYPE")
    dimension = header_value(text, "DIMENSION")
    edge = header_value(text, "EDGE_WEIGHT_TYPE")
    missing = [label for label, value in [
        ("NAME", name),
        ("TYPE", typ),
        ("DIMENSION", dimension),
        ("EDGE_WEIGHT_TYPE", edge),
    ] if not value]
    if missing:
        return False, dimension, edge, "missing header fields: " + ", ".join(missing)
    if typ.upper() != "TSP":
        return False, dimension, edge, f"TYPE is {typ}, expected TSP"
    if not re.fullmatch(r"\d+", dimension):
        return False, dimension, edge, f"invalid DIMENSION {dimension}"
    if "NODE_COORD_SECTION" not in text.upper() and "EDGE_WEIGHT_SECTION" not in text.upper():
        return False, dimension, edge, "missing supported section"
    if "EOF" not in text.upper():
        return False, dimension, edge, "missing EOF marker"
    return True, dimension, edge, "ok"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    output_doc = Path(args.output_doc)
    if not output_doc.is_absolute():
        output_doc = ROOT / output_doc
    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    selected = select_instances(config, args.tier, args.instances)
    rows: list[dict[str, str]] = []
    for tier, instance in selected:
        dest = data_dir / f"{instance}.tsp"
        if dest.exists() and not args.force:
            text = normalize_tsplib(dest.read_text(encoding="utf-8", errors="replace"))
            ok, dimension, edge, message = validate(text)
            if ok and text != dest.read_text(encoding="utf-8", errors="replace"):
                dest.write_text(text, encoding="utf-8", newline="\n")
            rows.append({
                "instance": instance, "tier": tier, "source": "existing", "status": "exists" if ok else "invalid_existing",
                "path": str(dest.relative_to(ROOT)), "dimension": dimension, "edge_weight_type": edge,
                "sha256": sha256_text(text) if ok else "", "message": message,
            })
            print(f"[skip] {instance}: existing ({message})")
            continue

        success = False
        last_message = ""
        for url in source_urls(instance):
            try:
                data = download(url, args.timeout)
                if is_html(data):
                    last_message = "html response rejected"
                    continue
                text = normalize_tsplib(data.decode("utf-8", errors="replace"))
                ok, dimension, edge, message = validate(text)
                if not ok:
                    last_message = message
                    continue
                dest.write_text(text, encoding="utf-8", newline="\n")
                digest = sha256_text(text)
                rows.append({
                    "instance": instance, "tier": tier, "source": url, "status": "downloaded",
                    "path": str(dest.relative_to(ROOT)), "dimension": dimension, "edge_weight_type": edge,
                    "sha256": digest, "message": "ok",
                })
                print(f"[ok] {instance}: {url}")
                success = True
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, gzip.BadGzipFile) as exc:
                last_message = f"{type(exc).__name__}: {exc}"
                continue
        if not success:
            if dest.exists() and args.force:
                try:
                    dest.unlink()
                except OSError:
                    pass
            rows.append({
                "instance": instance, "tier": tier, "source": "", "status": "failed",
                "path": str(dest.relative_to(ROOT)), "dimension": "", "edge_weight_type": "",
                "sha256": "", "message": last_message,
            })
            print(f"[failed] {instance}: {last_message}")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance", "tier", "source", "status", "path", "dimension", "edge_weight_type", "sha256", "message"
        ])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 大规模 TSPLIB95 数据来源记录",
        "",
        "本文件由 `scripts/download_large_tsplib_subset.py` 生成。下载脚本会拒绝 HTML 错误页，并在保存前检查 NAME、TYPE、DIMENSION、EDGE_WEIGHT_TYPE 和 TSPLIB section。",
        "",
        "| tier | instance | status | dimension | edge_weight_type | source | sha256 | message |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tier']} | {row['instance']} | {row['status']} | {row['dimension']} | "
            f"{row['edge_weight_type']} | {row['source']} | {row['sha256']} | {row['message']} |"
        )
    output_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output_csv.relative_to(ROOT)}")
    print(f"[ok] wrote {output_doc.relative_to(ROOT)}")
    failed = [r["instance"] for r in rows if r["status"] == "failed"]
    return 1 if failed and len(failed) == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
