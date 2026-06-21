#!/usr/bin/env python3
"""Inspect local TSPLIB95 large-instance files without downloading data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
DEFAULT_OUTPUT = ROOT / "docs" / "dev" / "large_instance_data_status.md"
DEFAULT_INVENTORY = ROOT / "results" / "final" / "large_instance_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", nargs="+", choices=["L1", "L2", "L3"], default=["L1", "L2", "L3"])
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_instances(config: dict, tiers: Iterable[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for tier in tiers:
        for instance in config["tiers"].get(tier, []):
            pairs.append((tier, instance))
    return pairs


def inferred_dimension(instance: str) -> int | None:
    match = re.search(r"(\d+)$", instance)
    return int(match.group(1)) if match else None


def parse_header(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION", "EOF"}:
            break
        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        fields[key.strip().upper()] = value.strip()
    return fields


def status_for(path_exists: bool, dimension: str, edge_weight_type: str) -> str:
    if not path_exists:
        return "missing"
    if not dimension:
        return "present_header_incomplete"
    if edge_weight_type.upper() == "CEIL_2D":
        return "present_check_bks_edge_weight_type"
    return "present"


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    inventory = Path(args.inventory)
    if not inventory.is_absolute():
        inventory = ROOT / inventory
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for tier, instance in iter_instances(config, args.tier):
        path = data_dir / f"{instance}.tsp"
        exists = path.exists()
        header = parse_header(path) if exists else {}
        dimension = header.get("DIMENSION", "")
        if not dimension and inferred_dimension(instance) is not None:
            dimension = str(inferred_dimension(instance))
        edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "")
        rows.append({
            "instance": instance,
            "tier": tier,
            "path_exists": "true" if exists else "false",
            "dimension": dimension,
            "edge_weight_type": edge_weight_type,
            "bks": str(config["bks"].get(instance, "")),
            "status": status_for(exists, dimension, edge_weight_type),
        })

    with inventory.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance", "tier", "path_exists", "dimension", "edge_weight_type", "bks", "status"
        ])
        writer.writeheader()
        writer.writerows(rows)

    present = [r for r in rows if r["path_exists"] == "true"]
    missing = [r for r in rows if r["path_exists"] == "false"]
    lines = [
        "# 大规模 TSPLIB95 实例数据状态",
        "",
        f"- 数据目录：`{data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir}`",
        f"- 已检查 tier：{', '.join(args.tier)}",
        f"- 已存在实例数：{len(present)}",
        f"- 缺失实例数：{len(missing)}",
        f"- Inventory CSV：`{inventory.relative_to(ROOT)}`",
        "",
        "## 实例清单",
        "",
        "| tier | instance | exists | dimension | edge_weight_type | BKS | status |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tier']} | {row['instance']} | {row['path_exists']} | {row['dimension']} | "
            f"{row['edge_weight_type']} | {row['bks']} | {row['status']} |"
        )
    lines.extend([
        "",
        "## 数据准备说明",
        "",
        "本脚本不依赖网络，也不会自动下载数据。若某个实例缺失，请从 TSPLIB95 镜像或课程允许的数据源手动下载对应 `.tsp` 文件，并放入 `data/` 目录后重新运行本脚本。",
        "",
        "特别注意：`dsj1000` 的 BKS 与 `EDGE_WEIGHT_TYPE` 有关；如果本地文件为 `CEIL_2D` 或其它格式，分析时需要核对对应 BKS。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    print(f"[ok] wrote {inventory.relative_to(ROOT)}")
    if missing:
        print("[warning] missing instances:", ", ".join(r["instance"] for r in missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
