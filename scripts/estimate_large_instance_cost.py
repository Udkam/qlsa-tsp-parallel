#!/usr/bin/env python3
"""Estimate DistanceMatrix memory and run cost for large TSPLIB tiers."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
DEFAULT_INVENTORY = ROOT / "results" / "final" / "large_instance_inventory.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "dev" / "large_instance_cost_estimate.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def inferred_dimension(instance: str) -> int | None:
    match = re.search(r"(\d+)$", instance)
    return int(match.group(1)) if match else None


def risk(n: int) -> str:
    if n <= 300:
        return "green"
    if n <= 800:
        return "yellow"
    if n <= 1500:
        return "orange"
    return "red"


def mib(bytes_value: int) -> float:
    return bytes_value / (1024.0 * 1024.0)


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    inventory = Path(args.inventory)
    if not inventory.is_absolute():
        inventory = ROOT / inventory
    if inventory.exists():
        with inventory.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for tier, instances in config["tiers"].items():
        for instance in instances:
            rows.append({
                "instance": instance,
                "tier": tier,
                "path_exists": "false",
                "dimension": str(inferred_dimension(instance) or ""),
                "edge_weight_type": "",
                "bks": str(config["bks"].get(instance, "")),
                "status": "not_inventory_checked",
            })
    return rows


def main() -> int:
    args = parse_args()
    rows = load_rows(args)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    moves = args.iterations * args.chains * args.repeat
    lines = [
        "# 大规模实例内存与复杂度预估",
        "",
        f"- 默认估算参数：iterations={args.iterations}, chains={args.chains}, repeat={args.repeat}",
        f"- 总 move 数估算：`{moves}`",
        "- DistanceMatrix 使用 int32 一维连续数组，内存约为 `n*n*4` bytes。",
        "",
        "| tier | instance | exists | n | matrix MiB | moves | risk | note |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    red_instances: list[str] = []
    for row in rows:
        n = int(row["dimension"]) if row.get("dimension", "").isdigit() else (inferred_dimension(row["instance"]) or 0)
        matrix_bytes = n * n * 4
        level = risk(n)
        if level == "red":
            red_instances.append(row["instance"])
        note = "skip by default; require --allow-huge" if level == "red" else ""
        lines.append(
            f"| {row.get('tier', '')} | {row['instance']} | {row.get('path_exists', '')} | {n} | "
            f"{mib(matrix_bytes):.2f} | {moves} | {level} | {note} |"
        )
    lines.extend([
        "",
        "## 风险等级",
        "",
        "- green: n <= 300，适合作为 L1 正式验证。",
        "- yellow: 300 < n <= 800，适合作为 L2 stress。",
        "- orange: 800 < n <= 1500，只建议短预算 L3 stress。",
        "- red: n > 1500，默认不运行，必须显式确认。",
    ])
    if red_instances:
        lines.extend(["", "默认禁止运行的 red 档实例：", "", ", ".join(red_instances)])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
