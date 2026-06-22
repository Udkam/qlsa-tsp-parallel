#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a short Nsight profiling pass for CUDA candidate mode when tools exist.

The script is intentionally conservative: it records tool availability and the
exact commands, but it does not invent occupancy/bandwidth conclusions when
Nsight is absent or metrics are unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="a280")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--candidates-per-iter", type=int, default=128)
    parser.add_argument("--policy", choices=["best", "random", "hybrid"], default="best")
    parser.add_argument("--output-dir", default="results/logs/nsight")
    parser.add_argument("--markdown", default="docs/dev/cuda_nsight_profile_analysis.md")
    return parser.parse_args()


def find_executable() -> Path:
    candidates = [
        ROOT / "build-cuda-ninja" / "tsp_sa.exe",
        ROOT / "build-cuda-ninja" / "tsp_sa",
        ROOT / "build" / "Release" / "tsp_sa.exe",
        ROOT / "build" / "tsp_sa",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("tsp_sa executable not found; build the project first")


def run_command(command: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout


def main() -> int:
    args = parse_args()
    exe = find_executable()
    input_path = ROOT / "data" / f"{args.instance}.tsp"
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = ROOT / args.markdown
    markdown.parent.mkdir(parents=True, exist_ok=True)

    base_command = [
        str(exe),
        "--input", str(input_path),
        "--parallel", "cuda",
        "--cuda_mode", "candidate",
        "--cuda_candidate_policy", args.policy,
        "--cuda_reversal_mode", "parallel",
        "--cuda_candidates_per_iter", str(args.candidates_per_iter),
        "--chains", str(args.chains),
        "--cuda_block_size", str(args.block_size),
        "--iterations", str(args.iterations),
        "--seed", "1",
        "--init", "nn",
        "--csv-only",
    ]

    lines: list[str] = [
        "# CUDA Nsight profiling analysis",
        "",
        f"- Instance: `{args.instance}`",
        f"- Iterations: `{args.iterations}`",
        f"- Candidate policy: `{args.policy}`",
        f"- Executable: `{exe}`",
        "",
    ]

    if not input_path.exists():
        lines.append(f"- Status: skipped, missing `{input_path}`.")
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[skip] missing {input_path}")
        return 0

    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    lines.append(f"- Nsight Systems: `{nsys or 'not found'}`")
    lines.append(f"- Nsight Compute: `{ncu or 'not found'}`")
    lines.append("")

    if nsys:
        nsys_out = out_dir / f"cuda_candidate_{args.instance}_{args.policy}_nsys"
        command = [
            nsys,
            "profile",
            "--stats=true",
            "--force-overwrite=true",
            "--output",
            str(nsys_out),
            *base_command,
        ]
        code, _ = run_command(command, out_dir / f"cuda_candidate_{args.instance}_{args.policy}_nsys.log")
        lines.append("## Nsight Systems")
        lines.append("")
        lines.append(f"- Command: `{' '.join(command)}`")
        lines.append(f"- Exit code: `{code}`")
        lines.append(f"- Output prefix: `{nsys_out}`")
        lines.append("")
    else:
        lines.append("## Nsight Systems")
        lines.append("")
        lines.append("- Tool not found in PATH; no Systems trace was captured.")
        lines.append("")

    if ncu:
        ncu_out = out_dir / f"cuda_candidate_{args.instance}_{args.policy}_ncu"
        command = [
            ncu,
            "--force-overwrite",
            "--set",
            "speedOfLight",
            "--target-processes",
            "all",
            "--export",
            str(ncu_out),
            *base_command,
        ]
        code, _ = run_command(command, out_dir / f"cuda_candidate_{args.instance}_{args.policy}_ncu.log")
        lines.append("## Nsight Compute")
        lines.append("")
        lines.append(f"- Command: `{' '.join(command)}`")
        lines.append(f"- Exit code: `{code}`")
        lines.append(f"- Output prefix: `{ncu_out}`")
        lines.append("")
    else:
        lines.append("## Nsight Compute")
        lines.append("")
        lines.append("- Tool not found in PATH; no Compute report was captured.")
        lines.append("")

    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append("- This file records profiler availability and generated reports.")
    lines.append("- If metrics are missing or the tool is unavailable, do not report occupancy, bandwidth, or CUDA advantage claims.")
    lines.append("- Runtime and quality conclusions should still come from CSV experiments.")

    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {markdown.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
