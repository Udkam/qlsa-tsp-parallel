#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run CUDA candidate serial-vs-parallel reversal experiments.

This is a convenience wrapper around run_cuda_candidate_experiments.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["ch130", "a280", "lin318", "rat575"])
    parser.add_argument("--algorithms", nargs="+", choices=["sa", "qlsa"], default=["sa"])
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--candidates-per-iter", nargs="+", type=int, default=[128])
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "cuda_reversal_raw.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_cuda_candidate_experiments.py"),
        "--instances",
        *args.instances,
        "--algorithms",
        *args.algorithms,
        "--iterations",
        str(args.iterations),
        "--repeat",
        str(args.repeat),
        "--chains",
        str(args.chains),
        "--block-sizes",
        *[str(x) for x in args.block_sizes],
        "--candidates-per-iter",
        *[str(x) for x in args.candidates_per_iter],
        "--reversal-modes",
        "serial",
        "parallel",
        "--output",
        args.output,
    ]
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
