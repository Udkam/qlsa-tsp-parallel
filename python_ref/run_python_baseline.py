#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a single Python faithful-baseline algorithm and emit C++-compatible CSV.

CSV columns match the C++ tool exactly:
algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,
best_length,final_length,elapsed_ms,accepted_moves,improved_moves
"""

from __future__ import annotations

import argparse
import sys

from tsplib_loader import load_instance
import sa_paper
import qlsa_paper
import sb_qlsa_paper

CSV_HEADER = (
    "algorithm,instance,dimension,iterations,seed,init,chains,threads,parallel,"
    "best_length,final_length,elapsed_ms,accepted_moves,improved_moves"
)

ALGORITHMS = {
    "sa": ("python-sa", lambda dist, it, seed, init: sa_paper.run_sa(dist, it, seed, init=init)),
    "qlsa-paper": ("python-qlsa-paper", lambda dist, it, seed, init: qlsa_paper.run_qlsa(dist, it, seed, init=init)),
    "sb-qlsa": ("python-sb-qlsa", lambda dist, it, seed, init: sb_qlsa_paper.run_sb_qlsa(dist, it, seed, init=init)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Python faithful TSP baseline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--init", default="nn", choices=["nn", "random"])
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    instance = load_instance(args.input)
    label, runner = ALGORITHMS[args.algorithm]

    if not args.csv_only:
        print(f"Instance: {instance.name} (n={instance.dimension})", file=sys.stderr)
        print(f"Algorithm: {label}", file=sys.stderr)
        print(CSV_HEADER)

    for r in range(args.repeat):
        seed = args.seed + r
        res = runner(instance.distances, args.iterations, seed, args.init)
        print(
            f"{label},{instance.name},{instance.dimension},{args.iterations},{seed},"
            f"{args.init},1,1,none,{res.best_length},{res.final_length},"
            f"{res.elapsed_ms:.3f},{res.accepted_moves},{res.improved_moves}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
