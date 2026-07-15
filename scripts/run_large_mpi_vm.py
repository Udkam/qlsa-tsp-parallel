#!/usr/bin/env python3
"""Run large-instance MPI + OpenMP experiments from VM1.

This script is intended for the Ubuntu VM environment where `mpirun`, hostfile
and `build-mpi/tsp_sa_mpi` are available. It never falls back to OpenMP-only
execution.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

try:
    from scripts.mpi_csv import (
        MPI_HEADER,
        MpiCsvError,
        MpiExecutionContract,
        parse_mpi_program_rows,
        validate_mpi_execution_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from mpi_csv import (  # type: ignore[no-redef]
        MPI_HEADER,
        MpiCsvError,
        MpiExecutionContract,
        parse_mpi_program_rows,
        validate_mpi_execution_contract,
    )

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "large_tsplib_instances.json"
PROGRAM_HEADER = [
    field for field in MPI_HEADER if field not in {"mpi_ranks", "communication_ms"}
]
EXTRA_HEADER = ["tier", "np", "threads_per_rank", "status", "log_file", "communication_ms", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", default=["ch130", "d198", "a280"])
    parser.add_argument("--algorithm", choices=["sa", "qlsa", "both"], default="both")
    parser.add_argument("--np", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--threads", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--chains", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--iterations", type=int, default=300_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default=str(ROOT / "data"))
    parser.add_argument("--hostfile", required=True)
    parser.add_argument("--exe", default=str(ROOT / "build-mpi" / "tsp_sa_mpi"))
    parser.add_argument("--mpi-prefix", default=str(Path.home() / "ompi-4.1.2"),
                        help="Open MPI prefix on the VM; used when the directory exists.")
    parser.add_argument("--disable-pmix-compress", action="store_true", default=True,
                        help="Add '--mca pmix_base_compress 0' to avoid mixed PMIx compression support across VMs.")
    parser.add_argument("--allow-oversubscribe", action="store_true",
                        help="Pass --oversubscribe to mpirun for rank/thread grid experiments.")
    parser.add_argument("--map-by", default="node",
                        help="Open MPI mapping policy; default node spreads ranks across VM1/VM2.")
    parser.add_argument("--output", default=str(ROOT / "results" / "raw" / "large_mpi_vm_raw.csv"))
    parser.add_argument("--summary", default="",
                        help="Optional summary CSV path; invokes analyze_large_mpi_vm.py after raw output is written.")
    parser.add_argument("--timeout", type=int, default=0)
    return parser.parse_args()


def load_tier_map() -> dict[str, str]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {instance: tier for tier, instances in config["tiers"].items() for instance in instances}


def parse_rows(stdout: str, algorithm: str | None = None) -> list[dict[str, str]]:
    return parse_mpi_program_rows(stdout, algorithm=algorithm)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def command_for(args: argparse.Namespace, input_path: Path, algorithm: str, np_value: int, threads: int, chains: int) -> list[str]:
    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = ROOT / exe
    hostfile = Path(args.hostfile)
    command = ["mpirun"]
    mpi_prefix = Path(args.mpi_prefix).expanduser() if args.mpi_prefix else None
    if mpi_prefix and mpi_prefix.exists():
        command.extend(["--prefix", str(mpi_prefix)])
    if args.disable_pmix_compress:
        command.extend(["--mca", "pmix_base_compress", "0"])
    if args.allow_oversubscribe:
        command.append("--oversubscribe")
    if args.map_by:
        command.extend(["--map-by", args.map_by])
    command.extend([
        "-np", str(np_value), "--hostfile", str(hostfile), str(exe),
        "--input", str(input_path), "--parallel", "mpi-omp",
        "--chains", str(chains), "--threads", str(threads),
        "--iterations", str(args.iterations), "--repeat", str(args.repeat),
        "--seed", str(args.seed), "--init", "nn", "--csv-only",
    ])
    if algorithm == "qlsa":
        command.extend(["--qlsa", "--alpha", "0.1", "--gamma", "0.9", "--epsilon", "0.1", "--policy", "epsilon-greedy"])
    return command


def run_one(args: argparse.Namespace, input_path: Path, tier: str, algorithm: str, np_value: int, threads: int, chains: int, log_dir: Path) -> list[dict[str, str]]:
    command = command_for(args, input_path, algorithm, np_value, threads, chains)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"{input_path.stem}_{algorithm}_np{np_value}_t{threads}_c{chains}_{timestamp}.log"
    print("[run]", " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=args.timeout if args.timeout > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        log_file.write_text("$ " + " ".join(command) + "\n\n[timeout]\n" + str(exc), encoding="utf-8")
        raise MpiCsvError(
            f"{input_path.name} {algorithm}: command timed out; see {log_file.relative_to(ROOT)}"
        ) from exc
    log_file.write_text(
        "$ " + " ".join(command) + f"\n\nreturncode={completed.returncode}\n\n[stdout]\n"
        + completed.stdout + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise MpiCsvError(
            f"{input_path.name} {algorithm}: command failed with return code {completed.returncode}; "
            f"see {log_file.relative_to(ROOT)}"
        )
    rows = parse_rows(completed.stdout, algorithm)
    try:
        validate_mpi_execution_contract(
            rows,
            MpiExecutionContract(
                algorithm=algorithm,
                iterations=args.iterations,
                chains=chains,
                threads=threads,
                ranks=np_value,
                repeat=args.repeat,
                seed=args.seed,
            ),
            source=f"{input_path.name} {algorithm}",
        )
    except MpiCsvError as exc:
        raise MpiCsvError(f"{exc}; see {log_file.relative_to(ROOT)}") from exc
    for row in rows:
        row["tier"] = tier
        row["np"] = str(np_value)
        row["threads_per_rank"] = str(threads)
        row["status"] = "ok"
        row["log_file"] = str(log_file.relative_to(ROOT))
        row["communication_ms"] = row.pop("communication_ms")
        row.pop("mpi_ranks")
        row["error"] = ""
    return rows


def main() -> int:
    args = parse_args()
    tmap = load_tier_map()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results" / "logs" / "large_mpi_vm"
    log_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["sa", "qlsa"] if args.algorithm == "both" else [args.algorithm]
    rows: list[dict[str, str]] = []
    for instance in args.instances:
        tier = tmap.get(instance, "custom")
        input_path = input_dir / f"{instance}.tsp"
        if not input_path.exists():
            raise FileNotFoundError(f"missing requested instance: {display_path(input_path)}")
        for algorithm in algorithms:
            for np_value in args.np:
                for threads in args.threads:
                    for chains in args.chains:
                        rows.extend(run_one(args, input_path, tier, algorithm, np_value, threads, chains, log_dir))

    if not rows:
        raise MpiCsvError("no MPI experiment rows were produced")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRAM_HEADER + EXTRA_HEADER)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in PROGRAM_HEADER + EXTRA_HEADER} for row in rows)
    print(f"[ok] wrote {len(rows)} rows to {output.relative_to(ROOT)}")
    if args.summary:
        summary = Path(args.summary)
        if not summary.is_absolute():
            summary = ROOT / summary
        analyze = ROOT / "scripts" / "analyze_large_mpi_vm.py"
        cmd = [__import__("sys").executable, str(analyze), "--input", str(output), "--output", str(summary)]
        print("[run]", " ".join(cmd))
        completed = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
