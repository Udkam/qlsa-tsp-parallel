#!/usr/bin/env python3
"""Run real MPI VM scaling experiments from VM1.

This script is intentionally strict: it requires mpirun/mpiexec, a hostfile, and
the MPI executable. It never falls back to OpenMP because VM scaling data must be
real MPI evidence.
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, NamedTuple, Optional, Tuple

try:
    from scripts.mpi_csv import (
        MpiExecutionContract,
        parse_mpi_program_rows,
        validate_mpi_execution_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from mpi_csv import (  # type: ignore[no-redef]
        MpiExecutionContract,
        parse_mpi_program_rows,
        validate_mpi_execution_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
BKS = {"berlin52": 7542, "eil51": 426, "st70": 675, "eil76": 538, "rat99": 1211, "eil101": 629}
RAW_HEADER = [
    "command_id",
    "algorithm",
    "instance",
    "dimension",
    "iterations",
    "seed",
    "init",
    "chains",
    "threads",
    "parallel",
    "best_length",
    "final_length",
    "elapsed_ms",
    "accepted_moves",
    "improved_moves",
    "mpi_ranks",
    "communication_ms",
    "actual_threads",
    "iterations_completed",
    "deadline_reached",
    "hostfile",
]
SUMMARY_HEADER = [
    "algorithm",
    "instance",
    "np",
    "threads",
    "chains",
    "runs",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "speedup_vs_np1",
    "efficiency",
    "communication_ms_mean",
    "best_length_min",
    "gap_percent",
]


class Experiment(NamedTuple):
    algorithm: str
    instance: str
    np: int
    threads: int
    chains: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real MPI VM scaling experiments.")
    parser.add_argument("--hostfile", type=Path, default=ROOT / "mpi_hosts.local")
    parser.add_argument("--executable", type=Path, default=ROOT / "build-mpi" / "tsp_sa_mpi")
    parser.add_argument("--instance", default="berlin52")
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--np", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--threads", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--chains", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--algorithm", choices=["sa", "qlsa", "both"], default="both")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "raw" / "mpi_vm_scaling_raw.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "summary" / "mpi_vm_scaling_summary.csv")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures" / "fig12_mpi_vm_scaling.png")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_launcher() -> Optional[str]:
    for name in ("mpirun", "mpiexec"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None


def mpi_runtime_args() -> List[str]:
    """Return VM-specific Open MPI launch arguments when the local prefix exists."""
    prefix = Path("/home/clj/ompi-4.1.2")
    if not prefix.exists():
        return []
    return ["--prefix", str(prefix), "--mca", "pmix_base_compress", "0"]


def instance_path(name: str) -> Path:
    if name == "square4":
        return ROOT / "tests" / "fixtures" / "square4.tsp"
    return ROOT / "data" / f"{name}.tsp"


def csv_rows(stdout: str, algorithm: str | None = None) -> List[Dict[str, str]]:
    return parse_mpi_program_rows(stdout, algorithm=algorithm)


def build_command(launcher: str, args: argparse.Namespace, exp: Experiment) -> List[str]:
    cmd = [
        launcher,
        *mpi_runtime_args(),
        "-np",
        str(exp.np),
        "--hostfile",
        str(args.hostfile),
        "--map-by",
        "node",
        str(args.executable),
        "--input",
        str(instance_path(exp.instance)),
        "--parallel",
        "mpi-omp",
        "--chains",
        str(exp.chains),
        "--threads",
        str(exp.threads),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--init",
        "nn",
        "--repeat",
        str(args.repeat),
        "--csv-only",
    ]
    if exp.algorithm == "qlsa":
        cmd.extend(
            [
                "--qlsa",
                "--alpha",
                "0.1",
                "--gamma",
                "0.9",
                "--epsilon",
                "0.1",
                "--policy",
                "epsilon-greedy",
            ]
        )
    return cmd


def run_command(command: List[str], log_path: Path, args: argparse.Namespace, exp: Experiment) -> List[Dict[str, str]]:
    print("Running:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n"
        + "[stdout]\n"
        + completed.stdout
        + "\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}; see {log_path}")
    rows = csv_rows(completed.stdout, exp.algorithm)
    validate_mpi_execution_contract(
        rows,
        MpiExecutionContract(
            algorithm=exp.algorithm,
            iterations=args.iterations,
            chains=exp.chains,
            threads=exp.threads,
            ranks=exp.np,
            repeat=args.repeat,
            seed=args.seed,
        ),
        source=f"{exp.instance} {exp.algorithm} np={exp.np}",
    )
    return rows


def normalize(command_id: int, raw: Mapping[str, str], hostfile: Path) -> Dict[str, str]:
    return {
        "command_id": str(command_id),
        "algorithm": raw["algorithm"],
        "instance": raw["instance"],
        "dimension": raw["dimension"],
        "iterations": raw["iterations"],
        "seed": raw["seed"],
        "init": raw["init"],
        "chains": raw["chains"],
        "threads": raw["threads"],
        "parallel": raw["parallel"],
        "best_length": raw["best_length"],
        "final_length": raw["final_length"],
        "elapsed_ms": raw["elapsed_ms"],
        "accepted_moves": raw["accepted_moves"],
        "improved_moves": raw["improved_moves"],
        "mpi_ranks": raw["mpi_ranks"],
        "communication_ms": raw["communication_ms"],
        "actual_threads": raw["actual_threads"],
        "iterations_completed": raw["iterations_completed"],
        "deadline_reached": raw["deadline_reached"],
        "hostfile": str(hostfile),
    }


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def stddev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups = defaultdict(list)  # type: DefaultDict[Tuple[str, str, int, int, int], List[Dict[str, str]]]
    for row in rows:
        key = (
            row["algorithm"],
            row["instance"],
            int(row["mpi_ranks"]),
            int(row["threads"]),
            int(row["chains"]),
        )
        groups[key].append(row)

    base_elapsed = {}  # type: Dict[Tuple[str, str, int, int], float]
    for (algorithm, instance, np_value, threads, chains), items in groups.items():
        if np_value == 1:
            base_elapsed[(algorithm, instance, threads, chains)] = mean([float(i["elapsed_ms"]) for i in items])

    summary = []  # type: List[Dict[str, str]]
    for (algorithm, instance, np_value, threads, chains), items in sorted(groups.items()):
        elapsed = [float(i["elapsed_ms"]) for i in items]
        comm = [float(i["communication_ms"]) for i in items]
        best_min = min(int(i["best_length"]) for i in items)
        bks = BKS.get(instance)
        gap = "" if bks is None else f"{((best_min - bks) / bks * 100.0):.4f}"
        baseline = base_elapsed.get((algorithm, instance, threads, chains))
        speedup = "" if baseline is None else f"{(baseline / mean(elapsed)):.4f}"
        efficiency = "" if baseline is None or np_value <= 0 else f"{(baseline / mean(elapsed) / np_value):.4f}"
        summary.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "np": str(np_value),
                "threads": str(threads),
                "chains": str(chains),
                "runs": str(len(items)),
                "elapsed_ms_mean": f"{mean(elapsed):.3f}",
                "elapsed_ms_std": f"{stddev(elapsed):.3f}",
                "speedup_vs_np1": speedup,
                "efficiency": efficiency,
                "communication_ms_mean": f"{mean(comm):.3f}",
                "best_length_min": str(best_min),
                "gap_percent": gap,
            }
        )
    return summary


def write_summary(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(summary: List[Dict[str, str]], args: argparse.Namespace) -> None:
    path = ROOT / "docs" / "dev" / "mpi_vm_scaling_analysis.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MPI VM Scaling Analysis",
        "",
        "This document is generated from real `mpirun`/`mpiexec` rows. It should not be used if the raw CSV is absent.",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Hostfile: `{args.hostfile}`",
        f"- Instance: `{args.instance}`",
        f"- Iterations: {args.iterations}",
        f"- Repeat: {args.repeat}",
        "",
        "| algorithm | instance | np | threads | chains | runs | elapsed_ms_mean | speedup_vs_np1 | efficiency | comm_ms_mean | best | gap_percent |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {algorithm} | {instance} | {np} | {threads} | {chains} | {runs} | "
            "{elapsed_ms_mean} | {speedup_vs_np1} | {efficiency} | "
            "{communication_ms_mean} | {best_length_min} | {gap_percent} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- `np=1` is the MPI single-node baseline for the same executable and hostfile environment.",
            "- `np=2` is the dual-VM distributed-memory run.",
            "- Efficiency is `speedup_vs_np1 / np`.",
            "- VMware NAT and virtualization overhead mean this is an engineering evidence run, not a production HPC benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(summary: List[Dict[str, str]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.font_manager as fm  # type: ignore
    except Exception:
        path.with_suffix(".svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='360'>"
            "<rect width='100%' height='100%' fill='white'/>"
            "<text x='30' y='50' font-size='18'>MPI VM scaling figure skipped: matplotlib unavailable.</text>"
            "</svg>\n",
            encoding="utf-8",
        )
        return

    filtered = [r for r in summary if r["speedup_vs_np1"]]
    if not filtered:
        return
    labels = [
        f"{'SA' if r['algorithm'].startswith('sa') else 'QLSA'}\n"
        f"进程={r['np']} 线程={r['threads']} 链={r['chains']}"
        for r in filtered
    ]
    values = [float(r["speedup_vs_np1"]) for r in filtered]
    colors = ["#1f77b4" if r["algorithm"].startswith("sa") else "#ff7f0e" for r in filtered]

    path.parent.mkdir(parents=True, exist_ok=True)
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(2.0, color="#1f77b4", linestyle="--", linewidth=1.0, label="理想 np=2")
    ax.set_ylabel("相对 np=1 的加速比")
    ax.set_title("双 VM MPI + OpenMP 扩展性")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", color="#d6e8ff", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.quick:
        args.instance = "berlin52"
        args.iterations = 100000
        args.repeat = 1
        args.np = [1, 2]
        args.threads = [2]
        args.chains = [16]
        args.algorithm = "sa"

    launcher = find_launcher()
    if launcher is None:
        print("Error: mpirun/mpiexec not found", file=sys.stderr)
        return 1
    if not args.hostfile.exists():
        print(f"Error: hostfile not found: {args.hostfile}", file=sys.stderr)
        return 1
    if not args.executable.exists():
        print(f"Error: tsp_sa_mpi not found: {args.executable}", file=sys.stderr)
        return 1
    if not instance_path(args.instance).exists():
        print(f"Error: instance file not found: {instance_path(args.instance)}", file=sys.stderr)
        return 1

    algorithms = ["sa", "qlsa"] if args.algorithm == "both" else [args.algorithm]
    experiments = [
        Experiment(algorithm, args.instance, np_value, threads, chains)
        for algorithm in algorithms
        for np_value in args.np
        for threads in args.threads
        for chains in args.chains
    ]

    logs_dir = ROOT / "results" / "logs" / "mpi_vm_scaling"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []  # type: List[Dict[str, str]]
    for command_id, exp in enumerate(experiments, start=1):
        log_path = logs_dir / (
            f"{command_id:03d}_{exp.algorithm}_{exp.instance}_np{exp.np}_t{exp.threads}_c{exp.chains}.log"
        )
        for row in run_command(build_command(launcher, args, exp), log_path, args, exp):
            raw_rows.append(normalize(command_id, row, args.hostfile))

    if not raw_rows:
        raise RuntimeError("no MPI experiment rows were produced")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_HEADER)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary = summarize(raw_rows)
    write_summary(args.summary, summary)
    write_analysis(summary, args)
    write_figure(summary, args.figure)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    print("Wrote docs/dev/mpi_vm_scaling_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
