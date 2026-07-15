#!/usr/bin/env python3
"""Run a small MPI+OpenMP smoke experiment with OpenMP fallback.

The script never fabricates MPI results. If tsp_sa_mpi or an MPI launcher is not
available, it runs the same cases through the existing OpenMP backend and marks
the rows as fallback.
"""

import csv
import argparse
import io
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

try:
    from scripts.experiment_csv import parse_program_rows, validate_command_output
    from scripts.mpi_csv import (
        MPI_SCHEMA_WIDTH,
        MpiExecutionContract,
        parse_mpi_program_rows,
        validate_mpi_execution_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from experiment_csv import parse_program_rows, validate_command_output  # type: ignore[no-redef]
    from mpi_csv import (  # type: ignore[no-redef]
        MPI_SCHEMA_WIDTH,
        MpiExecutionContract,
        parse_mpi_program_rows,
        validate_mpi_execution_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
TSP_CLI_SCHEMA_WIDTHS = {14, 22, 27, 28}
CSV_HEADER = [
    "mode",
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
    "fallback",
    "speedup_vs_openmp",
    "scaling_efficiency",
]


class Case(NamedTuple):
    instance_name: str
    input_path: Path
    iterations: int
    chains: int
    threads: int
    seed: int
    qlsa: bool


def find_executable(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_tsp_sa() -> Optional[Path]:
    return find_executable(
        [
            ROOT / "build" / "ninja-mpi-release" / "tsp_sa.exe",
            ROOT / "build" / "ninja-mpi-release" / "tsp_sa",
            ROOT / "build" / "ninja-cpu-release" / "tsp_sa.exe",
            ROOT / "build" / "ninja-cpu-release" / "tsp_sa",
            ROOT / "build" / "ninja-cuda-release" / "tsp_sa.exe",
            ROOT / "build" / "ninja-cuda-release" / "tsp_sa",
            ROOT / "build-cuda-ninja" / "tsp_sa.exe",
            ROOT / "build-cuda-ninja" / "tsp_sa",
            ROOT / "build-cuda-real" / "Release" / "tsp_sa.exe",
            ROOT / "build" / "Release" / "tsp_sa.exe",
            ROOT / "build" / "tsp_sa",
        ]
    )


def find_tsp_sa_mpi() -> Optional[Path]:
    return find_executable(
        [
            ROOT / "build" / "ninja-mpi-release" / "tsp_sa_mpi.exe",
            ROOT / "build" / "ninja-mpi-release" / "tsp_sa_mpi",
            ROOT / "build-cuda-ninja" / "tsp_sa_mpi.exe",
            ROOT / "build-cuda-ninja" / "tsp_sa_mpi",
            ROOT / "build-mpi" / "tsp_sa_mpi.exe",
            ROOT / "build-mpi" / "tsp_sa_mpi",
            ROOT / "build" / "Release" / "tsp_sa_mpi.exe",
            ROOT / "build" / "tsp_sa_mpi",
        ]
    )


def resolve_mpi_executable(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is None:
        return find_tsp_sa_mpi()
    path = explicit if explicit.is_absolute() else ROOT / explicit
    if not path.is_file():
        raise FileNotFoundError(f"explicit tsp_sa_mpi executable not found: {path}")
    return path.resolve()


def find_mpi_launcher() -> Optional[str]:
    for name in ("mpiexec", "mpirun"):
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


def data_lines(stdout: str) -> List[List[str]]:
    rows = []  # type: List[List[str]]
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("CSV:"):
            continue
        if "," not in stripped:
            continue
        first = stripped.split(",", 1)[0]
        if first.startswith("sa") or first.startswith("qlsa"):
            rows.append(next(csv.reader([stripped])))
    if not rows:
        raise RuntimeError("no CSV data rows found in command stdout")
    return rows


def run_command(command: List[str], log_path: Path) -> List[str]:
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
    rows = data_lines(completed.stdout)
    if len(rows) != 1:
        raise RuntimeError(f"expected one CSV data row, got {len(rows)}; see {log_path}")
    return rows[0]


def run_command_rows(command: List[str], log_path: Path) -> List[List[str]]:
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
    return data_lines(completed.stdout)


def openmp_command(exe: Path, case: Case) -> List[str]:
    cmd = [
        str(exe),
        "--input",
        str(case.input_path),
        "--parallel",
        "omp",
        "--chains",
        str(case.chains),
        "--threads",
        str(case.threads),
        "--iterations",
        str(case.iterations),
        "--seed",
        str(case.seed),
        "--init",
        "nn",
        "--repeat",
        "1",
        "--csv-only",
    ]
    if case.qlsa:
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


def mpi_command(launcher: str, exe: Path, case: Case, ranks: int) -> List[str]:
    cmd = [
        launcher,
        "-n",
        str(ranks),
        str(exe),
        "--input",
        str(case.input_path),
        "--parallel",
        "mpi-omp",
        "--chains",
        str(case.chains),
        "--threads",
        str(case.threads),
        "--iterations",
        str(case.iterations),
        "--seed",
        str(case.seed),
        "--init",
        "nn",
        "--repeat",
        "1",
        "--csv-only",
    ]
    if case.qlsa:
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


def mpi_hostfile_command(launcher: str,
                         exe: Path,
                         case: Case,
                         ranks: int,
                         hostfile: Path,
                         repeat: int) -> List[str]:
    cmd = [
        launcher,
        *mpi_runtime_args(),
        "-np",
        str(ranks),
        "--hostfile",
        str(hostfile),
        "--map-by",
        "node",
        str(exe),
        "--input",
        str(case.input_path),
        "--parallel",
        "mpi-omp",
        "--chains",
        str(case.chains),
        "--threads",
        str(case.threads),
        "--iterations",
        str(case.iterations),
        "--seed",
        str(case.seed),
        "--init",
        "nn",
        "--repeat",
        str(repeat),
        "--csv-only",
    ]
    if case.qlsa:
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


def mpi_extension(raw: List[str]) -> tuple[int, float]:
    if len(raw) != MPI_SCHEMA_WIDTH or raw[8] != "mpi-omp":
        raise RuntimeError(
            f"expected a {MPI_SCHEMA_WIDTH}-column mpi-omp CSV row, got {raw}"
        )
    if raw[0] not in {"sa-mpi-omp", "qlsa-mpi-omp"}:
        raise RuntimeError(f"unexpected MPI algorithm label: {raw[0]!r}")
    return int(raw[14]), float(raw[15])


def csv_text(rows: List[List[str]]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue()


def validate_openmp_raw(raw: List[str], command: List[str], source: str) -> None:
    rows = parse_program_rows(csv_text([raw]))
    validate_command_output(rows, command, source=source)


def validate_mpi_raw_rows(
    raw_rows: List[List[str]],
    case: Case,
    ranks: int,
    repeat: int,
    source: str,
) -> None:
    rows = parse_mpi_program_rows(
        csv_text(raw_rows), algorithm="qlsa" if case.qlsa else "sa"
    )
    validate_mpi_execution_contract(
        rows,
        MpiExecutionContract(
            algorithm="qlsa" if case.qlsa else "sa",
            iterations=case.iterations,
            chains=case.chains,
            threads=case.threads,
            ranks=ranks,
            repeat=repeat,
            seed=case.seed,
            init="nn",
        ),
        source=source,
    )


def normalize_row(
    raw: List[str],
    mode: str,
    fallback: bool,
    mpi_ranks: int,
    communication_ms: float,
    speedup: Optional[float],
    efficiency: Optional[float],
) -> Dict[str, str]:
    if len(raw) < 14:
        raise RuntimeError(f"unexpected CSV row with {len(raw)} columns: {raw}")
    ranks = mpi_ranks
    comm = communication_ms
    if raw[8] == "mpi-omp":
        ranks, comm = mpi_extension(raw)
        actual_threads = raw[16]
        iterations_completed = raw[17]
        deadline_reached = raw[18]
    elif len(raw) not in TSP_CLI_SCHEMA_WIDTHS:
        raise RuntimeError(
            f"non-MPI tsp_sa CSV row must have one of {sorted(TSP_CLI_SCHEMA_WIDTHS)} columns, "
            f"got {len(raw)}: {raw}"
        )
    else:
        actual_threads = raw[27] if len(raw) == 28 else ""
        iterations_completed = raw[20] if len(raw) >= 22 else ""
        deadline_reached = raw[21] if len(raw) >= 22 else ""
    return {
        "mode": mode,
        "algorithm": raw[0],
        "instance": raw[1],
        "dimension": raw[2],
        "iterations": raw[3],
        "seed": raw[4],
        "init": raw[5],
        "chains": raw[6],
        "threads": raw[7],
        "parallel": raw[8],
        "best_length": raw[9],
        "final_length": raw[10],
        "elapsed_ms": raw[11],
        "accepted_moves": raw[12],
        "improved_moves": raw[13],
        "mpi_ranks": str(ranks),
        "communication_ms": f"{comm:.3f}",
        "actual_threads": actual_threads,
        "iterations_completed": iterations_completed,
        "deadline_reached": deadline_reached,
        "fallback": "true" if fallback else "false",
        "speedup_vs_openmp": "" if speedup is None else f"{speedup:.4f}",
        "scaling_efficiency": "" if efficiency is None else f"{efficiency:.4f}",
    }


def write_analysis(rows: List[Dict[str, str]], mpi_available: bool, mpi_reason: str) -> None:
    docs_dir = ROOT / "docs" / "dev"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "mpi_analysis.md"

    lines = [
        "# MPI + OpenMP Smoke Analysis",
        "",
        "This smoke test validates the optional rank-level MPI backend when available. "
        "If MPI is unavailable, the script falls back to the existing OpenMP backend and marks the rows explicitly.",
        "",
        f"- MPI available for this run: {'yes' if mpi_available else 'no'}",
        f"- MPI status: {mpi_reason}",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Results",
        "",
        "| mode | algorithm | instance | chains | threads | ranks | best_length | elapsed_ms | comm_ms | speedup_vs_openmp | efficiency | fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {algorithm} | {instance} | {chains} | {threads} | {mpi_ranks} | "
            "{best_length} | {elapsed_ms} | {communication_ms} | {speedup_vs_openmp} | "
            "{scaling_efficiency} | {fallback} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `speedup_vs_openmp` is computed only for actual MPI rows by comparing the MPI elapsed time with the matching OpenMP smoke baseline.",
            "- `scaling_efficiency` is `speedup_vs_openmp / mpi_ranks`; it is a smoke-level indicator, not a full scaling study.",
            "- Communication overhead is reported by the MPI backend around MINLOC reduction, best-tour broadcast, and move-count reductions.",
            "- If `fallback=true`, no MPI performance claim should be made from that row.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MPI smoke or local fallback smoke.")
    parser.add_argument("--hostfile", type=Path, default=None, help="MPI hostfile for real VM/cluster mode.")
    parser.add_argument("--np", type=int, default=2, help="MPI process count in hostfile mode.")
    parser.add_argument("--remote", action="store_true", help="Require real mpirun/tsp_sa_mpi; never fallback.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "mpi_smoke.csv")
    parser.add_argument("--instance", action="append", choices=["square4", "berlin52"], help="Instance to run; repeatable.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--executable", type=Path, default=None, help="Explicit path to tsp_sa_mpi.")
    parser.add_argument("--no-qlsa", action="store_true", help="Run SA only.")
    return parser.parse_args()


def case_for_instance(name: str, iterations: Optional[int], chains: Optional[int], threads: Optional[int], qlsa: bool) -> Case:
    if name == "square4":
        return Case(
            "square4",
            ROOT / "tests" / "fixtures" / "square4.tsp",
            iterations if iterations is not None else 10000,
            chains if chains is not None else 16,
            threads if threads is not None else 2,
            1,
            qlsa,
        )
    if name == "berlin52":
        return Case(
            "berlin52",
            ROOT / "data" / "berlin52.tsp",
            iterations if iterations is not None else 1000000,
            chains if chains is not None else 64,
            threads if threads is not None else 4,
            1,
            qlsa,
        )
    raise ValueError(f"unsupported instance: {name}")


def run_hostfile_mode(args: argparse.Namespace) -> int:
    if args.hostfile is None:
        print("Error: --hostfile is required in --remote mode", file=sys.stderr)
        return 1
    if not args.hostfile.exists():
        print(f"Error: hostfile not found: {args.hostfile}", file=sys.stderr)
        return 1
    launcher = find_mpi_launcher()
    if launcher is None:
        print("Error: mpirun/mpiexec not found; real MPI smoke cannot run", file=sys.stderr)
        return 1
    try:
        mpi_exe = resolve_mpi_executable(args.executable)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if mpi_exe is None:
        print("Error: tsp_sa_mpi not found in known build directories", file=sys.stderr)
        return 1

    logs_dir = ROOT / "results" / "logs" / "mpi_vm_smoke"
    logs_dir.mkdir(parents=True, exist_ok=True)

    requested_instances = args.instance or ["square4", "berlin52"]
    rows = []  # type: List[Dict[str, str]]
    idx = 0
    for instance_name in requested_instances:
        for qlsa in ([False] if args.no_qlsa else [False, True]):
            case = case_for_instance(instance_name, args.iterations, args.chains, args.threads, qlsa)
            if not case.input_path.exists():
                print(f"Warning: missing {case.input_path}; skipping {case.instance_name}")
                continue
            idx += 1
            suffix = "qlsa" if qlsa else "sa"
            log_path = logs_dir / f"{idx:02d}_{case.instance_name}_{suffix}_mpi_vm.log"
            raw_rows = run_command_rows(
                mpi_hostfile_command(launcher, mpi_exe, case, args.np, args.hostfile, args.repeat),
                log_path,
            )
            validate_mpi_raw_rows(
                raw_rows,
                case,
                args.np,
                args.repeat,
                f"{case.instance_name} hostfile smoke; see {log_path}",
            )
            for raw in raw_rows:
                ranks, communication_ms = mpi_extension(raw)
                rows.append(
                    normalize_row(
                        raw,
                        mode="mpi-vm-smoke",
                        fallback=False,
                        mpi_ranks=ranks,
                        communication_ms=communication_ms,
                        speedup=None,
                        efficiency=None,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    write_analysis(rows, mpi_available=True, mpi_reason=f"real hostfile mode with np={args.np}")
    print(f"Wrote {args.output}")
    print("Wrote docs/dev/mpi_analysis.md")
    return 0


def main() -> int:
    args = parse_args()
    if args.remote or args.hostfile is not None:
        return run_hostfile_mode(args)

    results_dir = ROOT / "results"
    logs_dir = results_dir / "logs" / "mpi_smoke"
    logs_dir.mkdir(parents=True, exist_ok=True)

    openmp_exe = find_tsp_sa()
    if openmp_exe is None:
        print("Error: tsp_sa executable not found. Build the project first.", file=sys.stderr)
        return 1

    try:
        mpi_exe = resolve_mpi_executable(args.executable)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    launcher = find_mpi_launcher()
    mpi_available = mpi_exe is not None and launcher is not None
    mpi_reason = "tsp_sa_mpi and MPI launcher found"
    if mpi_exe is None:
        mpi_reason = "tsp_sa_mpi executable not found; using OpenMP fallback"
    elif launcher is None:
        mpi_reason = "MPI launcher not found; using OpenMP fallback"

    candidates = [
        Case("square4", ROOT / "tests" / "fixtures" / "square4.tsp", 20000, 8, 2, 1, False),
        Case("square4", ROOT / "tests" / "fixtures" / "square4.tsp", 20000, 8, 2, 1, True),
        Case("berlin52", ROOT / "data" / "berlin52.tsp", 100000, 16, 2, 1, False),
        Case("berlin52", ROOT / "data" / "berlin52.tsp", 100000, 16, 2, 1, True),
    ]
    cases = [case for case in candidates if case.input_path.exists()]
    for case in candidates:
        if not case.input_path.exists():
            print(f"Warning: missing {case.input_path}; skipping {case.instance_name}")

    rows = []  # type: List[Dict[str, str]]
    for idx, case in enumerate(cases, start=1):
        algo_suffix = "qlsa" if case.qlsa else "sa"
        base_log = logs_dir / f"{idx:02d}_{case.instance_name}_{algo_suffix}_openmp.log"
        openmp_cmd = openmp_command(openmp_exe, case)
        openmp_raw = run_command(openmp_cmd, base_log)
        validate_openmp_raw(openmp_raw, openmp_cmd, f"{case.instance_name} OpenMP smoke")
        openmp_elapsed = float(openmp_raw[11])
        rows.append(
            normalize_row(
                openmp_raw,
                mode="openmp-baseline" if mpi_available else "openmp-fallback",
                fallback=not mpi_available,
                mpi_ranks=1,
                communication_ms=0.0,
                speedup=None,
                efficiency=None,
            )
        )

        if mpi_available:
            mpi_log = logs_dir / f"{idx:02d}_{case.instance_name}_{algo_suffix}_mpi.log"
            assert launcher is not None and mpi_exe is not None
            mpi_cmd = mpi_command(launcher, mpi_exe, case, ranks=2)
            mpi_raw = run_command(mpi_cmd, mpi_log)
            validate_mpi_raw_rows(
                [mpi_raw], case, 2, 1, f"{case.instance_name} local MPI smoke"
            )
            mpi_elapsed = float(mpi_raw[11])
            ranks, communication_ms = mpi_extension(mpi_raw)
            speedup = openmp_elapsed / mpi_elapsed if mpi_elapsed > 0.0 else None
            efficiency = (speedup / ranks) if speedup is not None and ranks > 0 else None
            rows.append(
                normalize_row(
                    mpi_raw,
                    mode="mpi",
                    fallback=False,
                    mpi_ranks=ranks,
                    communication_ms=communication_ms,
                    speedup=speedup,
                    efficiency=efficiency,
                )
            )

    out_csv = args.output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    write_analysis(rows, mpi_available=mpi_available, mpi_reason=mpi_reason)
    print(f"Wrote {out_csv}")
    print("Wrote docs/dev/mpi_analysis.md")
    if not mpi_available:
        print(f"Warning: {mpi_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
