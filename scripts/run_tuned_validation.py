#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from scripts.experiment_csv import (
        CURRENT_HEADER,
        ExperimentCsvError,
        parse_program_rows,
        resolve_executable,
        row_for_output,
        validate_command_output,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` invocation.
    from experiment_csv import (  # type: ignore[no-redef]
        CURRENT_HEADER,
        ExperimentCsvError,
        parse_program_rows,
        resolve_executable,
        row_for_output,
        validate_command_output,
    )

BASE_FIELDS = CURRENT_HEADER

META_FIELDS = [
    "config_name",
    "family",
    "variant",
    "t0",
    "tf",
    "alpha",
    "gamma",
    "epsilon",
    "policy",
    "fallback",
]

CSV_FIELDS = BASE_FIELDS + META_FIELDS
ALGORITHMS = {
    "sa",
    "qlsa",
    "sa-multichain",
    "qlsa-multichain",
    "sa-omp",
    "qlsa-omp",
    "sa-cuda",
    "qlsa-cuda",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Step 6B independent validation for tuned parameters.")
    parser.add_argument("--config", default="configs/tuned_params.json")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", default="results/tuned_validation_raw.csv")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--executable", type=Path, help="Explicit tsp_sa executable path.")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_executable(root, explicit=None):
    candidates = [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]
    return resolve_executable(explicit, candidates, root=root, description="tsp_sa executable")


def safe_name(text):
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")[:180]


def load_configs(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs", [])
    if not isinstance(configs, list) or not configs:
        raise SystemExit(f"No configs found in {path}")
    return configs


def quick_filter(configs):
    selected = []
    for config in configs:
        if config.get("instance") == "eil76" and config.get("family") in ("sa", "qlsa"):
            selected.append(config)
    return selected


def extract_csv_rows(stdout, family=None):
    if family == "sa":
        predicate = lambda algorithm: algorithm.startswith("sa")
    elif family == "qlsa":
        predicate = lambda algorithm: algorithm.startswith("qlsa")
    else:
        predicate = lambda algorithm: algorithm.startswith(("sa", "qlsa"))
    return parse_program_rows(stdout, algorithm_predicate=predicate)


def write_log(log_dir, label, command, completed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{timestamp}_{safe_name(label)}.log"
    log_path.write_text(
        "\n".join(
            [
                f"command: {' '.join(command)}",
                f"returncode: {completed.returncode}",
                "",
                "===== STDOUT =====",
                completed.stdout,
                "",
                "===== STDERR =====",
                completed.stderr,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def metadata(config, fallback):
    params = config["parameters"]
    family = config["family"]
    return {
        "config_name": config["name"],
        "family": family,
        "variant": config["variant"],
        "t0": str(params["t0"]),
        "tf": str(params["tf"]),
        "alpha": "" if family == "sa" else str(params["alpha"]),
        "gamma": "" if family == "sa" else str(params["gamma"]),
        "epsilon": "" if family == "sa" else str(params["epsilon"]),
        "policy": "" if family == "sa" else params["policy"],
        "fallback": "1" if fallback else "0",
    }


def command_for(exe, input_path, config, repeat, seed):
    params = config["parameters"]
    command = [
        str(exe),
        "--input",
        str(input_path),
        "--iterations",
        str(params["iterations"]),
        "--seed",
        str(seed),
        "--init",
        params.get("init", "nn"),
        "--repeat",
        str(repeat),
        "--csv-only",
        "--parallel",
        params.get("parallel", "omp"),
        "--chains",
        str(params["chains"]),
        "--threads",
        str(params["threads"]),
        "--t0",
        str(params["t0"]),
        "--tf",
        str(params["tf"]),
    ]
    if config["family"] == "qlsa":
        command.extend(
            [
                "--qlsa",
                "--alpha",
                str(params["alpha"]),
                "--gamma",
                str(params["gamma"]),
                "--epsilon",
                str(params["epsilon"]),
                "--policy",
                params["policy"],
            ]
        )
    return command


def run_command(command, log_dir, config):
    label = config["name"]
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path = write_log(log_dir, label, command, completed)
    combined = f"{completed.stdout}\n{completed.stderr}"
    fallback = "falling back to serial multi-chain execution" in combined
    if fallback:
        print(f"!!! WARNING: fallback detected for {label}; see {log_path}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}; see {log_path}")
    rows = extract_csv_rows(completed.stdout, config["family"])
    try:
        validate_command_output(rows, command, source=label)
    except ExperimentCsvError as exc:
        raise SystemExit(f"{exc}; see {log_path}") from exc
    meta = metadata(config, False)
    for row in rows:
        row.update(meta)
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_output(row, CSV_FIELDS))


def main():
    args = parse_args()
    if args.quick:
        args.repeat = 1
        args.seed = 101

    root = Path.cwd()
    exe = find_executable(root, args.executable)
    configs = load_configs(Path(args.config))
    if args.quick:
        configs = quick_filter(configs)
    if not configs:
        raise SystemExit("No tuned validation configs selected.")

    input_dir = Path(args.input_dir)
    log_dir = Path("results") / "logs" / "tuned_validation"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in configs:
        input_path = input_dir / f"{config['instance']}.tsp"
        if not input_path.exists():
            raise FileNotFoundError(f"missing requested instance: {input_path}")
        command = command_for(exe, input_path, config, args.repeat, args.seed)
        rows.extend(run_command(command, log_dir, config))

    output_path = Path(args.output)
    if not rows:
        raise ExperimentCsvError("no experiment rows were produced")
    write_rows(output_path, rows)
    print(f"Wrote tuned validation raw CSV: {output_path}", flush=True)


if __name__ == "__main__":
    main()
