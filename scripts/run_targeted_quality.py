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
    parser = argparse.ArgumentParser(description="Run Step 6C targeted high-budget quality experiments.")
    parser.add_argument("--config", default="configs/targeted_quality_configs.json")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--output", default="results/targeted_quality_raw.csv")
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


def as_list(value):
    return value if isinstance(value, list) else [value]


def load_configs(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs", [])
    if not isinstance(configs, list) or not configs:
        raise SystemExit(f"No configs found in {path}")
    return configs


def quick_configs(configs):
    selected = []
    for config in configs:
        if config.get("instance") == "eil101" and config.get("family") == "sa":
            item = dict(config)
            params = dict(config["parameters"])
            params["iterations"] = [100000]
            params["chains"] = [16]
            item["name"] = "eil101-sa-quick"
            item["variant"] = "quick"
            item["parameters"] = params
            selected.append(item)
            break
    return selected


def expand_configs(configs):
    expanded = []
    for config in configs:
        params = config["parameters"]
        for iterations in as_list(params["iterations"]):
            for chains in as_list(params["chains"]):
                item = dict(config)
                expanded_params = dict(params)
                expanded_params["iterations"] = iterations
                expanded_params["chains"] = chains
                item["parameters"] = expanded_params
                item["run_name"] = f"{config['name']}-it{iterations}-c{chains}"
                expanded.append(item)
    return expanded


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
        "config_name": config.get("run_name", config["name"]),
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
        "omp",
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
    label = config.get("run_name", config["name"])
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
        args.seed = 201

    root = Path.cwd()
    exe = find_executable(root, args.executable)
    configs = load_configs(Path(args.config))
    if args.quick:
        configs = quick_configs(configs)
    configs = expand_configs(configs)
    if not configs:
        raise SystemExit("No targeted quality configs selected.")

    input_dir = Path(args.input_dir)
    log_dir = Path("results") / "logs" / "targeted_quality"
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
    print(f"Wrote targeted quality raw CSV: {output_path}", flush=True)


if __name__ == "__main__":
    main()
