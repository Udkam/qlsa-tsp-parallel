#!/usr/bin/env python3
import argparse
import csv
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


BASE_FIELDS = [
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
]

PARAM_FIELDS = [
    "t0",
    "tf",
    "alpha",
    "gamma",
    "epsilon",
    "policy",
    "tuning_stage",
    "fallback",
]

CSV_FIELDS = BASE_FIELDS + PARAM_FIELDS
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
    parser = argparse.ArgumentParser(description="Run Step 6A SA/QLSA parameter tuning.")
    parser.add_argument("--instances", nargs="+", default=["eil76", "rat99", "eil101"])
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output", default="results/tuning_raw.csv")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--algorithm", choices=["sa", "qlsa", "both"], default="both")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    return parser.parse_args()


def find_executable(root):
    candidates = [
        root / "build-cuda-ninja" / "tsp_sa.exe",
        root / "build-cuda-ninja" / "tsp_sa",
        root / "build" / "Release" / "tsp_sa.exe",
        root / "build" / "tsp_sa",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    lines = ["Could not find tsp_sa executable. Tried:"]
    lines.extend(f"  - {candidate}" for candidate in candidates)
    raise SystemExit("\n".join(lines))


def safe_name(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:180]


def extract_csv_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(parsed) == len(BASE_FIELDS) and parsed[0] in ALGORITHMS:
            rows.append(dict(zip(BASE_FIELDS, parsed)))
    return rows


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


def run_command(command, log_dir, label, params):
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path = write_log(log_dir, label, command, completed)
    combined = f"{completed.stdout}\n{completed.stderr}"
    fallback = "falling back to serial multi-chain execution" in combined
    if fallback:
        print(f"[warning] fallback detected for {label}; see {log_path}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}; see {log_path}")
    rows = extract_csv_rows(completed.stdout)
    if not rows:
        raise SystemExit(f"No CSV rows found in command output; see {log_path}")
    enriched = []
    for row in rows:
        row.update(params)
        row["fallback"] = "1" if fallback else "0"
        enriched.append(row)
    return enriched


def command_for(exe, input_path, args, config):
    command = [
        str(exe),
        "--input",
        str(input_path),
        "--iterations",
        str(config["iterations"]),
        "--seed",
        str(args.seed),
        "--init",
        "nn",
        "--repeat",
        str(args.repeat),
        "--csv-only",
        "--parallel",
        "omp",
        "--chains",
        str(args.chains),
        "--threads",
        str(args.threads),
        "--t0",
        str(config["t0"]),
        "--tf",
        str(config["tf"]),
    ]
    if config["family"] == "qlsa":
        command.extend(
            [
                "--qlsa",
                "--alpha",
                str(config["alpha"]),
                "--gamma",
                str(config["gamma"]),
                "--epsilon",
                str(config["epsilon"]),
                "--policy",
                config["policy"],
            ]
        )
    return command


def metadata_from_config(config):
    return {
        "t0": str(config["t0"]),
        "tf": str(config["tf"]),
        "alpha": "" if config["family"] == "sa" else str(config["alpha"]),
        "gamma": "" if config["family"] == "sa" else str(config["gamma"]),
        "epsilon": "" if config["family"] == "sa" else str(config["epsilon"]),
        "policy": "" if config["family"] == "sa" else config["policy"],
        "tuning_stage": config["stage"],
    }


def quick_settings(args):
    args.instances = ["eil76"]
    args.iterations = 200_000
    args.repeat = 1
    return {
        "sa_iterations": [200_000],
        "sa_t0": [1000, 3000],
        "sa_tf": [0.001, 0.0001],
        "qlsa_stage1_alpha": [0.1],
        "qlsa_stage1_gamma": [0.9],
        "qlsa_stage1_epsilon": [0.1],
        "qlsa_stage1_policy": ["epsilon-greedy"],
        "qlsa_stage1_iterations": 200_000,
        "qlsa_stage2_iterations": [200_000],
        "qlsa_stage2_t0": [1000, 3000],
        "qlsa_stage2_tf": [0.001, 0.0001],
        "qlsa_stage2_topk": 1,
    }


def full_settings(args):
    return {
        "sa_iterations": sorted({args.iterations, args.iterations * 2}),
        "sa_t0": [100, 300, 1000, 3000],
        "sa_tf": [0.001, 0.0001, 0.00001],
        "qlsa_stage1_alpha": [0.05, 0.1, 0.2],
        "qlsa_stage1_gamma": [0.8, 0.9, 0.95],
        "qlsa_stage1_epsilon": [0.05, 0.1, 0.2],
        "qlsa_stage1_policy": ["epsilon-greedy", "softmax"],
        "qlsa_stage1_iterations": args.iterations,
        "qlsa_stage2_iterations": sorted({args.iterations, args.iterations * 2}),
        "qlsa_stage2_t0": [300, 1000, 3000],
        "qlsa_stage2_tf": [0.001, 0.0001],
        "qlsa_stage2_topk": 3,
    }


def sa_configs(settings):
    for iterations in settings["sa_iterations"]:
        for t0 in settings["sa_t0"]:
            for tf in settings["sa_tf"]:
                yield {
                    "family": "sa",
                    "stage": "sa-grid",
                    "iterations": iterations,
                    "t0": t0,
                    "tf": tf,
                    "alpha": "",
                    "gamma": "",
                    "epsilon": "",
                    "policy": "",
                }


def qlsa_stage1_configs(settings):
    for alpha in settings["qlsa_stage1_alpha"]:
        for gamma in settings["qlsa_stage1_gamma"]:
            for epsilon in settings["qlsa_stage1_epsilon"]:
                for policy in settings["qlsa_stage1_policy"]:
                    yield {
                        "family": "qlsa",
                        "stage": "qlsa-stage1",
                        "iterations": settings["qlsa_stage1_iterations"],
                        "t0": 1000,
                        "tf": 0.001,
                        "alpha": alpha,
                        "gamma": gamma,
                        "epsilon": epsilon,
                        "policy": policy,
                    }


def qlsa_stage2_configs(base_config, settings):
    for iterations in settings["qlsa_stage2_iterations"]:
        for t0 in settings["qlsa_stage2_t0"]:
            for tf in settings["qlsa_stage2_tf"]:
                config = dict(base_config)
                config["stage"] = "qlsa-stage2"
                config["iterations"] = iterations
                config["t0"] = t0
                config["tf"] = tf
                yield config


def score_rows(rows):
    best = min(int(row["best_length"]) for row in rows)
    elapsed = sum(float(row["elapsed_ms"]) for row in rows) / len(rows)
    return best, elapsed


def select_stage1_configs(rows, topk):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("tuning_stage") != "qlsa-stage1":
            continue
        key = (
            row["instance"],
            row["alpha"],
            row["gamma"],
            row["epsilon"],
            row["policy"],
        )
        grouped[key].append(row)

    per_instance = defaultdict(list)
    for key, group in grouped.items():
        instance, alpha, gamma, epsilon, policy = key
        best, elapsed = score_rows(group)
        per_instance[instance].append((best, elapsed, alpha, gamma, epsilon, policy))

    selected = {}
    for instance, configs in per_instance.items():
        configs.sort(key=lambda item: (item[0], item[1]))
        selected[instance] = [
            {
                "family": "qlsa",
                "stage": "qlsa-stage1-selected",
                "iterations": 1,
                "t0": 1000,
                "tf": 0.001,
                "alpha": float(alpha),
                "gamma": float(gamma),
                "epsilon": float(epsilon),
                "policy": policy,
            }
            for _, _, alpha, gamma, epsilon, policy in configs[:topk]
        ]
    return selected


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def load_existing_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def should_run_sa(args):
    return args.algorithm in ("sa", "both")


def should_run_qlsa(args):
    return args.algorithm in ("qlsa", "both")


def main():
    args = parse_args()
    settings = quick_settings(args) if args.quick else full_settings(args)
    root = Path.cwd()
    exe = find_executable(root)
    output_path = Path(args.output)
    log_dir = Path("results") / "logs" / "tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    all_rows = []

    if should_run_sa(args):
        for instance in args.instances:
            input_path = input_dir / f"{instance}.tsp"
            if not input_path.exists():
                print(f"[warning] missing {input_path}; skipping {instance}", flush=True)
                continue
            for config in sa_configs(settings):
                command = command_for(exe, input_path, args, config)
                label = f"{instance}_sa_t0{config['t0']}_tf{config['tf']}_it{config['iterations']}"
                all_rows.extend(run_command(command, log_dir, label, metadata_from_config(config)))

    stage1_rows = []
    if should_run_qlsa(args) and args.stage in ("1", "all"):
        for instance in args.instances:
            input_path = input_dir / f"{instance}.tsp"
            if not input_path.exists():
                print(f"[warning] missing {input_path}; skipping {instance}", flush=True)
                continue
            for config in qlsa_stage1_configs(settings):
                command = command_for(exe, input_path, args, config)
                label = (
                    f"{instance}_qlsa_s1_a{config['alpha']}_g{config['gamma']}"
                    f"_e{config['epsilon']}_{config['policy']}"
                )
                rows = run_command(command, log_dir, label, metadata_from_config(config))
                stage1_rows.extend(rows)
                all_rows.extend(rows)

    if should_run_qlsa(args) and args.stage in ("2", "all"):
        source_rows = stage1_rows if stage1_rows else load_existing_rows(output_path)
        selected = select_stage1_configs(source_rows, settings["qlsa_stage2_topk"])
        if not selected:
            raise SystemExit("QLSA stage 2 requires qlsa-stage1 rows in this run or existing output CSV.")
        for instance in args.instances:
            input_path = input_dir / f"{instance}.tsp"
            if not input_path.exists():
                print(f"[warning] missing {input_path}; skipping {instance}", flush=True)
                continue
            for base_config in selected.get(instance, []):
                for config in qlsa_stage2_configs(base_config, settings):
                    command = command_for(exe, input_path, args, config)
                    label = (
                        f"{instance}_qlsa_s2_a{config['alpha']}_g{config['gamma']}"
                        f"_e{config['epsilon']}_{config['policy']}_t0{config['t0']}"
                        f"_tf{config['tf']}_it{config['iterations']}"
                    )
                    all_rows.extend(run_command(command, log_dir, label, metadata_from_config(config)))

    write_rows(output_path, all_rows)
    print(f"Wrote tuning raw CSV: {output_path}", flush=True)


if __name__ == "__main__":
    main()
