#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BKS = {
    "berlin52": 7542,
    "eil51": 426,
    "st70": 675,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "pr76": 108159,
}

SUMMARY_FIELDS = [
    "instance",
    "algorithm",
    "parallel",
    "chains",
    "threads",
    "iterations",
    "t0",
    "tf",
    "alpha",
    "gamma",
    "epsilon",
    "policy",
    "tuning_stage",
    "runs",
    "bks",
    "best_length_min",
    "best_length_mean",
    "gap_percent",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "score",
    "speedup_vs_previous_baseline",
    "accepted_moves_mean",
    "improved_moves_mean",
]


def family(algorithm):
    return "qlsa" if algorithm.startswith("qlsa") else "sa"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def load_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_int(row, field, default=0):
    value = row.get(field, "")
    return int(value) if value != "" else default


def parse_float(row, field, default=0.0):
    value = row.get(field, "")
    return float(value) if value != "" else default


def load_previous_baselines(path=Path("results/step5_multi_cpu_summary.csv")):
    baselines = {}
    if not path.exists():
        return baselines
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            algorithm = row.get("algorithm", "")
            if algorithm in ("sa-omp", "qlsa-omp"):
                baselines[(row["instance"], family(algorithm))] = {
                    "elapsed_ms_mean": float(row["elapsed_ms_mean"]),
                    "gap_percent": float(row["gap_percent"]),
                    "best_length_min": int(row["best_length_min"]),
                }
    return baselines


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["instance"],
            row["algorithm"],
            row["parallel"],
            row["chains"],
            row["threads"],
            row["iterations"],
            row.get("t0", ""),
            row.get("tf", ""),
            row.get("alpha", ""),
            row.get("gamma", ""),
            row.get("epsilon", ""),
            row.get("policy", ""),
            row.get("tuning_stage", ""),
        )
        groups[key].append(row)
    return groups


def summarize(rows):
    baselines = load_previous_baselines()
    summaries = []
    for key, group in group_rows(rows).items():
        (
            instance,
            algorithm,
            parallel,
            chains,
            threads,
            iterations,
            t0,
            tf,
            alpha,
            gamma,
            epsilon,
            policy,
            tuning_stage,
        ) = key
        best_lengths = [parse_int(row, "best_length") for row in group]
        elapsed = [parse_float(row, "elapsed_ms") for row in group]
        accepted = [parse_float(row, "accepted_moves") for row in group]
        improved = [parse_float(row, "improved_moves") for row in group]
        bks = BKS.get(instance)
        best_min = min(best_lengths)
        best_mean = mean(best_lengths)
        gap = 100.0 * (best_min - bks) / bks if bks else 0.0
        elapsed_mean = mean(elapsed)
        score = gap + 0.001 * elapsed_mean
        baseline = baselines.get((instance, family(algorithm)))
        speedup = ""
        if baseline and elapsed_mean > 0:
            speedup = baseline["elapsed_ms_mean"] / elapsed_mean
        summaries.append(
            {
                "instance": instance,
                "algorithm": algorithm,
                "parallel": parallel,
                "chains": int(chains),
                "threads": int(threads),
                "iterations": int(iterations),
                "t0": t0,
                "tf": tf,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "policy": policy,
                "tuning_stage": tuning_stage,
                "runs": len(group),
                "bks": bks if bks is not None else "",
                "best_length_min": best_min,
                "best_length_mean": best_mean,
                "gap_percent": gap,
                "elapsed_ms_mean": elapsed_mean,
                "elapsed_ms_std": sample_std(elapsed),
                "score": score,
                "speedup_vs_previous_baseline": speedup,
                "accepted_moves_mean": mean(accepted),
                "improved_moves_mean": mean(improved),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["instance"],
            family(row["algorithm"]),
            row["gap_percent"],
            row["elapsed_ms_mean"],
            row["algorithm"],
        )
    )
    return summaries


def fmt_csv(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt_csv(row.get(field, "")) for field in SUMMARY_FIELDS})


def best_rows_for_instance(rows, instance, fam):
    candidates = [row for row in rows if row["instance"] == instance and family(row["algorithm"]) == fam]
    if not candidates:
        return None, None
    quality = min(candidates, key=lambda row: (row["gap_percent"], row["best_length_min"], row["elapsed_ms_mean"]))
    tradeoff = min(candidates, key=lambda row: (row["score"], row["gap_percent"], row["elapsed_ms_mean"]))
    return quality, tradeoff


def config_text(row):
    if row is None:
        return "-"
    parts = [
        f"it={row['iterations']}",
        f"t0={row['t0']}",
        f"tf={row['tf']}",
    ]
    if family(row["algorithm"]) == "qlsa":
        parts.extend(
            [
                f"a={row['alpha']}",
                f"g={row['gamma']}",
                f"eps={row['epsilon']}",
                f"{row['policy']}",
            ]
        )
    return ", ".join(parts)


def md_float(value, digits=3):
    if value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def make_markdown(summary_rows, source_path):
    instances = sorted({row["instance"] for row in summary_rows})
    baselines = load_previous_baselines()
    lines = [
        "# Step 6A Parameter Tuning Analysis",
        "",
        "## Purpose",
        "",
        "This stage searches SA and QLSA parameters to reduce the remaining Gap on eil76, rat99, and eil101, while keeping the algorithm implementation unchanged.",
        "",
        "## Search Space",
        "",
        "- SA: iterations in {1e6, 2e6}, t0 in {100, 300, 1000, 3000}, tf in {1e-3, 1e-4, 1e-5}.",
        "- QLSA stage 1: t0=1000, tf=1e-3, iterations=1e6, alpha/gamma/epsilon/policy grid.",
        "- QLSA stage 2: top stage-1 configurations per instance are expanded over t0/tf/iterations.",
        "- Quick mode uses a reduced eil76-only grid for smoke testing.",
        "- Tradeoff score is empirical: score = gap_percent + 0.001 * elapsed_ms_mean.",
        "",
        f"Raw input: `{source_path.as_posix()}`",
        "",
        "## Best Configurations",
        "",
        "| Instance | Family | Best Quality Gap % | Best Quality Config | Tradeoff Gap % | Tradeoff Mean ms | Tradeoff Config |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for instance in instances:
        for fam in ("sa", "qlsa"):
            quality, tradeoff = best_rows_for_instance(summary_rows, instance, fam)
            lines.append(
                "| {instance} | {fam} | {qgap} | {qconf} | {tgap} | {tms} | {tconf} |".format(
                    instance=instance,
                    fam=fam.upper(),
                    qgap=md_float(quality["gap_percent"]) if quality else "-",
                    qconf=config_text(quality),
                    tgap=md_float(tradeoff["gap_percent"]) if tradeoff else "-",
                    tms=md_float(tradeoff["elapsed_ms_mean"]) if tradeoff else "-",
                    tconf=config_text(tradeoff),
                )
            )

    lines.extend(["", "## Comparison With Step 5B Defaults", ""])
    for instance in instances:
        lines.append(f"### {instance}")
        for fam in ("sa", "qlsa"):
            quality, _ = best_rows_for_instance(summary_rows, instance, fam)
            baseline = baselines.get((instance, fam))
            if quality and baseline:
                lines.append(
                    "- {fam}: previous gap {bgap:.3f}%, tuned best gap {tgap:.3f}%, previous best {bbest}, tuned best {tbest}.".format(
                        fam=fam.upper(),
                        bgap=baseline["gap_percent"],
                        tgap=quality["gap_percent"],
                        bbest=baseline["best_length_min"],
                        tbest=quality["best_length_min"],
                    )
                )
            elif quality:
                lines.append(
                    f"- {fam.upper()}: tuned best gap {quality['gap_percent']:.3f}% with best length {quality['best_length_min']}."
                )
        lines.append("")

    lines.extend(["## QLSA vs SA Observation", ""])
    for instance in instances:
        sa_quality, _ = best_rows_for_instance(summary_rows, instance, "sa")
        qlsa_quality, _ = best_rows_for_instance(summary_rows, instance, "qlsa")
        if not sa_quality or not qlsa_quality:
            continue
        if qlsa_quality["gap_percent"] < sa_quality["gap_percent"]:
            verdict = "QLSA found a lower Gap than SA in this tuning run."
        elif qlsa_quality["gap_percent"] == sa_quality["gap_percent"]:
            verdict = "QLSA matched SA quality in this tuning run."
        else:
            verdict = "SA remained better than QLSA in this tuning run."
        lines.append(f"- {instance}: {verdict}")

    lines.extend(["", "## Next Steps", ""])
    lines.append("- Run the full tuning grid before drawing final conclusions.")
    lines.append("- Promote the best SA/QLSA configurations into the final multi-instance experiment matrix.")
    lines.append("- Keep CUDA conclusions separate unless the CUDA runs are confirmed non-fallback and competitive on larger instances.")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path, rows, source_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_markdown(rows, source_path), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze Step 6A tuning results.")
    parser.add_argument("--input", default="results/tuning_raw.csv")
    parser.add_argument("--output", default="results/tuning_summary.csv")
    parser.add_argument("--markdown", default="docs/step6A_tuning_analysis.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)

    rows = load_rows(input_path)
    summaries = summarize(rows)
    write_summary(output_path, summaries)
    write_markdown(markdown_path, summaries, input_path)
    print(f"Wrote tuning summary CSV: {output_path}")
    print(f"Wrote tuning analysis markdown: {markdown_path}")


if __name__ == "__main__":
    main()
