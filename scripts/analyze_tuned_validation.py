#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BKS = {
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
}

DEFAULT_GAPS = {
    ("eil76", "sa"): 0.186,
    ("eil76", "qlsa"): 0.743,
    ("rat99", "sa"): 0.330,
    ("rat99", "qlsa"): 1.156,
    ("eil101", "sa"): 0.954,
    ("eil101", "qlsa"): 1.272,
}

SUMMARY_FIELDS = [
    "instance",
    "family",
    "variant",
    "config_name",
    "iterations",
    "chains",
    "threads",
    "parallel",
    "t0",
    "tf",
    "alpha",
    "gamma",
    "epsilon",
    "policy",
    "runs",
    "bks",
    "best_length_min",
    "best_length_mean",
    "gap_percent_min",
    "gap_percent_mean",
    "elapsed_ms_mean",
    "elapsed_ms_std",
    "accepted_moves_mean",
    "improved_moves_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Step 6B tuned validation results.")
    parser.add_argument("--input", default="results/tuned_validation_raw.csv")
    parser.add_argument("--output", default="results/tuned_validation_summary.csv")
    parser.add_argument("--markdown", default="docs/step6B_tuned_validation_analysis.md")
    return parser.parse_args()


def load_rows(path):
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_int(row, field, default=0):
    value = row.get(field, "")
    return int(value) if value != "" else default


def parse_float(row, field, default=0.0):
    value = row.get(field, "")
    return float(value) if value != "" else default


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["instance"],
            row["family"],
            row["variant"],
            row["config_name"],
            row["iterations"],
            row["chains"],
            row["threads"],
            row["parallel"],
            row["t0"],
            row["tf"],
            row.get("alpha", ""),
            row.get("gamma", ""),
            row.get("epsilon", ""),
            row.get("policy", ""),
        )
        groups[key].append(row)
    return groups


def summarize(rows):
    summaries = []
    for key, group in group_rows(rows).items():
        (
            instance,
            family,
            variant,
            config_name,
            iterations,
            chains,
            threads,
            parallel,
            t0,
            tf,
            alpha,
            gamma,
            epsilon,
            policy,
        ) = key
        bks = BKS.get(instance)
        best_lengths = [parse_int(row, "best_length") for row in group]
        elapsed = [parse_float(row, "elapsed_ms") for row in group]
        accepted = [parse_float(row, "accepted_moves") for row in group]
        improved = [parse_float(row, "improved_moves") for row in group]
        best_min = min(best_lengths)
        best_mean = mean(best_lengths)
        gap_min = 100.0 * (best_min - bks) / bks if bks else 0.0
        gap_mean = 100.0 * (best_mean - bks) / bks if bks else 0.0
        summaries.append(
            {
                "instance": instance,
                "family": family,
                "variant": variant,
                "config_name": config_name,
                "iterations": int(iterations),
                "chains": int(chains),
                "threads": int(threads),
                "parallel": parallel,
                "t0": t0,
                "tf": tf,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "policy": policy,
                "runs": len(group),
                "bks": bks if bks is not None else "",
                "best_length_min": best_min,
                "best_length_mean": best_mean,
                "gap_percent_min": gap_min,
                "gap_percent_mean": gap_mean,
                "elapsed_ms_mean": mean(elapsed),
                "elapsed_ms_std": sample_std(elapsed),
                "accepted_moves_mean": mean(accepted),
                "improved_moves_mean": mean(improved),
            }
        )
    summaries.sort(key=lambda row: (row["instance"], row["family"], row["variant"]))
    return summaries


def fmt_csv(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_summary(path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: fmt_csv(row.get(field, "")) for field in SUMMARY_FIELDS})


def md_float(value, digits=3):
    if value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def param_text(row):
    parts = [
        f"it={row['iterations']}",
        f"t0={row['t0']}",
        f"tf={row['tf']}",
        f"chains={row['chains']}",
        f"threads={row['threads']}",
    ]
    if row["family"] == "qlsa":
        parts.extend(
            [
                f"alpha={row['alpha']}",
                f"gamma={row['gamma']}",
                f"epsilon={row['epsilon']}",
                f"policy={row['policy']}",
            ]
        )
    return ", ".join(parts)


def stable_bks(row):
    return row["bks"] != "" and row["best_length_min"] == row["bks"] and abs(row["best_length_mean"] - row["bks"]) < 1e-9


def make_markdown(summaries, source_path):
    lines = [
        "# Step 6B Tuned Parameter Independent Validation",
        "",
        "## Purpose",
        "",
        "- Use the tuned parameters selected in Step 6A.",
        "- Validate with independent seeds starting at seed=101.",
        "- Use repeat=10 for the full validation run.",
        "- Avoid reporting only the best result selected during the tuning search.",
        "",
        f"Raw input: `{source_path.as_posix()}`",
        "",
        "## Validation Results",
        "",
        "| Instance | Family | Variant | Runs | BKS | Best Min | Best Mean | Gap Min % | Gap Mean % | Mean ms | Std ms | Parameters |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {instance} | {family} | {variant} | {runs} | {bks} | {best_min} | {best_mean} | {gap_min} | {gap_mean} | {ms} | {std} | {params} |".format(
                instance=row["instance"],
                family=row["family"].upper(),
                variant=row["variant"],
                runs=row["runs"],
                bks=row["bks"],
                best_min=row["best_length_min"],
                best_mean=md_float(row["best_length_mean"]),
                gap_min=md_float(row["gap_percent_min"]),
                gap_mean=md_float(row["gap_percent_mean"]),
                ms=md_float(row["elapsed_ms_mean"]),
                std=md_float(row["elapsed_ms_std"]),
                params=param_text(row),
            )
        )

    lines.extend(["", "## Comparison With Step 5B Default Parameters", ""])
    by_instance = defaultdict(list)
    for row in summaries:
        by_instance[row["instance"]].append(row)
    for instance in sorted(by_instance):
        lines.append(f"### {instance}")
        for row in sorted(by_instance[instance], key=lambda item: (item["family"], item["variant"])):
            default_gap = DEFAULT_GAPS.get((instance, row["family"]))
            if default_gap is None:
                continue
            lines.append(
                "- {family} {variant}: default gap {default_gap:.3f}%, tuned validation min gap {gap_min:.3f}%, mean gap {gap_mean:.3f}%.".format(
                    family=row["family"].upper(),
                    variant=row["variant"],
                    default_gap=default_gap,
                    gap_min=row["gap_percent_min"],
                    gap_mean=row["gap_percent_mean"],
                )
            )
        lines.append("")

    lines.extend(["## Conclusions From Available Validation Rows", ""])
    for instance in sorted(by_instance):
        stable = [row for row in by_instance[instance] if stable_bks(row)]
        if stable:
            names = ", ".join(f"{row['family'].upper()} {row['variant']}" for row in stable)
            lines.append(f"- {instance}: {names} reached BKS in every recorded validation run.")
        else:
            best = min(by_instance[instance], key=lambda row: (row["gap_percent_min"], row["gap_percent_mean"]))
            lines.append(
                f"- {instance}: best recorded validation config is {best['family'].upper()} {best['variant']} with min gap {best['gap_percent_min']:.3f}% and mean gap {best['gap_percent_mean']:.3f}%."
            )

    rat99_rows = by_instance.get("rat99", [])
    sa_rat99 = [row for row in rat99_rows if row["family"] == "sa"]
    qlsa_rat99 = [row for row in rat99_rows if row["family"] == "qlsa"]
    if sa_rat99 and qlsa_rat99:
        best_sa = min(sa_rat99, key=lambda row: (row["gap_percent_min"], row["gap_percent_mean"]))
        best_qlsa = min(qlsa_rat99, key=lambda row: (row["gap_percent_min"], row["gap_percent_mean"]))
        if best_qlsa["gap_percent_min"] < best_sa["gap_percent_min"]:
            lines.append("- rat99: QLSA still has a better minimum Gap than SA in this validation output.")
        elif best_qlsa["gap_percent_min"] == best_sa["gap_percent_min"]:
            lines.append("- rat99: QLSA and SA tie on minimum Gap in this validation output; compare mean Gap and runtime before final wording.")
        else:
            lines.append("- rat99: SA has a better minimum Gap than QLSA in this validation output.")

    lines.extend(["", "## Notes", ""])
    lines.append("- Tuning improves solution quality relative to default parameters, but some configurations increase runtime.")
    lines.append("- The final report should show both the default-parameter parallel speedup results and the tuned-parameter solution-quality results.")
    lines.append("- Quick mode is only a script smoke test; full repeat=10 validation is required before making final claims.")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path, summaries, source_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_markdown(summaries, source_path), encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)

    rows = load_rows(input_path)
    summaries = summarize(rows)
    write_summary(output_path, summaries)
    write_markdown(markdown_path, summaries, input_path)
    print(f"Wrote tuned validation summary CSV: {output_path}")
    print(f"Wrote tuned validation analysis markdown: {markdown_path}")


if __name__ == "__main__":
    main()
