#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BKS = {
    "rat99": 1211,
    "eil101": 629,
}

STEP6B_BASELINES = {
    ("eil101", "sa", "tuned-high-budget"): {
        "label": "eil101 SA tuned",
        "min_gap": 0.477,
        "mean_gap": 1.717,
    },
    ("eil101", "qlsa", "tuned-high-budget"): {
        "label": "eil101 QLSA tuned",
        "min_gap": 0.477,
        "mean_gap": 1.526,
    },
    ("rat99", "sa", "tuned-high-budget"): {
        "label": "rat99 SA tuned",
        "min_gap": 0.165,
        "mean_gap": 0.875,
    },
    ("rat99", "qlsa", "quality-first-high-budget"): {
        "label": "rat99 QLSA quality-first",
        "min_gap": 0.083,
        "mean_gap": 0.372,
    },
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
    "score",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Step 6C targeted high-budget quality experiments.")
    parser.add_argument("--input", default="results/targeted_quality_raw.csv")
    parser.add_argument("--output", default="results/targeted_quality_summary.csv")
    parser.add_argument("--markdown", default="docs/step6C_targeted_quality_analysis.md")
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
        elapsed_mean = mean(elapsed)
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
                "elapsed_ms_mean": elapsed_mean,
                "elapsed_ms_std": sample_std(elapsed),
                "accepted_moves_mean": mean(accepted),
                "improved_moves_mean": mean(improved),
                "score": gap_min + 0.001 * elapsed_mean,
            }
        )
    summaries.sort(key=lambda row: (row["instance"], row["family"], row["iterations"], row["chains"]))
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
        f"chains={row['chains']}",
        f"threads={row['threads']}",
        f"t0={row['t0']}",
        f"tf={row['tf']}",
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


def best_quality(rows):
    if not rows:
        return None
    return min(rows, key=lambda row: (row["gap_percent_min"], row["gap_percent_mean"], row["elapsed_ms_mean"]))


def best_tradeoff(rows):
    if not rows:
        return None
    return min(rows, key=lambda row: (row["score"], row["gap_percent_mean"], row["elapsed_ms_mean"]))


def baseline_for(row):
    return STEP6B_BASELINES.get((row["instance"], row["family"], row["variant"]))


def reached_bks(row):
    return row["bks"] != "" and row["best_length_min"] == row["bks"]


def make_markdown(summaries, source_path):
    by_instance = defaultdict(list)
    for row in summaries:
        by_instance[row["instance"]].append(row)

    lines = [
        "# Step 6C Targeted Quality Experiment Analysis",
        "",
        "## Purpose",
        "",
        "- This stage is not a new full parameter search; it expands search budget around selected Step 6B configurations.",
        "- Increasing chains launches more independent search chains in one experiment, which usually increases the chance of finding a better tour.",
        "- Increasing iterations lets each chain search more deeply, but directly increases runtime.",
        "- The final report should present both solution quality and runtime cost.",
        "- The time-quality tradeoff score used here is empirical: score = gap_percent_min + 0.001 * elapsed_ms_mean.",
        "",
        f"Raw input: `{source_path.as_posix()}`",
        "",
        "## Summary Table",
        "",
        "| Instance | Family | Variant | Runs | Best | Gap Min % | Gap Mean % | Mean ms | Std ms | Score | Parameters |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {instance} | {family} | {variant} | {runs} | {best} | {gap_min} | {gap_mean} | {ms} | {std} | {score} | {params} |".format(
                instance=row["instance"],
                family=row["family"].upper(),
                variant=row["variant"],
                runs=row["runs"],
                best=row["best_length_min"],
                gap_min=md_float(row["gap_percent_min"]),
                gap_mean=md_float(row["gap_percent_mean"]),
                ms=md_float(row["elapsed_ms_mean"]),
                std=md_float(row["elapsed_ms_std"]),
                score=md_float(row["score"]),
                params=param_text(row),
            )
        )

    lines.extend(["", "## Per-Instance Selection", ""])
    for instance in sorted(by_instance):
        rows = by_instance[instance]
        quality = best_quality(rows)
        tradeoff = best_tradeoff(rows)
        lines.append(f"### {instance}")
        if quality:
            bks_text = "reached BKS" if reached_bks(quality) else "did not reach BKS"
            lines.append(
                f"- Best quality: {quality['family'].upper()} {quality['variant']} ({param_text(quality)}), best={quality['best_length_min']}, min gap={quality['gap_percent_min']:.3f}%, mean gap={quality['gap_percent_mean']:.3f}%, {bks_text}."
            )
        if tradeoff:
            lines.append(
                f"- Best time-quality tradeoff: {tradeoff['family'].upper()} {tradeoff['variant']} ({param_text(tradeoff)}), score={tradeoff['score']:.3f}, mean_ms={tradeoff['elapsed_ms_mean']:.3f}."
            )
        lines.append("")

    lines.extend(["## Comparison With Step 6B", ""])
    for row in summaries:
        baseline = baseline_for(row)
        if not baseline:
            continue
        min_delta = row["gap_percent_min"] - baseline["min_gap"]
        mean_delta = row["gap_percent_mean"] - baseline["mean_gap"]
        min_word = "improved" if min_delta < 0 else "matched" if abs(min_delta) < 1e-9 else "worse"
        mean_word = "improved" if mean_delta < 0 else "matched" if abs(mean_delta) < 1e-9 else "worse"
        lines.append(
            "- {name} {params}: Step 6B min/mean gap {bmin:.3f}%/{bmean:.3f}%; Step 6C min/mean gap {cmin:.3f}%/{cmean:.3f}% ({min_word} min, {mean_word} mean).".format(
                name=baseline["label"],
                params=f"it={row['iterations']}, chains={row['chains']}",
                bmin=baseline["min_gap"],
                bmean=baseline["mean_gap"],
                cmin=row["gap_percent_min"],
                cmean=row["gap_percent_mean"],
                min_word=min_word,
                mean_word=mean_word,
            )
        )

    lines.extend(["", "## Notes", ""])
    lines.append("- Full conclusions require the complete repeat=5 targeted run; quick mode is only a smoke test.")
    lines.append("- Because this stage increases budget, any quality improvement should be discussed together with elapsed time.")
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
    print(f"Wrote targeted quality summary CSV: {output_path}")
    print(f"Wrote targeted quality analysis markdown: {markdown_path}")


if __name__ == "__main__":
    main()
