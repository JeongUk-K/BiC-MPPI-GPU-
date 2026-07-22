#!/usr/bin/env python3
"""Create publication-ready WMRobot statistics from per-trial CSV files.

Continuous outcomes are summarized over successful trials only. Success-rate
confidence intervals use Wilson's method; continuous-outcome confidence
intervals are two-sided 95% Student-t intervals for the mean. Pairwise tests
remain matched by (start_case, map): exact McNemar tests for success and exact
sign tests over jointly successful trials for continuous outcomes.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


METHODS = [
    ("MPPI", "result_mppi.csv"),
    ("Log-MPPI", "result_log_mppi.csv"),
    ("Cluster-MPPI-DBSCAN", "result_cluster_mppi.csv"),
    ("Cluster-MPPI-KMeans", "result_cluster_mppi_kmeans.csv"),
    ("BiC-MPPI-DBSCAN (ours)", "result_bi_mppi.csv"),
    ("BiC-MPPI-KMeans (ours)", "result_bi_mppi_kmeans.csv"),
]

METRICS = [
    ("iterations", "iter", "Iter.", 1),
    ("total_s", "elapsed", "Total [s]", 3),
    ("rollout_s", "elapsed_rollout", "Roll. [s]", 3),
    ("clustering_s", "elapsed_clustering", "Clust. [s]", 3),
    ("connection_s", "elapsed_connection", "Conn. [s]", 3),
    ("guide_s", "elapsed_guide", "Guide [s]", 3),
]


def flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "success"}


def finite_number(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values: list[float], probability: float) -> float:
    """R-7/NumPy-linear sample quantile."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def t_critical_975(degrees_freedom: int) -> float:
    """Accurate large-df Cornish-Fisher approximation to t_(0.975, df)."""
    if degrees_freedom <= 0:
        return math.nan
    z = 1.959963984540054
    df = float(degrees_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan, math.nan
    margin = t_critical_975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    numerator = sum(math.comb(trials, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**trials))


def holm_adjust(rows: list[dict[str, object]]) -> None:
    """Add Holm-adjusted p-values within each outcome family."""
    families: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        families.setdefault(str(row["outcome"]), []).append(row)
    for family in families.values():
        ordered = sorted(family, key=lambda row: float(row["p_value_raw"]))
        running = 0.0
        count = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (count - index) * float(row["p_value_raw"]))
            running = max(running, adjusted)
            row["p_value_holm"] = running


def read_methods(input_dir: Path) -> dict[str, dict[tuple[int, int], dict[str, str]]]:
    loaded = {}
    missing = []
    for method, filename in METHODS:
        path = input_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        indexed = {(int(row["start_case"]), int(row["map"])): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate (start_case, map) keys in {path}")
        loaded[method] = indexed
    if missing:
        raise FileNotFoundError("missing required result CSV(s):\n" + "\n".join(missing))
    return loaded


def descriptive_rows(data: dict[str, dict[tuple[int, int], dict[str, str]]]):
    output = []
    status = {}
    for method, _ in METHODS:
        rows = list(data[method].values())
        successful = [row for row in rows if flag(row.get("is_success", ""))]
        total = len(rows)
        successes = len(successful)
        collisions = sum(flag(row.get("is_collision", "")) for row in rows)
        ci_low, ci_high = wilson_interval(successes, total)
        status[method] = {
            "method": method,
            "trials": total,
            "successes": successes,
            "success_rate": successes / total,
            "success_rate_ci_low": ci_low,
            "success_rate_ci_high": ci_high,
            "failures": total - successes,
            "collisions": collisions,
        }
        for metric, field, label, _ in METRICS:
            values = [value for row in successful if (value := finite_number(row, field)) is not None]
            mean, mean_low, mean_high = mean_ci(values)
            output.append({
                **status[method],
                "metric": metric,
                "metric_label": label,
                "n_success_metric": len(values),
                "mean": mean,
                "mean_ci_low": mean_low,
                "mean_ci_high": mean_high,
                "median": statistics.median(values),
                "q1": quantile(values, 0.25),
                "q3": quantile(values, 0.75),
            })
    return status, output


def pairwise_rows(data: dict[str, dict[tuple[int, int], dict[str, str]]]):
    output: list[dict[str, object]] = []
    names = [method for method, _ in METHODS]
    for left_index, method_a in enumerate(names):
        for method_b in names[left_index + 1:]:
            common = sorted(set(data[method_a]) & set(data[method_b]))
            a_success = [flag(data[method_a][key].get("is_success", "")) for key in common]
            b_success = [flag(data[method_b][key].get("is_success", "")) for key in common]
            a_only = sum(a and not b for a, b in zip(a_success, b_success))
            b_only = sum(b and not a for a, b in zip(a_success, b_success))
            output.append({
                "outcome": "success_rate",
                "test": "exact McNemar",
                "method_a": method_a,
                "method_b": method_b,
                "matched_trials": len(common),
                "joint_successes": sum(a and b for a, b in zip(a_success, b_success)),
                "nonzero_pairs": a_only + b_only,
                "effect_a_minus_b": statistics.fmean(a_success) - statistics.fmean(b_success),
                "effect_unit": "proportion",
                "p_value_raw": exact_two_sided_binomial(a_only, a_only + b_only),
            })
            joint_keys = [key for key, a, b in zip(common, a_success, b_success) if a and b]
            for metric, field, _, _ in METRICS:
                pairs = []
                for key in joint_keys:
                    a_value = finite_number(data[method_a][key], field)
                    b_value = finite_number(data[method_b][key], field)
                    if a_value is not None and b_value is not None:
                        pairs.append((a_value, b_value))
                differences = [a - b for a, b in pairs]
                nonzero = [difference for difference in differences if difference != 0.0]
                positives = sum(difference > 0.0 for difference in nonzero)
                output.append({
                    "outcome": metric,
                    "test": "exact paired sign",
                    "method_a": method_a,
                    "method_b": method_b,
                    "matched_trials": len(common),
                    "joint_successes": len(pairs),
                    "nonzero_pairs": len(nonzero),
                    "effect_a_minus_b": statistics.median(differences),
                    "effect_unit": "iterations" if metric == "iterations" else "seconds",
                    "p_value_raw": exact_two_sided_binomial(positives, len(nonzero)),
                })
    holm_adjust(output)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def write_publication_table(path: Path, status, descriptions) -> None:
    by_method_metric = {(row["method"], row["metric"]): row for row in descriptions}
    fields = ["Method", "Succ.", "SR [95% CI]", "Fail.", "Coll."] + [label for _, _, label, _ in METRICS]
    rows = []
    for method, _ in METHODS:
        state = status[method]
        row = {
            "Method": method,
            "Succ.": state["successes"],
            "SR [95% CI]": (
                f'{state["success_rate"]:.3f} '
                f'[{state["success_rate_ci_low"]:.3f}, {state["success_rate_ci_high"]:.3f}]'
            ),
            "Fail.": state["failures"],
            "Coll.": state["collisions"],
        }
        for metric, _, label, digits in METRICS:
            item = by_method_metric[(method, metric)]
            row[label] = (
                f'{number(item["mean"], digits)} '
                f'({number(item["mean_ci_low"], digits)}, {number(item["mean_ci_high"], digits)}); '
                f'{number(item["median"], digits)} '
                f'[{number(item["q1"], digits)}, {number(item["q3"], digits)}]'
            )
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, status, descriptions) -> None:
    by_method_metric = {(row["method"], row["metric"]): row for row in descriptions}
    lines = [
        "# WMRobot statistical summary",
        "",
        "Success rate uses all 600 matched trials and a 95% Wilson CI. Failure and collision are counts over all trials.",
        "",
        "| Method | Succ. | SR [95% CI] | Fail. | Coll. |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, _ in METHODS:
        item = status[method]
        lines.append(
            f'| {method} | {item["successes"]} | {item["success_rate"]:.3f} '
            f'[{item["success_rate_ci_low"]:.3f}, {item["success_rate_ci_high"]:.3f}] '
            f'| {item["failures"]} | {item["collisions"]} |'
        )
    lines += [
        "",
        "Continuous outcomes below use successful trials only. Each cell is mean (95% CI); median [Q1, Q3].",
        "",
        "| Method | " + " | ".join(label for _, _, label, _ in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for method, _ in METHODS:
        cells = []
        for metric, _, _, digits in METRICS:
            item = by_method_metric[(method, metric)]
            cells.append(
                f'{number(item["mean"], digits)} '
                f'({number(item["mean_ci_low"], digits)}, {number(item["mean_ci_high"], digits)}); '
                f'{number(item["median"], digits)} '
                f'[{number(item["q1"], digits)}, {number(item["q3"], digits)}]'
            )
        lines.append(f'| {method} | ' + " | ".join(cells) + " |")
    lines += [
        "",
        "## Statistical comparison files",
        "",
        "`wmrobot_pairwise_comparisons.csv` contains exact paired comparisons because every method was run on the same `(start_case, map)` cases. Success uses an exact McNemar test. Continuous outcomes use an exact paired sign test on cases where both methods succeeded. `effect_a_minus_b` is the paired median difference for continuous outcomes and the success-rate difference for success. Holm-adjusted p-values are computed separately for each outcome.",
        "",
        "Interpretation: a negative continuous-outcome effect favors method A (fewer iterations/lower time); a positive success-rate effect favors method A.",
        "",
        "The confidence interval for a continuous outcome is a two-sided Student-t interval for its mean. Quartiles use the linear (R-7/NumPy) definition.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/wmrobot/raw_csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/wmrobot/statistics"))
    args = parser.parse_args()

    data = read_methods(args.input_dir)
    status, descriptions = descriptive_rows(data)
    comparisons = pairwise_rows(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "wmrobot_success_descriptive_stats.csv", descriptions)
    write_publication_table(args.output_dir / "wmrobot_publication_table.csv", status, descriptions)
    write_csv(args.output_dir / "wmrobot_pairwise_comparisons.csv", comparisons)
    write_markdown(args.output_dir / "wmrobot_statistical_report.md", status, descriptions)
    print(f"Wrote WMRobot statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
