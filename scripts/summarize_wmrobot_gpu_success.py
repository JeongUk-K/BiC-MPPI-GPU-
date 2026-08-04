#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path


RESULT_FILES = {
    "mppi": "result_mppi.csv",
    "log_mppi": "result_log_mppi.csv",
    "cluster_mppi_dbscan": "result_cluster_mppi.csv",
    "cluster_mppi_kmeans": "result_cluster_mppi_kmeans.csv",
    "bi_mppi_dbscan": "result_bi_mppi_se2.csv",
    "bi_mppi_kmeans": "result_bi_mppi_kmeans_se2.csv",
}


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def finite_values(rows, field):
    values = []
    for row in rows:
        value = float(row[field])
        if math.isfinite(value):
            values.append(value)
    return values


def describe(values):
    if not values:
        return {
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize only successful WMRobot GPU benchmark runs."
    )
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    successful_rows = []
    summary_rows = []

    for example, relative_path in RESULT_FILES.items():
        csv_path = result_dir / example / relative_path
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        successes = [row for row in rows if row["is_success"] == "1"]
        collisions = sum(row["is_collision"] == "1" for row in rows)

        for row in successes:
            successful_rows.append({"example": example, **row})

        executed_steps = [float(row["iter"]) + 1.0 for row in successes]
        per_iter_ms = [
            1000.0 * float(row["elapsed"]) / (float(row["iter"]) + 1.0)
            for row in successes
        ]
        per_iter_stats = describe(per_iter_ms)

        summary = {
            "example": example,
            "variant": successes[0]["variant"] if successes else rows[0]["variant"],
            "total_runs": len(rows),
            "num_success": len(successes),
            "success_rate": len(successes) / len(rows) if rows else math.nan,
            "num_collision": collisions,
            "num_non_success": len(rows) - len(successes),
            "num_timeout_or_other_failure": len(rows) - len(successes) - collisions,
        }
        for statistic, value in describe(executed_steps).items():
            summary[f"executed_steps_{statistic}"] = value
        for prefix, field in (
            ("iter", "iter"),
            ("elapsed_s", "elapsed"),
            ("rollout_s", "elapsed_rollout"),
            ("clustering_s", "elapsed_clustering"),
            ("connection_s", "elapsed_connection"),
            ("guide_s", "elapsed_guide"),
            ("goal_distance", "d_goal"),
            ("connection_distance", "d_conn"),
        ):
            for statistic, value in describe(
                finite_values(successes, field)
            ).items():
                summary[f"{prefix}_{statistic}"] = value
        for statistic, value in per_iter_stats.items():
            summary[f"elapsed_per_iter_ms_{statistic}"] = value
        summary_rows.append(summary)

    success_output = result_dir / "successful_runs.csv"
    success_fields = ["example"] + [
        "variant",
        "start_case",
        "map",
        "is_success",
        "is_collision",
        "iter",
        "elapsed",
        "elapsed_rollout",
        "elapsed_clustering",
        "elapsed_connection",
        "elapsed_guide",
        "d_goal",
        "d_conn",
    ]
    with success_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=success_fields)
        writer.writeheader()
        writer.writerows(successful_rows)

    summary_output = result_dir / "success_only_summary.csv"
    with summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"successful rows: {len(successful_rows)}")
    print(f"wrote: {success_output}")
    print(f"wrote: {summary_output}")


if __name__ == "__main__":
    main()
