#!/usr/bin/env python3
"""Compare matched WMRobot DBSCAN and GPU K-means result rows."""

import argparse
import csv
import math
from pathlib import Path


SOLVERS = {
    "cluster_mppi": "result_cluster_mppi.csv",
    "bi_mppi": "result_bi_mppi.csv",
}

SOLVER_LABELS = {
    "cluster_mppi": "Cluster-MPPI",
    "bi_mppi": "BiC-MPPI",
}

AGGREGATE_FIELDS = [
    "source", "group", "variant", "num_rows", "num_runs", "num_success",
    "success_rate", "num_collision", "num_landed", "landed_rate",
    "avg_iter", "avg_elapsed", "avg_rollout", "avg_clustering",
    "avg_connection", "avg_guide", "avg_f_err", "avg_final_error",
]

COMPARISON_AGGREGATE_FIELDS = [
    "backend", "comparison_solver", "matched_runs",
    "mean_clustering_s_per_step", "mean_total_s_per_step",
    "clustering_speedup_dbscan_over_kmeans",
    "total_step_speedup_dbscan_over_kmeans",
    "success_rate_delta_kmeans_minus_dbscan",
]


def read_rows(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(int(row["start_case"]), int(row["map"])): row for row in rows}


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def flag(row, key):
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def per_step(row, key):
    iterations = max(1.0, number(row, "iter"))
    return number(row, key) / iterations


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def fmt(value):
    return "nan" if not math.isfinite(value) else f"{value:.10g}"


def write_comparison_aggregate(path, detail_rows, summaries):
    """Add idempotent DBSCAN/K-means rows to the standard aggregate CSV."""
    existing_fields = []
    existing_rows = []
    if path.is_file():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fields = list(existing_fields)
    for field in AGGREGATE_FIELDS + COMPARISON_AGGREGATE_FIELDS:
        if field not in fields:
            fields.append(field)

    # Re-running the comparator replaces its own rows but preserves aggregates
    # produced by other WMRobot experiments in the same output directory.
    rows = [row for row in existing_rows
            if row.get("group") != "backend_comparison"]
    summary_by_solver = {row["solver"]: row for row in summaries}

    for solver in SOLVERS:
        matched = [row for row in detail_rows if row["solver"] == solver]
        if not matched:
            continue
        summary = summary_by_solver[solver]
        count = len(matched)
        label = SOLVER_LABELS[solver]
        for backend in ("dbscan", "kmeans"):
            successes = sum(row[f"{backend}_success"] for row in matched)
            iterations = [float(row[f"{backend}_iter"]) for row in matched]
            per_step_clustering = [
                float(row[f"{backend}_clustering_s_per_step"])
                for row in matched
            ]
            per_step_total = [
                float(row[f"{backend}_total_s_per_step"])
                for row in matched
            ]
            total_elapsed = [
                float(row[f"{backend}_total_s"]) for row in matched
            ]
            total_clustering = [
                step_time * iterations[index]
                for index, step_time in enumerate(per_step_clustering)
            ]
            rows.append({
                "source": "backend_comparison.csv",
                "group": "backend_comparison",
                "variant": f"{label} ({backend.upper()})",
                "num_rows": str(count),
                "num_runs": str(count),
                "num_success": str(successes),
                "success_rate": fmt(successes / count),
                "avg_iter": fmt(mean(iterations)),
                "avg_elapsed": fmt(mean(total_elapsed)),
                "avg_clustering": fmt(mean(total_clustering)),
                "backend": backend,
                "comparison_solver": solver,
                "matched_runs": str(count),
                "mean_clustering_s_per_step": fmt(mean(per_step_clustering)),
                "mean_total_s_per_step": fmt(mean(per_step_total)),
                "clustering_speedup_dbscan_over_kmeans":
                    summary["clustering_speedup_dbscan_over_kmeans"],
                "total_step_speedup_dbscan_over_kmeans":
                    summary["total_step_speedup_dbscan_over_kmeans"],
                "success_rate_delta_kmeans_minus_dbscan":
                    summary["success_rate_delta_kmeans_minus_dbscan"],
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbscan-dir", required=True, type=Path)
    parser.add_argument("--kmeans-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--aggregate", type=Path,
        help="Add DBSCAN/K-means rows to an aggregate_timing_success CSV",
    )
    args = parser.parse_args()

    detail_rows = []
    summaries = []
    for solver, filename in SOLVERS.items():
        dbscan = read_rows(args.dbscan_dir / filename)
        kmeans = read_rows(args.kmeans_dir / filename)
        if not dbscan and not kmeans:
            continue
        keys = sorted(set(dbscan) & set(kmeans))
        for start_case, map_id in keys:
            drow, krow = dbscan[(start_case, map_id)], kmeans[(start_case, map_id)]
            dcluster = per_step(drow, "elapsed_clustering")
            kcluster = per_step(krow, "elapsed_clustering")
            dtotal = per_step(drow, "elapsed")
            ktotal = per_step(krow, "elapsed")
            detail_rows.append({
                "solver": solver,
                "start_case": start_case,
                "map": map_id,
                "dbscan_success": int(flag(drow, "is_success")),
                "kmeans_success": int(flag(krow, "is_success")),
                "dbscan_iter": int(number(drow, "iter")),
                "kmeans_iter": int(number(krow, "iter")),
                "dbscan_clustering_s_per_step": fmt(dcluster),
                "kmeans_clustering_s_per_step": fmt(kcluster),
                "clustering_speedup_dbscan_over_kmeans":
                    fmt(dcluster / kcluster) if kcluster > 0 else "nan",
                "dbscan_total_s_per_step": fmt(dtotal),
                "kmeans_total_s_per_step": fmt(ktotal),
                "total_step_speedup_dbscan_over_kmeans":
                    fmt(dtotal / ktotal) if ktotal > 0 else "nan",
                "dbscan_total_s": fmt(number(drow, "elapsed")),
                "kmeans_total_s": fmt(number(krow, "elapsed")),
            })

        matched = [row for row in detail_rows if row["solver"] == solver]
        dcluster_values = [float(row["dbscan_clustering_s_per_step"])
                           for row in matched]
        kcluster_values = [float(row["kmeans_clustering_s_per_step"])
                           for row in matched]
        dtotal_values = [float(row["dbscan_total_s_per_step"])
                         for row in matched]
        ktotal_values = [float(row["kmeans_total_s_per_step"])
                         for row in matched]
        dcluster_mean = mean(dcluster_values)
        kcluster_mean = mean(kcluster_values)
        dtotal_mean = mean(dtotal_values)
        ktotal_mean = mean(ktotal_values)
        count = len(matched)
        dbscan_success = sum(row["dbscan_success"] for row in matched)
        kmeans_success = sum(row["kmeans_success"] for row in matched)
        summaries.append({
            "solver": solver,
            "matched_runs": count,
            "dbscan_success_rate": fmt(dbscan_success / count) if count else "nan",
            "kmeans_success_rate": fmt(kmeans_success / count) if count else "nan",
            "success_rate_delta_kmeans_minus_dbscan":
                fmt((kmeans_success - dbscan_success) / count) if count else "nan",
            "dbscan_mean_clustering_s_per_step": fmt(dcluster_mean),
            "kmeans_mean_clustering_s_per_step": fmt(kcluster_mean),
            "clustering_speedup_dbscan_over_kmeans":
                fmt(dcluster_mean / kcluster_mean) if kcluster_mean > 0 else "nan",
            "dbscan_mean_total_s_per_step": fmt(dtotal_mean),
            "kmeans_mean_total_s_per_step": fmt(ktotal_mean),
            "total_step_speedup_dbscan_over_kmeans":
                fmt(dtotal_mean / ktotal_mean) if ktotal_mean > 0 else "nan",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    detail_fields = [
        "solver", "start_case", "map", "dbscan_success", "kmeans_success",
        "dbscan_iter", "kmeans_iter", "dbscan_clustering_s_per_step",
        "kmeans_clustering_s_per_step",
        "clustering_speedup_dbscan_over_kmeans", "dbscan_total_s_per_step",
        "kmeans_total_s_per_step", "total_step_speedup_dbscan_over_kmeans",
        "dbscan_total_s", "kmeans_total_s",
    ]
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_fields = list(summaries[0].keys()) if summaries else [
        "solver", "matched_runs"
    ]
    with args.summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    if args.aggregate is not None:
        write_comparison_aggregate(args.aggregate, detail_rows, summaries)


if __name__ == "__main__":
    main()
