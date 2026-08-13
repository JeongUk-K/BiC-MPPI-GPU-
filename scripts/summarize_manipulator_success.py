#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result" / "manipulator"
SOLVERS = (
    "mppi", "logmppi", "clustermppi_kmeans", "clustermppi_dbscan",
    "bicmppi_kmeans", "bicmppi_dbscan",
)
FIELDS = (
    "solver", "solver_label", "total_scenarios", "reached_scenarios",
    "successful_scenarios", "failed_scenarios", "success_rate_pct",
    "collision_scenarios", "mean_total_solver_time_s_all",
    "mean_wall_time_s_all", "mean_solve_iterations_all",
    "mean_total_solver_time_s_success",
    "mean_solver_time_per_iteration_ms_success", "mean_wall_time_s_success",
    "mean_solve_iterations_success", "total_solver_time_success_q1_s",
    "total_solver_time_success_q2_s", "total_solver_time_success_q3_s",
    "mean_final_q_error_success", "mean_final_ee_error_success",
    "mean_final_qdot_norm_success", "mean_collision_count_success",
)


def mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def fmt(value):
    return f"{value:.10g}" if isinstance(value, float) else value


def main():
    summaries = []
    expected_count = None
    for solver in SOLVERS:
        path = RESULT_DIR / solver / f"manipulator_random_pose_benchmark_{solver}_stats.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"no benchmark rows in {path}")
        if expected_count is None:
            expected_count = len(rows)
        if len(rows) != expected_count:
            raise RuntimeError(
                f"incomplete result: {solver} has {len(rows)} rows; expected {expected_count}"
            )

        successes = [row for row in rows if row["success"] == "1"]
        if not successes:
            raise RuntimeError(f"no successful scenarios for {solver}")
        elapsed = [float(row["total_solver_elapsed_s"]) for row in successes]
        q1, q2, q3 = np.quantile(elapsed, (0.25, 0.50, 0.75))
        summaries.append({
            "solver": solver,
            "solver_label": rows[0]["solver_label"],
            "total_scenarios": len(rows),
            "reached_scenarios": sum(row["reached"] == "1" for row in rows),
            "successful_scenarios": len(successes),
            "failed_scenarios": len(rows) - len(successes),
            "success_rate_pct": 100.0 * len(successes) / len(rows),
            "collision_scenarios": sum(float(row["collision_count"]) > 0 for row in rows),
            "mean_total_solver_time_s_all": mean(rows, "total_solver_elapsed_s"),
            "mean_wall_time_s_all": mean(rows, "wall_elapsed_s"),
            "mean_solve_iterations_all": mean(rows, "solve_iterations"),
            "mean_total_solver_time_s_success": mean(successes, "total_solver_elapsed_s"),
            "mean_solver_time_per_iteration_ms_success":
                1000.0 * mean(successes, "mean_solver_elapsed_s"),
            "mean_wall_time_s_success": mean(successes, "wall_elapsed_s"),
            "mean_solve_iterations_success": mean(successes, "solve_iterations"),
            "total_solver_time_success_q1_s": float(q1),
            "total_solver_time_success_q2_s": float(q2),
            "total_solver_time_success_q3_s": float(q3),
            "mean_final_q_error_success": mean(successes, "final_q_error"),
            "mean_final_ee_error_success": mean(successes, "final_ee_error"),
            "mean_final_qdot_norm_success": mean(successes, "final_qdot_norm"),
            "mean_collision_count_success": mean(successes, "collision_count"),
        })

    output = RESULT_DIR / "manipulator_summary_success_only.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: fmt(value) for key, value in summary.items()})
    print(output)


if __name__ == "__main__":
    main()
