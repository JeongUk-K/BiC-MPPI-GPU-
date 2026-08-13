#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result" / "wmrobot"
METHODS = {
    "MPPI": RESULT_DIR / "mppi" / "result_mppi_progress.csv",
    "Log-MPPI": RESULT_DIR / "log_mppi" / "result_log_mppi_progress.csv",
    "Cluster-DB": RESULT_DIR / "cluster_mppi_dbscan" / "result_cluster_mppi_progress.csv",
    "Cluster-KM": RESULT_DIR / "cluster_mppi_kmeans" / "result_cluster_mppi_kmeans_progress.csv",
    "BiC-DB": RESULT_DIR / "bi_mppi_dbscan" / "result_bi_mppi_se2_progress.csv",
    "BiC-KM": RESULT_DIR / "bi_mppi_kmeans" / "result_bi_mppi_kmeans_se2_progress.csv",
}
COLORS = {
    "MPPI": "#4E5964", "Log-MPPI": "#7B8794",
    "Cluster-DB": "#D66B5D", "Cluster-KM": "#E39A45",
    "BiC-DB": "#4A83AD", "BiC-KM": "#285F8F",
}
LINESTYLES = {
    "MPPI": "-", "Log-MPPI": "--", "Cluster-DB": "--",
    "Cluster-KM": "-", "BiC-DB": "--", "BiC-KM": "-",
}
MAX_VALID_COST = 1.0e7


def load_progress(path):
    trials = defaultdict(list)
    successful = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["start_case"]), int(row["map"]))
            trials[key].append((int(row["iter"]), float(row["cost"])))
            if row["is_success"] == "1":
                successful.add(key)
    return trials, successful


def trace(rows, iterations, normalize):
    finite = sorted(
        (iteration, cost) for iteration, cost in rows
        if math.isfinite(cost) and 0.0 < cost < MAX_VALID_COST
    )
    if len(finite) < 2:
        raise ValueError("trial has fewer than two finite cost observations")
    x = np.fromiter((row[0] for row in finite), dtype=float)
    y = np.fromiter((row[1] for row in finite), dtype=float)
    if normalize:
        y /= y[0]
    return np.interp(iterations, x, y, left=y[0], right=y[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true",
                        help="plot unnormalized predicted trajectory cost")
    args = parser.parse_args()
    normalize = not args.raw

    loaded = {method: load_progress(path) for method, path in METHODS.items()}
    common_success = set.intersection(
        *(successful for _, successful in loaded.values())
    )
    if not common_success:
        raise RuntimeError("no trials were successful for every solver")

    max_iteration = max(
        iteration
        for method in METHODS
        for key in common_success
        for iteration, _ in loaded[method][0][key]
    )
    iterations = np.arange(max_iteration + 1, dtype=float)
    statistics = {}
    for method in METHODS:
        traces = np.vstack([
            trace(loaded[method][0][key], iterations, normalize)
            for key in sorted(common_success)
        ])
        statistics[method] = {
            "q1": np.quantile(traces, 0.25, axis=0),
            "median": np.quantile(traces, 0.50, axis=0),
            "q3": np.quantile(traces, 0.75, axis=0),
        }

    output_name = "wmrobot_cost_convergence_common_success"
    if args.raw:
        output_name += "_raw"
    aggregate_path = RESULT_DIR / f"{output_name}.csv"
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "iteration", "q1", "median", "q3",
                         "num_common_success_trials"))
        for method, values in statistics.items():
            for iteration in range(max_iteration + 1):
                writer.writerow((method, iteration, values["q1"][iteration],
                                 values["median"][iteration],
                                 values["q3"][iteration], len(common_success)))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.1, 3.8), constrained_layout=True)
    for method, values in statistics.items():
        color = COLORS[method]
        ax.fill_between(iterations, values["q1"], values["q3"],
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(iterations, values["median"], color=color, linewidth=1.8,
                linestyle=LINESTYLES[method], label=method)

    ax.set_yscale("log")
    ax.set_xlim(0, max_iteration)
    ax.set_xlabel("Control iteration $k$")
    ax.set_ylabel(r"Predicted trajectory cost $C_k$" if args.raw else
                  r"Normalized predicted cost $C_k / C_0$")
    ax.set_title(
        f"WMRobot predicted cost over control iterations ($n={len(common_success)}$)",
        loc="left", fontweight="bold", pad=8,
    )
    ax.grid(color="#C8D0D7", linewidth=0.55, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#75808A")
    ax.tick_params(colors="#52606B")
    ax.legend(ncol=3, frameon=False, loc="upper right",
              columnspacing=1.2, handlelength=2.5)

    stem = RESULT_DIR / output_name
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"common successful trials: {len(common_success)}")
    print(aggregate_path)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
