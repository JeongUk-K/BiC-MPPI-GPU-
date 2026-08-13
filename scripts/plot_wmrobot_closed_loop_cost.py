#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from plot_wmrobot_cost_convergence import (
    COLORS, LINESTYLES, METHODS, RESULT_DIR,
)

# Edit these values to change only the names displayed in the legend.
LEGEND_LABELS = {
    "MPPI": "MPPI",
    "Log-MPPI": "Log-MPPI",
    "Cluster-DB": "Cluster-DB",
    "Cluster-KM": "Cluster-KM",
    "BiC-DB": "BiC-MPPI-DB (ours)",
    "BiC-KM": "BiC-MPPI-KM (ours)",
}


def load_progress(path):
    trials = defaultdict(list)
    successful = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["start_case"]), int(row["map"]))
            trials[key].append((int(row["iter"]), float(row["d_goal"])))
            if row["is_success"] == "1":
                successful.add(key)
    return trials, successful


def closed_loop_trace(rows, iterations, cumulative):
    finite = sorted(
        (iteration, distance) for iteration, distance in rows
        if math.isfinite(distance) and distance >= 0.0
    )
    if not finite:
        raise ValueError("trial has no finite goal-distance observations")
    x = np.fromiter((row[0] for row in finite), dtype=float)
    values = np.fromiter((row[1] for row in finite), dtype=float)
    if cumulative:
        values = np.cumsum(values)
    return np.interp(iterations, x, values, left=values[0], right=values[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-cost", action="store_true",
                        help="plot per-iteration state cost instead of its sum")
    parser.add_argument("--legend-fontsize", type=float, default=20.0,
                        help="legend font size in points (default: 8)")
    parser.add_argument("--axis-label-fontsize", type=float, default=11.0,
                        help="x/y-axis label font size in points (default: 9)")
    parser.add_argument("--max-iteration", type=int,
                        help="last control iteration to include")
    args = parser.parse_args()
    cumulative = not args.state_cost

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
    if args.max_iteration is not None:
        if args.max_iteration < 0:
            parser.error("--max-iteration must be non-negative")
        max_iteration = min(max_iteration, args.max_iteration)
    iterations = np.arange(max_iteration + 1, dtype=float)
    traces = {}
    statistics = {}
    for method in METHODS:
        traces[method] = {
            key: closed_loop_trace(loaded[method][0][key], iterations,
                                   cumulative)
            for key in sorted(common_success)
        }
        values = np.vstack(list(traces[method].values()))
        statistics[method] = {
            "q1": np.quantile(values, 0.25, axis=0),
            "median": np.quantile(values, 0.50, axis=0),
            "q3": np.quantile(values, 0.75, axis=0),
        }

    metric = ("closed_loop_state_cost" if args.state_cost else
              "closed_loop_cumulative_cost")
    stem = RESULT_DIR / f"wmrobot_{metric}_common_success"
    with stem.with_name(stem.name + "_trials.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "start_case", "map", "iteration", metric))
        for method, method_traces in traces.items():
            for (start_case, map_id), values in method_traces.items():
                for iteration, value in enumerate(values):
                    writer.writerow((method, start_case, map_id, iteration, value))

    aggregate_path = stem.with_suffix(".csv")
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
                linestyle=LINESTYLES[method], label=LEGEND_LABELS[method])

    ax.set_xlim(0, max_iteration)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Executed control iteration $k$",
                  fontsize=args.axis_label_fontsize)
    ax.set_ylabel((r"Closed-loop state cost $c_k$" if args.state_cost else
                   r"Cumulative closed-loop cost $J_k^{\mathrm{CL}}$"),
                  fontsize=args.axis_label_fontsize)
    ax.grid(color="#C8D0D7", linewidth=0.55, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#75808A")
    ax.tick_params(colors="#52606B")
    ax.legend(ncol=3, frameon=False, loc="upper right",
              fontsize=args.legend_fontsize,
              columnspacing=1.2, handlelength=2.5)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"common successful trials: {len(common_success)}")
    print(aggregate_path)
    print(stem.with_name(stem.name + "_trials.csv"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
