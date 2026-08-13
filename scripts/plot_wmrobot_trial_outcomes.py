#!/usr/bin/env python3
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result" / "wmrobot"
FILES = {
    "MPPI": RESULT_DIR / "mppi" / "result_mppi.csv",
    "Log-MPPI": RESULT_DIR / "log_mppi" / "result_log_mppi.csv",
    "Cluster-DB": RESULT_DIR / "cluster_mppi_dbscan" / "result_cluster_mppi.csv",
    "Cluster-KM": RESULT_DIR / "cluster_mppi_kmeans" / "result_cluster_mppi_kmeans.csv",
    "BiC-DB": RESULT_DIR / "bi_mppi_dbscan" / "result_bi_mppi_se2.csv",
    "BiC-KM": RESULT_DIR / "bi_mppi_kmeans" / "result_bi_mppi_kmeans_se2.csv",
}
OUTCOME_COLORS = ("#285F8F", "#D66B5D", "#DDE3E9")
OUTCOME_HATCHES = (None, "...", "///")
OUTCOME_LABELS = ("Success", "Collision", "Non-collision failure")
METHOD_COLORS = {
    "MPPI": "#4E5964", "Log-MPPI": "#7B8794",
    "Cluster-DB": "#D66B5D", "Cluster-KM": "#E39A45",
    "BiC-DB": "#4A83AD", "BiC-KM": "#285F8F",
}
METHOD_MARKERS = {
    "MPPI": "o", "Log-MPPI": "X", "Cluster-DB": "^",
    "Cluster-KM": "s", "BiC-DB": "D", "BiC-KM": "*",
}
BENCHMARK_COLORS = {"WMRobot": "#285F8F", "Manipulator": "#D66B5D"}


def load_results():
    results = []
    for method, path in FILES.items():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        successful = [row for row in rows if row["is_success"] == "1"]
        collision = sum(row["is_collision"] == "1" for row in rows)
        results.append({
            "method": method,
            "trials": len(rows),
            "success": len(successful),
            "collision": collision,
            "other": len(rows) - len(successful) - collision,
            "success_pct": 100 * len(successful) / len(rows),
            "time": statistics.mean(float(row["elapsed"])
                                    for row in successful),
        })
    return results


def load_manipulator_results():
    names = {
        "mppi": "MPPI", "logmppi": "Log-MPPI",
        "clustermppi_dbscan": "Cluster-DB",
        "clustermppi_kmeans": "Cluster-KM",
        "bicmppi_dbscan": "BiC-DB", "bicmppi_kmeans": "BiC-KM",
    }
    path = RESULT_DIR.parent / "manipulator" / "manipulator_summary_success_only.csv"
    with path.open(newline="") as handle:
        rows = {row["solver"]: row for row in csv.DictReader(handle)}
    return [{
        "method": method,
        "trials": int(rows[solver]["total_scenarios"]),
        "success": int(rows[solver]["successful_scenarios"]),
        "success_pct": float(rows[solver]["success_rate_pct"]),
        "time": float(rows[solver]["mean_total_solver_time_s_success"]),
    } for solver, method in names.items()]


def plot_outcomes(ax, results, title):
    for y, row in enumerate(results):
        left = 0.0
        counts = (row["success"], row["collision"], row["other"])
        for count, color, hatch, label in zip(
                counts, OUTCOME_COLORS, OUTCOME_HATCHES, OUTCOME_LABELS):
            width = 100 * count / row["trials"]
            ax.barh(y, width, left=left, height=0.58, color=color,
                    edgecolor="white", linewidth=0.9, hatch=hatch,
                    label=label if y == 0 else None)
            if width >= 5:
                ax.text(left + width / 2, y, f"{width:.1f}%", ha="center",
                        va="center", fontsize=7.5, fontweight="bold",
                        color="white" if label != "Non-collision failure"
                        else "#26323B")
            left += width

    ax.set_yticks(range(len(results)), [row["method"] for row in results])
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Trial outcome [%]")
    ax.set_title(title, loc="left", fontweight="bold", pad=31)
    ax.axhline(3.5, color="#B9C2CA", linewidth=0.8)
    ax.grid(axis="x", color="#C8D0D7", linewidth=0.55, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#75808A")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", colors="#52606B")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3,
              frameon=False, handlelength=2.2, columnspacing=1.2,
              borderaxespad=0, fontsize=8)


def plot_tradeoff(ax, results, title, detailed=False):
    frontier = []
    best = -1.0
    for row in sorted(results, key=lambda item: item["time"]):
        if row["success_pct"] > best:
            frontier.append(row)
            best = row["success_pct"]
    ax.plot([row["time"] for row in frontier],
            [row["success_pct"] for row in frontier],
            color="#9AA6B2", linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)

    offsets = {
        "MPPI": (6, 9), "Log-MPPI": (6, -14),
        "Cluster-DB": (6, -14), "Cluster-KM": (6, 9),
        "BiC-DB": (7, 8), "BiC-KM": (-8, 9),
    }
    label_positions = {
        "MPPI": (0.55, 56.5), "Log-MPPI": (0.55, 46.0),
        "Cluster-DB": (2.60, 46.0), "Cluster-KM": (1.65, 55.0),
        "BiC-DB": (3.35, 89.0), "BiC-KM": (3.78, 99.0),
    }
    for row in results:
        method = row["method"]
        ax.scatter(row["time"], row["success_pct"], s=80 if detailed else 65,
                   marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                   edgecolor="white", linewidth=0.8, zorder=3)
        if detailed:
            xtext, ytext = label_positions[method]
            ax.annotate(f"{method}\n{row['time']:.2f} s · {row['success_pct']:.1f}%",
                        (row["time"], row["success_pct"]), xytext=(xtext, ytext),
                        textcoords="data", ha="center", va="center",
                        fontsize=7.2, fontweight="bold", color="#26323B",
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                              "edgecolor": "#D1D8DE", "linewidth": 0.6},
                        arrowprops={"arrowstyle": "-", "color": "#9AA6B2",
                                    "linewidth": 0.7})
        else:
            dx, dy = offsets[method]
            ax.annotate(method, (row["time"], row["success_pct"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center",
                        fontsize=7.5, fontweight="bold", color="#26323B")

    ax.set_xlim(0, 4.5)
    ax.set_ylim((43, 102) if detailed else (45, 100))
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Mean computation time on successful trials [s]")
    ax.set_ylabel("Success rate [%]")
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.grid(color="#C8D0D7", linewidth=0.55, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#75808A")
    ax.tick_params(colors="#52606B")


def plot_cross_benchmark_tradeoff(ax, benchmarks):
    for benchmark, results in benchmarks.items():
        for row in results:
            highlighted = row["method"].startswith("BiC-")
            ax.scatter(row["time"], row["success_pct"],
                       s=210 if highlighted else 75,
                       marker=METHOD_MARKERS[row["method"]],
                       color=BENCHMARK_COLORS[benchmark], edgecolor="white",
                       linewidth=1.4 if highlighted else 0.6,
                       alpha=1 if highlighted else 0.42,
                       zorder=4 if highlighted else 2)

    ax.set_xscale("log")
    ax.set_xlim(0.1, 13)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10],
                  ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Mean computation time on successful trials [s, log scale]")
    ax.set_ylabel("Success rate [%]")
    ax.grid(color="#C8D0D7", linewidth=0.55, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#75808A")
    ax.tick_params(colors="#52606B")

    benchmark_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=color, markeredgecolor="white",
               label=f"{name} (n={benchmarks[name][0]['trials']})")
        for name, color in BENCHMARK_COLORS.items()
    ]
    solver_handles = [
        Line2D([], [], marker=METHOD_MARKERS[name], linestyle="none",
               markersize=9 if name.startswith("BiC-") else 6.5,
               markerfacecolor="#34495E", markeredgecolor="white",
               alpha=1 if name.startswith("BiC-") else 0.42,
               label=name) for name in FILES
    ]
    first_legend = ax.legend(handles=benchmark_handles, title="Benchmark",
                             loc="upper left", frameon=False)
    ax.add_artist(first_legend)
    ax.legend(handles=solver_handles, title="Solver", loc="upper center",
              bbox_to_anchor=(0.5, -0.17), frameon=False,
              ncol=3, fontsize=7.5, borderaxespad=0)


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(RESULT_DIR / f"{stem}.{suffix}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    results = load_results()
    manipulator_results = load_manipulator_results()

    with (RESULT_DIR / "wmrobot_trial_outcomes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "trials", "success", "collision",
                         "non_collision_failure", "success_pct",
                         "collision_pct", "non_collision_failure_pct"))
        for row in results:
            total = row["trials"]
            writer.writerow((row["method"], total, row["success"],
                             row["collision"], row["other"],
                             row["success_pct"], 100 * row["collision"] / total,
                             100 * row["other"] / total))

    with (RESULT_DIR / "wmrobot_reliability_computation_tradeoff.csv").open(
            "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "success_rate_pct",
                         "mean_elapsed_success_s", "num_success", "num_trials"))
        for row in results:
            writer.writerow((row["method"], row["success_pct"], row["time"],
                             row["success"], row["trials"]))

    cross_path = RESULT_DIR / "wmrobot_manipulator_reliability_computation_tradeoff.csv"
    with cross_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("benchmark", "method", "success_rate_pct",
                         "mean_elapsed_success_s", "num_success", "num_trials"))
        for benchmark, rows in (("WMRobot", results),
                                ("Manipulator", manipulator_results)):
            for row in rows:
                writer.writerow((benchmark, row["method"], row["success_pct"],
                                 row["time"], row["success"], row["trials"]))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    plot_outcomes(ax, results, "WMRobot benchmark")
    ax.text(1, 1.105, f"n = {results[0]['trials']} trials per method",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#52606B")
    fig.tight_layout()
    save(fig, "wmrobot_trial_outcomes")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plot_tradeoff(ax, results, "Reliability–computation trade-off", detailed=True)
    fig.tight_layout()
    save(fig, "wmrobot_reliability_computation_tradeoff")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_cross_benchmark_tradeoff(
        ax, {"WMRobot": results, "Manipulator": manipulator_results})
    fig.tight_layout()
    save(fig, "wmrobot_manipulator_reliability_computation_tradeoff")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.15),
                             gridspec_kw={"width_ratios": (1.2, 1)})
    plot_outcomes(axes[0], results, "(a) Trial outcome breakdown")
    plot_tradeoff(axes[1], results,
                  "(b) Reliability–computation trade-off")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.78, wspace=0.3)
    save(fig, "wmrobot_benchmark_combined")


if __name__ == "__main__":
    main()
