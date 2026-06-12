#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_bool_int(value):
    if isinstance(value, (int, float)):
        return value != 0 and not math.isnan(value)
    return str(value).strip() in {"1", "1.0", "true", "True"}


def read_summary(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"method", "notes"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = math.nan
    return rows


def read_trajectories(path):
    data = {}
    segments = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            kind = row["trajectory"]
            idx = int(row["index"])
            point = (
                int(row["t"]),
                float(row["x"]),
                float(row["y"]),
                float(row["theta"]),
            )
            if kind == "segment":
                segments.setdefault((method, idx), []).append(point)
            else:
                data.setdefault(method, []).append(point)
    for series in data.values():
        series.sort(key=lambda p: p[0])
    for series in segments.values():
        series.sort(key=lambda p: p[0])
    return data, segments


def read_waypoints(path):
    waypoints = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            waypoints.append((float(row["x"]), float(row["y"]), float(row["theta"])))
    return waypoints


def add_world(ax):
    length = 15.0
    width = 3.0
    wall = 0.18
    gap_half = 0.52
    portals = [(3.0, 0.95), (6.0, 2.05), (9.0, 0.95), (12.0, 2.05)]

    rects = [
        (-0.5, -0.5, length + 1.0, 0.5),
        (-0.5, width, length + 1.0, 0.5),
        (-0.5, -0.5, 0.5, width + 1.0),
        (length, -0.5, 0.5, width + 1.0),
    ]
    for x, gy in portals:
        lower_h = max(0.0, gy - gap_half)
        upper_y = gy + gap_half
        upper_h = max(0.0, width - upper_y)
        if lower_h > 0.0:
            rects.append((x - wall * 0.5, 0.0, wall, lower_h))
        if upper_h > 0.0:
            rects.append((x - wall * 0.5, upper_y, wall, upper_h))

    for x, y, w, h in rects:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="#2f3437",
                edgecolor="#1f2326",
                linewidth=0.8,
                alpha=0.88,
            )
        )

    ax.set_xlim(-0.2, 15.2)
    ax.set_ylim(-0.2, 3.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="#d9dee2", linewidth=0.45, alpha=0.7)


def plot_timing(summary, out_dir):
    methods = [row["method"] for row in summary]
    measured = [row["measured_wall_s"] * 1000.0 for row in summary]
    effective = [row["effective_parallel_wall_s"] * 1000.0 for row in summary]
    success = [parse_bool_int(row["success"]) for row in summary]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = list(range(len(methods)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], measured, width, label="measured wall", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], effective, width, label="effective parallel wall", color="#f58518")

    for i, ok in enumerate(success):
        marker = "success" if ok else "fail"
        ymax = max(measured[i], effective[i])
        ax.text(i, ymax * 1.03 if ymax > 0 else 0.1, marker, ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("time [ms]")
    ax.set_title("WMRobot waypoint-parallel runtime")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9dee2", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "wmrobot_waypoint_timing.png", dpi=180)
    plt.close(fig)


def plot_breakdown(summary, out_dir):
    methods = [row["method"] for row in summary]
    fields = [
        ("rollout_s", "rollout", "#4c78a8"),
        ("clustering_s", "clustering", "#54a24b"),
        ("connection_s", "connection", "#e45756"),
        ("guide_s", "guide", "#b279a2"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = list(range(len(methods)))
    bottoms = [0.0] * len(methods)
    for key, label, color in fields:
        values = [row[key] * 1000.0 for row in summary]
        ax.bar(x, values, bottom=bottoms, label=label, color=color)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("accumulated solver time [ms]")
    ax.set_title("WMRobot solver-time breakdown")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#d9dee2", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "wmrobot_waypoint_runtime_breakdown.png", dpi=180)
    plt.close(fig)


def plot_paths(paths, segments, waypoints, summary, out_dir):
    colors = {
        "MPPI": "#4c78a8",
        "Cluster-MPPI": "#54a24b",
        "BiC-MPPI": "#e45756",
        "Waypoint-Parallel BiC": "#f58518",
    }
    success = {row["method"]: parse_bool_int(row["success"]) for row in summary}

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    add_world(ax)

    for (_, idx), series in sorted(segments.items(), key=lambda item: item[0][1]):
        xs = [p[1] for p in series]
        ys = [p[2] for p in series]
        label = "waypoint segment BiC" if idx == 0 else None
        ax.plot(xs, ys, color="#8a8f94", linewidth=1.0, linestyle="--", alpha=0.55, label=label)

    for method, series in paths.items():
        xs = [p[1] for p in series]
        ys = [p[2] for p in series]
        label = f"{method} ({'success' if success.get(method) else 'fail'})"
        ax.plot(xs, ys, color=colors.get(method, "#333333"), linewidth=2.0, label=label)
        if xs:
            ax.scatter(xs[-1], ys[-1], s=24, color=colors.get(method, "#333333"), zorder=5)

    if waypoints:
        wx = [p[0] for p in waypoints]
        wy = [p[1] for p in waypoints]
        ax.scatter(wx, wy, marker="o", s=22, color="#111111", zorder=6, label="waypoints")
        ax.scatter(wx[0], wy[0], marker="s", s=52, color="#111111", zorder=6)
        ax.scatter(wx[-1], wy[-1], marker="*", s=90, color="#111111", zorder=6)

    ax.set_title("WMRobot waypoint-parallel trajectories")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "wmrobot_waypoint_paths.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="results/wmrobot_waypoint_parallel",
        help="Directory containing summary.csv, trajectories.csv, waypoints.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.dir)
    summary = read_summary(out_dir / "summary.csv")
    paths, segments = read_trajectories(out_dir / "trajectories.csv")
    waypoints = read_waypoints(out_dir / "waypoints.csv")

    plot_timing(summary, out_dir)
    plot_breakdown(summary, out_dir)
    plot_paths(paths, segments, waypoints, summary, out_dir)

    print(out_dir / "wmrobot_waypoint_timing.png")
    print(out_dir / "wmrobot_waypoint_runtime_breakdown.png")
    print(out_dir / "wmrobot_waypoint_paths.png")


if __name__ == "__main__":
    main()
