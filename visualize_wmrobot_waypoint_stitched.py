#!/usr/bin/env python3
import argparse
import csv
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


SOLVERS = [
    ("mppi", "MPPI", "#4c78a8"),
    ("log_mppi", "Log-MPPI", "#b279a2"),
    ("cluster_mppi", "Cluster-MPPI", "#54a24b"),
    ("bi_mppi", "BiC-MPPI (parallel rollout)", "#e45756"),
]


def parse_number(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_summary(root):
    rows = []
    for solver, label, color in SOLVERS:
        for row in read_csv_rows(root / solver / "summary.csv"):
            row["solver"] = solver
            row["solver_label"] = row.get("solver_label") or label
            row["color"] = color
            for key in [
                "scenario",
                "success",
                "is_collision",
                "iter",
                "reached_waypoints",
                "total_waypoints",
                "elapsed",
                "elapsed_rollout",
                "elapsed_clustering",
                "elapsed_connection",
                "elapsed_guide",
                "final_error",
                "path_points",
            ]:
                row[key] = parse_number(row.get(key))
            rows.append(row)
    return rows


def read_paths(root):
    paths = {}
    for solver, _, _ in SOLVERS:
        rows = read_csv_rows(root / solver / "paths.csv")
        grouped = {}
        for row in rows:
            scenario = int(float(row["scenario"]))
            grouped.setdefault(scenario, []).append(
                (
                    int(float(row["step"])),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["theta"]),
                )
            )
        for scenario, points in grouped.items():
            points.sort(key=lambda p: p[0])
            arr = np.array([[p[1], p[2], p[3]] for p in points], dtype=float).T
            paths[(scenario, solver)] = arr
    return paths


def read_waypoints(root, scenario):
    for solver, _, _ in SOLVERS:
        rows = read_csv_rows(root / solver / "waypoints.csv")
        points = []
        for row in rows:
            if int(float(row["scenario"])) == scenario:
                points.append(
                    (
                        int(float(row["index"])),
                        float(row["x"]),
                        float(row["y"]),
                        float(row["theta"]),
                    )
                )
        if points:
            points.sort(key=lambda p: p[0])
            return np.array([[p[1], p[2], p[3]] for p in points], dtype=float).T
    return np.zeros((3, 0))


def read_stitched_map_csv(root, scenario):
    name = f"scenario_{scenario:02d}.csv"
    for solver, _, _ in SOLVERS:
        path = root / solver / "stitched_maps" / name
        if not path.exists():
            continue
        rows = []
        with path.open(newline="") as f:
            for row in csv.reader(f):
                if row:
                    rows.append([float(v) for v in row])
        if rows:
            return np.array(rows, dtype=float)
    raise FileNotFoundError(f"no stitched map CSV found for scenario {scenario}")


def read_vis_bin(path):
    trajectories = []
    if not path.exists():
        return trajectories
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            return trajectories
        n_traj, rows = struct.unpack("<ii", header)
        for _ in range(n_traj):
            raw_cols = f.read(4)
            if len(raw_cols) != 4:
                break
            (cols,) = struct.unpack("<i", raw_cols)
            count = rows * cols
            data = f.read(count * 8)
            if len(data) != count * 8:
                break
            trajectories.append(
                np.frombuffer(data, dtype="<f8").reshape(rows, cols).copy()
            )
    return trajectories


def stride_limit(items, limit):
    if limit <= 0 or len(items) <= limit:
        return items
    stride = max(1, math.ceil(len(items) / limit))
    return items[::stride][:limit]


def add_map(ax, grid, resolution=0.1):
    rows, cols = grid.shape
    occ = (grid.T >= 9.5).astype(float)
    cmap = ListedColormap(["#ffffff", "#303437"])
    ax.imshow(
        occ,
        origin="lower",
        extent=[0.0, rows * resolution, 0.0, cols * resolution],
        cmap=cmap,
        interpolation="nearest",
        alpha=0.92,
        aspect="equal",
    )
    ax.set_xlim(-0.05, rows * resolution + 0.05)
    ax.set_ylim(-0.05, cols * resolution + 0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d3d7dc", linewidth=0.45, alpha=0.65)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def write_aggregate(summary, out_path):
    grouped = {}
    for row in summary:
        grouped.setdefault(row["solver"], []).append(row)

    fields = [
        "solver",
        "solver_label",
        "scenarios",
        "success_rate",
        "mean_elapsed",
        "mean_iter",
        "mean_final_error",
        "mean_reached_waypoints",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for solver, label, _ in SOLVERS:
            rows = grouped.get(solver, [])
            if not rows:
                continue
            writer.writerow(
                {
                    "solver": solver,
                    "solver_label": label,
                    "scenarios": len(rows),
                    "success_rate": np.nanmean([r["success"] for r in rows]),
                    "mean_elapsed": np.nanmean([r["elapsed"] for r in rows]),
                    "mean_iter": np.nanmean([r["iter"] for r in rows]),
                    "mean_final_error": np.nanmean([r["final_error"] for r in rows]),
                    "mean_reached_waypoints": np.nanmean(
                        [r["reached_waypoints"] for r in rows]
                    ),
                }
            )


def plot_timing(summary, out_path):
    grouped = {solver: [] for solver, _, _ in SOLVERS}
    for row in summary:
        grouped.setdefault(row["solver"], []).append(row)

    labels = []
    colors = []
    elapsed = []
    rollout = []
    clustering = []
    connection = []
    guide = []
    success = []
    for solver, label, color in SOLVERS:
        rows = grouped.get(solver, [])
        if not rows:
            continue
        labels.append(label)
        colors.append(color)
        elapsed.append(np.nanmean([r["elapsed"] for r in rows]) * 1000.0)
        rollout.append(np.nanmean([r["elapsed_rollout"] for r in rows]) * 1000.0)
        clustering.append(
            np.nanmean([r["elapsed_clustering"] for r in rows]) * 1000.0
        )
        connection.append(
            np.nanmean([r["elapsed_connection"] for r in rows]) * 1000.0
        )
        guide.append(np.nanmean([r["elapsed_guide"] for r in rows]) * 1000.0)
        success.append(np.nanmean([r["success"] for r in rows]))

    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for values, name, color in [
        (rollout, "rollout", "#4c78a8"),
        (clustering, "clustering", "#54a24b"),
        (connection, "connection", "#e45756"),
        (guide, "guide", "#b279a2"),
    ]:
        ax.bar(x, values, bottom=bottom, label=name, color=color)
        bottom += np.nan_to_num(np.array(values))

    for i, rate in enumerate(success):
        ymax = bottom[i] if bottom[i] > 0 else elapsed[i]
        ax.text(i, ymax * 1.03 if ymax > 0 else 1.0, f"{rate:.0%}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("mean accumulated solver time [ms]")
    ax.set_title("WMRobot 5-map waypoint runtime")
    ax.grid(axis="y", color="#d3d7dc", linewidth=0.55, alpha=0.8)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_paths_for_scenario(root, paths, summary, scenario, out_path):
    grid = read_stitched_map_csv(root, scenario)
    waypoints = read_waypoints(root, scenario)
    summary_by_solver = {
        row["solver"]: row for row in summary if int(row["scenario"]) == scenario
    }

    fig, ax = plt.subplots(figsize=(7.2, 12.0))
    add_map(ax, grid)

    for solver, label, color in SOLVERS:
        path = paths.get((scenario, solver))
        if path is None or path.shape[1] == 0:
            continue
        row = summary_by_solver.get(solver, {})
        ok = bool(parse_number(row.get("success"), 0.0))
        elapsed = parse_number(row.get("elapsed"), math.nan)
        ax.plot(
            path[0],
            path[1],
            color=color,
            linewidth=2.0,
            label=f"{label} ({'success' if ok else 'fail'}, {elapsed:.2f}s)",
            zorder=5,
        )
        ax.scatter(path[0, -1], path[1, -1], s=28, color=color, zorder=6)

    if waypoints.shape[1] > 0:
        ax.plot(
            waypoints[0],
            waypoints[1],
            color="#111111",
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            zorder=7,
        )
        ax.scatter(waypoints[0], waypoints[1], s=28, color="#111111", zorder=8)
        ax.scatter(waypoints[0, 0], waypoints[1, 0], marker="s", s=72, color="#111111", zorder=9)
        ax.scatter(waypoints[0, -1], waypoints[1, -1], marker="*", s=120, color="#111111", zorder=9)

    map_ids = ""
    if summary_by_solver:
        map_ids = next(iter(summary_by_solver.values())).get("map_ids", "")
    ax.set_title(f"WMRobot stitched BARN scenario {scenario:02d}  |  maps {map_ids}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def choose_step(solver_dir, requested):
    steps = sorted([p.name for p in solver_dir.glob("step_*") if p.is_dir()])
    if not steps:
        return ""
    if requested == "first":
        return steps[0]
    if requested == "middle":
        return steps[len(steps) // 2]
    if requested == "last":
        return steps[-1]
    if requested in steps:
        return requested
    return steps[0]


def rollout_groups(solver_dir, step_name, max_rollouts, base_color):
    step_dir = solver_dir / step_name
    specs = [
        ("rollouts", "rollouts", base_color, "-", 0.23),
        ("clusters", "clusters", "#1f7a36", "-", 0.55),
        ("forward_clusters", "forward clusters", "#e45756", "--", 0.45),
        ("backward_clusters", "backward clusters", "#4c78a8", "--", 0.45),
        ("guide_candidates", "guide candidates", "#b279a2", "-", 0.45),
    ]
    groups = []
    for file_label, display, color, linestyle, alpha in specs:
        trajs = stride_limit(read_vis_bin(step_dir / f"{file_label}.bin"), max_rollouts)
        if trajs:
            groups.append((display, trajs, color, linestyle, alpha))
    return groups


def plot_rollouts(root, paths, scenario, step_selector, max_rollouts, out_path):
    grid = read_stitched_map_csv(root, scenario)
    waypoints = read_waypoints(root, scenario)

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 13.0), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (solver, label, color) in zip(axes, SOLVERS):
        add_map(ax, grid)
        solver_dir = root / "vis_data" / f"stitched_{solver}_scenario_{scenario:02d}"
        step_name = choose_step(solver_dir, step_selector)
        if step_name:
            for display, trajs, group_color, linestyle, alpha in rollout_groups(
                solver_dir, step_name, max_rollouts, color
            ):
                for k, traj in enumerate(trajs):
                    ax.plot(
                        traj[0],
                        traj[1],
                        color=group_color,
                        linestyle=linestyle,
                        linewidth=0.85,
                        alpha=alpha,
                        label=display if k == 0 else None,
                    )

        path = paths.get((scenario, solver))
        if path is not None and path.shape[1] > 0:
            ax.plot(path[0], path[1], color="#111111", linewidth=1.4, alpha=0.9)

        if waypoints.shape[1] > 0:
            ax.scatter(waypoints[0, 0], waypoints[1, 0], marker="s", s=34, color="#111111", zorder=8)
            ax.scatter(waypoints[0, -1], waypoints[1, -1], marker="*", s=70, color="#111111", zorder=8)

        suffix = f" ({step_name})" if step_name else " (no rollout log)"
        ax.set_title(f"{label}{suffix}")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=7, loc="upper right")

    for ax in axes[len(SOLVERS) :]:
        ax.axis("off")

    fig.suptitle(f"WMRobot stitched BARN sampled rollouts, scenario {scenario:02d}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/wmrobot_waypoint_stitched")
    parser.add_argument("--rollout-scenario", type=int, default=0)
    parser.add_argument(
        "--rollout-step",
        default="first",
        help="first, middle, last, or a concrete step name such as step_0000",
    )
    parser.add_argument("--max-rollouts", type=int, default=24)
    args = parser.parse_args()

    root = Path(args.dir)
    summary = read_summary(root)
    if not summary:
        raise FileNotFoundError(f"no solver summary.csv files found under {root}")
    paths = read_paths(root)
    scenarios = sorted({int(row["scenario"]) for row in summary})

    write_aggregate(summary, root / "wmrobot_stitched_aggregate.csv")
    plot_timing(summary, root / "wmrobot_stitched_timing.png")

    for scenario in scenarios:
        plot_paths_for_scenario(
            root,
            paths,
            summary,
            scenario,
            root / f"wmrobot_stitched_paths_scenario_{scenario:02d}.png",
        )

    rollout_scenario = args.rollout_scenario
    if rollout_scenario not in scenarios:
        rollout_scenario = scenarios[0]
    plot_rollouts(
        root,
        paths,
        rollout_scenario,
        args.rollout_step,
        args.max_rollouts,
        root / f"wmrobot_stitched_rollouts_scenario_{rollout_scenario:02d}.png",
    )

    print(root / "wmrobot_stitched_aggregate.csv")
    print(root / "wmrobot_stitched_timing.png")
    for scenario in scenarios:
        print(root / f"wmrobot_stitched_paths_scenario_{scenario:02d}.png")
    print(root / f"wmrobot_stitched_rollouts_scenario_{rollout_scenario:02d}.png")


if __name__ == "__main__":
    main()
