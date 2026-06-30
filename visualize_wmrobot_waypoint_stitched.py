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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap


SOLVERS = [
    ("mppi", "MPPI", "#4c78a8"),
    ("log_mppi", "Log-MPPI", "#b279a2"),
    ("cluster_mppi", "Cluster-MPPI", "#54a24b"),
    ("bi_mppi", "BiC-MPPI (parallel rollout)", "#e45756"),
]

PATH_FIGSIZE = (12.0, 4.8)


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


def read_vis_bin_limited(path, limit):
    trajectories = []
    if not path.exists():
        return trajectories
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            return trajectories
        n_traj, rows = struct.unpack("<ii", header)
        if n_traj <= 0 or rows <= 0:
            return trajectories
        if limit <= 0 or n_traj <= limit:
            wanted = set(range(n_traj))
        else:
            stride = max(1, math.ceil(n_traj / limit))
            wanted = set(range(0, n_traj, stride))
        for i in range(n_traj):
            raw_cols = f.read(4)
            if len(raw_cols) != 4:
                break
            (cols,) = struct.unpack("<i", raw_cols)
            count = rows * cols
            if i in wanted and len(trajectories) < limit:
                data = f.read(count * 8)
                if len(data) != count * 8:
                    break
                trajectories.append(
                    np.frombuffer(data, dtype="<f8").reshape(rows, cols).copy()
                )
            else:
                f.seek(count * 8, 1)
    return trajectories


def stride_limit(items, limit):
    if limit <= 0 or len(items) <= limit:
        return items
    stride = max(1, math.ceil(len(items) / limit))
    return items[::stride][:limit]


def plot_xy(traj):
    return traj[1], traj[0]


def add_map(ax, grid, resolution=0.1):
    rows, cols = grid.shape
    occ = (grid >= 9.5).astype(float)
    cmap = ListedColormap(["#ffffff", "#303437"])
    ax.imshow(
        occ,
        origin="lower",
        extent=[0.0, cols * resolution, 0.0, rows * resolution],
        cmap=cmap,
        interpolation="nearest",
        alpha=0.92,
        aspect="equal",
    )
    ax.set_xlim(-0.05, cols * resolution + 0.25)
    ax.set_ylim(-0.05, rows * resolution + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d3d7dc", linewidth=0.45, alpha=0.65)
    ax.set_xlabel("stitched y [m]")
    ax.set_ylabel("x [m]")


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

    fig, ax = plt.subplots(figsize=PATH_FIGSIZE)
    add_map(ax, grid)

    for solver, label, color in SOLVERS:
        path = paths.get((scenario, solver))
        if path is None or path.shape[1] == 0:
            continue
        row = summary_by_solver.get(solver, {})
        ok = bool(parse_number(row.get("success"), 0.0))
        px, py = plot_xy(path)
        ax.plot(
            px,
            py,
            color=color,
            linewidth=1.0,
            label=f"{label} ({'success' if ok else 'fail'})",
            zorder=5,
        )
        ax.scatter(px[-1], py[-1], s=28, color=color, zorder=6)

    if waypoints.shape[1] > 0:
        wx, wy = plot_xy(waypoints)
        ax.plot(
            wx,
            wy,
            color="#111111",
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            zorder=7,
        )
        ax.scatter(wx, wy, s=28, color="#111111", zorder=8)
        ax.scatter(wx[0], wy[0], marker="s", s=72, color="#111111", zorder=9)
        ax.scatter(wx[-1], wy[-1], marker="*", s=120, color="#111111", zorder=9)

    map_ids = ""
    if summary_by_solver:
        map_ids = next(iter(summary_by_solver.values())).get("map_ids", "")
    ax.set_title(f"WMRobot stitched BARN scenario {scenario:02d}  |  maps {map_ids}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)
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


def trajectory_display_limit(solver_dir, labels, max_rollouts):
    if isinstance(labels, str):
        labels = [labels]
    if "bi_mppi" in solver_dir.name and any(label != "optimal" for label in labels):
        return max(max_rollouts, 128)
    return max_rollouts


def rollout_groups(solver_dir, step_name, max_rollouts, base_color):
    step_dir = solver_dir / step_name
    specs = [
        ("rollouts", "rollouts / guide candidates", "#7f878f", "-", 0.40, 0.48, 3),
        ("guide_candidates", "rollouts / guide candidates", "#7f878f", "-", 0.40, 0.48, 3),
        ("clusters", "clustered paths", "#f58518", "-", 0.88, 0.72, 5),
        ("forward_clusters", "clustered paths", "#f58518", "-", 0.88, 0.72, 5),
        ("backward_clusters", "clustered paths", "#f58518", "-", 0.88, 0.72, 5),
        ("optimal", "optimal path", "#d62728", "-", 1.0, 1.45, 9),
    ]
    groups = []
    seen_displays = set()
    for file_label, display, color, linestyle, alpha, linewidth, zorder in specs:
        if file_label == "optimal":
            trajs = read_vis_bin(step_dir / f"{file_label}.bin")
        else:
            limit = trajectory_display_limit(solver_dir, file_label, max_rollouts)
            trajs = stride_limit(
                read_vis_bin(step_dir / f"{file_label}.bin"), limit
            )
        if trajs:
            label = display if display not in seen_displays else None
            seen_displays.add(display)
            groups.append((label, trajs, color, linestyle, alpha, linewidth, zorder))
    return groups


def gif_trajectory_groups(solver_dir, step_name, max_rollouts):
    step_dir = solver_dir / step_name
    specs = [
        (
            ["rollouts", "guide_candidates"],
            "#7f878f",
            0.40,
            0.48,
            "-",
            3,
        ),
        (
            ["clusters", "forward_clusters", "backward_clusters"],
            "#f58518",
            0.88,
            0.72,
            "-",
            5,
        ),
        (
            ["optimal"],
            "#d62728",
            1.0,
            1.5,
            "-",
            9,
        ),
    ]
    groups = []
    for labels, color, alpha, linewidth, linestyle, zorder in specs:
        trajectories = []
        group_limit = trajectory_display_limit(solver_dir, labels, max_rollouts)
        per_file_limit = group_limit
        for label in labels:
            file_path = step_dir / f"{label}.bin"
            if label == "optimal":
                trajectories.extend(read_vis_bin(file_path))
            else:
                trajectories.extend(read_vis_bin_limited(file_path, per_file_limit))
        if "optimal" not in labels:
            trajectories = stride_limit(trajectories, group_limit)
        if trajectories:
            groups.append((trajectories, color, alpha, linewidth, linestyle, zorder))
    return groups


def logged_step_numbers_for_solver(root, scenario, solver):
    steps = set()
    solver_dir = root / "vis_data" / f"stitched_{solver}_scenario_{scenario:02d}"
    for step_dir in solver_dir.glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            steps.add(int(step_dir.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def draw_waypoints(ax, waypoints):
    if waypoints.shape[1] <= 0:
        return
    wx, wy = plot_xy(waypoints)
    ax.plot(
        wx,
        wy,
        color="#111111",
        linewidth=0.9,
        linestyle=":",
        alpha=0.7,
        zorder=7,
    )
    ax.scatter(wx, wy, s=18, color="#111111", zorder=8)
    ax.scatter(
        wx[0],
        wy[0],
        marker="s",
        s=42,
        color="#111111",
        zorder=9,
    )
    ax.scatter(
        wx[-1],
        wy[-1],
        marker="*",
        s=72,
        color="#111111",
        zorder=9,
    )


def make_solver_rollout_path_gif(
    root, paths, scenario, solver_info, max_rollouts, out_path, fps
):
    solver, label, color = solver_info
    grid = read_stitched_map_csv(root, scenario)
    waypoints = read_waypoints(root, scenario)
    logged_steps = logged_step_numbers_for_solver(root, scenario, solver)
    path = paths.get((scenario, solver))
    max_path_step = 0
    if path is not None and path.shape[1] > 1:
        max_path_step = max(max_path_step, path.shape[1] - 2)
    if logged_steps:
        max_path_step = max(max_path_step, max(logged_steps))
    frame_steps = list(range(max_path_step + 1))
    if not frame_steps:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 4.2))

    def update(frame_index):
        step = frame_steps[frame_index]
        ax.clear()
        add_map(ax, grid)
        draw_waypoints(ax, waypoints)

        solver_dir = root / "vis_data" / f"stitched_{solver}_scenario_{scenario:02d}"
        step_name = f"step_{step:04d}"
        if (solver_dir / step_name).is_dir():
            for trajs, group_color, alpha, linewidth, linestyle, zorder in (
                gif_trajectory_groups(solver_dir, step_name, max_rollouts)
            ):
                for traj in trajs:
                    tx, ty = plot_xy(traj)
                    ax.plot(
                        tx,
                        ty,
                        color=group_color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        alpha=alpha,
                        zorder=zorder,
                    )

        if path is not None and path.shape[1] > 0:
            end = min(path.shape[1], step + 2)
            px, py = plot_xy(path[:, :end])
            ax.plot(
                px,
                py,
                color="#111111",
                linewidth=0.68,
                alpha=0.78,
                zorder=8,
            )
            ax.scatter(
                px[-1],
                py[-1],
                s=34,
                color="#111111",
                edgecolor="#ffffff",
                linewidth=0.5,
                zorder=12,
            )

        ax.set_title(f"{label} | scenario {scenario:02d} | step {step:04d}", fontsize=11)
        fig.tight_layout()
        return (ax,)

    anim = FuncAnimation(fig, update, frames=len(frame_steps), blit=False)
    writer = PillowWriter(fps=max(1, fps))
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    return True


def plot_rollouts(root, paths, scenario, step_selector, max_rollouts, out_path):
    grid = read_stitched_map_csv(root, scenario)
    waypoints = read_waypoints(root, scenario)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (solver, label, color) in zip(axes, SOLVERS):
        add_map(ax, grid)
        solver_dir = root / "vis_data" / f"stitched_{solver}_scenario_{scenario:02d}"
        step_name = choose_step(solver_dir, step_selector)
        if step_name:
            for (
                display,
                trajs,
                group_color,
                linestyle,
                alpha,
                linewidth,
                zorder,
            ) in rollout_groups(
                solver_dir, step_name, max_rollouts, color
            ):
                for k, traj in enumerate(trajs):
                    tx, ty = plot_xy(traj)
                    ax.plot(
                        tx,
                        ty,
                        color=group_color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=zorder,
                        label=display if k == 0 else None,
                    )

        path = paths.get((scenario, solver))
        if path is not None and path.shape[1] > 0:
            px, py = plot_xy(path)
            ax.plot(px, py, color="#111111", linewidth=0.62, alpha=0.70, zorder=8)

        if waypoints.shape[1] > 0:
            wx, wy = plot_xy(waypoints)
            ax.scatter(wx[0], wy[0], marker="s", s=34, color="#111111", zorder=8)
            ax.scatter(wx[-1], wy[-1], marker="*", s=70, color="#111111", zorder=8)

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
    parser.add_argument(
        "--rollout-scenario",
        type=int,
        default=0,
        help="kept for compatibility; rollout PNG/GIF outputs are generated for all scenarios",
    )
    parser.add_argument(
        "--rollout-step",
        default="first",
        help="first, middle, last, or a concrete step name such as step_0000",
    )
    parser.add_argument("--max-rollouts", type=int, default=24)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--no-gif", action="store_true")
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

    rollout_paths = []
    for scenario in scenarios:
        rollout_path = root / f"wmrobot_stitched_rollouts_scenario_{scenario:02d}.png"
        plot_rollouts(
            root,
            paths,
            scenario,
            args.rollout_step,
            args.max_rollouts,
            rollout_path,
        )
        rollout_paths.append(rollout_path)
    gif_paths = []
    if not args.no_gif:
        gif_dir = root / "rollout_gifs"
        for scenario in scenarios:
            for solver_info in SOLVERS:
                solver, _, _ = solver_info
                gif_path = gif_dir / f"wmrobot_stitched_{solver}_scenario_{scenario:02d}.gif"
                if make_solver_rollout_path_gif(
                    root,
                    paths,
                    scenario,
                    solver_info,
                    args.max_rollouts,
                    gif_path,
                    args.gif_fps,
                ):
                    gif_paths.append(gif_path)

    print(root / "wmrobot_stitched_aggregate.csv")
    print(root / "wmrobot_stitched_timing.png")
    for scenario in scenarios:
        print(root / f"wmrobot_stitched_paths_scenario_{scenario:02d}.png")
    for rollout_path in rollout_paths:
        print(rollout_path)
    if gif_paths:
        print(f"wrote {len(gif_paths)} solver/scenario GIFs under {root / 'rollout_gifs'}")
        for gif_path in gif_paths:
            print(gif_path)


if __name__ == "__main__":
    main()
