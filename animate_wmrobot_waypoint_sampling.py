#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visualize_wmrobot_waypoint_parallel import (
    add_world,
    parse_bool_int,
    read_summary,
    read_trajectories,
    read_waypoints,
)


COLORS = {
    "MPPI": "#4c78a8",
    "Cluster-MPPI": "#54a24b",
    "BiC-MPPI": "#e45756",
    "Waypoint-Parallel BiC": "#f58518",
}


def read_metadata(path):
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_vis_bin(path):
    trajectories = []
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            return trajectories
        n_traj, rows = struct.unpack("<ii", header)
        for _ in range(n_traj):
            cols_raw = f.read(4)
            if len(cols_raw) != 4:
                break
            (cols,) = struct.unpack("<i", cols_raw)
            count = rows * cols
            data = f.read(count * 8)
            if len(data) != count * 8:
                break
            arr = np.frombuffer(data, dtype="<f8").reshape(rows, cols).copy()
            trajectories.append(arr)
    return trajectories


def stride_limit(items, limit):
    if limit <= 0 or len(items) <= limit:
        return items
    stride = max(1, math.ceil(len(items) / limit))
    return items[::stride][:limit]


def solver_method(solver_name):
    if solver_name == "wmrobot_waypoint_mppi":
        return "MPPI"
    if solver_name == "wmrobot_waypoint_cluster_mppi":
        return "Cluster-MPPI"
    if solver_name == "wmrobot_waypoint_bic_mppi":
        return "BiC-MPPI"
    if solver_name.startswith("wmrobot_waypoint_segment_"):
        return "Waypoint-Parallel BiC"
    if solver_name == "wmrobot_waypoint_final_guide":
        return "Waypoint-Parallel BiC"
    return "Other"


def label_style(solver_name, label):
    if solver_name == "wmrobot_waypoint_mppi":
        return "MPPI rollouts", "#4c78a8", 0.28, 1.05, "-"
    if solver_name == "wmrobot_waypoint_cluster_mppi":
        if label == "clusters":
            return "Cluster representatives", "#1f7a36", 0.55, 1.25, "-"
        return "Cluster-MPPI rollouts", "#54a24b", 0.28, 1.05, "-"
    if solver_name == "wmrobot_waypoint_bic_mppi":
        if label == "guide_candidates":
            return "BiC guide candidates", "#b279a2", 0.46, 1.2, "-"
        return "BiC cluster candidates", "#e45756", 0.36, 1.05, "--"
    if solver_name.startswith("wmrobot_waypoint_segment_"):
        if label == "guide_candidates":
            return "Waypoint segment guide samples", "#f58518", 0.42, 1.2, "-"
        return "Waypoint segment clusters", "#7d858c", 0.34, 1.0, "--"
    if solver_name == "wmrobot_waypoint_final_guide":
        return "Waypoint final guide samples", "#f58518", 0.50, 1.25, "-"
    return label, "#777777", 0.25, 1.0, "-"


def parse_segment_index(solver_name):
    prefix = "wmrobot_waypoint_segment_"
    if not solver_name.startswith(prefix):
        return None
    try:
        return int(solver_name[len(prefix):])
    except ValueError:
        return None


def sample_offset(solver_name, label, trajectories, metadata):
    segment_t = int(metadata.get("segment_T", 45))
    global_t = int(metadata.get("global_T", segment_t * 5))
    cols = trajectories[0].shape[1] if trajectories else 1
    horizon = max(0, cols - 1)
    segment_idx = parse_segment_index(solver_name)

    if segment_idx is not None:
        segment_start = segment_idx * segment_t
        if label == "backward_clusters":
            return max(0, segment_start + segment_t - horizon)
        return segment_start

    if label == "backward_clusters":
        return max(0, global_t - horizon)
    return 0


def collect_sample_groups(vis_root, metadata, max_samples_per_file):
    groups = []
    labels = [
        "rollouts",
        "clusters",
        "forward_clusters",
        "backward_clusters",
        "guide_candidates",
        "optimal",
    ]

    for solver_dir in sorted(Path(vis_root).glob("wmrobot_waypoint_*")):
        step_dir = solver_dir / "step_0000"
        if not step_dir.exists():
            continue

        for label in labels:
            path = step_dir / f"{label}.bin"
            if not path.exists():
                continue
            trajectories = read_vis_bin(path)
            if label == "optimal":
                continue
            trajectories = stride_limit(trajectories, max_samples_per_file)
            if not trajectories:
                continue
            display, color, alpha, linewidth, linestyle = label_style(
                solver_dir.name, label
            )
            offset = sample_offset(solver_dir.name, label, trajectories, metadata)
            groups.append(
                {
                    "name": display,
                    "method": solver_method(solver_dir.name),
                    "solver": solver_dir.name,
                    "label": label,
                    "offset": offset,
                    "trajectories": trajectories,
                    "color": color,
                    "alpha": alpha,
                    "linewidth": linewidth,
                    "linestyle": linestyle,
                }
            )
    return groups


def series_to_arrays(series):
    if not series:
        return np.zeros((0,)), np.zeros((0,))
    xs = np.array([p[1] for p in series], dtype=float)
    ys = np.array([p[2] for p in series], dtype=float)
    return xs, ys


def draw_waypoints(ax, waypoints):
    if not waypoints:
        return
    wx = [p[0] for p in waypoints]
    wy = [p[1] for p in waypoints]
    ax.scatter(wx, wy, marker="o", s=18, color="#111111", zorder=8)
    ax.scatter(wx[0], wy[0], marker="s", s=44, color="#111111", zorder=8)
    ax.scatter(wx[-1], wy[-1], marker="*", s=84, color="#111111", zorder=8)


def draw_samples(ax, groups, frame_t, forecast_window):
    seen = set()
    for group in groups:
        local_t = frame_t - group["offset"]
        if local_t < 0:
            continue
        label = group["name"] if group["name"] not in seen else None
        seen.add(group["name"])
        for traj in group["trajectories"]:
            if traj.shape[0] < 2 or traj.shape[1] < 2:
                continue
            start = int(local_t)
            if start >= traj.shape[1] - 1:
                continue
            if forecast_window > 0:
                end = min(start + forecast_window + 1, traj.shape[1])
            else:
                end = traj.shape[1]
            if end - start < 2:
                continue
            ax.plot(
                traj[0, start:end],
                traj[1, start:end],
                color=group["color"],
                alpha=group["alpha"],
                linewidth=group["linewidth"],
                linestyle=group["linestyle"],
                label=label,
                zorder=2,
            )
            label = None


def draw_plans(ax, paths, summary, frame_t):
    success = {row["method"]: parse_bool_int(row["success"]) for row in summary}
    for method, series in paths.items():
        xs, ys = series_to_arrays(series)
        if xs.size < 2:
            continue
        upto = min(frame_t + 1, xs.size)
        if upto < 2:
            continue
        linewidth = 2.7 if method == "Waypoint-Parallel BiC" else 1.9
        alpha = 1.0 if method == "Waypoint-Parallel BiC" else 0.86
        label = f"{method} ({'success' if success.get(method) else 'fail'})"
        ax.plot(
            xs[:upto],
            ys[:upto],
            color=COLORS.get(method, "#333333"),
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=6,
        )
        ax.scatter(
            xs[upto - 1],
            ys[upto - 1],
            s=20 if method != "Waypoint-Parallel BiC" else 34,
            color=COLORS.get(method, "#333333"),
            zorder=7,
        )


def render_variant_frame(
    frame_t,
    max_t,
    groups,
    paths,
    summary,
    waypoints,
    frame_path,
    dpi,
    forecast_window,
):
    methods = ["MPPI", "Cluster-MPPI", "BiC-MPPI", "Waypoint-Parallel BiC"]
    success = {row["method"]: parse_bool_int(row["success"]) for row in summary}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    fig.suptitle(f"Waypoint-aligned sampled rollouts, step {frame_t:03d}/{max_t:03d}")

    for ax, method in zip(axes.ravel(), methods):
        add_world(ax)
        draw_waypoints(ax, waypoints)
        method_groups = [group for group in groups if group["method"] == method]
        draw_samples(ax, method_groups, frame_t, forecast_window)
        if method in paths:
            draw_plans(ax, {method: paths[method]}, summary, frame_t)
        status = "success" if success.get(method) else "fail"
        ax.set_title(f"{method} ({status})", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        dedup = {}
        for handle, label in zip(handles, labels):
            if label and label not in dedup:
                dedup[label] = handle
        ax.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper right",
            frameon=True,
            fontsize=6.5,
            framealpha=0.76,
        )

    fig.text(
        0.5,
        0.025,
        "transparent lines: sampled rollout forecasts at the current timestep; bold line: selected trajectory prefix",
        ha="center",
        fontsize=9,
        color="#222222",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def render_combined_frame(
    frame_t,
    max_t,
    groups,
    paths,
    summary,
    waypoints,
    frame_path,
    dpi,
    forecast_window,
):
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    add_world(ax)
    draw_waypoints(ax, waypoints)
    draw_samples(ax, groups, frame_t, forecast_window)
    draw_plans(ax, paths, summary, frame_t)
    ax.set_title(f"Waypoint-aligned sampled rollouts, step {frame_t:03d}/{max_t:03d}")
    ax.text(
        0.01,
        0.98,
        "faint lines: sampled rollout forecasts at the current timestep; bold lines: selected trajectories",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 3},
    )
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def render_frames(
    out_dir,
    vis_root,
    frame_stride,
    max_samples_per_file,
    dpi,
    layout,
    forecast_window,
):
    metadata = read_metadata(out_dir / "metadata.json")
    summary = read_summary(out_dir / "summary.csv")
    paths, _ = read_trajectories(out_dir / "trajectories.csv")
    waypoints = read_waypoints(out_dir / "waypoints.csv")
    groups = collect_sample_groups(vis_root, metadata, max_samples_per_file)

    if not groups:
        raise RuntimeError(f"No sampled trajectory logs found under {vis_root}")

    max_t = 0
    for series in paths.values():
        if series:
            max_t = max(max_t, max(p[0] for p in series))

    frames_dir = out_dir / "sampling_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    frame_paths = []
    for frame_t in range(0, max_t + 1, frame_stride):
        frame_path = frames_dir / f"frame_{frame_t:04d}.png"
        if layout == "combined":
            render_combined_frame(
                frame_t, max_t, groups, paths, summary, waypoints, frame_path, dpi,
                forecast_window,
            )
        else:
            render_variant_frame(
                frame_t, max_t, groups, paths, summary, waypoints, frame_path, dpi,
                forecast_window,
            )
        frame_paths.append(frame_path)

    return frame_paths


def write_gif(frame_paths, gif_path, duration_ms):
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    if not images:
        raise RuntimeError("No frames generated")
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    for image in images:
        image.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/wmrobot_waypoint_parallel")
    parser.add_argument("--vis-root", default="build/vis_data")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples-per-file", type=int, default=45)
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--layout", choices=["variants", "combined"], default="variants")
    parser.add_argument(
        "--forecast-window",
        type=int,
        default=45,
        help="Future rollout steps shown at each frame; <=0 shows the full remaining rollout.",
    )
    args = parser.parse_args()

    out_dir = Path(args.dir)
    vis_root = Path(args.vis_root)
    frame_paths = render_frames(
        out_dir,
        vis_root,
        max(1, args.stride),
        args.max_samples_per_file,
        args.dpi,
        args.layout,
        args.forecast_window,
    )
    gif_path = out_dir / "wmrobot_waypoint_sampling.gif"
    write_gif(frame_paths, gif_path, args.duration_ms)

    print(gif_path)
    print(out_dir / "sampling_frames")
    print(f"frames={len(frame_paths)}")


if __name__ == "__main__":
    main()
