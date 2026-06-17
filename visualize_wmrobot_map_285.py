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
    ("mppi", "MPPI", "result_mppi.csv", "#4c78a8"),
    ("log_mppi", "Log-MPPI", "result_log_mppi.csv", "#b279a2"),
    ("cluster_mppi", "Cluster-MPPI", "result_cluster_mppi.csv", "#54a24b"),
    ("bi_mppi", "BiC-MPPI", "result_bi_mppi.csv", "#e45756"),
]


def read_meta(path):
    meta = {}
    if not path.exists():
        return meta
    with path.open() as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            meta[key] = value
    return meta


def parse_vector(text):
    return np.array([float(v) for v in text.split()], dtype=float)


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
            cols_raw = f.read(4)
            if len(cols_raw) != 4:
                break
            (cols,) = struct.unpack("<i", cols_raw)
            count = rows * cols
            data = f.read(count * 8)
            if len(data) != count * 8:
                break
            trajectories.append(
                np.frombuffer(data, dtype="<f8").reshape(rows, cols).copy()
            )
    return trajectories


def read_map_bin(path):
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"invalid map header: {path}")
        rows, cols = struct.unpack("<ii", header)
        data = f.read(rows * cols * 8)
        if len(data) != rows * cols * 8:
            raise ValueError(f"invalid map payload: {path}")
    return np.frombuffer(data, dtype="<f8").reshape(rows, cols).copy()


def read_result(path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[0]
    out = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def read_positions(solver_dir):
    points = []
    for step_dir in sorted(solver_dir.glob("step_*")):
        trajs = read_vis_bin(step_dir / "x_pos.bin")
        if trajs:
            points.append(trajs[0][:, 0])
    if not points:
        return np.zeros((3, 0))
    return np.stack(points, axis=1)


def stride_limit(items, limit):
    if limit <= 0 or len(items) <= limit:
        return items
    stride = max(1, math.ceil(len(items) / limit))
    return items[::stride][:limit]


def collect_dataset(vis_root, result_dir):
    dataset = []
    first_map = None
    first_meta = {}
    for solver_name, label, result_file, color in SOLVERS:
        solver_dir = vis_root / solver_name
        meta = read_meta(solver_dir / "meta.txt")
        path = read_positions(solver_dir)
        result = read_result(result_dir / result_file)
        if first_map is None and (solver_dir / "map.bin").exists():
            first_map = read_map_bin(solver_dir / "map.bin")
            first_meta = meta
        dataset.append(
            {
                "solver": solver_name,
                "label": label,
                "dir": solver_dir,
                "color": color,
                "meta": meta,
                "path": path,
                "result": result,
            }
        )
    if first_map is None:
        raise FileNotFoundError(f"no map.bin found under {vis_root}")
    return dataset, first_map, first_meta


def result_float(result, key, default=0.0):
    value = result.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def success_text(result):
    success = result_float(result, "is_success", 0.0)
    return "success" if success != 0.0 else "fail"


def add_map(ax, grid, resolution):
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
    ax.set_ylim(-0.05, cols * resolution + 0.18)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d3d7dc", linewidth=0.45, alpha=0.65)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_paths(dataset, grid, meta, out_path):
    resolution = float(meta.get("map_resolution", 0.1))
    start = parse_vector(meta.get("x_init", "0.5 0.0 1.57079632679"))
    target = parse_vector(meta.get("x_target", "1.5 5.0 1.57079632679"))

    fig, ax = plt.subplots(figsize=(7.2, 10.0))
    add_map(ax, grid, resolution)
    for item in dataset:
        path = item["path"]
        if path.shape[1] == 0:
            continue
        result = item["result"]
        elapsed = result_float(result, "elapsed", math.nan)
        label = f"{item['label']} ({success_text(result)}, {elapsed:.3f}s)"
        ax.plot(
            path[0],
            path[1],
            color=item["color"],
            linewidth=2.0,
            label=label,
            zorder=5,
        )
        ax.scatter(path[0, -1], path[1, -1], s=28, color=item["color"], zorder=6)

    ax.scatter(start[0], start[1], marker="s", s=70, color="#111111", zorder=8, label="start")
    ax.scatter(target[0], target[1], marker="*", s=120, color="#111111", zorder=8, label="target")
    ax.set_title("WMRobot map 285 closed-loop paths")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def logged_step_numbers(item):
    steps = []
    for step_dir in item["dir"].glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            steps.append(int(step_dir.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def logged_step_name_at_or_before(item, step_idx):
    steps = logged_step_numbers(item)
    if not steps:
        return ""
    candidate = steps[0]
    for step in steps:
        if step > step_idx:
            break
        candidate = step
    return f"step_{candidate:04d}"


def gif_rollout_groups(item, step_name, max_rollouts):
    step_dir = item["dir"] / step_name
    specs = [
        ("rollouts", "rollouts", "#7f878f", "-", 0.26, 0.55, 3),
        ("clusters", "clusters", "#f58518", "-", 0.78, 1.05, 5),
        ("forward_clusters", "forward clusters", "#f58518", "-", 0.78, 1.05, 5),
        ("backward_clusters", "backward clusters", "#e45756", "--", 0.70, 1.05, 5),
    ]
    groups = []
    for file_label, display, color, linestyle, alpha, linewidth, zorder in specs:
        trajectories = stride_limit(read_vis_bin(step_dir / f"{file_label}.bin"), max_rollouts)
        if trajectories:
            groups.append((display, trajectories, color, linestyle, alpha, linewidth, zorder))
    return groups


def draw_gif_rollouts(ax, item, step_idx, max_rollouts):
    step_name = logged_step_name_at_or_before(item, step_idx)
    if not step_name:
        return
    seen = set()
    for display, trajs, color, linestyle, alpha, linewidth, zorder in gif_rollout_groups(
        item, step_name, max_rollouts
    ):
        label = display if display not in seen else None
        seen.add(display)
        for k, traj in enumerate(trajs):
            ax.plot(
                traj[0],
                traj[1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label if k == 0 else None,
                zorder=zorder,
            )


def plot_paths_gif(
    dataset,
    grid,
    meta,
    out_path,
    max_frames,
    duration_ms,
    max_rollouts=24,
    title_prefix="WMRobot map 285 path evolution",
    show_rollouts=False,
):
    from PIL import Image

    resolution = float(meta.get("map_resolution", 0.1))
    start = parse_vector(meta.get("x_init", "0.5 0.0 1.57079632679"))
    target = parse_vector(meta.get("x_target", "1.5 5.0 1.57079632679"))
    max_len = max((item["path"].shape[1] for item in dataset), default=0)
    if max_len == 0:
        return False

    indices = list(range(max_len))
    if max_frames > 0 and len(indices) > max_frames:
        stride = math.ceil(len(indices) / max_frames)
        indices = indices[::stride]
        if indices[-1] != max_len - 1:
            indices.append(max_len - 1)

    frames = []
    for step_idx in indices:
        fig, ax = plt.subplots(figsize=(5.8, 8.0), dpi=110)
        add_map(ax, grid, resolution)
        for item in dataset:
            if show_rollouts:
                draw_gif_rollouts(ax, item, step_idx, max_rollouts)
            path = item["path"]
            if path.shape[1] == 0:
                continue
            end = min(step_idx + 1, path.shape[1])
            if end <= 0:
                continue
            ax.plot(
                path[0, :end],
                path[1, :end],
                color=item["color"],
                linewidth=2.15,
                label=item["label"],
                zorder=9,
            )
            ax.scatter(
                path[0, end - 1],
                path[1, end - 1],
                s=22,
                color=item["color"],
                edgecolor="#ffffff",
                linewidth=0.45,
                zorder=10,
            )

        ax.scatter(start[0], start[1], marker="s", s=48, color="#111111", zorder=8)
        ax.scatter(target[0], target[1], marker="*", s=82, color="#111111", zorder=8)
        ax.set_title(f"{title_prefix}  |  step {step_idx:03d}")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2,
                  frameon=False, fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy()))
        plt.close(fig)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return True


def plot_timing(dataset, out_path):
    labels = [item["label"] for item in dataset]
    elapsed = [result_float(item["result"], "elapsed", math.nan) * 1000.0 for item in dataset]
    rollout = [
        result_float(item["result"], "elapsed_rollout", 0.0) * 1000.0
        for item in dataset
    ]
    clustering = [
        result_float(item["result"], "elapsed_clustering", 0.0) * 1000.0
        for item in dataset
    ]
    connection = [
        result_float(item["result"], "elapsed_connection", 0.0) * 1000.0
        for item in dataset
    ]
    guide = [
        result_float(item["result"], "elapsed_guide", 0.0) * 1000.0
        for item in dataset
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for values, name, color in [
        (rollout, "rollout", "#4c78a8"),
        (clustering, "clustering", "#54a24b"),
        (connection, "connection", "#e45756"),
        (guide, "guide", "#b279a2"),
    ]:
        ax.bar(x, values, bottom=bottom, label=name, color=color)
        bottom += np.array(values)

    for i, total in enumerate(elapsed):
        if math.isfinite(total):
            ax.text(i, total * 1.02 if total > 0 else 1.0, success_text(dataset[i]["result"]),
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("accumulated solver time [ms]")
    ax.set_title("WMRobot map 285 runtime")
    ymax = max([v for v in elapsed if math.isfinite(v)] + [1.0])
    ax.set_ylim(0.0, ymax * 1.18)
    ax.grid(axis="y", color="#d3d7dc", linewidth=0.55, alpha=0.8)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def rollout_groups(item, step_name, max_rollouts):
    step_dir = item["dir"] / step_name
    groups = []
    specs = [
        ("rollouts", "rollouts", item["color"], "-", 0.23),
        ("clusters", "clusters", "#1f7a36", "-", 0.55),
        ("forward_clusters", "forward clusters", "#e45756", "--", 0.45),
        ("backward_clusters", "backward clusters", "#4c78a8", "--", 0.45),
        ("guide_candidates", "guide candidates", "#b279a2", "-", 0.45),
    ]
    for file_label, display, color, linestyle, alpha in specs:
        trajs = read_vis_bin(step_dir / f"{file_label}.bin")
        trajs = stride_limit(trajs, max_rollouts)
        if trajs:
            groups.append((display, trajs, color, linestyle, alpha))
    return groups


def choose_step(item, requested):
    steps = sorted([p.name for p in item["dir"].glob("step_*") if p.is_dir()])
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


def plot_rollouts(dataset, grid, meta, out_path, step_selector, max_rollouts):
    resolution = float(meta.get("map_resolution", 0.1))
    start = parse_vector(meta.get("x_init", "0.5 0.0 1.57079632679"))
    target = parse_vector(meta.get("x_target", "1.5 5.0 1.57079632679"))

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 9.0), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, item in zip(axes, dataset):
        add_map(ax, grid, resolution)
        step_name = choose_step(item, step_selector)
        for display, trajs, color, linestyle, alpha in rollout_groups(
            item, step_name, max_rollouts
        ):
            for k, traj in enumerate(trajs):
                ax.plot(
                    traj[0],
                    traj[1],
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.85,
                    alpha=alpha,
                    label=display if k == 0 else None,
                )
        path = item["path"]
        if path.shape[1] > 0:
            ax.plot(path[0], path[1], color="#111111", linewidth=1.5, alpha=0.9)
        ax.scatter(start[0], start[1], marker="s", s=34, color="#111111", zorder=8)
        ax.scatter(target[0], target[1], marker="*", s=70, color="#111111", zorder=8)
        ax.set_title(f"{item['label']} samples ({step_name})")
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    for ax in axes[len(dataset):]:
        ax.axis("off")

    fig.suptitle("WMRobot map 285 sampled rollouts", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(dataset, out_path):
    fields = [
        "solver",
        "label",
        "map",
        "is_success",
        "iter",
        "elapsed",
        "elapsed_rollout",
        "elapsed_clustering",
        "elapsed_connection",
        "elapsed_guide",
        "f_err",
        "path_points",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in dataset:
            result = item["result"]
            row = {
                "solver": item["solver"],
                "label": item["label"],
                "path_points": int(item["path"].shape[1]),
            }
            for key in fields:
                if key in row:
                    continue
                row[key] = result.get(key, "")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-root", default="build/vis_data")
    parser.add_argument("--result-dir", default="build")
    parser.add_argument("--out-dir", default="results/wmrobot_map_285")
    parser.add_argument(
        "--rollout-step",
        default="first",
        help="first, middle, last, or a concrete step name such as step_0000",
    )
    parser.add_argument("--max-rollouts", type=int, default=24)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    args = parser.parse_args()

    vis_root = Path(args.vis_root)
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, grid, meta = collect_dataset(vis_root, result_dir)
    write_summary(dataset, out_dir / "wmrobot_map_285_summary.csv")
    plot_paths(dataset, grid, meta, out_dir / "wmrobot_map_285_paths.png")
    plot_timing(dataset, out_dir / "wmrobot_map_285_timing.png")
    plot_rollouts(
        dataset,
        grid,
        meta,
        out_dir / "wmrobot_map_285_rollouts.png",
        args.rollout_step,
        args.max_rollouts,
    )
    gif_path = out_dir / "wmrobot_map_285_paths.gif"
    gif_created = False
    solver_gif_paths = []
    if not args.no_gif:
        gif_created = plot_paths_gif(
            dataset,
            grid,
            meta,
            gif_path,
            args.gif_frames,
            args.gif_duration_ms,
            args.max_rollouts,
        )
        solver_gif_dir = out_dir / "solver_gifs"
        solver_gif_dir.mkdir(parents=True, exist_ok=True)
        for item in dataset:
            solver_gif_path = solver_gif_dir / f"wmrobot_map_285_{item['solver']}_paths.gif"
            if plot_paths_gif(
                [item],
                grid,
                meta,
                solver_gif_path,
                args.gif_frames,
                args.gif_duration_ms,
                args.max_rollouts,
                f"WMRobot map 285 {item['label']} path evolution",
                True,
            ):
                solver_gif_paths.append(solver_gif_path)

    print(out_dir / "wmrobot_map_285_summary.csv")
    print(out_dir / "wmrobot_map_285_paths.png")
    print(out_dir / "wmrobot_map_285_timing.png")
    print(out_dir / "wmrobot_map_285_rollouts.png")
    if gif_created:
        print(gif_path)
    for solver_gif_path in solver_gif_paths:
        print(solver_gif_path)


if __name__ == "__main__":
    main()
