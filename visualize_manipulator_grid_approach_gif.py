#!/usr/bin/env python3
"""Render the grid-approach manipulator trajectory as a GIF."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


DH_A = np.array([0.0, -0.427, -0.357, 0.0, 0.0, 0.0])
DH_D = np.array([0.15, 0.0, 0.0, 0.11, 0.09, 0.09])
DH_ALPHA = np.array([math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0])

X_INSERT = -0.56
X_RETRACT = -0.33
Y_CENTERS = np.array([-0.36, -0.16, 0.04])
Z_CENTERS = np.array([0.18, 0.36, 0.54])
WALL_BOXES = (
    ("vertical_01", ((-0.62, -0.42), (-0.2775, -0.2425), (0.09, 0.63))),
    ("vertical_12", ((-0.62, -0.42), (-0.0775, -0.0425), (0.09, 0.63))),
    ("horizontal_01", ((-0.62, -0.42), (-0.46, 0.14), (0.2525, 0.2875))),
    ("horizontal_12", ((-0.62, -0.42), (-0.46, 0.14), (0.4325, 0.4675))),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a GIF from build/manipulator_grid_approach_bicmppi_executed_x.csv"
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        type=Path,
        help="Directory containing manipulator_grid_approach_bicmppi_*.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="GIF output directory. Defaults to <build-dir>/manipulator_grid_approach_gifs",
    )
    parser.add_argument("--fps", default=12, type=int, help="GIF frame rate")
    parser.add_argument(
        "--max-frames",
        default=120,
        type=int,
        help="Maximum animation frames sampled from the trajectory",
    )
    parser.add_argument(
        "--prefix",
        default="manipulator_grid_approach_bicmppi_",
        help="CSV filename prefix inside build-dir",
    )
    return parser.parse_args()


def load_states(csv_path):
    states = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            states.append([float(row[f"x{i}"]) for i in range(1, 13)])
    if not states:
        raise RuntimeError(f"{csv_path} has no state rows")
    return np.asarray(states, dtype=float)


def load_metadata(csv_path):
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{csv_path} has no metadata rows")
    return rows[0]


def metadata_point(row, prefix):
    return np.array(
        [
            float(row[f"{prefix}_x"]),
            float(row[f"{prefix}_y"]),
            float(row[f"{prefix}_z"]),
        ],
        dtype=float,
    )


def dh_transform(theta, a, d, alpha):
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def joint_positions(q):
    transform = np.eye(4)
    positions = [np.zeros(3)]
    for i in range(6):
        transform = transform @ dh_transform(q[i], DH_A[i], DH_D[i], DH_ALPHA[i])
        positions.append(transform[:3, 3].copy())
    return np.asarray(positions)


def box_vertices(bounds):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    return np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )


def obstacle_corners():
    return np.concatenate([box_vertices(bounds) for _, bounds in WALL_BOXES], axis=0)


def draw_box(ax, bounds):
    vertices = box_vertices(bounds)
    faces = [
        [vertices[i] for i in (0, 1, 2, 3)],
        [vertices[i] for i in (4, 5, 6, 7)],
        [vertices[i] for i in (0, 1, 5, 4)],
        [vertices[i] for i in (2, 3, 7, 6)],
        [vertices[i] for i in (1, 2, 6, 5)],
        [vertices[i] for i in (0, 3, 7, 4)],
    ]
    collection = Poly3DCollection(
        faces,
        facecolors=(0.84, 0.20, 0.16, 0.18),
        edgecolors=(0.58, 0.05, 0.04, 0.70),
        linewidths=0.85,
    )
    ax.add_collection3d(collection)


def grid_points(x_value):
    points = []
    for z in Z_CENTERS:
        for y in Y_CENTERS:
            points.append([x_value, y, z])
    return np.asarray(points, dtype=float)


def sampled_frame_indices(count, max_frames):
    if count <= 1:
        return [0]
    stride = max(1, math.ceil(count / max_frames))
    frames = list(range(0, count, stride))
    if frames[-1] != count - 1:
        frames.append(count - 1)
    return frames


def set_axes_from_points(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0 + 0.15, 0.78)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.85))
    except AttributeError:
        pass


def render_grid_approach(build_dir, out_dir, prefix, fps, max_frames):
    trajectory_csv = build_dir / f"{prefix}executed_x.csv"
    metadata_csv = build_dir / f"{prefix}task_metadata.csv"
    if not trajectory_csv.exists():
        raise FileNotFoundError(
            f"{trajectory_csv} not found. Run ./run_manipulator_grid_wall_gif.sh first."
        )
    if not metadata_csv.exists():
        raise FileNotFoundError(f"{metadata_csv} not found")

    states = load_states(trajectory_csv)
    metadata = load_metadata(metadata_csv)
    qs = states[:, :6]
    pose_sequence = np.asarray([joint_positions(q) for q in qs])

    free_opposite = metadata_point(metadata, "free_opposite")
    free_aligned = metadata_point(metadata, "free_aligned")
    goal_retract = metadata_point(metadata, "goal_retract")
    goal_insert = metadata_point(metadata, "goal_insert")
    nominal_path = np.vstack([free_opposite, free_aligned, goal_retract, goal_insert])

    insert_points = grid_points(X_INSERT)
    retract_points = grid_points(X_RETRACT)
    all_points = np.concatenate(
        [
            pose_sequence.reshape(-1, 3),
            obstacle_corners(),
            insert_points,
            retract_points,
            nominal_path,
        ],
        axis=0,
    )
    frame_indices = sampled_frame_indices(len(qs), max_frames)

    fig = plt.figure(figsize=(8, 6.5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    for _, bounds in WALL_BOXES:
        draw_box(ax, bounds)

    ax.scatter(
        insert_points[:, 0],
        insert_points[:, 1],
        insert_points[:, 2],
        color="#8c8c8c",
        s=18,
        depthshade=False,
        label="insert cells",
    )
    ax.scatter(
        retract_points[:, 0],
        retract_points[:, 1],
        retract_points[:, 2],
        color="#b8b8b8",
        s=14,
        depthshade=False,
        label="retract cells",
    )
    ax.plot(
        nominal_path[:, 0],
        nominal_path[:, 1],
        nominal_path[:, 2],
        "o--",
        color="#20945f",
        linewidth=1.8,
        markersize=4.5,
        alpha=0.88,
        label="nominal route",
    )
    ax.scatter(
        [free_opposite[0]],
        [free_opposite[1]],
        [free_opposite[2]],
        color="#1f5fd1",
        s=56,
        depthshade=False,
        label="free opposite",
    )
    ax.scatter(
        [goal_insert[0]],
        [goal_insert[1]],
        [goal_insert[2]],
        color="#d12f1f",
        s=56,
        depthshade=False,
        label="goal insert",
    )

    current_line, = ax.plot(
        [],
        [],
        [],
        "o-",
        color="#101820",
        linewidth=3.0,
        markersize=5.4,
        label="manipulator",
    )
    ee_trace, = ax.plot([], [], [], color="#1f5fd1", linewidth=1.5, alpha=0.78)
    step_text = ax.text2D(
        0.035,
        0.955,
        "",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#101820",
    )

    set_axes_from_points(ax, all_points)
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.28)

    goal_cell = metadata.get("goal_cell", "?")

    def update(frame):
        pose = pose_sequence[frame]
        trace = pose_sequence[: frame + 1, -1, :]
        current_line.set_data(pose[:, 0], pose[:, 1])
        current_line.set_3d_properties(pose[:, 2])
        ee_trace.set_data(trace[:, 0], trace[:, 1])
        ee_trace.set_3d_properties(trace[:, 2])
        step_text.set_text(
            f"Grid-approach BiC-MPPI | goal cell {goal_cell} | step {frame}/{len(qs) - 1}"
        )
        return current_line, ee_trace, step_text

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / max(1, fps),
        blit=False,
        repeat=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "manipulator_grid_approach_bicmppi.gif"
    animation.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return gif_path


def main():
    args = parse_args()
    build_dir = args.build_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else build_dir / "manipulator_grid_approach_gifs"
    )
    gif_path = render_grid_approach(build_dir, out_dir, args.prefix, args.fps, args.max_frames)
    print(f"[gif] {gif_path}")


if __name__ == "__main__":
    main()
