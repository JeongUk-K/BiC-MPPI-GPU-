#!/usr/bin/env python3
"""Render manipulator pinkNplace example trajectories as GIF files."""

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

HOME_Q = np.array([0.00, -1.00, 1.20, 0.00, 0.80, 0.00])
PHASE_TARGETS = (
    ("pre-pick", np.array([-0.65, -1.18, 1.22, 0.00, 0.78, 0.00])),
    ("pick", np.array([-0.65, -0.82, 1.52, 0.00, 0.38, 0.00])),
    ("transfer", np.array([0.10, -1.45, 1.25, 0.00, 1.05, 0.00])),
    ("pre-place", np.array([0.90, -1.18, 1.18, 0.00, 0.80, 0.00])),
    ("place", np.array([0.90, -0.78, 1.52, 0.00, 0.36, 0.00])),
)
OBSTACLE_BOXES = (
    ("central_baffle", ((-0.48, -0.30), (-0.22, 0.06), (0.05, 0.38))),
    ("pick_bin_rear_wall", ((-0.72, -0.44), (0.30, 0.36), (0.02, 0.15))),
    ("pick_bin_left_wall", ((-0.72, -0.66), (0.04, 0.36), (0.02, 0.15))),
    ("place_bin_front_wall", ((-0.34, -0.04), (-0.72, -0.66), (0.02, 0.16))),
    ("place_bin_right_wall", ((-0.04, 0.02), (-0.72, -0.42), (0.02, 0.16))),
)

SOLVERS = (
    ("mppi", "MPPI"),
    ("logmppi", "Log-MPPI"),
    ("clustermppi", "Cluster-MPPI"),
    ("bicmppi", "BiC-MPPI"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create GIFs from build/manipulator_pinkNplace_*_executed_x.csv"
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        type=Path,
        help="Directory containing manipulator_pinkNplace_*_executed_x.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="GIF output directory. Defaults to <build-dir>/manipulator_pinkNplace_gifs",
    )
    parser.add_argument("--fps", default=12, type=int, help="GIF frame rate")
    parser.add_argument(
        "--max-frames",
        default=120,
        type=int,
        help="Maximum animation frames sampled from each trajectory",
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=[name for name, _ in SOLVERS],
        help="Render only the selected solver. Can be passed multiple times.",
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
    return np.concatenate([box_vertices(bounds) for _, bounds in OBSTACLE_BOXES], axis=0)


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
        facecolors=(0.86, 0.23, 0.18, 0.16),
        edgecolors=(0.60, 0.06, 0.04, 0.66),
        linewidths=0.8,
    )
    ax.add_collection3d(collection)


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


def render_solver(build_dir, out_dir, solver_name, label, fps, max_frames):
    csv_path = build_dir / f"manipulator_pinkNplace_{solver_name}_executed_x.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run ./manipulaotr.sh pinkNplace first.")

    states = load_states(csv_path)
    qs = states[:, :6]
    pose_sequence = np.asarray([joint_positions(q) for q in qs])
    home_pose = joint_positions(HOME_Q)
    phase_poses = [(name, joint_positions(q)) for name, q in PHASE_TARGETS]
    phase_ee = np.asarray([pose[-1] for _, pose in phase_poses])

    all_points = np.concatenate(
        [
            pose_sequence.reshape(-1, 3),
            home_pose,
            *[pose for _, pose in phase_poses],
            obstacle_corners(),
        ],
        axis=0,
    )
    frame_indices = sampled_frame_indices(len(qs), max_frames)

    fig = plt.figure(figsize=(8, 6.5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    for _, bounds in OBSTACLE_BOXES:
        draw_box(ax, bounds)

    ax.plot(
        home_pose[:, 0],
        home_pose[:, 1],
        home_pose[:, 2],
        "o--",
        color="#7a7a7a",
        linewidth=1.4,
        markersize=3.8,
        alpha=0.72,
        label="home",
    )
    ax.scatter(
        phase_ee[:, 0],
        phase_ee[:, 1],
        phase_ee[:, 2],
        color="#20945f",
        s=34,
        depthshade=False,
        label="phase targets",
    )

    current_line, = ax.plot(
        [],
        [],
        [],
        "o-",
        color="#1f5fd1",
        linewidth=3.0,
        markersize=5.4,
        label="executed",
    )
    ee_trace, = ax.plot([], [], [], color="#101820", linewidth=1.3, alpha=0.72)
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

    def update(frame):
        pose = pose_sequence[frame]
        trace = pose_sequence[: frame + 1, -1, :]
        current_line.set_data(pose[:, 0], pose[:, 1])
        current_line.set_3d_properties(pose[:, 2])
        ee_trace.set_data(trace[:, 0], trace[:, 1])
        ee_trace.set_3d_properties(trace[:, 2])
        step_text.set_text(f"{label} pinkNplace motion | step {frame}/{len(qs) - 1}")
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
    gif_path = out_dir / f"manipulator_pinkNplace_{solver_name}.gif"
    animation.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return gif_path


def main():
    args = parse_args()
    build_dir = args.build_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else build_dir / "manipulator_pinkNplace_gifs"
    )
    selected = set(args.solver or [name for name, _ in SOLVERS])

    generated = []
    for solver_name, label in SOLVERS:
        if solver_name not in selected:
            continue
        gif_path = render_solver(build_dir, out_dir, solver_name, label, args.fps, args.max_frames)
        generated.append(gif_path)
        print(f"[gif] {gif_path}")

    if not generated:
        raise RuntimeError("No solvers selected")


if __name__ == "__main__":
    main()
