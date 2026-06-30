#!/usr/bin/env python3
"""Render random-pose manipulator benchmark trajectories and aggregate stats."""

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

DEFAULT_OBSTACLE_BOUNDS = ((-0.34, -0.20), (-0.52, -0.38), (0.28, 0.42))
SOLVERS = (
    ("mppi", "MPPI"),
    ("logmppi", "Log-MPPI"),
    ("clustermppi", "Cluster-MPPI"),
    ("bicmppi", "BiC-MPPI"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create GIFs and aggregate stats for the random pose benchmark"
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        type=Path,
        help="Build directory used to resolve relative trajectory paths",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        type=Path,
        help="Benchmark output directory. Defaults to random_pose_benchmark/results",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="GIF output directory. Defaults to <benchmark-dir>/gifs",
    )
    parser.add_argument("--fps", default=12, type=int, help="GIF frame rate")
    parser.add_argument(
        "--max-frames",
        default=72,
        type=int,
        help="Maximum animation frames sampled from each trajectory",
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=[name for name, _ in SOLVERS],
        help="Render only the selected solver. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Only rebuild the final aggregate stats CSV",
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


def load_rows(csv_path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def obstacle_bounds_from_scenario(scenario):
    required = (
        "obstacle_xmin",
        "obstacle_xmax",
        "obstacle_ymin",
        "obstacle_ymax",
        "obstacle_zmin",
        "obstacle_zmax",
    )
    if not all(key in scenario for key in required):
        return DEFAULT_OBSTACLE_BOUNDS
    return (
        (float(scenario["obstacle_xmin"]), float(scenario["obstacle_xmax"])),
        (float(scenario["obstacle_ymin"]), float(scenario["obstacle_ymax"])),
        (float(scenario["obstacle_zmin"]), float(scenario["obstacle_zmax"])),
    )


def obstacle_corners(bounds):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    return np.asarray(
        [
            [x, y, z]
            for x in (xmin, xmax)
            for y in (ymin, ymax)
            for z in (zmin, zmax)
        ],
        dtype=float,
    )


def draw_obstacle(ax, bounds):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    vertices = np.asarray(
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
        facecolors=(0.86, 0.23, 0.18, 0.18),
        edgecolors=(0.60, 0.06, 0.04, 0.70),
        linewidths=1.0,
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
    radius = max(float(np.max(maxs - mins)) / 2.0 + 0.15, 0.75)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.85))
    except AttributeError:
        pass


def scenario_map(benchmark_dir):
    path = benchmark_dir / "manipulator_random_pose_benchmark_scenarios.csv"
    rows = load_rows(path)
    return {int(row["scenario"]): row for row in rows}


def trajectory_path_from_row(build_dir, row):
    path = Path(row["trajectory_csv"])
    if path.is_absolute():
        return path
    return build_dir / path


def render_row(build_dir, out_dir, scenario_rows, solver_key, solver_label, row, fps, max_frames):
    scenario_index = int(row["scenario"])
    traj_csv = trajectory_path_from_row(build_dir, row)
    if not traj_csv.exists():
        raise FileNotFoundError(f"{traj_csv} not found")

    scenario = scenario_rows[scenario_index]
    obstacle_bounds = obstacle_bounds_from_scenario(scenario)
    goal_q = np.asarray([float(scenario[f"goal_q{i}"]) for i in range(1, 7)])

    states = load_states(traj_csv)
    qs = states[:, :6]
    pose_sequence = np.asarray([joint_positions(q) for q in qs])
    start_pose = joint_positions(qs[0])
    goal_pose = joint_positions(goal_q)

    all_points = np.concatenate(
        [
            pose_sequence.reshape(-1, 3),
            start_pose,
            goal_pose,
            obstacle_corners(obstacle_bounds),
        ],
        axis=0,
    )
    frame_indices = sampled_frame_indices(len(qs), max_frames)

    fig = plt.figure(figsize=(8, 6.5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    draw_obstacle(ax, obstacle_bounds)
    ax.plot(
        start_pose[:, 0],
        start_pose[:, 1],
        start_pose[:, 2],
        "o--",
        color="#7a7a7a",
        linewidth=1.4,
        markersize=3.8,
        alpha=0.75,
        label="start",
    )
    ax.plot(
        goal_pose[:, 0],
        goal_pose[:, 1],
        goal_pose[:, 2],
        "o--",
        color="#20945f",
        linewidth=1.6,
        markersize=3.8,
        alpha=0.85,
        label="goal",
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
        fontsize=10.5,
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

    init_name = row["init_name"]
    goal_name = row["goal_name"]
    success = int(float(row["success"]))
    solve_iterations = int(float(row["solve_iterations"]))

    def update(frame):
        pose = pose_sequence[frame]
        trace = pose_sequence[: frame + 1, -1, :]
        current_line.set_data(pose[:, 0], pose[:, 1])
        current_line.set_3d_properties(pose[:, 2])
        ee_trace.set_data(trace[:, 0], trace[:, 1])
        ee_trace.set_3d_properties(trace[:, 2])
        step_text.set_text(
            f"{solver_label} {scenario_index:02d}: {init_name} -> {goal_name} | "
            f"step {frame}/{len(qs) - 1} | success={success} iter={solve_iterations}"
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

    solver_out_dir = out_dir / solver_key
    solver_out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = solver_out_dir / f"scenario_{scenario_index:02d}_{solver_key}.gif"
    animation.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return gif_path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def csv_mean(values):
    return mean(values) if values else ""


def stddev(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def csv_stddev(values):
    return stddev(values) if values else ""


def csv_max(values):
    return max(values) if values else ""


def as_float(row, key):
    return float(row[key])


def as_int(row, key):
    return int(float(row[key]))


def row_thresholds(row):
    return (
        as_float(row, "q_tol"),
        as_float(row, "ee_tol"),
        as_float(row, "qdot_tol"),
    )


def row_q_ok(row):
    q_tol, _, _ = row_thresholds(row)
    return as_float(row, "final_q_error") < q_tol


def row_ee_ok(row):
    _, ee_tol, _ = row_thresholds(row)
    return as_float(row, "final_ee_error") < ee_tol


def row_qdot_ok(row):
    _, _, qdot_tol = row_thresholds(row)
    return as_float(row, "final_qdot_norm") < qdot_tol


def row_goal_reached_strict(row):
    return row_q_ok(row) and row_ee_ok(row) and row_qdot_ok(row)


def row_goal_reached_ee(row):
    return row_ee_ok(row) and row_qdot_ok(row)


def row_collision_free(row):
    return as_int(row, "collision_count") == 0


def count_true(rows, predicate):
    return sum(1 for row in rows if predicate(row))


def rate(count, total):
    return count / total if total else 0.0


def write_final_stats(benchmark_dir, all_solver_rows):
    final_path = benchmark_dir / "manipulator_random_pose_benchmark_final_stats.csv"
    headers = [
        "solver",
        "solver_label",
        "scenario_count",
        "q_tol",
        "ee_tol",
        "qdot_tol",
        "success_count",
        "success_rate",
        "success_mean_count",
        "goal_reached_strict_count",
        "goal_reached_strict_rate",
        "goal_reached_ee_count",
        "goal_reached_ee_rate",
        "collision_free_count",
        "collision_free_rate",
        "near_goal_ee05_count",
        "near_goal_ee05_rate",
        "near_goal_ee07_count",
        "near_goal_ee07_rate",
        "q_ok_count",
        "q_ok_rate",
        "ee_ok_count",
        "ee_ok_rate",
        "qdot_ok_count",
        "qdot_ok_rate",
        "q_fail_count",
        "ee_fail_count",
        "qdot_fail_count",
        "collision_fail_count",
        "mean_solve_iterations",
        "std_solve_iterations",
        "mean_executed_steps",
        "mean_total_solver_elapsed_s",
        "std_total_solver_elapsed_s",
        "mean_wall_elapsed_s",
        "std_wall_elapsed_s",
        "mean_final_q_error",
        "mean_final_ee_error",
        "mean_final_qdot_norm",
        "mean_collision_count",
        "max_collision_count",
        "all_mean_total_solver_elapsed_s",
        "all_mean_wall_elapsed_s",
        "all_mean_final_q_error",
        "all_mean_final_ee_error",
        "all_mean_final_qdot_norm",
        "all_mean_collision_count",
        "mean_samples_total_per_iter",
    ]

    with final_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for solver_key, rows in all_solver_rows:
            if not rows:
                continue
            q_tol, ee_tol, qdot_tol = row_thresholds(rows[0])
            scenario_count = len(rows)
            successful_rows = [row for row in rows if as_int(row, "success") == 1]
            success_count = len(successful_rows)
            goal_reached_strict_count = count_true(rows, row_goal_reached_strict)
            goal_reached_ee_count = count_true(rows, row_goal_reached_ee)
            collision_free_count = count_true(rows, row_collision_free)
            near_goal_ee05_count = count_true(
                rows, lambda row: as_float(row, "final_ee_error") < 0.05
            )
            near_goal_ee07_count = count_true(
                rows, lambda row: as_float(row, "final_ee_error") < 0.07
            )
            q_ok_count = count_true(rows, row_q_ok)
            ee_ok_count = count_true(rows, row_ee_ok)
            qdot_ok_count = count_true(rows, row_qdot_ok)
            q_fail_count = scenario_count - q_ok_count
            ee_fail_count = scenario_count - ee_ok_count
            qdot_fail_count = scenario_count - qdot_ok_count
            collision_fail_count = scenario_count - collision_free_count
            solve_iterations = [
                as_float(row, "solve_iterations") for row in successful_rows
            ]
            executed_steps = [
                as_float(row, "executed_steps") for row in successful_rows
            ]
            solver_elapsed = [
                as_float(row, "total_solver_elapsed_s") for row in successful_rows
            ]
            wall_elapsed = [
                as_float(row, "wall_elapsed_s") for row in successful_rows
            ]
            q_error = [as_float(row, "final_q_error") for row in successful_rows]
            ee_error = [as_float(row, "final_ee_error") for row in successful_rows]
            qdot_norm = [
                as_float(row, "final_qdot_norm") for row in successful_rows
            ]
            collisions = [as_float(row, "collision_count") for row in successful_rows]
            all_solver_elapsed = [
                as_float(row, "total_solver_elapsed_s") for row in rows
            ]
            all_wall_elapsed = [as_float(row, "wall_elapsed_s") for row in rows]
            all_q_error = [as_float(row, "final_q_error") for row in rows]
            all_ee_error = [as_float(row, "final_ee_error") for row in rows]
            all_qdot_norm = [as_float(row, "final_qdot_norm") for row in rows]
            all_collisions = [as_float(row, "collision_count") for row in rows]
            samples_total = [as_float(row, "samples_total_per_iter") for row in rows]
            writer.writerow(
                {
                    "solver": solver_key,
                    "solver_label": rows[0]["solver_label"],
                    "scenario_count": scenario_count,
                    "q_tol": q_tol,
                    "ee_tol": ee_tol,
                    "qdot_tol": qdot_tol,
                    "success_count": success_count,
                    "success_rate": rate(success_count, scenario_count),
                    "success_mean_count": success_count,
                    "goal_reached_strict_count": goal_reached_strict_count,
                    "goal_reached_strict_rate": rate(
                        goal_reached_strict_count, scenario_count
                    ),
                    "goal_reached_ee_count": goal_reached_ee_count,
                    "goal_reached_ee_rate": rate(
                        goal_reached_ee_count, scenario_count
                    ),
                    "collision_free_count": collision_free_count,
                    "collision_free_rate": rate(
                        collision_free_count, scenario_count
                    ),
                    "near_goal_ee05_count": near_goal_ee05_count,
                    "near_goal_ee05_rate": rate(
                        near_goal_ee05_count, scenario_count
                    ),
                    "near_goal_ee07_count": near_goal_ee07_count,
                    "near_goal_ee07_rate": rate(
                        near_goal_ee07_count, scenario_count
                    ),
                    "q_ok_count": q_ok_count,
                    "q_ok_rate": rate(q_ok_count, scenario_count),
                    "ee_ok_count": ee_ok_count,
                    "ee_ok_rate": rate(ee_ok_count, scenario_count),
                    "qdot_ok_count": qdot_ok_count,
                    "qdot_ok_rate": rate(qdot_ok_count, scenario_count),
                    "q_fail_count": q_fail_count,
                    "ee_fail_count": ee_fail_count,
                    "qdot_fail_count": qdot_fail_count,
                    "collision_fail_count": collision_fail_count,
                    "mean_solve_iterations": csv_mean(solve_iterations),
                    "std_solve_iterations": csv_stddev(solve_iterations),
                    "mean_executed_steps": csv_mean(executed_steps),
                    "mean_total_solver_elapsed_s": csv_mean(solver_elapsed),
                    "std_total_solver_elapsed_s": csv_stddev(solver_elapsed),
                    "mean_wall_elapsed_s": csv_mean(wall_elapsed),
                    "std_wall_elapsed_s": csv_stddev(wall_elapsed),
                    "mean_final_q_error": csv_mean(q_error),
                    "mean_final_ee_error": csv_mean(ee_error),
                    "mean_final_qdot_norm": csv_mean(qdot_norm),
                    "mean_collision_count": csv_mean(collisions),
                    "max_collision_count": csv_max(collisions),
                    "all_mean_total_solver_elapsed_s": mean(all_solver_elapsed),
                    "all_mean_wall_elapsed_s": mean(all_wall_elapsed),
                    "all_mean_final_q_error": mean(all_q_error),
                    "all_mean_final_ee_error": mean(all_ee_error),
                    "all_mean_final_qdot_norm": mean(all_qdot_norm),
                    "all_mean_collision_count": mean(all_collisions),
                    "mean_samples_total_per_iter": mean(samples_total),
                }
            )
    return final_path


def failure_flags(row):
    flags = []
    if not row_q_ok(row):
        flags.append("q")
    if not row_ee_ok(row):
        flags.append("ee")
    if not row_qdot_ok(row):
        flags.append("vel")
    if not row_collision_free(row):
        flags.append("col")
    return "+".join(flags) if flags else "ok"


def write_failure_breakdown(benchmark_dir, all_solver_rows):
    breakdown_path = (
        benchmark_dir / "manipulator_random_pose_benchmark_failure_breakdown.csv"
    )
    headers = [
        "solver",
        "solver_label",
        "scenario",
        "init_name",
        "goal_name",
        "q_tol",
        "ee_tol",
        "qdot_tol",
        "final_q_error",
        "final_ee_error",
        "final_qdot_norm",
        "collision_count",
        "goal_reached_strict",
        "goal_reached_ee",
        "collision_free",
        "near_goal_ee05",
        "near_goal_ee07",
        "q_ok",
        "ee_ok",
        "qdot_ok",
        "failure_flags",
    ]
    with breakdown_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for solver_key, rows in all_solver_rows:
            for row in rows:
                q_tol, ee_tol, qdot_tol = row_thresholds(row)
                writer.writerow(
                    {
                        "solver": solver_key,
                        "solver_label": row["solver_label"],
                        "scenario": row["scenario"],
                        "init_name": row["init_name"],
                        "goal_name": row["goal_name"],
                        "q_tol": q_tol,
                        "ee_tol": ee_tol,
                        "qdot_tol": qdot_tol,
                        "final_q_error": row["final_q_error"],
                        "final_ee_error": row["final_ee_error"],
                        "final_qdot_norm": row["final_qdot_norm"],
                        "collision_count": row["collision_count"],
                        "goal_reached_strict": int(row_goal_reached_strict(row)),
                        "goal_reached_ee": int(row_goal_reached_ee(row)),
                        "collision_free": int(row_collision_free(row)),
                        "near_goal_ee05": int(
                            as_float(row, "final_ee_error") < 0.05
                        ),
                        "near_goal_ee07": int(
                            as_float(row, "final_ee_error") < 0.07
                        ),
                        "q_ok": int(row_q_ok(row)),
                        "ee_ok": int(row_ee_ok(row)),
                        "qdot_ok": int(row_qdot_ok(row)),
                        "failure_flags": failure_flags(row),
                    }
                )
    return breakdown_path


def main():
    args = parse_args()
    build_dir = args.build_dir.resolve()
    benchmark_dir = (
        args.benchmark_dir.resolve()
        if args.benchmark_dir is not None
        else Path("random_pose_benchmark/results").resolve()
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else benchmark_dir / "gifs"
    )

    selected = set(args.solver or [name for name, _ in SOLVERS])
    scenarios = scenario_map(benchmark_dir)
    expected_scenario_count = len(scenarios)

    all_solver_rows = []
    generated = []
    for solver_key, solver_label in SOLVERS:
        if solver_key not in selected:
            continue
        stats_path = benchmark_dir / f"manipulator_random_pose_benchmark_{solver_key}_stats.csv"
        rows = load_rows(stats_path)
        if len(rows) != expected_scenario_count:
            raise RuntimeError(
                f"{stats_path} should contain {expected_scenario_count} "
                f"scenarios, found {len(rows)}"
            )
        all_solver_rows.append((solver_key, rows))

        if args.skip_gifs:
            continue
        for row in rows:
            gif_path = render_row(
                build_dir,
                out_dir,
                scenarios,
                solver_key,
                solver_label,
                row,
                args.fps,
                args.max_frames,
            )
            generated.append(gif_path)
            print(f"[gif] {gif_path}")

    final_path = write_final_stats(benchmark_dir, all_solver_rows)
    print(f"[stats] {final_path}")
    breakdown_path = write_failure_breakdown(benchmark_dir, all_solver_rows)
    print(f"[stats] {breakdown_path}")

    expected_gif_count = sum(len(rows) for _, rows in all_solver_rows)
    if not args.skip_gifs and len(generated) != expected_gif_count:
        raise RuntimeError(
            f"expected {expected_gif_count} GIFs, generated {len(generated)}"
        )


if __name__ == "__main__":
    main()
