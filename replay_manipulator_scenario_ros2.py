#!/usr/bin/python3
"""Replay one manipulator benchmark trajectory through a ROS 2 controller.

The MuJoCo/ros2_control process must already expose a
std_msgs/msg/Float64MultiArray subscriber on /position_controller/commands.
The CSV state order is q1..q6 followed by q_dot1..q_dot6; only q is sent to
the position controller.
"""

"python3 replay_manipulator_scenario_ros2.py   --solver bicmppi   --scenario 3  --speed 0.3"

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# model/ur5e.xml: ur5e default range, with elbow's size3_limited override.
Q_MIN = (-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319)
Q_MAX = (6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a benchmark executed_x.csv on a MuJoCo ROS 2 position "
            "controller. Defaults to MPPI scenario 00."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("result/manipulator"),
        help="benchmark output root (default: result/manipulator)",
    )
    parser.add_argument(
        "--solver",
        choices=("mppi", "logmppi", "clustermppi", "bicmppi"),
        default="mppi",
    )
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="explicit executed_x.csv; overrides benchmark-dir/solver/scenario",
    )
    parser.add_argument(
        "--topic",
        default="/position_controller/commands",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="trajectory sample interval in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="hold the first pose before replay (default: 2.0)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="hold the final pose before exit/repeat (default: 2.0)",
    )
    parser.add_argument(
        "--subscriber-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the controller subscriber (default: 10)",
    )
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument(
        "--show-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw saved EE rollouts in the MuJoCo viewer (default: enabled)",
    )
    parser.add_argument(
        "--rollout-topic",
        default="/ee_rollout_paths",
        help="MarkerArray topic consumed by the patched MuJoCo bridge",
    )
    parser.add_argument(
        "--rollout-data",
        type=Path,
        help="explicit scenario_XX_ee_packed.bin.zst archive",
    )
    parser.add_argument("--rollout-branch", default="forward")
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=32,
        help="evenly sampled paths shown per iteration (default: 32)",
    )
    parser.add_argument(
        "--guide-rollout-count",
        type=int,
        default=8,
        help="paths shown for each BiC-MPPI guide branch (default: 8)",
    )
    parser.add_argument(
        "--rollout-point-stride",
        type=int,
        default=2,
        help="keep every Nth EE point in each line (default: 2)",
    )
    parser.add_argument("--rollout-line-width", type=float, default=0.003)
    parser.add_argument("--rollout-alpha", type=float, default=0.35)
    args = parser.parse_args(argv)
    if args.scenario < 0:
        parser.error("--scenario must be non-negative")
    if args.dt <= 0.0 or args.speed <= 0.0:
        parser.error("--dt and --speed must be positive")
    if args.settle_seconds < 0.0 or args.hold_seconds < 0.0:
        parser.error("hold durations must be non-negative")
    if args.subscriber_timeout < 0.0:
        parser.error("--subscriber-timeout must be non-negative")
    if (
        args.rollout_count <= 0
        or args.guide_rollout_count <= 0
        or args.rollout_point_stride <= 0
    ):
        parser.error("rollout counts and --rollout-point-stride must be positive")
    if args.rollout_line_width <= 0.0:
        parser.error("--rollout-line-width must be positive")
    if not 0.0 <= args.rollout_alpha <= 1.0:
        parser.error("--rollout-alpha must be between 0 and 1")
    return args


def default_trajectory_path(args: argparse.Namespace) -> Path:
    tag = f"scenario_{args.scenario:02d}_executed_x.csv"
    return args.benchmark_dir / "trajectories" / args.solver / tag


def default_rollout_path(args: argparse.Namespace) -> Path:
    tag = f"scenario_{args.scenario:02d}_ee_packed.bin.zst"
    return args.benchmark_dir / "rollout_ee" / args.solver / tag


class RolloutEEArchive:
    """One-time decompression plus cheap per-iteration rollout decoding."""

    def __init__(self, data_path: Path, np_module) -> None:
        self.data_path = data_path
        self.index_path = data_path.with_name(
            data_path.name.replace("_ee_packed.bin.zst", "_index.csv")
        )
        if not data_path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(
                f"rollout archive/index not found: {data_path}, {self.index_path}"
            )
        self.np = np_module
        with self.index_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = {
            (int(row["iteration"]), row["branch"]): row for row in rows
        }
        self.branches = sorted({row["branch"] for row in rows})
        try:
            self.raw = subprocess.run(
                ["zstd", "-q", "-d", "-c", str(data_path)],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"failed to decompress {data_path}: {error}") from error

    def max_iteration(self, branch: str) -> int:
        iterations = [iteration for iteration, name in self.rows if name == branch]
        if not iterations:
            raise ValueError(
                f"branch {branch!r} absent from {self.index_path}; "
                f"available={self.branches}"
            )
        return max(iterations)

    def load(self, iteration: int, branch: str):
        max_iteration = self.max_iteration(branch)
        min_iteration = min(i for i, name in self.rows if name == branch)
        iteration = min(max(iteration, min_iteration), max_iteration)
        row = self.rows[(iteration, branch)]
        offset = int(row["uncompressed_offset_bytes"])
        size = int(row["uncompressed_bytes"])
        rollout_count = int(row["rollout_count"])
        point_count = int(row["point_count"])
        scale = float(row["coordinate_scale_m"])
        if row["layout"] == "delta-packed-rollout-time-xyz":
            record_size = 6 + (point_count - 1) * 3
            records = self.np.frombuffer(
                self.raw, dtype=self.np.uint8, count=size, offset=offset
            ).reshape(rollout_count, record_size)
            initial = records[:, :6].copy().view("<i2").reshape(
                rollout_count, 1, 3
            )
            deltas = records[:, 6:].view(self.np.int8).reshape(
                rollout_count, point_count - 1, 3
            )
            encoded = self.np.concatenate((initial, deltas), axis=1)
            encoded = self.np.cumsum(encoded, axis=1, dtype=self.np.int32)
        else:
            values = self.np.frombuffer(
                self.raw, dtype="<i2", count=size // 2, offset=offset
            )
            encoded = values.reshape(rollout_count, point_count, 3)
            if row["layout"] == "delta-rollout-time-xyz":
                encoded = self.np.cumsum(encoded, axis=1, dtype=self.np.int32)
        return encoded.astype(self.np.float32) * scale


def state_columns(fieldnames: Sequence[str] | None) -> tuple[str, ...]:
    if not fieldnames:
        raise ValueError("trajectory CSV has no header")
    # writeMatrixCsv() uses x1..x12. Also accept explicit q1..q6 files.
    if all(f"x{i}" in fieldnames for i in range(1, 7)):
        return tuple(f"x{i}" for i in range(1, 7))
    if all(f"q{i}" in fieldnames for i in range(1, 7)):
        return tuple(f"q{i}" for i in range(1, 7))
    raise ValueError("trajectory CSV must contain x1..x6 or q1..q6")


def load_trajectory(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"trajectory not found: {path}\n"
            "Run ./manipulator_rsndom.sh first, or pass --trajectory."
        )
    samples: list[list[float]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = state_columns(reader.fieldnames)
        for row_index, row in enumerate(reader):
            q = [float(row[column]) for column in columns]
            if not all(math.isfinite(value) for value in q):
                raise ValueError(f"non-finite q at CSV row {row_index + 2}")
            for joint, value, lower, upper in zip(
                JOINT_NAMES, q, Q_MIN, Q_MAX
            ):
                if value < lower or value > upper:
                    raise ValueError(
                        f"{joint}={value} outside UR5e range "
                        f"[{lower}, {upper}] at CSV row {row_index + 2}"
                    )
            samples.append(q)
    if not samples:
        raise ValueError(f"trajectory has no samples: {path}")
    return samples


def load_scenario_description(
    benchmark_dir: Path, scenario_index: int
) -> str | None:
    path = benchmark_dir / "manipulator_random_pose_benchmark_scenarios.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["scenario"]) != scenario_index:
                continue
            obstacle = (
                row.get("obstacle_xmin"),
                row.get("obstacle_xmax"),
                row.get("obstacle_ymin"),
                row.get("obstacle_ymax"),
                row.get("obstacle_zmin"),
                row.get("obstacle_zmax"),
            )
            return (
                f"{row['init_name']} -> {row['goal_name']}; "
                f"obstacle bounds={obstacle}"
            )
    return None


def load_scenario_row(
    benchmark_dir: Path, scenario_index: int
) -> dict[str, str] | None:
    path = benchmark_dir / "manipulator_random_pose_benchmark_scenarios.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["scenario"]) == scenario_index:
                return row
    return None


def executed_ee_path(samples, scenario_row, np_module):
    """Convert executed joint states with the DH set used to create the data."""
    candidates = (
        ("ur5e", (0.425, 0.392), (0.163, 0.127, 0.1, 0.1)),
        ("legacy_rb5", (0.427, 0.357), (0.15, 0.11, 0.09, 0.09)),
    )
    alpha = (math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0)

    def fk(q, lengths, offsets):
        a = (0.0, -lengths[0], -lengths[1], 0.0, 0.0, 0.0)
        d = (offsets[0], 0.0, 0.0, offsets[1], offsets[2], offsets[3])
        transform = np_module.eye(4, dtype=np_module.float64)
        for theta, link_a, link_d, link_alpha in zip(q, a, d, alpha):
            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(link_alpha), math.sin(link_alpha)
            transform = transform @ np_module.array(
                (
                    (ct, -st * ca, st * sa, link_a * ct),
                    (st, ct * ca, -ct * sa, link_a * st),
                    (0.0, sa, ca, link_d),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        return transform[:3, 3]

    chosen = candidates[0]
    if scenario_row is not None:
        reference = np_module.array(
            [
                float(scenario_row["init_ee_x"]),
                float(scenario_row["init_ee_y"]),
                float(scenario_row["init_ee_z"]),
            ]
        )
        chosen = min(
            candidates,
            key=lambda candidate: np_module.linalg.norm(
                fk(samples[0], candidate[1], candidate[2]) - reference
            ),
        )
    positions = np_module.stack(
        [fk(q, chosen[1], chosen[2]) for q in samples], axis=0
    )
    return positions, chosen[0]


def load_scenario_obstacle(
    benchmark_dir: Path, scenario_index: int
) -> tuple[float, float, float, float, float, float] | None:
    """Return the scenario AABB as xmin,xmax,ymin,ymax,zmin,zmax."""
    path = benchmark_dir / "manipulator_random_pose_benchmark_scenarios.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["scenario"]) == scenario_index:
                return tuple(
                    float(row[name])
                    for name in (
                        "obstacle_xmin",
                        "obstacle_xmax",
                        "obstacle_ymin",
                        "obstacle_ymax",
                        "obstacle_zmin",
                        "obstacle_zmax",
                    )
                )
    return None


def main() -> int:
    # ROS 2 Humble on Ubuntu 22.04 is built for the system Python 3.10. The
    # user's `python3` may resolve to /usr/local/bin/python3 (currently 3.14),
    # which cannot load Humble's CPython extension. Re-exec with the same
    # interpreter used by /opt/ros/humble/bin/ros2.
    ros_python = Path("/usr/bin/python3")
    if sys.version_info[:2] != (3, 10) and ros_python.is_file():
        os.execv(
            str(ros_python),
            [str(ros_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    try:
        import numpy as np
        import rclpy
        from geometry_msgs.msg import Point
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.utilities import remove_ros_args
        from std_msgs.msg import Float64MultiArray
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError as error:
        print(
            "ROS 2 Python packages are unavailable. Source your ROS 2 setup "
            "before running this script.",
            file=sys.stderr,
        )
        print(error, file=sys.stderr)
        return 2

    ros_free_args = remove_ros_args(args=sys.argv)[1:]
    args = parse_args(ros_free_args)
    trajectory_path = (args.trajectory or default_trajectory_path(args)).resolve()
    try:
        samples = load_trajectory(trajectory_path)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    rollout_archive = None
    rollout_path = (args.rollout_data or default_rollout_path(args)).resolve()
    rollout_branches: list[str] = []
    if args.show_rollouts:
        try:
            rollout_archive = RolloutEEArchive(rollout_path, np)
            rollout_archive.max_iteration(args.rollout_branch)
            rollout_branches.append(args.rollout_branch)
            if args.solver == "bicmppi" and "backward" not in rollout_branches:
                rollout_archive.max_iteration("backward")
                rollout_branches.append("backward")
            if args.solver == "bicmppi":
                rollout_branches.extend(
                    branch
                    for branch in rollout_archive.branches
                    if branch.startswith("guide_")
                    and branch not in rollout_branches
                )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"warning: EE rollout lines disabled: {error}", file=sys.stderr)
            rollout_archive = None
            rollout_branches.clear()

    scenario_obstacle = load_scenario_obstacle(
        args.benchmark_dir.resolve(), args.scenario
    )
    scenario_row = load_scenario_row(
        args.benchmark_dir.resolve(), args.scenario
    )
    optimal_ee_path, optimal_fk_model = executed_ee_path(
        samples, scenario_row, np
    )

    class ScenarioReplayNode(Node):
        def __init__(self) -> None:
            super().__init__("manipulator_benchmark_scenario_replay")
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.publisher = self.create_publisher(
                Float64MultiArray, args.topic, qos
            )
            marker_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.rollout_publisher = self.create_publisher(
                MarkerArray, args.rollout_topic, marker_qos
            )
            self.period = args.dt / args.speed
            self.settle_ticks = math.ceil(args.settle_seconds / self.period)
            self.hold_ticks = math.ceil(args.hold_seconds / self.period)
            self.tick = 0
            self.sample_index = 0
            self.phase = "settle"
            self.timer = None
            self.published_rollout_iteration = None
            self.done = False

        def publish_q(self, q: Sequence[float]) -> None:
            message = Float64MultiArray()
            message.data = list(q)
            self.publisher.publish(message)

        def publish_rollouts(self, iteration: int) -> None:
            if iteration == self.published_rollout_iteration:
                return

            marker_array = MarkerArray()
            now = self.get_clock().now().to_msg()
            branch_colors = (
                {
                    "forward": (1.00, 0.24, 0.62),
                    "backward": (0.10, 0.82, 0.92),
                }
                if args.solver == "bicmppi"
                else {"forward": (0.10, 0.72, 1.00)}
            )
            marker_id = 0
            if rollout_archive is not None:
                for branch in rollout_branches:
                    branch_iteration = min(
                        iteration, rollout_archive.max_iteration(branch)
                    )
                    positions = rollout_archive.load(branch_iteration, branch)
                    requested_count = (
                        args.guide_rollout_count
                        if branch.startswith("guide_")
                        else args.rollout_count
                    )
                    count = min(requested_count, positions.shape[0])
                    selected = np.linspace(
                        0, positions.shape[0] - 1, count, dtype=np.int64
                    )
                    color = (
                        (0.18, 0.95, 0.28)
                        if branch.startswith("guide_")
                        else branch_colors.get(branch, (0.95, 0.72, 0.15))
                    )
                    for rollout_index in selected:
                        path = positions[rollout_index]
                        point_indices = list(
                            range(0, path.shape[0], args.rollout_point_stride)
                        )
                        if point_indices[-1] != path.shape[0] - 1:
                            point_indices.append(path.shape[0] - 1)

                        marker = Marker()
                        marker.header.stamp = now
                        marker.header.frame_id = "base_link"
                        marker.ns = f"ee_rollouts_{branch}"
                        marker.id = marker_id
                        marker_id += 1
                        marker.type = Marker.LINE_STRIP
                        marker.action = Marker.ADD
                        marker.pose.orientation.w = 1.0
                        marker.scale.x = args.rollout_line_width
                        marker.color.r = color[0]
                        marker.color.g = color[1]
                        marker.color.b = color[2]
                        marker.color.a = args.rollout_alpha
                        marker.points = [
                            Point(
                                x=float(path[i, 0]),
                                y=float(path[i, 1]),
                                z=float(path[i, 2]),
                            )
                            for i in point_indices
                        ]
                        marker_array.markers.append(marker)

            # The solver's accepted receding-horizon solution. The benchmark
            # does not store Uo for every iteration, so executed_x.csv is the
            # authoritative optimized path available for every solver.
            optimal = Marker()
            optimal.header.stamp = now
            optimal.header.frame_id = "base_link"
            optimal.ns = f"optimal_path_{args.solver}"
            optimal.id = marker_id
            marker_id += 1
            optimal.type = Marker.LINE_STRIP
            optimal.action = Marker.ADD
            optimal.pose.orientation.w = 1.0
            optimal.scale.x = args.rollout_line_width * 3.0
            optimal.color.r = 1.0
            optimal.color.g = 0.88
            optimal.color.b = 0.05
            optimal.color.a = 1.0
            optimal.points = [
                Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                for point in optimal_ee_path
            ]
            marker_array.markers.append(optimal)

            if scenario_obstacle is not None:
                xmin, xmax, ymin, ymax, zmin, zmax = scenario_obstacle
                obstacle = Marker()
                obstacle.header.stamp = now
                obstacle.header.frame_id = "base_link"
                obstacle.ns = "scenario_aabb"
                obstacle.id = marker_id
                obstacle.type = Marker.CUBE
                obstacle.action = Marker.ADD
                obstacle.pose.position.x = 0.5 * (xmin + xmax)
                obstacle.pose.position.y = 0.5 * (ymin + ymax)
                obstacle.pose.position.z = 0.5 * (zmin + zmax)
                obstacle.pose.orientation.w = 1.0
                obstacle.scale.x = 2*(xmax - xmin)
                obstacle.scale.y = 2*(ymax - ymin)
                obstacle.scale.z = 2*(zmax - zmin)
                obstacle.color.r = 0.90
                obstacle.color.g = 0.18
                obstacle.color.b = 0.10
                obstacle.color.a = 0.55
                marker_array.markers.append(obstacle)

            self.rollout_publisher.publish(marker_array)
            self.published_rollout_iteration = iteration

        def start(self) -> None:
            self.get_logger().info(f"trajectory: {trajectory_path}")
            self.get_logger().info(f"topic: {args.topic}")
            self.get_logger().info(f"joint order: {list(JOINT_NAMES)}")
            if rollout_archive is not None:
                self.get_logger().info(
                    f"EE rollout lines: {rollout_path} "
                    f"({args.rollout_count} forward/backward, "
                    f"{args.guide_rollout_count} per guide, "
                    f"branches={rollout_branches})"
                )
                if self.rollout_publisher.get_subscription_count() == 0:
                    self.get_logger().warning(
                        f"no MuJoCo rollout-line subscriber on {args.rollout_topic}; "
                        "rebuild/restart the patched mujoco_ros2_control node"
                    )
            self.get_logger().info(
                f"highlighted optimal executed path: {len(optimal_ee_path)} "
                f"points (FK={optimal_fk_model})"
            )
            description = load_scenario_description(
                args.benchmark_dir.resolve(), args.scenario
            )
            if description:
                self.get_logger().info(f"scenario {args.scenario}: {description}")
            if scenario_obstacle is not None:
                self.get_logger().info(
                    f"scenario AABB is rendered from CSV bounds: "
                    f"{scenario_obstacle}"
                )
            else:
                self.get_logger().warning(
                    "scenario AABB bounds were not found; obstacle is not rendered"
                )
            self.publish_rollouts(0)
            self.timer = self.create_timer(self.period, self.on_timer)

        def on_timer(self) -> None:
            if self.phase == "settle":
                self.publish_q(samples[0])
                self.tick += 1
                if self.tick >= self.settle_ticks:
                    self.phase = "play"
                    self.tick = 0
                    self.sample_index = 0
                    self.get_logger().info(
                        f"replaying {len(samples)} samples at {self.period:.6f} s"
                    )
                return

            if self.phase == "play":
                self.publish_rollouts(self.sample_index)
                self.publish_q(samples[self.sample_index])
                self.sample_index += 1
                if self.sample_index >= len(samples):
                    self.phase = "hold"
                    self.tick = 0
                    self.get_logger().info("trajectory complete; holding goal")
                return

            self.publish_q(samples[-1])
            self.tick += 1
            if self.tick < self.hold_ticks:
                return
            if args.repeat:
                self.phase = "settle"
                self.tick = 0
                self.sample_index = 0
                self.get_logger().info("restarting trajectory")
            else:
                self.get_logger().info("replay finished")
                self.done = True
                self.timer.cancel()

    rclpy.init(args=sys.argv)
    node = ScenarioReplayNode()
    try:
        deadline = time.monotonic() + args.subscriber_timeout
        while rclpy.ok() and node.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                node.get_logger().error(
                    f"no subscriber on {args.topic} after "
                    f"{args.subscriber_timeout:.1f} s"
                )
                return 1
            rclpy.spin_once(node, timeout_sec=0.1)
        node.get_logger().info("position controller subscriber connected")
        node.start()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        node.get_logger().info("replay interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
