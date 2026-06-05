#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


SOLVERS = [
    ("mppi", "MPPI", "#1f77b4"),
    ("bi_mppi", "Bi-MPPI", "#2ca02c"),
]

OBSTACLES = [
    (-0.50, -0.50, 5.00, 0.50),
    (-0.50, 3.00, 5.00, 0.50),
    (-0.50, -0.50, 0.50, 4.00),
    (4.00, -0.50, 0.50, 4.00),
    (1.05, 1.45, 0.82, 0.72),
    (3.15, 1.45, 0.82, 0.72),
    (1.95, 2.38, 1.10, 0.18),
]

START = np.array([0.55, 0.65, 0.0])
TARGET = np.array([2.55, 1.82, np.pi])
REQ_FORWARD = 1.05
REQ_REVERSE = 0.95


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows, name):
    return np.array([float(row[name]) for row in rows], dtype=float)


def load_solver(input_dir, name):
    return {
        "path": read_csv(input_dir / f"path_parking_hybrid_{name}.csv"),
        "control": read_csv(input_dir / f"control_parking_hybrid_{name}.csv"),
        "result": read_csv(input_dir / f"result_parking_hybrid_{name}.csv")[0],
    }


def draw_heading(ax, state, color, label):
    x, y, theta = state
    ax.arrow(
        x,
        y,
        0.22 * np.cos(theta),
        0.22 * np.sin(theta),
        head_width=0.06,
        head_length=0.08,
        color=color,
        length_includes_head=True,
        label=label,
    )


def plot(input_dir, output):
    data = {name: load_solver(input_dir, name) for name, _, _ in SOLVERS}

    fig = plt.figure(figsize=(11, 8), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.0, 1.0], hspace=0.34)
    ax_path = fig.add_subplot(gs[0, :])
    ax_speed = fig.add_subplot(gs[1, :])
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_score = fig.add_subplot(gs[2, 1])

    for x, y, w, h in OBSTACLES:
      ax_path.add_patch(
          Rectangle((x, y), w, h, facecolor="#222222", edgecolor="none", alpha=0.85)
      )

    draw_heading(ax_path, START, "#555555", "start")
    draw_heading(ax_path, TARGET, "#d62728", "target")

    for name, label, color in SOLVERS:
        rows = data[name]["path"]
        x = col(rows, "x")
        y = col(rows, "y")
        ax_path.plot(x, y, color=color, lw=2.2, label=label)
        ax_path.scatter(x[-1], y[-1], color=color, s=35, zorder=4)

        control_rows = data[name]["control"]
        it = col(control_rows, "iter")
        speed = col(control_rows, "signed_speed")
        ax_speed.plot(it, speed, color=color, lw=1.8, label=label)
        ax_speed.fill_between(it, 0.0, speed, where=speed >= 0.0, color=color, alpha=0.18)
        ax_speed.fill_between(it, 0.0, speed, where=speed < 0.0, color=color, alpha=0.08)

        pf = col(rows, "forward_distance")
        pr = col(rows, "reverse_distance")
        ax_dist.plot(col(rows, "iter"), pf, color=color, lw=1.8, label=f"{label} forward")
        ax_dist.plot(
            col(rows, "iter"),
            pr,
            color=color,
            lw=1.8,
            ls="--",
            label=f"{label} reverse",
        )

    ax_path.set_title("Hybrid parking trajectory")
    ax_path.set_xlim(0.0, 4.0)
    ax_path.set_ylim(0.0, 3.0)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.grid(True, alpha=0.25)
    ax_path.legend(loc="lower right", frameon=True)

    ax_speed.axhline(0.0, color="#333333", lw=1.0)
    ax_speed.set_title("Applied signed-speed control")
    ax_speed.set_ylabel("signed speed [m/s]")
    ax_speed.set_xlabel("MPC iteration")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(loc="upper right")
    ax_speed.text(
        0.01,
        0.92,
        "positive = forward, negative = reverse",
        transform=ax_speed.transAxes,
        fontsize=9,
        va="top",
    )

    ax_dist.axhline(REQ_FORWARD, color="#777777", lw=1.0, ls=":")
    ax_dist.axhline(REQ_REVERSE, color="#777777", lw=1.0, ls=":")
    ax_dist.set_title("Required mode coverage")
    ax_dist.set_xlabel("MPC iteration")
    ax_dist.set_ylabel("distance [m]")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(loc="lower right", fontsize=8)

    labels = []
    scores = []
    pose_errors = []
    for name, label, _ in SOLVERS:
        result = data[name]["result"]
        labels.append(label)
        scores.append(float(result["score"]))
        pose_errors.append(float(result["pose_error"]))

    x = np.arange(len(labels))
    width = 0.34
    ax_score.bar(x - width / 2, scores, width, label="score", color="#8c564b")
    ax_score.bar(x + width / 2, pose_errors, width, label="pose error", color="#9467bd")
    ax_score.set_xticks(x, labels)
    ax_score.set_title("Lower is better")
    ax_score.grid(True, axis="y", alpha=0.25)
    ax_score.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"saved {output}")

    for name, label, _ in SOLVERS:
        result = data[name]["result"]
        controls = data[name]["control"]
        speed = col(controls, "signed_speed")
        print(
            f"{label}: success={result['is_success']} failed={result['is_failed']} "
            f"iter={result['iter']} score={float(result['score']):.4f} "
            f"pose_error={float(result['pose_error']):.4f} "
            f"forward={float(result['forward_distance']):.3f} "
            f"reverse={float(result['reverse_distance']):.3f} "
            f"has_pos={bool(np.any(speed > 1e-3))} "
            f"has_neg={bool(np.any(speed < -1e-3))}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("build"))
    parser.add_argument(
        "--output", type=Path, default=Path("build/plots/parking_hybrid_comparison.png")
    )
    args = parser.parse_args()
    plot(args.input_dir, args.output)


if __name__ == "__main__":
    main()
