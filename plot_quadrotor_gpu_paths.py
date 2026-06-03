#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = [
    ("mppi", "MPPI", "#1f77b4"),
    ("log_mppi", "Log-MPPI", "#9467bd"),
    ("cluster_mppi", "Cluster-MPPI", "#ff7f0e"),
    ("bi_mppi", "BiC-MPPI", "#2ca02c"),
    ("svgd_mppi", "SVGD-MPPI", "#d62728"),
]


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_paths(build_dir):
    by_variant = {}
    for variant, _, _ in VARIANTS:
        path_file = build_dir / f"path_quadrotor_{variant}.csv"
        maps = defaultdict(list)
        if not path_file.exists():
            by_variant[variant] = maps
            continue

        with path_file.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                map_id = int(row["map"])
                maps[map_id].append(
                    {
                        "iter": int(row["iter"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]),
                    }
                )

        for rows in maps.values():
            rows.sort(key=lambda item: item["iter"])
        by_variant[variant] = maps
    return by_variant


def load_results(build_dir):
    by_variant = {}
    for variant, _, _ in VARIANTS:
        result_file = build_dir / f"result_quadrotor_{variant}.csv"
        results = {}
        if not result_file.exists():
            by_variant[variant] = results
            continue

        with result_file.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[int(row["map"])] = row
        by_variant[variant] = results
    return by_variant


def load_map(map_dir, map_id):
    map_file = map_dir / f"output_{map_id}.txt"
    if not map_file.exists():
        return None
    return np.loadtxt(map_file)


def result_suffix(result):
    if not result:
        return ""
    iter_count = result.get("iter", "?")
    if parse_bool(result.get("is_failed", "0")):
        status = "failed"
    elif parse_bool(result.get("is_landed", "0")):
        status = "landed"
    else:
        status = "stopped"
    return f" ({status}, iter {iter_count})"


def plot_one_map(map_id, grid, paths, results, output_path, resolution, dpi):
    fig, ax = plt.subplots(figsize=(7.0, 9.0), dpi=dpi)

    if grid is not None:
        rows, cols = grid.shape
        extent = (0, rows * resolution, 0, cols * resolution)
        obstacles = (grid == 10).T
        ax.imshow(
            obstacles,
            origin="lower",
            extent=extent,
            cmap="Greys",
            interpolation="nearest",
            alpha=0.45,
            aspect="equal",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(0, 3.0)
        ax.set_ylim(0, 5.0)

    plotted = 0
    for variant, label, color in VARIANTS:
        rows = paths.get(variant, {}).get(map_id, [])
        if not rows:
            continue
        xs = [row["x"] for row in rows]
        ys = [row["y"] for row in rows]
        zs = [row["z"] for row in rows]
        suffix = result_suffix(results.get(variant, {}).get(map_id))
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            label=f"{label}{suffix}, z_end={zs[-1]:.2f}",
        )
        ax.scatter(xs[0], ys[0], color=color, s=18, marker="o")
        ax.scatter(xs[-1], ys[-1], color=color, s=28, marker="x")
        plotted += 1

    ax.scatter([1.5], [0.0], color="black", s=45, marker="o", label="start")
    ax.scatter([1.5], [5.0], color="black", s=70, marker="*", label="target")

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No quadrotor path CSV found for this map",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    ax.set_title(f"Quadrotor GPU MPPI Variants - Map {map_id:03d}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot quadrotor GPU MPPI variant paths for every BARN map."
    )
    parser.add_argument("--build-dir", default="build", type=Path)
    parser.add_argument("--map-dir", default="BARN_dataset/txt_files", type=Path)
    parser.add_argument(
        "--output-dir", default="build/plots/quadrotor_gpu", type=Path
    )
    parser.add_argument("--maps", default=300, type=int)
    parser.add_argument("--resolution", default=0.1, type=float)
    parser.add_argument("--dpi", default=150, type=int)
    args = parser.parse_args()

    paths = load_paths(args.build_dir)
    results = load_results(args.build_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for map_id in range(args.maps):
        grid = load_map(args.map_dir, map_id)
        output_path = args.output_dir / f"quadrotor_map_{map_id:03d}.png"
        plot_one_map(
            map_id,
            grid,
            paths,
            results,
            output_path,
            args.resolution,
            args.dpi,
        )

    print(f"Saved {args.maps} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
