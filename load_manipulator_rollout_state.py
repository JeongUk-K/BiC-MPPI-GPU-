#!/usr/bin/env python3
"""Load one q + qdot rollout batch from a manipulator state archive."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_batch(
    data_path: Path,
    index_path: Path,
    iteration: int,
    branch: str = "forward",
) -> np.ndarray:
    """Return float32 states shaped (rollout, time, q_and_qdot)."""
    matches = [
        row
        for row in read_index(index_path)
        if int(row["iteration"]) == iteration and row["branch"] == branch
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one batch for iteration={iteration}, branch={branch!r}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    compressed_offset = int(row["compressed_offset_bytes"])
    compressed_bytes = int(row["compressed_bytes"])
    with data_path.open("rb") as stream:
        stream.seek(compressed_offset)
        frame = stream.read(compressed_bytes)
    if len(frame) != compressed_bytes:
        raise ValueError("state archive is shorter than its index")
    raw = subprocess.run(
        ["zstd", "-q", "-d", "-c"],
        input=frame,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    expected_bytes = int(row["uncompressed_bytes"])
    if len(raw) != expected_bytes:
        raise ValueError(
            f"decompressed {len(raw)} bytes; expected {expected_bytes}"
        )
    rollout_count = int(row["rollout_count"])
    point_count = int(row["point_count"])
    state_dim = int(row["state_dim"])
    return (
        np.frombuffer(raw, dtype="<f2")
        .reshape(rollout_count, point_count, state_dim)
        .astype(np.float32)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="scenario_XX_state_f16.bin.zst")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--branch", default="forward")
    parser.add_argument("--output", type=Path, help="optional .npy output")
    args = parser.parse_args()
    index_path = args.index or args.data.with_name(
        args.data.name.replace("_state_f16.bin.zst", "_index.csv")
    )
    states = load_batch(args.data, index_path, args.iteration, args.branch)
    if args.output:
        np.save(args.output, states)
    dof = states.shape[-1] // 2
    print(
        f"shape={states.shape} "
        f"q_range=({states[..., :dof].min():.6g}, {states[..., :dof].max():.6g}) "
        f"qdot_range=({states[..., dof:].min():.6g}, "
        f"{states[..., dof:].max():.6g})"
    )


if __name__ == "__main__":
    main()

