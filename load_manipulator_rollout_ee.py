#!/usr/bin/env python3
"""Load one iteration/branch from the manipulator rollout EE archive."""

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
    """Return EE positions with shape (rollout, time, xyz), in metres."""
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
    raw = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(data_path)],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    offset = int(row["uncompressed_offset_bytes"])
    size = int(row["uncompressed_bytes"])
    rollout_count = int(row["rollout_count"])
    point_count = int(row["point_count"])
    scale = float(row["coordinate_scale_m"])
    if row["layout"] == "delta-packed-rollout-time-xyz":
        record_size = 6 + (point_count - 1) * 3
        records = np.frombuffer(
            raw, dtype=np.uint8, count=size, offset=offset
        ).reshape(rollout_count, record_size)
        initial = records[:, :6].copy().view("<i2").reshape(rollout_count, 1, 3)
        deltas = records[:, 6:].view(np.int8).reshape(
            rollout_count, point_count - 1, 3
        )
        encoded = np.concatenate((initial, deltas), axis=1)
        encoded = np.cumsum(encoded, axis=1, dtype=np.int32)
    else:
        values = np.frombuffer(raw, dtype="<i2", count=size // 2, offset=offset)
        encoded = values.reshape(rollout_count, point_count, 3)
        if row["layout"] == "delta-rollout-time-xyz":
            encoded = np.cumsum(encoded, axis=1, dtype=np.int32)
    return encoded.astype(np.float32) * scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="scenario_XX_ee_packed.bin.zst")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--branch", default="forward")
    parser.add_argument("--output", type=Path, help="optional .npy output")
    args = parser.parse_args()

    index_path = args.index or args.data.with_name(
        args.data.name.replace("_ee_packed.bin.zst", "_index.csv")
    )
    positions = load_batch(args.data, index_path, args.iteration, args.branch)
    if args.output:
        np.save(args.output, positions)
    print(
        f"shape={positions.shape} min={positions.min(axis=(0, 1))} "
        f"max={positions.max(axis=(0, 1))}"
    )


if __name__ == "__main__":
    main()
