#!/usr/bin/env python3
"""Rerun only Chapter 15 Nsight geometry profiles for an existing run."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_module(root: Path):
    path = root / "scripts/39_ch15_channels_chunks_buffers.py"
    spec = importlib.util.spec_from_file_location("ch15_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    args = parser.parse_args()

    ch15 = load_module(args.root)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    private_dir = args.root / "private/ch15_pipeline" / args.run_dir.name
    rows = ch15.run_geometry_profiles(
        args.root, args.run_dir, private_dir, binary
    )
    ch15.write_csv(args.run_dir / "kernel_geometry.csv", rows)
    (args.run_dir / "geometry_manifest.txt").write_text(
        "geometry_profiles=7\n"
        "messages=1M,4K\n"
        f"private_nsys_dir={private_dir}\n",
        encoding="utf-8",
    )
    for row in rows:
        print(
            row["label"],
            f"bytes={row['message_bytes']}",
            f"grid={row['grid_x_values']}",
            f"block={row['block_x_values']}",
            f"matches_requested={row['grid_matches_requested_channels']}",
        )


if __name__ == "__main__":
    main()
