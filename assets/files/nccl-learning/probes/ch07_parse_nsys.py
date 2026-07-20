#!/usr/bin/env python3
"""Parse Nsight Systems NVTX-to-kernel summary for chapter 7."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse(path: Path, mode: str) -> list[dict[str, object]]:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith(f":{mode},"):
            continue
        values = next(csv.reader([raw_line]))
        rows.append(
            {
                "mode": mode,
                "nvtx_instances": int(values[4]),
                "kernel_instances": int(values[5]),
                "total_time_ns": int(values[6]),
                "average_ns": float(values[7]),
                "kernel_name": values[12],
            }
        )
    if not rows:
        raise SystemExit(f"no :{mode} NVTX kernel rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--decomposed", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = parse(args.direct, "direct") + parse(args.decomposed, "decomposed")
    totals = {
        mode: sum(int(row["kernel_instances"]) for row in rows if row["mode"] == mode)
        for mode in ("direct", "decomposed")
    }
    expected_direct = args.iterations * args.gpus
    expected_decomposed = expected_direct * 2
    if totals["direct"] != expected_direct:
        raise SystemExit(f"expected {expected_direct} direct kernels, got {totals['direct']}")
    if totals["decomposed"] != expected_decomposed:
        raise SystemExit(
            f"expected {expected_decomposed} decomposed kernels, got {totals['decomposed']}"
        )
    if not all("ncclDevKernel" in str(row["kernel_name"]) for row in rows):
        raise SystemExit("non-NCCL kernel unexpectedly attributed to measured NVTX range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"direct_kernel_instances={totals['direct']}")
    print(f"decomposed_kernel_instances={totals['decomposed']}")
    for row in rows:
        print(
            f"mode={row['mode']} instances={row['kernel_instances']} "
            f"kernel={row['kernel_name']}"
        )


if __name__ == "__main__":
    main()
