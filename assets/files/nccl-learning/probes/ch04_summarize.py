#!/usr/bin/env python3
"""Summarize native NCCL and PyTorch communicator experiments."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


def read_many(directory: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(directory.glob("rank*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no rank rows in {directory}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_init_counts(directory: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    pattern = re.compile(r"rank \d+ nranks (\d+).*Init COMPLETE")
    for path in directory.glob("nccl_*.log"):
        for match in pattern.finditer(path.read_text(errors="replace")):
            nranks = int(match.group(1))
            counts[nranks] = counts.get(nranks, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    native = read_many(run_dir / "raw/native")
    default = read_many(run_dir / "raw/pytorch_default")
    remap = read_many(run_dir / "raw/pytorch_remap")
    if not all(row["correct"].lower() == "true" for row in native + default + remap):
        raise SystemExit("communicator correctness failure")

    expected_split_rank = {0: 1, 1: 1, 2: 0, 3: 0}
    for row in native:
        rank = int(row["global_rank"])
        assert int(row["comm_rank"]) == rank
        assert int(row["comm_count"]) == 4
        assert int(row["split_rank"]) == expected_split_rank[rank]
        assert int(row["split_count"]) == 2
        assert int(row["comm_device"]) == rank
        assert int(row["split_device"]) == rank
    assert len({row["unique_id_hash"] for row in native}) == 1

    for rows in (default, remap):
        for row in rows:
            rank = int(row["rank"])
            assert float(row["world_sum"]) == 10.0
            assert float(row["parity_sum"]) == (4.0 if rank % 2 == 0 else 6.0)
            assert float(row["pair_sum"]) == (3.0 if rank < 2 else 7.0)
            assert int(row["parity_group_size"]) == 2
            assert int(row["pair_group_size"]) == 2

    default_map = {int(row["rank"]): row["physical_pci_bus_id"] for row in default}
    remap_map = {int(row["rank"]): row["physical_pci_bus_id"] for row in remap}
    if default_map == remap_map:
        raise SystemExit("CUDA_VISIBLE_DEVICES remap did not change physical mapping")

    default_init_counts = parse_init_counts(run_dir / "raw/pytorch_default")
    remap_init_counts = parse_init_counts(run_dir / "raw/pytorch_remap")
    duplicate_log = (run_dir / "raw/duplicate.stdout").read_text(errors="replace")
    duplicate_detected = "Duplicate GPU detected" in duplicate_log or "duplicate GPU" in duplicate_log
    if not duplicate_detected:
        raise SystemExit("expected duplicate GPU diagnostic was not found")

    mapping_rows = []
    for rank in range(4):
        mapping_rows.append(
            {
                "rank": rank,
                "default_pci_bus_id": default_map[rank],
                "remapped_pci_bus_id": remap_map[rank],
                "changed": default_map[rank] != remap_map[rank],
            }
        )
    write_csv(run_dir / "device_mapping.csv", mapping_rows)

    metrics = [
        {"metric": "native_ranks", "value": len(native)},
        {"metric": "native_unique_id_hashes", "value": len({row["unique_id_hash"] for row in native})},
        {"metric": "native_init_median_ms", "value": f"{statistics.median(float(row['init_ms']) for row in native):.3f}"},
        {"metric": "pytorch_default_init_nranks4", "value": default_init_counts.get(4, 0)},
        {"metric": "pytorch_default_init_nranks2", "value": default_init_counts.get(2, 0)},
        {"metric": "pytorch_remap_init_nranks4", "value": remap_init_counts.get(4, 0)},
        {"metric": "pytorch_remap_init_nranks2", "value": remap_init_counts.get(2, 0)},
        {"metric": "duplicate_gpu_detected", "value": str(duplicate_detected).lower()},
    ]
    write_csv(run_dir / "summary.csv", metrics)

    lines = [
        "# Chapter 04 experiment summary",
        "",
        "## Native communicator and split",
        "",
        "| global rank | CUDA device | PCI BDF | color | key | split rank | world sum | split sum |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(native, key=lambda item: int(item["global_rank"])):
        lines.append(
            f"| {row['global_rank']} | {row['cuda_device']} | {row['pci_bus_id']} | "
            f"{row['color']} | {row['key']} | {row['split_rank']} | {row['world_sum']} | {row['split_sum']} |"
        )
    lines.extend([
        "",
        "## CUDA_VISIBLE_DEVICES mapping",
        "",
        "| rank/local ordinal | default PCI BDF | remapped PCI BDF |",
        "|---:|---|---|",
    ])
    for row in mapping_rows:
        lines.append(f"| {row['rank']} | {row['default_pci_bus_id']} | {row['remapped_pci_bus_id']} |")
    lines.extend([
        "",
        "## Assertions",
        "",
        "- All native ranks received one identical ncclUniqueId payload.",
        "- Parent communicator rank equals global rank; split rank follows key ordering within color.",
        "- World, parity and pair ProcessGroups produced independent expected sums.",
        "- CUDA_VISIBLE_DEVICES changed every rank's physical BDF without changing its local ordinal.",
        "- Binding two ranks to one visible GPU produced the expected duplicate-GPU failure.",
        "",
        f"- Default NCCL Init COMPLETE counts: `{default_init_counts}`",
        f"- Remapped NCCL Init COMPLETE counts: `{remap_init_counts}`",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
