#!/usr/bin/env python3
"""Validate and benchmark AllReduce versus ReduceScatter plus AllGather."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def full_constant_check(tensor: torch.Tensor, expected: int | float) -> bool:
    # Bound temporary memory while still inspecting every element.
    chunk_elements = 1 << 20
    flat = tensor.reshape(-1)
    for offset in range(0, flat.numel(), chunk_elements):
        if not torch.all(flat[offset : offset + chunk_elements] == expected).item():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes-bytes", default="1024,65536,1048576,16777216,268435456")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--memory-bytes", type=int, default=268435456)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("chapter 7 requires exactly four ranks")

    contracts: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []

    def contract(name: str, correct: bool, detail: str) -> None:
        contracts.append({"rank": rank, "name": name, "correct": correct, "detail": detail})
        if not correct:
            raise AssertionError(f"contract failed: {name}: {detail}")

    # Exact, nonuniform integer input proves rank-major shard placement.
    exact_count = 20
    base = torch.arange(exact_count, dtype=torch.int64, device=device) + rank * 100
    direct = base.clone()
    dist.all_reduce(direct, op=dist.ReduceOp.SUM)
    shard = torch.empty(exact_count // world, dtype=torch.int64, device=device)
    gathered = torch.empty(exact_count, dtype=torch.int64, device=device)
    dist.reduce_scatter_tensor(shard, base, op=dist.ReduceOp.SUM)
    dist.all_gather_into_tensor(gathered, shard)
    torch.cuda.synchronize(device)
    expected = torch.arange(exact_count, dtype=torch.int64, device=device) * world + 600
    contract(
        "exact-sum-equivalence",
        torch.equal(direct, gathered) and torch.equal(gathered, expected),
        "int64 full tensor direct==RS+AG==CPU formula",
    )

    # AllGather is rank-major. Rotating its chunks is an explicit non-equivalent transform.
    wrong_order = gathered.reshape(world, -1).roll(shifts=1, dims=0).reshape(-1)
    contract(
        "wrong-shard-order-rejected",
        not torch.equal(direct, wrong_order),
        "rank-major chunks rotated by one rank",
    )

    max_shard = torch.empty_like(shard)
    max_gathered = torch.empty_like(gathered)
    dist.reduce_scatter_tensor(max_shard, base, op=dist.ReduceOp.MAX)
    dist.all_gather_into_tensor(max_gathered, max_shard)
    torch.cuda.synchronize(device)
    contract(
        "wrong-reduction-op-rejected",
        not torch.equal(direct, max_gathered),
        "direct SUM compared with ReduceScatter MAX plus AllGather",
    )

    # Equal-size ReduceScatter requires divisibility. Padding and trimming restores
    # equivalence for SUM when the padding identity is zero.
    logical_count = 1003
    padded_count = math.ceil(logical_count / world) * world
    logical = torch.arange(logical_count, dtype=torch.int64, device=device) + rank * 10000
    direct_uneven = logical.clone()
    dist.all_reduce(direct_uneven, op=dist.ReduceOp.SUM)
    padded = torch.zeros(padded_count, dtype=torch.int64, device=device)
    padded[:logical_count].copy_(logical)
    padded_shard = torch.empty(padded_count // world, dtype=torch.int64, device=device)
    padded_gathered = torch.empty(padded_count, dtype=torch.int64, device=device)
    dist.reduce_scatter_tensor(padded_shard, padded, op=dist.ReduceOp.SUM)
    dist.all_gather_into_tensor(padded_gathered, padded_shard)
    torch.cuda.synchronize(device)
    contract(
        "padding-and-trim-equivalence",
        torch.equal(direct_uneven, padded_gathered[:logical_count]),
        f"logical={logical_count};padded={padded_count};identity=0",
    )

    invalid_rejected = False
    invalid_message = ""
    try:
        invalid_output = torch.empty(logical_count // world, dtype=torch.int64, device=device)
        dist.reduce_scatter_tensor(invalid_output, logical, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
    except Exception as error:
        invalid_rejected = True
        invalid_message = str(error).replace("\n", " ")[:400]
    contract(
        "nondivisible-without-padding-rejected",
        invalid_rejected,
        invalid_message,
    )

    del base, direct, shard, gathered, expected, wrong_order, max_shard, max_gathered
    del logical, direct_uneven, padded, padded_shard, padded_gathered
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    sizes = [int(value) for value in args.sizes_bytes.split(",")]
    for size_bytes in sizes:
        if size_bytes % (world * 4) != 0:
            raise ValueError("benchmark byte sizes must divide by world_size*sizeof(float32)")
        elements = size_bytes // 4
        direct_input = torch.empty(elements, dtype=torch.float32, device=device)
        decomp_input = torch.empty(elements, dtype=torch.float32, device=device)
        decomp_shard = torch.empty(elements // world, dtype=torch.float32, device=device)
        decomp_output = torch.empty(elements, dtype=torch.float32, device=device)

        def run_direct() -> None:
            dist.all_reduce(direct_input, op=dist.ReduceOp.SUM)

        def run_decomposed() -> None:
            dist.reduce_scatter_tensor(decomp_shard, decomp_input, op=dist.ReduceOp.SUM)
            dist.all_gather_into_tensor(decomp_output, decomp_shard)

        for _ in range(args.warmups):
            direct_input.fill_(rank + 1)
            run_direct()
            decomp_input.fill_(rank + 1)
            run_decomposed()
        torch.cuda.synchronize(device)

        for cycle in range(args.cycles):
            modes = ("direct", "decomposed") if cycle % 2 == 0 else ("decomposed", "direct")
            for mode in modes:
                if mode == "direct":
                    direct_input.fill_(rank + 1)
                    operation = run_direct
                else:
                    decomp_input.fill_(rank + 1)
                    operation = run_decomposed
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                end.record()
                end.synchronize()
                timing_rows.append(
                    {
                        "rank": rank,
                        "size_bytes": size_bytes,
                        "mode": mode,
                        "cycle": cycle,
                        "duration_us": start.elapsed_time(end) * 1000.0,
                        "correct": True,
                    }
                )

        direct_correct = full_constant_check(direct_input, 10.0)
        decomp_correct = full_constant_check(decomp_output, 10.0)
        if not direct_correct or not decomp_correct or not torch.equal(direct_input, decomp_output):
            raise AssertionError(f"benchmark correctness failed at {size_bytes} bytes")

        del direct_input, decomp_input, decomp_shard, decomp_output
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    memory_elements = args.memory_bytes // 4
    if args.memory_bytes % (world * 4) != 0:
        raise ValueError("memory bytes must divide by world_size*sizeof(float32)")

    def clean_baseline() -> int:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)

    def memory_record(mode: str, baseline: int, steady: int, peak: int, correct: bool) -> None:
        memory_rows.append(
            {
                "rank": rank,
                "mode": mode,
                "payload_bytes": args.memory_bytes,
                "baseline_bytes": baseline,
                "steady_delta_bytes": steady - baseline,
                "peak_delta_bytes": peak - baseline,
                "correct": correct,
            }
        )
        if not correct:
            raise AssertionError(f"memory mode {mode} correctness failed")

    dist.barrier()
    baseline = clean_baseline()
    torch.cuda.reset_peak_memory_stats(device)
    full = torch.full((memory_elements,), float(rank + 1), device=device)
    dist.all_reduce(full)
    torch.cuda.synchronize(device)
    memory_record(
        "allreduce-full",
        baseline,
        torch.cuda.memory_allocated(device),
        torch.cuda.max_memory_allocated(device),
        full_constant_check(full, 10.0),
    )
    del full

    dist.barrier()
    baseline = clean_baseline()
    torch.cuda.reset_peak_memory_stats(device)
    full = torch.full((memory_elements,), float(rank + 1), device=device)
    shard = torch.empty(memory_elements // world, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(shard, full)
    torch.cuda.synchronize(device)
    del full
    gc.collect()
    memory_record(
        "reduce-scatter-shard",
        baseline,
        torch.cuda.memory_allocated(device),
        torch.cuda.max_memory_allocated(device),
        full_constant_check(shard, 10.0),
    )
    del shard

    dist.barrier()
    baseline = clean_baseline()
    torch.cuda.reset_peak_memory_stats(device)
    full = torch.full((memory_elements,), float(rank + 1), device=device)
    shard = torch.empty(memory_elements // world, dtype=torch.float32, device=device)
    gathered = torch.empty(memory_elements, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(shard, full)
    dist.all_gather_into_tensor(gathered, shard)
    torch.cuda.synchronize(device)
    del full, shard
    gc.collect()
    memory_record(
        "rs-ag-naive-lifetime",
        baseline,
        torch.cuda.memory_allocated(device),
        torch.cuda.max_memory_allocated(device),
        full_constant_check(gathered, 10.0),
    )
    del gathered

    dist.barrier()
    baseline = clean_baseline()
    torch.cuda.reset_peak_memory_stats(device)
    full = torch.full((memory_elements,), float(rank + 1), device=device)
    shard = torch.empty(memory_elements // world, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(shard, full)
    torch.cuda.synchronize(device)
    del full
    gc.collect()
    gathered = torch.empty(memory_elements, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(gathered, shard)
    torch.cuda.synchronize(device)
    del shard
    gc.collect()
    memory_record(
        "rs-ag-optimized-lifetime",
        baseline,
        torch.cuda.memory_allocated(device),
        torch.cuda.max_memory_allocated(device),
        full_constant_check(gathered, 10.0),
    )
    del gathered

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("contracts", contracts),
        ("timings", timing_rows),
        ("memory", memory_rows),
    ):
        with (args.output_dir / f"rank{rank}_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    dist.barrier()
    if rank == 0:
        print(
            f"contracts_per_rank={len(contracts)} timings_per_rank={len(timing_rows)} "
            f"memory_modes_per_rank={len(memory_rows)}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
