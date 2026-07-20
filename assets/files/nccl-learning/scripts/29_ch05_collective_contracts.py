#!/usr/bin/env python3
"""Validate collective contracts with rank-encoded tensors and CPU references."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def encode(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().reshape(-1).tolist()
    return ";".join(str(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("this experiment requires exactly four ranks")

    rows: list[dict[str, object]] = []

    def check(
        operation: str,
        variant: str,
        actual: torch.Tensor,
        expected: torch.Tensor | None,
        note: str = "",
    ) -> None:
        torch.cuda.synchronize(device)
        if expected is None:
            correct = True
            expected_text = "undefined-on-this-rank"
        else:
            expected = expected.to(actual.device)
            correct = torch.equal(actual, expected)
            expected_text = encode(expected)
        rows.append(
            {
                "rank": rank,
                "operation": operation,
                "variant": variant,
                "shape": "x".join(map(str, actual.shape)) or "0",
                "dtype": str(actual.dtype).removeprefix("torch."),
                "actual": encode(actual),
                "expected": expected_text,
                "correct": bool(correct),
                "note": note,
            }
        )
        if not correct:
            raise AssertionError(
                f"rank={rank} operation={operation}/{variant} "
                f"actual={encode(actual)} expected={expected_text}"
            )

    value = torch.full((5,), rank + 1, dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    check("all_reduce", "sum-int64-in-place", value, torch.full((5,), 10, dtype=torch.int64))

    value = torch.tensor([rank - 2.0, rank * 3.0, 10.0 - rank], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    check("all_reduce", "max-float32-in-place", value, torch.tensor([1.0, 9.0, 10.0]))

    empty = torch.empty(0, dtype=torch.float32, device=device)
    dist.all_reduce(empty)
    check("all_reduce", "zero-count", empty, torch.empty(0, dtype=torch.float32))

    for root in (0, 2):
        value = torch.tensor([rank * 10 + index for index in range(4)], dtype=torch.int32, device=device)
        dist.broadcast(value, src=root)
        expected = torch.tensor([root * 10 + index for index in range(4)], dtype=torch.int32)
        check("broadcast", f"root-{root}", value, expected)

    for root in (0, 3):
        value = torch.tensor([rank + 1, (rank + 1) * 10], dtype=torch.int64, device=device)
        dist.reduce(value, dst=root, op=dist.ReduceOp.SUM)
        expected = torch.tensor([10, 100], dtype=torch.int64) if rank == root else None
        check(
            "reduce",
            f"sum-root-{root}",
            value,
            expected,
            "non-root recv contents are not part of the Reduce contract" if rank != root else "",
        )

    gathered_input = torch.tensor([rank, rank + 10, rank + 100], dtype=torch.int64, device=device)
    gathered_list = [torch.empty_like(gathered_input) for _ in range(world)]
    dist.all_gather(gathered_list, gathered_input)
    gathered = torch.stack(gathered_list)
    gathered_expected = torch.tensor([[r, r + 10, r + 100] for r in range(world)], dtype=torch.int64)
    check("all_gather", "list-rank-major", gathered, gathered_expected)

    gathered_flat = torch.empty(world * gathered_input.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered_flat, gathered_input)
    check("all_gather", "into-tensor-rank-major", gathered_flat, gathered_expected.reshape(-1))

    chunk = 3
    rs_input = torch.arange(world * chunk, dtype=torch.int64, device=device) + rank * 100
    rs_output = torch.empty(chunk, dtype=torch.int64, device=device)
    dist.reduce_scatter_tensor(rs_output, rs_input, op=dist.ReduceOp.SUM)
    rank_slice = torch.arange(rank * chunk, (rank + 1) * chunk, dtype=torch.int64)
    rs_expected = rank_slice * world + 100 * world * (world - 1) // 2
    check("reduce_scatter", "sum-contiguous-rank-chunk", rs_output, rs_expected)

    equal_chunk = 2
    equal_input_chunks = []
    for dst in range(world):
        equal_input_chunks.extend([rank * 1000 + dst * 100 + index for index in range(equal_chunk)])
    equal_input = torch.tensor(equal_input_chunks, dtype=torch.int64, device=device)
    equal_output = torch.empty_like(equal_input)
    dist.all_to_all_single(equal_output, equal_input)
    equal_expected = torch.tensor(
        [src * 1000 + rank * 100 + index for src in range(world) for index in range(equal_chunk)],
        dtype=torch.int64,
    )
    check("all_to_all", "equal-splits-source-major-output", equal_output, equal_expected)

    input_splits = [rank + dst + 1 for dst in range(world)]
    output_splits = [src + rank + 1 for src in range(world)]
    uneven_values = []
    for dst, count in enumerate(input_splits):
        uneven_values.extend(rank * 10000 + dst * 100 + index for index in range(count))
    uneven_input = torch.tensor(uneven_values, dtype=torch.int64, device=device)
    uneven_output = torch.empty(sum(output_splits), dtype=torch.int64, device=device)
    dist.all_to_all_single(
        uneven_output,
        uneven_input,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    uneven_expected_values = []
    for src, count in enumerate(output_splits):
        uneven_expected_values.extend(src * 10000 + rank * 100 + index for index in range(count))
    check(
        "all_to_all",
        "uneven-splits-source-major-output",
        uneven_output,
        torch.tensor(uneven_expected_values, dtype=torch.int64),
        f"input_splits={input_splits};output_splits={output_splits}",
    )

    next_rank = (rank + 1) % world
    previous_rank = (rank - 1 + world) % world
    send = torch.tensor([rank, rank + 100], dtype=torch.int64, device=device)
    receive = torch.empty_like(send)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send, next_rank),
            dist.P2POp(dist.irecv, receive, previous_rank),
        ]
    )
    for request in requests:
        request.wait()
    check(
        "send_recv",
        "grouped-ring",
        receive,
        torch.tensor([previous_rank, previous_rank + 100], dtype=torch.int64),
    )

    invalid_root_rejected = False
    invalid_message = ""
    try:
        invalid = torch.tensor([rank], dtype=torch.int64, device=device)
        dist.broadcast(invalid, src=world)
        torch.cuda.synchronize(device)
    except Exception as error:  # The contract requires a deterministic local rejection.
        invalid_root_rejected = True
        invalid_message = str(error).replace("\n", " ")[:300]
    rows.append(
        {
            "rank": rank,
            "operation": "broadcast",
            "variant": "invalid-root-rejected",
            "shape": "1",
            "dtype": "int64",
            "actual": str(invalid_root_rejected).lower(),
            "expected": "true",
            "correct": invalid_root_rejected,
            "note": invalid_message,
        }
    )
    if not invalid_root_rejected:
        raise AssertionError("invalid broadcast root was not rejected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rank{rank}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dist.barrier()
    if rank == 0:
        print(f"rows_per_rank={len(rows)} output_dir={args.output_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
