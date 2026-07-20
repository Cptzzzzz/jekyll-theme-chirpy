#!/usr/bin/env python3
import os
from datetime import timedelta

# This lab image has an onnx/protobuf mismatch that can be reached when DDP
# imports torch._dynamo internals. Keep the example focused on DDP/NCCL.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def main() -> None:
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234)
    x = torch.randn(4096, 16)
    true_w = torch.randn(16, 4)
    y = x @ true_w + 0.01 * torch.randn(4096, 4)
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    if rank == 0:
        print(f"world_size={world}; one process per GPU; backend=nccl", flush=True)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        total = 0.0
        for step, (bx, by) in enumerate(loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = ddp_model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            if step == 9:
                break

        loss_tensor = torch.tensor([total / 10], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"epoch={epoch} avg_loss_across_ranks={loss_tensor.item():.6f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
