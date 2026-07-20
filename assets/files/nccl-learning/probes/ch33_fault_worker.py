#!/usr/bin/env python3
"""Inject one distributed failure with explicit causal markers."""
from __future__ import annotations
import argparse,os,time
from datetime import timedelta
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION","python")
import torch
import torch.distributed as dist

CASES=("control","op_mismatch","shape_mismatch","order_mismatch","skip","lazy_skip","rank_exit","peer_exception")
def mark(rank,case,event,**fields):
  extra=" ".join(f"{k}={v}" for k,v in fields.items())
  print(f"CH33_EVENT rank={rank} case={case} event={event} mono_ns={time.monotonic_ns()} {extra}".rstrip(),flush=True)
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--case",choices=CASES,required=True);a=ap.parse_args()
  local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local)
  dist.init_process_group("nccl",timeout=timedelta(seconds=4));rank=dist.get_rank()
  if dist.get_world_size()!=2:raise RuntimeError("chapter 33 requires two ranks")
  mark(rank,a.case,"pg_initialized")
  if a.case!="lazy_skip":
    x=torch.tensor([float(rank+1)],device=dev);dist.all_reduce(x);torch.cuda.synchronize(dev)
    if x.item()!=3:raise RuntimeError("warmup failed")
    mark(rank,a.case,"healthy_collective_complete",seq=0)
  if a.case=="control":
    x=torch.tensor([float(rank+1)],device=dev);mark(rank,a.case,"collective_enter",seq=1,op="allreduce",count=1)
    dist.all_reduce(x);torch.cuda.synchronize(dev);mark(rank,a.case,"collective_complete",seq=1,value=float(x))
    dist.destroy_process_group();return
  if a.case=="op_mismatch":
    x=torch.tensor([float(rank+1)],device=dev);op="broadcast" if rank==0 else "allreduce";mark(rank,a.case,"collective_enter",seq=1,op=op,count=1)
    (dist.broadcast(x,src=0) if rank==0 else dist.all_reduce(x));torch.cuda.synchronize(dev)
    mark(rank,a.case,"unexpected_collective_return",seq=1,value=float(x));time.sleep(6);raise RuntimeError("op mismatch returned")
  if a.case=="shape_mismatch":
    count=4 if rank==0 else 8;x=torch.ones(count,device=dev);mark(rank,a.case,"collective_enter",seq=1,op="allreduce",count=count)
    dist.all_reduce(x);torch.cuda.synchronize(dev);mark(rank,a.case,"unexpected_collective_return",seq=1);time.sleep(6);raise RuntimeError("shape mismatch returned")
  if a.case=="order_mismatch":
    x=torch.tensor([float(rank+1)],device=dev);first="broadcast" if rank==0 else "allreduce";second="allreduce" if rank==0 else "broadcast"
    mark(rank,a.case,"collective_enter",seq=1,op=first,count=1);(dist.broadcast(x,src=0) if rank==0 else dist.all_reduce(x))
    mark(rank,a.case,"collective_enter",seq=2,op=second,count=1);(dist.all_reduce(x) if rank==0 else dist.broadcast(x,src=0));torch.cuda.synchronize(dev)
    mark(rank,a.case,"unexpected_collective_return",seq=2);raise RuntimeError("order mismatch returned")
  if a.case in ("skip","lazy_skip"):
    seq=0 if a.case=="lazy_skip" else 1
    if rank==0:
      x=torch.ones(1,device=dev);mark(rank,a.case,"collective_enter",seq=seq,op="allreduce",count=1);dist.all_reduce(x);torch.cuda.synchronize(dev)
      mark(rank,a.case,"unexpected_collective_return",seq=seq);raise RuntimeError("skipped collective returned")
    mark(rank,a.case,"fault_injected_skip",seq=seq)
    if a.case=="lazy_skip":time.sleep(30)
    else:time.sleep(8);raise RuntimeError("skip fixture finished")
    raise RuntimeError("lazy skip fixture unexpectedly resumed")
  if a.case=="rank_exit":
    if rank==1:
      mark(rank,a.case,"fault_injected_process_exit",exitcode=42);os._exit(42)
    x=torch.ones(1,device=dev);mark(rank,a.case,"collective_enter",seq=1,op="allreduce",count=1);dist.all_reduce(x);torch.cuda.synchronize(dev)
    raise RuntimeError("collective after peer exit returned")
  if rank==1:
    mark(rank,a.case,"fault_injected_application_exception");raise RuntimeError("CH33 injected application failure before collective")
  x=torch.ones(1,device=dev);mark(rank,a.case,"collective_enter",seq=1,op="allreduce",count=1);dist.all_reduce(x);torch.cuda.synchronize(dev)
  raise RuntimeError("collective after peer exception returned")
if __name__=="__main__":main()
