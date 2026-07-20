#!/usr/bin/env python3
"""Measure WorkNCCL enqueue, wait, CUDA-aware Future, and device completion."""
from __future__ import annotations
import argparse,csv,os,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
  ap.add_argument("--action",choices=("work_wait","future_wait","device_sync"),required=True)
  ap.add_argument("--cycles",type=int,default=5);ap.add_argument("--bytes",type=int,default=16<<20)
  ap.add_argument("--sleep-cycles",type=int,default=80_000_000);a=ap.parse_args()
  local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local)
  dist.init_process_group("nccl",timeout=timedelta(seconds=60));rank=dist.get_rank();world=dist.get_world_size()
  if world!=4 or not hasattr(torch.cuda,"_sleep"):raise RuntimeError("requires four GPUs and torch.cuda._sleep")
  x=torch.full((a.bytes//4,),float(rank+1),device=dev);dist.all_reduce(x);torch.cuda.synchronize(dev)
  free_stream=torch.cuda.Stream(device=dev);rows=[]
  for cycle in range(a.cycles):
    x.fill_(float(rank+1));torch.cuda.synchronize(dev);dist.barrier();torch.cuda.synchronize(dev)
    sleep_start=torch.cuda.Event(enable_timing=True);sleep_end=torch.cuda.Event(enable_timing=True)
    sleep_start.record();torch.cuda._sleep(a.sleep_cycles);sleep_end.record()
    torch.cuda.nvtx.range_push(f"ch29_measure_rank{rank}")
    t0=time.perf_counter_ns();work=dist.all_reduce(x,async_op=True);t1=time.perf_counter_ns()
    future=work.get_future();completed_enqueue=int(work.is_completed());future_enqueue=int(future.done())
    t2=time.perf_counter_ns()
    if a.action=="work_wait":work.wait()
    elif a.action=="future_wait":future.wait()
    t3=time.perf_counter_ns()
    completed_action=int(work.is_completed());future_action=int(future.done())
    dependent=torch.cuda.Event();dependent.record()
    free=torch.cuda.Event()
    with torch.cuda.stream(free_stream):free.record(free_stream)
    deadline=time.monotonic()+2
    while not free.query() and time.monotonic()<deadline:time.sleep(.0001)
    free_before_dependent=int(free.query() and not dependent.query())
    t4=time.perf_counter_ns();torch.cuda.synchronize(dev);t5=time.perf_counter_ns()
    torch.cuda.nvtx.range_pop();sleep_end.synchronize()
    correct=int(bool(torch.all(x==10.0).item()))
    if not correct:raise RuntimeError(f"incorrect rank={rank}")
    rows.append({"rank":rank,"cycle":cycle,"action":a.action,
      "blocking_wait":int(os.environ.get("TORCH_NCCL_BLOCKING_WAIT","0")=="1"),
      "bytes":a.bytes,"sleep_cycles":a.sleep_cycles,"sleep_gpu_us":sleep_start.elapsed_time(sleep_end)*1000,
      "enqueue_host_us":(t1-t0)/1000,"query_host_us":(t2-t1)/1000,"action_host_us":(t3-t2)/1000,
      "device_sync_host_us":(t5-t4)/1000,"completed_after_enqueue":completed_enqueue,
      "completed_after_action":completed_action,"future_done_after_enqueue":future_enqueue,
      "future_done_after_action":future_action,"future_done_after_sync":int(future.done()),
      "free_stream_before_dependent":free_before_dependent,"correct":correct})
  a.output_dir.mkdir(parents=True,exist_ok=True)
  with (a.output_dir/f"rank{rank}.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()
