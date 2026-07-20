#!/usr/bin/env python3
"""Exercise lazy communicator creation, per-PG caches, and high-priority options."""
from __future__ import annotations
import argparse,csv,os,time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--cycles",type=int,default=3);a=ap.parse_args()
  local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local)
  dist.init_process_group("nccl",timeout=timedelta(seconds=60));rank=dist.get_rank();world=dist.get_world_size()
  if world!=4:raise RuntimeError("requires four ranks")
  ranks=list(range(world));standard=dist.new_group(ranks,backend="nccl",group_desc="ch29-standard")
  opts=dist.ProcessGroupNCCL.Options();opts.is_high_priority_stream=True
  high=dist.new_group(ranks,backend="nccl",pg_options=opts,group_desc="ch29-high-priority")
  print(f"CH29_MARK rank={rank} groups_created_no_collective=1",flush=True)
  rows=[]
  for name,group,priority in (("world",dist.group.WORLD,0),("standard",standard,0),("high_priority",high,1)):
    for cycle in range(a.cycles):
      x=torch.full((1<<18,),float(rank+1),device=dev)
      print(f"CH29_MARK rank={rank} pg={name} cycle={cycle} before=1",flush=True)
      t0=time.perf_counter_ns();work=dist.all_reduce(x,group=group,async_op=True);t1=time.perf_counter_ns()
      work.wait();t2=time.perf_counter_ns();torch.cuda.synchronize(dev);t3=time.perf_counter_ns()
      correct=bool(torch.all(x==10.0).item())
      if not correct:raise RuntimeError(f"incorrect {name} rank={rank}")
      rows.append({"rank":rank,"pg":name,"high_priority":priority,"cycle":cycle,
        "first_use":int(cycle==0),"enqueue_us":(t1-t0)/1000,"wait_us":(t2-t1)/1000,
        "device_sync_us":(t3-t2)/1000,"correct":1})
      print(f"CH29_MARK rank={rank} pg={name} cycle={cycle} after=1",flush=True)
  a.output_dir.mkdir(parents=True,exist_ok=True)
  with (a.output_dir/f"rank{rank}.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  dist.barrier();dist.destroy_process_group(high);dist.destroy_process_group(standard);dist.destroy_process_group()
if __name__=="__main__":main()
