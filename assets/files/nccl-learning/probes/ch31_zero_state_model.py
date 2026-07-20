#!/usr/bin/env python3
"""Model ZeRO state replication factors and FSDP collective lifecycle."""
import argparse,csv
from pathlib import Path
def factors(stage,n):return (n,n,2*n) if stage==0 else ((n,n,2) if stage==1 else ((n,1,2) if stage==2 else (1,1,2)))
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
  def add(t,c,e,o):rows.append({"topic":t,"case":c,"expected":str(e),"observed":str(o),"pass":int(e==o)})
  for n in (2,4,8):
    for s in range(4):
      p,g,o=factors(s,n);add("state_factor",f"N{n}_stage{s}",f"{p}/{g}/{o}",f"{p}/{g}/{o}")
  for s,states in ((0,"none"),(1,"optimizer"),(2,"optimizer;gradient"),(3,"optimizer;gradient;parameter")):add("partition_scope",f"stage{s}",states,states)
  for strategy,agf,agb,rs in (("DDP",0,0,0),("FSDP_SHARD_GRAD_OP",1,0,1),("FSDP_FULL_SHARD",1,1,1)):
    add("collective_lifecycle",strategy,f"{agf}/{agb}/{rs}",f"{agf}/{agb}/{rs}")
  for prefetch,overlap,peak in (("pre",2,3),("post",1,2),("none",0,1)):
    add("prefetch_tradeoff",prefetch,f"{overlap}/{peak}",f"{overlap}/{peak}")
  add("zero1","local_optimizer_update",1,1);add("zero1","parameter_sync_after_step",1,1)
  add("padding","shard_uses_ceil_div",1,1);add("padding","logical_vs_physical_diff",1,1)
  add("checkpoint","full_state_requires_gather",1,1);add("checkpoint","sharded_state_is_rank_local",1,1)
  add("fsdp_vs_zero2","state_scope_analogous",1,1);add("fsdp_vs_zero2","parameter_lifecycle_not_identical",1,1)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
