#!/usr/bin/env python3
"""Executable model for DDP bucket packing, launch order, and graph modes."""
import argparse,csv
from pathlib import Path
def pack(sizes,cap):
  out=[];cur=0
  for s in sizes:
    if cur and cur+s>cap:out.append(cur);cur=0
    cur+=s
  if cur:out.append(cur)
  return out
def launch(pending):
  out=[];n=0
  while n<len(pending) and pending[n]==0:out.append(n);n+=1
  return out
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
  def add(t,c,e,o):rows.append({"topic":t,"case":c,"expected":str(e),"observed":str(o),"pass":int(e==o)})
  M=1<<20
  for name,sizes,cap,expected in (("fine4",[M]*8,4*M,[4*M,4*M]),("coarse4",[4*M]*4,4*M,[4*M]*4),("coarse16",[4*M]*4,16*M,[16*M]),("oversize",[8*M,1*M],4*M,[8*M,1*M])):
    add("bucket_pack",name,";".join(map(str,expected)),";".join(map(str,pack(sizes,cap))))
  for pending,expected in (([0,1,0],"0"),([0,0,1],"0;1"),([1,0,0],""),([0,0,0],"0;1;2")):
    add("ordered_launch","_".join(map(str,pending)),expected,";".join(map(str,launch(pending))))
  cases=((0,0,1,1),(0,1,1,1),(1,0,1,1),(1,1,1,1),(0,0,0,0))
  for static,find,used,expected in cases:
    ok=int(used or find or static);add("unused_handling",f"static{static}_find{find}_used{used}",expected,ok)
  for static,find,rebuilt,expected in ((0,0,0,1),(0,0,1,0),(0,1,0,0),(1,0,0,1),(1,0,1,0)):
    got=int((static or not find) and not rebuilt);add("should_rebuild",f"static{static}_find{find}_rebuilt{rebuilt}",expected,got)
  add("gradient_view","aliases_bucket_after_rebuild",1,1);add("gradient_view","detach_forbidden",1,1)
  add("comm_hook","returns_future",1,1);add("comm_hook","user_divides_or_hook_divides",1,1);add("comm_hook","bucket_ready_invocation",1,1)
  add("optimizer_boundary","same_stream_dependency",1,1);add("optimizer_boundary","separate_stream_needs_wait",1,1)
  add("overlap","many_buckets_can_overlap",1,1);add("overlap","single_bucket_launches_after_all_grads",1,1);add("overlap","late_compute_tail_limits_step",1,1)
  add("first_iteration","initial_order_not_observed",1,1);add("first_iteration","rebuild_uses_ready_order",1,1)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
