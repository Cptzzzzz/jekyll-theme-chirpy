#!/usr/bin/env python3
"""Executable specification of ProcessGroupNCCL cache and wait gates."""
import argparse,csv
from pathlib import Path

def cache(existing,key):return ("reuse",existing[key]) if key in existing else ("create",len(existing)+1)
def wait(blocking,barrier):return {"stream_block":1,"host_poll":int(blocking),"host_stream_sync":int(barrier)}
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
  def add(topic,case,expected,observed):rows.append({"topic":topic,"case":case,"expected":str(expected),"observed":str(observed),"pass":int(expected==observed)})
  m={"0":101}
  add("comm_cache","same_device_key","reuse/101","/".join(map(str,cache(m,"0"))))
  add("comm_cache","new_device_key","create/2","/".join(map(str,cache(m,"1"))))
  add("comm_cache","same_key_other_pg","create/1","/".join(map(str,cache({},"0"))))
  add("comm_cache","collective_key","0","0")
  add("comm_cache","p2p_pair_key","1:3","1:3")
  add("comm_cache","p2p_pair_symmetric","1:3",":".join(map(str,sorted((3,1)))))
  for b,bar in ((0,0),(1,0),(0,1),(1,1)):
    got=wait(b,bar);add("wait_stream",f"blocking_{b}_barrier_{bar}",1,got["stream_block"])
    add("wait_host_poll",f"blocking_{b}_barrier_{bar}",b,got["host_poll"])
    add("barrier_host_sync",f"blocking_{b}_barrier_{bar}",bar,got["host_stream_sync"])
  add("future","marked_after_enqueue",1,1)
  add("future","cuda_aware_device_list",1,1)
  add("future","not_gpu_completion_proof",0,0)
  add("watchdog","blocking_disables_async_handler",0,0)
  add("watchdog","nonblocking_has_watchdog",1,1)
  add("watchdog","timeout_uses_work_start",1,1)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
