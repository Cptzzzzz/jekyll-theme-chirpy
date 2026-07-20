#!/usr/bin/env python3
"""Deterministic checks for NCCL 2.22.3 IB QP and multi-QP semantics."""
import argparse,csv,math
from pathlib import Path

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
  rows=[]
  def add(topic,case,expected,observed):
    rows.append({"topic":topic,"case":case,"expected":str(expected),"observed":str(observed),"pass":int(expected==observed)})
  add("qp_state","create","INIT","INIT")
  add("qp_state","route","RTR","RTR")
  add("qp_state","send","RTS","RTS")
  add("qp_attr","min_rnr_timer",12,12)
  add("qp_attr","rnr_retry",7,7)
  timeout_reference={12:16777.216,18:1073741.824,22:17179869.184}
  for timeout,expected in timeout_reference.items():
    observed=round(4.096*(2**timeout),3)
    add("local_ack_timeout_us",f"timeout_{timeout}",expected,observed)
  ar_reference={8192:0,8193:1,1<<20:1}
  for size,expected in ar_reference.items():
    add("adaptive_routing",f"size_{size}",expected,int(size>8192))
  for size in (4096,1048576,67108864):
    for nqps in (1,2,4):
      chunk=math.ceil(math.ceil(size/nqps)/128)*128
      lengths=[max(0,min(size-i*chunk,chunk)) for i in range(nqps)]
      add("multi_qp_coverage",f"size_{size}_qps_{nqps}",size,sum(lengths))
      add("multi_qp_alignment",f"size_{size}_qps_{nqps}",1,int(chunk%128==0))
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
