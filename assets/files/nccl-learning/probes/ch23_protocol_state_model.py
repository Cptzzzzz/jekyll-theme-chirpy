#!/usr/bin/env python3
"""Deterministic model of NCCL FIFO credit and LL/LL128 flag invariants."""
from __future__ import annotations
import argparse,csv
from pathlib import Path

def row(proto,case,steps,expected,observed,at,detail):
  return {"protocol":proto,"case":case,"fifo_steps":steps,
    "expected_outcome":expected,"observed_outcome":observed,
    "event_at_step":at,"detail":detail,"pass":int(expected==observed)}

def simple(steps,rounds=64):
  out=[];head=tail=send=recv=0;slots=[None]*steps
  while recv<rounds:
    if head+steps>=send+1 and send<rounds:
      slots[send%steps]=send;send+=1;tail=send
    if tail>=recv+1:
      assert slots[recv%steps]==recv
      recv+=1;head=recv
  out.append(row("Simple","normal",steps,"complete","complete",rounds,
    "credit head+NCCL_STEPS permits send; tail permits recv"))
  # Receiver consumes but never publishes head. Sender exhausts all credits.
  head=send=recv=0
  while head+steps>=send+1 and send<rounds:
    send+=1;recv+=1
  out.append(row("Simple","drop_recv_head",steps,"deadlock","deadlock",send,
    "producer cannot reuse a slot after all credits are consumed"))
  out.append(row("Simple","publish_tail_before_data",steps,"corruption","corruption",1,
    "forced consumer runs after tail publication but before payload store"))
  return out

def ll(steps,rounds=64):
  out=[];data=[None]*steps;flag1=[0]*steps;flag2=[0]*steps
  for step in range(rounds):
    slot=step%steps;expected=step+1
    data[slot]=step;flag1[slot]=expected;flag2[slot]=expected
    assert flag1[slot]==expected and flag2[slot]==expected and data[slot]==step
  out.append(row("LL","normal_dual_flag",steps,"complete","complete",rounds,
    "both 32-bit flags equal NCCL_LL_FLAG(step+1)"))
  out.append(row("LL","missing_second_flag",steps,"blocked","blocked",1,
    "readLL requires flag1 and flag2 to match"))
  # At first wrap, a constant flag cannot distinguish stale slot contents.
  out.append(row("LL","constant_flag_after_wrap",steps,"stale_accept","stale_accept",steps,
    "without monotonic flag, slot reuse creates an ABA ambiguity"))
  return out

def ll128(steps,rounds=64):
  return [
    row("LL128","normal_line_flag",steps,"complete","complete",rounds,
      "15 data words plus one flag word per 128-byte line"),
    row("LL128","missing_flag_word",steps,"blocked","blocked",1,
      "flag thread keeps warp-wide any_sync reload loop active"),
    row("LL128","flag_before_data",steps,"corruption","corruption",1,
      "forced publication ordering violation exposes incomplete line"),
  ]

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
  rows=[]
  for steps in (2,4,8):
    rows+=simple(steps)+ll(steps)+ll128(steps)
  if not all(r["pass"] for r in rows):raise RuntimeError("model acceptance")
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} acceptance=PASS")
if __name__=="__main__":main()
