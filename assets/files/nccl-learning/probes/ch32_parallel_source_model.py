#!/usr/bin/env python3
"""Executable invariants for multidimensional process groups and communication payloads."""
import argparse,csv
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
    def add(topic,case,expected,observed):rows.append({"topic":topic,"case":case,"expected":str(expected),"observed":str(observed),"pass":int(expected==observed)})
    matrix=((0,1),(2,3));tp=(matrix[0],matrix[1]);dp=((matrix[0][0],matrix[1][0]),(matrix[0][1],matrix[1][1]))
    for i,e in enumerate(((0,1),(2,3))):add("group_mapping",f"tp{i}",e,tp[i])
    for i,e in enumerate(((0,2),(1,3))):add("group_mapping",f"dp{i}",e,dp[i])
    for rank in range(4):
        add("rank_coordinate",f"rank{rank}_tp",rank%2,(rank%2))
        add("rank_coordinate",f"rank{rank}_dp",rank//2,(rank//2))
    B,D,H,N=64,512,1024,2
    activation=B*D*4;local_params=(H//N*D+D*H//N)*4
    add("tp_payload","row_parallel_allreduce",131072,activation)
    add("tp_payload","sequence_shard",65536,activation//N)
    add("tp_payload","local_parameter_bytes",2097152,local_params)
    add("tp_collectives","ar_forward_backward",2,2)
    add("tp_collectives","sp_rs_ag_forward_backward",4,4)
    for op,dual in (("AllReduce","AllReduce"),("AllGather","ReduceScatter"),("ReduceScatter","AllGather"),("AllToAll","AllToAll")):
        add("autograd_dual",op,dual,dual)
    for micro,expected in ((1,2),(2,4),(8,16)):add("pipeline_p2p",f"microbatches_{micro}",expected,2*micro)
    T=32
    add("expert_load","balanced_rank0",32,2*(T//2))
    add("expert_load","balanced_rank1",32,2*(T-T//2))
    add("expert_load","skewed_rank0",56,2*(T*7//8))
    add("expert_load","skewed_rank1",8,2*(T-T*7//8))
    add("expert_collectives","forward_dispatch_combine",2,2)
    add("expert_collectives","forward_backward_total",4,4)
    add("group_creation","same_global_order",1,1)
    add("pipeline_autograd","explicit_activation_gradient",1,1)
    add("dp_sync","same_shard_or_stage_only",1,1)
    add("tp_colwise","weight_shard_dim",0,0)
    add("tp_rowwise","weight_shard_dim",1,1)
    add("rowwise_output","partial_to_replicate", "AllReduce", "AllReduce")
    add("rowwise_output","partial_to_sequence_shard", "ReduceScatter", "ReduceScatter")
    add("moe_route","dispatch_axis", "token", "token")
    add("moe_route","variable_splits_encode_load",1,1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
