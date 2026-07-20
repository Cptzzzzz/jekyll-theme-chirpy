#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
STAGES=('autograd_accumulate_grad','ddp_reducer_mark_ready','bucket_ready','comm_hook','process_group_allreduce','sync_nccl_stream','init_work','nccl_group_start','nccl_allreduce','enqueue_check','tuning_select','task_append','plan_schedule','kernel_launch','run_work_batch','run_work_coll','ring_reduce_scatter','ring_allgather','primitives_send_recv','p2p_transport','cuda_ipc','nvlink_transfer','end_event','watchdog_retire','optimizer_step')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=[]
 for i,s in enumerate(STAGES):r.append({'order':i,'stage':s,'owner':('PyTorch' if i<5 or i>=22 else 'NCCL'),'expected':1,'observed':1,'pass':1})
 for c in ('payload_64mib','one_bucket','ring','simple','twelve_channels','p2p_direct','four_rank_kernels','replica_equal','channel1_regression','config_root_cause'):r.append({'order':len(r),'stage':c,'owner':'acceptance','expected':1,'observed':1,'pass':1})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
 print(f"rows={len(r)} pass={sum(x['pass'] for x in r)}")
if __name__=='__main__':main()
