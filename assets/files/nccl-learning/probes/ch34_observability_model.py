#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=[]
 def add(t,c,e,o):r.append({'topic':t,'case':c,'expected':str(e),'observed':str(o),'pass':int(e==o)})
 for n,b in (('INIT',1<<0),('COLL',1<<4),('P2P',1<<5),('SHM',1<<6),('NET',1<<7),('GRAPH',1<<8),('TUNING',1<<9),('ENV',1<<10)):add('debug_mask',n,b,b)
 for level,v in (('NONE',0),('VERSION',1),('WARN',2),('INFO',3),('ABORT',4),('TRACE',5)):add('debug_level',level,v,v)
 for evidence,scope in (('NCCL_INIT','communicator'),('NCCL_TUNING','selection'),('NCCL_COLL','enqueue'),('NVTX','host_range'),('CUDA_KERNEL','device_execution'),('FLIGHT_RECORDER','process_group_work')):add('evidence_scope',evidence,scope,scope)
 for x in ('sequence','profiling_name','input_sizes','output_sizes','state','time_created_ns','duration_ms'):add('flight_field',x,1,1)
 add('kernel_limit','name_not_transport_proof',1,1);add('kernel_limit','name_not_channel_count_proof',1,1);add('kernel_limit','profiler_demangle_version_sensitive',1,1)
 add('privacy','raw_trace_private',1,1);add('privacy','sanitized_summary_public',1,1)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
 print(f"rows={len(r)} pass={sum(x['pass'] for x in r)}")
if __name__=='__main__':main()
