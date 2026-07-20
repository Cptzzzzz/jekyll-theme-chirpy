#!/usr/bin/env python3
"""Prove one 64 MiB DDP AllReduce end to end and diagnose a channel regression."""
import argparse,csv,json,os,re,statistics,subprocess
from datetime import datetime,timezone
from pathlib import Path
def env():
 e=dict(os.environ)
 for k in tuple(e):
  if k.startswith('NCCL_') or k.startswith('TORCH_NCCL_') or k in ('MASTER_ADDR','MASTER_PORT','LD_LIBRARY_PATH'):e.pop(k)
 e['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python';return e
def run(cmd,e,p,cwd,timeout=600):
 x=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x.stdout)
 if x.returncode:raise RuntimeError(f'{p}\n{x.stdout[-3000:]}');return x.stdout
 return x.stdout
def reads(d,prefix):
 r=[]
 for p in sorted(d.glob(prefix+'*.csv')):r+=list(csv.DictReader(p.open()))
 return r
def outcsv(p,r):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def cmd(base,w,out,cycles,profile=False):
 x=base+[str(w),'--output-dir',str(out),'--config','capstone64','--param-mib','64','--param-count','1','--bucket-cap-mb','64','--warmup','3','--cycles',str(cycles)]
 if profile:x.append('--profile-capture')
 return x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/root/nccl-learning'));ap.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));a=ap.parse_args();d=a.root/'logs/ch36_capstone'/a.run_id;raw=d/'raw';private=d/'private';private.mkdir(parents=True,exist_ok=True);base=['torchrun','--standalone','--nproc_per_node=4'];w=a.root/'probes/ch30_ddp_reducer_worker.py'
 allrows=[];proof=[]
 for name,opts in (('baseline',{}),('channel1',{'NCCL_MIN_NCHANNELS':'1','NCCL_MAX_NCHANNELS':'1'})):
  e=env();e.update(opts);e.update({'NCCL_DEBUG':'INFO','NCCL_DEBUG_SUBSYS':'INIT,GRAPH,TUNING,COLL'});od=raw/'rows'/name;text=run(cmd(base,w,od,6),e,raw/f'{name}.log',a.root)
  rows=reads(od,'iterations_rank');buckets=reads(od,'buckets_rank');states=[json.loads(p.read_text()) for p in od.glob('logging_rank*.json')]
  if len(rows)!=24 or any(x['correct']!='1' for x in rows) or len(buckets)!=24 or not all(x['replicas_equal'] for x in states):raise RuntimeError(name)
  for x in rows:x['variant']=name
  allrows+=rows
  sel=re.findall(r'67108864 Bytes -> Algo (\d+) proto (\d+) time ([0-9.]+)',text);channels=re.findall(r'(\d+) coll channels',text);p2p=len(re.findall(r'via P2P',text))
  proof.append({'variant':name,'bucket_bytes':int(buckets[0]['bucket_bytes']),'buckets_per_rank_iteration':len(buckets)//24,'algo':sel[-1][0] if sel else '','proto':sel[-1][1] if sel else '','model_time_us':sel[-1][2] if sel else '','coll_channels':channels[-1] if channels else '','p2p_lines':p2p,'median_step_gpu_us':statistics.median(float(x['step_gpu_us']) for x in rows)})
 outcsv(d/'iteration_measurements.csv',allrows);outcsv(d/'end_to_end_proof.csv',proof)
 b=proof[0];c=proof[1];reg=(c['median_step_gpu_us']/b['median_step_gpu_us']-1)*100
 acceptance=[('payload',67108864,b['bucket_bytes']),('bucket_count',1,b['buckets_per_rank_iteration']),('algorithm',1,int(b['algo'])),('protocol',2,int(b['proto'])),('channels',12,int(b['coll_channels'])),('p2p_present',1,int(b['p2p_lines']>0)),('regression_channels',1,int(c['coll_channels'])),('regression_detected',1,int(reg>50))]
 outcsv(d/'prediction_acceptance.csv',[{'check':x,'expected':y,'observed':z,'pass':int(y==z)} for x,y,z in acceptance])
 if any(y!=z for _,y,z in acceptance):raise RuntimeError(acceptance)
 nsys=subprocess.check_output(['which','nsys'],text=True).strip();prefix=private/'nsys';od=private/'rows';e=env();e['NCCL_DEBUG']='WARN';run([nsys,'profile','--force-overwrite=true','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop','-o',str(prefix)]+cmd(base,w,od,1,True),e,raw/'nsys.log',a.root,900);rep=Path(str(prefix)+'.nsys-rep');gpu=run([nsys,'stats','--force-export=true','--report','cuda_gpu_trace','--format','csv',str(rep)],env(),private/'gpu.csv',a.root);lines=gpu.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith('Start (ns),'));kr=[]
 for x in csv.DictReader(lines[i:]):
  if 'ncclDevKernel_AllReduce' in x['Name']:kr.append({'device':x['Device'],'duration_ns':x['Duration (ns)'],'grid_x':x['GrdX'],'block_x':x['BlkX'],'kernel_name':x['Name']})
 if len(kr)!=4:raise RuntimeError(len(kr))
 outcsv(d/'nsys_kernel_summary.csv',kr)
 model=d/'callgraph_model.csv';run(['python3',str(a.root/'probes/ch36_callgraph_model.py'),'--output',str(model)],env(),raw/'model.log',a.root);mr=list(csv.DictReader(model.open()))
 if len(mr)!=35 or any(x['pass']!='1' for x in mr):raise RuntimeError(len(mr))
 manifest=['experiment=ch36_capstone',f'run_id={a.run_id}',f'timestamp_utc={datetime.now(timezone.utc).isoformat()}','prediction_acceptance=8/8 PASS','callgraph_model=35/35 PASS','nsys_devices=4/4 PASS','replica_equality=PASS',f'channel1_step_regression_percent={reg:.2f}','raw_logs_and_profiles=NOT_FOR_PUBLICATION','pytorch_runtime=2.5.0a0+872d972e41.nv24.08','nccl_runtime=2.22.3']
 (d/'manifest.txt').write_text('\n'.join(manifest)+'\n');lines=['# Chapter 36 Capstone summary','','- Prediction acceptance: 8/8 PASS','- Call graph model: 35/35 PASS','- Nsight AllReduce kernels: 4/4 PASS','- Replica equality: PASS',f'- Channel1 step regression: {reg:.2f}%','','| variant | bucket | algo/proto | channels | step GPU us |','|---|---:|---|---:|---:|']
 for x in proof:lines.append(f"| {x['variant']} | {x['bucket_bytes']} | {x['algo']}/{x['proto']} | {x['coll_channels']} | {x['median_step_gpu_us']:.1f} |")
 (d/'summary.md').write_text('\n'.join(lines)+'\n');print(f'run_dir={d}')
if __name__=='__main__':main()
