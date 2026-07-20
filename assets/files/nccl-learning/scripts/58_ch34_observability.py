#!/usr/bin/env python3
"""Align NCCL debug, Nsight/NVTX, and ProcessGroupNCCL flight records."""
import argparse,csv,hashlib,os,pickle,re,statistics,subprocess
from datetime import datetime,timezone
from pathlib import Path
def env():
 e=dict(os.environ)
 for k in tuple(e):
  if k.startswith('NCCL_') or k.startswith('TORCH_NCCL_') or k in ('MASTER_ADDR','MASTER_PORT','LD_LIBRARY_PATH'):e.pop(k)
 e['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python';e['TORCH_NCCL_TRACE_BUFFER_SIZE']='128';return e
def run(cmd,e,p,cwd,timeout=600):
 x=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x.stdout,encoding='utf-8')
 if x.returncode:raise RuntimeError(f"{p}\n{x.stdout[-3000:]}");return x.stdout
 return x.stdout
def outcsv(p,r):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/root/nccl-learning'));ap.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));a=ap.parse_args();d=a.root/'logs/ch34_observability'/a.run_id;raw=d/'raw';private=d/'private';private.mkdir(parents=True,exist_ok=True)
 base=['torchrun','--standalone','--nproc_per_node=2',str(a.root/'probes/ch34_observability_worker.py')];allrows=[]
 configs=(('init_graph','INFO','INIT,GRAPH'),('tuning','INFO','TUNING'),('coll_trace','TRACE','COLL,TUNING'))
 for name,level,subsys in configs:
  e=env();e.update({'NCCL_DEBUG':level,'NCCL_DEBUG_SUBSYS':subsys});od=raw/'rows'/name;text=run(base+['--output-dir',str(od)],e,raw/f'{name}.log',a.root)
  rows=[]
  for p in sorted(od.glob('measurements_rank*.csv')):rows+=list(csv.DictReader(p.open()))
  if len(rows)!=6 or any(x['correct']!='1' for x in rows):raise RuntimeError(name)
  for x in rows:x['debug_config']=name
  allrows+=rows
 outcsv(d/'measurement_summary.csv',allrows)
 # Parse sanitized NCCL evidence; raw logs remain private.
 evidence=[]
 for name,_,_ in configs:
  text=(raw/f'{name}.log').read_text()
  for kind,pat in (('ring',r'Ring \d+ :[^\n]+'),('tuning',r'\d+ Bytes -> Algo \d+ proto \d+ time [0-9.]+'),('collective',r'Collective AllReduce[^\n]+'),('coll_info',r'AllReduce: opCount[^\n]+')):
   for i,m in enumerate(re.finditer(pat,text)):
    sanitized=re.sub(r'0x[0-9a-fA-F]+','<ptr>',m.group(0))
    evidence.append({'config':name,'kind':kind,'ordinal':i,'evidence':sanitized[:500]})
 if not any(x['kind']=='tuning' for x in evidence) or not any(x['kind'] in ('collective','coll_info') for x in evidence):raise RuntimeError('debug evidence')
 outcsv(d/'nccl_debug_evidence.csv',evidence)
 # Parse flight recorder pickles from a completed run.
 flights=[]
 for p in sorted((raw/'rows'/'coll_trace').glob('flight_rank*.pkl')):
  rank=int(re.search(r'(\d+)',p.stem).group(1));obj=pickle.loads(p.read_bytes());entries=obj.get('entries',[]) if isinstance(obj,dict) else []
  for z in entries:
   if 'all_reduce' in str(z.get('profiling_name','')).lower():
    created=z.get('time_created_ns');completed=z.get('time_discovered_completed_ns');duration=(completed-created)/1e6 if created and completed else ''
    flights.append({'rank':rank,'record_id':z.get('record_id',''),'process_group':str(z.get('process_group','')),'collective_seq_id':z.get('collective_seq_id',''),'p2p_seq_id':z.get('p2p_seq_id',''),'profiling_name':z.get('profiling_name',''),'input_sizes':str(z.get('input_sizes','')),'output_sizes':str(z.get('output_sizes','')),'state':z.get('state',''),'duration_ms':duration})
 if len(flights)<6:raise RuntimeError(f'flight entries {len(flights)}')
 outcsv(d/'flight_recorder_summary.csv',flights)
 # Nsight one aligned run.
 nsys=subprocess.check_output(['which','nsys'],text=True).strip();prefix=private/'nsys';od=private/'rows';e=env();e['NCCL_DEBUG']='WARN'
 run([nsys,'profile','--force-overwrite=true','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop','-o',str(prefix)]+base+['--output-dir',str(od),'--profile'],e,raw/'nsys.log',a.root,900)
 rep=Path(str(prefix)+'.nsys-rep');gpu=run([nsys,'stats','--force-export=true','--report','cuda_gpu_trace','--format','csv',str(rep)],env(),private/'gpu.csv',a.root);nvtx=run([nsys,'stats','--force-export=true','--report','nvtx_gpu_proj_trace','--format','csv',str(rep)],env(),private/'nvtx.csv',a.root)
 def table(text):
  lines=text.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith('Start (ns),'));return list(csv.DictReader(lines[i:]))
 kernels=[]
 for x in table(gpu):
  if 'ncclDevKernel_AllReduce' in x['Name']:kernels.append({'device':x['Device'],'start_ns':x['Start (ns)'],'duration_ns':x['Duration (ns)'],'kernel_name':x['Name']})
 ranges=[];nlines=nvtx.splitlines();ni=next(i for i,x in enumerate(nlines) if x.startswith('Name,Projected Start (ns),'))
 for x in csv.DictReader(nlines[ni:]):
  if 'ch34_seq' in x['Name']:ranges.append({'range':x['Name'].lstrip(':'),'start_ns':x['Projected Start (ns)'],'duration_ns':x['Projected Duration (ns)']})
 if len(kernels)!=6 or len(ranges)<6:raise RuntimeError(f'nsys kernels={len(kernels)} ranges={len(ranges)}')
 outcsv(d/'nsys_kernel_summary.csv',kernels);outcsv(d/'nvtx_range_summary.csv',ranges)
 model=d/'observability_source_model.csv';run(['python3',str(a.root/'probes/ch34_observability_model.py'),'--output',str(model)],env(),raw/'model.log',a.root);mr=list(csv.DictReader(model.open()))
 if len(mr)!=32 or any(x['pass']!='1' for x in mr):raise RuntimeError(len(mr))
 import torch
 manifest=['experiment=ch34_observability',f'run_id={a.run_id}',f'timestamp_utc={datetime.now(timezone.utc).isoformat()}',f'measurement_rows={len(allrows)}',f'debug_evidence_rows={len(evidence)}',f'flight_rows={len(flights)}',f'nsys_kernels={len(kernels)}',f'nvtx_ranges={len(ranges)}','correctness=PASS','source_model=32/32 PASS','raw_debug_and_profiles=NOT_FOR_PUBLICATION',f'pytorch_runtime={torch.__version__}','nccl_runtime=2.22.3']
 (d/'manifest.txt').write_text('\n'.join(manifest)+'\n')
 med={}
 for n in (4096,1<<20,64<<20):med[n]=statistics.median(float(x['gpu_us']) for x in allrows if int(x['bytes'])==n)
 lines=['# Chapter 34 experiment summary','',f'- Measurements: {len(allrows)}',f'- NCCL debug evidence: {len(evidence)}',f'- Flight-recorder AllReduce rows: {len(flights)}',f'- Nsight kernels: {len(kernels)}; NVTX ranges: {len(ranges)}','- Source model: 32/32 PASS','','| bytes | median GPU us |','|---:|---:|']+[f'| {n} | {med[n]:.1f} |' for n in med]
 (d/'summary.md').write_text('\n'.join(lines)+'\n');print(f'run_dir={d}')
if __name__=='__main__':main()
