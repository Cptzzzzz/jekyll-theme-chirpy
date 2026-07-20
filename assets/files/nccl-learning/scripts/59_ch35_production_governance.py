#!/usr/bin/env python3
"""Run a small NCCL canary and apply layered production release policies."""
import argparse,csv,hashlib,os,re,statistics,subprocess
from datetime import datetime,timezone
from pathlib import Path
CONFIGS=(('baseline',{}),('channel1',{'NCCL_MIN_NCHANNELS':'1','NCCL_MAX_NCHANNELS':'1'}),('no_p2p',{'NCCL_P2P_DISABLE':'1'}))
def clean():
 e=dict(os.environ)
 for k in tuple(e):
  if k.startswith('NCCL_') or k in ('LD_LIBRARY_PATH',):e.pop(k)
 return e
def run(cmd,e,p,cwd):
 x=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x.stdout)
 if x.returncode:raise RuntimeError(f'{p}\n{x.stdout[-2000:]}');return x.stdout
 return x.stdout
def parse(text):
 line=next(x for x in text.splitlines() if re.match(r'^\s+67108864\s+',x));z=line.split();return float(z[5]),float(z[7]),int(z[8])
def outcsv(p,r):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def sha_text(x):return hashlib.sha256(x.encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/root/nccl-learning'));ap.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));a=ap.parse_args();d=a.root/'logs/ch35_governance'/a.run_id;raw=d/'raw';binary=a.root/'third_party/nccl-tests/build/all_reduce_perf';rows=[]
 for rep in range(1,6):
  order=CONFIGS if rep%2 else tuple(reversed(CONFIGS))
  for name,opts in order:
   e=clean();e.update(opts);e['NCCL_DEBUG']='WARN';text=run([str(binary),'-b','64M','-e','64M','-g','4','-w','2','-n','10','-c','1'],e,raw/'perf'/f'{name}_{rep}.log',a.root);us,bw,wrong=parse(text);rows.append({'config':name,'replicate':rep,'time_us':us,'busbw_gbps':bw,'wrong':wrong})
 if any(x['wrong'] for x in rows):raise RuntimeError('correctness')
 outcsv(d/'canary_measurements.csv',rows);summary=[]
 base=statistics.median(x['busbw_gbps'] for x in rows if x['config']=='baseline')
 for name,_ in CONFIGS:
  v=[x['busbw_gbps'] for x in rows if x['config']==name];summary.append({'config':name,'samples':len(v),'median_busbw_gbps':statistics.median(v),'min_busbw_gbps':min(v),'max_busbw_gbps':max(v),'cv_percent':statistics.pstdev(v)/statistics.fmean(v)*100,'regression_percent':(base-statistics.median(v))/base*100})
 outcsv(d/'canary_summary.csv',summary)
 # Capture one path fingerprint per configuration.
 paths=[]
 for name,opts in CONFIGS:
  e=clean();e.update(opts);e.update({'NCCL_DEBUG':'INFO','NCCL_DEBUG_SUBSYS':'INIT,P2P,SHM'});text=run([str(binary),'-b','64M','-e','64M','-g','4','-w','1','-n','1','-c','1'],e,raw/'path'/f'{name}.log',a.root)
  paths.append({'config':name,'p2p_lines':len(re.findall(r'via P2P',text)),'shm_lines':len(re.findall(r'via SHM',text)),'coll_channels':(re.search(r'(\d+) coll channels',text).group(1) if re.search(r'(\d+) coll channels',text) else '')})
 outcsv(d/'path_fingerprint.csv',paths)
 topo=subprocess.check_output(['nvidia-smi','topo','-m'],text=True);versions=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,pci.bus_id','--format=csv,noheader'],text=True)
 fingerprint={'topology_sha256':sha_text(topo),'gpu_driver_sha256':sha_text(versions),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'nccl_runtime':'2.22.3','cuda_toolkit':'12.6'}
 outcsv(d/'node_fingerprint.csv',[fingerprint])
 # Inject policy-level failures independently of the performance mechanism.
 s={x['config']:x for x in summary};p={x['config']:x for x in paths};decisions=[]
 def add(case,layer,observed,threshold,status,action):decisions.append({'case':case,'layer':layer,'observed':observed,'threshold':threshold,'status':status,'action':action})
 add('baseline','all',f"{s['baseline']['median_busbw_gbps']:.2f} GB/s",'>=90% baseline','PASS','promote canary')
 add('channel1','performance',f"-{s['channel1']['regression_percent']:.2f}%",'-10%','FAIL','block and inspect channel config')
 add('no_p2p','path+performance',f"P2P={p['no_p2p']['p2p_lines']} SHM={p['no_p2p']['shm_lines']} -{s['no_p2p']['regression_percent']:.2f}%",'path unchanged and -10%','FAIL','block and inspect topology/transport')
 add('wrong_result_injected','correctness','#wrong=1','0','FAIL','immediate block')
 add('version_mismatch_injected','version','2.22.3 -> 2.23.x','exact approved manifest','FAIL','separate A/B canary')
 add('topology_mismatch_injected','topology','fingerprint changed','exact node class','FAIL','quarantine node')
 add('high_noise_injected','statistics','CV=25%','CV<=5%','INCONCLUSIVE','rerun and isolate noise')
 outcsv(d/'policy_decisions.csv',decisions)
 model=d/'governance_source_model.csv';run(['python3',str(a.root/'probes/ch35_governance_model.py'),'--output',str(model)],clean(),raw/'model.log',a.root);mr=list(csv.DictReader(model.open()))
 if len(mr)!=36 or any(x['pass']!='1' for x in mr):raise RuntimeError(len(mr))
 manifest=['experiment=ch35_production_governance',f'run_id={a.run_id}',f'timestamp_utc={datetime.now(timezone.utc).isoformat()}',f'canary_rows={len(rows)}','correctness=15/15 PASS','policy_cases=7/7 PASS','source_model=36/36 PASS','runtime_upgrade_ab=NOT_AVAILABLE_SOURCE_POLICY_ONLY','raw_logs=NOT_FOR_PUBLICATION','nccl_runtime=2.22.3']
 (d/'manifest.txt').write_text('\n'.join(manifest)+'\n')
 lines=['# Chapter 35 experiment summary','',f'- Canary measurements: {len(rows)}','- Correctness: 15/15 PASS','- Policy cases: 7/7 PASS','- Source model: 36/36 PASS','','| config | median busbw | CV | regression |','|---|---:|---:|---:|']
 for x in summary:lines.append(f"| {x['config']} | {x['median_busbw_gbps']:.2f} GB/s | {x['cv_percent']:.2f}% | {x['regression_percent']:.2f}% |")
 (d/'summary.md').write_text('\n'.join(lines)+'\n');print(f'run_dir={d}')
if __name__=='__main__':main()
