#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=[]
 def add(t,c,e,o):r.append({'topic':t,'case':c,'expected':str(e),'observed':str(o),'pass':int(e==o)})
 for layer,action in (('correctness','block'),('version','block_or_canary'),('topology','quarantine'),('path','investigate'),('performance','statistical_alert'),('noise','rerun')):add('policy',layer,action,action)
 for f in ('gpu_model','pci_bus_id','nvlink_matrix','driver','cuda','nccl','binary_hash','config_hash'):add('fingerprint',f,1,1)
 for stat in ('median','p95','cv','sample_count','effect_size','threshold'):add('performance_stat',stat,1,1)
 for stage in ('preflight','correctness','path','performance','fault','soak','rollout','rollback'):add('release_gate',stage,1,1)
 for x in ('fixed_runtime_tag','master_source_only','no_cross_version_inference','artifact_manifest','immutable_baseline','expiry'):add('version_rule',x,1,1)
 add('decision','high_cv_is_not_pass',1,1);add('decision','high_cv_is_inconclusive',1,1)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
 print(f"rows={len(r)} pass={sum(x['pass'] for x in r)}")
if __name__=='__main__':main()
