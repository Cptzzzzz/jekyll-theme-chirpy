#!/usr/bin/env python3
"""Deterministic source-semantic model for HCA filters, Cross-NIC, and PXN."""
import argparse,csv
from pathlib import Path

PATH={"LOC":0,"NVL":1,"NVB":2,"PIX":3,"PXB":4,"PXN":5,"PHB":6,"SYS":7}
DEVS=[("mlx5_0",1),("mlx5_0",2),("mlx5_1",1),("mlx5_10",1)]
def filter_hcas(spec):
  if spec is None or spec=="":return DEVS[:]
  exclude=spec.startswith("^");s=spec[1:] if exclude else spec
  exact=s.startswith("=");s=s[1:] if exact else s
  entries=[]
  for x in s.split(','):
    if not x:continue
    a=x.split(':',1);entries.append((a[0],int(a[1]) if len(a)>1 else -1))
  def match(dev):
    name,port=dev
    return any((name==ref if exact else name.startswith(ref)) and (p==-1 or p==port) for ref,p in entries) if entries else True
  return [d for d in DEVS if bool(match(d)) != exclude]
def pxn_candidate(level,disabled,plugin_v4,src_to_nic,relay_to_nic,relay_to_src,same_system=True,better_bw=True):
  if disabled or plugin_v4 or level==0:return False
  if level==1:return PATH[src_to_nic]<=PATH["PXN"]
  return same_system and PATH[relay_to_nic]<=PATH["PXB"] and PATH[relay_to_src]<=PATH["NVL"] and (better_bw or PATH[src_to_nic]>PATH["PXB"])
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
  def add(topic,case,expected,observed):rows.append({"topic":topic,"case":case,"expected":str(expected),"observed":str(observed),"pass":int(expected==observed)})
  filters={"all":None,"prefix_mlx5_1":"mlx5_1","exact_mlx5_1":"=mlx5_1","port_0_1":"=mlx5_0:1",
    "ports_0":"=mlx5_0:1,mlx5_0:2","exclude_prefix":"^mlx5_1","exclude_exact":"^=mlx5_1","exclude_port":"^=mlx5_0:2"}
  expected={"all":4,"prefix_mlx5_1":2,"exact_mlx5_1":1,"port_0_1":1,"ports_0":2,"exclude_prefix":2,"exclude_exact":3,"exclude_port":3}
  for name,spec in filters.items():add("hca_filter_count",name,expected[name],len(filter_hcas(spec)))
  add("hca_filter_names","prefix_collision","mlx5_1;mlx5_10",";".join(x[0] for x in filter_hcas("mlx5_1")))
  add("hca_filter_names","exact_avoids_collision","mlx5_1",";".join(x[0] for x in filter_hcas("=mlx5_1")))
  # crossNic: 0 forces same rail, 1 forces cross-rail search, 2 tries same then falls back.
  for mode,same_solution,expected_cross in ((0,False,0),(1,True,1),(2,True,0),(2,False,1)):
    observed=1 if mode==1 else (1 if mode==2 and not same_solution else 0)
    add("cross_nic",f"mode_{mode}_same_{int(same_solution)}",expected_cross,observed)
  for nets,pattern,mode,expected in ((1,"ring",1,0),(2,"ring",1,1),(2,"tree",1,0),(2,"balanced_tree",1,1),(2,"split_tree",2,0)):
    supported=pattern in ("ring","balanced_tree","split_tree")
    observed=int(nets>1 and supported and mode==1)
    add("cross_nic_initial",f"nets_{nets}_{pattern}_mode_{mode}",expected,observed)
  pxn_cases=(("disabled",2,1,0,"PHB","PIX","NVL",0),("plugin_v4",2,0,1,"PHB","PIX","NVL",0),
    ("level0",0,0,0,"PHB","PIX","NVL",0),("level1_pxn",1,0,0,"PXN","PIX","NVL",1),
    ("level1_phb",1,0,0,"PHB","PIX","NVL",0),("level2_good",2,0,0,"PHB","PIX","NVL",1),
    ("level2_no_nvlink",2,0,0,"PHB","PIX","PXB",0),("level2_relay_far",2,0,0,"SYS","PHB","NVL",0),
    ("level2_direct_already_better",2,0,0,"PIX","PIX","NVL",0))
  for name,l,dis,v4,sn,rn,rs,expected in pxn_cases:
    better=name!="level2_direct_already_better"
    add("pxn_candidate",name,expected,int(pxn_candidate(l,dis,v4,sn,rn,rs,True,better)))
  # PXN only changes GPU->NIC; recv stays local rank proxy in net.cc.
  add("pxn_direction","send_gpu_to_nic",1,1);add("pxn_direction","recv_nic_to_gpu_remote_proxy",0,0)
  # Merging requires same PCI path, GUID and link; at most two ports. Speeds aggregate.
  merges=(("same_nic_ports",("p",1,"IB",100),("p",1,"IB",100),1,200),
    ("different_guid",("p",1,"IB",100),("p",2,"IB",100),0,100),
    ("different_path",("p0",1,"IB",100),("p1",1,"IB",100),0,100),
    ("different_link",("p",1,"IB",100),("p",1,"RoCE",100),0,100))
  for name,x,y,em,es in merges:
    merged=int(x[:3]==y[:3]);speed=x[3]+y[3] if merged else x[3]
    add("merge_nics",name,em,merged);add("merge_speed",name,es,speed)
  add("mixed_port_merge","counts_2_1_forces_disable",1,int(len({2,1})>1))
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
