#!/usr/bin/env python3
"""Model NCCL plugin ABI fallback, CollNet eligibility, and NVLS gates."""
import argparse,csv
from pathlib import Path
def abi(symbols,prefix):
  for v in (8,7,6,5):
    if f"{prefix}_v{v}" in symbols:return v
  return 0
def nvls(enable,gpus,mc_api,mc_attr,nvs_nodes,cc):
  if enable==0 or gpus<=2:return (0,0)
  support=mc_attr if enable==2 and mc_api else (1 if enable==1 else 0)
  channels=int(support and nvs_nodes>0 and cc>=90)
  return support,channels
def collnet(symbol,init_ok,devices,env_enable,graph_channels,nodes,threshold,local_ranks,arity=8):
  discovered=int(symbol and init_ok and devices>0)
  enabled=int(discovered and env_enable==1)
  graph=int(enabled and graph_channels>0)
  final=int(graph and nodes>=threshold and local_ranks<=arity)
  return discovered,enabled,graph,final
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
  def add(topic,case,expected,observed):rows.append({"topic":topic,"case":case,"expected":str(expected),"observed":str(observed),"pass":int(expected==observed)})
  for name,s,expected in (("v8",{"ncclNetPlugin_v8"},8),("v7",{"ncclNetPlugin_v7","ncclNetPlugin_v5"},7),("v5",{"ncclNetPlugin_v5"},5),("v4_reject",{"ncclNetPlugin_v4"},0),("none",set(),0)):
    add("net_abi",name,expected,abi(s,"ncclNetPlugin"))
  for name,s,expected in (("coll_v8",{"ncclCollNetPlugin_v8"},8),("coll_v6",{"ncclCollNetPlugin_v6"},6),("coll_v4_reject",{"ncclCollNetPlugin_v4"},0),("none",set(),0)):
    add("collnet_abi",name,expected,abi(s,"ncclCollNetPlugin"))
  collcases=(("no_symbol",0,1,1,1,4,2,4,0),("init_fail",1,0,1,1,4,2,4,0),("zero_devices",1,1,0,1,4,2,4,0),
    ("env_off",1,1,1,0,4,2,4,0),("zero_graph",1,1,1,1,0,2,4,0),("below_nodes",1,1,1,1,4,1,4,0),
    ("too_many_local",1,1,1,1,4,2,9,0),("all_pass",1,1,1,1,4,2,8,1))
  for name,s,i,d,e,g,n,t,expected in collcases:
    final=collnet(s,i,d,e,g,n,2,t)[-1];add("collnet_final",name,expected,final)
  nvlscases=(("disabled",0,4,1,1,1,90,(0,0)),("two_gpus",2,2,1,1,1,90,(0,0)),("auto_no_api",2,4,0,1,1,90,(0,0)),
    ("auto_attr0",2,4,1,0,1,90,(0,0)),("forced_v100",1,4,1,0,0,70,(1,0)),("hopper_no_nvs",2,4,1,1,0,90,(1,0)),
    ("hopper_nvs",2,4,1,1,1,90,(1,1)))
  for name,e,g,api,attr,nvs,cc,expected in nvlscases:
    observed=nvls(e,g,api,attr,nvs,cc);add("nvls_support_channels",name,f"{expected[0]}/{expected[1]}",f"{observed[0]}/{observed[1]}")
  # Algorithm table eligibility after final communicator capabilities.
  algos=("Ring","Tree","CollNetDirect","CollNetChain","NVLS","NVLSTree")
  for coll,nv in ((0,0),(1,0),(0,1),(1,1)):
    observed=[]
    for algo in algos:
      ok=algo in ("Ring","Tree") or (algo.startswith("CollNet") and coll) or (algo.startswith("NVLS") and nv)
      if ok:observed.append(algo)
    expected={ (0,0):"Ring;Tree",(1,0):"Ring;Tree;CollNetDirect;CollNetChain",(0,1):"Ring;Tree;NVLS;NVLSTree",(1,1):"Ring;Tree;CollNetDirect;CollNetChain;NVLS;NVLSTree"}[(coll,nv)]
    add("algorithm_eligibility",f"coll_{coll}_nvls_{nv}",expected,";".join(observed))
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
