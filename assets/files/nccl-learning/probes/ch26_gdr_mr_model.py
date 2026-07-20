#!/usr/bin/env python3
"""Independent invariants for NCCL GDR topology gates and registration caches."""
import argparse,csv
from pathlib import Path

LEVEL={"LOC":0,"NVL":1,"NVB":2,"PIX":3,"PXB":4,"PXN":5,"PHB":6,"SYS":7}
def gdr(net,gpu,read,cc,nvlink,dist,level="PXB",read_param=-2,proxy_dist=None):
  if not net or not gpu:return 0
  if read:
    if read_param==0:return 0
    if read_param<0 and cc<80 and not nvlink:return 0
  d=LEVEL[proxy_dist] if dist=="PXN" else LEVEL[dist]
  return int(d<=LEVEL[level])
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
  def add(topic,case,expected,observed):rows.append({"topic":topic,"case":case,"expected":expected,"observed":observed,"pass":int(expected==observed)})
  cases=[("no_net",0,1,0,70,1,"PIX","PXB",-2,None,0),("no_gpu",1,0,0,70,1,"PIX","PXB",-2,None,0),
    ("recv_pix",1,1,0,70,1,"PIX","PXB",-2,None,1),("recv_phb",1,1,0,70,1,"PHB","PXB",-2,None,0),
    ("read_v100_nvlink",1,1,1,70,1,"PXB","PXB",-2,None,1),("read_v100_no_nvlink",1,1,1,70,0,"PIX","PXB",-2,None,0),
    ("read_disabled",1,1,1,90,1,"PIX","SYS",0,None,0),("level_sys",1,1,0,70,1,"SYS","SYS",-2,None,1),
    ("pxn_proxy_pix",1,1,0,70,1,"PXN","PXB",-2,"PIX",1),("pxn_proxy_phb",1,1,0,70,1,"PXN","PXB",-2,"PHB",0)]
  for c in cases:add("gdr_gate",c[0],c[-1],gdr(*c[1:-1]))
  page=4096
  for addr,size,expected_addr,expected_pages in ((0x1003,1,0x1000,1),(0x1003,4096,0x1000,2),(0x2000,8192,0x2000,2),(0x2fff,2,0x2000,2)):
    base=addr&-page;pages=(addr+size-base+page-1)//page
    add("page_align",f"{addr:x}_{size}_addr",expected_addr,base);add("page_align",f"{addr:x}_{size}_pages",expected_pages,pages)
  # User cache: exact/contained registrations share a record; partial overlap does not.
  parent=(0x1000,4);queries=[("exact",0x1000,4,1),("contained",0x2000,2,1),("left_overlap",0x0800,2,0),("right_overlap",0x4000,2,0),("disjoint",0x8000,1,0)]
  for name,addr,pages,expected in queries:
    hit=int(addr>=parent[0] and ((addr-parent[0])//page+pages)<=parent[1]);add("cache_containment",name,expected,hit)
  for fd,relaxed,expected in ((-1,0,"ibv_reg_mr"),(-1,1,"ibv_reg_mr_iova2"),(7,0,"ibv_reg_dmabuf_mr"),(7,1,"ibv_reg_dmabuf_mr")):
    observed="ibv_reg_dmabuf_mr" if fd!=-1 else ("ibv_reg_mr_iova2" if relaxed else "ibv_reg_mr")
    add("registration_api",f"fd_{fd}_ro_{relaxed}",expected,observed)
  for old,new,expected in ((0,1,32),(32,33,64),(64,65,128)):
    cap=32 if old<32 else 2*old;add("cache_growth",f"{old}_to_{new}",expected,cap)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
