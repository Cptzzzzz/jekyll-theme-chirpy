# NCCL / DDP Infra 学习教程

这份教程面向 GPU infra / ML infra 工程师，目标不是“会改 NCCL”，而是能把多卡训练慢、卡住、报错的问题定位到具体层：训练代码、rank 启动、DDP、NCCL、GPU 拓扑、NVLink/PCIe、IB/RDMA、容器权限或版本兼容。

当前机器适合做单机实验：

- GPU: 4 x Tesla V100-SXM2-32GB
- GPU 拓扑: 4 张 GPU 两两 `NV2`
- NVIDIA driver: 580.159.03
- CUDA toolkit: 12.6, `nvcc` 位于 `/usr/local/cuda/bin/nvcc`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- PyTorch CUDA: 12.6
- PyTorch NCCL: 2.22.3

## 先回答：要不要从源码学

不要从 NCCL 源码开始。推荐路线是：

```text
训练现象 -> DDP/rank 概念 -> collective 语义 -> NCCL 日志
        -> nccl-tests benchmark -> 拓扑/NVLink/RDMA -> 再读源码入口
```

源码在这里的作用是把日志、实验和真实实现对应起来。你应该先能跑、能测、能读日志，再回头看源码。

## 本目录内容

```text
/root/nccl-learning
├── README.md
├── blogs/
├── logs/
├── scripts/
│   ├── 00_env_report.sh
│   ├── 01_collectives.py
│   ├── 02_ddp_toy_train.py
│   ├── 03_mismatch_hang_demo.py
│   ├── 04_build_nccl_tests.sh
│   ├── 05_run_nccl_tests_single_node.sh
│   ├── 06_compare_p2p.sh
│   ├── 07_nccl_debug_ddp.sh
│   └── 08_multinode_rdma_check.sh
└── third_party/
    ├── nccl/
    └── nccl-tests/
```

开源仓库：

- NCCL: <https://github.com/NVIDIA/nccl>
- nccl-tests: <https://github.com/NVIDIA/nccl-tests>
- PyTorch distributed docs: <https://docs.pytorch.org/docs/stable/distributed.html>
- NCCL docs: <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>

如果你想按“博客 + 实验 + 结论”的方式学习，直接从这里开始：

```text
blogs/README.md
```

其中 `blogs/11-parameter-sweeps.md` 记录了参数 sweep：message size、GPU 数、collective 类型、P2P/SHM 开关、channel 数和 Ring/Tree 算法对性能的影响。

专家课程的当前规范和正式结果索引：

- [CURRICULUM.md](CURRICULUM.md)：36 章专家路线、实验和验收标准
- [EXPERIMENT_STANDARD.md](EXPERIMENT_STANDARD.md)：正确性、性能、路径和源码证据规范
- [ARTICLE_STANDARD.md](ARTICLE_STANDARD.md)：深度文章结构、源码摘录与注释规范
- [SOURCE_EVIDENCE_AUDIT.md](SOURCE_EVIDENCE_AUDIT.md)：37 篇声明、实现与省略边界审计
- [RESULTS_INDEX.md](RESULTS_INDEX.md)：正式 run ID、关键结果和硬件阻塞项

## 快速开始

先采集环境：

```bash
cd /root/nccl-learning
./scripts/00_env_report.sh | tee logs/env_report.log
```

跑 PyTorch collective：

```bash
torchrun --standalone --nproc_per_node=4 ./scripts/01_collectives.py
```

跑一个最小 DDP 训练：

```bash
torchrun --standalone --nproc_per_node=4 ./scripts/02_ddp_toy_train.py
```

本机 Python 环境有 onnx/protobuf 版本不兼容；脚本内部设置了 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，避免 DDP 初始化经过 `torch._dynamo` 时被这个非 NCCL 问题打断。

构建并运行 nccl-tests：

```bash
./scripts/04_build_nccl_tests.sh
MAX_BYTES=1G GPUS=4 ./scripts/05_run_nccl_tests_single_node.sh
```

打开 NCCL debug 日志：

```bash
./scripts/07_nccl_debug_ddp.sh
```

比较是否禁用 P2P/NVLink 路径后的性能变化：

```bash
MAX_BYTES=1G GPUS=4 ./scripts/06_compare_p2p.sh
```

## 学习路线总表

| 阶段 | 知识点 | 实验脚本 | 关联源码/资料 | 学到什么程度 |
|---|---|---|---|---|
| 1 | rank/local_rank/world_size/process group | `01_collectives.py`, `02_ddp_toy_train.py` | PyTorch distributed docs | 能解释每个进程绑定哪张 GPU |
| 2 | collective 语义 | `01_collectives.py` | `nccl-tests/src/all_reduce.cu`, `all_gather.cu`, `reduce_scatter.cu` | 知道 all-reduce/all-gather/reduce-scatter 在训练里的作用 |
| 3 | DDP 底层通信 | `02_ddp_toy_train.py`, `07_nccl_debug_ddp.sh` | PyTorch DDP docs, NCCL logs | 知道 DDP backward 时会触发梯度同步 |
| 4 | communicator 初始化 | `05_run_nccl_tests_single_node.sh` | `nccl-tests/src/common.cu`, `nccl/src/init.cc` | 知道 rank 如何加入同一个 NCCL communicator |
| 5 | NCCL 日志 | `07_nccl_debug_ddp.sh` | `nccl/src/debug.cc`, NCCL env docs | 能从日志判断 INIT/NET/GRAPH/COLL/P2P/SHM |
| 6 | 单机拓扑/NVLink/P2P | `00_env_report.sh`, `06_compare_p2p.sh` | `nvidia-smi topo -m`, `nccl/src/transport.cc` | 能判断是否走 P2P/NVLink，禁用后性能是否下降 |
| 7 | benchmark baseline | `05_run_nccl_tests_single_node.sh` | `nccl-tests/doc/PERFORMANCE.md`, `util.cu` | 能读 `algbw`/`busbw`，建立节点基线 |
| 8 | 多机 RDMA | `08_multinode_rdma_check.sh` | `nccl/src/transport/net_ib`, GPUDirect RDMA docs | 能判断是否看到 IB/RDMA、是否可能 fallback TCP |
| 9 | 故障注入 | `03_mismatch_hang_demo.py` | PyTorch/NCCL timeout 日志 | 理解 rank 代码路径不一致为什么会 hang |
| 10 | 源码阅读 | 不需要先跑 | `nccl/src/{init,collectives,transport,graph}` | 能把日志和源码模块对应起来 |

## 阶段 1：DDP 和 rank 模型

DDP 通常是“一张 GPU 一个进程”。在这台机器上：

```bash
torchrun --standalone --nproc_per_node=4 ./scripts/01_collectives.py
```

你会得到 4 个 rank。关键概念：

```text
rank        全局进程编号
local_rank  当前节点内的 GPU 编号
world_size  总进程数
backend     GPU 训练通常是 nccl
```

`01_collectives.py` 里的核心代码：

```python
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
dist.all_gather(gathered, y)
dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
```

你要能解释：

```text
每个 rank 都必须进入同一个 collective。
只要某个 rank 少执行了一次 all_reduce，其他 rank 就会等它，最后 hang 或 timeout。
```

故障注入：

```bash
torchrun --standalone --nproc_per_node=2 ./scripts/03_mismatch_hang_demo.py
```

这个脚本故意让 rank 0 跳过 `all_reduce`，其他 rank 会等待，最终触发 timeout。生产里的很多 NCCL timeout 就是这种逻辑问题伪装成通信问题。

## 阶段 2：collective 和训练动作的对应关系

常见 collective：

```text
all_reduce       DDP 梯度同步
broadcast        初始化参数、状态同步
all_gather       参数/激活/张量并行收集
reduce_scatter   ZeRO/FSDP 梯度或参数分片
barrier          所有 rank 等齐
```

在 `nccl-tests` 里，对应源码入口非常直接：

```text
third_party/nccl-tests/src/all_reduce.cu:507        ncclAllReduce(...)
third_party/nccl-tests/src/all_gather.cu:50         ncclAllGather(...)
third_party/nccl-tests/src/reduce_scatter.cu:49     ncclReduceScatter(...)
third_party/nccl-tests/src/broadcast.cu:50          ncclBroadcast(...)
```

这几个文件比 NCCL 主仓库更适合初学，因为它们展示“如何调用 NCCL”，而不是 NCCL 内部如何实现。

## 阶段 3：DDP 底层是什么

运行：

```bash
torchrun --standalone --nproc_per_node=4 ./scripts/02_ddp_toy_train.py
```

逻辑是：

```text
每个 rank 有完整模型副本
每个 rank 读不同 data shard
forward/backward 在本地 GPU 上执行
DDP 在 backward 过程中触发梯度 all-reduce
每个 rank 得到相同平均梯度
每个 rank 各自 optimizer.step()
```

底层栈：

```text
训练代码
  -> PyTorch DDP
  -> torch.distributed/c10d
  -> ProcessGroupNCCL
  -> NCCL
  -> CUDA
  -> NVLink / PCIe / IB RDMA / TCP
```

打开 NCCL 日志看 DDP 背后的通信：

```bash
./scripts/07_nccl_debug_ddp.sh
```

重点看这些词：

```text
Init COMPLETE
Channel
Connected
P2P
SHM
NET
Socket
IB
busId
cudaDev
```

## 阶段 4：NCCL communicator 初始化

NCCL collective 不是直接“随便几个 GPU 通信”。它先要创建 communicator：所有 rank 拿到同一个 `ncclUniqueId`，然后各自调用 `ncclCommInitRank` 加入同一个通信域。

`nccl-tests` 的关键入口：

```text
third_party/nccl-tests/src/common.cu:211-221
```

关键逻辑：

```cpp
ncclGroupStart();
cudaSetDevice(cudaDevs[i]);
ncclCommInitRankConfig(comms+i, nRanks, ncclId, rank, &config);
ncclGroupEnd();
```

NCCL 主仓库入口：

```text
third_party/nccl/src/init.cc:2561-2572
```

关键逻辑：

```cpp
ncclInitEnv();
cudaGetDevice(&cudaDev);
ncclCommInitRankDev(..., myrank, cudaDev, ...);
```

你要形成的理解：

```text
rank、cuda device、world size、唯一通信 ID 必须一致。
duplicate GPU、rank 数不一致、master 地址不通，都会在初始化阶段暴露。
```

## 阶段 5：NCCL debug 日志

最常用：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL,P2P,SHM,ENV
```

NCCL 源码里 debug 子系统定义在：

```text
third_party/nccl/src/debug.cc:55-72
```

对应关系：

```text
INIT      communicator 初始化
BOOTSTRAP rank 之间建立初始连接
ENV       环境变量和配置
GRAPH     拓扑搜索
P2P       GPU peer-to-peer transport
SHM       shared memory transport
NET       network transport
COLL      collective 调用
NVLS      NVLink SHARP
```

实验：

```bash
./scripts/07_nccl_debug_ddp.sh
```

这会把完整日志写到 `logs/ddp_nccl_debug_*.log`，并额外 grep 出关键行。

如果命令退出码是 0，日志末尾的 `Abort COMPLETE`、`socketProgressOpt: abort called` 一类清理信息通常只是 communicator teardown，不要单独当成训练失败。

## 阶段 6：单机拓扑、NVLink 和 P2P

先看拓扑：

```bash
nvidia-smi topo -m
```

当前机器显示 GPU 两两 `NV2`，说明 GPU 间有 NVLink 路径。NCCL 在单机通信时会在 P2P/SHM/NET 等 transport 中选择可用路径。

NCCL transport 选择入口：

```text
third_party/nccl/src/transport.cc:15-18
third_party/nccl/src/transport.cc:27-36
```

关键逻辑是按 transport 顺序检查 `canConnect`，能连就 `setup`：

```cpp
&p2pTransport, &shmTransport, &netTransport, &collNetTransport
transport->canConnect(...)
transportComm->setup(...)
```

实验：

```bash
MAX_BYTES=1G GPUS=4 ./scripts/06_compare_p2p.sh
```

它会跑三组：

```text
baseline
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
```

如果禁用 P2P 后性能明显下降，说明 baseline 很可能利用了 GPU P2P/NVLink 路径。

## 阶段 7：nccl-tests benchmark

构建：

```bash
./scripts/04_build_nccl_tests.sh
```

单机 4 卡测试：

```bash
MAX_BYTES=1G GPUS=4 ./scripts/05_run_nccl_tests_single_node.sh
```

重点列：

```text
time   单次 collective 耗时
algbw  算法视角的吞吐
busbw  总线/链路视角的有效吞吐
#wrong 数据校验错误数，正常应为 0
```

输出列定义入口：

```text
third_party/nccl-tests/src/util.cu:756-760
```

读结果时不要只看最大值。至少看三件事：

```text
小消息是否延迟异常高
大消息是否达到稳定带宽
禁用 P2P/SHM/IB 后性能是否符合预期下降
```

建议保存基线：

```bash
mkdir -p logs/baseline
MAX_BYTES=1G GPUS=4 ./scripts/05_run_nccl_tests_single_node.sh
cp logs/*_g4_*.log logs/baseline/
```

以后驱动、CUDA、容器、NCCL、PyTorch 镜像升级后，先和这个 baseline 对比。

## 阶段 8：多机 RDMA/IB/RoCE

当前机器可以检查 RDMA 工具，但是否有实际 IB/RoCE fabric 要看节点环境。

检查：

```bash
./scripts/08_multinode_rdma_check.sh
```

多机时你要排查：

```text
/dev/infiniband 是否在容器内可见
ibv_devinfo 是否能看到 mlx5 设备
GPU 和 NIC 的 NUMA/PCIe 距离
NCCL 日志是否出现 NET/IB
是否 fallback 到 Socket/TCP
RoCE 的 GID index、MTU、PFC/ECN 是否正确
```

NCCL IB transport 入口：

```text
third_party/nccl/src/transport/net_ib/init.cc
third_party/nccl/src/transport/net_ib/connect.cc
```

常见变量在源码里对应：

```text
NCCL_IB_DISABLE      net_ib/init.cc:27
NCCL_IB_HCA          net_ib/init.cc:317-318
NCCL_IB_GID_INDEX    net_ib/connect.cc:12
```

典型多机 torchrun：

```bash
# node 0
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=<node0_ip> \
  --master_port=29500 \
  ./scripts/02_ddp_toy_train.py

# node 1
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=4 \
  --master_addr=<node0_ip> \
  --master_port=29500 \
  ./scripts/02_ddp_toy_train.py
```

典型多机 nccl-tests 需要 MPI 版本：

```bash
make MPI=1 CUDA_HOME=/usr/local/cuda
mpirun -np 8 -N 4 ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

没有 MPI 或没有多机环境时，不要在单机上硬套这些命令。

## 阶段 9：源码阅读顺序

推荐顺序：

1. `nccl-tests/src/all_reduce.cu`
2. `nccl-tests/src/common.cu`
3. `nccl-tests/doc/PERFORMANCE.md`
4. `nccl/src/collectives.cc`
5. `nccl/src/init.cc`
6. `nccl/src/debug.cc`
7. `nccl/src/transport.cc`
8. `nccl/src/transport/p2p.cc`
9. `nccl/src/transport/shm.cc`
10. `nccl/src/transport/net.cc`
11. `nccl/src/transport/net_ib/`
12. `nccl/src/graph/`
13. `nccl/src/device/`

前 8 个够大部分 infra 排障。`src/device/` 和协议/kernel 细节属于通信库工程师方向，不是第一优先级。

## 排障 runbook

遇到 “NCCL error / hang / timeout / 多卡慢” 时按这个顺序：

```text
1. nvidia-smi 是否正常，GPU 是否被占满或有 Xid/ECC 问题
2. torch.cuda.is_available() 和 torch.cuda.device_count()
3. torchrun 的 nnodes/node_rank/nproc_per_node/master_addr/master_port 是否一致
4. 每个 rank 是否绑定唯一 GPU，是否 duplicate GPU
5. 是否所有 rank 都进入同一个 collective
6. NCCL_DEBUG=INFO 看卡在 INIT、NET、GRAPH 还是 COLL
7. 单机先跑 nccl-tests，确认节点内 P2P/NVLink 正常
8. 多机再查 IB/RDMA、容器 /dev/infiniband、路由、防火墙、GID/MTU
9. 对比历史 baseline，判断是平台退化还是训练代码变化
10. 必要时用 NCCL_P2P_DISABLE/NCCL_IB_DISABLE/NCCL_SOCKET_IFNAME 做隔离实验
```

注意：`NCCL_*_DISABLE` 这类变量主要用于隔离问题，不要随手固化到生产默认配置。

## 你应该达到的标准

第一档，单机合格：

```text
能跑 DDP
能跑 collective demo
能跑 nccl-tests
能看 nvidia-smi topo -m
能打开 NCCL_DEBUG 并读关键行
能用 P2P/SHM disable 做隔离实验
```

第二档，GPU 集群 infra：

```text
能判断多机是否走 IB/RDMA
能识别 Socket/TCP fallback
能处理容器/K8s RDMA 设备暴露
能建立节点级 NCCL baseline
能区分训练代码问题和平台问题
```

第三档，通信库/超大规模优化：

```text
读 NCCL transport/graph/device 层源码
理解 ring/tree/channel/protocol
调 SHARP/NVLS/CollNet
写或调 NCCL net plugin/topology config
```

作为 infra 工程师，先稳定到第二档。源码读到能解释日志和性能现象即可。
