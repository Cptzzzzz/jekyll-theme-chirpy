# NCCL 专家课程大纲 v2

这套课程面向 GPU Infra / AI Infra 工程师。目标不是记住 NCCL 环境变量，而是能够预测执行路径、读懂实现、设计可证伪实验，并处理生产中的性能回归和通信故障。

课程固定使用以下主线：

```text
训练框架
  -> ProcessGroupNCCL / communicator
  -> collective contract / task queue
  -> planner / tuning / work batch
  -> algorithm / protocol / channel
  -> transport / proxy / device primitive
  -> NVLink / PCIe / Socket / IB / RoCE
```

文章规范见 [ARTICLE_STANDARD.md](ARTICLE_STANDARD.md)，实验规范见 [EXPERIMENT_STANDARD.md](EXPERIMENT_STANDARD.md)。旧文章和旧实验只能作为基线材料；只有按新版规范补齐源码、参数实验和推导后，章节才算完成。

## 环境与源码边界

```text
GPU                 4 x Tesla V100-SXM2-32GB
GPU topology        GPU 两两 NV2
NUMA                GPU 位于 NUMA 0
Driver              580.159.03
CUDA toolkit        12.6
PyTorch             2.5.0a0+872d972e41.nv24.08
Runtime NCCL        2.22.3
Runtime source      v2.22.3-1 @ 178b6b7
RDMA HCA            当前机器无可见 HCA
```

状态定义：

- `PLANNED`：实验和文章尚未按新版规范完成。
- `IN_PROGRESS`：正在分析、运行实验或撰写。
- `VALIDATED`：源码、实验、博客和公网附件均已复核。
- `RDMA`：需要带 IB/RoCE HCA 的多机环境。
- `NVSWITCH`：需要支持 NVLS 的 GPU/NVSwitch 环境。

## 卷一：系统基础

### 01. NCCL 边界、软件栈、版本与动态链接

- 核心：驱动、CUDA Driver/Runtime、NCCL、PyTorch c10d、MPI 各自负责什么；编译版本、运行版本、SONAME、ABI 和符号解析。
- 源码：`src/nccl.h.in`、`src/init.cc`；PyTorch 的 NCCL version API；ELF dynamic section。
- 实验：比较包版本、header 宏、`ldd`、`ldconfig`、`LD_DEBUG=libs` 和 `/proc/<pid>/maps`；用不同 `LD_LIBRARY_PATH` 做正反例。
- 验收：能证明某进程实际加载哪个 `libnccl.so`，并解释为什么 `nvidia-smi` 显示的 CUDA 版本不是 toolkit 版本。
- 状态：`VALIDATED`，正式运行 `ch01_stack_version/20260710T065547Z`。

### 02. GPU、PCIe、NVLink、NVSwitch 与 NUMA 拓扑

- 核心：PCI domain/bus/device、PCI switch、PHB、NUMA、NVLink lane、NVSwitch；`PIX/PXB/PHB/SYS/NVL` 的含义。
- 源码：`graph/xml.cc`、`graph/topo.cc` 中物理节点和 link 的构造入口。
- 实验：GPU 两两 peer access、单/双向 copy、不同 payload、CPU NUMA 绑定；关联 `nvidia-smi topo -m`、NVML 与 PCI sysfs。
- 验收：从硬件图预测 P2P 路径和性能矩阵，并识别异常链路或错误 rank 放置。
- 状态：`VALIDATED`，正式运行 `ch02_topology/20260710T070553Z`。

### 03. CUDA stream、event、异步完成与内存可见性

- 核心：host enqueue 与 device completion、legacy/default stream、event dependency、pinned/device memory、buffer lifetime 和 memory ordering。
- 源码：NCCL API 的 stream 参数、`src/group.cc`、`src/enqueue.cc` 的提交边界。
- 实验：测 host 返回、event 完成和同步等待；跨 stream 有/无 event；同步前读取结果和提前复用 buffer 的反例。
- 验收：能画出 CPU、两个 CUDA stream 和 NCCL kernel 的时序，并证明 API 返回不等于通信完成。
- 状态：`VALIDATED`，正式运行 `ch03_async/20260710T071511Z`。

### 04. Process、rank、device 与 communicator 生命周期

- 核心：global/local/node rank、world size、unique ID、device mapping、communicator split、blocking/nonblocking init、abort/destroy。
- 源码：`src/init.cc` 的 `ncclCommInitRank*`、`src/group.cc`、`include/comm.h`。
- 实验：world 与交叉 subgroup、communicator split、改变 `CUDA_VISIBLE_DEVICES`、duplicate device、原生 C API 生命周期。
- 验收：能从启动命令推导每个 rank 的 GPU 和 communicator，并定位 rank mapping 错误。
- 状态：`VALIDATED`，正式运行 `ch04_communicators/20260710T072334Z`。

## 卷二：Collective 契约与组合

### 05. Collective 的严格语义与数据布局

- 核心：Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、AllToAll、Send/Recv 的输入输出、count、root、in-place 和 datatype。
- 源码：`src/nccl.h.in`、`src/collectives.cc`、`src/include/info.h`、`src/misc/argcheck.cc`、`src/enqueue.cc`；PyTorch AllToAll 的 P2P 组合实现。
- 实验：rank 编码输入，CPU reference 逐元素断言；覆盖 out-of-place/in-place、不同 root 和非整齐值。
- 验收：不运行程序也能手算每个 rank 的输出和 buffer 大小。
- 状态：`VALIDATED`，正式运行 `ch05_collective_contracts/20260710T073928Z`；原生 C API 36 条与 PyTorch 56 条逐 rank 记录全部通过。

### 06. 调用顺序、Group、异步错误与完成语义

- 核心：同一 communicator 的全局操作顺序、`ncclGroupStart/End` 只保证批量提交而非同步完成、async error 和 teardown。
- 源码：`src/group.cc` 的 thread-local group state、`ncclGroupEndInternal` 和 async job。
- 实验：顺序一致/不一致、grouped send/recv、nested group、多 stream 和 rank skip。
- 验收：区分 host 阻塞、CUDA stream 依赖、collective rendezvous 和 watchdog timeout。
- 状态：`VALIDATED`，正式运行 `ch06_group_ordering/20260710T075642Z`；原生 Group/async 76 条记录通过，DETAIL mismatch、raw partial progress 和 rank-skip watchdog 均完成负向验证。

### 07. Collective 分解、等价变换与通信模式选型

- 核心：AllReduce = ReduceScatter + AllGather 的条件；分片训练为什么直接保留 ReduceScatter 结果；AllToAll 与点对点组合。
- 源码：`device/all_reduce.h`、`all_gather.h`、`reduce_scatter.h` 的执行结构。
- 实验：相同 payload 比较 AR 与 RS+AG 的结果、显存、kernel 数和时间；构造不等价的错误切分。
- 验收：能根据训练状态是否复制/分片选择 collective，而不是机械使用 AllReduce。
- 状态：`VALIDATED`，正式运行 `ch07_decomposition/20260710T081207Z`；20 条 contract、400 条 timing、16 条 memory 记录通过，Nsight 证明 direct/分解路径为 20/40 个 measured NCCL kernel。

## 卷三：性能测量与建模

### 08. 通信复杂度、algbw 与 busbw

- 核心：每 rank payload、链路字节数、Ring AllReduce 的 `2(N-1)/N` 系数，以及不同 collective 的 busbw 归一化。
- 源码：`nccl-tests/src/common.cu` 与各 collective 的 `GetBw` 实现。
- 实验：从原始时间独立重算 algbw/busbw，并对 world size 2/3/4 验证公式。
- 验收：能说明 busbw 是模型化指标，不是某条 NVLink 的直接计数器。
- 状态：`VALIDATED`，正式运行 `ch08_busbw/20260710T082109Z`；6 类 collective、3 个 world size、2 个消息尺寸共 360 条 out-of-place 记录全部正确，独立复算 algbw/busbw 的最大绝对误差为 0.00891/0.00912 GB/s。

### 09. 可重复 Benchmark 与统计证据

- 核心：warmup、同步边界、GPU clock/P-state、NUMA、in-place、迭代与独立 cycle、median/P95/CV。
- 源码：nccl-tests timer、aggregation 和 correctness 路径。
- 实验：warmup 数、iteration 数、blocking 模式和后台负载的控制实验；至少十个 cycle。
- 验收：CV 大于 5% 时拒绝给出精确优化结论，并能定位噪声来源。
- 状态：`VALIDATED`，正式运行 `ch09_benchmark/20260710T150350Z`；46 次独立进程、920 行 out-of-place 记录与对应 in-place 检查全部正确，完成 warmup/iteration/blocking/instrumentation/affinity/noise 六组控制变量实验。

### 10. 延迟-带宽模型与消息区间

- 核心：`T = alpha + n/beta`、固定开销、流水填充、延迟区/过渡区/饱和区和算法交叉点。
- 源码：`graph/tuning.cc` 的 latency/bandwidth table 与估时入口。
- 实验：1 B 到 1 GiB 密集扫描、分段拟合、残差分析，并比较模型预测与实测。
- 验收：知道为何小包看 latency、大包看 busbw，不能用一个 64 MiB 点代表全部 workload。
- 状态：`VALIDATED`，正式运行 `ch10_latency_model/20260710T151843Z`；1 B–1 GiB 共 31 个 size、310 行测量全部正确，验证 Ring+LL/LL128/Simple 切换、全局高 R2 失真与分段 alpha-beta 模型。

### 11. 性能回归的分层归因

- 核心：GPU kernel、channel、protocol、transport、CPU proxy、NUMA、网络和应用调度的瓶颈边界。
- 源码：把 debug 日志中的选择映射到 tuning、transport 和 enqueue 分支。
- 实验：分别注入 channel 限制、P2P 禁用、CPU 争用和错误 NUMA 绑定，建立可识别指纹。
- 验收：面对 20% 回归先定位层级，再决定是否调整 NCCL 环境变量。
- 状态：`VALIDATED`，正式运行 `ch11_regression/20260710T152727Z`；14 个性能进程、420 行测量全部正确，提取单 channel、P2P→SHM 跨层 fallback、NUMA 负向对照与同核 CPU 长尾四类指纹。

## 卷四：算法、协议与流水线

### 12. Ring AllReduce 的逐步推导

- 核心：ReduceScatter + AllGather、`2(N-1)` step、chunk owner、链路流水和慢链路上限。
- 源码：`device/all_reduce.h` 的 Ring 分支、`graph/rings.cc`。
- 实验：world size、消息大小、channel 三维 Sweep；从 GRAPH 日志重建每条 ring。
- 验收：能逐 step 写出四 rank 的 send/recv/reduce 数据块。
- 状态：`VALIDATED`，正式运行 `ch12_ring/20260710T154018Z`。

### 13. Tree、双树与层次化算法

- 核心：up/down tree、深度、双树并发、节点内外层次化，以及 Tree 延迟和 Ring 带宽的取舍。
- 源码：`device/all_reduce.h` Tree 分支、`graph/trees.cc`。
- 实验：Ring/Tree 全消息区间对照，并从日志复原 parent/children。
- 验收：找到本机真实交叉区间，不背诵“Tree 小包、Ring 大包”。
- 状态：`VALIDATED`，正式运行 `ch13_tree/20260710T155522Z`。

### 14. Simple、LL 与 LL128 协议

- 核心：payload 编码、flag、同步粒度、有效带宽和硬件限制；算法与协议是两个独立选择维度。
- 源码：`device/prims_simple.h`、`prims_ll.h`、`prims_ll128.h`。
- 实验：固定 Ring/Tree 后扫描三协议与完整消息区间；关联 kernel/trace 和正确性。
- 验收：从源码解释 LL 的低延迟与 payload 效率代价，以及 LL128 的适用边界。
- 状态：`VALIDATED`，正式运行 `ch14_protocol/20260710T161632Z`。

### 15. Channel、chunk、slice、step 与 buffer

- 核心：channel 与 CUDA block、chunk 切分、slice 粒度、FIFO step、`NCCL_BUFFSIZE` 和线程数。
- 源码：`include/channel.h`、`device/primitives.h`、`enqueue.cc` 的 chunk 计算。
- 实验：channel、buffer、thread、protocol 联合但分层的 Sweep；Nsight 验证 block 数。
- 验收：解释 channel 太少、过多和 buffer 改变为何可能分别变慢。
- 状态：`VALIDATED`，正式运行 `ch15_pipeline/20260710T191306Z`。

### 16. Tuning 模型与算法/协议自动选择

- 核心：latency/bandwidth table、拓扑修正、backup 路径和 tuner plugin 介入点。
- 源码：`graph/tuning.cc::ncclTopoTuneModel/ncclTopoGetAlgoTime`、`misc/tuner.cc`。
- 实验：重算小/中/大消息候选时间；比较自动选择、强制选择和实测最优。
- 验收：能解释自动选择为何合理或为何在特定 workload 上失准。
- 状态：`VALIDATED`，正式运行 `ch16_tuning/20260710T194200Z`。

## 卷五：初始化、拓扑、Transport 与执行引擎

### 17. Bootstrap 与 communicator 初始化状态机

- 核心：unique ID、bootstrap socket、peer info allgather、local rank、topology、graph、transport connect 和 async init。
- 源码：`bootstrap.cc`、`init.cc::ncclCommInitRankFunc/ncclCommInitRankDev`。
- 实验：rank 数 scaling、阶段耗时、错误地址/缺 rank/duplicate GPU 故障注入。
- 验收：初始化 hang 时能定位在 bootstrap、topology、graph 还是 connect。
- 状态：`VALIDATED`，正式运行 `ch17_init/20260710T200500Z`。

### 18. 拓扑 XML、节点模型与路径计算

- 核心：GPU/CPU/PCI/NIC/NET/NVS 节点、link 类型/带宽和最宽路径计算。
- 源码：`graph/xml.cc`、`graph/topo.cc`、`graph/paths.cc`。
- 实验：导出 XML，与 sysfs/NVML 对照；对 XML 做受控修改并观察模拟路径结果。
- 验收：从 XML 推导 NCCL 的路径等级，而不是只复述 `nvidia-smi topo`。
- 状态：`VALIDATED`，正式运行 `ch18_topology_paths/20260711T031500Z`。

### 19. Graph Search、Ring/Tree 与 Channel 构造

- 核心：pattern、crossNic、sameChannels、speed/type 约束、递归搜索、preset/postset。
- 源码：`graph/search.cc`、`graph/connect.cc`。
- 实验：dump/replay graph，修改搜索约束或拓扑 fixture，观察 channel 数和顺序变化。
- 验收：能解释 NCCL 为什么选择当前 graph，并识别 search fallback。
- 状态：`VALIDATED`，正式运行 `ch19_graph_search/20260711T044500Z`。

### 20. Transport 选择与连接生命周期

- 核心：P2P/SHM/NET/CollNet transport 的 `canConnect/setup/connect/free`，send/recv connector 和 connIndex。
- 源码：`transport.cc` 与各 `transport/*.cc` vtable。
- 实验：逐层禁用 transport，保存连接日志和性能；验证 fallback 顺序。
- 验收：同时用源码分支、日志和性能证明真实路径。
- 状态：`VALIDATED`，正式运行 `ch20_transport_selection/20260711T061500Z`。

### 21. P2P、CUDA IPC、direct pointer 与 SHM

- 核心：peer access、IPC handle、P2P read/write、direct pointer、SHM staging 和 memcpy mode。
- 源码：`transport/p2p.cc`、`transport/shm.cc`。
- 实验：GPU pair、P2P level、direct disable、P2P/SHM disable、memcpy mode 多变量分组测试。
- 验收：解释 NVLink 存在但 NCCL 不走 P2P 的所有主要原因。
- 状态：`VALIDATED`，正式运行 `ch21_p2p_ipc_shm/20260711T140000Z`。

### 22. Enqueue、Task、Plan 与 Work Batch

- 核心：`ncclInfo`、collective task、planner、channel schedule、work batch、kernel upload 和 launch。
- 源码：`collectives.cc -> enqueue.cc::ncclEnqueueCheck` 及 plan 构造函数。
- 实验：单操作、grouped 操作和 CUDA Graph；用 TRACE/NVTX 对齐 task 与 kernel。
- 验收：从 API 参数追到 device work element 中的字段。
- 状态：`VALIDATED`，正式运行 `ch22_enqueue_plan/20260711T151500Z`。

### 23. Device Kernel、Primitive 与同步协议

- 核心：`RunWorkColl`、send/recv/reduce/copy primitives、warp/thread role、step、flag、barrier 和 direct path。
- 源码：`device/common.h`、`all_reduce.h`、`primitives.h` 与三种 `prims_*`。
- 实验：Nsight kernel/block 证据；可控 debug build 或最小仿真验证 step/flag 行为。
- 验收：能逐层解释一个 Ring Simple chunk 在 GPU 上如何移动和归约。
- 状态：`VALIDATED`，正式运行 `ch23_device_primitives/20260711T154500Z`。

### 24. CPU Proxy、NET Progress 与 GPU/CPU 协作

- 核心：proxy connection、op pool、progress loop、sub operation、network completion 和共享 buffer。
- 源码：`proxy.cc`、`transport/net.cc`、`transport/net_socket.cc`。
- 实验：强制 NET/socket 路径，观察 proxy CPU、线程亲和性、系统调用和 GPU timeline。
- 验收：区分 GPU kernel 空转、proxy 不推进和网络 completion 慢。
- 状态：`VALIDATED`，正式运行 `ch24_proxy_socket/20260711T170000Z`。

## 卷六：多机网络与高级硬件

### 25. Socket、InfiniBand、RoCE 与 Verbs

- 核心：bootstrap socket 与 data transport、QP/CQ/MR、GID、MTU、PFC/ECN、retry/timeout。
- 源码：`transport/net_socket.cc`、`transport/net_ib.cc`。
- 实验：Socket/IB 对照、`ib_write_bw` 对照、GID/MTU/QP 参数矩阵和网络故障注入。
- 验收：能区分 NCCL、verbs、NIC、交换机和 TCP 问题。
- 状态：Socket 与源码模型 `VALIDATED`，正式运行 `ch25_socket_ib_roce/20260711T180000Z`；IB/RoCE 性能为 `HARDWARE_BLOCKED`。

### 26. GPUDirect RDMA、MR 与注册缓存

- 核心：NIC DMA GPU memory、DMA-BUF、`nvidia-peermem`、MR、registration cache、host bounce、ACS/IOMMU。
- 源码：`register.cc`、`transport/net.cc`、`transport/net_ib.cc`。
- 实验：GDR 开关、GPU/NIC 距离、注册机制和消息大小矩阵。
- 验收：用拓扑、日志和性能共同证明是否走 GDR。
- 状态：CUDA capability、registration cache、Socket 负对照与源码模型 `VALIDATED`，正式运行 `ch26_gdr_registration/20260711T163000Z`；端到端 GDRDMA 为 `HARDWARE_BLOCKED`。

### 27. Multi-NIC、Rail、Cross-NIC 与 PXN

- 核心：NIC affinity、rail、crossNic、PXN intermediate rank、HCA filtering 和非对称拓扑。
- 源码：`graph/paths.cc` 的 PXN 逻辑、graph search 的 net selection。
- 实验：单/双 NIC、同/跨 NUMA、HCA mask、`CROSS_NIC`、`PXN_DISABLE`。
- 验收：能设计 GPU/NIC 亲和性并发现单 rail、错绑 NUMA或非对称瓶颈。
- 状态：单 NIC 负对照、HCA filter 与双 NIC/PXN 源码模型 `VALIDATED`，正式运行 `ch27_multinic_pxn/20260711T170000Z`；真实 multi-rail/PXN 性能为 `HARDWARE_BLOCKED`。

### 28. CollNet、SHARP、NVLS 与插件

- 核心：网络聚合、NVSwitch multicast/reduction、NVLS Tree、net/tuner plugin ABI 和 fallback。
- 源码：`transport/coll_net.cc`、`transport/nvls.cc`、`include/nccl_net.h`、`include/nccl_tuner.h`。
- 实验：插件加载/拒绝/fallback；在支持硬件上对照 Ring/Tree/CollNet/NVLS。
- 验收：明确能力发现、启用条件、实际选择和失败回退四个阶段。
- 状态：插件 ABI/加载/拒绝/回退、Socket 负对照、强制算法回退与源码门控模型 `VALIDATED`，正式运行 `ch28_offload_plugins/20260711T173000Z`；CollNet/SHARP 与 NVLS 端到端性能为 `HARDWARE_BLOCKED`。

## 卷七：训练框架集成

### 29. ProcessGroupNCCL、Stream、Event 与 WorkNCCL

- 核心：communicator cache、NCCL stream、event record/wait、WorkNCCL future、watchdog 和 timeout。
- 源码：当前 PyTorch commit 的 `ProcessGroupNCCL.cpp`。
- 实验：多 process group、多 stream、async_op、blocking wait 和 communicator cache。
- 验收：从 `torch.distributed.all_reduce` 追到 NCCL API，并解释完成语义差异。
- 状态：communicator cache、专用 stream/event、Work/Future 完成边界、blocking wait 与 watchdog 源码模型 `VALIDATED`，正式运行 `ch29_process_group_nccl/20260711T183000Z`。

### 30. DDP Reducer、Bucket 与计算通信重叠

- 核心：autograd hook、variable index、bucket rebuild、ready order、AllReduce hook 和 optimizer 边界。
- 源码：PyTorch `Reducer.cpp`、DDP Python wrapper 和 ProcessGroupNCCL。
- 实验：参数粒度、bucket cap、层计算时长、static graph、unused parameter；采集 overlap timeline。
- 验收：区分通信慢、bucket 调度差、计算尾部和同步等待。
- 状态：autograd hook、bucket rebuild/ready order、comm hook、unused/static graph、optimizer stream 边界与 Nsight overlap `VALIDATED`，正式运行 `ch30_ddp_reducer_overlap/20260711T190000Z`；旧 bucket 实验保留为基线。

### 31. ZeRO-0/1/2/3 与 FSDP

- 核心：参数、梯度、optimizer state 的复制/分片；AllGather、ReduceScatter、prefetch、reshard 和显存峰值。
- 源码：PyTorch FSDP 与 DeepSpeed ZeRO 对应版本源码。
- 实验：相同模型/optimizer 比较 DDP、ZeRO/FSDP 的显存、collective 序列、payload 和 step 时间。
- 验收：能独立推导每个 stage 的每 rank 状态量和通信量。
- 状态：ZeRO 状态公式、DDP/ZRO/FSDP 四级状态字节、collective 指纹、prefetch/reshard 与数值等价 `VALIDATED`，正式运行 `ch31_zero_fsdp_states/20260711T200000Z`；DeepSpeed v0.14.4 仅固定源码，runtime 未安装；FSDP 精确性能仍为 `PERF_UNSTABLE`。

### 32. TP、PP、EP/MoE 与多维 Process Group

- 核心：TP 的 AR/AG/RS、PP 的 Send/Recv、EP 的 AllToAll、DP 组合和 group mapping。
- 源码：Megatron-LM/DeepSpeed/PyTorch parallel APIs 的固定版本入口。
- 实验：最小可训练 TP/PP/EP 模型，而非仅合成 collective；输出每层消息大小和 timeline。
- 验收：从模型 shape 和并行拓扑预测 collective 顺序、payload 与热点。
- 状态：二维 rank/group 映射、可训练 TP/SP/PP/EP、变长 AllToAll、上游 activation gradient、DP shard 同步、payload/timeline 与 Nsight 指纹 `VALIDATED`，正式运行 `ch32_multidim_parallel/20260711T213000Z`；Megatron-LM core_v0.8.0 与 DeepSpeed v0.14.4 仅固定源码，精确微基准性能为 `PERF_UNSTABLE`。

## 卷八：故障、可观测性与生产工程

### 33. Collective Mismatch、Hang、Rank Crash 与网络故障

- 核心：调用分歧、数据形状分歧、lazy init、watchdog、async error、abort 和 teardown 连锁日志。
- 源码：NCCL group/error lifecycle 与 PyTorch watchdog。
- 实验：skip/op/shape/order/exit/timeout/网络中断矩阵，记录第一异常和后续错误。
- 验收：从第一现场区分应用逻辑、初始化、transport 和网络根因。
- 状态：op/shape/order fingerprint、raw partial-return、steady Work timeout、lazy-init外层deadline、rank exit、应用异常、error handling与abort/destroy因果链 `VALIDATED`，正式运行 `ch33_fault_root_cause/20260711T223000Z`；真实网络中断因单节点无HCA为 `BLOCKED`。

### 34. NCCL Debug、NVTX、Nsight 与 Flight Recorder

- 核心：debug subsystem、TRACE/COLL/TUNING/GRAPH/NET 日志，NVTX range，GPU kernel，CPU proxy 和 PyTorch trace buffer。
- 源码：`debug.cc`、`init_nvtx.cc`、PyTorch ProcessGroupNCCL tracing。
- 实验：同一操作生成四种观测证据并对齐时间戳；识别 profiler 名称误导的反例。
- 验收：不能仅凭 kernel 名称声称使用某协议，必须组合日志和源码。
- 状态：INIT/GRAPH/TUNING/COLL、NVTX投影、Nsight逐kernel与ProcessGroupNCCL Flight Recorder四层证据对齐 `VALIDATED`，正式运行 `ch34_observability/20260711T235500Z`；实测kernel符号`RING_LL`与大消息runtime proto2冲突，证明不能仅凭kernel名判协议。

### 35. 生产基线、回归判定与版本治理

- 核心：节点验收、拓扑 fingerprint、canary、基线阈值、配置审计、版本 A/B、升级和回滚。
- 源码：固定 runtime tag，并建立 master 版本演进 diff，而非混用实现。
- 实验：自动运行 correctness/performance/path/fault 套件，注入已知回归检验报警规则。
- 验收：能判断变化来自软件版本、算法选择、拓扑、链路还是统计噪声。
- 状态：节点/版本fingerprint、correctness/path/performance gate、channel与P2P真实回归、噪声INCONCLUSIVE、分阶段发布与回滚policy `VALIDATED`，正式运行 `ch35_governance/20260712T001500Z`；第二套NCCL runtime不可用，真实升级A/B未执行。

### 36. Capstone：一条 AllReduce 的端到端证明

- 核心：从 DDP autograd hook 追到 ProcessGroupNCCL、NCCL API、task、tuning、plan、device primitive、transport 和物理链路。
- 源码：串联本课程所有固定版本入口，形成可导航调用图和关键结构体生命周期图。
- 实验：运行前预测 algorithm/protocol/channel/transport；运行后用日志、profile、计数和性能模型逐项证明；再诊断一个未知回归。
- 验收：提交源码注释、实验包、结果重算、故障根因和生产修复方案。
- 状态：64 MiB DDP单bucket从autograd/Reducer到PGNCCL、NCCL tuning/plan/kernel、P2P/NVLink与optimizer闭环 `VALIDATED`，正式运行 `ch36_capstone/20260712T003000Z`；8/8预测、35/35调用图、4/4设备kernel和1-channel回归根因全部通过。

## 推荐顺序与阶段验收

1. 先完成 01-11，建立不会误测、不会误读 collective 的基础。
2. 再完成 12-24，真正理解 NCCL 算法和实现，而不是停留在环境变量层面。
3. 有 RDMA/NVSwitch 环境后完成 25-28，补齐多机数据面。
4. 完成 29-32，把 NCCL 机制映射回真实训练框架。
5. 最后完成 33-36，达到生产排障、版本治理和端到端源码证明能力。
