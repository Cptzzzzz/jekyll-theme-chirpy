---
title: 'NCCL 面试题详解：54 题从集合通信到大模型推理与生产排障'
author: Cptz
date: 2026-07-15 09:00:00 +0800
categories: [AI Infra, nccl]
tags: [nccl, interview, gpu, tensor parallel, vllm, sglang, troubleshooting]
math: true
mermaid: true
---

## 这份题库如何使用

这不是一份背诵名词的题库。面试官真正要判断的是候选人能否在下面四层之间建立闭环：

```text
数学语义 -> 并行策略 -> NCCL 实现 -> 生产观测与故障证据
```

完成本篇知识面筛查后，可以继续阅读[20 道 NCCL 进阶面试实验题](/posts/nccl-interview-deep-dive-labs/)；后者会把完成语义、Ring/device primitive、TP Decode 小消息、FP16 数值反例、transport fallback 和 hang 根因落实到源码与实测结果。

全文共 54 题，每题包含：

- **核心回答**：面试时首先应给出的结论。
- **深入展开**：公式、数据布局、执行链和边界条件。
- **涉及知识点**：这道题连接到的知识图谱。
- **追问**：面试官可以继续验证候选人是否真正理解。
- **追问回答**：对二次追问给出结论、推理与工程边界。
- **评分要点**：满分 10 分，给出可操作的分档标准。

本文以课程实验环境的 **NCCL 2.22.3** 为源码基线。NCCL 新版文档已经增加了新的 collective、注册方式和容错能力；凡涉及版本差异的地方都会单独说明，不能用 2026 年的新 API 解释 2.22.3 的运行结果。

通用评分含义：

| 分数 | 能力表现 |
|---:|---|
| 0-2 | 只会说名词，或者存在根本性语义错误 |
| 3-5 | API 语义正确，但解释不了数据流和性能 |
| 6-8 | 能结合算法、拓扑、框架和排障证据 |
| 9-10 | 能推导公式，指出版本边界，并设计可证伪实验 |

---

## Level 1：基础概念

### 1. NCCL 解决什么问题？不解决什么问题？

**核心回答**

NCCL 是面向 NVIDIA GPU 的集合通信和点对点通信库。它接收 GPU buffer、元素数量、数据类型、归约操作、communicator 和 CUDA stream，在单机或多机拓扑上完成 AllReduce、AllGather、ReduceScatter、Broadcast、Reduce 与 Send/Recv 等通信。

NCCL 不负责定义数据并行或张量并行，不负责启动进程、调度请求、切分模型、构建计算图，也不负责在 rank 失败后自动恢复整个训练或推理任务。这些策略由 PyTorch、Megatron、vLLM、SGLang 或上层控制器决定。

**深入展开**

可以把职责边界表示为：

```text
应用：决定为什么通信，例如 TP 合并局部 GEMM 结果
框架：决定谁和谁通信、tensor shape、调用顺序
NCCL：决定如何在给定 communicator 和拓扑上搬运及归约
硬件：NVLink / PCIe / NIC / InfiniBand / Ethernet 执行传输
```

NCCL 是 topology-aware 的数据面，但“感知拓扑”不等于“理解模型”。它知道 GPU 之间是否有 NVLink、GPU 到 NIC 的 PCIe 距离，却不知道某个 communicator 代表 TP 组还是 DP 组。若上层把跨节点 rank 错误地放进延迟敏感的 TP 组，NCCL只能尽量优化这组既定通信，不能替应用改并行策略。

**涉及知识点**：collective、communicator、CUDA stream、拓扑感知、控制面与数据面、框架职责边界。

**追问**：NCCL 能否替代 MPI？为什么一个系统可能同时使用 NCCL、Gloo 和 TCPStore？

**追问回答**

不能把 NCCL 当成 MPI 的完整替代品。MPI 除了 collective，还定义进程间点对点消息、tag、communicator 管理以及更广泛的 CPU/异构通信语义；NCCL 的核心边界是 CUDA buffer 上的 GPU collective 和 Send/Recv。纯 GPU 数据面可以用 NCCL 取代一部分 MPI 通信，但进程启动、服务发现、CPU object 传输和任务控制仍需要其他组件。

一个 PyTorch 系统同时使用三者并不重复：TCPStore 保存 rendezvous 信息并让进程找到彼此；Gloo 处理 CPU tensor、小型控制消息，或者广播 `ncclUniqueId`；NCCL 在 communicator 建立后传输梯度、激活等 GPU payload。判断某段流量属于哪个 backend，应看 buffer device、ProcessGroup 和具体调用，而不能只看进程初始化时加载过哪些库。

**评分要点（10 分）**：说清 GPU collective 得 2 分；指出单机和多机得 1 分；区分框架并行策略与 NCCL 执行得 3 分；说明 topology-aware 的边界得 2 分；能给出训练或推理实例再得 2 分。

### 2. NCCL、CUDA、NVLink、PCIe、RDMA、InfiniBand 分别处于什么层次？

**核心回答**

CUDA 提供 GPU 编程、内存和 stream 执行模型；NCCL 是建立在 CUDA 之上的通信库；NVLink 与 PCIe 是节点内互连；InfiniBand 或以太网是节点间网络；RDMA 是网卡直接访问远端注册内存的传输机制，而不是某一种物理网络的同义词。

**深入展开**

一条跨节点 AllReduce 可以经过如下层次：

```text
torch.distributed / PyNccl
  -> NCCL collective、算法和协议
  -> NCCL NET plugin / CPU proxy
  -> libibverbs 或 Socket
  -> IB / RoCE / Ethernet fabric
  -> 对端 NIC
  -> GPU memory 或 host staging memory
```

GPUDirect RDMA 表示 NIC 可以直接 DMA 到 GPU 显存，避免 payload 必须先落到 CPU 内存。它仍然需要 CPU 完成连接建立、队列管理和控制路径，也不表示数据绕过 PCIe，因为 NIC 和 GPU 通常仍通过 PCIe fabric 连接。

NVLink 是链路，NVSwitch 是交换结构，NCCL 则在这些链路上构造 Ring、Tree 或其他通信图。把“使用 NCCL”等价成“必然使用 NVLink”是错误的；如果 P2P 条件不满足，NCCL 可能退化到 SHM，跨节点还可能使用 Socket。

**涉及知识点**：CUDA、GPU Direct、PCIe BAR、DMA、IB Verbs、RoCE、NVLink、NVSwitch、NET plugin。

**追问**：RoCE 也能做 RDMA，为什么生产排障时仍要区分 InfiniBand 与 RoCE？

**追问回答**

RDMA 描述的是远端内存访问能力，不规定底层 fabric。原生 InfiniBand 使用 IB 链路层、Subnet Manager、LID/credit-based flow control；RoCE 把 RDMA verbs 映射到以太网，生产环境还涉及 GID、VLAN、路由、MTU，以及 PFC/ECN 等拥塞与无损配置。两者可能暴露相似的 verbs API，却有不同的控制面、计数器和故障模式。

因此排障证据也不同：IB 需要检查端口状态、LID、SM、VL 和 fabric counter；RoCE 需要同时检查 GID index、IP 路由、交换机 PFC/ECN、优先级映射与丢包/暂停帧。`ib_write_bw` 或 NCCL 日志里出现 `NET/IB` 只说明使用了 verbs 类插件，不能据此把 RoCE 当成原生 IB，也不能证明拥塞配置正确。

**评分要点（10 分）**：正确分层得 4 分；说明 RDMA 不是网络类型得 2 分；解释 GDR 仍有控制路径得 2 分；指出 NCCL 会按条件选择 transport 得 2 分。

### 3. 什么是 rank、world size 和 communicator？

**核心回答**

rank 是某个 communicator 内的逻辑编号，取值范围为 $[0,N)$；world size 是该 communicator 的参与者数量 $N$；communicator 是一组参与者及其通信状态、拓扑、连接和全局操作顺序的上下文。

**深入展开**

rank 不是 GPU 编号，也不天然等于进程编号。最简单的一进程一卡程序中，global rank、local rank、CUDA device ordinal 和 NCCL rank 可能恰好相同，但这只是部署映射的结果。

同一个进程可以同时属于多个 communicator。例如 16 张 GPU 配置 `DP=2, TP=8`：

```text
TP group 0: global rank [0,1,2,3,4,5,6,7]
TP group 1: global rank [8,9,10,11,12,13,14,15]
DP group 0: global rank [0,8]
DP group 1: global rank [1,9]
...
```

global rank 8 在 world communicator 中是 rank 8，在第二个 TP communicator 中可能是 rank 0，在某个 DP communicator 中可能是 rank 1。collective 的匹配、顺序和 root 都以具体 communicator 的 rank 空间为准。

**涉及知识点**：global/local/group rank、ProcessGroup、NCCL communicator、多维并行组、rank-device 映射。

**追问**：成员集合相同但分别创建的两个 communicator，能否自动共享 collective 顺序？

**追问回答**

不能。两个 communicator 即使包含完全相同的进程和 GPU，也有各自的 unique ID、连接、拓扑计划、`opCount` 和 collective 顺序域。某个 communicator 上的第 10 次 AllReduce，只与其他 rank 在同一 communicator 上的第 10 次匹配，不会与另一个 communicator 的操作自动配对或排序。

这也意味着两个 communicator 可以并发执行，但应用必须显式处理跨 communicator 的依赖。若同一批 rank 在不同主机线程或 CUDA stream 上以相反顺序启动两个 communicator 的工作，仍可能形成循环等待；成员相同并不会让 NCCL 替应用推导一个全局顺序。

**评分要点（10 分）**：定义三者得 3 分；指出 rank 不等于 GPU id 得 2 分；能画出 TP/DP 子组得 3 分；说明每个 communicator 有独立顺序域得 2 分。

### 4. 分别解释 Broadcast、Reduce、AllReduce、AllGather、ReduceScatter 和 Send/Recv。

**核心回答**

设有 $N$ 个 rank，每个 rank 的输入为 $x^{(r)}$，每块有 $q$ 个元素：

| 操作 | 输出语义 | 典型输出元素数 |
|---|---|---:|
| Broadcast | root 的输入复制到所有 rank | 每 rank $q$ |
| Reduce | 所有输入归约到 root | root 为 $q$ |
| AllReduce | 所有输入归约，结果复制到所有 rank | 每 rank $q$ |
| AllGather | 按 rank 顺序收集所有输入分块 | 每 rank $Nq$ |
| ReduceScatter | 先逐元素归约总输入，再把结果分片 | 每 rank $q$，每 rank 输入通常为 $Nq$ |
| Send/Recv | 指定 peer 之间传输一段数据 | 由双方 count 决定 |

AllReduce 的逐元素语义为：

$$
y_j^{(r)}=\bigoplus_{k=0}^{N-1}x_j^{(k)}
$$

AllGather 的 rank-major 布局为：

$$
y^{(r)}=[x^{(0)},x^{(1)},\ldots,x^{(N-1)}]
$$

ReduceScatter 中每个 rank 通常提供 $Nq$ 个元素。完整归约结果也有 $Nq$ 个元素，rank $r$ 只保留区间 $[rq,(r+1)q)$。

**深入展开**

必须区分“数学语义”和“实现算法”。AllReduce 在语义上可以看成 Reduce 后再 Broadcast，但高性能实现不会先把全部数据集中到一个 root。Ring 通常将其实现为 ReduceScatter 加 AllGather，Tree 则采用层次归约和分发。

还要注意 in-place 契约不是统一的 `sendbuff == recvbuff`。例如 AllGather 的本地输入需要位于接收 buffer 中属于本 rank 的 offset；ReduceScatter 也有相应布局要求。Reduce 的非 root 输出没有有效语义，不能把旧 buffer 内容当成归约结果。

**涉及知识点**：collective 数学契约、count、rank-major 布局、root、in-place、归约操作。

**追问**：为什么 `AllReduce = Reduce + Broadcast` 在语义上成立，但性能分析不能直接按中心化实现计算？

**追问回答**

等式只说明最终结果相同：先把所有输入归约到 root，再把结果复制给所有 rank，确实满足 AllReduce 的数学契约。它没有规定数据必须经过 root，也没有规定使用两个独立 API。

高性能实现会把流量和计算分散到所有 rank。Ring 将 AllReduce 实现为流水化的 ReduceScatter 加 AllGather，每个 rank 发送约 $2(N-1)M/N$；Tree 则沿多级父子关系归约和广播。若按中心化 Reduce+Broadcast 估算，会虚构 root 的带宽热点、错误的并行度和链路利用率。性能模型必须使用实际选择的算法、协议、channel 和物理拓扑，而不是只使用语义等价式。

**评分要点（10 分）**：六种操作全部正确得 4 分；shape/count 正确得 2 分；能写 AllReduce 和 ReduceScatter 公式得 2 分；指出 in-place 与非 root 边界得 2 分。

### 5. `AllReduce(SUM)` 后每个 rank 得到什么？输入输出 shape 是否变化？

**核心回答**

每个 rank 都得到所有 rank 对应位置元素之和，输入和输出的逻辑元素数相同。若 4 个 rank 分别输入全部为 1、2、3、4 的长度 $q$ tensor，则每个 rank 的输出都是长度 $q$、元素值为 10 的 tensor。

**深入展开**

AllReduce 不会沿 tensor 某个维度自动拼接，也不会改变 shape。`count=q` 表示每个 rank 对 $q$ 个同类型元素参与逐元素归约。拼接分片是 AllGather 的语义。

浮点 SUM 还存在归约顺序问题。加法在实数上满足结合律，但 IEEE 浮点加法不严格满足结合律。不同算法、channel 切分或拓扑可能改变累加顺序，因此结果通常在误差容限内正确，却未必逐 bit 相同。需要 bitwise determinism 的系统不能只检查“都使用 NCCL”，还要固定算法、拓扑、版本和上层执行顺序，并验证实际可重复性。

**涉及知识点**：逐元素归约、count、shape、浮点结合律、数值可重复性。

**追问**：AllReduce 的输入为 FP16，但累加精度由谁决定？能否假设一定使用 FP32 累加？

**追问回答**

不能假设。NCCL API 传入的是输入/输出 datatype 和 reduction op，并没有单独的“累加 dtype”参数；实际归约运算由该 NCCL 版本、GPU 指令和对应 datatype 的设备端实现决定。上层框架也可能先把 tensor 转成 FP32 再调用 NCCL，但那时 NCCL 看到的 datatype 已经是 FP32，而不是“FP16 输入、隐式 FP32 累加”。

工程上应把累加精度视为需要验证的契约：检查调用点传给 NCCL 的 datatype，阅读该版本 reduction primitive，并用大动态范围、易产生舍入误差的输入与 FP32 reference 比较。只看到模型参数是 FP16/BF16，既不能推出 wire datatype，也不能推出内部一定采用 FP32 accumulation。

**评分要点（10 分）**：结果和 shape 得 4 分；区分 AllGather 得 2 分；指出 dtype 与 reduction op 得 1 分；解释浮点顺序和非 bitwise deterministic 得 3 分。

### 6. NCCL 操作是同步还是异步的？函数返回是否意味着 GPU 通信完成？

**核心回答**

常规 NCCL collective 是相对于 CPU 异步的 CUDA stream 操作。API 成功返回通常表示工作已经被提交或排入给定 stream，不表示 GPU 通信已经完成。完成性需要通过 stream 同步、CUDA Event 或上层框架的 Work/Event 语义确认。

**深入展开**

若在 stream 上依次提交：

```text
GEMM -> ncclAllReduce -> next kernel
```

同一 stream 的顺序保证 next kernel 看到 AllReduce 完成后的数据。但 CPU 在 `ncclAllReduce` 返回后立即读取 host 计时器，测到的主要是 enqueue 时间，而不是通信时间。

PyTorch ProcessGroupNCCL 通常使用专用通信 stream。它会记录 event，使 NCCL stream 等待生产输入的 application stream；通信完成后再通过 Work/Event 给后续消费者建立依赖。`async_op=True` 返回 Work 也不代表 GPU 已完成。`wait()` 的 host 阻塞行为还受 ProcessGroup 配置影响，应区分：

```text
host API return
work submitted
CUDA stream dependency established
GPU kernel completed
result safe for another stream/CPU 使用
```

异步错误同样可能在稍后的查询、同步或 watchdog 中才暴露，因此错误栈不一定指向最初的错误 collective。

**涉及知识点**：CUDA stream、Event、host/GPU completion、ProcessGroupNCCL Work、异步错误、正确计时。

**追问**：两个不同 CUDA stream 访问同一个通信 buffer 时，需要怎样建立依赖？

**追问回答**

生产输入的 stream 应在写完 buffer 后记录 CUDA Event，NCCL 所在 stream 先 `cudaStreamWaitEvent` 再启动 collective；通信完成后在 NCCL stream 记录另一个 Event，消费输出的 stream 等待该 Event 后才能读或覆盖 buffer。若始终使用同一 stream，stream 内顺序天然提供这些依赖。

只靠 host 上先后调用两个异步 API 不足以建立跨 stream 的 GPU happens-before。allocator 也必须知道 buffer 仍被通信 stream 使用，例如通过框架的 `record_stream`/Work 生命周期避免显存提前复用。多个 stream 对同一地址并发写，或者在 collective 完成前释放、重用 buffer，都是数据竞争。

**评分要点（10 分）**：指出异步 enqueue 得 3 分；能区分 host return 与 GPU completion 得 2 分；说明 stream/event 顺序得 2 分；联系 PyTorch Work 和正确计时得 3 分。

### 7. PyTorch 调用 `torch.distributed.all_reduce` 时，什么情况下底层使用 NCCL？

**核心回答**

当 tensor 位于 CUDA device，目标 ProcessGroup 的 CUDA backend 是 NCCL 时，`torch.distributed.all_reduce` 会进入 ProcessGroupNCCL，并最终调用 NCCL collective。CPU tensor 通常使用 Gloo 等 CPU backend，不能因为函数名相同就假设底层都是 NCCL。

**深入展开**

典型调用链是：

```text
torch.distributed.all_reduce(tensor, group)
  -> c10d dispatcher
  -> ProcessGroupNCCL::allreduce
  -> communicator cache / NCCL stream / Work
  -> ncclAllReduce(buffer, count, dtype, op, comm, stream)
```

需要同时核对 tensor device 和 group backend。现代 PyTorch 可以为 CPU、CUDA 等设备配置不同 backend，因此仅看到 `init_process_group()` 不足以判断某次操作的实际路径。

在 vLLM、SGLang 等框架中，模型语义上的 `all_reduce` 还可能先被自定义通信层截获，改走 Custom AllReduce、FlashInfer 或 MSCCL++。这时即使框架初始化过 NCCL communicator，该次操作也未必调用 `ncclAllReduce`。

**涉及知识点**：c10d、ProcessGroupNCCL、backend 与 device、communicator cache、框架自定义 collective。

**追问**：如何用 Nsight Systems 和 NCCL 日志证明某一次模型 AllReduce 真的进入了 NCCL？

**追问回答**

先给目标算子加 NVTX range，并用 Nsight Systems 同时采集 CUDA API、CUDA kernel 和 NVTX。时间线上应能从该 range 关联到 `ncclAllReduce` 的 host 调用或 ProcessGroupNCCL enqueue，再关联到 NCCL 设备 kernel；若看到的是框架自定义 kernel 名而没有 NCCL API 调用，就不能声称该次操作进入了 NCCL。

同时启用该版本支持的 NCCL `INFO/TRACE` 与 `COLL` 日志，核对 communicator、rank、count、datatype、stream/opCount 和时间窗口。最强的证据是三方一致：模型层 tensor 大小与 dtype、日志中的 collective 元数据、Nsight 中相同时间段的 NCCL kernel。还应做一次可控 A/B，例如禁用 Custom AllReduce；只凭“进程加载了 `libnccl.so`”或只看到 NCCL 初始化日志都不够。

**评分要点（10 分）**：CUDA tensor 与 NCCL backend 两个条件得 4 分；写出 ProcessGroupNCCL 链路得 3 分；指出 CPU/Gloo 边界得 1 分；能识别自定义 AR 绕过 NCCL 得 2 分。

### 8. NCCL 和 Gloo 有什么区别？为什么 GPU 通信通常选择 NCCL，而控制面常用 Gloo/TCP？

**核心回答**

NCCL 针对 GPU buffer、CUDA stream 和 GPU 拓扑优化，能够使用 NVLink、PCIe P2P、共享内存和 RDMA 网络。Gloo 更偏向 CPU tensor 和通用网络通信。GPU payload 用 NCCL 可以避免不必要的 host staging，并利用 GPU kernel 完成归约；小型 Python object、状态同步和 bootstrap 元数据则更适合 Gloo、TCPStore 或普通 Socket。

**深入展开**

大模型系统经常同时存在两类流量：

```text
控制面：rank 地址、请求元数据、心跳、调度状态、ncclUniqueId
数据面：激活、梯度、参数、KV block、MoE token
```

将控制面和数据面分离有两个好处。第一，CPU object 不需要为了通信临时搬到 GPU；第二，当 NCCL communicator 正在执行长 collective 或发生故障时，独立控制通道仍可能用于上报状态和触发 teardown。

但不能简单说“小消息都应该走 Gloo”。如果小消息位于 GPU 且后续 GPU kernel 立即消费，转到 CPU 会引入同步和拷贝，NCCL 或 GPU 自定义通信往往更合适。选型取决于 buffer 所在位置、消费者、延迟要求和故障域。

**涉及知识点**：GPU/CPU backend、控制面、数据面、host staging、bootstrap、故障域隔离。

**追问**：为什么使用 Gloo 广播 `ncclUniqueId` 并不意味着后续 AllReduce 经过 Gloo？

**追问回答**

`ncclUniqueId` 只是创建 NCCL communicator 所需的小型 bootstrap token。Gloo 在这里承担控制面广播：让所有 rank 拿到同一个 ID；随后每个 rank 调用 `ncclCommInitRank`，由 NCCL 建立自己的 peer 连接、拓扑图和数据通道。

后续 GPU AllReduce 调用的是该 NCCL communicator，payload 走 NCCL 选择的 P2P、SHM、IB/RoCE 或 Socket transport，不会因为 ID 曾由 Gloo 搬运就经过 Gloo。判断数据路径要看 collective 的 ProcessGroup/backend、NCCL transport 日志和运行时 trace，而不是 bootstrap 使用了什么工具。

**评分要点（10 分）**：GPU 与 CPU 定位得 3 分；解释拓扑和 stream 优势得 2 分；区分控制面与数据面得 3 分；指出不能按消息大小机械选型得 2 分。

---

## Level 2：算法与并行策略

### 9. Ring AllReduce 包含哪两个阶段？

**核心回答**

经典 Ring AllReduce 把每个 rank 的长度 $M$ 数据均分为 $N$ 个 chunk，先执行 $N-1$ 步 ReduceScatter，使每个 rank 得到一个完整归约后的 chunk；再执行 $N-1$ 步 AllGather，把这些 chunk 传播到所有 rank。

**深入展开**

ReduceScatter 阶段中，每一步都同时完成发送、接收和局部 reduction。某个 chunk 绕 Ring 前进时不断吸收沿途 rank 的对应输入，经过 $N-1$ 次传递后聚合全部 $N$ 个贡献。AllGather 阶段不再做 reduction，只复制已经归约完成的 chunk。

因此：

$$
\operatorname{AllReduce}(x)
=\operatorname{AllGather}(\operatorname{ReduceScatter}(x))
$$

这个等价需要相同的分块布局、归约操作和 rank 顺序。它是数学与数据布局等价，不表示一次 NCCL AllReduce 一定启动两个独立 API 或两个 kernel。NCCL 可以在一个设备执行计划中流水化两个阶段。

**涉及知识点**：Ring、chunk owner、ReduceScatter、AllGather、流水线、collective 分解。

**追问**：既然 AllReduce 可以分解，为什么框架有时直接保留 ReduceScatter 的分片结果？

**追问回答**

因为下一步计算未必需要每个 rank 都持有完整结果。若优化器状态、参数、梯度或激活本来就是分片布局，ReduceScatter 已经同时完成“归约”和“分片”；再执行 AllGather 只会增加约 $(N-1)M/N$ 的单 rank 流量，并产生一份本来不需要的完整副本。

ZeRO/FSDP 的梯度分片和某些 sequence-parallel 路径都利用这一点。正确性条件是下游算子明确知道每个 rank 持有哪些区间，并在真正需要完整 tensor 的边界才 AllGather。保留分片不是少做了 AllReduce 的一半，而是把程序的合法数据布局从 replicated 改成 sharded，从而消除不必要的复制阶段。

**评分要点（10 分）**：两个阶段得 2 分；各 $N-1$ 步得 2 分；解释 chunk 如何归约和传播得 3 分；指出 API 分解不等于实现分裂得 3 分。

### 10. 用 4 个 rank 描述 Ring AllReduce 中数据块如何移动和累加。

**核心回答**

把每个 rank 的输入分为 $c_0,c_1,c_2,c_3$，并设通信方向为 $0\rightarrow1\rightarrow2\rightarrow3\rightarrow0$。ReduceScatter 的一种合法调度如下：

```text
step 0: rank r 发送自己的 c_r，接收前驱的 chunk，并加上本地对应 chunk
step 1: 发送上一步得到的 2-rank partial，接收并形成 3-rank partial
step 2: 发送 3-rank partial，接收并形成 4-rank complete chunk
```

以 rank 0 为例：

```text
step 0: 从 rank 3 收到 c3 的贡献，形成 rank{3,0} 对 c3 的 partial
step 1: 把 c3 partial 发给 rank 1，同时收到 c2 partial 并加入本地 c2
step 2: 把 c2 partial 发给 rank 1，同时收到 c1 的 3-rank partial，加入本地 c1
结果：rank 0 持有完整归约后的 c1
```

chunk 最终由哪个 rank 持有取决于索引约定，不影响算法。随后 AllGather 再进行 3 步：每个 rank 把自己已有的 complete chunk 发给后继，并逐步收齐其余三个 complete chunk。

**深入展开**

关键不是背某张固定表，而是能验证两个不变量：

1. ReduceScatter 每一步后，正在传递的 partial 多包含一个 rank 的贡献。
2. 每个 complete chunk 在 AllGather 中只被复制，不再归约，最终每个 rank 恰好拥有全部四块。

真实 NCCL 还会把每个 chunk 继续切成 slice/step，由多个 channel 并行流水；逻辑 Ring 的 step 不等于 timeline 上只能看到一个串行 memcpy。

**涉及知识点**：Ring 邻居、chunk index、partial reduction、step、slice、channel。

**追问**：如果 rank 2 的链路速度只有其他链路的一半，单 Ring 的稳态吞吐会怎样？

**追问回答**

单 Ring 的每个 chunk 都必须依次通过 Ring 上的每条边，流水线周期由最慢边决定。rank 2 相邻方向上的有效链路带宽减半后，其他边即使更快也会等待这个 stage，理想稳态吞吐上限大致降到原来的一半；慢边前后还会出现队列积压和空泡。

真实 NCCL 可能用多个 channel、多个 Ring 或不同方向绕开/分摊某条物理链路，所以最终下降未必精确为 50%。验证时应固定为单 Ring/单 channel 做基线，再看拓扑图和链路计数器；否则“rank 2 慢”可能被多路径调度掩盖，或者同时拖慢多个逻辑 Ring。

**评分要点（10 分）**：能描述 3 步 RS 和 3 步 AG 得 4 分；partial 每步增加贡献得 2 分；最终每 rank 完整得 2 分；指出 channel 流水与逻辑 step 的区别得 2 分。

### 11. Ring AllReduce 每个 rank 的通信量如何随 world size 变化？

**核心回答**

设每个 rank 的输入大小为 $M$ 字节。Ring 每一步发送约 $M/N$ 字节，ReduceScatter 和 AllGather 各有 $N-1$ 步，因此每个 rank 的发送量为：

$$
V_{send}=2\frac{N-1}{N}M
$$

接收量相同。随着 $N$ 增大，单 rank 发送量趋近 $2M$，但串行依赖 step 数增长到 $2(N-1)$。

**深入展开**

理想单向 Ring 模型可写成：

$$
T_{ring}\approx2(N-1)\alpha
+2\frac{N-1}{N}\frac{M}{B}
$$

$\alpha$ 表示每一步不可隐藏的固定延迟，$B$ 表示稳态链路带宽。这个式子解释了 Ring 的两面性：大消息时，通信量接近理论下界并能充分流水；小消息时，$2(N-1)$ 个依赖 step 的固定成本会随着 rank 数增长。

这里的“发送量”和“接收量”应分别统计。若把双向字节简单相加，会得到两倍数值，容易和 `nccl-tests busbw` 的归一化定义混淆。busbw 是算法模型指标，不是 NIC 或某条 NVLink 的硬件计数器。

**涉及知识点**：通信复杂度、Ring 流量、延迟带宽模型、algbw、busbw。

**追问**：为什么“world size 越大，每 rank 流量趋近 2M”不代表扩到无限 rank 仍能保持同样时延？

**追问回答**

$2(N-1)M/N\rightarrow2M$ 只描述带宽项中的字节数。Ring 仍有 $2(N-1)$ 个逻辑 step，每个 step 都要付启动、同步、传播和协议开销，因而延迟项约为 $2(N-1)\alpha$，会随 rank 数线性增长。

扩到更多节点后还会遇到更长网络直径、更多交换跳数、链路共享、chunk 变小导致的协议效率下降，以及最慢 rank 的尾延迟。固定 $M$ 时，最终可能从带宽受限转为延迟受限。容量规划必须同时看字节数、step 深度和 topology contention，不能只用单 rank 总流量外推。

**评分要点（10 分）**：推导 $2(N-1)M/N$ 得 4 分；区分发送和接收得 1 分；写出包含 $\alpha$ 的时间模型得 3 分；解释大规模小消息退化得 2 分。

### 12. Ring 和 Tree AllReduce 分别适合什么消息规模？

**核心回答**

Ring 通常具有较高的大消息带宽利用率，但依赖步数为 $O(N)$；Tree 的层数约为 $O(\log N)$，通常更适合延迟敏感的小消息。这个结论只是起点，真实选择还取决于拓扑、协议、channel、节点数和 NCCL tuning model。

**深入展开**

Tree 将 reduction 沿叶子到根传播，再把结果从根向下分发。其关键路径较短，因此固定延迟项通常低于 Ring。NCCL 的双树设计可以让不同数据块在互补树上并行，减少单 root 的瓶颈。

Ring 的优势是每个 rank 可以持续在相邻链路上发送等大小 chunk，稳态阶段容易接近瓶颈链路带宽。Tree 在大消息时可能受父子链路并发、树节点度数或不均匀拓扑限制。

不能在生产中永远强制“Tree 小包、Ring 大包”。正确方法是扫描真实 message size，保存 `NCCL_DEBUG_SUBSYS=TUNING,GRAPH` 证据，并比较自动选择、强制候选和业务 workload。NVSwitch、跨节点层次化拓扑、CollNet/NVLS 等都会改变交叉点。

**涉及知识点**：Ring、Tree、双树、关键路径、tuning model、层次化 collective。

**追问**：为什么同样是 1 MiB，单机 8 卡和 64 节点各 8 卡的最优算法可能不同？

**追问回答**

消息大小不是算法选择的唯一输入。单机 8 卡可能处于低延迟、高带宽、近乎对称的 NVSwitch 域；64 节点则同时包含节点内和节点间层级，跨 NIC/交换机的延迟、带宽、oversubscription 和 GDR 路径成为主要约束。相同 1 MiB 在前者可能足以让 Ring 饱和，在后者却会被大量跨节点 step 的 $\alpha$ 项主导。

因此多节点常更偏好 Tree、分层算法或 CollNet/NVLS 类路径，用较小的跨节点深度换取更复杂的节点内阶段；实际选择还依赖 NCCL 版本和硬件支持。应按目标拓扑做 size sweep，并记录实际 algorithm/protocol，而不是把单机 crossover 阈值直接复制到集群。

**评分要点（10 分）**：Ring 带宽与 Tree 延迟得 3 分；复杂度得 2 分；解释双树或层次化得 2 分；拒绝固定阈值并提出实测方法得 3 分。

### 13. 用延迟带宽模型分析小消息和大消息通信。

**核心回答**

最简单模型为：

$$
T(M)=\alpha+\beta M
=\alpha+\frac{M}{B}
$$

小消息满足 $\alpha\gg M/B$，性能由 launch、同步、协议握手和依赖深度决定；大消息满足 $M/B\gg\alpha$，性能主要由链路、算法流量系数和稳态流水带宽决定。

**深入展开**

collective 需要加入算法相关系数。例如 Ring AllReduce：

$$
T(M,N)\approx2(N-1)\alpha_s
+2\frac{N-1}{N}\frac{M}{B_{eff}}
$$

Tree 则可粗略写成 $2\log_2N$ 层的固定开销，再加上各层传输项。真实曲线通常不是一条直线，因为 NCCL 会在不同消息区间切换 algorithm、protocol、channel 数和 chunk 大小，内存注册、CUDA Graph 与网络拥塞也会引入分段。

拟合时应对消息大小做对数扫描，至少分别拟合延迟区、过渡区和饱和区，并检查残差。全区间得到一个很高的 $R^2$ 也不代表模型正确，因为大消息点会主导误差。

**涉及知识点**：$\alpha$-$\beta$ 模型、分段拟合、算法交叉点、固定开销、稳态带宽。

**追问**：如何从实测曲线识别 protocol 或 algorithm 切换，而不是把跳变误判为随机噪声？

**追问回答**

先对每个 message size 做足够的 warmup 和多次重复，画中位数、P95/P99 与置信区间。若跳变在相同 size 附近可重复出现，并且同时伴随 NCCL `TUNING/GRAPH` 或 trace 中 algorithm、protocol、channel 数的变化，它才像稳定的选择边界；随机噪声通常不会严格锁定在同一个尺寸。

再用 `NCCL_ALGO`、`NCCL_PROTO` 等受该版本支持的开关分别强制候选路径做 A/B。若强制后曲线变为两条连续曲线，而自动模式恰好在两者交点切换，因果证据就比较完整。实验还要固定拓扑、时钟、并发流量与 buffer 注册状态，防止 DVFS、缓存和网络争用制造伪跳变。

**评分要点（10 分）**：模型和变量定义得 3 分；区分小包/大包得 2 分；加入 collective 系数得 2 分；提出分段拟合和残差验证得 3 分。

### 14. 为什么 Transformer TP 通常采用 `ColumnParallel -> RowParallel -> AllReduce`？

**核心回答**

这是对相邻两个不同 GEMM 使用互补切分。第一个权重按输出列切分，产生各 rank 独立的中间特征；第二个权重按对应输入行切分，每个 rank 计算最终输出的局部和，最后一次 AllReduce 得到完整输出。这样避免在两个 GEMM 中间 AllGather 完整激活。

**深入展开**

对两层 MLP：

$$
Y=\phi(XW_1)W_2
$$

令：

$$
W_1=[W_{1,0},\ldots,W_{1,p-1}],\qquad
W_2=\begin{bmatrix}W_{2,0}\\ \vdots \\ W_{2,p-1}\end{bmatrix}
$$

则：

$$
Y=\sum_{i=0}^{p-1}\phi(XW_{1,i})W_{2,i}
$$

rank $i$ 只需保存 $Z_i=\phi(XW_{1,i})$，计算 $Y_i=Z_iW_{2,i}$，最后对 $Y_i$ 求和。这里不是利用 $\phi(\sum Z_i)=\sum\phi(Z_i)$，而是 Column Parallel 直接算出了完整中间 tensor 的一个精确列分片，从未产生需要先求和的 $W_1$ partial。

Attention 也是同一结构：QKV Column Parallel 让不同 rank 持有不同 head，按 head 独立完成 Attention；O Projection Row Parallel 后再 AllReduce。

**涉及知识点**：1D Tensor Parallel、Column/Row Parallel、局部和、Megatron TP、Attention head 切分。

**追问**：为什么说“同一个矩阵先按列切再按行切”是不准确的？

**追问回答**

MLP 中按列切和按行切的是两个不同权重矩阵。若 $H=XW_1$，Column Parallel 把 $W_1$ 沿输出维切为 $W_1^{(r)}$，每个 rank 得到 $H^{(r)}$；随后 $Y=HW_2$，Row Parallel 把另一个矩阵 $W_2$ 沿输入维切为与 $H^{(r)}$ 对齐的 $W_2^{(r)}$，计算局部 partial $Y^{(r)}$。

最终有 $Y=\sum_r H^{(r)}W_2^{(r)}$，所以只需一次跨 rank reduction。说“同一个矩阵先按列切再按行切”会混淆两个 GEMM 的维度契约，也解释不了为什么中间激活分片能直接成为第二个 GEMM 的本地输入。

**评分要点（10 分）**：指出是相邻两个矩阵得 2 分；正确推导公式得 3 分；解释避免中间 AllGather 得 2 分；能映射到 Attention 和 MLP 得 3 分。

### 15. 为什么 MLP 中间的 SiLU/GELU 不需要通信？哪些算子会破坏分片独立性？

**核心回答**

SiLU、GELU、逐元素乘加等操作只读取同一元素或同一分片内的数据。Column Parallel 已经让每个 rank 获得中间输出的精确列分片，因此这些操作可在本地执行。若算子需要跨被切分维度做统计、归一化、排序或混合，就必须通信。

**深入展开**

适合留在分片状态中的典型操作有：

- GELU、SiLU、ReLU。
- SwiGLU 中对齐分片的 `silu(gate_i) * up_i`。
- 每个输出分片自己的 bias、scale、quantize/dequantize。
- 按 head 切分后的 RoPE 和每个 head 内 Attention。

会破坏独立性的操作包括：

- 在被切分 hidden 维度上做 RMSNorm/LayerNorm，因为需要全局平方和、均值或方差。
- 在被切分维度上做 Softmax，因为需要全局 max 和 sum。
- 全局 Top-K、排序或跨 shard 路由决策。
- 任意跨分片通道混合的稠密变换。

“element-wise 没有通信”不代表任意 fused kernel 都可以本地执行。必须逐项检查其 reduction axis、数据依赖和输出布局。

**涉及知识点**：分片可分性、reduction axis、SwiGLU、Norm、Softmax、分布式 Top-K。

**追问**：RMSNorm 放在第一个 Column Parallel GEMM 之前，为什么通常不引入 TP 通信？

**追问回答**

标准 Megatron 式 TP 在进入第一个 Column Parallel GEMM 前，每个 rank 对自己负责的 token 持有完整 hidden vector；RMSNorm 的均方统计只沿 hidden 维计算，所以每个 rank 能对本地 token 独立得到完整统计量，不需要 TP collective。若启用了 sequence parallel，各 rank 可以持有不同 token 区间，但每个 token 的 hidden 维仍然完整，结论不变。

只有当输入真的沿 hidden 维分片时，每个 rank 才只能算局部平方和，必须 AllReduce 全局 sum-of-squares 后才能得到精确 RMS。判断是否通信应看归一化维度是否被切分，而不是只看“模型启用了 TP”。

**评分要点（10 分）**：解释逐元素局部性得 3 分；列出可本地操作得 2 分；指出 Norm/Softmax 的跨分片统计得 3 分；能按 axis 分析而不是背名单得 2 分。

### 16. 为什么标准 Decoder 层通常有约两次 TP AllReduce，而不是整个层只有一次？

**核心回答**

Attention 子层的 O Projection 是一次 Row Parallel，需要一次 AllReduce；MLP 子层的 Down Projection 又是一次 Row Parallel，需要第二次 AllReduce。两者之间存在残差、Norm 和下一子层计算依赖，通常不能把两次归约合并到层末尾。

**深入展开**

典型执行顺序为：

```text
RMSNorm
QKV Column -> Attention -> O Row -> AllReduce
Residual Add
RMSNorm
Gate/Up Column -> Activation -> Down Row -> AllReduce
Residual Add
```

MLP 的输入依赖 Attention 全局求和后的输出。若跳过第一次 AllReduce，各 rank 持有的是不同的 O Projection partial，后续 RMSNorm 的统计值和 MLP 输入都会错误。因此两次 collective 存在真实的数据依赖。

框架可以把 AllReduce 与 residual、RMSNorm 或量化融合，也可以通过 Sequence Parallel 把 AllReduce 改写为 ReduceScatter 加后续 AllGather，但这些优化改变的是 launch、布局或重叠，不一定消除了逻辑通信量。

**涉及知识点**：Transformer 数据依赖、残差、Pre-Norm、collective fusion、Sequence Parallel。

**追问**：所谓 fused AllReduce + RMSNorm 消除的是 collective，还是 kernel launch 和中间访存？

**追问回答**

通常没有消除 AllReduce 的数学语义和网络字节。各 rank 的 Row Parallel partial 仍必须求和，融合实现只是把 reduction 的收尾、residual add、RMSNorm 统计或写回合到更少的 kernel/流水阶段中，减少 kernel launch、全局显存中间写回和同步间隙。

只有改变数据布局或下游算法，使完整归约结果不再需要，才算真正消除 collective。评估融合时要分别报告网络 payload、collective kernel 时长、launch gap 和 HBM 流量；端到端加速不能直接解释成“少了一次通信”。

**评分要点（10 分）**：指出 Attention/MLP 各一次得 3 分；解释中间依赖得 3 分；说明融合不等于通信消失得 2 分；能讨论 SP 变换得 2 分。

### 17. TP 的激活 AllReduce 和 DDP 的梯度 AllReduce 有什么本质区别？

**核心回答**

TP AllReduce 合并同一次前向中不同模型分片产生的局部激活，一个请求需要所有 TP rank 协同才能继续；DDP AllReduce 在反向中同步不同数据副本计算出的参数梯度，每个 DP rank 拥有完整模型并处理不同样本。

**深入展开**

| 维度 | TP 激活通信 | DDP 梯度通信 |
|---|---|---|
| 发生阶段 | 每层前向，训练时也可出现在反向 | 主要在 backward |
| 数据 | `[tokens, hidden]` 等激活 partial | 参数梯度 bucket |
| 频率 | 每层多次 | 每个 bucket 每轮一次 |
| 延迟敏感性 | Decode 极强，位于逐层关键路径 | 可与后续 backward 计算重叠 |
| rank 关系 | 多 rank 共同完成一个模型副本 | 每 rank 是完整模型副本 |
| 推理时 | 常见 | 无梯度时通常不存在 |

DDP 还会把梯度放进 bucket，某个 bucket 的参数梯度 ready 后即可启动 AllReduce，以隐藏通信。TP 的逐层输出常被下一算子立即消费，可重叠窗口更窄。这也是推理 TP 更关注微秒级延迟和长尾的原因。

**涉及知识点**：Tensor Parallel、Data Parallel、DDP Reducer、bucket、计算通信重叠、推理关键路径。

**追问**：为什么 DDP 可以调 bucket size，而 TP 通常不能随意把多层 AllReduce 合成一个大 bucket？

**追问回答**

DDP 的不同参数梯度在反向传播中逐步 ready，但通常直到 optimizer step 才要求全部完成。框架可以把多个梯度放进 bucket，在不改变数学依赖的前提下延迟一部分通信，并与剩余反向计算重叠。

TP 的 AllReduce 位于前向或反向层间关键路径，某一层的归约结果往往是下一层 GEMM、residual 或 normalization 的立即输入。把多层结果攒成一个 bucket 会要求先计算尚且依赖前层结果的后续层，形成不可能的依赖。TP 优化通常依靠算子融合、通信计算 overlap、sequence parallel 或布局变化，而不是任意跨层聚合。

**评分要点（10 分）**：数据和并行语义区别得 4 分；前向/反向区别得 2 分；说明 bucket overlap 得 2 分；联系推理延迟得 2 分。

### 18. 为什么 Dense 推理的 `TP=1, DP=8` 通常没有逐层 DP AllReduce？

**核心回答**

Dense DP 推理中，每个 rank 持有完整模型，处理独立请求 batch，没有梯度需要同步，也不需要合并其他副本的局部激活。因此 DP rank 之间通常没有逐层 collective，前端只需把请求路由到不同副本。

**深入展开**

DP 推理的主要目标是扩展吞吐和并发，而不是让多个 rank 共同完成一次 GEMM。每个副本有自己的调度器、KV Cache 和连续批处理状态。系统仍可能使用控制通信完成健康检查、全局空闲判断、权重热更新或统计聚合，但这些不是 Dense Transformer 每层的计算依赖。

例外包括 DP Attention 与 Expert Parallel 组合、某些全局采样方案以及在线训练向推理引擎广播权重。特别是 MoE 模型中，Attention 可以按 DP 处理不同 token，而 Expert 层跨 DP/EP rank 重新分发 token，此时各 rank 的 forward 需要对齐，不能再视为完全独立副本。

**涉及知识点**：推理数据并行、请求路由、KV Cache、DP Attention、MoE、在线权重更新。

**追问**：`TP=8, DP=4` 时，哪些通信发生在 8-rank TP 组，哪些可能发生在 4-rank DP 组？

**追问回答**

每个模型副本内部的 activation collective 发生在 8-rank TP 组，例如 Row Parallel 输出的 AllReduce、sequence-parallel 的 ReduceScatter/AllGather，以及某些 vocab-parallel 通信。四个副本中同一 TP 位置的 rank 组成 4-rank DP 组。

训练时，DP 组负责对应参数梯度的 AllReduce 或 ReduceScatter，以及与 ZeRO/FSDP 状态分片相关的 AllGather；纯 dense 推理通常不需要逐层 DP collective，因为四个副本独立服务不同请求。不能看到总 world size 32 就把一次 TP AllReduce 建模成 32 rank，必须从具体 ProcessGroup 成员推导流量和拓扑。

**评分要点（10 分）**：完整模型副本和独立 batch 得 4 分；指出无梯度得 1 分；区分控制通信得 2 分；给出 MoE/权重更新例外得 3 分。

### 19. Sequence Parallel 为什么常使用 `AllGather + ReduceScatter`？它相对 AllReduce 有什么收益？

**核心回答**

Sequence Parallel 把原本在 TP rank 上复制的激活沿 token/sequence 维度切分。Row Parallel 输出后使用 ReduceScatter，只保留每个 rank 的 token 子集；进入需要完整 token 集合的下一段 Column Parallel 计算前再 AllGather。它主要减少复制激活的显存，并创造更细的通信重叠机会，不是免费消除通信量。

**深入展开**

从代数上看：

$$
\operatorname{AllReduce}
=\operatorname{ReduceScatter}+\operatorname{AllGather}
$$

传统 TP 在子层边界完成整个 AllReduce，使所有 rank 都持有 `[T,H]`。SP 则在边界只完成 ReduceScatter，让 rank $r$ 持有 `[T/p,H]`。LayerNorm、Dropout 或残差等逐 token 操作可以在本地 token 子集上完成，直到下一次确实需要复制输入时再 AllGather。

收益包括激活显存约按 TP 度下降、Norm 等操作不重复计算，以及可能把 AllGather/ReduceScatter 与 GEMM 分块重叠。代价是布局更复杂，对不均匀 token 数、CUDA Graph 和算子支持提出更高要求。

**涉及知识点**：Sequence Parallel、token shard、激活显存、AllReduce 分解、通信计算重叠。

**追问**：SP 为什么适合按 sequence 维切，而不是让 RMSNorm 在 hidden 维分片上直接本地计算？

**追问回答**

RMSNorm、LayerNorm、dropout 等操作通常对每个 token 独立，但需要该 token 的完整 hidden vector。沿 sequence 维切后，每个 rank 只处理一部分 token，却保留这些 token 的全部 hidden 元素，因此归一化统计和 element-wise 操作都能本地完成。

沿 hidden 维切时，每个 rank 只有局部平方和，精确 RMSNorm 至少要归约统计量；LayerNorm 还要归约 sum 和 sum-of-squares。SP 的价值正是让这类非 GEMM 区域避免 TP 通信，并降低每 rank activation memory，而在 GEMM 边界用 ReduceScatter/AllGather 转换布局。

**评分要点（10 分）**：说明 RS/AG 布局得 3 分；显存收益得 2 分；指出总通信并未自然消失得 2 分；解释本地 Norm 和重叠得 3 分。

### 20. Vocab Parallel Embedding 为什么可能需要 AllReduce？分片 LM Head 为什么可能需要 AllGather？

**核心回答**

Vocab Parallel Embedding 按词表行切权重。每个 token id 只由拥有该词表区间的 rank 查到向量，其他 rank 产生零向量，随后 AllReduce(SUM) 让所有 TP rank 得到完整 embedding。LM Head 按词表输出切分后，每个 rank 只有部分 logits；若采样逻辑需要完整词表 logits，就要 AllGather。

**深入展开**

Embedding 的本地计算可以写成：

```python
owned = vocab_start <= token_id < vocab_end
local_id = token_id - vocab_start
local_embedding = table[local_id] if owned else zeros(H)
embedding = all_reduce(local_embedding, SUM)
```

由于每个合法 token 只有一个 owner，SUM 实际承担“选择并复制 owner 结果”的作用。若词表存在 replicated padding 或重复 ownership，则必须额外保证掩码契约，否则会重复相加。

LM Head 得到的是 `[T,V/p]` logits。最直接方案是 AllGather 成 `[T,V]` 再做 softmax/top-k；更高效的 distributed sampling 可以先归约全局 max/sum，或者只交换局部 top-k 候选，避免完整 logits AllGather，但算法必须保证全局采样语义和数值稳定性。

**涉及知识点**：Vocab Parallel、mask、embedding lookup、distributed softmax、distributed top-k、采样。

**追问**：为什么 argmax 可以只收集每个 rank 的局部最大值和 token id，而精确随机采样通常更复杂？

**追问回答**

全局 argmax 可写成对二元组 `(logit, token_id)` 的结合性归约：每个 rank 先求本地最大值，再在 rank 间选最大二元组，通信量与词表大小无关。只要定义相同的 tie-break 规则，结果就是精确的。

随机采样需要构造全局概率分布。softmax normalization 依赖全局最大值和全局指数和，top-k/top-p 还需要知道跨 shard 的候选集合与累计概率；最终所有 rank 还要对同一次随机选择达成一致。可以设计分布式精确采样而不 AllGather 全词表，但至少需要多轮统计/候选通信，且必须严谨处理 RNG 和截断边界，所以比 argmax 的一次小型归约复杂得多。

**评分要点（10 分）**：Embedding owner/mask/AllReduce 得 4 分；LM Head 分片得 2 分；解释 AllGather 得 1 分；提出正确的分布式采样替代方案得 3 分。

---

## Level 3：NCCL 内部机制

### 21. 多进程如何创建同一个 NCCL communicator？`ncclUniqueId` 起什么作用？

**核心回答**

一个进程调用 `ncclGetUniqueId()` 获得 opaque unique ID，再通过 NCCL 之外的通道把它发送给所有参与进程。每个进程设置自己的 CUDA device，并以相同 `nranks`、唯一的 communicator rank 和相同 unique ID 调用 `ncclCommInitRank()`，共同创建 communicator。

**深入展开**

典型伪代码为：

```cpp
if (globalRank == 0) ncclGetUniqueId(&id);
bootstrapBroadcast(&id, sizeof(id));

cudaSetDevice(localDevice);
ncclCommInitRank(&comm, nranks, id, commRank);
```

unique ID 不是 rank 表，也不保存完整 GPU 拓扑，更不是数据加密密钥。它提供同一次初始化 rendezvous 所需的 bootstrap 标识和连接信息。rank、nranks 和 GPU 绑定仍由每个参与者显式提供。

初始化不只是“交换一个 ID”。后续还要交换 peer info、识别同主机/同进程关系、构建 topology、搜索 Ring/Tree/channel，并为 P2P、SHM 或 NET transport 建立 connector。ProcessGroupNCCL 还可能延迟到该 ProcessGroup 的第一次 CUDA collective 才真正创建 NCCL communicator，所以 `init_process_group()` 返回不等于所有 NCCL 连接均已完成。

**涉及知识点**：unique ID、out-of-band broadcast、`cudaSetDevice`、`ncclCommInitRank`、lazy communicator initialization。

**追问**：两个 rank 误用相同 communicator rank，或者分别使用不同 unique ID，会在哪个阶段失败？

**追问回答**

同一 unique ID 下两个进程都声明为相同 rank，会造成 peer 表和连接身份冲突：真正缺失的那个 rank 永远不会加入。错误可能在 `ncclCommInitRank` 的 bootstrap/connect 阶段直接暴露，也可能因连接按需建立而在第一次 collective 才超时，不能假设一定返回一个清晰的“duplicate rank”错误。

若参与者分成了两个不同 unique ID，它们实际上在创建两个互不相干的 communicator；每一组都等待自己 world size 中缺失的成员，通常表现为初始化或首个通信 hang。排查时要记录每个进程的 unique ID 摘要、声明的 rank/world size、hostname/PID/device，并找“最早发生分叉”的证据，而不是只看最后超时的 rank。

**评分要点（10 分）**：标准初始化流程得 4 分；解释 unique ID 的边界得 2 分；指出 device/rank 显式映射得 2 分；了解框架 lazy init 得 2 分。

### 22. NCCL 初始化时为什么需要 bootstrap 网络？bootstrap 流量和正式数据流量有什么区别？

**核心回答**

多机进程在正式 GPU 通信连接建立前还不知道全部 peer、拓扑和 transport endpoint，因此需要一个最小 bootstrap 通道完成 rendezvous、地址交换和 peer 信息收集。正式 collective 的 payload 则走 NCCL 最终选择的 P2P、SHM、IB/RoCE 或 Socket transport。

**深入展开**

初始化可以分成以下阶段：

```text
unique ID / root endpoint
  -> bootstrap connect
  -> peer info allgather
  -> local rank 与 topology 识别
  -> Ring / Tree / channel 搜索
  -> transport canConnect / setup / connect
  -> communicator ready
```

bootstrap 使用 TCP 并不意味着正式数据一定走 TCP；正式 NET transport 使用 IB Verbs 也不意味着 bootstrap 可以不依赖 IP 可达性。排障时必须先判断 hang 位于初始化的哪一层：

- 连 root endpoint 都无法连接，属于 bootstrap 地址、接口或防火墙问题。
- 已打印 topology，但卡在 connect，可能是 NET plugin、RDMA endpoint 或 rank 缺失。
- communicator 已 ready，首个 collective 才卡住，更应检查 collective 顺序、数据路径和某个 rank 的运行状态。

**涉及知识点**：rendezvous、bootstrap socket、peer info、transport connect、初始化状态机。

**追问**：为什么 `MASTER_ADDR/PORT` 可达仍不能证明 RDMA 数据路径一定正常？

**追问回答**

`MASTER_ADDR/PORT` 通常只承载 rendezvous 或 TCPStore 控制流量。它可达只能证明进程能通过某个 TCP 地址交换元数据；NCCL payload 可能走另一块 NIC、另一张网、不同端口/QP，并要求 verbs、内存注册和 GDR 全部可用。

RDMA 路径还依赖 HCA 选择、GID/LID、MTU、路由、PFC/ECN、`/dev/infiniband` 权限、memlock、GPU-NIC 拓扑和 peer-memory/dma-buf 支持。完整验证至少包括 NCCL transport 日志、目标 GPU buffer 的注册/GDR 证据、端口计数器和多 rank payload benchmark；`telnet MASTER_PORT` 成功只验证了最前面的 bootstrap。

**评分要点（10 分）**：bootstrap 目的得 3 分；区分控制流量和 payload 得 3 分；列出初始化阶段得 2 分；能按日志阶段定位得 2 分。

### 23. NCCL 如何发现 GPU、PCIe、NVLink、NIC 拓扑？拓扑为什么影响 Ring/Tree 构造？

**核心回答**

NCCL 综合 CUDA/NVML、PCI/sysfs、CPU NUMA 信息和网络 plugin 提供的 NIC 属性，构建包含 GPU、CPU、PCI bridge、NIC、NVLink/NVSwitch 等节点与链路的内部拓扑图，再基于链路类型、带宽和路径约束搜索 collective graph。

**深入展开**

在 NCCL 2.22.3 中，可以沿以下源码模块理解：

```text
graph/xml.cc       读取或导出 topology XML
graph/topo.cc      构造与规范化拓扑节点
graph/paths.cc     计算 GPU/GPU、GPU/NET 路径
graph/search.cc    搜索满足带宽和类型约束的 graph
graph/connect.cc   生成 Ring/Tree 的 rank 连接关系
transport.cc       为 graph edge 选择实际 transport
```

Ring 的排列需要避免把高带宽 NVLink 域切碎，并尽量让每个 channel 使用均衡链路。Tree 要考虑 parent/child 的路径和并发。多 NIC 系统还要决定 GPU 到近端 NIC 的映射、是否跨 NIC rail，以及跨节点 channel 如何连接。

rank placement 会改变图搜索结果。即使物理硬件不变，把 `CUDA_VISIBLE_DEVICES` 顺序或进程 GPU 绑定打乱，也可能让逻辑相邻 rank 对应到更差的物理路径。

**涉及知识点**：NVML、sysfs、NUMA、topology XML、路径等级、graph search、rank placement。

**追问**：`nvidia-smi topo -m` 和 `NCCL_TOPO_DUMP_FILE` 分别证明什么，为什么二者不能互相替代？

**追问回答**

`nvidia-smi topo -m` 从驱动/系统视角展示 GPU、NIC、CPU affinity 以及 NVLink/PCIe 距离，是物理和 OS 可见拓扑的基线。`NCCL_TOPO_DUMP_FILE` 则导出 NCCL 在当前进程、容器、版本和环境变量下实际发现并建模的节点、链路、带宽与路径。

前者不能证明 NCCL plugin 看到了哪块 NIC、如何解释虚拟化设备或最终构造了什么 Ring；后者也不能取代硬件事实，因为错误的 sysfs mount、容器设备映射或 topology override 可能让 NCCL 的视图不完整。排障应把两份图对齐：先确认硬件关系，再解释 NCCL 为什么选出当前路径。

**评分要点（10 分）**：能列出拓扑信息来源得 3 分；描述内部图和路径得 2 分；解释如何影响 Ring/Tree/channel 得 3 分；联系 rank placement 得 2 分。

### 24. 单机通信可能使用哪些路径？

**核心回答**

常见路径包括 GPU P2P、共享内存和必要时的网络 transport。GPU P2P 可能经 NVLink/NVSwitch 或 PCIe peer access；跨进程 P2P 通常还需要 CUDA IPC 映射。若 P2P 不可用，NCCL 可以退化到 SHM staging；特定硬件和版本还可能使用 NVLS/NVSwitch offload 等能力。

**深入展开**

需要区分“物理路径”和“NCCL transport 名称”：

| NCCL 路径 | 可能的数据方式 | 常见前提 |
|---|---|---|
| P2P | 直接 peer pointer、CUDA IPC、P2P read/write | peer access、拓扑和安全策略允许 |
| SHM | 共享 host buffer 加 GPU/CPU copy/progress | 同主机、共享内存可见且容量足够 |
| NET | Socket 或网络 plugin，即使 peer 在同机也可作为 fallback | 可用网络接口/plugin |
| NVLS | NVSwitch multicast/reduction 能力 | 支持的 GPU/NVSwitch、版本和注册条件 |

看到物理 NVLink 不足以证明 NCCL 正在使用 P2P。还应结合 `NCCL_DEBUG_SUBSYS=GRAPH,P2P,SHM`、拓扑 dump、Nsight kernel 和禁用 P2P 的受控性能对照。反之，性能下降也不能单独证明走了 SHM，必须有路径日志。

**涉及知识点**：CUDA peer access、CUDA IPC、P2P、SHM、NVLS、transport fallback。

**追问**：容器没有共享 IPC namespace 时，为什么 P2P 可能正常而 SHM fallback 失败？

**追问回答**

GPU P2P transport 主要依赖 CUDA peer access/CUDA IPC、GPU 可见性和驱动能力，数据可直接经 NVLink 或 PCIe 在 GPU memory 间传输。NCCL 的 SHM fallback 则需要进程通过同一个可见的 POSIX shared-memory 区域交换 staging buffer，并受到 `/dev/shm` mount、IPC namespace、容量和权限约束。

因此某些部署中 GPU peer path 可用，P2P benchmark 正常；一旦因拓扑或环境变量禁用 P2P，NCCL 改走 SHM 时才暴露 namespace 隔离或 `/dev/shm` 太小。验证要分别强制/禁用 `P2P` 与 `SHM`，不能用一种 transport 成功推断另一种也正常。

**评分要点（10 分）**：列出 P2P/SHM/NET 得 3 分；解释 NVLink/PCIe 与 P2P 的关系得 2 分；了解 CUDA IPC 得 2 分；提出日志加对照实验的证据链得 3 分。

### 25. 跨节点通信可能使用哪些路径？什么是 GPUDirect RDMA？

**核心回答**

跨节点 NCCL 可以使用 Socket 或网络 plugin。高性能 NVIDIA GPU 集群通常通过 IB Verbs 在 InfiniBand 或 RoCE 网络上传输。GPUDirect RDMA 允许第三方 PCIe 设备，例如 NIC，直接 DMA 读写 GPU memory，减少 CPU host staging 和额外拷贝。

**深入展开**

典型 GDR 数据路径为：

```text
GPU memory
  <-> PCIe fabric
  <-> local NIC
  <-> IB/RoCE network
  <-> remote NIC
  <-> PCIe fabric
  <-> remote GPU memory
```

“直接”指 payload 不必经过系统内存，不表示没有 PCIe，也不表示没有 CPU proxy。连接创建、内存注册、work request 提交、completion 处理和错误管理仍需要软件控制路径。

GDR 需要 GPU、NIC、驱动、内核模块或 DMA-BUF 路径、IOMMU/ACS 配置、内存注册和 PCIe 拓扑共同满足条件。NIC 与 GPU 跨 CPU socket 时可能仍能工作，但性能和稳定性会受到 UPI/NUMA 路径影响。

**涉及知识点**：IB Verbs、RoCE、Socket、GPUDirect RDMA、DMA-BUF、`nvidia-peermem`、PCIe root complex。

**追问**：看到日志中的 IB plugin 就能证明 payload 使用了 GDR 吗？还需要什么证据？

**追问回答**

不能。IB/verbs plugin 被加载只说明 NCCL 找到了一个候选网络后端；payload 仍可能先从 GPU 拷到 host pinned memory，由 CPU proxy 通过 NIC 发送，再在对端搬回 GPU。也可能只有部分方向、部分 GPU-NIC 组合启用了 GDR。

还需要看到该连接对 GPU pointer 的 direct registration/GDR 判定、GPU-NIC 拓扑距离和相关驱动能力，并用 NIC/PCIe 计数器、CPU proxy 占用或受控关闭 GDR 的 A/B 验证。最好对每个 rank、channel、send/recv 方向取证；“加载了 `NET/IB`”与“所有 payload 都直接 DMA 到显存”是两个不同命题。

**评分要点（10 分）**：跨节点 transport 得 2 分；准确解释 GDR 得 3 分；指出控制路径仍存在得 2 分；列出注册、驱动和拓扑条件得 3 分。

### 26. 没有 GPUDirect RDMA 时，GPU 数据可能经过怎样的 staging 路径？

**核心回答**

常见 fallback 是先把 GPU 数据复制到 page-locked host staging buffer，NIC 从 host memory 发送；对端 NIC 写入 host buffer 后，再复制到目标 GPU。相比 GDR，它增加 GPU/CPU 内存路径、PCIe 流量、copy engine 或 GPU kernel 工作以及 CPU proxy 压力。

**深入展开**

逻辑路径可表示为：

```text
source GPU
  -> source pinned host buffer
  -> source NIC
  -> network
  -> destination NIC
  -> destination pinned host buffer
  -> destination GPU
```

实现不一定为每条消息同步执行一整次 D2H 和 H2D。NCCL 会使用 buffer、step 和 proxy 流水化多个阶段，使网络和拷贝尽量重叠。但最终瓶颈可能落在 PCIe、host memory bandwidth、NUMA、copy engine 或 CPU progress 上。

排障时可比较 GDR 开关前后的大消息带宽、CPU 使用率、PCIe 计数器和日志中的 GDR read/write 选择。若 workload 是极小消息，注册和启动开销可能比额外 payload 拷贝更显著，不能仅凭理论带宽预测收益。

**涉及知识点**：host staging、pinned memory、copy engine、CPU proxy、流水线、NUMA。

**追问**：为什么 staging 路径仍可能通过流水获得不错吞吐，但延迟和 CPU 占用明显恶化？

**追问回答**

host staging 可把 payload 切成多个 slice，让 `GPU->host`、NIC send/recv 和 `host->GPU` 三个阶段并行。消息足够大、队列足够深时，稳态吞吐由最慢 stage 决定，而不是三段耗时简单相加，所以带宽曲线可能仍然可观。

但第一个字节必须经过额外拷贝和同步，最后一个字节也要完成回拷，短消息延迟一定更敏感；CPU proxy 还要轮询、推进队列并管理 host buffer，同时消耗 PCIe 带宽和内存带宽。应同时观察小消息 latency、CPU core 使用率、PCIe traffic 与大消息 algbw，单看后者容易漏掉 staging 代价。

**评分要点（10 分）**：画出完整 staging 路径得 4 分；说明额外资源成本得 2 分；解释流水重叠得 2 分；提出验证指标得 2 分。

### 27. NCCL 的 Channel 是什么？增加 Channel 为什么可能提高带宽，也可能增加开销？

**核心回答**

Channel 是 NCCL 将 collective 数据和通信图并行化的逻辑流水线。经典 Ring/Tree 路径中，一个 channel 通常对应一套独立的 rank 连接和一个 CUDA block/CTA，数据按 channel 分片并行推进。

**深入展开**

增加 channel 可能带来：

- 更多 GPU blocks 和并行数据片段，提高链路与多路径利用率。
- 多个 channel 映射到不同 NVLink、NIC queue 或 rail，提高 aggregate bandwidth。
- 更短 chunk，改善流水填充。

同时也可能带来：

- 更多 CTA 占用 SM，和 GEMM/Attention 争抢执行资源。
- 更多 connector、buffer、proxy work 和调度开销。
- 小消息被切得过碎，固定开销占比上升。
- channel 超过真实独立链路数后只增加竞争，不增加物理带宽。

因此 channel 数是 topology、algorithm、protocol 和 message size 联合 tuning 的结果。强制 `NCCL_MIN_NCHANNELS` 或 `NCCL_MAX_NCHANNELS` 适合诊断，不应在缺少完整 size sweep 时直接作为生产优化。

**涉及知识点**：Channel、CTA、chunk、slice、SM occupancy、多 rail、tuning。

**追问**：如何用 Nsight Systems 验证某次 collective 实际使用的 channel/CTA 数？

**追问回答**

先在 NCCL `GRAPH/INFO` 日志中记录该 communicator 配置的 channel 数，再在 Nsight Systems 中定位同一次 collective 的 NCCL kernel，查看 kernel launch 的 grid dimensions。经典 NCCL kernel 通常以 block/CTA 承载 channel 工作，`grid.x` 因而是实际活跃 CTA 数的重要证据。

但不能机械地把所有版本的 `grid.x` 等同于“逻辑 channel 总数”：一个 kernel 可能批处理多个 work element，部分 channel 可能因消息大小未启用，新实现也可能改变 CTA 映射。应以相同 opCount/时间戳关联日志与 launch 参数，并结合源码中的 kernel launch 和 work dispatch 逻辑解释。做 `NCCL_MIN_NCHANNELS/MAX_NCHANNELS` A/B 后若 grid、带宽同时按预期变化，证据更完整。

**评分要点（10 分）**：定义 channel 得 3 分；说明带宽收益得 2 分；列出 GPU/proxy/碎片开销得 3 分；提出 profiler 验证得 2 分。

### 28. `Simple`、`LL`、`LL128` 协议分别倾向于解决什么问题？

**核心回答**

它们是 NCCL 数据传输和同步协议，不是 collective 算法。Simple 追求高 payload 效率和大消息带宽；LL 使用更细粒度的数据加 flag 协议降低同步延迟，但有效 payload 比例较低；LL128 在低延迟同步与较高 payload 效率之间折中，并受 GPU 架构和路径支持限制。

**深入展开**

Algorithm 决定“数据经过哪些 rank 和多少 step”，Protocol 决定“每个 step 如何把数据装入通信 buffer、如何发布 ready flag、以什么粒度同步”。因此存在 `Ring+Simple`、`Ring+LL`、`Tree+LL128` 等组合。

在 NCCL 2.22.3 的典型实现中：

- LL 把数据和同步 flag 交织，接收端可以在较小粒度上轮询并立即消费，代价是线上的非 payload 字节比例较高。
- LL128 以接近 128-byte line 的方式组织 payload 和 flag，提高有效载荷比例，同时保留细粒度流水。
- Simple 使用较大的 buffer step 和显式同步，适合带宽饱和区。

实际阈值不能死记。拓扑、GPU 架构、算法、channel 和 NCCL 版本都会改变 tuner 估时；某些 transport 对 LL128 还有额外限制。

**涉及知识点**：Algorithm/Protocol 正交维度、flag、payload efficiency、buffer step、tuner。

**追问**：为什么强制 LL 可能改善 4 KiB 延迟，却明显降低 256 MiB 带宽？

**追问回答**

LL 协议使用更细粒度的 ready/flag 机制，减少大 buffer 切片和同步的启动成本，小消息时可以更早地流水转发，因此 4 KiB 的 $\alpha$ 项可能更低。代价是每单位有效 payload 携带更多 flag/协议元数据，并采用不以峰值带宽为首要目标的数据搬运方式。

对 256 MiB 消息，启动成本已被摊薄，决定性能的是长期有效 payload 吞吐；LL 的编码开销和较低带宽效率会持续作用于整个消息，Simple 往往更优。协议阈值必须通过 size sweep 确定，不能用一个小消息胜点推断所有尺寸都应强制 LL。

**评分要点（10 分）**：区分协议与算法得 3 分；三协议定位得 3 分；解释 flag 与 payload 效率得 2 分；指出版本/硬件边界得 2 分。

### 29. `ncclGroupStart/ncclGroupEnd` 有什么作用？为什么多组 Send/Recv 经常需要成组提交？

**核心回答**

Group Calls 可以把多个 NCCL 操作作为一组延迟提交，用于单线程管理多 GPU、聚合多个小操作以及同时发布多组 Send/Recv。`ncclGroupEnd()` 结束 host 侧组提交，不表示组内 GPU 通信已经完成。

**深入展开**

逐个调用 P2P 可能形成 host 侧依赖：某个 `Send` 在对端 `Recv` 尚未发布时等待，而同一线程永远走不到后续 `Recv`。分组后，NCCL 先收集完整 P2P schedule，再统一建立连接和 launch：

```cpp
ncclGroupStart();
for (peer : peers) {
  ncclSend(..., peer, comm, stream);
  ncclRecv(..., peer, comm, stream);
}
ncclGroupEnd();
```

Group 还可减少多个小 collective 的 launch 开销，但“聚合提交”不保证把任意操作自动融合成数学等价的一个 collective，也不解除各 rank 的顺序契约。所有参与 rank 仍需以兼容顺序构造同一组操作。

NCCL 2.22.3 中，group state 是 thread-local；最外层 `ncclGroupEndInternal()` 才会触发各 communicator 的 prepare、plan 和 launch。嵌套 Group 只有最外层结束时真正提交。

**涉及知识点**：Group Calls、P2P matching、aggregated launch、thread-local group state、host/GPU completion。

**追问**：GroupEnd 成功后立刻复用或释放 input buffer 是否安全？

**追问回答**

不安全。`ncclGroupEnd` 结束的是 host 侧分组、初始化和 enqueue 过程；成功返回并不意味着组内 CUDA 工作已执行完。buffer 的内容和生命周期仍受传入 stream 的异步完成性约束。

应用应在通信 stream 完成 event 后再让其他 stream 覆盖该地址，host 释放也必须由 stream-aware allocator 延迟到最后使用结束。PyTorch 中需要正确保留 Work/tensor 生命周期并记录 stream；原生 CUDA 可用 event 或 stream synchronization。只有 allocator 的“free”本身能可靠记录该 stream 依赖时，提前调用逻辑释放才可能安全复用。

**评分要点（10 分）**：多 GPU/P2P/聚合三个用途得 3 分；解释避免顺序阻塞得 3 分；指出 GroupEnd 非 GPU 完成得 2 分；说明顺序契约仍存在得 2 分。

### 30. 为什么不同 rank 以不同顺序调用 collective 可能永久阻塞？

**核心回答**

同一 communicator 上的 collective 按全局操作序列匹配，而不是靠用户提供的 message tag 匹配。若 rank 0 的第 $k$ 个操作是 AllReduce，而 rank 1 的第 $k$ 个操作是 Broadcast 或跳过了该操作，各 rank 会等待不存在的匹配参与者，通常表现为 hang 或 watchdog timeout。

**深入展开**

错误示例：

```text
rank 0: AllReduce A -> Broadcast B
rank 1: Broadcast B -> AllReduce A
```

即使最终每个 rank 都“调用了同样两个 collective”，序列位置仍不一致。NCCL 不解析 tensor 业务名称 A/B，只看到 communicator 上 opCount/sequence 对应的操作参数。

大模型中常见根因包括：

- 某个 rank 因空 batch 提前 return，其他 rank 仍执行 TP collective。
- MoE 路由或 speculative decoding 的条件分支没有全 rank 对齐。
- 异常只在一个 rank 抛出，其余 rank 继续进入下一次 collective。
- 多 communicator 在不同线程并发使用时，host launch 顺序不一致并形成跨 communicator 循环等待。

修复不是增加 timeout，而是让控制流、collective sequence 和错误传播一致。可以在框架层记录 process group、sequence number、op type、shape 和调用栈，找出第一个分歧点。

**涉及知识点**：collective ordering、opCount、SPMD、条件分支、watchdog、多 communicator 并发。

**追问**：两个完全不同的 communicator 是否也可能因 host launch 顺序形成死锁？

**追问回答**

可能，尤其在本文基线 NCCL 2.22.3 中，多个 communicator 共享同一批设备资源且各 rank 启动顺序不一致时。例如 rank 0 先在 comm A 启动再等待 comm B，而 rank 1 先启动 B 再等待 A；若 host 阻塞、group call、stream 依赖或资源获取形成环，就会互相等待。

解决方法是让所有重叠成员对 communicator 使用一致的全局 launch 顺序，或者用显式 CUDA Event/host 序列化打破环，并遵循目标 NCCL 版本的 multi-communicator ordering 约束。communicator 的匹配域彼此独立，不代表调度资源和 host 控制流也彼此隔离。

**评分要点（10 分）**：指出同 communicator 全局顺序得 3 分；能分析示例得 2 分；给出模型分支根因得 2 分；提出 sequence/shape 证据定位得 3 分。

### 31. 相同 collective 但 count、dtype 或 communicator 不一致可能产生什么结果？

**核心回答**

collective 名称相同并不足够。参与 rank 还必须在 communicator、count、datatype、reduction op、root 和 buffer 契约上兼容。NCCL 不一定在执行前跨 rank 完整比较这些元数据，不匹配可能导致 hang、越界访问、数据截断、silent corruption 或异步 CUDA 错误。

**深入展开**

典型错误包括：

- rank 0 用 FP16 count 1024，rank 1 用 FP32 count 1024，字节数和 reduction 解释都不同。
- ReduceScatter 的 input 总元素数没有满足 `nranks * recvcount`。
- Broadcast 的 root 在不同 rank 上配置不同。
- 两组成员相同但某个 rank 使用了另一个 communicator 实例。
- AllGather in-place pointer 没有指向本 rank 对应 offset。

有些框架会在 ProcessGroup 层做 shape 检查，但不能依赖所有错误都被同步检测。生产系统应在 scheduler 或通信封装层记录并校验 collective fingerprint：

```text
process-group id + sequence + op + dtype + shape/count + root
```

故障注入必须设置外部 watchdog，不能让错误实验无限等待。

**涉及知识点**：collective contract、dtype/count、root、in-place pointer、fingerprint、silent corruption。

**追问**：为什么仅比较 tensor shape 仍可能漏掉 dtype 或 storage offset 导致的错误？

**追问回答**

相同 shape 只说明逻辑维度相同。FP16 与 FP32 的元素字节数、NCCL datatype 和归约解释不同；两个 view 即使 shape 相同，也可能有不同 stride、storage offset、alias 关系或所在 device。NCCL 最终接收的是裸指针、count 和 datatype，并按连续元素契约访问。

因此跨 rank 一致性检查至少要覆盖 collective 类型、communicator/op 序号、shape/count、dtype、device、root/peer，以及 wrapper 对 contiguity 的要求。本地还要检查 `data_ptr` 对应区域是否足够、view 是否重叠。只比较打印出来的 `torch.Size`，可能让各 rank 传入相同元素数却用不同字节解释，甚至从不同 offset 读写错误区域。

**评分要点（10 分）**：列出完整匹配字段得 4 分；说明可能不被预检查得 2 分；覆盖 hang 与 corruption 两类结果得 2 分；提出 fingerprint/负向实验得 2 分。

### 32. NCCL 如何与 CUDA Stream、Event 和 CUDA Graph 配合？Graph Capture 有哪些约束？

**核心回答**

NCCL 操作提交到调用者给定的 CUDA stream，并服从该 stream 的依赖。框架使用 CUDA Event 在计算 stream 和通信 stream 之间传递 producer/consumer 依赖。CUDA Graph capture 会把 NCCL launch 纳入图，replay 时复用已捕获的执行计划，因此 communicator、调用顺序、shape、buffer 生命周期和图并发方式必须满足版本对应的约束。

**深入展开**

ProcessGroupNCCL 的常见 stream 关系为：

```text
application stream: GEMM -> record input-ready event
NCCL stream:        wait event -> NCCL kernel -> record work-end event
consumer stream:    wait work-end event -> next kernel
```

没有这些依赖时，NCCL 可能读取尚未写完的 input，或消费者读取尚未完成的 output。allocator 的 `recordStream` 只能避免 storage 过早复用，不能自动解决无依赖的并发读写。

在 NCCL 2.22.3 中，capture 首次经过 host enqueue、tuning、task/plan 构建和图节点记录；replay 不会重新运行完整 Python/C++ 调度路径。推理框架通常为不同 batch shape 捕获多个 graph，并使用稳定地址的 graph buffer。动态改变 collective 序列、临时分配不同 buffer，或让不同 rank replay 不同图，都可能破坏正确性。

新版 NCCL 对 buffer registration、混合多个 stream 和并发 graph 有持续演进，必须以部署版本文档为准，不能套用单一规则。

**涉及知识点**：CUDA stream/event、ProcessGroupNCCL stream、allocator safety、CUDA Graph capture/replay、静态地址。

**追问**：为什么 CUDA Graph 能降低 CPU launch 开销，却不能消除 collective 的网络延迟？

**追问回答**

CUDA Graph 把一串已捕获的 kernel 和依赖作为整体 replay，减少 Python/C++ 调度、CUDA API 调用和每次 kernel launch 的 CPU 开销，也能让 launch gap 更稳定。它优化的是提交路径，而不是改变 collective 的数学数据依赖。

replay 时，相同 payload 仍要经过 NVLink/PCIe/NIC，远端 rank 仍要到达匹配 collective，Ring/Tree 的 step 和链路传播延迟也仍存在。Graph 对很多小 kernel 之间的 host 空洞可能很有效，但 NCCL kernel 内部的网络时间不会凭空消失；必须把 CPU gap、rank arrival skew 和 kernel duration 分开测量。

**评分要点（10 分）**：stream 异步模型得 2 分；画出 event 依赖得 3 分；说明 capture/replay 与稳定 buffer 得 3 分；指出版本约束得 2 分。

---

## Level 4：大模型推理框架与生产问题

### 33. 描述 vLLM/SGLang 中一次 TP AllReduce 的完整调用链。

**核心回答**

模型的 Row Parallel 层先计算局部 GEMM 结果，再调用 TP group 的 AllReduce。框架通信抽象根据硬件、消息大小和 Graph 状态选择 Custom AllReduce、FlashInfer/MSCCL++/对称内存、直接 PyNccl 或 PyTorch ProcessGroupNCCL；只有后两类 NCCL 路径最终进入 `libnccl.so`。

**深入展开**

vLLM 的典型链路为：

```text
RowParallelLinear.forward
  -> tensor_model_parallel_all_reduce
  -> TP GroupCoordinator.all_reduce
  -> CudaCommunicator.all_reduce
  -> Custom AR / PyNccl / torch.distributed fallback
  -> ncclAllReduce
  -> Ring/Tree + protocol + channel
  -> P2P/SHM/NET transport
```

当前 vLLM 的 `CudaCommunicator` 会依次判断多个优化 backend 是否适用，再回退到 PyNccl 或 `torch.distributed`。PyNccl 直接加载 NCCL C API，可以显式传入 output buffer 和 CUDA stream；ProcessGroupNCCL 则由 c10d 管理 communicator、stream、Work 和错误监控。

SGLang 的结构相似，但还根据配置尝试 Custom AR、MSCCL++、Torch symmetric memory、PyNccl 或 ProcessGroupNCCL。两个框架都会创建 TP、PP、DP/EP、CP 等不同 communicator；“初始化 NCCL world group”不表示每次模型通信都在整个 world 上进行。

源码入口：

- [vLLM RowParallelLinear](https://github.com/vllm-project/vllm/blob/3ad85e0de4ee88b8fe9d7fde8a3a024c92cb3c7b/vllm/model_executor/layers/linear.py#L1537-L1692)
- [vLLM CudaCommunicator](https://github.com/vllm-project/vllm/blob/3ad85e0de4ee88b8fe9d7fde8a3a024c92cb3c7b/vllm/distributed/device_communicators/cuda_communicator.py#L273-L338)
- [SGLang RowParallelLinear](https://github.com/sgl-project/sglang/blob/43124cdd90e543ea79b63800390e917063809510/python/sglang/srt/layers/linear.py#L1371-L1599)
- [SGLang GroupCoordinator](https://github.com/sgl-project/sglang/blob/43124cdd90e543ea79b63800390e917063809510/python/sglang/srt/distributed/parallel_state.py#L579-L785)

**涉及知识点**：Row Parallel、GroupCoordinator、Custom AR、PyNccl、ProcessGroupNCCL、多维 communicator。

**追问**：PyNccl 和 ProcessGroupNCCL 都调用 NCCL，它们在 stream、buffer 和错误管理上有什么区别？

**追问回答**

PyNccl 通常是框架为了低开销或特定执行路径封装的直接 NCCL binding：框架自己选择 communicator、CUDA stream、输入输出地址和 capture 方式，也必须自己管理 buffer 生命周期、跨 stream event 与错误查询。它便于和 Custom AllReduce、KV transfer 或 graph replay 紧密集成，但正确性责任更多落在框架通信层。

ProcessGroupNCCL 是 PyTorch c10d 的完整 backend。它维护 communicator cache、专用 NCCL stream/event、`WorkNCCL` completion、allocator stream safety、timeout/watchdog 和异步错误传播，并把 collective 接入 ProcessGroup 语义。两者最终都可能调用同一个 `libnccl.so`，但不能混用其 Work、stream 或生命周期假设；性能差异也可能来自 wrapper 调度，而非 NCCL 算法本身。

**评分要点（10 分）**：模型到 group 链路得 3 分；通信 backend 分派得 3 分；能区分 PyNccl/c10d/custom 得 2 分；说明最终算法和 transport 得 2 分。

### 34. 模型语义上调用了 AllReduce，为什么 Nsight Systems 中不一定出现 NCCL kernel？

**核心回答**

AllReduce 是模型所需的数学语义，不是实现名称。框架可能用 Custom AllReduce、FlashInfer、MSCCL++、对称内存或 fused kernel 实现同一语义；world size 为 1 时还会直接返回。因此 timeline 可能出现框架自定义 CUDA kernel，而没有 `ncclKernel_*`。

**深入展开**

常见情况有：

- 单机 NVLink 拓扑下，小消息命中 one-shot/two-shot Custom AR。
- AllReduce 与 residual、RMSNorm 或 quantization 被融合。
- CUDA Graph 中 kernel 名称或调用栈和 eager 模式不同。
- Sequence Parallel 将一个 AllReduce 改写为 ReduceScatter 和 AllGather。
- TP size 为 1，通信封装成为 no-op。

验证不能只搜索 kernel 名称，应建立三层证据：

1. 配置和启动日志说明 backend 是否启用。
2. 框架分派点的 NVTX/日志说明该 shape 命中哪个分支。
3. Nsight Systems 同时检查 CUDA kernel、NCCL API trace 和 stream 依赖。

做 A/B 时可禁用 Custom AllReduce，使其回退 NCCL，再比较 kernel、正确性和性能。若只比较总 latency 而没有路径证据，无法证明差异来自哪种实现。

**涉及知识点**：语义与实现、Custom AR、kernel fusion、CUDA Graph、Nsight Systems、路径证据。

**追问**：禁用 Custom AR 后性能变慢，为什么仍不能直接断言 Custom AR kernel 本身更快？

**追问回答**

这个开关通常同时改变多项条件：fallback 可能使用不同 stream、buffer 注册、CUDA Graph capture、同步点、fusion 边界和消息阈值，甚至改变 warmup 与内存分配。端到端变慢只能证明“启用 Custom AR 的整条路径更快”，不能把差值全部归因于一个 kernel 的设备执行效率。

要判断 kernel 本身，应在相同输入、拓扑、stream 依赖和已完成 warmup 的条件下，用 GPU event/Nsight 对成对调用计时，并验证传输字节和结果一致；再单独测 host gap、前后同步和内存开销。最终应同时报告 kernel-level 与 end-to-end 结果，避免把系统集成收益误写成算法内核收益。

**评分要点（10 分）**：语义/实现区别得 3 分；列出替代路径得 3 分；提出日志+trace+A/B 证据得 3 分；考虑 no-op 得 1 分。

### 35. 为什么推理框架喜欢为单机小消息实现 Custom AllReduce，而跨节点通常回退 NCCL？

**核心回答**

Decode 中很多 TP AllReduce 是延迟敏感的小消息。单机固定 GPU 集合可以预先交换 CUDA IPC/对称内存 handle，让一个定制 kernel 直接读写 peer buffer，减少通用 NCCL 的调度、协议和 proxy 开销。跨节点则需要处理 NIC、RDMA、拥塞、多 rail、注册和故障，NCCL 的成熟 transport 和 tuning 更有优势。

**深入展开**

Custom AR 常见设计包括：

- one-shot：每个 rank 写出本地数据，再由每个 rank 读取所有 peer 对应位置并完成 reduction。
- two-shot：先 ReduceScatter 到 owner，再 AllGather complete shard。
- 预注册稳定 buffer，配合 CUDA Graph 复用地址。
- 直接融合 reduction 与 residual/normalization，减少中间显存流量。

其适用条件通常较苛刻：固定 rank 数、支持 peer access、特定 dtype/对齐/消息上限、完整 NVLink 或可接受的 PCIe 拓扑。消息变大后，所有 rank 直接读取所有 peer 的 one-shot 流量和内存压力可能变差；跨节点也无法用 CUDA IPC 直接映射远端显存。

NCCL 提供更完整的 Ring/Tree、channel、NET plugin、GDR 和错误处理，所以框架一般对不支持的 size、world size、拓扑或多机配置回退 NCCL。

**涉及知识点**：one-shot/two-shot、CUDA IPC、对称内存、小消息延迟、跨节点 transport、fallback。

**追问**：Custom AR 的阈值应只由 tensor 字节数决定吗？还要考虑哪些变量？

**追问回答**

不能。tensor 字节数只是最显著的维度；最优路径还取决于 TP world size、NVLink/PCIe 拓扑、是否跨节点、dtype 与对齐、buffer 是否注册、CUDA Graph 模式、可用 peer access，以及 Custom AR 支持的 shape/layout。

运行时并发也会改变 crossover：Decode 的小 batch、SM 正在执行的 GEMM、通信 stream 优先级、多个 replica 的链路争用和 rank arrival skew 都可能使同一尺寸选择不同。阈值应来自目标 workload 的多维 sweep，并保留正确 fallback；若只能用单一 size threshold，至少要按拓扑、world size 和执行阶段分别校准。

**评分要点（10 分）**：小消息固定开销得 2 分；解释直接 peer 访问得 2 分；描述 one/two-shot 得 2 分；说明适用边界得 2 分；解释跨节点回退得 2 分。

### 36. Prefill 和 Decode 对 NCCL 的性能需求有什么不同？

**核心回答**

Prefill 一次处理大量 token，TP activation 较大，更关注带宽、channel、协议和计算通信重叠；Decode 每个 active sequence 通常只推进一个或少量 token，单次 tensor 较小但每层都在关键路径上，更关注启动延迟、同步、CUDA Graph 与 P95/P99 抖动。

**深入展开**

Row Parallel AllReduce 的逻辑 payload 近似为：

$$
M=T\times H\times b
$$

$T$ 是本轮实际 scheduled token 数，$H$ 是 hidden size，$b$ 是每元素字节数。Prefill 的 $T$ 可以是几百到几千，容易进入带宽区；Decode 的 $T$ 取决于连续批处理中活跃序列数，在低负载时可能很小。

两阶段的优化方向不同：

| 阶段 | 主要瓶颈 | 重点优化 |
|---|---|---|
| Prefill | 大消息链路带宽、GEMM 与通信暴露 | chunked prefill、RS/AG overlap、拓扑和大包带宽 |
| Decode | 每层 collective 固定延迟、最慢 rank、launch 抖动 | CUDA Graph、Custom AR、fusion、TP placement、P99 |

高并发 Decode 的 $T$ 也可能达到中等消息区间，不能把 Decode 永久等同于“几 KiB”。应从调度日志提取真实 token histogram，再映射到通信 size sweep。

**涉及知识点**：Prefill/Decode、continuous batching、message histogram、带宽区、延迟区、长尾。

**追问**：为什么只用 `nccl-tests` 的 1 GiB AllReduce 带宽无法预测在线 Decode 性能？

**追问回答**

1 GiB AllReduce 主要测量协议进入稳态后的峰值带宽，启动和 rank 到达偏差被巨大 payload 摊薄。在线 Decode 往往每层有一个或多个相互依赖的中小消息 collective，单 token step 要连续支付许多次 launch、同步和网络延迟，属于 latency-sensitive 路径。

此外真实服务还叠加 GEMM、CUDA Graph、Custom AR 阈值、动态 batch、请求调度和多 replica/NIC 争用。正确预测需要用实际 TP tensor size 分布和调用频率做 sweep/replay，并观察 P50/P99、rank skew 与端到端 ITL；大消息 busbw 只能证明峰值数据面能力，不能代表 Decode 尾延迟。

**评分要点（10 分）**：阶段区别得 3 分；写出 payload 公式得 2 分；对应优化方法得 3 分；考虑动态 batching 得 2 分。

### 37. 一个 80 层模型、每层两次 AllReduce，如何估算每个 Decode iteration 的通信次数和逻辑消息量？

**核心回答**

主干 TP AllReduce 次数约为 $80\times2=160$。每次逻辑 payload 为 $M=T\times H\times b$，160 次总逻辑 payload 为 $160M$。Ring TP=$p$ 时，每 rank 的近似发送量为：

$$
V_{send}\approx160\times2\frac{p-1}{p}\times T\times H\times b
$$

接收量相同。还应另外计算 embedding、logits、Sequence Parallel、MoE 或 speculative model 的通信。

**深入展开**

例如 $T=32$、$H=8192$、BF16、TP=8：

$$
M=32\times8192\times2=524288\text{ B}=512\text{ KiB}
$$

160 次的逻辑 payload 为 80 MiB。Ring 系数为 $2(8-1)/8=1.75$，因此每 rank 约发送 140 MiB，并接收相近字节数。这不是总线计数器精确值，因为真实算法可能为 Tree、NVLS 或 Custom AR，channel padding、协议 flag 和融合也会改变线上流量。

通信时间不能只用总字节除以峰值带宽。160 次 collective 之间存在逐层依赖，低负载 Decode 中固定延迟可能占主导：

$$
T_{comm}\approx\sum_{k=1}^{160}(\alpha_k+M_k/B_k)
$$

**涉及知识点**：每层 collective 计数、TP payload、Ring 流量系数、关键路径、逻辑字节与线上字节。

**追问**：为什么把 160 个 512 KiB AllReduce 等价成一个 80 MiB AllReduce 会严重低估时延？

**追问回答**

二者字节总量相同，但固定开销完全不同。160 个 collective 要支付 160 次 host/device launch、rank 匹配、协议启动和层间同步，近似包含 $160\alpha$；一个 80 MiB collective 只支付一次 $\alpha$，并能在大 payload 上充分流水和饱和链路。

更关键的是这 160 次通常位于逐层依赖链：第 $l+1$ 层需要第 $l$ 层归约后的激活，不能任意合并。容量模型应计算 $\sum_i(\alpha_i+M_i/B_i)$ 并加入 arrival skew，而不是先把 $\sum_iM_i$ 合成一个大消息再套带宽公式。

**评分要点（10 分）**：160 次得 2 分；payload 公式得 2 分；Ring 流量计算得 2 分；示例计算正确得 2 分；指出固定延迟和额外通信得 2 分。

### 38. `TP=8, DP=4` 会形成一个 32-rank AllReduce 吗？通信组通常如何划分？

**核心回答**

Dense 模型通常形成 4 个独立的 8-rank TP 组，而不是让每层在 32 个 rank 上 AllReduce。每个 TP 组共同完成一个模型副本，四个 DP 副本处理不同请求。

**深入展开**

一种 rank layout 为：

```text
TP0 = [0..7]
TP1 = [8..15]
TP2 = [16..23]
TP3 = [24..31]

DP group for tp_rank 0 = [0,8,16,24]
DP group for tp_rank 1 = [1,9,17,25]
...
```

Transformer 层的 TP activation AllReduce 只发生在每个 `TPi`。Dense 推理的 DP groups 通常不做逐层激活同步。训练时对应 DP group 会对同一参数 shard 的梯度做 AllReduce；MoE/DP Attention 或在线权重更新也可能使用 DP/EP 方向的 group。

实际 rank 排列可能把 PP、CP 和 EP 维度也编码进去，因此不能从 global rank 连续性猜 group。应打印每种 group 的成员和 group-local rank，并把它映射到节点、GPU 和 NIC 拓扑。

**涉及知识点**：多维 process group、TP/DP rank layout、Dense 推理、group-local rank、拓扑映射。

**追问**：若每个节点 8 卡，应该怎样排列上面四个 TP 组，才能让 TP 不跨节点？

**追问回答**

把每个连续 8-rank TP 组完整放在一个节点最直观：节点 0 为 `[0..7]`，节点 1 为 `[8..15]`，节点 2 为 `[16..23]`，节点 3 为 `[24..31]`。这样每层延迟敏感的 TP collective 留在节点内 NVLink/NVSwitch 域。

对应的 4-rank DP 组按相同 local TP index 跨节点组成，例如 `[0,8,16,24]`、`[1,9,17,25]`。这要求 launcher 的 global-rank 排列与物理 node/local-rank 映射一致；仅在配置里写 `TP=8` 不足以保证 placement，必须打印 group membership 并对照 hostname、CUDA device 和 NCCL topology。

**评分要点（10 分）**：4 个 8-rank TP 组得 3 分；画出 DP group 得 2 分；区分训练与推理得 2 分；考虑 PP/EP 和物理 placement 得 3 分。

### 39. Pipeline Parallel 如何使用 NCCL Send/Recv？为什么 PP 可以减少跨节点 collective，但会引入流水线气泡？

**核心回答**

PP 把不同层放到不同 stage。上游 stage 完成本地层后，把边界激活通过 NCCL Send 发送给下游 stage；下游 Recv 后继续前向。它把每层跨所有 TP rank 的 collective 换成 stage 边界的点对点传输，但不同 microbatch 的依赖会造成 pipeline fill/drain 和负载不均衡气泡。

**深入展开**

对于 stage $s$ 和 $s+1$：

```text
stage s:   layers -> activation[T,H] -> ncclSend
stage s+1: ncclRecv -> layers -> next activation
```

GPU tensor payload 可走 NCCL P2P，shape、dtype 等 metadata 常通过 CPU 控制通道预先交换。所有 stage 必须以一致的 microbatch 顺序发布 Send/Recv，否则可能形成 P2P 死锁。

PP 的优势是每个 stage 只保存一部分层，且跨节点通常只传边界 activation，不要求每个 Transformer 子层都执行跨节点 AllReduce。代价包括：

- 第一个 microbatch 要依次填满所有 stage，最后还要排空。
- 最慢或层数最重的 stage 决定吞吐。
- Decode 依赖逐 token 前向，microbatch 数不足时气泡更显著。
- KV Cache、采样和请求状态要与 stage 映射一致。

**涉及知识点**：Pipeline Parallel、NCCL P2P、microbatch、pipeline bubble、stage balance、metadata channel。

**追问**：为什么 PP 并不自动比跨节点 TP 延迟更低，需要结合 batch 和 stage 数判断？

**追问回答**

PP 把跨节点通信从每层 TP AllReduce 变成 stage 边界的 activation Send/Recv，字节和 collective 次数可能减少；但它引入了 stage 串行依赖、pipeline bubble、microbatch 调度和 stage imbalance。低 batch Decode 很难填满 pipeline，单个 token 仍要依次穿过所有 stage，边界延迟直接叠加到 ITL。

跨节点 TP 虽然每层通信频繁，却让所有 GPU 同时处理同一层，且在某些高速 fabric 上可能有更低的单步关键路径。选择必须对同一模型做 batch/microbatch sweep，比较 TTFT、ITL、吞吐、显存和 P99，并验证 `TP=8, PP=2` 的 stage 是否均衡；不能仅按“跨节点次数更少”下结论。

**评分要点（10 分）**：Send/Recv 数据流得 3 分；解释减少跨层 collective 得 2 分；说明 fill/drain 和负载气泡得 3 分；联系 Decode 得 2 分。

### 40. MoE Expert Parallel 的 Token dispatch/combine 为什么是 All-to-All 语义？

**核心回答**

Router 为每个 token 选择 top-k expert，而 expert 分布在不同 EP rank 上。每个源 rank 都可能向多个目标 rank 发送 token，每个目标 rank 也可能接收所有源 rank 的 token；expert 计算后结果还要按原 token 和 top-k 权重返回并合并，因此是可变长度 All-to-All/All-to-All-v 语义。

**深入展开**

一次 MoE 层包含：

```text
router logits / top-k
  -> 统计每个目标 rank 的 token count
  -> pack / permute
  -> dispatch all-to-all-v
  -> local grouped GEMM by expert
  -> combine all-to-all-v
  -> unpermute + top-k weighted sum
```

通信 shape 由实时路由结果决定，通常不均匀。热点 expert 会同时造成网络 incast、目标 rank token 堆积和 grouped GEMM 负载不平衡。即使网络平均带宽充足，最慢 EP rank 仍决定整层完成时间。

生产优化不仅是换通信库，还包括 capacity、expert placement、冗余 expert、负载均衡、token drop 策略、低精度 dispatch 和通信计算 overlap。

**涉及知识点**：MoE Router、Expert Parallel、All-to-All-v、pack/unpack、grouped GEMM、负载不均衡。

**追问**：为什么随机均匀生成 token destination 的 All-to-All benchmark 可能高估真实 MoE 性能？

**追问回答**

均匀随机流量让每个 expert/NIC 收到近似相同 token 数，buffer 尺寸稳定，链路也容易保持对称满载。这恰好消除了真实 router 常见的热门 expert、burst、top-k 相关性、容量裁剪和某些 rank 的 incast，因此测到的是理想负载均衡上限。

真实 MoE 的慢 rank 还会同时承担更多 pack/unpack 和 expert GEMM，下一次 collective 要等待它完成，通信与计算不均衡相互放大。benchmark 应从线上 trace 重放 destination histogram 和时间相关性，并同时报告均匀、可控 skew 与真实分布；只调总 token 数而不调路由偏斜，无法覆盖 P99。

**评分要点（10 分）**：解释多源多目标得 3 分；描述 dispatch/compute/combine 得 3 分；指出可变长度和热点得 2 分；联系专家负载均衡得 2 分。

### 41. 为什么 MoE 通信可能使用 `AllGather + ReduceScatter`，也可能使用 DeepEP、NIXL 等专用后端？

**核心回答**

AllGather + ReduceScatter 容易复用成熟 collective、支持 CUDA Graph 并简化动态 count 管理，但会把许多与本 rank expert 无关的 token 复制过来。专用 EP backend 则按路由结果只向目标 peer 发送 token，并针对低延迟 Decode、高吞吐 Prefill、RDMA 和量化 dispatch 做优化。

**深入展开**

在 vLLM 当前实现中，默认 `allgather_reducescatter` manager 使用可变长度 AllGather 收集 token/路由信息，再用 ReduceScatter 合并输出。它的优点是语义规则、fallback 稳定，缺点是在 EP 规模大或 token 稀疏时放大流量。

DeepEP 等 backend 通常区分：

- normal/high-throughput 模式：面向 Prefill 大 token batch，强调带宽和计算重叠。
- low-latency 模式：面向 Decode 小 batch，强调固定延迟和 CUDA Graph。
- 节点内 NVLink 与节点间 RDMA 的分层 dispatch。

版本边界很重要：课程基线 NCCL 2.22.3 的 public C API 没有通用 `ncclAllToAll`，PyTorch 和框架常用 grouped Send/Recv 或其他 collective 组合实现 All-to-All 语义；当前新版 NCCL 文档已出现新的 AllToAll API，但框架是否使用仍取决于构建版本和分派实现。

源码参考：[vLLM All2All manager](https://github.com/vllm-project/vllm/blob/3ad85e0de4ee88b8fe9d7fde8a3a024c92cb3c7b/vllm/distributed/device_communicators/all2all.py#L90-L138)。

**涉及知识点**：AllGather/ReduceScatter、All-to-All-v、DeepEP、low-latency/high-throughput、API 版本边界。

**追问**：如何用 token destination histogram 估算 AllGather 方案相对 direct dispatch 的流量放大？

**追问回答**

设 EP world size 为 $P$，$C_{s,d}$ 表示源 rank $s$ 发往目标 expert-owner $d$ 的 token 数，每个 token payload 为 $S$ 字节。direct dispatch 的单向跨 rank 逻辑流量为

$$
V_{direct}=S\sum_{s\ne d}C_{s,d}.
$$

若 AllGather 方案把每个源 rank 的全部 $T_s=\sum_dC_{s,d}$ payload 复制到其余 $P-1$ 个 rank，再由各 rank 本地过滤，则

$$
V_{gather}=S(P-1)\sum_sT_s,
\qquad A=V_{gather}/V_{direct}.
$$

计算时要把 top-k 产生的 token 副本、padding、metadata 以及 combine 返回流量按实现加入。若某方案只 AllGather routing metadata 而非 hidden payload，上式不能直接套用；必须先明确到底被复制的是哪些字节。

**评分要点（10 分）**：两类方案语义得 3 分；分析流量与实现复杂度得 3 分；区分 Prefill/Decode 模式得 2 分；指出 NCCL 版本边界得 2 分。

### 42. PD 分离时，哪些通信通常使用 NCCL，哪些 KV Cache 传输可能使用 NIXL/Mooncake/RDMA？

**核心回答**

Prefill engine 内部和 Decode engine 内部若采用 TP/PP，逐层 activation collective 和 stage P2P 仍通常使用 NCCL。Prefill 到 Decode 的跨实例 KV Cache 传输则由 KV connector 负责，可能使用 NIXL、Mooncake、MoRI、LMCache、RDMA 或存储系统，不应一概称为 NCCL。

**深入展开**

应按通信域拆开：

```text
Prefill TP/PP group       -> NCCL 或框架 Custom AR
Prefill scheduler/control -> ZMQ / RPC / HTTP / TCP
Prefill -> Decode KV      -> NIXL / Mooncake / RDMA / cache backend
Decode TP/PP group        -> NCCL 或框架 Custom AR
训练 -> 推理权重更新      -> NCCL broadcast/send-recv 或专用传输引擎
```

KV transfer 与 collective 的流量形态也不同。它通常是按请求、按 KV block 的单向或多目标传输，需要处理 cache address、block ownership、完成通知、失败重试和生命周期；TP AllReduce 则是固定 group 上高频、严格顺序的同步 collective。

即使 KV connector 使用 RDMA，也不能据此断言它经过 NCCL。NCCL 和 NIXL 都可能使用同一块 NIC/RDMA fabric，进而相互争抢带宽和 queue resources，这属于共享底层资源，不等于共享软件路径。

**涉及知识点**：Disaggregated Prefill、KV connector、NIXL、Mooncake、RDMA、控制面、在线权重更新。

**追问**：PD 分离后 Decode P99 变差，如何判断是 KV transfer 争用 NIC，还是 Decode TP collective 本身变慢？

**追问回答**

先在同一时间轴关联请求、KV transfer、TP collective 和 NIC counter。若开启 KV transfer 后 NCCL collective 的 GPU kernel duration/网络阶段本身增长，并与端口吞吐、拥塞或 pause/ECN counter 峰值重合，且限速 KV 或分离 NIC 后恢复，证据支持共享 fabric 争用。

若 NCCL kernel duration保持正常，但 Decode stream 在 collective 前出现更长空洞，或者请求要等待 KV ready/event，问题更可能在 transfer completion、调度或跨 stream 依赖。实验应包括无 KV、同流量 dummy KV、KV 限速、独立 rail 四组，并比较 collective entry skew 与 kernel duration；只看最终 Decode P99 无法定位是哪一段变慢。

**评分要点（10 分）**：正确划分 engine 内外得 4 分；列出 KV backend 得 2 分；区分 collective 与 block transfer 得 2 分；识别共享 NIC 争用得 2 分。

### 43. 为什么 TP rank 的调度批次必须对齐？某个 rank 少执行一个请求会发生什么？

**核心回答**

TP ranks 共同执行同一个模型副本，每层必须以相同顺序进入对应 collective，并满足约定的 shape/count。某个 rank 因调度差异跳过一个 forward 或提前退出，会让其余 rank 等待缺失的 collective，或者让后续 collective 错位匹配，最终 hang、timeout 或错误结果。

**深入展开**

连续批处理虽然允许请求动态加入和结束，但通常由一个 leader 生成本轮 batch metadata，再同步给所有 TP worker。各 rank 对相同 token position 执行自己的权重 shard，KV Cache 的物理内容可以按 rank 不同，但逻辑 request/token 顺序必须一致。

需要区分：

- **合法分片**：Sequence Parallel 或 EP 允许不同 rank 持有不同 token 数，但 collective 使用显式 sizes，所有 rank 对全局 layout 达成一致。
- **非法分歧**：某个 rank 独立判断 batch 为空而跳过整个层，其他 rank 仍执行 AllReduce。
- **危险分支**：speculative decoding、MoE、LoRA 热切换或 OOM recovery 只在部分 rank 改变执行路径。

健壮系统会把调度 epoch、batch id、token counts 和 collective sequence 写入 trace，并在任一 worker 失败时尽快让整个 TP group 停止，而不是让其他 worker继续排队。

**涉及知识点**：SPMD、continuous batching、batch metadata、变长 collective、sequence number、错误传播。

**追问**：不同 rank 的本地 token count 不同是否一定错误？什么情况下 ReduceScatterV/AllGatherV 可以保证正确？

**追问回答**

不一定。MoE、动态 batching 等场景天然会产生 variable count；只要所有 rank 对全局 `counts/displacements` 有一致认识，发送总量与接收总量守恒，每个 peer 的 dtype、顺序和区间完全匹配，variable-size collective 或等价的 P2P schedule 就能正确执行。

必须注意本文基线 NCCL 2.22.3 的标准 API 没有直接名为 `ncclAllGatherV`/`ncclReduceScatterV` 的调用，框架通常用 padding、分组 Send/Recv 或其他通信库/新能力模拟。不能把不同 count 直接传给固定 count 的 `ncclAllGather`：即使总字节碰巧相同，也会破坏 rank-major layout 和调用匹配。

**评分要点（10 分）**：顺序和 shape 契约得 3 分；解释 leader metadata 得 2 分；区分合法分片与非法分歧得 3 分；提出 trace/错误传播得 2 分。

### 44. 为什么通常建议将 TP 放在 NVLink/NVSwitch 域内，把 DP 或 PP 扩展到节点之间？

**核心回答**

TP collective 位于几乎每个 Transformer 子层的关键路径，Decode 一轮可能执行上百次，任何跨节点延迟和抖动都会逐层累积。NVLink/NVSwitch 的带宽和延迟通常显著优于节点间网络。Dense DP 副本可独立处理请求，PP 只在 stage 边界传 activation，因此更适合跨节点扩展。

**深入展开**

并行维度的通信频率不同：

```text
TP: 每层约 2 次 collective，延迟高度串行
PP: 每个 stage 边界每 microbatch 一次 P2P
Dense DP inference: token-forward 热路径通常无跨副本 collective
EP: 每个 MoE 层 dispatch/combine，常需要高性能节点间网络
```

因此 placement 应先保护最延迟敏感的维度。例如每节点 8 张 NVSwitch GPU 时，`TP=8` 通常放在节点内，再用 DP 扩节点。若模型单节点放不下，可引入 PP；若使用大规模 MoE EP，则需要同时设计 GPU-NIC affinity、多 rail 和 expert placement。

这不是绝对规则。低速节点内 PCIe、超大 Prefill batch、极高带宽网络、模型层数不均衡或 PP 气泡都可能改变结果。最终仍要用端到端 TTFT、ITL、throughput 和通信 trace 验证，而不是只比较理论链路峰值。

**涉及知识点**：parallelism placement、NVLink/NVSwitch、跨节点 TP、PP bubble、EP multi-rail、TTFT/ITL。

**追问**：模型需要 16 卡但每节点 8 卡时，你会如何在 `TP=16` 与 `TP=8, PP=2` 之间做实验决策？

**追问回答**

先建立两条真实执行路径。`TP=16` 会让每层 TP collective 跨节点，但没有 pipeline stage；`TP=8, PP=2` 把 TP 留在节点内，跨节点只传 stage activation，却增加 bubble、stage 边界和负载不均。两者都要确认显存可容纳相同模型、KV cache 和目标 batch。

然后按线上 prefill/decode 长度与 batch 分布做 sweep，记录 TTFT、ITL P50/P99、吞吐、显存、每层 collective/SendRecv timeline、NIC/NVLink 利用率和 stage idle。对 PP 还要尝试 microbatch 与切层位置，对 TP16 要验证跨节点算法/rail。最终按服务 SLO 下的 Pareto 结果选型，而不是只比较一个离线 tokens/s 数字。

**评分要点（10 分）**：按通信频率解释 placement 得 4 分；联系 Decode 累积延迟得 2 分；考虑 EP/PP 代价得 2 分；提出端到端验证得 2 分。

---

## Level 5：故障诊断与专家题

### 45. `nccl-tests` 中 `algbw` 和 `busbw` 分别表示什么？

**核心回答**

`algbw` 是 collective 的逻辑数据量除以时间：

$$
algbw=\frac{S}{t}
$$

`busbw` 再乘 collective 对应的理论流量系数，使不同 rank 数和 collective 更容易与硬件瓶颈带宽比较。对 AllReduce：

$$
busbw=algbw\times2\frac{N-1}{N}
$$

**深入展开**

根据 `nccl-tests` 官方定义：

| Collective | busbw 修正系数 |
|---|---:|
| AllReduce | $2(N-1)/N$ |
| AllGather | $(N-1)/N$ |
| ReduceScatter | $(N-1)/N$ |
| Broadcast | $1$ |
| Reduce | $1$ |

busbw 是根据理想点对点流量模型归一化出来的指标，不是某条 NVLink、PCIe 或 NIC 端口的硬件 counter。它适合判断 collective 对瓶颈链路的利用程度，但不能告诉你具体哪条链路拥塞，也不能跨完全不同的拓扑直接用同一个峰值比较。

看结果时还必须确认：

- `#wrong == 0`，否则性能数据没有意义。
- out-of-place 与 in-place 都符合预期。
- message size、dtype、rank 数、warmup 和 iteration 相同。
- 小消息看 time/latency，大消息再看 algbw/busbw。
- 报告多次独立运行的 median、P95 和 CV，不取一次最快值。

**涉及知识点**：nccl-tests、algbw、busbw、流量归一化、正确性、统计基线。

**追问**：4-rank AllReduce 的 algbw 为 100 GB/s，busbw 是多少？它是否表示每条 NVLink 都跑到 150 GB/s？

**追问回答**

按 `nccl-tests` 对 AllReduce 使用的归一化系数，

$$
busbw=algbw\times\frac{2(N-1)}{N}
=100\times\frac{2\times3}{4}=150\ \text{GB/s}.
$$

它不是任一物理链路的实测值。`busbw` 用算法的理想通信量对耗时做归一化，便于跨 collective/world size 比较；多个 channel、双向链路和多条 NVLink 可能共同承载这些流量。要判断每条 NVLink 是否达到 150 GB/s，必须读取链路级计数器并结合 Ring 映射，不能从这个派生指标反推。

**评分要点（10 分）**：两个定义得 3 分；AllReduce 系数得 2 分；指出 busbw 非硬件 counter 得 2 分；加入正确性和统计要求得 3 分。

### 46. 单机 8 卡 AllReduce 带宽只有预期的一半，你会按什么顺序排查？

**核心回答**

先确认预期和测量有效，再确认物理拓扑与链路状态，然后检查 NCCL 实际选择的 algorithm/protocol/channel/transport，最后才做环境变量干预。排障顺序应从事实到机制，而不是先随机设置 `NCCL_*`。

**深入展开**

建议流程：

1. **验证 benchmark**：确认 `#wrong=0`、size 已进入带宽区、迭代充分、没有并发 workload，比较同版本历史基线。
2. **验证硬件状态**：保存 `nvidia-smi topo -m`、NVLink link status、GPU clocks/P-state、功耗与 ECC/Xid；确认八张卡和预期一致。
3. **拆成 pair matrix**：运行 CUDA P2P bandwidth/latency 和 2-rank nccl-tests，识别是否某个 GPU pair 或某个 baseboard 异常。
4. **验证 NCCL 路径**：保存 `NCCL_DEBUG=INFO` 以及 `GRAPH,P2P,SHM,TUNING` 日志，比较是否从 P2P 退化到 SHM，channel/protocol 是否变化。
5. **检查进程放置**：核对 rank-device、CPU affinity、NUMA、容器 `/dev/shm` 和 IPC；CPU proxy 是否与高负载线程争抢。
6. **查看 timeline**：用 Nsight Systems 判断是 NCCL kernel 自身变长、kernel 之间有 host gap，还是和其他 kernel 争抢 SM/copy engine。
7. **最小 A/B**：一次只禁用 P2P、限制 channel 或强制候选算法，用故障指纹验证假设，完成后恢复自动选择。

“预期的一半”还可能是把单链路峰值、双向聚合峰值和 busbw 混为一谈。第一步必须明确理论上限对应哪个拓扑 cut 和哪个指标。

**涉及知识点**：基线、拓扑 cut、P2P matrix、transport fallback、GPU clock、NUMA、Nsight Systems。

**追问**：禁用 P2P 后带宽几乎不变，这能支持哪些假设，又不能证明什么？

**追问回答**

它支持的假设包括：基线可能本来就没走 P2P、测试尺寸受其他瓶颈限制、SHM fallback 在该尺寸上性能接近，或者目标 GPU 对之间没有被本次 collective 大量使用。首先应确认环境变量生效，并比较 NCCL transport 日志和 topology plan。

它不能证明“P2P 一定没用”或“SHM 与 P2P 永远一样快”。短测试可能被 launch latency 主导，多 channel 可能绕开被禁路径，缓存/warmup 也会掩盖差异。需要覆盖 latency 与 bandwidth 尺寸、不同 GPU pair，并同时观察 CPU/PCIe traffic，才能把性能不敏感转化为路径结论。

**评分要点（10 分）**：先验证 benchmark 得 2 分；硬件和 pair matrix 得 2 分；路径日志得 2 分；NUMA/容器/timeline 得 2 分；坚持单变量证伪得 2 分。

### 47. 单机正常、跨节点 NCCL hang，如何区分 collective 不匹配、网络配置、RDMA 和 rank 崩溃？

**核心回答**

先按时间点分类：初始化前、communicator connect、首个 collective、运行一段时间后。然后收集所有 rank 的带时间戳日志和进程状态，用最小 2-rank 跨节点 `nccl-tests` 将应用控制流问题与基础网络问题分离。

**深入展开**

诊断矩阵如下：

| 现象 | 优先怀疑 | 关键证据 |
|---|---|---|
| bootstrap 尚未完成 | 地址、接口、DNS、防火墙、端口 | INIT/BOOTSTRAP 日志、TCP 连通性 |
| topology 完成但 NET connect 卡住 | HCA、GID、RoCE、Verbs、plugin | NET 日志、`ibv_devinfo`、link/GID |
| nccl-tests 也跨机失败 | 集群数据路径 | 2-rank/多 rank nccl-tests、RDMA 基准 |
| nccl-tests 正常，业务首轮卡住 | group/order/shape 不一致 | ProcessGroup sequence、op、shape trace |
| 运行后随机卡住 | rank crash、Xid、拥塞、链路错误 | 进程退出码、dmesg、GPU/NIC counters |

必须从所有 rank 找第一个异常时间点。最后打印 timeout 的 rank 往往只是受害者。若一个 rank OOM 后退出，其余 rank 会停在下一次 collective；若不同 rank op sequence 分歧，所有进程可能仍然存活。

生产环境应配置有界 timeout、ProcessGroup flight recorder 和 coordinated teardown。调试变量要区分 `NCCL_*` 与 PyTorch 的 `TORCH_NCCL_*`，后者控制 watchdog、trace buffer 和错误处理，不是 NCCL transport 参数。

**涉及知识点**：bootstrap、IB/RoCE、GID、collective mismatch、rank crash、watchdog、flight recorder。

**追问**：为什么单独执行 `ib_write_bw` 成功仍不能证明 NCCL 多 rank AllReduce 正常？

**追问回答**

`ib_write_bw` 通常验证一对 endpoint、一个或少量 QP、特定 memory type 和明确指定的 HCA/port。NCCL AllReduce 还要完成多 rank bootstrap、拓扑选路、多个 channel/QP、双向流量、GPU buffer 注册/GDR、collective 顺序和所有 peer 的一致进度。

因此 pairwise verbs 成功只能排除部分底层故障，无法覆盖某个 rank 选错 NIC、某条 rail/GID 不通、多对多拥塞、GDR 失败或 collective mismatch。应测试所有相关 GPU-NIC/rank pair，再运行与生产 world size、buffer 类型和并发模式一致的 NCCL collective，并核对每个 rank 的日志。

**评分要点（10 分）**：按阶段分类得 2 分；最小 nccl-tests 隔离得 2 分；网络/RDMA 检查得 2 分；sequence 与 rank liveness 得 2 分；全 rank 时间线和有界 teardown 得 2 分。

### 48. NCCL 日志中的 `INIT`、`GRAPH`、`NET`、`COLL` 分别能帮助确认什么？

**核心回答**

`INIT` 观察 communicator、rank、device、bootstrap 与初始化完成状态；`GRAPH` 观察 topology、Ring/Tree/channel 构造；`NET` 观察网络 plugin、NIC、连接和 GDR 相关选择；`COLL` 观察 collective 任务信息。算法、协议和估时选择还应结合 `TUNING`。具体字段随 NCCL 版本变化。

**深入展开**

常见使用方式：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,COLL,TUNING
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log
```

不同子系统解决的问题不同：

- `INIT`：各 rank 是否加入同一 communicator、绑定哪个 CUDA/NVML device、初始化停在哪一步。
- `GRAPH`：检测到哪些 GPU/NIC 路径、生成多少 channel、Ring/Tree 邻居是谁。
- `NET`：加载哪个 plugin、使用哪些 HCA/interface、连接方式及部分 GDR 信息。
- `COLL`：collective 类型、count、datatype、opCount 等执行信息，详细度取决于日志级别和版本。
- `TUNING`：候选 algorithm/protocol 的估时与最终选择，适合解释 size crossover。

日志证明的是软件选择和控制路径，不直接证明实际链路达到某带宽，也不替代 GPU timeline。最可靠的结论通常需要日志、Nsight 和性能曲线三者一致。

**涉及知识点**：NCCL debug、subsystem、日志聚合、algorithm/protocol selection、证据强度。

**追问**：GRAPH 日志显示 Ring 经过 NVLink，为什么还需要 Nsight 和 P2P 性能对照？

**追问回答**

GRAPH 日志描述的是 NCCL 构造出的计划和逻辑边，不等于每条物理链路在运行时达到预期带宽，也不能显示 rank arrival skew、kernel 空洞或某个 channel 的动态瓶颈。计划经过 NVLink，只能证明 NCCL 认为这条路径可用且选择了它。

Nsight 用来确认目标 collective 实际何时执行、kernel/CTA 配置和等待发生在哪里；P2P 微基准与链路计数器则验证对应 GPU pair 的真实单向/双向性能。三者结合才能区分“拓扑选择正确但硬件/运行时慢”和“通信计划本身选错”，单份 GRAPH 日志不足以闭环。

**评分要点（10 分）**：四个子系统各 1 分；补充 TUNING 得 1 分；给出可用日志配置得 2 分；解释日志证据边界得 3 分。

### 49. 为什么平均延迟正常但 P99 很差？如何分析网络、rank straggler、CUDA Stream 和 CPU 抖动？

**核心回答**

collective 完成时间由最慢 rank 和最慢路径决定。偶发网络拥塞、某个 GPU 降频、CPU proxy 被抢占、stream 前序 kernel 变长、内存注册 miss、动态算法切换或某个请求产生异常大 batch，都可能只影响少量样本，因此平均值正常而 P99 恶化。

**深入展开**

应将一次业务延迟拆成：

$$
T_{observed}=T_{queue}+T_{launch-gap}+T_{stream-wait}+T_{collective}+T_{consumer-wait}
$$

分析步骤：

1. 按 request/iteration、collective sequence、rank 记录分布，不只记录全局平均。
2. 对齐所有 rank 时间线，确认长尾前谁最晚进入 collective，谁最后完成。
3. 用 Nsight 区分 host launch gap、前序 GEMM 延长和 NCCL kernel 延长。
4. 关联 GPU clocks、功耗、thermal、Xid，NIC queue/drop/ECN/PFC，CPU run queue、IRQ 和 context switch。
5. 按 message size、batch token 数、algorithm/protocol 分桶，排除 workload composition 变化。
6. 注入可控 CPU/网络/GPU 干扰建立故障指纹，再与生产长尾比较。

仅在 rank 0 记录 elapsed time 会把“rank 0 提交晚”和“对端 rank 慢”混在一起。对 NCCL 本身的时间还需使用正确 CUDA Event 边界。

**涉及知识点**：tail latency、straggler、最慢 rank、queueing、ECN/PFC、GPU clock、per-rank trace。

**追问**：所有 NCCL kernel duration 都正常，但 kernel 之间出现毫秒级空洞，应优先检查什么？

**追问回答**

优先检查 NCCL 之外的提交与依赖路径：各 rank 是否同时到达 collective、CPU 线程是否被抢占、Python/调度器是否停顿、前序 GEMM/event 是否迟迟未完成，以及 ProcessGroup/通信 stream 的 launch 是否被阻塞。kernel duration 正常说明已进入设备执行后的通信大致正常，空洞多半发生在 enqueue 前或 stream wait 上。

在 Nsight 中把 NVTX、CPU thread、CUDA API、application stream 和 NCCL stream 对齐，并比较所有 rank 的 collective entry timestamp。若只有一个 rank 晚到，应向它的前序计算/调度追溯；若所有 rank 都晚提交，则检查 host 调度、GC、锁和 graph miss。不要先用 `NCCL_ALGO` 调参来解决根本没有在 kernel 内发生的问题。

**评分要点（10 分）**：最慢 rank 模型得 2 分；拆分延迟组成得 2 分；跨 rank timeline 得 2 分；关联 GPU/NIC/CPU 指标得 2 分；按 workload 分桶和注入实验得 2 分。

### 50. 容器中运行 NCCL 需要重点检查哪些资源？

**核心回答**

需要检查 GPU device 和驱动库、进程间 IPC 与 `/dev/shm`、memlock、RDMA device、网络 namespace/interface、sysfs/PCI 拓扑可见性以及容器内 CUDA/NCCL 与宿主驱动 ABI 兼容性。

**深入展开**

检查清单：

- `/dev/nvidia*`、CUDA driver library 和 GPU runtime 是否正确注入。
- 多进程是否共享所需 IPC namespace；`/dev/shm` 容量是否足够，是否被意外设为很小。
- `ulimit -l`、IPC lock capability 和 pinned/registered memory 限制。
- `/dev/infiniband/*`、RDMA userspace library、NIC plugin 和对应网卡是否在容器中可见。
- host network 或 CNI 是否允许 bootstrap/interface/端口通信，DNS 和 hostname 是否稳定。
- `/sys/bus/pci`、NUMA 和 GPU/NIC topology 信息是否可读，避免 NCCL 得到残缺拓扑。
- 宿主 driver、container CUDA、NCCL、OFED/rdma-core 与 plugin 版本是否兼容。
- 安全策略、seccomp、SELinux/AppArmor 是否阻止 IPC、共享内存或设备访问。

`--ipc=host` 可以简化共享内存可见性，但会扩大容器间故障和安全域，不是唯一正确方案。更严格的部署可以显式配置足够 shm、同 Pod IPC 和必要 capability。

**涉及知识点**：容器 GPU 注入、IPC namespace、shared memory、memlock、RDMA device、CNI、ABI。

**追问**：为什么容器内 `nvidia-smi` 正常仍不能证明 NCCL P2P/SHM/RDMA 都可用？

**追问回答**

`nvidia-smi` 正常只证明容器能访问 NVIDIA 驱动管理接口和可见 GPU。P2P 还依赖 peer-access、GPU 拓扑与 CUDA IPC；SHM 依赖 IPC namespace、`/dev/shm` mount/容量；RDMA 依赖 `/dev/infiniband`、sysfs 拓扑、memlock、verbs library、NIC/GID 和 GDR 支持。

这些资源可以彼此独立地缺失。例如 GPU kernel 正常但容器未映射 HCA，会退到 Socket；P2P 正常但 `/dev/shm` 过小，会在 fallback 时失败。验收脚本应分别运行 CUDA P2P、强制 SHM 和跨节点 RDMA/GDR 测试，并保存 transport 日志，不能把 `nvidia-smi` 当作通信栈健康检查。

**评分要点（10 分）**：GPU/驱动得 2 分；IPC/shm/memlock 得 3 分；RDMA/network 得 2 分；topology/版本/安全策略得 2 分；指出 ipc=host 权衡得 1 分。

### 51. 一个 rank OOM 或进程退出后，其他 rank 为什么可能表现为 NCCL hang？如何设计超时、abort 和故障传播？

**核心回答**

collective 需要所有参与 rank 推进相同操作。一个 rank 在进入 collective 前 OOM、崩溃或失去 GPU context 后，其余 rank 已提交的 GPU/网络工作可能一直等待缺失 peer，直到 transport 报错或上层 watchdog 超时。因此表面症状在健康 rank 上是 NCCL hang，根因却是另一个 rank 更早失败。

**深入展开**

健壮设计应包含：

1. 每个 worker 的异常、退出码、CUDA error 和 heartbeat 立即上报控制面。
2. ProcessGroup 设置有界 timeout，并启用适合版本的 trace/flight recorder。
3. 任一 rank 判定不可恢复后，协调其他 rank 停止提交新 collective。
4. 调用 communicator abort/销毁，并终止可能持有失效 CUDA context 的进程。
5. 由作业控制器从 checkpoint 或完整模型状态重建整个通信组。

不能假设捕获 Python OOM 后清空 cache 就能让该 rank 跳过当前 batch、其他 rank 继续。communicator 的 collective sequence 已经分歧，其他 GPU 上还可能有在途 kernel。某些异步 CUDA 错误发生后，最稳妥边界是退出进程并重新初始化。

新版 NCCL 增加了更多 communicator shrink、RAS 或容错接口，但能否局部恢复仍取决于上层状态分片、请求重放和框架支持，不能把库级能力等价成业务透明容错。

**涉及知识点**：rank failure、async error、watchdog、communicator abort、coordinated teardown、checkpoint/restart。

**追问**：为什么只杀死 timeout 的 rank 往往无法恢复，甚至可能杀错根因进程？

**追问回答**

collective 是多 rank 协议，报 timeout 的 rank 常常只是最先超过 watchdog 阈值的等待者；真正根因可能是另一个 rank 更早 OOM、CUDA fault、退出或走错 collective。此时 communicator 中已有未完成工作，各 rank 的 opCount、stream 和 buffer 状态可能已经分叉。

只重启一个进程无法让旧 communicator 自动重新达成一致，其余 rank 还可能继续等待旧连接。正确流程是找到所有 rank 中时间最早的异常，协调 abort 整个受影响 communicator/worker group，并由上层从明确 checkpoint 或请求边界整体重建。排序日志要用统一时钟和 communicator/opCount，不能按“谁最后打印 timeout”认定根因。

**评分要点（10 分）**：解释缺失参与者得 3 分；指出错误可能异步传播得 2 分；给出完整 teardown 流程得 3 分；说明库级与业务级容错边界得 2 分。

### 52. 如何证明性能问题发生在 NCCL，而不是 GEMM、Attention、调度器或 CPU launch？

**核心回答**

需要把端到端时间分解到 CUDA timeline，并用匹配真实 shape/topology 的通信 microbenchmark 和单变量 A/B 建立因果关系。仅看到 GPU utilization 低或 NCCL 出现在调用栈中，不足以证明 NCCL 是瓶颈。

**深入展开**

证据链应包括：

1. **模型级定位**：为 Attention、GEMM、collective、scheduler 加 NVTX，测量每层 exposed 时间和 overlap。
2. **GPU timeline**：检查 NCCL/custom AR kernel duration、前后 stream wait 和 host launch gap。
3. **shape 对齐**：从线上提取 `(collective, group size, dtype, bytes)` histogram，在同拓扑运行 nccl-tests 或最小 PyTorch benchmark。
4. **并行 A/B**：比较 TP=1 与 TP>1 时不能只看总吞吐，还要保持单卡计算工作量可解释；禁用 Custom AR 时只改变通信 backend。
5. **计算验证**：对 GEMM/Attention 使用算子 profiler，检查实际 FLOPS、shape、quantization 和 kernel fallback。
6. **调度验证**：CPU profiler 和 timeline 判断 GPU 空洞是否来自 batch 准备、Python/C++ scheduler 或进程抢占。

优化价值应看 exposed communication，而不是 NCCL kernel 总时长。若 NCCL 与计算完全重叠，即使 kernel 占用时间长，也未必在关键路径；反之，一个很短但串行出现 160 次的 AllReduce 可能主导 ITL。

**涉及知识点**：critical path、exposed communication、NVTX、Nsight Systems、microbenchmark、因果 A/B。

**追问**：为什么把 NCCL kernel 全部删除后端到端变快，仍不一定代表真实系统可以获得同样收益？

**追问回答**

删除 collective 通常也删除了它建立的数据依赖：后续 rank 可能直接读取未归约的局部 partial，得到错误结果；timeline 还可能因此重排，让本应等待远端的计算提前执行。测到的是一个不满足模型语义的理论上界，不是可部署优化。

要证明收益可实现，必须用保持结果等价的替代方案，例如 ReduceScatter 后消费分片、Custom AR、通信计算 overlap 或并行布局重写，并做数值正确性与压力测试。只有替代路径保持相同输入输出契约时，它与原路径的端到端差值才是可兑现的优化空间。

**评分要点（10 分）**：提出 timeline 分解得 2 分；shape 对齐 microbenchmark 得 2 分；单变量 A/B 得 2 分；区分 exposed 与 overlapped 得 2 分；同时验证计算/调度得 2 分。

### 53. 从 `ncclAllReduce()` 开始，如何追踪 NCCL 源码中的 enqueue、拓扑选择、transport、proxy 和 device kernel？

**核心回答**

应按生命周期追踪，而不是只在 `collectives.cc` 找 Ring 循环：communicator 初始化阶段已经完成 topology graph 和 transport connector；每次 API 调用构造 `ncclInfo` 并追加 task；GroupEnd 将 task 调度成 plan/work，上传或嵌入 kernel arguments，启动设备 kernel，并把网络/proxy 操作交给 CPU progress 路径。

**深入展开**

以 NCCL 2.22.3 为基线，Host 执行链为：

```text
src/collectives.cc::ncclAllReduce
  -> 构造 ncclInfo
  -> src/enqueue.cc::ncclEnqueueCheck
  -> taskAppend
  -> src/group.cc::ncclGroupEndInternal
  -> ncclLaunchPrepare
  -> scheduleCollTasksToPlan / scheduleP2pTasksToPlan
  -> finishPlan
  -> uploadWork
  -> ncclLaunchKernel
  -> ncclLaunchKernelAfter_NoCuda
```

communicator 初始化和连接链为：

```text
bootstrap / init
  -> graph/topo.cc + graph/paths.cc
  -> graph/search.cc + graph/connect.cc
  -> transport.cc::selectTransport
  -> P2P / SHM / NET / CollNet canConnect, setup, connect
  -> channel send/recv connector ready
```

设备侧根据 `coll + redop + dtype + algo + proto` 选择生成的 device function：

```text
device/common.h::ncclKernelMain
  -> RunWorkBatch
  -> RunWorkColl<AllReduce,...>
  -> device/all_reduce.h 的 Ring/Tree/NVLS/CollNet 分支
  -> primitives 的 send/recv/reduce/copy
```

NET 路径还会创建 proxy operation。GPU 通过 step/flag 与通信 buffer 协作，CPU proxy 调用 network plugin 推进 send/recv completion。源码阅读必须把日志和实验对应回来：API/COLL 日志验证参数，TUNING 验证算法协议，GRAPH/NET 验证连接，Nsight 验证 kernel 与 block，proxy 日志验证网络 progress。

**涉及知识点**：`ncclInfo`、task、plan、work batch、graph search、transport vtable、proxy、device primitive。

**追问**：为什么 `ncclAllReduce()` 的函数体里看不到 Ring step，并不表示 NCCL 源码没有开源算法实现？

**追问回答**

host API 的职责主要是校验参数、构造 `Info/Task`、加入 communicator 队列并触发 enqueue，它不会在 CPU 函数里逐步执行 GPU Ring。algorithm/protocol/channel 在后续规划阶段确定，再由模板化 device kernel 读取 work descriptor 执行。

Ring 的 send/recv/reduce step 通常位于设备端 collective 与 primitive 模板中，例如 AllReduce 的 Ring 实现调用 `Primitives` 的 `send`、`recvReduceSend`、`directRecvReduceCopySend` 等组合。阅读源码要沿“API -> enqueue/plan -> work descriptor -> kernel dispatch -> collective template -> transport primitive”追踪；只停在公开 C API wrapper，自然看不到数据面循环。

**评分要点（10 分）**：API 到 enqueue 得 2 分；task/plan/launch 得 2 分；初始化 graph/transport 得 2 分；device kernel 与 proxy 得 2 分；能设计日志/trace 对照得 2 分。

### 54. 设计一组实验，比较 TP=2/4/8、不同消息大小、单机/跨节点、NCCL/Custom AR。

**核心回答**

实验必须同时提供正确性、性能、路径和系统证据，并一次只改变一个主要变量。不能把 TP、batch、模型 shape、节点数和 backend 全部一起改变后，把差异归因给 NCCL。

**深入展开**

建议矩阵：

| 维度 | 取值 |
|---|---|
| TP size | 2、4、8 |
| topology | 单机 NVLink/NVSwitch、单机 PCIe、跨节点 |
| bytes | 1 KiB 到 1 GiB 对数扫描，加真实 Decode/Prefill size |
| dtype | FP16/BF16/FP32，按业务选择 |
| backend | 自动、禁用 Custom AR 的 NCCL、Custom AR |
| mode | eager、CUDA Graph；out-of-place/in-place 按能力 |

每个 case 固定：GPU 集合与 rank mapping、CPU/NUMA affinity、CUDA/NCCL/框架 commit、GPU clocks、warmup、iteration、并发 workload、全部 `NCCL_*`/`TORCH_NCCL_*` 和 batch shape。

证据要求：

- **正确性**：rank 编码输入，与 FP32/CPU reference 比较；报告最大误差和 `#wrong`。
- **性能**：至少 10 次独立进程 cycle，报告 median、P95、CV、小包 latency、大包 algbw/busbw。
- **路径**：NCCL GRAPH/TUNING/NET 日志，框架 backend 分派日志，Nsight kernel/API timeline。
- **系统**：GPU/NIC/NUMA 拓扑、clock、功耗、NIC counters、manifest。

分析时先回答可证伪问题：Custom AR 在什么 size crossover 前优于 NCCL？TP 增大后是 $\alpha$ 项还是带宽项恶化？跨节点首次出现的额外固定延迟是多少？Prefill/Decode 的实际 message histogram 落在哪些区间？

若 CV 超过 5%，应先报告不稳定并定位噪声，不应给出“快 3.7%”之类精确结论。

**涉及知识点**：实验设计、控制变量、正确性 oracle、参数 sweep、统计、路径证据、crossover。

**追问**：如何避免 Custom AR 只支持某些 shape，导致对比样本选择偏差？

**追问回答**

先从真实 workload trace 得到完整的 `(size, shape, dtype, world size, phase)` 分布，再标记 Custom AR 支持域。对 kernel 公平比较，应在两条实现都支持的交集上做成对 A/B；同时必须单独报告交集覆盖了线上多少调用、多少字节和多少关键路径时间，不能悄悄丢掉 unsupported 样本。

系统级评估则按真实频率重放全部样本：支持的走 Custom AR，不支持的走 NCCL fallback，并把 dispatch/判定开销计入。再分别给出 supported、fallback 和 overall 指标及正确性结果。这样既能回答“实现本身谁快”，也能回答“上线后整体能快多少”，避免只挑有利 shape 得出偏置结论。

**评分要点（10 分）**：完整参数矩阵得 2 分；控制变量得 2 分；四类证据各 1 分；独立 cycle 和统计得 1 分；提出可证伪假设与边界得 1 分。

---

## 总分与岗位能力判断

54 题满分 540 分。建议先逐题独立作答，再根据评分点打分：

| 总分 | 参考能力层级 | 主要缺口或能力 |
|---:|---|---|
| 0-180 | 基础不足 | collective、rank、异步语义尚未建立 |
| 181-300 | 初级 Infra | 能使用 NCCL，但算法、拓扑和证据链较弱 |
| 301-400 | 中级 Infra | 能完成常见部署、benchmark 与故障定位 |
| 401-480 | 高级 Infra | 能分析框架通信关键路径并治理生产性能 |
| 481-540 | 专家候选 | 能推导、读源码、设计实验并说明结论边界 |

总分不能掩盖关键短板。以下题目属于硬门槛：

- 题 4、6：collective 契约和异步完成语义。
- 题 14、17、18：TP 与 DP 的计算和通信本质。
- 题 30、31：顺序与参数不匹配，这是生产 hang 的高频根因。
- 题 45、46：benchmark 指标和分层归因。
- 题 47、51：跨节点故障和 rank failure 的安全处置。
- 题 52、54：能否用证据而不是经验猜测建立因果关系。

高级或专家候选人不一定记得每个环境变量名称，但必须知道到哪里核对版本文档、如何证明实际路径，以及什么证据足以支持结论。

## 推荐面试组合

45 分钟筛选可以选：1、4、6、9、14、25、33、47。

90 分钟高级面试可以选：3、11、16、23、28、30、36、40、45、49、52，并从 53 或 54 中选择一道深入题。

面试官应要求候选人画数据布局、推导至少一个公式，并对一个生产故障给出按顺序执行的证据收集方案。只问 API 名词无法区分“使用过 NCCL”和“能负责大模型通信基础设施”。

## 官方资料与课程延伸

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)
- [NCCL 2.22.3-1 固定源码](https://github.com/NVIDIA/nccl/tree/v2.22.3-1)
- [nccl-tests PERFORMANCE：algbw 与 busbw](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [CUDA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [PyTorch Distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch ProcessGroupNCCL 环境变量](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)
- [课程 05：Collective 契约、内存布局与逐元素证明](/posts/nccl-collectives/)
- [课程 12：Ring AllReduce 的逐步推导](/posts/nccl-ring-allreduce-steps/)
- [课程 17：Bootstrap 与 communicator 初始化状态机](/posts/nccl-bootstrap-communicator-init-state-machine/)
- [课程 29：ProcessGroupNCCL、Stream、Event 与 Work](/posts/nccl-processgroupnccl-stream-event-work/)
- [课程 33：Collective mismatch、hang 与 rank crash](/posts/nccl-collective-mismatch-hang-rank-crash-root-cause/)
- [课程 36：AllReduce 端到端源码与实验闭环](/posts/nccl-allreduce-end-to-end-capstone/)
