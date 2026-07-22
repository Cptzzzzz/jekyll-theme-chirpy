---
title: 'CUDA Features 4.1：统一内存（Unified Memory）'
author: Cptz
date: 2026-07-22 08:01:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, unified-memory, hmm, memory]
math: true
pin: true
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.1 Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)，以中文技术转述保留 API 语义、支持条件和限制。示例为基于原理重新编写的最小代码。

## CUDA Features 中文系列目录

本系列将 CUDA Programming Guide 13.3 的第四章整理为 **20 篇中文技术文章**。建议先按顺序建立内存、提交、片上流水和资源隔离四条主线，再针对工作中的 API 回查单篇。

| 章节 | 主题 | 核心问题 |
|---|---|---|
| 4.1 | [Unified Memory](/posts/cuda-features-01-unified-memory/) | 统一地址、分页迁移、一致性与 oversubscription |
| 4.2 | [CUDA Graphs](/posts/cuda-features-02-cuda-graphs/) | 低开销重复提交、更新、条件节点与设备端发射 |
| 4.3 | [Stream-Ordered Memory Allocator](/posts/cuda-features-03-stream-ordered-memory-allocator/) | 异步 allocation 生命周期与 memory pool |
| 4.4 | [Cooperative Groups](/posts/cuda-features-04-cooperative-groups/) | 显式线程组、同步与 collective |
| 4.5 | [Programmatic Dependent Launch](/posts/cuda-features-05-programmatic-dependent-launch/) | 将“允许发射”和“允许消费”分阶段 |
| 4.6 | [Green Contexts](/posts/cuda-features-06-green-contexts/) | SM/work queue 分区与干扰控制 |
| 4.7 | [Lazy Loading](/posts/cuda-features-07-lazy-loading/) | Module 按需加载与冷启动成本 |
| 4.8 | [Error Log Management](/posts/cuda-features-08-error-log-management/) | Driver 结构化错误诊断 |
| 4.9 | [Asynchronous Barriers](/posts/cuda-features-09-asynchronous-barriers/) | Arrival、transaction 与 phase 状态机 |
| 4.10 | [Pipelines](/posts/cuda-features-10-pipelines/) | 多 stage 生产者-消费者流水 |
| 4.11 | [Asynchronous Data Copies](/posts/cuda-features-11-asynchronous-data-copies/) | LDGSTS、TMA 与 STAS |
| 4.12 | [Cluster Launch Control](/posts/cuda-features-12-cluster-launch-control/) | 取消未启动任务并执行工作窃取 |
| 4.13 | [L2 Cache Control](/posts/cuda-features-13-l2-cache-control/) | Persisting access 与 set-aside cache |
| 4.14 | [Memory Synchronization Domains](/posts/cuda-features-14-memory-synchronization-domains/) | 隔离 fence 等待的无关流量 |
| 4.15 | [Interprocess Communication](/posts/cuda-features-15-interprocess-communication/) | 跨进程共享 GPU allocation 与同步 |
| 4.16 | [Virtual Memory Management](/posts/cuda-features-16-virtual-memory-management/) | VA、物理内存、映射与权限解耦 |
| 4.17 | [Extended GPU Memory](/posts/cuda-features-17-extended-gpu-memory/) | 经 NVLink fabric 访问系统级内存 |
| 4.18 | [CUDA Dynamic Parallelism](/posts/cuda-features-18-dynamic-parallelism/) | GPU 端动态产生 child grid |
| 4.19 | [CUDA Interoperability](/posts/cuda-features-19-graphics-interop/) | 与 Vulkan、D3D、OpenGL、NvSci 共享资源 |
| 4.20 | [Driver Entry Point Access](/posts/cuda-features-20-driver-entry-point-access/) | 动态符号、ABI 版本与兼容回退 |

完整英文原文可在 [CUDA Programming Guide Part 4](https://docs.nvidia.com/cuda/cuda-programming-guide/part4.html) 或[官方 PDF](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)中核对。

Unified Memory 的目标是让 CPU 和 GPU 使用同一虚拟地址访问数据。它消除的是手工维护两套指针和副本的必要性，不是物理层的数据移动成本：驱动仍可能迁移页面、建立远端映射、处理缺页并维护一致性。

## 4.1.1 完整 Unified Memory 支持

完整支持意味着 GPU 不只可以访问 `cudaMallocManaged` 的分配，还能访问系统分配器、栈、全局变量和文件映射产生的 pageable memory。Linux 上这依赖 HMM（Heterogeneous Memory Management）。文档给出的 HMM 组合包括：

- 受支持的 Linux 内核，文档列出的修复版本为 6.1.24+、6.2.11+ 或 6.3+。
- 计算能力 7.5 或更高的 GPU。
- 535 或更高版本的驱动。
- NVIDIA Open GPU Kernel Modules；专有 kernel module 不提供这里所述的 HMM 完整能力。

即使满足这些条件，也应查询设备属性而不是仅按产品名判断。虚拟机、容器、WSL、IOMMU 和多 GPU 拓扑会改变实际能力。

### 4.1.1.1 深入示例：普通系统分配

在完整支持的系统上，普通 `malloc` 指针可被 kernel 使用：

```cpp
float* data = static_cast<float*>(malloc(n * sizeof(float)));
initialize_on_cpu(data, n);

kernel<<<grid, block>>>(data, n);
cudaDeviceSynchronize();

consume_on_cpu(data, n);
free(data);
```

这里的关键仍是同步。CPU 在 kernel 未完成时读取或写入同一区域，是否有定义取决于平台一致性能力和所用同步原语。一个地址能从 GPU 解引用，并不自动消除数据竞争。

#### 4.1.1.1.1 文件后备 Unified Memory

文件映射页面在完整支持系统上也可被 GPU 访问。因此可以把大文件 `mmap` 到地址空间，再让 GPU 直接使用相同指针。GPU 修改最终如何反映到文件，仍受 OS 的 page cache、映射模式、刷新和文件系统语义控制。

```text
文件 -> CPU 虚拟地址映射 -> GPU 缺页/映射或迁移 -> kernel 访问
```

应用必须自己处理：

- 文件长度与映射范围是否覆盖 GPU 访问区间；
- `MAP_SHARED`/`MAP_PRIVATE` 的写入语义；
- GPU 完成后何时 `msync`/`munmap`；
- 存储 IO、页面迁移和计算之间是否存在抖动。

#### 4.1.1.1.2 Unified Memory 与 IPC

不能把 `cudaMallocManaged` 指针直接交给传统 `cudaIpcGetMemHandle`。需要跨进程共享时，原文给出的思路是让进程映射同一个文件，或使用 CUDA VMM / 可导出的 memory pool 等具有明确句柄语义的机制。

传递裸指针也不够：不同进程的虚拟地址和页表独立，必须先共享可导入的底层对象或同一文件映射，并在每个进程建立映射。

## 4.1.1.2 性能调优

Unified Memory 性能的主要变量是：页面大小、页面当前驻留位置、访问者、访问局部性、一致性方式和迁移能否与计算重叠。

### 4.1.1.2.1 分页与页大小

GPU 访问尚未建立映射的页会触发 fault，驱动随后决定迁移或映射。较大的页面有三方面好处：更少的缺页事件、更少的页表项、更大的 TLB 覆盖；代价是只访问少量字节时搬运更多无用数据，并可能扩大 CPU/GPU 所有权来回切换的粒度。

因此：

- 连续扫描和密集访问通常受益于大页或批量预取。
- 稀疏随机访问可能因过度迁移而受损。
- 同一页被 CPU 和 GPU 交替写，会形成 page ping-pong。
- 页迁移时间应从 kernel 时间中单独测量，否则容易误判算子性能。

### 4.1.1.2.2 主机直接访问 Unified Memory

在具有硬件一致性的系统上，CPU 可以直接访问 GPU 驻留的托管内存，而不一定先把页面迁回系统内存。应查询 `cudaDevAttrDirectManagedMemAccessFromHost`。该能力可能降低一次性 CPU 访问的迁移成本，但远端访问带宽和延迟未必优于迁移后本地访问。

适合直接访问还是先迁移，取决于访问规模：少量控制字段更适合远端直接访问，大范围重复扫描更可能适合预取到 CPU。

### 4.1.1.2.3 Host Native Atomics

`cudaDevAttrHostNativeAtomicSupported` 表示 GPU 对主机内存执行的某些原子操作可由系统原生支持。属性为真也不代表所有类型、宽度和 memory order 都有效，仍要遵守相应 CUDA 原子 API 的作用域与支持矩阵。

若没有原生支持，驱动或硬件不能凭空把任意主机地址变成高性能系统原子。跨 CPU/GPU 的 lock-free 数据结构必须针对目标平台验证。

### 4.1.1.2.4 原子访问与同步原语

普通 load/store 的可见性、原子性和顺序是三个不同问题。barrier 只约束参与的 GPU 线程；stream synchronization 能建立主机观察 GPU 完成的边界；system-scope atomic/fence 才用于设备与主机或其他设备之间的相应作用域。

一个安全阶段通常是：

```text
CPU 写输入
  -> 发射 kernel（提交建立先后）
  -> GPU 读写
  -> stream/event/device 同步
  -> CPU 读取输出
```

若 CPU 和 GPU 真正并发读写，则必须使用原文允许的原子/一致性组合，不能用 `volatile` 代替同步。

### 4.1.1.2.5 Memcpy/Memset 行为

`cudaMemcpy`、`cudaMemcpyAsync` 和 `cudaMemset` 可以作用于 unified memory。运行时可根据指针属性判断内存位置，因此常见代码可使用 `cudaMemcpyDefault`。同步版本可能引入主机等待；异步版本按指定 stream 排序。

复制不等于永久 placement。后续处理器访问仍可能引发迁移。把 memcpy 当作“预取”时，应明确自己究竟需要复制语义还是驻留提示。

### 4.1.1.2.6 Unified Memory 分配器概览

相关来源可分为：

| 来源 | 地址可见性 | 典型适用范围 |
|---|---|---|
| `cudaMallocManaged` | CPU/GPU 同一托管地址 | 跨平台 managed memory |
| 系统 `malloc/new` | 完整 HMM 系统上 GPU 可访问 | 复用普通 CPU 数据结构 |
| 栈/静态存储 | 完整 HMM 条件下可访问 | 小型已有对象，需谨慎生命周期 |
| `mmap` 文件 | 完整 HMM 条件下可访问 | 文件后备大数据 |
| pinned/mapped host memory | 显式注册并映射 | 稳定的主机驻留与直接访问 |

选择分配器会影响可移植性、缺页方式、注册成本和 IPC 能力。不能因为最终拿到的是统一地址，就认为这些来源完全等价。

### 4.1.1.2.7 Access Counter Migration

支持 access counter 的系统能观察处理器对远端页面的频繁访问，并据此迁移页面，减少长期远端访问。它是运行时自适应机制，不是应用局部性的替代品；短暂或来回变化的模式仍可能在策略收敛前付出较高成本。

### 4.1.1.2.8 避免 CPU 频繁写 GPU 驻留内存

CPU 对 GPU 驻留页持续小写入，可能触发迁移、失效或昂贵的远端一致性流量。常用改法是：

- 把 CPU 更新聚合到单独控制区；
- 使用双缓冲，在阶段边界交换所有权；
- CPU 写完后一次预取到 GPU；
- 对只读共享数据使用 read-mostly hint；
- 用事件明确通知，而不是轮询大数据页。

### 4.1.1.2.9 利用系统内存异步访问

GPU 可在访问某些系统内存的同时继续执行其他 warp。要让这种延迟隐藏真正生效，需要足够并发、合并访问和可推进的独立工作。完全依赖每次远端 load 的串行 pointer chasing 无法靠“异步”自动变快。

## 4.1.2 仅支持 Managed Memory 的设备

这类设备支持 `cudaMallocManaged`，但不支持让 GPU 普遍访问任意系统分配。程序应使用设备属性区分能力，并限制自己只把合法指针传入 kernel。

`concurrentManagedAccess` 等属性还决定 CPU/GPU 是否能在 kernel 运行期间并发访问 managed memory。属性不支持时，主机在 GPU 活跃期间触碰相关 managed allocation 可能导致错误或未定义行为，必须先同步。

## 4.1.3 Windows、WSL 与 Tegra

这些平台不能直接套用完整 Linux HMM 规则。原文将其单独列出，是因为 managed memory 的 placement、并发访问和多 GPU 行为可能退化到 CUDA 6.x 模式，或受到平台特定限制。

### 4.1.3.1 多 GPU

创建 managed allocation 时，运行时会考虑当前进程可见的 GPU 是否都能访问该内存以及它们之间的 peer 关系。某些组合会让内存位于系统内存并由 GPU 通过 PCIe 访问，性能明显不同于 GPU 本地驻留。

`CUDA_VISIBLE_DEVICES` 不只改变设备编号，也可能改变运行时看到的 peer 集合和 placement 决策。多 GPU 程序应在最终部署拓扑上测试，而不是只在单卡环境验证。

### 4.1.3.2 一致性与并发

`cudaDevAttrConcurrentManagedAccess` 用来识别 GPU 执行时主机能否并发访问 managed memory。即使允许并发，程序仍需消除数据竞争。不能把“系统不报错”和“读取值有确定顺序”混为一谈。

### 4.1.3.3 Stream 关联的 Unified Memory

`cudaStreamAttachMemAsync` 可把 managed allocation 与 stream 关联，并表达 global、host 或 single-stream 可见性。它允许运行时以更细粒度判断访问者和生命周期。

关联操作本身按 stream 排序。内存何时真正变为新的关联状态，取决于该操作何时执行，而不是 API 在主机返回的瞬间。

#### 4.1.3.3.1 Stream Callback

callback 在前序 stream 工作完成后由主机执行。对 stream 关联内存，callback 所处时刻会影响主机访问是否合法。回调不应调用禁止的 CUDA API，也不应假设其他 stream 已完成。

#### 4.1.3.3.2 更细粒度控制

把不同 allocation 关联到不同 stream，可让多个独立任务各自管理数据，而不必每次同步整个设备。只有确实共享的对象才需要通过事件或全局关联建立跨 stream 顺序。

#### 4.1.3.3.3 多线程主机示例的含义

主机线程与 CUDA stream 并非自动一一绑定。多线程程序必须明确每个线程使用哪个 stream、哪个 allocation 与哪条 stream 关联，以及对象销毁前由谁同步。否则一个线程可能在另一个线程的 GPU 工作尚未结束时访问或释放内存。

#### 4.1.3.3.4 关联内存的数据移动

stream association 是访问和并发提示，不等于立即执行一次确定方向的复制。数据的实际迁移仍由运行时根据后续访问与平台决定。需要确定 placement 时，应使用 prefetch；需要值复制时，应使用 memcpy。

## 4.1.4 性能提示

### 4.1.4.1 数据预取

`cudaMemPrefetchAsync` 把一段内存的预期位置设为 CPU 或某个 GPU，操作在 stream 中排序：

```cpp
cudaMemPrefetchAsync(data, bytes, gpu, stream);
kernel<<<grid, block, 0, stream>>>(data, n);

cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
consume_on_cpu(data);
```

预取适合已知下一阶段处理器的工作流。过早预取会占用容量，过晚预取无法隐藏延迟，错误方向则增加额外搬运。

### 4.1.4.2 数据使用提示

`cudaMemAdvise` 可表达：

- preferred location：希望长期驻留的位置；
- accessed by：预先建立指定处理器的访问条件；
- read mostly：主要只读，允许运行时维护只读副本等优化；
- unset 操作：撤销之前的提示。

提示不是强制命令，也不改变数据竞争规则。`read mostly` 数据若被频繁写入，优化可能被撤销并产生额外一致性成本。

### 4.1.4.3 Memory Discarding

当旧内容不再需要时，可以通过对应 discard 操作告知运行时无需保留原值。这样后续完整覆写不必先迁移旧页面。discard 后读取未重新定义的数据没有意义，应用必须保证新的写入覆盖消费范围。

### 4.1.4.4 查询属性

`cudaMemRangeGetAttribute(s)` 可查询 read-mostly、preferred location、accessed-by 和 last prefetch location 等信息。查询得到的是管理状态与提示结果，不是每一时刻所有物理页面的精确性能画像。

### 4.1.4.5 GPU 显存过量订阅

Unified Memory 允许工作集大于显存，但超额部分需要迁移或远端访问。可运行性提高不等于吞吐不变：若每轮都扫描远超显存的数据，页面会不断淘汰和重新迁入。

调优应记录 fault 数、迁移字节、PCIe/NVLink 流量和 kernel stall，并通过分块、预取、热点/冷数据拆分、read-mostly 以及更小并发工作集控制抖动。

**总结：Unified Memory 统一的是虚拟地址和数据管理模型；要获得稳定性能，仍需把页面驻留、访问阶段、一致性和拓扑当成显式系统设计问题。**

[下一篇：4.2 CUDA Graphs](/posts/cuda-features-02-cuda-graphs/)
