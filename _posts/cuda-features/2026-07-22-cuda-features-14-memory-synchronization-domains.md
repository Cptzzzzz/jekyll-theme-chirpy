---
title: 'CUDA Features 4.14：Memory Synchronization Domains'
author: Cptz
date: 2026-07-22 08:14:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, memory-model, fence, hopper]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.14 Memory Synchronization Domains](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)。

## 4.14.1 Memory Fence Interference

CUDA 内存模型具有 cumulativity。若 GPU 线程 2 通过 device-scope acquire 看到了线程 1 的写，再用 system-scope release 通知 CPU，那么 CPU 在 acquire 后不仅要看到线程 2 自己的写，也要看到经线程 2 累积进来的线程 1 写。

硬件执行 fence 时不知道哪些在途写是源代码同步关系真正要求的，哪些只是恰好可见，因此往往保守等待更大范围的事务。结果是 fence/flush 被无关慢事务拖延。

典型例子是：本地计算 kernel 访问 HBM，通信 kernel 同时通过 NVLink/PCIe 写远端。计算 kernel 结束时的隐式同步可能等待通信流量，即便其下游只依赖本地结果。这就是 memory fence interference。

fence 不只来自显式 `__threadfence*` 或 atomic；kernel/task 边界为实现 synchronizes-with 也可能隐式产生。

## 4.14.2 用 Domain 隔离流量

计算能力 9.0+、CUDA 12.0+ 为每次 kernel launch 标记 memory synchronization domain。写和 fence 带 domain ID，device-scope fence 只排序同一 domain 的写。把远端通信 kernel 放入另一个 domain，可让本地计算 fence 不再保守等待它的远端事务。

代价是应用必须遵守新规则：同一 GPU 上跨 domain 的排序需要 system-scope fence。原因正是 cumulativity；domain A 的 device fence 不再包含 domain B 的写。

```text
local domain: 计算、本地 HBM 流量、device-scope 顺序
remote domain: NCCL/远端 NVLink/PCIe 流量
跨 domain 传递数据: system-scope 顺序
```

domain 不改变 kernel 合法访问哪些地址，也不是内存保护或带宽分区。它只改变 fence 要覆盖的事务集合。

## 4.14.3 CUDA 中的用法

`cudaLaunchAttributeMemSyncDomain` 选择逻辑 domain：`Default` 或意在放置远端流量的 `Remote`。`cudaLaunchAttributeMemSyncDomainMap` 将逻辑 domain 映射到物理 domain ID，使框架可以和其他组件协调映射。

```cpp
cudaLaunchConfig_t cfg{};
cfg.stream = stream;
cfg.gridDim = grid;
cfg.blockDim = block;

cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeMemSyncDomain;
attr.val.memSyncDomain = cudaLaunchMemSyncDomainRemote;
cfg.attrs = &attr;
cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, comm_kernel, args...);
```

用 `cudaDevAttrMemSyncDomainCount` 查询数量。Hopper 提供 4 个物理 domain；旧于计算能力 9.0 的设备报告 1，因此相同代码可运行但不会得到隔离收益。默认 kernel 位于 domain 0，保持旧代码兼容。

框架集成时要避免两个库各自随意修改逻辑到物理映射。较好的做法是由顶层 runtime 统一分配 domain，通信库使用 Remote，计算使用 Default，并审计所有跨域事件/原子的数据可见性。

**总结：Memory Synchronization Domains 用显式 domain 缩小 fence 覆盖的在途事务，减少本地计算被远端通信 flush 拖累，但跨域正确性必须升级到 system-scope 同步。**

[上一篇：4.13 L2 Cache Control](/posts/cuda-features-13-l2-cache-control/) · [下一篇：4.15 Interprocess Communication](/posts/cuda-features-15-interprocess-communication/)
