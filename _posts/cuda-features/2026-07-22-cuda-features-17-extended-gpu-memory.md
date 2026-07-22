---
title: 'CUDA Features 4.17：Extended GPU Memory（EGM）'
author: Cptz
date: 2026-07-22 08:17:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, egm, nvlink-c2c, numa, grace-hopper]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.17 Extended GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html)。

Extended GPU Memory（EGM）面向集成 CPU-GPU NVIDIA 系统。它把 CPU attached memory 与 GPU HBM 放入可经 NVLink-C2C、GPU NVLink 和 NVSwitch fabric 高带宽访问的资源体系，让 GPU thread 能访问本地或远端 socket 的系统内存。

## 4.17.1 预备知识

### 4.17.1.1 平台拓扑

原章列出三类：

1. 单节点单 GPU：Arm CPU、CPU memory 与 GPU 通过 NVLink-C2C 相连。
2. 单节点多 GPU：多个 CPU/memory socket 与 GPU 经 NVLink 网络连接。
3. 多节点多 GPU：多个上述节点通过 NVLink-based network 连接。

本地访问通常走 NVLink-C2C；远端路径还会经过 GPU NVLink/NVSwitch，部分拓扑再经过远端 C2C。EGM 的价值来自保证流量走高带宽 NVLink 路径。

原文特别指出：用 cgroup 限制可见 device 会阻断 EGM 路由并造成性能问题，应使用 `CUDA_VISIBLE_DEVICES` 控制应用可见设备。部署容器时这条约束需要与编排平台的设备隔离方式共同验证。

### 4.17.1.2 Socket / NUMA ID

EGM 用 OS 分配的 NUMA node ID 表示 host memory location。它不是 CUDA device ordinal。可查询离某个 GPU 最近的 host NUMA：

```cpp
int numa_id;
cuDeviceGetAttribute(&numa_id,
                     CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                     device_ordinal);
```

在多 socket 系统上，allocation 的 NUMA ID决定物理 memory 所属 host node，也影响实际访问路径。

### 4.17.1.3 支持 EGM 的分配器

当前原章使用两条路径：Driver VMM 的 `cuMemCreate` 与 Runtime stream-ordered pool 的 `cudaMemPoolCreate`。普通 host/system allocation 在单节点单 GPU 的 C2C 系统上也能受益于本地高带宽访问，但显式 EGM 共享与多 socket placement 使用上述 location 类型。

### 4.17.1.4 现有 API 的扩展

VMM 增加 `CU_MEM_LOCATION_TYPE_HOST_NUMA`，memory pool 使用 `cudaMemLocationTypeHostNuma`。应用负责在每个所需 socket/设备建立 physical allocation、VA mapping 与 access rights。

多节点场景还要结合 CUDA IPC/VMM fabric handle；EGM 不替代跨进程句柄交换协议。

## 4.17.2 使用 EGM

### 4.17.2.1 单节点单 GPU

CPU 与 GPU 已由高带宽 C2C 紧密连接时，现有 CUDA host allocator 和系统内存可用于本地访问。仍要考虑 page size、访问局部性和 CPU/GPU 一致性，而不能只因链路快就忽略 NUMA placement。

### 4.17.2.2 单节点多 GPU

需要明确选定 host NUMA node，并给一个或多个 GPU 访问。两种 API 路径最终都表达同一件事：memory physically belongs to host NUMA N，GPU D 对它有权限。

#### 4.17.2.2.1 VMM

```cpp
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numa_id;
```

查询 granularity 并向上取整后 `cuMemCreate`；然后 `cuMemAddressReserve`、`cuMemMap`。最后用 `cuMemSetAccess` 同时为 host NUMA location 和目标 GPU device 设置 read-write。未 set access 的映射被访问会 fault。

#### 4.17.2.2.2 Memory Pool

创建 pool 时设置 pinned allocation 和 `cudaMemLocationTypeHostNuma`，再用 `cudaMemPoolSetAccess` 给 peer GPU 权限。把 pool 设为目标 device 当前 pool 后，可用异步分配接口取得 EGM pointer。

EGM 使用 2 MiB 页面。超大、低局部性 allocation 仍可能产生较多 TLB miss；高带宽链路不会消除地址转换成本。

### 4.17.2.3 多节点多 GPU

物理 allocation 仍设置 `HOST_NUMA` 与 NUMA ID，同时 requested handle type 使用 `CU_MEM_HANDLE_TYPE_FABRIC`。导出 fabric handle，经 MPI/TCP 等传给远端进程，再导入、reserve、map、set access。

```text
Node A host NUMA physical memory
  -> fabric handle
  -> Node B import
  -> Node B process-local VA
  -> remote GPU access over NVLink fabric
```

各 rank 必须协调对象生命周期。导出端释放 backing 前，所有远端 kernel、mapping 和 imported handle 都要结束。访问性能还取决于实际路由、NUMA placement、page size 与并发流量，应逐路径测本地/远端带宽和延迟。

**总结：EGM 用 host NUMA placement 加 VMM/pool 权限把系统内存纳入 NVLink fabric，但高性能依赖正确的 socket 选择、路由可见性、页行为和跨进程生命周期。**

[上一篇：4.16 Virtual Memory Management](/posts/cuda-features-16-virtual-memory-management/) · [下一篇：4.18 CUDA Dynamic Parallelism](/posts/cuda-features-18-dynamic-parallelism/)
