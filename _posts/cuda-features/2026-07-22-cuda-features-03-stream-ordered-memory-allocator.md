---
title: 'CUDA Features 4.3：流顺序内存分配器'
author: Cptz
date: 2026-07-22 08:03:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, memory-pool, cuda-malloc-async, allocator]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.3 Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)。

## 4.3.1 引言

传统 `cudaMalloc`/`cudaFree` 在主机上管理设备内存生命周期，释放还可能等待设备工作。流顺序分配器将分配和释放变成 stream 中的操作，让内存生命周期能由事件与 stream 依赖描述，并用 memory pool 缓存物理页。

## 4.3.2 内存管理

### 4.3.2.1 分配

`cudaMallocAsync(&ptr, size, stream)` 立即向主机返回指针，但其“可供设备工作使用”的时间点由 stream 中分配节点决定。同一 stream 中排在它之后的工作天然有序；其他 stream 必须等待事件或建立其他显式依赖。

```cpp
cudaMallocAsync(&p, bytes, producer);
produce<<<g, b, 0, producer>>>(p);

cudaEventRecord(ready, producer);
cudaStreamWaitEvent(consumer, ready);
consume<<<g, b, 0, consumer>>>(p);
```

API 的异步性质不是“后台 malloc 线程”，而是把 allocation lifetime 纳入设备执行顺序。分配失败仍可能在 API 调用处返回，也可能由后续异步错误报告机制体现，所有 CUDA 返回码都要检查。

### 4.3.2.2 释放

`cudaFreeAsync(ptr, stream)` 排入释放操作。释放必须在所有用户之后有序：

```cpp
cudaEventRecord(consumed, consumer);
cudaStreamWaitEvent(reclaimer, consumed);
cudaFreeAsync(p, reclaimer);
```

不能因为物理页可能仍在 pool 中，就在 free 之后继续查询、复制或解引用旧指针。其逻辑生命周期已经结束，地址稍后也可能被另一分配重用。

## 4.3.3 Memory Pools

pool 同时管理虚拟地址、已提交物理页、未使用但保留的页以及访问权限。缓存使下一次分配常能避免昂贵的 OS/driver 分配路径。

### 4.3.3.1 默认池

每个支持异步分配的设备有默认 pool，`cudaMallocAsync` 从当前设备的默认 pool 分配。也可用 `cudaDeviceGetDefaultMemPool` 取得句柄，设置 release threshold、读取统计或 trim。

应用可以设置当前 pool，使后续普通 `cudaMallocAsync` 使用另一个池。默认 pool 的存在不代表跨 context、跨 device 任意共享。

### 4.3.3.2 显式池

`cudaMemPoolCreate` 接收分配位置、allocation type 和 handle type 等属性。显式 pool 适合：

- 为不同模型/租户分离统计和回收策略；
- 配置 IPC 可导出句柄；
- 为 peer GPU 配置访问权限；
- 在确定时刻 destroy 整个分配域。

销毁 pool 前必须保证来自它的 allocation 不再被使用。pool handle 生命周期和其中各指针的 stream 生命周期是两层责任。

### 4.3.3.3 多 GPU 可访问性

设备具有 `cudaDeviceCanAccessPeer` 能力，不等于它能自动访问另一设备 pool 的分配。需要用 `cudaMemPoolSetAccess` 为目标设备设置 read/write 等访问权限。

默认 pool 的 peer access 状态也不同于 legacy `cudaDeviceEnablePeerAccess` 的全部行为，代码应按 pool API 明确配置并查询。

### 4.3.3.4 为 IPC 启用 Memory Pool

#### 4.3.3.4.1 创建和共享 Pool

创建 pool 时选择平台支持的可导出 handle type。导出的是 OS 可传递句柄，应用再用 Unix domain socket、Windows handle duplication 等机制把它交给另一进程。接收方导入后得到本进程的 pool handle。

#### 4.3.3.4.2 导入进程设置访问

导入 pool 后，接收进程仍要为本地目标 GPU 配置访问权限。共享底层对象并不会绕过设备 peer、拓扑或权限限制。

#### 4.3.3.4.3 共享具体 Allocation

除共享 pool 外，生产进程还要把某个 allocation 的导出数据传给接收进程；接收方再从导入 pool 获得可用指针。这个 allocation token 与裸虚拟地址不同，可以在另一地址空间重建引用。

#### 4.3.3.4.4 导出 Pool 限制

可导出性在创建时决定，不能把普通 pool 事后变成任意 handle type。导出进程必须维持底层 pool/allocation 所需生命周期，且 OS handle 自身也要正确关闭。

#### 4.3.3.4.5 导入 Pool 限制

导入的 pool 不能像本地 pool 一样执行所有管理操作；创建新分配、修改特定属性或重新导出可能受限。接收方只应依赖文档明确允许的操作。

## 4.3.4 最佳实践与调优

### 4.3.4.1 查询支持

使用 `cudaDevAttrMemoryPoolsSupported` 查询设备支持；涉及可导出句柄、peer 或特定 location 时继续查询相应能力。只按 CUDA 版本判断不足以覆盖平台差异。

### 4.3.4.2 物理页缓存

free 后，pool 通常把物理内存保留供复用。同步点可能触发把超出 release threshold 的未使用页归还系统。阈值为 0 偏向降低驻留占用，高阈值偏向减少后续 page allocation。

`cudaMemPoolTrimTo` 是回收请求，不应理解为强制把占用精确降到某个字节数；仍在使用、粒度对齐和驱动状态都会影响结果。

### 4.3.4.3 资源统计

pool 可报告 reserved/current、reserved/high、used/current、used/high 等指标：

- used 表示分配生命周期内正在使用的容量；
- reserved 表示 pool 从系统保留的物理容量；
- high watermark 用于观察历史峰值，可按 API 重置。

区分 used 与 reserved 能解释“业务已 free，但 `nvidia-smi` 显存未下降”的正常缓存行为。

### 4.3.4.4 内存复用策略

#### Follow Event Dependencies

启用后，分配器可沿显式 event dependency 判断另一 stream 的 free 是否先于当前 allocation，从而安全复用。事件是应用已经表达的依赖，不额外改变并行性。

#### Allow Opportunistic

若运行时发现旧 free 实际已经完成，可机会式复用，即使没有可追踪的显式依赖。它降低峰值，但行为会随时序变化，内存高水位可能不够确定。

#### Allow Internal Dependencies

分配器可以在 stream 之间插入内部依赖以实现复用。这可能减少显存，却把原本并行的工作变成有序。延迟敏感应用要同时测吞吐、尾延迟和峰值显存。

#### 禁用策略

关闭上述策略可获得更直接的依赖模型，但需要更多物理内存。正确性不能依赖某项复用策略是否碰巧开启；策略只应影响何时复用，不应修复本来缺失的用户数据依赖。

### 4.3.4.5 同步 API 的动作

device、stream、event 同步等操作为 pool 提供已完成工作的观察点，并可能触发按 release threshold 回收缓存页。一次同步因此可能同时承担等待和内存管理成本，benchmark 应避免把它误计入下一个 kernel。

## 4.3.5 补充事项

### 4.3.5.1 `cudaMemcpyAsync` 的当前 Context/Device

异步 pool 指针可携带所属设备信息，但某些 memcpy 行为仍对调用线程当前 context/device 敏感。多线程、多 context 代码应在调用前设置正确设备并检查指针所属。

### 4.3.5.2 `cudaPointerGetAttributes`

释放后查询旧指针没有合法生命周期保证；不能用属性查询来判断地址“是否还碰巧没被复用”。在有效期内，pointer attributes 可识别设备、内存类型和相关句柄。

### 4.3.5.3 `cudaGraphAddMemsetNode`

图中的 memset 参数和目标指针必须符合 graph memory node 的依赖与有效期。不能让 memset 节点位于 alloc 之前或 free 之后，也不能在更新时突破节点支持的尺寸/布局约束。

### 4.3.5.4 指针属性

异步分配的指针在逻辑上属于指定 pool/location。不要用数值地址范围推断内存种类，使用正式属性 API。

### 4.3.5.5 CPU 虚拟内存

pool 可能预留较大的虚拟地址范围。Linux 进程的 `ulimit -v` 等限制过低时，即便设备物理显存足够也可能失败。虚拟地址预留和实际物理提交要分开诊断。

**总结：流顺序分配器把 allocation lifetime 变成依赖图的一部分，并用 pool 在显存占用、并行度和分配开销之间提供可调节的复用策略。**

[上一篇：4.2 CUDA Graphs](/posts/cuda-features-02-cuda-graphs/) · [下一篇：4.4 Cooperative Groups](/posts/cuda-features-04-cooperative-groups/)
