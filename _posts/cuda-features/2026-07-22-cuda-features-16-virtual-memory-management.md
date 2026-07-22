---
title: 'CUDA Features 4.16：Virtual Memory Management'
author: Cptz
date: 2026-07-22 08:16:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, vmm, virtual-memory, multicast, ipc]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.16 Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)。VMM 是 Driver API 的低层能力，示例省略了逐调用错误检查，实际代码不可省略。

`cudaMalloc` 同时给出地址和背后的物理显存；传统 peer enable 还可能把过去与未来的全部 allocation 都映射给 peer。VMM 将虚拟地址、物理 allocation、映射和访问权限拆开，让应用只共享指定区域，甚至用来自不同 GPU 的物理页拼成连续 VA。

## 4.16.1 预备概念

### 4.16.1.1 定义

- Virtual Address（VA）：kernel 使用的地址，不表示物理位置。
- Physical Allocation：由 `cuMemCreate` 创建的 backing store，尚未 map 时没有可解引用 pointer。
- Mapping：把 VA 区间关联到 physical allocation 的某一 offset。
- Access Rights：指定某个 device/host NUMA location 对映射是 none、read 或 read-write。
- Shareable Handle：跨进程/OS 实例标识底层 allocation 的 POSIX FD、Win32 或 fabric handle。
- Unicast：一个地址访问一个具体 physical allocation。
- Multicast：一个映射代表多个设备副本，配合 multimem 指令完成一对多或归约访问。

区分 address 与 memory 后，可以预留很大 VA，按需逐段映射物理页；也可保持 pointer 不变而 unmap/remap backing，适合可增长 allocator、跨 GPU 参数映射和通信 buffer。

### 4.16.1.2 查询支持

先查询 `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`。fabric handle 还需查询 `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`，multicast 查询 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`。

fabric memory 的多用户安全依赖 IMEX domain/channel。导出者与导入者必须有权访问同一个 `/dev/nvidia-caps-imex-channels/channelN`。channel 由管理员配置；缺少它不是应用重试能解决的普通 allocation failure。

旧设备可不支持 VMM，设备支持 VMM 也不表示每种 handle、compression 或 multicast 都支持。按 feature 分别查询。

## 4.16.2 API 总览

最小单设备 VMM 生命周期：

```text
计算 allocation granularity
  -> cuMemCreate（物理）
  -> cuMemAddressReserve（虚拟）
  -> cuMemMap（VA -> physical）
  -> cuMemSetAccess（授权）
  -> kernel 使用 pointer
  -> cuMemUnmap
  -> cuMemAddressFree
  -> cuMemRelease
```

每个 size/offset 都应按查询到的 granularity 对齐。VA reserve 的 alignment 传 0 表示选择适当默认值，但映射大小仍必须满足约束。

释放顺序不能颠倒：仍在运行的 kernel 不得访问即将 unmap 的 VA；VA 必须全部 unmap 后才能 free；generic allocation handle 的引用要在所有映射/导入使用结束后 release。

## 4.16.3 Unicast Memory Sharing

unicast 跨进程/设备的五阶段是 allocate/export、share/import、reserve/map、set access、release。

### 4.16.3.1 Allocate and Export

构造 `CUmemAllocationProp` 时指定 allocation type、location 和 requested handle type。先查最小 granularity，再向上取整：

```cpp
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.location.id = device;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

size_t granularity;
cuMemGetAllocationGranularity(&granularity, &prop,
                              CU_MEM_ALLOC_GRANULARITY_MINIMUM);
size_t alloc_bytes = round_up(bytes, granularity);

CUmemGenericAllocationHandle physical;
cuMemCreate(&physical, alloc_bytes, &prop, 0);
```

`cuMemCreate` 只创建物理对象，没有 CPU/GPU mapping。随后按 handle type 导出：单节点 Linux 通常为 POSIX FD，Windows 为 Win32 handle，多节点为 fabric handle。

OS handle 是系统资源，也有 close/ownership 规则。CUDA generic handle 与导出的 FD/HANDLE 不是同一个生命周期概念。

### 4.16.3.2 Share and Import

CUDA 不负责传输 OS handle。Linux FD 通常通过 Unix domain socket 的 `SCM_RIGHTS`，Windows handle 需要适当 duplicate 到目标进程，fabric handle 可作为固定数据通过 MPI/其他通信发送。

接收方用 `cuMemImportFromShareableHandle` 得到本地 `CUmemGenericAllocationHandle`。POSIX/Win32 句柄限单节点相应 OS 实例；fabric handle 可跨 NVLink fabric 节点，前提是 IMEX 权限和平台支持。

不要通过普通 socket 只发送 FD 整数值；FD number 仅在进程自己的 descriptor table 内有意义，必须用 OS 的 descriptor passing。

### 4.16.3.3 Reserve and Map

`cuMemAddressReserve` 只取得连续 VA，不提交物理内存：

```cpp
CUdeviceptr va;
cuMemAddressReserve(&va, alloc_bytes, 0, 0, 0);
cuMemMap(va, alloc_bytes, 0, physical, 0);
```

一段大 VA 可以按 offset 映射多个 device 的 physical chunks，从 kernel 视角仍是连续 pointer。反过来，同一 physical allocation 也可在受支持条件下映射到多个 VA。

已映射区不能再次重叠 map；要替换 backing，先确保无访问，再 `cuMemUnmap`。reserve 本身不授予访问权，map 后仍需 set access。

### 4.16.3.4 Access Rights

`cuMemSetAccess` 为一个或多个 `CUmemAccessDesc` 指定权限：

```cpp
CUmemAccessDesc desc{};
desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
desc.location.id = consumer_device;
desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(va, alloc_bytes, &desc, 1);
```

访问权限是逐映射范围、逐 location 的。能导入 physical handle 不等于能访问；没有权限的 kernel 解引用会 fault。只读 consumer 应按 API 支持设置最小权限。

VMM 的价值之一正是只给必要 allocation 授权，而不是像全局 peer enable 那样为所有分配建立 peer mapping。

### 4.16.3.5 释放

每个进程先停止并同步所有使用，然后：

1. `cuMemUnmap` 本进程 VA。
2. `cuMemAddressFree` 释放 reserve。
3. `cuMemRelease` 释放导入/创建的 generic handle 引用。
4. 关闭仍由进程持有的 OS handle。

导出者不能在导入者尚在访问时销毁底层对象。跨进程控制协议必须有“最后一个使用者”或明确所有权。

## 4.16.4 Multicast Memory Sharing

multicast object 将多个设备的 physical memory 绑定为一个逻辑对象。multimem PTX 指令可把写/原子操作广播到副本，或从多个副本做 reduce load，适合 NVLink/NVSwitch 集体通信原语。

### 4.16.4.1 创建 Multicast Object

`cuMulticastCreate` 的属性包含参与设备数、size 和 handle type。先用 `cuMulticastGetGranularity` 对齐 size。新对象最初既没有 device team，也没有 physical backing。

单节点可选 POSIX FD 等 handle，多节点通常使用 fabric handle。每个参与进程需导入同一个 multicast object 身份。

### 4.16.4.2 添加设备

所有参与进程对各自控制的 device 调用 `cuMulticastAddDevice`。必须先完成整个 team 的 device 加入，再绑定物理内存。进程间需要 barrier 协议，避免一个 rank 过早 bind。

### 4.16.4.3 绑定内存

每个 device 用 `cuMemCreate` 创建满足要求的 backing，再以 `cuMulticastBindMem` 绑定 multicast offset 到 physical allocation offset。所有 replica 的范围和粒度要一致。

绑定并不会自动创建应用 VA；后续仍 reserve/map multicast handle 和相应 unicast mapping，供不同类型指令使用。

### 4.16.4.4 使用 Multicast Mapping

普通 C++ load/store 不能表达所有 multicast 语义，需要 libcu++/inline PTX 的 `multimem` 指令。例如：

- `multimem.red`：对所有 replica 执行原子归约写；
- `multimem.ld_reduce`：读取各 replica 并归约；
- 相应 load/store/atomic 变体按 scope 与 memory order 工作。

同一 physical memory 可能有 multicast 与 unicast 两个虚拟别名。跨这两种 proxy 传递数据时，需要 `fence.proxy.alias` 等 alias proxy fence；普通线程 fence 不一定建立不同访问 proxy 间顺序。

原章的 all-reduce/barrier 示例还强调代际混淆：下一轮某个快 rank 的 arrival 不能错误解除上一轮慢 rank 的 barrier，外层要有迭代间同步或 generation counter。

## 4.16.5 高级配置

### 4.16.5.1 Memory Type

`CUmemAllocationProp` 的 type/location 决定 physical allocation 的性质与位置。当前典型 VMM device backing 使用 pinned allocation type，location 可为 device 或 EGM 的 host NUMA。支持组合必须通过 granularity/capability 查询，不要按枚举存在就认为可用。

### 4.16.5.2 Compressible Memory

支持设备可以请求 generic compression type，使 allocation 有机会使用硬件压缩以降低显存/带宽压力。compression 是请求并可能回退，创建后应查询实际属性；数据分布不可压缩时也不能保证收益。

压缩 allocation 的 mapping、sharing 和访问仍遵守普通 VMM 生命周期。不要用压缩能力改变数据格式或正确性假设。

### 4.16.5.3 Virtual Aliasing

同一 physical allocation 可映射到多个虚拟地址，便于 ring buffer 连续窗口、不同权限 view 或 multicast/unicast 双视图。但两个 VA 指向同一物理位置，会产生 alias：

- 写一个别名会改变另一个看到的值；
- cache/proxy 可见性要按 CUDA memory model 建立；
- 同时写多个别名仍是数据竞争；
- unmap 某个 alias 不会自动销毁其他 mapping。

### 4.16.5.4 OS 特定 IPC Handle

Linux POSIX FD 通过 `SCM_RIGHTS` 传递，不能只发送整数；Windows HANDLE 需使用目标进程权限和 `DuplicateHandle`；fabric handle 不依赖单一 OS descriptor table，但依赖平台 fabric 与 IMEX 安全域。

handle type 是 allocation 创建属性的一部分。导出/导入两端必须使用相同类型，并正确管理原生句柄关闭、错误回滚和进程异常退出。

**总结：CUDA VMM 将 VA、physical memory、mapping、permission 和 portable handle 解耦，带来精确共享与 multicast 能力，同时把对齐、权限、proxy fence 和生命周期责任交给了应用。**

[上一篇：4.15 Interprocess Communication](/posts/cuda-features-15-interprocess-communication/) · [下一篇：4.17 Extended GPU Memory](/posts/cuda-features-17-extended-gpu-memory/)
