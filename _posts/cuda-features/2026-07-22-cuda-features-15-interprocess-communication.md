---
title: 'CUDA Features 4.15：Interprocess Communication'
author: Cptz
date: 2026-07-22 08:15:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, ipc, vmm, multi-gpu]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.15 Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)。

同一进程内的线程共享 CUDA device pointer 和 event handle；不同进程有各自虚拟地址空间，不能把指针数值直接发送过去。CUDA IPC 的共同模式是导出 process-portable handle，用 OS IPC/MPI 等传递，再由接收进程导入为本地 device pointer/对象。

```text
进程 A：CUDA 对象 -> portable handle
           | OS IPC / shared memory / socket / MPI
进程 B：portable handle -> 本地 CUDA 对象/指针
```

单节点可使用 OS 句柄；多节点 NVLink fabric 使用 fabric handle，在独立 OS 实例中导入本地映射。

## 4.15.1 Legacy IPC API

legacy 路径通常包括：

- `cudaIpcGetMemHandle` 导出合适的 device allocation；
- 接收方 `cudaIpcOpenMemHandle` 获得本进程 pointer；
- 使用结束后 `cudaIpcCloseMemHandle`；
- event 通过 `cudaEventInterprocess` 创建，再导出/打开 event handle 用于跨进程顺序。

它当前只在 Linux 平台支持。导出 allocation 必须一直存活到所有导入者停止使用；接收方关闭本地映射不等于释放生产者 allocation。

IPC 句柄解决可引用性，不自动解决生产/消费同步。生产者写完、消费者开始读、消费者结束、生产者释放这四个阶段需要 event、进程协议或通信库共同建立。

子分配器还要特别小心：若从一个大 `cudaMalloc` 区域切出小 buffer 再导出，底层映射粒度可能让接收方看到同一 allocation 的其他数据。安全设计应使用合适粒度的独立 allocation 或 VMM 权限边界。

## 4.15.2 VMM IPC

VMM 使用 Driver API，在创建物理 allocation 时决定可导出的 handle type，并允许逐 allocation 设置 peer 可访问性。流程是：

1. `cuMemCreate` 创建满足 granularity 的物理内存。
2. `cuMemExportToShareableHandle` 导出 POSIX FD、Win32 或 fabric handle。
3. 接收方 `cuMemImportFromShareableHandle` 导入 generic allocation handle。
4. 各进程独立 reserve VA、map 物理对象并设置 device access。
5. 用完后按各自映射与对象生命周期逆序释放。

VMM 的优势是地址与物理内存解耦、逐 allocation 权限和多节点 fabric 支持；代价是接口更低层，必须管理 granularity、OS handle、VA 映射、access descriptor 和清理顺序。

选择 legacy IPC 还是 VMM，不应只看 API 数量。简单 Linux 单节点共享可用 legacy；需要精确 peer 映射、可控粒度、Windows OS handle 或 multi-node fabric 时，VMM 更合适。

**总结：CUDA IPC 共享的是可导入的底层对象句柄，而不是裸 device pointer；跨进程正确性还需要独立管理映射生命周期和执行同步。**

[上一篇：4.14 Memory Synchronization Domains](/posts/cuda-features-14-memory-synchronization-domains/) · [下一篇：4.16 Virtual Memory Management](/posts/cuda-features-16-virtual-memory-management/)
