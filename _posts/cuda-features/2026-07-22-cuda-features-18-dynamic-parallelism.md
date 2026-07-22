---
title: 'CUDA Features 4.18：CUDA Dynamic Parallelism（CDP2）'
author: Cptz
date: 2026-07-22 08:18:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, dynamic-parallelism, cdp2, device-runtime]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.18 CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)。本章描述 CUDA 12.0+ 默认的 CDP2；很多旧教程讲的是 CDP1，尤其是 device-side `cudaDeviceSynchronize`，不能直接套用。

## 4.18.1 引言

### 4.18.1.1 概览

CUDA Dynamic Parallelism（CDP）允许正在 GPU 上执行的 thread 发射新的 grid。递归细分、数据依赖的自适应并行和不规则任务生成可以在设备端做决策，避免先把控制权和判定数据交还 CPU。

CUDA 12.0+ 默认使用 CDP2；计算能力 9.0+ 只支持 CDP2。较旧设备可用编译宏 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 选择 legacy CDP1，但 CDP1 计划在未来移除。

CDP 提供表达能力，不保证比扁平化 kernel、persistent kernel、work queue 或 device graph 更快。每个 child launch 都有运行时与资源管理成本。

## 4.18.2 执行环境

### 4.18.2.1 Parent 与 Child Grid

发起 device-side launch 的线程属于 parent grid，新 invocation 是 child grid。关系 properly nested：parent grid 在它所有线程发起的 child work 完成前，不会被视为完成。因此 host 等待 parent 完成，也间接等待其后代完成。

proper nesting 不表示 parent thread 可以在任意位置同步 child 并读取结果。CDP2 删除了 device-side `cudaDeviceSynchronize`；child 结果在 parent 结束前要由 tail launch kernel 消费。

### 4.18.2.2 CUDA Primitive 的作用域

Device Runtime 提供一组类似 Runtime API 的设备端接口，但能力有限。由 device API 创建的 stream/event 对创建它们的 grid 内所有线程可见，只在该 grid scope 有效。

以下跨界使用均未定义：

- 把 host 创建的 stream handle 传入 kernel 使用；
- parent grid 创建 stream 后让 child grid 使用；
- 把一个 grid 创建的 event handle 交给另一个 grid；
- grid 结束后继续使用其 device-created CUDA object。

### 4.18.2.3 Streams 与 Events

同一显式 stream 中 child launch 有序，不同 stream 可建立 event 依赖。stream/event handle 虽由某一线程创建，但 grid 内其他线程可使用，所以对象表是 grid-scope 而不是 thread-scope。

grid 退出时，所有由其发射的工作按 proper nesting 隐式完成，包含各 stream 的依赖。event handle 不保证跨 grid 唯一，数值相同也不能越域使用。

CDP2 还提供特殊 stream：

- `cudaStreamTailLaunch`：当前 grid 及其 child 工作完成后执行，用于接续和消费 child 结果；
- `cudaStreamFireAndForget`：派生工作不按普通同 stream 顺序等待发起点，但仍受 execution environment 规则管理。

### 4.18.2.4 顺序与并发

多个 parent 线程向同一显式 stream 发射时，最终顺序取决于线程实际调度；若业务要求确定顺序，先用 block/grid 可用同步选出单一发射者或建立事件顺序。

device-side 隐式 NULL stream 的范围是 thread block：同一 block 多线程向 NULL stream 发射会有序，不同 block 的 NULL streams 可以并行。想让同一 block 的多个发射并发，应创建显式 named streams。

不同 stream 只允许并发，不保证并发。程序不能让两个 child grid 互等，并假设硬件一定同时驻留来解除死锁。

## 4.18.3 内存一致性

parent/child 共享 global 和 constant storage，但各自有独立 shared 与 local memory。映射内存可用相同 pointer；texture 可读共享 backing，但有独立一致性边界。

| Memory space | Parent/child 使用同一指针 |
|---|---|
| global | 是 |
| mapped | 是 |
| local | 否 |
| shared | 否 |
| texture | 是，只读访问路径 |

### 4.18.3.1 Global Memory

child launch 时是明确一致性点：发射线程在 launch 之前可见的 global writes 对 child 可见。其他 parent 线程先写的数据，需要先通过 `__syncthreads()` 等让发射线程看到，才能保证 child 看见。

launch 之后 parent 与 child 并发访问同一位置，没有额外同步就是 race。CDP2 parent 不能显式等待 child，因此 child 修改不能保证被继续运行的 parent 线程看到。

需要在 parent grid 完成前消费 child 结果时，把 consumer kernel 发射到 `cudaStreamTailLaunch`：

```cpp
__global__ void parent(State* s) {
  child<<<child_grid, child_block>>>(s);
  consume_child<<<1, 256, 0, cudaStreamTailLaunch>>>(s);
}
```

tail kernel 不是 parent thread 的继续执行，而是一个后继 grid，因此资源和逻辑应按新的 kernel 边界组织。

### 4.18.3.2 Mapped Memory

zero-copy/mapped host memory 由 parent 和 child 通过 global pointer 访问。它遵守相同 launch 可见性和 race 规则，还要考虑 system scope 与主机并发访问。host 不能因为 child 已发射就提前读取结果，应等待顶层 parent 完成或相应 host-visible 事件。

### 4.18.3.3 Shared Memory

每个 grid 的每个 block 有自己的 shared memory。parent shared pointer 不能传给 child，child 也不可能访问 parent block 的 shared lifetime。需要共享的数据必须先写入 global/mapped memory。

编译器和 runtime 会拒绝一些可识别的 shared pointer 参数，但不要依赖所有非法情况都能静态检测。

### 4.18.3.4 Local Memory

局部数组、寄存器溢出和线程栈位于 parent thread 的 local memory，不属于 child。把设备函数局部变量地址传给 child 是错误的，即使 C++ 类型只是普通 pointer。

安全规则是：传给 child 的存储显式来自 global heap（合适的设备分配/设备端 `new`）或全局 `__device__` storage。

#### 4.18.3.4.1 Texture Memory

child launch 时，launch 前对 texture backing global memory 的写可反映到 child texture 读取。child 写 global backing 后，parent 的 texture path 在 parent grid 内不保证看到新值；同样应由 tail kernel 继续消费。parent/child 并发对 backing 写与 texture 读可能不一致。

## 4.18.4 编程接口

### 4.18.4.1 基础

Device code 使用普通 `<<<grid, block, shared, stream>>>` 语法发射 child。编译通常需要 relocatable device code（`-rdc=true`）并链接 `cudadevrt`。

launch 后检查 device-side `cudaGetLastError()` 只能报告提交相关错误；child 执行期错误最终仍需在上层同步/错误检查中观察。

### 4.18.4.2 C++ Device Runtime

Device Runtime API 是逐线程调用，允许位于分歧分支，不要求整个 block 同时调用，因此本身不会因为只有一个线程 launch 就造成 collective deadlock。大量线程各自发射 child 会产生大量 launch，应主动聚合。

#### 4.18.4.2.1 Device-Side Kernel Launch

`Dg`/`Db` 是 grid/block `dim3`，`Ns` 为每 block 动态 shared memory 字节，`S` 必须是同一 grid 内创建的合法 stream，省略时用对应 NULL stream。

device-side launch 对发起线程异步返回。child 可能很快开始，也可能直到 parent 到达隐式 launch synchronization point 才开始。独立 stream 并发仍是机会性的。

child 继承全局设备配置，例如 cache config 与 device limits；不能在 child launch 参数中把它变成另一 device 的工作。

#### 4.18.4.2.2 Events

device-created event 可记录 child stream 进度并让另一 device-created stream wait。event 不能跨 grid 传递，且只表达其合法 scope 内的依赖。

#### 4.18.4.2.3 同步

CDP2 没有 parent thread 显式等待 child 的通用 API。若一个 parent thread 要依赖其他 parent 线程发射的 child，可用 grid 内共享 stream/event 组织 child 之间的顺序，但 parent thread 本身仍不能因此直接读取 child 结果。

可见 child 修改的合法设备端后继是 tail launch grid；最终 host 可在 parent completion 后读取。

#### 4.18.4.2.4 Device Management

device runtime 只能控制当前执行 device，不支持 device-side `cudaSetDevice()`。`cudaGetDevice()` 与 host 所见 ordinal 一致。可用 `cudaDeviceGetAttribute` 按 ID 查询属性，但 device runtime 不提供整体 `cudaGetDeviceProperties()`。

## 4.18.5 编程指南

### 4.18.5.1 性能

CDP 适合每次 child 有足够工作量、并行形状只有设备端才能有效决定的场景。若每个数据元素都发射一个极小 child，launch/跟踪开销会压倒计算。

常见替代方案：单 kernel 内循环、全局 work queue、persistent kernel、prefix sum 后批量发射、CUDA Graph 条件节点或 device graph launch。应以端到端时间和代码复杂度比较。

#### 4.18.5.1.1 启用 CDP 的 Kernel 开销

链接/使用 device runtime 会启用 dynamic launch 跟踪软件，其开销可能影响同时运行的 kernel，即使某个 kernel 自己没有 launch child。因此 benchmark 要比较未链接 CDP 的基线，而不只是比较 parent 内某个分支。

### 4.18.5.2 实现限制

嵌套深度、pending launches、device runtime 内存、stream/event 数和参数存储都受实现资源限制。语义得到保证，不表示规模可以无限增长。

#### 4.18.5.2.1 Runtime

Device runtime 会为 launch tracking 预留内存，尤其是 pending grid launch pool。每个 pending launch 的配置和参数要保存到 child 完成。

host 可通过 `cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, n)` 调整固定 launch pool。提高它增加内存占用，降低它可能使发射在压力下失败。应用要检查 launch error 并控制任务生成速率。

### 4.18.5.3 兼容与互操作

CDP2 函数不能 launch CDP1 函数，反之亦然；调用图混用会在 module load 得到 `cudaErrorCdpVersionMismatch`。计算能力 9.0+ 上 device-side `cudaDeviceSynchronize` 不可用。

旧设备上强制 CDP1 的代码也不能引用 CDP2 的 tail/fire-and-forget stream。Fatbin/JIT 在不同架构上可能暴露版本不匹配，部署测试必须覆盖实际 GPU。

## 4.18.6 从 PTX Device Launch

这一节主要面向编译器和语言实现者。低层需要 `cudaGetParameterBuffer` 取得参数区，按 ABI 布局写参数，再用 `cudaLaunchDevice` 提交 kernel pointer、参数 buffer、grid/block、shared bytes 和 stream。

### 4.18.6.1 Kernel Launch API

#### 4.18.6.1.1 `cudaLaunchDevice`

PTX 需按 32/64 位 address size 声明正确函数签名。实现位于 `cudadevrt`，必须链接。无参数 kernel 的 parameter buffer 可为 null。

#### 4.18.6.1.2 `cudaGetParameterBuffer`

调用传入最大参数对齐和总字节数。当前实现返回 64 字节对齐 buffer，虽然 alignment 参数当前可能被忽略，仍应传真实要求保证未来兼容。

### 4.18.6.2 Parameter Buffer Layout

参数按 CUDA kernel ABI 顺序放入 buffer，每个参数 offset 对齐到自身 alignment，最终 buffer 覆盖全部参数。编译器实现不能简单无 padding 地拼接字段；结构体、向量类型和指针宽度尤其需要按目标 ABI 计算。

**总结：CDP2 让 GPU 动态生成嵌套 grid，并用 tail launch 替代 parent 内显式等待；正确设计必须同时处理 grid-scope 对象、launch 时一致性、global-only 数据传递和有限 launch pool。**

[上一篇：4.17 Extended GPU Memory](/posts/cuda-features-17-extended-gpu-memory/) · [下一篇：4.19 CUDA Interoperability](/posts/cuda-features-19-graphics-interop/)
