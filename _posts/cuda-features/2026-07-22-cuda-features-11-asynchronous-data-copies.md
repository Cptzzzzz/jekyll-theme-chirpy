---
title: 'CUDA Features 4.11：Asynchronous Data Copies'
author: Cptz
date: 2026-07-22 08:11:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, cp-async, tma, stas, pipeline]
math: true
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.11 Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)，分别讨论 LDGSTS、TMA 与 STAS。三者方向、粒度、完成机制和硬件代际不同，不能互换理解。

GPU kernel 经常把 global memory tile 搬到 shared memory，再执行复用密集的计算。异步复制允许发起线程在数据移动期间推进独立计算，并避免某些“global load 到寄存器、再 store 到 shared”的中间路径。

## 4.11.1 LDGSTS

LDGSTS（计算能力 8.0+）面向小粒度、逐元素的 global-to-shared 异步复制。它支持每条操作 4、8 或 16 字节：4/8 字节使用 L1 access 模式，16 字节还可使用 L1 bypass，减少对 L1 的污染。

约束包括：

- 唯一异步方向是 global memory 到 shared memory；
- 源与目标按 4/8/16 字节复制尺寸对齐；
- 两端 128 字节对齐通常获得最佳访问行为；
- 操作被建模为 async-thread proxy，实际重叠程度由硬件实现决定；
- 完成通过 shared barrier 或 pipeline 观察。

默认每个线程只等待自己发起的 LDGSTS。如果线程 A 搬的数据要由线程 B 使用，A 等待自己的 copy 完成仍不足以自动同步 B，需要在相应等待之后加 group/block 同步建立跨线程可见性。

```cpp
auto pipe = cuda::make_pipeline();
pipe.producer_acquire();
cuda::memcpy_async(smem + tid, gmem + tid,
                   cuda::aligned_size_t<16>(16), pipe);
pipe.producer_commit();

pipe.consumer_wait();
__syncthreads();  // 后续会读取其他线程搬入的数据
consume_tile(smem);
pipe.consumer_release();
```

### 4.11.1.1 条件代码中的批量 Load

边界 tile 常让部分线程没有合法输入。若每个 lane 在分歧分支中独立提交 pipeline batch，会引发 warp entanglement，并让 batch 数远多于预期。

更稳妥的设计是让 warp 统一执行 commit，越界 lane 使用合法的零填充/不发起元素复制路径，但仍参与 pipeline 控制；或者先通过 predicate 形成明确的有效区，再在 commit 前重新收敛。

```text
计算每 lane valid
  -> valid lane 发起元素 copy
  -> warp 收敛
  -> 全体以一致控制流 commit batch
```

不能为了保持收敛而让越界 lane 构造非法 global pointer；即使结果“不会使用”，非法地址计算/访问仍不成立。

### 4.11.1.2 预取

N-stage pipeline 的 prologue 先填充若干 stage，steady state 同时计算当前 batch 和预取未来 batch，epilogue 排空剩余 stage。

设复制延迟为 $L$，每 tile 计算时间为 $C$，理想上需要足够 stage 使在途计算覆盖 $L$。但 stage 增加会消耗 shared memory 并降低 occupancy，因此应找最小的充分 stage 数。

原章分别用 `cuda::memcpy_async`、`cooperative_groups::memcpy_async` 与低层 pipeline primitive 展示同一思想。高层接口更便于维护，低层接口用于需要直接控制 batch 和 barrier 的实现。

如果每个线程只访问自己搬入的元素，等待自己的复制后未必需要全 block barrier；如果 tile 在计算阶段跨线程复用，则必须在所有相关复制完成后同步。

### 4.11.1.3 Warp Specialization 的 Producer-Consumer

warp specialization 把第一组 warp 设为 producer，其他 warp 设为 consumer。producer 负责填充双缓冲，consumer 等 full barrier、计算，再通过 ready barrier 归还 buffer。

```text
producer warp: wait ready -> LDGSTS -> signal full
consumer warp: wait full  -> compute -> signal ready
```

分区 pipeline 版本由 API 管理 stage；低层版本用 `__pipeline_memcpy_async` 加 barrier，并通过 `__pipeline_arrive_on` 把复制完成关联到 full barrier。barrier 初始计数、角色线程数和退出路径必须一致。

## 4.11.2 Tensor Memory Accelerator（TMA）

TMA（计算能力 9.0+）面向 bulk 复制。LDGSTS 通常由许多线程各搬 4/8/16 字节，TMA 则可由一个 elected thread 发起较大的一维或多维 tile 传输，其他线程用于计算。

TMA 支持的核心方向包括 global-to-shared 与 shared-to-global；在 cluster 场景还可涉及 distributed shared memory。global-to-shared 通常以 mbarrier transaction bytes 报告完成，shared-to-global 则用 async group 的 commit/wait 确认 TMA 已读完 shared 源。

TMA 位于 async proxy。普通线程写 shared 后，在 TMA 读取前要用 `fence_proxy_async` 等正确 proxy fence，再由 block 同步确保所有写入发布。普通 `__syncthreads()` 与 proxy fence解决的层次不同，不能随意少一个。

### 4.11.2.1 一维数组

一维 bulk copy 不需要多维 tensor descriptor。典型 global-to-shared 流程：

1. 初始化 block-scope barrier。
2. 一个 elected thread 发起 bulk copy。
3. 由高层 API 自动，或由低层 API 显式设置 expected transaction bytes。
4. block 线程 arrive。
5. barrier 同时等 arrival 与传输字节完成。
6. 所有线程安全读取 shared buffer。

```text
thread arrivals == 0 AND transaction bytes == 0
  => tile 可消费
```

使用 `cuda::memcpy_async` 时 transaction accounting 可由库完成；使用 `cuda::device::memcpy_async_tx` 或 PTX `cp_async_bulk` 时，需要显式告知 barrier 预期字节。多个线程都更新 expected bytes 时会累加，因此通常只选一个发起线程。

shared-to-global 时，发起后把 bulk 操作 commit 成 async group，再等待 group 已完成读取 shared 源，之后才能覆盖/释放该 shared buffer。目标 global 数据何时可被其他工作读取，还需更外层依赖。

#### 4.11.2.1.1 一维预取

一维 TMA 同样可以与多 stage pipeline 组合。一个 producer thread 为下一 tile 发起大块复制，其余线程计算当前 tile。相比让每线程各做 LDGSTS，它减少了地址生成和复制指令压力。

最佳策略通常是少量、更大的 bulk operation；把一个连续大 tile 切成很多小 TMA 并不会自然更异步，反而增加发起和 barrier bookkeeping。

## 4.11.2.2 多维数组

多维 TMA 通过 `CUtensorMap` 描述 global tensor：基础地址、rank、各维大小、stride、shared box 大小、element stride、interleave、swizzle、L2 promotion 和越界填充方式。硬件根据 tile 坐标生成地址并处理边界。

主机常用 `cuTensorMapEncodeTiled` 生成 descriptor，再以 `const __grid_constant__` kernel 参数传入。也可放在 device constant memory；若放在可修改的 global memory，使用前需要 tensor-map proxy fence。

目标 shared buffer应满足 TMA 对齐，原章示例采用 128 字节对齐。加载 tile 后，barrier 完成意味着数据可供线程读取；写回前，线程写 shared 的结果必须先对 TMA proxy 可见。

### 4.11.2.2.1 在设备端编码 Tensor Map

需要完全设备侧生成 descriptor 时，可使用相应 tensormap 指令/API 在 shared 或 global 存储中编码字段。rank 可覆盖多维张量，字段之间有取值和对齐约束。

设备编码适合地址、shape 或 stride 只能在设备端确定的动态工作流，但成本高于复用一个主机预编码 descriptor。应尽量让多个 tile 共享模板，只修改必要字段。

### 4.11.2.2.2 设备端修改 Tensor Map

PTX tensormap replace 指令可修改 global address、box dimensions、global dimensions/strides、element strides、元素类型、interleave、swizzle 和 fill mode 等字段。

修改通常由一个 warp 协作：在 shared memory 创建/修改临时 descriptor，warp 同步后用 `tensormap_cp_fenceproxy` 把完整 descriptor 复制并 release 到 global memory。128 字节 descriptor 的发布必须原子地遵守规定流程，不能由任意线程用普通 stores 拼接后立即消费。

### 4.11.2.2.3 使用已修改 Tensor Map

消费者 block 在使用 global memory 中刚更新的 map 前，需要执行相应 acquire tensor-map proxy fence。每次修改后首次使用要重新建立可见性；同一 block 对未再次修改的 map 后续使用不必重复同样 fence。

descriptor 的生命周期要覆盖所有 TMA 操作。生产者修改下一 descriptor 时，不能与仍使用旧值的 operation 发生未排序竞争。

### 4.11.2.2.4 用 Driver API 创建模板

主机可用 Driver API 编码一个合法模板，填入固定的数据类型、rank、布局、swizzle 等字段；device 端复制模板后只替换动态 base/shape。这样比设备端从空白描述符逐字段构建更易保证保留位和字段组合合法。

模板本身仍需匹配最终字段的约束。例如替换 stride 后，新的 stride 必须满足 TMA 对齐和范围要求。

### 4.11.2.2.5 Shared-Memory Bank Swizzling

TMA 可在搬入 shared 时应用 32B、64B 或 128B swizzle，写回时还原。这改变 16 字节 chunk 到 bank subgroup 的映射，用来降低矩阵行/列访问的 bank conflict。

关键约束：

- global memory 按 128 字节对齐；
- shared memory 至少按要求对齐，原文建议 128 字节；
- swizzle 粒度固定为 16 字节；
- shared box 的 inner dimension 不得超过 swizzle span；
- 128B/64B/32B 模式的图案分别在 1024/512/256 字节后重复；
- shared 指针相对重复周期的偏移会进入索引变换，不能只写一个固定 XOR 公式而忽略 base offset。

swizzle 只改变 shared 布局，不会自动让任意计算访问无冲突。consumer 必须按同一映射寻址，并通过 profiler 验证 shared bank conflict 是否真正下降。

## 4.11.3 STAS

STAS（计算能力 9.0+）把寄存器中的 4、8 或 16 字节异步写到 cluster 内 distributed shared memory。它只通过 libcu++ 的低层 `cuda::ptx::st_async` 暴露，目标按传输尺寸对齐，完成由 shared-memory barrier 报告。

它解决的是 block cluster 内的小数据点对点传递，不是 global-to-shared bulk copy。原章示例把 8 个 block 排成环：每个 block 向右邻居生产，同时从左邻居消费。

每个 block 维护：

- `filled` barrier：远端写入完成，本地 consumer 可读；
- `ready` barrier：本地已消费，远端 producer 可覆盖。

cluster 启动后先 `cluster.sync()`，保证所有 block 的 barrier 和 shared buffer 已初始化。之后用 `map_shared_rank` 取得邻居 buffer 与 barrier 的 distributed shared 地址。

```text
向 next block 发起 st_async
  -> 本地 filled 登记预期接收字节
  -> 等 filled phase
  -> 消费 prev block 写入的数据
  -> 到达 prev block 的 ready
  -> 等本地 ready，确认 next 已消费
  -> phase 翻转
```

地址空间必须匹配：本地 barrier 使用 `space_shared`，映射到其他 block 的远端 barrier 使用 `space_cluster`。混淆两者不是性能问题，而是错误的 PTX 内存空间语义。

**总结：LDGSTS 负责小粒度 global-to-shared，TMA 负责大块一维/多维传输，STAS 负责寄存器到 distributed shared；选型必须同时匹配数据方向、粒度、硬件代际和完成协议。**

[上一篇：4.10 Pipelines](/posts/cuda-features-10-pipelines/) · [下一篇：4.12 Cluster Launch Control](/posts/cuda-features-12-cluster-launch-control/)
