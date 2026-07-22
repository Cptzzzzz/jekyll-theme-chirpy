---
title: 'CUDA Features 4.10：Pipelines'
author: Cptz
date: 2026-07-22 08:10:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, pipeline, double-buffering, async-copy]
math: true
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.10 Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)。

Pipeline 把异步批次组织成固定 stage 数的 FIFO。producer 获取空 stage 并提交工作，consumer 等待已提交 stage、使用结果后释放。它把常见的多 barrier 双缓冲协议封装为更清晰的状态机。

## 4.10.1 初始化

block-scope pipeline 的 shared state 放在 shared memory，模板参数包含 scope 和 stage 数。创建 pipeline 时还要决定线程角色：所有线程同时生产/消费，或划分 producer 与 consumer。

```cpp
namespace cg = cooperative_groups;
constexpr int stages = 2;

__shared__ cuda::pipeline_shared_state<
    cuda::thread_scope_block, stages> state;

auto block = cg::this_thread_block();
auto pipe = cuda::make_pipeline(block, &state);
```

state 的生命周期覆盖所有异步操作。参与集合必须和初始化 group 一致；不能让半个 block 使用按全 block 建立的统一 pipeline。

## 4.10.2 提交工作

producer 对当前可写 stage 执行：

1. `producer_acquire()`：等待并取得空 stage。
2. 发起与 pipeline 关联的异步操作。
3. `producer_commit()`：把该批次发布给 consumer，推进 producer sequence。

```cpp
pipe.producer_acquire();
cuda::memcpy_async(block, smem[write_slot],
                   gmem + offset, bytes, pipe);
pipe.producer_commit();
```

acquire 之后必须 commit；遗漏 commit 会让 consumer 等不到该 stage，也可能耗尽所有可写 stage。commit 表示提交批次，不表示数据已完成搬运。

底层 intrinsic 中，`__pipeline_memcpy_async` 追加异步复制，`__pipeline_commit` 把此前操作组成 batch。高层 pipeline API 更明确地维护角色与 stage，通常优先使用。

## 4.10.3 消费工作

consumer 执行：

1. `consumer_wait()`：等 FIFO 头部 stage 的异步工作完成。
2. 读取该 stage 数据并计算。
3. `consumer_release()`：确认不再访问，归还给 producer。

```cpp
pipe.consumer_wait();
compute(smem[read_slot]);
pipe.consumer_release();
```

不能在 wait 前读取目标，也不能在所有 consumer 结束前 release。release 太早会让 producer 覆盖尚在读取的数据。

低层 `__pipeline_wait_prior(N)` 表示等待到只有最近 N 个 batch 可能仍未完成。N 的语义基于感知到的批次序列，不应和“等待第 N 个 stage”混淆。

## 4.10.4 Warp Entanglement

pipeline 的实际 batch sequence（PB）由收敛 warp 推进，而单线程看到的是其 perceived sequence（TB）。完全收敛时一次 commit 对整个 warp 形成一个 batch；完全分歧时，32 个 lane 可能把序列推进 32 次。

后果包括：

- `consumer_wait_prior<N>` 等待比线程本意更老/更多的 batch；
- barrier update 次数被放大；
- pipeline 失去本想得到的重叠；
- lane 间对 slot/sequence 的理解不一致。

因此在 `producer_commit` 和会影响共享 pipeline sequence 的操作前用 `__syncwarp()` 重新收敛。最好让整个 pipeline 控制路径本身保持 warp-uniform。

## 4.10.5 提前退出

partitioned pipeline 中，producer 和 consumer 的参与义务在循环尾部必须完成。producer 不能留下 acquire 但未 commit 的 stage，consumer 不能拿到 stage 后不 release。

线程永久退出时，按 API 的角色协议调用相应退出/收尾操作，使共享状态不再等待它。若不同线程处理的迭代数不同，应把尾部不规则工作与主稳态 pipeline 分开。

## 4.10.6 跟踪异步内存操作

`cuda::memcpy_async(..., pipeline)` 把 copy 加入当前 producer batch。硬件支持且对齐满足时可使用异步 global-to-shared 路径；否则库可能使用等价 fallback，但 pipeline 的完成语义仍保持。

源/目标有效期、地址空间和对齐仍是调用方责任。推荐让 tile 首地址和复制长度满足 16 字节对齐，避免每线程复制产生碎片访问。

Pipeline 只跟踪与它关联的异步操作。普通 load/store 或另一个 barrier 上的事务不会自动成为当前 stage 的完成条件。

## 4.10.7 Producer-Consumer 模式

统一 pipeline 中，每个线程既协作搬运也计算：

```text
prologue：预填前 stages-1 个 tile
steady state：
  acquire/提交下一 tile
  wait/计算当前 tile
  release 当前 tile
epilogue：排空剩余 tile
```

partitioned 模式可做 warp specialization：一个或多个 warp 专门发起 TMA/copy，其他 warp 计算。它减少每次搬运的发起线程数，但必须正确设置角色和参与者，并给 producer 足够工作避免 consumer 饥饿。

stage 数的取舍为：

| 增加 stage 的收益 | 增加 stage 的代价 |
|---|---|
| 覆盖更长 copy latency | 更多 shared memory |
| producer 可更早预取 | occupancy 可能降低 |
| 对短暂延迟波动更鲁棒 | prologue/epilogue 与状态开销增加 |

只有当 copy engine/异步路径与计算能并行，且每 tile 计算足以覆盖搬运时，pipeline 才提升吞吐。应比较 1、2、3 stage 的 kernel 时间、occupancy、stall reason 和内存吞吐，而不是默认 stage 越多越快。

**总结：Pipeline 用 acquire/commit/wait/release 管理有界 FIFO stage；性能来自稳态重叠，正确性来自生产者和消费者对每个 stage 生命周期的完整交接。**

[上一篇：4.9 Asynchronous Barriers](/posts/cuda-features-09-asynchronous-barriers/) · [下一篇：4.11 Asynchronous Data Copies](/posts/cuda-features-11-asynchronous-data-copies/)
