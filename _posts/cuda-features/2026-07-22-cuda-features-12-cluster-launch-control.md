---
title: 'CUDA Features 4.12：Cluster Launch Control 工作窃取'
author: Cptz
date: 2026-07-22 08:12:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, cluster-launch-control, work-stealing, scheduling]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.12 Work Stealing with Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)。

固定“每 block 一份任务”具有低调度开销和良好抢占点，但尾部 block 负载不均会拖慢 grid；固定少量常驻 block 用 grid-stride loop 能自行取任务，却会让一个长 kernel 缺少细粒度的调度让路机会。Cluster Launch Control（CLC）试图结合两者。

运行中的 block 请求取消一个尚未开始的 block。若成功，它取得被取消 block 的 index，并执行那份工作，相当于从未来 grid 中窃取任务；若失败，说明没有可取消 index 或调度器因高优先级工作等原因拒绝，当前 block 可退出，让调度器处理其他工作。

## 4.12.1 API 细节

取消请求是异步 proxy 操作，结果写入 shared memory，并通过 mbarrier 同步。libcu++ PTX API 提供请求指令与结果解码指令。

### 4.12.1.1 Thread Block Cancellation

原章把流程分为 setup 和循环五步：

1. 在 shared memory 声明取消结果与 barrier，线程持有 phase。
2. 单线程初始化 arrival count 为 1 的 mbarrier，随后 block 同步。
3. 单线程发起取消请求，并设置对应 transaction count。
4. 等待 barrier phase 完成，确保结果可读。
5. 解码成功状态与被取消的 block index；成功则处理该 index，失败则结束。

```cpp
// 结构性伪代码，低层函数名以 libcu++ PTX API 为准。
while (true) {
  if (threadIdx.x == 0)
    request_cancel(result, barrier);
  __syncthreads();

  wait_for_cancel_result(barrier, phase);
  bool ok = decode_success(result);
  auto stolen = decode_block_index(result);
  __syncthreads();

  if (!ok) break;
  process_work_for(stolen);
  phase ^= 1;
}
```

取消成功意味着原 block index 不会再作为独立 block 启动，当前 block 必须完整承担其逻辑工作。算法不能依赖被取消 block 的 shared 状态，因为它从未开始执行。

### 4.12.1.2 取消约束

- 推荐每个 block 由单线程一次发起一个请求。
- 多线程同时请求会取消多个 block，需要每个请求有独立 shared 结果区，barrier arrival/transaction count 也要对应调整。
- 只能取消尚未开始的 block，已调度/运行的 block 不可窃取。
- 取消可能因没有 index、高优先级 kernel 调度等原因失败；失败是正常控制流，不是必然错误。
- 请求、结果和 barrier 都要遵守 async proxy 与 shared-memory 可见性规则。
- 对 cluster kernel，窃取单位和 index 必须保持 cluster 结构完整，不能把一个 cluster 拆成不合法的部分。

CLC 不保证完美负载均衡。靠后的任务在请求前可能已被调度，任务成本也可能高度不均；它提供的是硬件调度队列中的机会式 work stealing。

## 4.12.2 Vector-Scalar 示例

示例共同执行 $v_i \leftarrow \alpha v_i$，并假设每个 block 先计算一次有成本的 `alpha`。

### 4.12.2.1 Thread Block

三种设计的差异：

| 设计 | 优点 | 代价 |
|---|---|---|
| 每 block 固定一个 tile | 索引简单，调度器有很多 block | 每个 block 重复 prologue |
| 固定 SM 数 block + grid-stride | prologue 次数少 | 长驻留 block 不利于让路，负载由软件循环固定 |
| CLC | 复用 prologue，并保留未启动 block 可取消的调度粒度 | 有取消请求/barrier 开销，且成功非保证 |

CLC 版本先处理自己的 `blockIdx`，随后不断请求另一个未启动 index。因当前 block 已经算好 `alpha`，它可以复用 prologue，减少重复成本。

### 4.12.2.2 Thread Block Cluster

cluster 版本按 cluster 为调度/窃取单位。一个 cluster 内 block 可使用 distributed shared memory 和 cluster sync，故被窃取工作必须映射成完整 cluster index，并由当前 cluster 的所有 block 协同处理。

这对“每个任务需要一个 cluster、任务数很多、每任务都有昂贵共同 setup”的场景有意义。若单任务很小，取消、同步和索引重映射可能比省下的 prologue 更贵。

评估时至少比较：总 kernel 时间、prologue 执行次数、取消成功/失败次数、尾部 SM 利用率，以及高优先级 kernel 的可调度延迟。CLC 的价值不仅是当前 kernel 快，也包括失败后让调度器更容易切换到高优先级工作。

**总结：Cluster Launch Control 通过取消尚未启动的 block/cluster 来复用当前执行实体的 setup，并在失败时保留调度让路能力，是一种硬件队列感知的机会式工作窃取。**

[上一篇：4.11 Asynchronous Data Copies](/posts/cuda-features-11-asynchronous-data-copies/) · [下一篇：4.13 L2 Cache Control](/posts/cuda-features-13-l2-cache-control/)
