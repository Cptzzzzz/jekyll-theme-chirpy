---
title: 'CUDA Features 4.4：Cooperative Groups'
author: Cptz
date: 2026-07-22 08:04:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, cooperative-groups, synchronization, collectives]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.4 Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)。

## 4.4.1 引言

Cooperative Groups（CG）把“共同参与某个操作的线程集合”表示为显式 group。设备函数可在接口中声明同步范围，避免把“必须由整个 block 调用”藏在实现内部。

## 4.4.2 Group Handle 与成员函数

group handle 提供 `size()`、当前线程 `thread_rank()`、同步以及集合操作。具体 group 还可提供 group/grid index、维度和有效性查询。

```cpp
namespace cg = cooperative_groups;

template<class Group>
__device__ float sum(Group const& g, float x) {
  for (int offset = g.size() / 2; offset; offset /= 2)
    x += g.shfl_down(x, offset);
  return x;
}
```

handle 表示参与集合和能力，不意味着所有成员在任意控制流位置都自动收敛。集体操作仍要求合法成员共同调用。

## 4.4.3 Groupless 默认行为

不接收 group 的重载根据调用上下文隐式选择参与者，写法短，但函数签名无法告诉调用者需要哪些线程。复杂库代码优先使用显式 group。

### 4.4.3.1 尽早创建隐式 Group Handle

`this_thread_block()`、`this_grid()` 等隐式 handle 应在进入分歧控制流前创建。这样每个参与线程以相同上下文构造集合，也让后续函数只消费一个已确定 group。

### 4.4.3.2 只按引用传递

原文要求 group handle 只按引用传递，避免复制破坏某些 handle 的实现与生命周期假设。模板通常写成 `Group const&` 或库 API 要求的引用形式。

## 4.4.4 创建 Cooperative Groups

常见创建方式：

- `this_thread_block()`：当前 block 全体线程；
- `tiled_partition<N>(parent)`：把父组分成固定 tile；
- `coalesced_threads()`：当前控制流点活跃且收敛的线程；
- cluster/grid/multi-grid group：在对应 cooperative launch 与硬件能力下建立大范围集合。

静态 tile 大小允许编译器针对 warp shuffle 等能力优化；动态分组更灵活，但可能使用更通用实现。

### 4.4.4.1 避免 Group Creation Hazard

在分支内部创建组时，各线程观察到的活跃集合可能不同。尤其是 `coalesced_threads()`，它描述的是创建瞬间的收敛集合，重新分歧后不能假设原集合自动改变。

```cpp
if (predicate) {
  auto active = cg::coalesced_threads();
  // 只有进入分支的 active 成员参与这里的 collective。
  auto prefix = cg::exclusive_scan(active, value);
}
```

## 4.4.5 同步

### 4.4.5.1 Sync

`g.sync()` 让 group 成员会合，并提供该 group 作用域相应的内存排序。若只有部分成员到达，行为不成立。同步作用域不会自动扩展到组外线程。

### 4.4.5.2 Barriers

CG barrier 接口可表达分离式 arrive/wait 和多阶段同步，适合线程到达后继续独立工作。barrier 的 expected participants 必须与 group/partition 一致，线程提前退出时需按协议 drop。

## 4.4.6 集合操作

### 4.4.6.1 Reduce

`reduce` 把组内值按二元操作归约。操作应满足库对类型与结合性的要求。浮点加法并非数学上严格结合，因此不同树形顺序可能产生末位差异。

### 4.4.6.2 Scan

inclusive/exclusive scan 为每个线程产生前缀结果，可用于压缩、队列槽位分配和分支内活跃元素编号。结果顺序由 group rank 定义，而不是全局 thread index 的任意假设。

### 4.4.6.3 Invoke One

`invoke_one` 让 group 选择一个线程执行 callable，并可把结果在组内共享。它比硬编码 `thread_rank()==0` 更能表达“只执行一次”的集合语义；callable 内仍需遵守允许的同步和副作用范围。

## 4.4.7 异步数据移动

CG 可协作发起 `memcpy_async`，把 global 数据搬到 shared memory，再通过 wait/sync 确认可消费。参与线程共同覆盖复制范围，调用必须对 group 成员一致。

### 4.4.7.1 对齐要求

源和目标地址及元素尺寸要满足接口对齐。原文强调至少 4 字节的适用粒度，16 字节对齐通常最有利于底层异步复制。若编译器无法证明对齐，可能退回普通复制路径或要求显式 `aligned_size_t` 承诺；错误承诺会导致未定义行为。

```cpp
auto block = cg::this_thread_block();
cg::memcpy_async(block, smem, gmem,
                 cuda::aligned_size_t<16>(bytes));
cg::wait(block);
consume(smem);
```

## 4.4.8 大规模 Group

Grid group 的同步要求 cooperative kernel 的所有 block 能按规则共同驻留/推进。发射网格不能超过设备在该 kernel 资源配置下允许的 cooperative resident blocks。

CUDA 13 已移除旧的 multi-device cooperative kernel launch API。跨设备全局协调应使用当前支持的通信/执行机制，而不是依赖已移除接口。

### 4.4.8.1 何时使用 `cudaLaunchCooperativeKernel`

当算法确实需要单 kernel 内跨 block 的全 grid barrier，并且拆成多个 kernel 的边界成本或状态保存不合适时使用。发射前通过 occupancy API 计算每 SM 可驻留 block 数，再限制 grid。

如果只需要阶段间全局同步，拆成两个 kernel 往往更简单：kernel 边界天然提供全局执行分段，也不会受 cooperative grid 驻留上限约束。

**总结：Cooperative Groups 的核心是把参与集合、集合能力和同步作用域变成显式接口，从而让线程协作代码可组合且更容易验证。**

[上一篇：4.3 Stream-Ordered Memory Allocator](/posts/cuda-features-03-stream-ordered-memory-allocator/) · [下一篇：4.5 Programmatic Dependent Launch](/posts/cuda-features-05-programmatic-dependent-launch/)
