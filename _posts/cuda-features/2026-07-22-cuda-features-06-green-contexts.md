---
title: 'CUDA Features 4.6：Green Contexts'
author: Cptz
date: 2026-07-22 08:06:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, green-context, qos, resource-partition]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.6 Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)。Green Contexts API 在 CUDA 版本中仍在演进，部署时应以目标 Toolkit 的头文件与 release notes 为准。

## 4.6.1 动机与使用场景

同一 GPU 上的吞吐任务和延迟敏感任务会争用 SM、work queue 以及共享内存系统。stream priority 能影响待调度工作的优先次序，却不能为某个 workload 预留一组执行资源。Green Contexts 允许应用在一个进程内划分 GPU 执行资源，并让指定 stream 使用相应 execution context。

适用场景包括：

- 大型后台 kernel 与短时延 kernel 共存；
- 按业务 QoS 把不同 kernel 限制在不同 SM 集合；
- 研究资源缩放曲线，而不让 kernel 自动使用整卡；
- 在已有 MIG 分区内部进一步组织进程内工作。

它不是硬实时机制，也不是 MIG 替代品。MIG 在硬件层隔离更多资源；MPS 面向多进程共享；green context 主要提供可编程的执行资源/队列分区。L2、显存带宽和互连仍可能形成相互干扰。

## 4.6.2 易用模型

应用不需要手工选择具体 SM ID。它描述需要的资源数量与分区方式，Driver/Runtime 返回 resource descriptor，再由 descriptor 创建 green context 和 stream。

抽象链条是：

```text
Device Resources
  -> Resource Partition
  -> Resource Descriptor
  -> Green Context / Execution Context
  -> Green-Context Stream
  -> Kernel / Graph Work
```

资源描述与执行上下文分开，使应用可以先规划资源，再创建实际提交环境。所有步骤都要检查返回码并在失败时释放已构造对象。

## 4.6.3 Device Resource 与 Resource Descriptor

device resource 表示可被划分的资源类别，例如 SM。resource descriptor 表示一组已经选择并满足约束的资源。SM 不是唯一考虑因素，work queue resource 也要加入 descriptor，才能形成可发射工作的执行上下文。

分区受到以下约束：

- 设备支持的最小 SM 数与分配粒度；
- 总请求不能超过输入 resource 中可用数量；
- cluster kernel 需要满足 cluster 大小和可用 SM 的组合；
- 某些设备/模式下的最小 partition 可能不同；
- descriptor 消费/所有权规则要按 API 遵守，不能重复释放或越过生命周期使用。

资源数是上限/集合语义的一部分，不保证 kernel 始终占满它们，也不保证两个 context 必然同时运行。

## 4.6.4 创建示例

下面按原章的四步流程解释，伪代码省略具体类型名和错误处理宏，真实代码应使用当前 Toolkit API 声明。

### 4.6.4.1 Step 1：取得可用 GPU 资源

先为目标 device 查询可用于 green context 的 resource。返回对象属于特定设备，不能拿到另一设备创建 context。

```cpp
// 伪代码：名称以目标 CUDA 头文件为准。
DeviceResource all_resources = query_green_context_resources(device);
```

应用还应同时查询设备 SM 总数、green context 支持、cluster 能力和后续 kernel 的 occupancy，用于判断期望分区是否有意义。

### 4.6.4.2 Step 2：划分 SM Resource

从总 resource 按最小/最大数量或等分策略产生一个或多个 partition。若请求 16 个 SM，实际划分必须满足硬件粒度，API 可能返回符合规则的数量或错误。

```text
全部 SM resource
  -> latency partition（较小、保留）
  -> throughput partition（其余大部分）
```

选择分区不是“越小越低延迟”。太少的 SM 会延长关键 kernel 本身；目标是减少排队时间后的总延迟最小。

### 4.6.4.3 Step 2 续：加入 Workqueue Resource

只有 SM partition 仍不足以形成完整执行环境。按文档流程把 work queue resource 加入分区，使该资源集合可以承载 stream 提交。

work queue 影响工作如何进入硬件调度，但不是无限数量。创建很多小 context 会消耗管理资源，也可能增加调度复杂度。

### 4.6.4.4 Step 3：创建 Resource Descriptor

把选定资源转换为 descriptor。descriptor 是创建 green context 的输入，也是资源归属与销毁管理的关键对象。

```cpp
ResourceDescriptor desc = make_descriptor(sm_and_queue_partition);
```

若构造失败，不能继续创建 context；应按逆序清理已取得的资源/中间对象。

### 4.6.4.5 Step 4：创建 Green Context

最后用 descriptor 和目标 device 创建 green context。Runtime API 可进一步得到 execution context 与可提交工作的 stream。

```cpp
GreenContext ctx = create_green_context(desc, device);
cudaStream_t stream = create_stream_for(ctx);
```

这条 stream 携带 execution context 身份。把 kernel 发射到普通 stream 不会自动使用刚创建的资源分区。

## 4.6.5 发射工作

kernel 的 launch 语法与普通 stream 相似，核心区别是 stream 所属 execution context。kernel 的 grid、block、动态 shared memory 和 cluster 属性仍要合法。

```cpp
latency_kernel<<<grid, block, smem, green_stream>>>(args);
```

occupancy 估算必须基于分区的 SM 数，而不是整卡 SM 数。若 kernel 要求固定 cluster size，分区必须容纳至少一个合法 cluster；否则发射失败或无法实现预期并行度。

CUDA Graph 在 green context 上执行时，节点参数需要能携带 execution context 的 V2 形式。不能把在默认 context 下构造的全部假设无条件搬到 green context 图中。

## 4.6.6 额外 Execution Context API

原章列出的辅助能力包括：

- 在 execution context 上记录 event；
- 让 execution context 等待 event；
- 同步 execution context；
- 查询其 device；
- 获取唯一 ID 用于诊断；
- 销毁 execution context。

这些 API 使事件顺序和诊断明确指向某个执行环境。跨 context event 依赖仍必须满足 event 类型与设备支持规则。

销毁时应遵循：停止接收新工作、等待或确认所有工作完成、销毁关联 stream/graph 引用、再销毁 execution context 和底层 green context。不要依赖进程退出替代正常资源管理。

## 4.6.7 完整示例的性能含义

原章示例在 H100 上把资源划为大约 112 SM 的主分区和 16 SM 的关键分区，并比较关键 kernel 的开始时间。没有划分时，先提交的长 kernel 可能占据大部分执行资源；划分后，关键 kernel 有自己的 SM 集合，因此能更早开始。

这个结果应这样理解：

```text
未分区：关键延迟 = 排队/资源等待 + 自身执行
已分区：关键延迟 = 较少等待 + 在较少 SM 上的自身执行
```

收益取决于两项变化的差。关键 kernel 若非常并行，缩到 16 SM 后的执行变慢可能抵消等待改善。后台吞吐也会因可用 SM 减少而下降。

实际调优建议同时记录：

| 指标 | 目的 |
|---|---|
| 关键任务 P50/P95/P99 延迟 | 验证 QoS 是否改善 |
| 关键 kernel 起始排队时间 | 区分等待与执行 |
| 背景任务吞吐 | 衡量隔离代价 |
| DRAM/L2/NVLink 利用率 | 发现非 SM 资源争用 |
| 分区 SM 数扫描 | 找到延迟/吞吐平衡点 |

Green context 与 stream priority 可以组合：资源分区控制“在哪组资源上运行”，priority 影响该上下文内部或可比较队列上的调度。但二者都不能替代正确的同步和容量规划。

**总结：Green Contexts 把 GPU 执行资源划分提升为应用可控制的 execution context，能够降低 SM 调度干扰，但最终 QoS 仍由分区大小、kernel 伸缩性和共享内存系统共同决定。**

[上一篇：4.5 Programmatic Dependent Launch](/posts/cuda-features-05-programmatic-dependent-launch/) · [下一篇：4.7 Lazy Loading](/posts/cuda-features-07-lazy-loading/)
