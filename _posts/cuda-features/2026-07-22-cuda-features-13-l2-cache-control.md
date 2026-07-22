---
title: 'CUDA Features 4.13：L2 Cache Control'
author: Cptz
date: 2026-07-22 08:13:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, l2-cache, persisting-cache, memory]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.13 L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)。

重复访问的数据称为 persisting access，只访问一次或很少复用的数据更接近 streaming access。计算能力 8.0+ 可用 Runtime API 影响数据在 L2 中的持久倾向；libcu++ 还提供 `cuda::annotated_ptr` 表达类似访问属性。

## 4.13.1 为 Persisting Access 预留 L2

应用可从 L2 中 set aside 一部分容量。persisting 访问优先使用该区域；normal/streaming 访问只有在未被 persisting 数据使用时才能占用。预留上限受 `persistingL2CacheMaxSize` 约束：

```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, device);
size_t bytes = std::min<size_t>(prop.l2CacheSize * 3 / 4,
                                prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);
```

MIG 模式下该 set-aside 功能禁用。MPS 环境的设置方式与普通进程内调用不同，需要按 MPS 配置/环境变量管理。

## 4.13.2 Persisting Access Policy

access policy window 可以设置在 stream 或 CUDA Graph kernel node 上，字段包括：

- `base_ptr` 与 `num_bytes`：候选地址窗口；
- `hitRatio`：窗口访问中赋予 hit property 的期望比例；
- `hitProp`：被选中访问的属性，通常 persisting；
- `missProp`：其他访问的属性，常设 streaming 或 normal。

窗口大小不能超过 `accessPolicyMaxWindowSize`。

```cpp
cudaStreamAttrValue attr{};
attr.accessPolicyWindow.base_ptr  = hot;
attr.accessPolicyWindow.num_bytes = hot_bytes;
attr.accessPolicyWindow.hitRatio  = 0.6;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

`hitRatio=0.6` 不是承诺 60% 运行时 cache hit，而是提示约 60% 的窗口访问获得 `hitProp`。选取可随实现变化，也不能把它用作功能正确性的随机采样。

## 4.13.3 L2 Access Properties

- `persisting`：尽量保留，适合高复用热点。
- `streaming`：降低长期占据倾向，适合一次性流量。
- `normal`：恢复普通缓存策略。

它们是替换优先级提示，不是 cache pin。竞争、容量、地址映射和其他 workload 仍可能逐出数据。

## 4.13.4 Persistence 示例的含义

假设热点窗口大于 set-aside 容量，如果所有访问都标为 persisting，热点彼此争抢并产生 thrash。降低 `hitRatio`，让被赋予 persisting 的工作集约等于预留容量，往往更稳定。

可用近似关系指导初值：

$$
\text{window bytes} \times \text{hitRatio}
\lesssim \text{set-aside bytes}
$$

它不是严格 cache 容量公式，最终应扫描 ratio 并用 L2 hit rate、DRAM bytes 和 kernel time 验证。

## 4.13.5 恢复 Normal

任务结束后应撤销 access window，避免后续无关数据继承策略。可把 window 的 `num_bytes` 设为 0，或把访问属性恢复 normal。还可调用 `cudaCtxResetPersistingL2Cache` 请求清除 persisting 状态。

重置不是业务数据同步，调用前后仍需用 stream/event 管理 kernel 顺序。

## 4.13.6 管理 Set-Aside 利用率

多个并发 stream/graph 都请求 persisting 时，共享同一个有限 set-aside。每个任务单独按全容量配置会过度订阅。服务应从全局视角划分窗口和 ratio，并在模型切换后撤销旧策略。

## 4.13.7 查询 L2 属性

通过 `cudaGetDeviceProperties` 查询：L2 总大小、最大 persisting set-aside、最大 access policy window 等。不同 GPU、MIG 配置和运行模式下属性不同，不应写死为某一代产品规格。

## 4.13.8 控制预留大小

`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)` 控制 context 的预留量，size 需不超过设备上限。预留过大可能压缩普通/streaming 流量的有效 cache 空间；设置为最大值不一定让整体 workload 最快。

**总结：L2 Cache Control 通过 set-aside 与 access policy window 调整热点和流式流量的替换优先级，收益来自减少热点逐出，而不是把数据永久锁在 L2。**

[上一篇：4.12 Cluster Launch Control](/posts/cuda-features-12-cluster-launch-control/) · [下一篇：4.14 Memory Synchronization Domains](/posts/cuda-features-14-memory-synchronization-domains/)
