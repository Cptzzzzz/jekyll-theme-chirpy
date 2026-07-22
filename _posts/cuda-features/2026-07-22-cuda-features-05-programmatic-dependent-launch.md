---
title: 'CUDA Features 4.5：程序化依赖发射与同步'
author: Cptz
date: 2026-07-22 08:05:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, dependent-launch, synchronization, hopper]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.5 Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

## 4.5.1 Background

同一 stream 中，后一个 kernel 通常要等前一个 kernel 完成才能开始。程序化依赖发射（PDL）允许主 kernel 在主动声明“发射条件已满足”后，让次 kernel 的独立前导部分先运行；次 kernel 到真正依赖数据的位置再等待主 kernel 完成。

## 4.5.2 API Description

该机制要求计算能力 9.0 及以上设备。主 kernel 在已经不再需要阻止后继发射的位置调用 `cudaTriggerProgrammaticLaunchCompletion()`。如果它没有显式调用，退出时也会隐式触发，因此不会永久丢失发射完成信号。

次 kernel 的 launch 需要设置 programmatic stream serialization 属性。它可以先做与主 kernel 无关的工作，到依赖边界调用 `cudaGridDependencySynchronize()`；该调用之后才可读取主 kernel 的结果。

```cpp
__global__ void primary(Input in, SharedData* out) {
  finish_launch_sensitive_phase();
  cudaTriggerProgrammaticLaunchCompletion();
  write_primary_result(in, out);
}

__global__ void secondary(SharedData* in) {
  prepare_independent_state();
  cudaGridDependencySynchronize();
  consume_primary_result(in);
}
```

这里存在两个不同事件：允许后继 grid 被发射，以及主 grid 的内存结果可消费。把 trigger 当作普通 release fence 会过早读取数据；把 dependency synchronize 放在次 kernel 开头则会丢掉本可重叠的前导工作。

### 机会性执行

PDL 提供的是允许重叠，不保证一定重叠。资源不足、调度策略或其他高优先级工作都可能让次 kernel 晚于预期开始。因此程序不能依靠并发发生来保证活性，也不能让两个 kernel 互相等待形成只有并发才能解除的死锁。

## 4.5.3 CUDA Graph 中的 PDL

图边可以表达 programmatic dependency。上游节点触发 launch completion，下游节点运行独立部分并在依赖点同步。图仍需满足边类型和节点属性限制；如果下游的所有工作都依赖上游，普通完整依赖边往往更简单。

### 适用场景与边界

- 次 kernel 有足够长且真正独立的 prologue，重叠才可能覆盖调度空洞。
- 主 kernel 应在最早安全点 trigger，但不得在仍会影响后继发射正确性的阶段之前触发。
- 性能收益依赖资源余量；两个高 occupancy kernel 通常很难有效重叠。
- 正确性必须在“完全不重叠”的合法调度下仍然成立。

**总结：PDL 将同一 stream 的完整完成依赖拆成“允许发射”和“允许消费”两个阶段，以机会性重叠换取更短的流水间隙。**

[上一篇：4.4 Cooperative Groups](/posts/cuda-features-04-cooperative-groups/) · [下一篇：4.6 Green Contexts](/posts/cuda-features-06-green-contexts/)
