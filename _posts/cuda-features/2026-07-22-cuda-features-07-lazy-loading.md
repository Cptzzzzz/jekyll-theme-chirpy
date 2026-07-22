---
title: 'CUDA Features 4.7：Lazy Loading'
author: Cptz
date: 2026-07-22 08:07:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, lazy-loading, module, startup]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.7 Lazy Loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)。

## 4.7.1 Introduction

CUDA Lazy Loading 延迟准备尚未使用的 module 和 kernel，减少进程初始化时间与初始设备内存占用。它改变的是“何时加载”，不改变 kernel 的计算结果；代价是第一次真正使用时可能出现一次性延迟。

## 4.7.2 Change History

CUDA 11.7 引入用户态 module 延迟加载；11.8 将粒度推进到 kernel function。CUDA 12.3 起，在支持平台上默认启用 lazy mode。版本演进意味着旧 benchmark 和新部署的首次 launch 路径可能不同。

## 4.7.3 Requirements

### 4.7.3.1 Runtime Version

应用使用的 CUDA Runtime 至少为 11.7，且应用依赖的静态/动态库也要链接支持 lazy loading 的 Runtime。进程中旧 Runtime 组件可能让相关 module 保持 eager 行为。

### 4.7.3.2 Driver Version

原章给出的最低 driver 分支为 515。实际部署还要满足 Toolkit 与 driver 的兼容规则，并在目标平台查询当前 loading mode。

### 4.7.3.3 Compiler

Lazy Loading 主要是 Runtime/Driver 能力，不要求必须用 11.7+ 编译器重新编译所有 device code。编译器版本仍决定生成代码与其他 CUDA 特性，不能由 lazy loading 替代兼容性检查。

### 4.7.3.4 Kernel / Module

包含 managed variables 的 module 可能需要 eager load，以便运行时建立变量状态。部分 module 依赖或平台限制也可能阻止细粒度延迟，因此“LAZY 模式”不承诺所有 module 都延迟到首次 kernel。

## 4.7.4 Usage

### 4.7.4.1 Enable / Disable

在进程启动前设置：

```bash
CUDA_MODULE_LOADING=LAZY ./app
CUDA_MODULE_LOADING=EAGER ./app
```

环境变量在 CUDA 初始化前读取。进程已经建立 context 后再改环境变量，不应期待已有状态重新加载。

### 4.7.4.2 Runtime Query

Driver API `cuModuleGetLoadingMode` 返回当前 eager/lazy mode。框架启动日志应记录查询结果，而不是只记录环境变量，因为最终行为还受 Runtime、driver 和平台支持影响。

### 4.7.4.3 Force Eager Loading

`cuModuleLoad` 本身不保证 module 内每个 kernel 都已准备。Driver 程序可对关键函数调用 `cuModuleGetFunction`，Runtime 程序可用 `cudaFuncGetAttributes` 触发 function 准备。

```text
加载 module
  -> 逐个解析关键 function
  -> 执行一次 warm-up launch
  -> 进入延迟敏感并发阶段
```

这样保留不常用代码的 lazy 优势，同时把热路径首次成本移到可控的预热阶段。

## 4.7.5 Potential Hazards

### 4.7.5.1 Concurrent Kernel Execution

历史程序有时发射一个占据 GPU 的长 kernel，再从 host 线程首次调用另一 kernel。首次函数加载可能需要运行时同步/资源准备，暴露程序对“长 kernel 与加载一定并发”的错误依赖。进入强并发区前主动准备所有会用到的 function。

程序正确性不能要求两个 kernel 必须并发；lazy loading 只是让这种未经保证的设计更容易失败。

### 4.7.5.2 Large Memory Allocations

应用若启动时占满几乎全部显存，延迟到运行期的 code/module state 可能没有空间，导致首次使用失败。显存预算要包括尚未加载的代码、runtime metadata、JIT 和库 workspace。

### 4.7.5.3 Performance Measurements

首次 kernel 测量可能包含 module/function loading、JIT 和 context 准备。应区分：

```text
cold start = 进程/上下文初始化 + 首次函数准备 + kernel
warm steady state = 预热后重复 kernel
```

报告吞吐时预热，报告冷启动时保留首次成本，并同时注明 loading mode。否则两个环境的数字不可比较。

**总结：Lazy Loading 以按需加载降低启动和初始占用，工程上必须用查询、预热和显存余量把首次使用成本变成可控事件。**

[上一篇：4.6 Green Contexts](/posts/cuda-features-06-green-contexts/) · [下一篇：4.8 Error Log Management](/posts/cuda-features-08-error-log-management/)
