---
title: 'CUDA Features 4.8：Error Log Management'
author: Cptz
date: 2026-07-22 08:08:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, logging, diagnostics, driver-api]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.8 Error Log Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html)。

## 4.8.1 Background

错误码只告诉调用方某次 API 成功或失败，往往缺少驱动内部的诊断上下文。Error Log Management 为 Driver API 提供结构化日志输出、回调和缓冲区导出，使库作者能在不改每一处错误处理逻辑的情况下收集更多信息。

## 4.8.2 Activation

最直接的方式是设置 `CUDA_LOG_FILE`：

```bash
CUDA_LOG_FILE=stderr ./app
CUDA_LOG_FILE=stdout ./app
CUDA_LOG_FILE=/var/log/my-cuda-app.log ./app
```

## 4.8.3 Output

日志包含时间、线程 ID、来源、严重级别、API 入口点和消息等字段。它用于诊断，不替代 API 返回值检查，也不承诺记录所有 Driver API 或每一种内部错误。

## 4.8.4 API Description

Driver API 可以注册和注销日志 callback。callback 适合接入应用已有观测系统，但必须保持轻量、线程安全，并避免在回调中递归调用会再次产生日志的接口。

应用还可以遍历当前日志项，或把日志 dump 到文件/内存。文档规定内部最多保留 100 条记录，内存 dump 上限为 25,600 字节；它是有界诊断缓存，不应被当作永久审计日志。

```text
CUDA Driver 产生诊断
  -> 有界日志缓冲区
  -> callback / iterator / file dump / memory dump
  -> 应用日志系统
```

## 4.8.5 Limitations and Known Issues

- 当前能力面向 Driver API，而不是覆盖所有 Runtime API 行为。
- flag 参数当前应传 0，为未来扩展保留。
- 日志文本当前为英文，应用不应解析自然语言来驱动控制流。
- 记录数量和 dump 容量有上限，故障风暴中较早记录可能被覆盖。
- 日志输出可能影响时序，复现竞态时应同时保留无日志运行证据。

推荐在错误报告中同时保存 API 返回码、设备/驱动版本、调用线程、关键参数以及日志 dump。这样日志是额外证据，而非唯一证据。

**总结：Error Log Management 为 CUDA Driver 错误增加了有界、结构化的诊断通道，但程序正确性仍必须以返回码和明确的错误处理为基础。**

[上一篇：4.7 Lazy Loading](/posts/cuda-features-07-lazy-loading/) · [下一篇：4.9 Asynchronous Barriers](/posts/cuda-features-09-asynchronous-barriers/)
