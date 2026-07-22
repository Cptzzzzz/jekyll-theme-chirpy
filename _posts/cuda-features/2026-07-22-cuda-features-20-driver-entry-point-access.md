---
title: 'CUDA Features 4.20：Driver Entry Point Access'
author: Cptz
date: 2026-07-22 08:20:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, driver-api, abi, dynamic-loading]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.20 Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html)。

## 4.20.1 引言

从 CUDA 11.3 起，应用可按名称和 ABI 版本取得 Driver API 函数地址，作用类似 POSIX `dlsym` 或 Windows `GetProcAddress`，但 CUDA API 额外理解 Driver 函数版本和 default stream 语义。

用途包括：

- 通过 Driver API `cuGetProcAddress` 动态取函数；
- 通过 Runtime API 取 Driver entry point；
- 选择 legacy default stream 或 per-thread default stream 版本；
- 用较旧 Toolkit 编译，但在较新 driver 上机会式使用新能力。

动态取得符号不等于忽略 ABI。函数 pointer 类型、请求版本和实际 driver 支持必须同时匹配。

## 4.20.2 Driver Function Typedefs

Toolkit include 目录提供 Driver API 对应的函数指针 typedef headers。`cudaTypedefs.h` 覆盖核心 Driver API，图形/EGL 等互操作接口有相应 typedef header。

typedef 名通常包含 API 引入/ABI 版本，例如：

```cpp
PFN_cuMemAlloc_v2000  old_alloc;
PFN_cuMemAlloc_v3020  current_alloc;
```

Driver 导出符号以 `_v2` 等后缀区分发生过 ABI/语义变化的版本，而 typedef 使用数值版本标识其对应 CUDA release。第一次版本通常没有 `_v1` 符号后缀。

必须使用与目标 ABI 对应的正式 typedef，不能把 `void*` 强转为“看起来参数差不多”的自定义签名。调用约定、结构体版本或参数宽度不同都会破坏 ABI。

## 4.20.3 取得 Driver Function

### 4.20.3.1 使用 Driver API

`cuGetProcAddress` 接收 symbol 名、输出 pointer、CUDA ABI version、flags 和 query result。假设需要 CUDA 10.1 引入的 `cuStreamBeginCapture` 新签名：

```cpp
PFN_cuStreamBeginCapture_v10010 begin_capture = nullptr;
CUdriverProcAddressQueryResult query{};

CUresult status = cuGetProcAddress(
    "cuStreamBeginCapture",
    reinterpret_cast<void**>(&begin_capture),
    10010,
    CU_GET_PROC_ADDRESS_DEFAULT,
    &query);

if (status == CUDA_SUCCESS &&
    query == CU_GET_PROC_ADDRESS_SUCCESS && begin_capture) {
  begin_capture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
}
```

请求字符串通常是基础 API 名，CUDA 根据 version/flags 选择正确实现；不要自行猜测应该在字符串末尾拼 `_v2`，以该 API 文档为准。

### 4.20.3.2 使用 Runtime API

Runtime 提供 `cudaGetDriverEntryPoint`，以及可明确 version 的 `cudaGetDriverEntryPointByVersion`。返回的是 Driver API 函数 pointer，调用结果类型仍是 `CUresult`/Driver ABI，不会自动变成 Runtime API。

库使用 Runtime 入口可以避免直接静态调用目标 Driver symbol，但仍应包含正确 typedef header，并检查 Runtime 返回码和 driver query status。

### 4.20.3.3 取得 Per-Thread Default Stream 版本

同一个 Driver API 可能有 legacy default stream 与 per-thread default stream（PTDS）语义。flags 可请求相应版本：

- legacy：stream 0 具有传统跨 stream 隐式同步行为；
- per-thread：每个 host thread 有自己的默认 stream。

如果函数带显式 stream 参数，传 0 时使用哪种语义取决于所取 entry point。动态库必须把自己的 stream 模式显式固定，不能受最终应用的编译宏偶然影响。

### 4.20.3.4 从旧 Toolkit 访问新特性

只要编译环境能声明新函数的 ABI（例如自带兼容 typedef 或更新 typedef header），并且运行 driver 足够新，就可动态查询，不需要静态链接时解析该 symbol。

正确流程：

```text
检查 cuDriverGetVersion >= 新 API 最低版本
  -> 用“该 typedef 对应的固定版本号”查询 symbol
  -> 检查 CUresult
  -> 检查 query result
  -> 检查 pointer 非空
  -> 调用；否则走旧能力 fallback
```

这用于可选能力，不应在没有 fallback 时假装应用仍兼容旧 driver。部署最低 driver 仍要明确声明。

Minor release 新增的 API ABI 可能尚未通过 `cuda.h` 的通用宏切换到新版本，但 typedef header 会提供精确版本 typedef。原章以 `cuDeviceGetUuid` 的 v2 版本说明应按 11040 请求，而不是只看未带版本的源代码名称。

## 4.20.4 `cuGetProcAddress` 指南

最重要的规则：传给 `cuGetProcAddress` 的 CUDA version 必须与选定 typedef 的版本一致并写为该 ABI 的固定值。

不要传：

- 编译时的通用 `CUDA_VERSION`：它可能比目标函数 ABI 新，选出不同版本；
- `cuDriverGetVersion` 的动态结果：driver 越新，错误地请求到更新 ABI 的风险越高。

`cuDriverGetVersion` 的用途是先判断 driver 是否至少支持目标 API，而不是决定函数 pointer 的类型。pointer 的静态类型与 version 参数必须是一对。

应用还应缓存查询结果，避免在热路径反复解析；按 driver/context 生命周期安全地初始化函数表；为每个 optional symbol 单独设计 fallback。

### 4.20.4.1 Runtime API 指南

`cudaGetDriverEntryPointByVersion` 同样遵守固定 ABI version 原则。Runtime 返回的错误与 `CUdriverProcAddressQueryResult` 提供不同层的信息，两者都要记录。

库若可能被不同 default-stream 模式的应用加载，应显式请求目标 flag，且不要在同一函数表里混用 PTDS 与 legacy entry point。

## 4.20.5 判断查询失败原因

失败分两层：

1. `CUresult` 表示 API 使用错误，例如输出 pointer 为 null、flag 非法或未正确初始化 Driver。
2. `CUdriverProcAddressQueryResult` 表示 symbol 查找结果。

主要 query result：

- `CU_GET_PROC_ADDRESS_SUCCESS`：找到符合版本/flag 的 entry point；仍检查 pointer 非空。
- `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`：driver 中知道该 symbol，但调用方请求的 CUDA ABI version 早于它的引入版本。
- `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`：当前 driver 没有 symbol，也可能是名称大小写/拼写错误。

`VERSION_NOT_SUFFICIENT` 与“driver 太旧”不同：前者可能说明 driver 足够新，但传入 version 太低。`SYMBOL_NOT_FOUND` 既可能是部署 driver 老，也可能只是把 `cu...` 错写成 `CU...`。

诊断日志至少输出 symbol、固定 ABI version、flags、`cuDriverGetVersion`、`CUresult` 和 query result。只打印“函数不可用”会掩盖配置错误。

**总结：Driver Entry Point Access 提供 CUDA-aware 的动态符号解析；其安全性来自函数 typedef、固定 ABI version、stream flag 和 driver capability 四者精确匹配。**

[上一篇：4.19 CUDA Interoperability](/posts/cuda-features-19-graphics-interop/) · [返回 CUDA Part 4 官方目录](https://docs.nvidia.com/cuda/cuda-programming-guide/part4.html)
