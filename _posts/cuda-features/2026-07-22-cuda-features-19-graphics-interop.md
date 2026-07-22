---
title: 'CUDA Features 4.19：CUDA 与图形及外部 API 互操作'
author: Cptz
date: 2026-07-22 08:19:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, vulkan, opengl, direct3d, nvsci, interop]
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.19 CUDA Interoperability with APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html)。互操作 API 的 handle ownership 和平台差异非常细，生产代码应同时核对 CUDA 与对端 API 的规范。

互操作的目标是让 CUDA kernel 直接读写其他 GPU API 创建的数据，避免 staging 到 CPU。原章分两种模型：

1. Graphics Interoperability：OpenGL/Direct3D 资源注册到 CUDA，反复 map/unmap。
2. External Resource Interoperability：Vulkan、D3D12、NvSci 等导出 OS-level memory/sync handle，CUDA 导入为 external object。

共享 memory 只解决“访问同一数据”，共享 semaphore/fence 才解决“谁先访问”。两者缺一不可。

## 4.19.1 Graphics Interoperability

通用生命周期：

```text
图形 API 创建 resource
  -> CUDA register
  -> 每帧 map
  -> 取得 device pointer 或 CUDA array
  -> CUDA kernel
  -> unmap（把所有权交还图形 API）
  -> 多帧重复
  -> unregister
```

register 是相对昂贵的 setup 操作，应长期保留；map/unmap 是每次所有权转换。资源被 CUDA map 期间，图形 API 不应并发读写，除非规范提供另一套明确同步。

map flag 表达意图：read-only、write-discard 或 none。`write-discard` 允许驱动为整资源覆写优化，不能在 kernel 中依赖旧内容。

### 4.19.1.1 OpenGL

OpenGL buffer 可通过 `cudaGraphicsGLRegisterBuffer` 注册，image/texture/renderbuffer 使用 `cudaGraphicsGLRegisterImage`。map 后：

- buffer 用 `cudaGraphicsResourceGetMappedPointer` 取得 pointer/size；
- image 用 `cudaGraphicsSubResourceGetMappedArray` 取得 CUDA array，再用 surface/texture object 访问。

简化的 VBO 帧循环：

```cpp
cudaGraphicsMapResources(1, &vbo_resource, stream);
float4* positions;
size_t bytes;
cudaGraphicsResourceGetMappedPointer(
    reinterpret_cast<void**>(&positions), &bytes, vbo_resource);

generate_vertices<<<grid, block, 0, stream>>>(positions, count);
cudaGraphicsUnmapResources(1, &vbo_resource, stream);
// OpenGL 在 CUDA 归还资源后绘制 VBO。
```

CUDA device 必须与持有 OpenGL context 的 GPU 匹配。多 GPU/显示器环境不能默认 device 0；使用相应 GL device query 选择。注册后图形 API 重建/resize resource 时，先 unregister 旧对象并重新注册。

### 4.19.1.2 Direct3D

Direct3D 9/10/11 资源可通过各版本 graphics interop API 注册。CUDA device 要和 D3D adapter 匹配，资源类型、usage 与 format 必须在支持集合内。

map 后的 buffer 取 device pointer，texture 取 CUDA array。unmap 是把 CUDA 修改发布给后续 D3D 工作的重要边界。D3D context/device 销毁前应先清理 CUDA registration。

### 4.19.1.3 SLI 配置

AFR/SLI 中当前帧和下一帧可能由不同 GPU 处理。CUDA 提供查询与图形 context 关联的 current-frame/next-frame device 集合。应用要在正确 device 上执行 CUDA 工作，否则发生 peer copy、注册失败或资源不可访问。

frame ownership 可随帧变化，因此不能启动时只选一次固定 GPU 后永久假设。现代系统是否启用相关模式也应运行时查询。

## 4.19.2 External Resource Interoperability

外部资源模型把两个独立对象导入 CUDA：

- external memory：底层 allocation，之后映射为 linear buffer 或 mipmapped array；
- external semaphore：binary/timeline semaphore 或 fence，用 stream wait/signal 排序。

导入描述符中的 handle type、size、offset、dedicated flag 必须与导出端创建方式一致。错误的 size/format 不会被“同一个 handle”自动修正。

## 4.19.2.1 Vulkan

### 4.19.2.1.1 设置 Vulkan Device

创建 Vulkan instance/device 时启用所需 external memory 与 external semaphore 扩展，并为 allocation/semaphore 声明可导出的 handle type。POSIX 常用 opaque FD，Windows 使用 opaque Win32/NT handle；timeline semaphore 还需对应 timeline 能力。

并非任意 `VkDeviceMemory` 都能事后导出。export info 要进入创建时的 `pNext` 链，并选择 Vulkan 查询为 compatible 的 handle type。

### 4.19.2.1.2 用 UUID 匹配 CUDA Device

比较 Vulkan physical device UUID 与 `cudaDeviceProp::uuid`，选择同一物理 GPU。在 Windows 还可能涉及 LUID/device node mask。只比较枚举 ordinal 不可靠，因为两套 API 枚举顺序可能不同。

```text
遍历 Vulkan physical devices，读取 UUID
  -> 遍历 CUDA devices，读取 UUID
  -> UUID 相等才建立互操作 pair
```

### 4.19.2.1.3 导出 Vulkan Memory

Vulkan 创建 buffer/image，查询 memory requirements，选择兼容 memory type，以 external-memory export info 分配并 bind。导出的 allocation size、dedicated allocation 与资源 offset 要保存给 CUDA。

共享整个 `VkDeviceMemory` allocation 时，CUDA 可能映射其中一个 offset 区域；应用必须保证不越过 Vulkan 分配边界，也不能与同 allocation 中其他资源冲突。

### 4.19.2.1.4 导出 Vulkan 同步对象

Vulkan GPU 调用异步，需要 semaphore/fence 表达队列顺序。binary semaphore 只有 signaled/unsignaled 状态；timeline semaphore 有 64 位单调 value，可在同一对象上表达连续帧依赖。

创建 semaphore 时配置 export handle type，timeline 还要在 `pNext` 指定类型。Vulkan signal 的 value 与 CUDA wait 的 value 必须一致。

### 4.19.2.1.5 导入 Memory Object

CUDA 填充 `cudaExternalMemoryHandleDesc`，指定 handle type、FD/Win32 handle、总 size 和 dedicated flag，再 `cudaImportExternalMemory`。

handle ownership 因类型/平台不同：某些 FD 在成功导入后由 CUDA 接管，某些 Win32 handle 仍由应用关闭。必须按对应 descriptor 文档处理，不能统一写“导入后都 close”或“永不 close”。

`cudaDestroyExternalMemory` 销毁 CUDA wrapper，不等于允许 Vulkan 在 CUDA 仍有映射/工作时释放 backing。

### 4.19.2.1.6 映射 Buffer

`cudaExternalMemoryGetMappedBuffer` 接收 offset/size，返回 device pointer。offset 与 size 要匹配 Vulkan suballocation 并满足对齐。用完的 mapped pointer 通过 `cudaFree` 释放，external memory object 则另行 destroy。

pointer 的 `cudaFree` 只清理 CUDA mapping，不释放 Vulkan `VkDeviceMemory`。

### 4.19.2.1.7 映射 Mipmapped Array

image 通过 `cudaExternalMemoryGetMappedMipmappedArray` 映射，描述 channel format、extent、flags、mip levels 和 offset。它们必须与 Vulkan image format/layout 和 allocation 完全匹配；render target 对应 CUDA color attachment flag。

用完调用 `cudaFreeMipmappedArray`。CUDA kernel 使用期间图像 layout/ownership 必须由 Vulkan barrier 与 external semaphore 协调。

### 4.19.2.1.8 导入同步对象

`cudaImportExternalSemaphore` 根据 opaque FD/Win32、timeline 等类型创建 `cudaExternalSemaphore_t`。导入前保存 handle 类型和 timeline/binary 属性，不能把 timeline handle 按 binary 导入。

### 4.19.2.1.9 Signal/Wait

`cudaWaitExternalSemaphoresAsync` 与 `cudaSignalExternalSemaphoresAsync` 在 CUDA stream 排序。常见一帧：

```text
Vulkan 渲染/生产
  -> Vulkan signal ready=N
  -> CUDA stream wait N
  -> CUDA kernel 修改共享 memory
  -> CUDA stream signal done=N
  -> Vulkan queue wait N 后消费
```

timeline value 必须单调推进；binary semaphore signal/wait 后的重用按两端 API 规则执行。memory visibility 依赖对端 queue barrier/layout transition 与 semaphore 配合，不只是时间顺序。

### 4.19.2.1.10 与 OpenGL 组合

OpenGL 可通过支持的外部 memory/semaphore 扩展导入相同 OS handle，形成 Vulkan/CUDA/OpenGL 共享。每一对 API 的 device identity、handle type 和 ownership 都要匹配；同步链必须明确唯一生产者和下一消费者，避免两个 API 同时拥有写权限。

## 4.19.2.2 Direct3D External Interop

### 4.19.2.2.1 匹配 LUID

Windows 上比较 D3D adapter LUID 与 CUDA device 的 LUID，并核对 node mask。LUID 只在当前系统启动实例中有意义，不应持久化到另一台机器或重启后使用。

### 4.19.2.2.2 导入 Memory

D3D12 heap 和 committed resource 使用不同 CUDA handle type。创建时要允许 shared handle；dedicated committed resource 设置 `cudaExternalMemoryDedicated`。可按 Win32 handle 或 named handle 导入。

导入 size 要覆盖对象真实 allocation，handle 的关闭责任按 Win32 类型处理。D3D11 resource 也有对应 keyed mutex/外部资源路径，不能与 D3D12 fence 语义混用。

### 4.19.2.2.3 映射 Buffer

外部 buffer 描述的 offset/size 必须与 D3D12 placement 一致。返回 pointer 最终 `cudaFree`；D3D resource/heap 仍由 D3D 管理。

### 4.19.2.2.4 映射 Mipmapped Array

extent、format、mip level 与 D3D resource description 一致。可作为 render target 的资源要设置 `cudaArrayColorAttachment`。释放 CUDA mipmapped array 不代表释放 D3D texture。

### 4.19.2.2.5 导入同步对象

D3D12 fence 可作为 external semaphore 导入，fence value 对应 timeline 语义。导入的 CUDA semaphore 与原 D3D fence 共享底层同步对象。

### 4.19.2.2.6 Signal/Wait

CUDA stream wait 某个 fence value 后访问共享 resource，完成后 signal 新 value；D3D command queue wait 该 value 再使用。D3D resource state transition 与 UAV/cache barrier 仍要正确设置，external fence 不替代 resource state 管理。

## 4.19.2.3 NvSci Interop

NvSciBuf/NvSciSync 面向跨进程、跨引擎的 NVIDIA SoC/安全相关通信。属性协商先由参与者声明需求，再由 NvSci reconcile 得到兼容对象，CUDA 导入已协调对象。

### 4.19.2.3.1 导入 Memory

`NvSciBufObj` 导入为 CUDA external memory。CUDA 要读取 reconciled attributes，确认 size、权限、GPU ID 以及是否需要 cache coherency。权限请求不能超过 NvSciBuf 授予的权限。

### 4.19.2.3.2 映射 Buffer

按 reconciled size/offset 映射 linear pointer。若属性指出需要 GPU cache coherency，访问顺序必须使用匹配的 NvSciSync object，不能只靠进程间消息通知。

### 4.19.2.3.3 映射 Mipmapped Array

根据 NvSciBuf image attributes 构造 channel/extent/mipmapped 描述。plane、pitch、format 和权限必须与协商结果一致。映射后由 CUDA array/surface API 使用。

### 4.19.2.3.4 导入同步对象

NvSciSync 属性协商区分 signaler 与 waiter 能力。CUDA 导入时必须选择与自身角色匹配的 external semaphore 类型和权限。

### 4.19.2.3.5 Signal/Wait

CUDA 用 external semaphore async API 等待或发信号，另一参与者通过 NvSciSync fence 对应操作衔接。每次迭代要传递正确 fence/generation，避免复用旧信号造成提前消费。

## 共同清理顺序

1. 停止提交新工作。
2. 双方通过 semaphore/fence 确认最后使用完成。
3. 释放 CUDA mapped buffer/mipmapped array。
4. destroy CUDA external semaphore/memory wrapper。
5. 按 ownership 关闭 OS handle。
6. 最后由创建 API 释放原始 memory/resource/sync object。

**总结：CUDA 互操作的本质是同一物理资源的多 API 映射加跨队列所有权交接；device 匹配、描述符一致性、handle ownership 和外部 semaphore 缺一不可。**

[上一篇：4.18 CUDA Dynamic Parallelism](/posts/cuda-features-18-dynamic-parallelism/) · [下一篇：4.20 Driver Entry Point Access](/posts/cuda-features-20-driver-entry-point-access/)
