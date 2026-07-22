---
title: 'CUDA Features 4.2：CUDA Graphs'
author: Cptz
date: 2026-07-22 08:02:00 +0800
categories: [AI Infra, CUDA Features]
tags: [cuda, cuda-graphs, launch-overhead, runtime]
math: true
---

> 本文逐节对应 CUDA Programming Guide v13.3 的 [4.2 CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)，以中文技术转述覆盖原章的结构、更新、条件节点、内存节点和设备端发射语义。

传统 stream 模型由 CPU 一次次提交 kernel 和复制。操作很短、每轮拓扑重复时，主机提交开销会与设备执行时间相当。CUDA Graph 把操作及依赖预先描述为图，实例化时集中完成验证与准备，之后以较小、较稳定的成本重复发射。

## 4.2.1 图结构

CUDA graph 是有向无环图。节点代表工作，边代表依赖。没有依赖路径的节点可以并行，存在边的节点必须遵守相应顺序。图对象 `cudaGraph_t` 是可编辑描述，可执行图 `cudaGraphExec_t` 才是实例化后能发射的对象。

### 4.2.1.1 节点类型

原章涉及的主要节点包括：

| 节点 | 表达的工作 |
|---|---|
| kernel | 发射一个 CUDA kernel |
| memcpy / memset | 内存复制或填充 |
| host | 在 CPU 执行回调 |
| child graph | 嵌套另一个图的工作 |
| event record / wait | 与图内外 stream 建立事件顺序 |
| empty | 只用于组织依赖，不执行工作 |
| memory alloc / free | 将分配生命周期纳入图 |
| conditional | 根据设备侧条件执行 body graph |

不同节点有不同参数、更新限制和设备发射支持范围。empty node 虽不做计算，却可把多个前驱汇聚成一个逻辑边界，减少上层图构造复杂度。

### 4.2.1.2 Edge Data

普通边表示完整完成依赖。支持 edge data 的接口还能附带依赖类型和端口信息，表达更细的行为，例如 programmatic dependent launch。没有识别某种边语义的代码不应擅自把它当作普通边重建，否则可能改变执行时机。

## 4.2.2 构建与运行

完整生命周期是：

```text
创建/捕获 cudaGraph_t
  -> 实例化 cudaGraphExec_t
  -> 可选 cudaGraphUpload
  -> 一次或多次 cudaGraphLaunch
  -> 更新或销毁
```

原始 graph 与 executable graph 生命周期独立。实例化完成后可以销毁原 graph，而 executable 仍可执行；但以后若要做整图更新，应用通常需要保留或重新构造新的描述图。

### 4.2.2.1 图创建

CUDA 提供显式 Graph API 和 stream capture 两条路径。两者也能组合：显式建立总体结构，再将已有 stream 工作捕获成子图或加入依赖。

#### 4.2.2.1.1 Graph API

显式构图一般先创建图，再逐节点添加并记录返回的 node handle，最后添加边：

```cpp
cudaGraph_t graph;
cudaGraphNode_t load, compute, store;
cudaGraphCreate(&graph, 0);

cudaGraphAddMemcpyNode1D(&load, graph, nullptr, 0,
                         d_x, h_x, bytes, cudaMemcpyHostToDevice);
cudaGraphAddKernelNode(&compute, graph, &load, 1, &kernel_params);
cudaGraphAddMemcpyNode1D(&store, graph, &compute, 1,
                         h_y, d_y, bytes, cudaMemcpyDeviceToHost);
```

优点是节点身份稳定、拓扑直观、容易单节点更新；代价是要把原本顺序代码显式转换成节点和边。

#### 4.2.2.1.2 Stream Capture

捕获期间，提交到 origin stream 的操作不会立即发射，而是记录到正在构建的图。事件可把捕获传播到另一条 stream：被捕获 stream 记录事件，其他 stream 等待该事件后加入同一 capture，最后必须正确回到 origin stream 结束捕获。

捕获模式控制其他线程对潜在不安全 API 的容忍程度：global 最严格地监控进程中的冲突操作，thread-local 限制到当前线程，relaxed 允许更多调用但把正确性责任交给程序。

以下行为容易导致 capture invalidated：

- 调用会隐式同步整个设备的 legacy API；
- 对正在捕获的工作做阻塞查询；
- 错误使用 legacy default stream；
- 跨线程或跨 stream 形成未闭合依赖；
- 在不支持捕获的库/API 内部改变全局 CUDA 状态。

捕获失败后，必须结束并丢弃无效 capture，不能继续假定之前的操作已经正常入图。

#### 4.2.2.1.3 组合示例

典型做法是在 setup 阶段创建资源并预热，在 capture 阶段只记录稳定异步路径，然后实例化：

```cpp
warm_up_once();

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
enqueue_preprocess(stream);
enqueue_model(stream);
enqueue_postprocess(stream);
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&exec, graph, 0);
```

如果库调用内部用多条 stream，只要它正确用事件加入并退出 capture，最终仍可成为一个图。

### 4.2.2.2 图实例化

实例化把通用描述变成针对执行环境的可执行对象，并检查非法拓扑、参数与节点组合。实例化可能明显比一次 launch 贵，应从热路径移出。

可使用实例化 flag 改变行为，例如允许设备端 launch、上传、自动释放图内存等。flag 之间存在兼容限制，必须按照最终 launch 模式选择。

错误报告可以指出失败节点和日志信息。生产框架应把这部分保存到诊断日志，而不是只报告一个通用失败码。

### 4.2.2.3 图执行

`cudaGraphLaunch(exec, stream)` 把一次图执行排入 stream。图内仍按图边调度，图整体与该 stream 前后工作保持顺序。对同一个 executable 的并发 launch 能力和限制应按文档/版本处理，不能在实例仍运行时无条件修改它。

`cudaGraphUpload` 可提前完成首次 launch 的部分准备，减少关键路径抖动；upload 在指定 stream 中排序，发射前要保证其完成或通过同一 stream 依赖自然排序。

## 4.2.3 更新实例化图

Graph update 的目标是复用昂贵的实例化结果。更新不是任意修改：只有运行时能证明新描述与已准备资源兼容时，才会成功。

### 4.2.3.1 整图更新

`cudaGraphExecUpdate` 比较新的 `cudaGraph_t` 与现有 executable。成功时 executable 采用新参数/兼容变化；失败时返回 update result 和相关 error node。调用方应记录原因并重新实例化新图。

整图更新适合捕获路径每轮重建图，但拓扑实际上稳定的场景。要提高匹配成功率，构图顺序、依赖结构和节点类型应保持确定。

### 4.2.3.2 单节点更新

如果只变 kernel 参数、memcpy 地址/大小等，节点 setter 更直接。调用方必须保存 executable 对应的 node handle 或建立从业务节点到 CUDA node 的映射。

参数更新不会自动延长资源生命周期。新指针指向的内存必须在本轮图执行期间有效，尺寸和设备归属也需符合节点限制。

### 4.2.3.3 节点启用状态

部分节点类型可以在 executable 中 enable/disable。禁用节点意味着这次不执行节点工作，但它仍位于拓扑中，依赖结构并未删除。这适合可选 kernel 或 memset，不适合表达完全不同的 DAG。

### 4.2.3.4 更新限制

常见不可兼容变化包括节点类型改变、依赖拓扑改变、某些 kernel/function 属性改变、设备归属变化以及 graph memory node 的受限修改。运行时版本可能扩展能力，因此代码应按返回结果回退，而不是硬编码“某种更新一定成功”。

## 4.2.4 条件图节点

条件节点把控制流留在设备侧，避免每轮把判定值复制到 CPU、同步并由 CPU 再发射后续工作。支持 IF、WHILE 和 SWITCH。

### 4.2.4.1 条件句柄

conditional handle 保存条件值，与图及默认值配置关联。设备代码通过相应 API 写条件值，条件节点读取它决定 body。条件生产节点必须在图上先于条件节点，否则可能读取初始化值或上一轮值。

句柄值不是通用全局同步对象。应用要明确每次 launch 是否重新初始化，以及多次/并发执行是否共享状态。

### 4.2.4.2 Body Graph 要求

条件节点包含一个或多个 body graph。body 允许的节点类型、嵌套层次、设备发射和更新方式受到约束。body 的资源必须在外部 executable 生命周期内有效，且不能构造破坏 DAG/循环语义的依赖。

### 4.2.4.3 IF

IF 节点根据句柄是否为 0 决定是否执行 body；带两个 body 时可表达 if/else。没有选中的 body 不执行，但后继节点仍需等待条件节点按定义完成。

```text
produce_condition -> IF(cond) -> selected body -> continuation
```

### 4.2.4.4 WHILE

WHILE 在每次 body 结束后重新检查条件。body 必须负责更新条件，否则非零值会导致无限循环。循环次数在设备侧动态决定，适合收敛迭代，但单次图执行时间也会变得不固定。

### 4.2.4.5 SWITCH

SWITCH 把非负条件值映射到多个 body；越界值按文档规则不选择有效 body。它适合少量离散分支。大量形状或任意动态拓扑仍更适合上层调度和 graph cache。

## 4.2.5 Graph Memory Nodes

### 4.2.5.1 目的

普通分配发生在图外时，框架必须为最坏情况长期保留内存。图内 alloc/free 节点让 CUDA 看到精确生命周期，从而在不重叠的区间复用物理内存。

### 4.2.5.2 API 基础

alloc node 产生一个虚拟地址供后继节点使用，free node 在所有用户完成后释放。地址在实例化/执行语义下具有特定有效期，不能把它当作永久 `cudaMalloc` 指针。

#### 4.2.5.2.1 节点 API

显式 API 为分配节点指定 pool properties、access descriptors 和字节数，再把返回地址填入 kernel 或 memcpy 参数。free node 必须依赖最后一个用户。

#### 4.2.5.2.2 捕获

捕获 `cudaMallocAsync`/`cudaFreeAsync` 会生成相应 memory nodes。捕获把原本 stream 顺序转成图边，因此跨 stream 使用仍需事件依赖。

#### 4.2.5.2.3 图外访问与释放

图分配的内存可以按允许的 API 在图外访问或释放，但必须在执行顺序上位于 alloc 完成之后、free 之前。图外释放会影响图再次发射，应用必须遵守文档的重新分配与有效期规则。

#### 4.2.5.2.4 AutoFreeOnLaunch

`cudaGraphInstantiateFlagAutoFreeOnLaunch` 使前一次 launch 遗留的图分配在下一次 launch 前自动释放，便于没有显式 free node 的图反复运行。它并不允许应用在仍运行时提前失去引用，也会影响允许的设备发射/更新组合。

#### 4.2.5.2.5 子图中的内存节点

child graph 对 memory node 有额外限制，因为地址必须正确传播到父图，并让实例化器证明生命周期。构造复杂嵌套前应先检查节点是否允许，而不是到实例化时才处理通用错误。

### 4.2.5.3 优化内存复用

#### 4.2.5.3.1 图内地址复用

两个 allocation 生命周期没有重叠时，运行时可复用同一虚拟或物理资源。图边越精确，运行时越能确认生命周期；无必要的全局依赖会串行化，缺失依赖则会造成危险的重叠。

#### 4.2.5.3.2 物理内存管理与共享

不同 graph executable 的虚拟地址可以由底层图内存池支持，运行时在安全点复用物理页。销毁 graph 不一定立即把所有保留页归还 OS，具体取决于仍存 executable、stream 工作和 trim 行为。

### 4.2.5.4 性能考虑

分配节点并非每次都做昂贵 OS 分配；稳态性能依赖缓存和复用。应同时测量首次执行、稳定执行与显存高水位。

#### 4.2.5.4.1 首次 Launch 与 Upload

首次 launch 可能建立映射和准备物理内存。`cudaGraphUpload` 可把一部分工作移出关键路径，但 upload 自身也消耗时间和 stream 顺序位置。

### 4.2.5.5 物理内存占用

可通过 graph memory attributes 查询当前和高水位 reserved/used bytes，并通过 trim 请求归还未使用物理内存。高水位反映曾经需求，不代表当前仍被节点使用。

### 4.2.5.6 Peer Access

多 GPU 使用图内分配时，需要为目标设备配置访问权。设备 peer capability 与实际 allocation access 是两个层次。

#### 4.2.5.6.1 显式节点 API

显式 alloc node 的 access descriptor 可指定哪些设备以何种权限访问。图中在其他设备执行的节点必须具备相应映射。

#### 4.2.5.6.2 Stream Capture

捕获路径下，peer access 状态和异步分配所属设备必须在 capture/实例化规则内一致。不要在 capture 中临时改变会影响全局 peer 状态的配置。

## 4.2.6 Device Graph Launch

设备端发射让正在运行的图不回到 CPU 就能调度另一个预先准备的图，减少动态工作流的主机往返。

### 4.2.6.1 创建设备图

#### 4.2.6.1.1 要求

设备可发射图必须用相应 instantiate flag 创建，并满足允许的节点、拓扑和资源限制。host node 等无法在设备执行环境中成立的节点不能随意加入。

#### 4.2.6.1.2 Upload

设备端 launch 前，可执行图必须完成 device upload。主机可显式 upload，也可按文档允许的路径在首次使用前准备。设备代码不能把尚未准备的普通 executable 当作 device graph 发射。

#### 4.2.6.1.3 更新

更新 device graph 后需保证新状态已对设备 launch 环境可见，并遵守不能与正在执行实例竞态修改的要求。频繁更新可能抵消省下的主机发射成本。

### 4.2.6.2 发射模式

#### 4.2.6.2.1 Fire-and-forget

子图被加入当前 execution environment 后可独立推进；发射者不以该操作作为等待点。它适合不要求当前图尾部等待的派生工作，但资源和结果仍需由明确依赖管理。

#### 4.2.6.2.2 Tail Launch

tail graph 在当前图及其相应环境工作完成后执行，天然形成接续阶段。self-tail launch 可让图把自己排到尾部实现设备侧循环，但必须有终止条件，否则会无限延续。

#### 4.2.6.2.3 Sibling Launch

同一发射环境中产生的多个图可作为 siblings。它们之间没有凭空产生的先后关系；需要顺序时，应使用合适的 tail 结构或图内依赖。

## 4.2.7 使用 Graph API 的设计方式

工程上通常建立三层：业务执行计划、CUDA graph 描述、可执行图缓存。缓存 key 至少可能包括设备、shape、batch、kernel 变体、内存布局和条件配置。

```text
请求形状/配置
  -> 查 graph cache
  -> 命中：更新参数并 launch
  -> 未命中：capture/build -> instantiate -> warmup -> 缓存
```

图优化的是重复路径。如果每次拓扑都不同，缓存膨胀、实例化和内存占用会成为新问题。

## 4.2.8 CUDA User Objects

捕获可能让资源寿命超出创建它的 C++ 作用域。CUDA User Object 用引用计数把资源绑定到 graph、executable 和正在运行的实例；最后一个引用释放后，运行时调用析构 callback。

引用可以 move 到 graph，也可按 API retain/release。析构 callback 运行环境有限制，不能在其中调用会非法重入 CUDA 的操作。回调应只做安全的主机资源释放或把清理转交其他线程。

这套机制解决的是生命周期，不是设备同步。即使 user object 仍存活，资源内容是否可访问仍取决于图依赖和执行完成。

**总结：CUDA Graph 把重复工作流从逐次命令提交变为可实例化、可更新的执行对象；真正的工程难点是稳定拓扑、资源生命周期、更新回退和设备侧动态控制。**

[上一篇：4.1 Unified Memory](/posts/cuda-features-01-unified-memory/) · [下一篇：4.3 Stream-Ordered Memory Allocator](/posts/cuda-features-03-stream-ordered-memory-allocator/)
