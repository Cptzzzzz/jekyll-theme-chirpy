# NCCL 专家课程源码证据审计

审计日期：2026-07-13  
课程范围：系列索引 + 36 章，共 37 篇  
NCCL 基准版本：`v2.22.3-1`，commit `178b6b7`

## 审计目标

本审计专门检查“文章是否把声明、路径或伪代码误当成实现”。通过条件是：

1. public header、ABI 和函数指针表明确标为接口声明。
2. 讨论运行时行为时，证据继续进入函数体、关键状态和下一跳。
3. `原始源码` 不包含未说明的裁剪；有省略时改称关键摘录并说明省略范围。
4. 文件路径后存在实际代码块，且代码后有调用链、实验或日志解释。
5. 当前仓库无法提供 vendor 实现时，文章明确写出证据边界。

## 自动检查结果

| 检查项 | 结果 |
|---|---:|
| 文章数 | 37 |
| Markdown code blocks | 1310 |
| C/C++/CUDA code blocks | 453 |
| 未闭合 code fence | 0 |
| 只有路径、同节没有代码的源码段 | 0 |
| 纯 public prototype 段 | 3 |
| 未标明性质的 public prototype 段 | 0 |
| `原始源码` 中未说明的人工省略 | 0 |

三段 public prototype 分别是第 5 章两组 collective API 和第 6 章 Group API。
它们现在都明确标为 public header 声明；第 5 章继续展示四个 Host wrapper，
第 6 章继续进入 `group.cc`、`group.h` 和 `enqueue.cc`。第 19 章源码注释中的
`...` 是上游注释用于描述路径序列的原文，不是文章裁剪。

## 逐章证据链

| 章节 | 主要实现证据链 | 审计结果 |
|---|---|---|
| 系列索引 | 课程导航、标准和实验结果索引，不声称实现细节 | 通过（索引页） |
| 01 运行版本 | ELF `DT_NEEDED` / loader / `/proc/maps` -> `ncclGetVersion` -> runtime probe | 通过 |
| 02 拓扑与 P2P | `topo.h` / topology XML -> NVML/sysfs -> P2P matrix 与并发实验 | 通过 |
| 03 异步完成 | `distributed_c10d.py` -> `ProcessGroupNCCL.cpp` event/wait -> NCCL stream enqueue | 通过（省略标签已校正） |
| 04 Communicator | `init.cc` init/destroy/query -> rank/device mapping -> duplicate GPU 反例 | 通过 |
| 05 Collective 契约 | `nccl.h.in` 声明 -> `collectives.cc` 完整 Host wrappers -> `ncclInfo` -> `ncclEnqueueCheck` / `ArgsCheck` | **已修复**：不再以声明代替实现 |
| 06 Group 与顺序 | `nccl.h.in` 声明 -> TLS group state -> nested depth -> enqueue / outermost launch | **已澄清**：声明与实现分层 |
| 07 Collective 分解 | enqueue task -> tuning fields -> `device/all_reduce.h` -> direct/decomposed trace | 通过 |
| 08 algbw / busbw | nccl-tests byte layout -> timer -> per-collective `GetBw` -> 手算复核 | 通过（省略标签已校正） |
| 09 Benchmark 方法 | nccl-tests warmup/sync/timer -> 独立重复 -> median/P95/CV | 通过 |
| 10 延迟带宽模型 | tuning latency/bandwidth table -> segmented fit -> residual 与反例 | 通过 |
| 11 回归归因 | 参数门控 -> controlled sweeps -> 路径日志 -> regression fingerprint | 通过 |
| 12 Ring AllReduce | graph ring order -> `runRing` step loop -> channel partition -> payload oracle | 通过 |
| 13 Tree / 双树 | `connect.cc` / `trees.cc` -> tree peers -> device tree kernel -> 交叉区间实验 | 通过 |
| 14 LL / LL128 / Simple | protocol types -> FIFO line/flag -> primitive wait/post -> planner selection | 通过（六处省略标签已校正） |
| 15 Channel / Chunk / Step | chunk planner -> CBD partition -> device slice -> kernel grid/block launch | 通过（四处省略标签已校正） |
| 16 Tuning / Plugin | `tuning.cc` model -> cost table -> final selection -> tuner ABI 与真实 callback | 通过（省略边界已校正） |
| 17 Bootstrap | API async job -> unique ID/root rendezvous -> bootstrap ring -> peerInfo/state publish | 通过（四段关键路径已显式标注） |
| 18 Topology Path | XML ingest -> node/link model -> path computation -> policy rewrite -> fixture replay | 通过（省略标签已校正） |
| 19 Graph Search | search pattern -> path bandwidth reservation -> rewind -> ring/tree/channel output | 通过 |
| 20 Transport | transport registry -> `canConnect` -> setup/connect -> connector publish | 通过 |
| 21 P2P / IPC / SHM | P2P capability -> CUDA IPC/direct pointer -> read/write mode -> SHM fallback | 通过 |
| 22 Enqueue / Plan | `taskAppend` -> plan construction -> work batch -> kernel launch / graph capture | 通过 |
| 23 Device Primitive | kernel dispatch -> primitive roles -> credit/step -> flag publication and wait | 通过 |
| 24 CPU Proxy / Socket | proxy op -> progress state -> NET request -> socket helper thread completion | 通过 |
| 25 Socket / IB / RoCE | NET backend -> socket/verbs request lifecycle -> CQ/test -> transport model | 通过 |
| 26 GDR / MR | GDR capability gate -> DMA-BUF/peermem -> MR registration/cache -> fallback | 通过 |
| 27 Multi-NIC / PXN | NIC scoring -> rail/crossNic policy -> PXN intermediate rank -> channel mapping | 通过 |
| 28 CollNet / NVLS | plugin ABI 声明 -> loader/capability gates -> proxy function-pointer call -> fallback | **已澄清**：ABI 不再写成 vendor 实现 |
| 29 ProcessGroupNCCL | PyTorch collective -> NCCL stream/event -> Work completion/error state | 通过 |
| 30 DDP Reducer | autograd hook -> bucket readiness -> ProcessGroup AllReduce -> optimizer visibility | 通过 |
| 31 ZeRO / FSDP | state ownership model -> reduce-scatter/all-gather trace -> memory/communication result | 通过 |
| 32 TP / PP / EP | multidimensional group construction -> per-dimension collective -> ordering experiment | 通过 |
| 33 故障根因 | sequence/watchdog/error propagation -> mismatch/crash injection -> rank-level evidence | 通过 |
| 34 可观测性 | NCCL debug/NVTX -> Nsight/flight recorder -> correlation keys -> diagnosis exercise | 通过 |
| 35 生产治理 | baseline schema -> statistical policy -> rollout/rollback model；不声称新的 NCCL 实现 | 通过（方法论章） |
| 36 端到端证明 | DDP Reducer -> ProcessGroupNCCL -> Host wrapper/planner -> kernel/proxy -> trace join | 通过（综合章） |

## 本次修复

1. 第 5 章把 public `nccl.h.in` 原型明确降级为“契约证据”，新增
   `AllReduce`、`Reduce`、`AllGather`、`ReduceScatter` 的 Host 函数体、`ncclInfo`
   字段解释和 `ncclEnqueueCheck` 下一跳。
2. 第 6 章明确 `ncclGroupStart/End` 是 header 声明，并指出后续三层实现位置。
3. 第 28 章明确 `ncclCollNet_v8_t` 是函数指针 ABI；vendor/SHARP 实现不在 NCCL 仓库，
   NCCL 侧证据闭环到实际 `iallreduce` 调用与 fallback gate。
4. 第 3、8、14、15、16、17、18 章把裁剪代码统一标成关键摘录，并写明省略内容。

## 后续写作门禁

新增或修改章节时，应把本审计与 `ARTICLE_STANDARD.md` 一起执行。任何“实现”结论如果只能
指向 header prototype、类型定义或文件路径，文章不能标记为 `VALIDATED`。
