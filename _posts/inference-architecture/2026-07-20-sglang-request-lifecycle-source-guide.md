---
title: 'SGLang 请求处理全流程：从 Tokenizer 到 Scheduler、Radix 与输出'
author: Cptz
date: 2026-07-20 00:20:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, architecture, scheduler, kv cache, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_architecture_posts.py. -->

> 本文按固定源码版本逐段跟踪一个 SGLang 请求。先用 30 秒建立全局路径，再按模块解释问题、Input、实现、Output、性能收益、代价与正确性不变量。

[另一篇：vLLM 请求全流程](/posts/vllm-request-lifecycle-source-guide/) · [面试题系列](/posts/vllm-sglang-interview-series/) · [源码实验手册](/posts/vllm-sglang-interview-source-labs/)

## 30 秒全流程概览

```text
HTTP / OpenAI Chat Completion
  -> TokenizerManager：建立 ReqState、tokenize、多模态预处理
  -> ZMQ transport：发送 TokenizedGenerateReqInput
  -> Scheduler：DTO 转 Req，管理 waiting/running 与 batch 状态机
  -> RadixCache + SchedulePolicy：prefix match、cache-aware 排序
  -> PrefillAdder：token/KV/request-row 多资源 admission
  -> ScheduleBatch / TpModelWorker / ModelRunner：EXTEND 或 DECODE forward、sample
  -> Scheduler result processor：提交 token、Radix/KV 状态与停止条件
  -> DetokenizerManager：独立进程增量 detokenize
  -> TokenizerManager：按 rid 唤醒 HTTP coroutine，返回 JSON / SSE
```

一句话理解：SGLang 用 Tokenizer、Scheduler、Detokenizer 多进程隔离输入输出 CPU 工作，把 Radix prefix reuse、准入与 batch 状态集中在 Scheduler；overlap 模式再用 batch snapshot、FutureMap 和 CUDA event 保证跨轮异步执行仍能正确提交。

下面进入完整源码路径。文中的类、函数和状态均固定到文章开头声明的 fork commit；伪代码用于压缩控制流，不替代链接中的源码条件分支。

---

本文固定到已运行验证的 fork commit
[`9529c2e96`](https://github.com/Cptzzzzz/sglang/commit/9529c2e96351f3b1d5715eaa89989b20723d60b5)。
该提交基于 `a3194d358`，只增加默认关闭的 `SGLANG_FLOW_TRACE` 注释与日志，不改变推理语义。

本文是生命周期骨架：沿一个 OpenAI-compatible generation 请求，跟踪它从 HTTP 输入、tokenization、
Radix prefix 匹配、EXTEND/DECODE 调度、GPU forward、投机验证、detokenization 到 HTTP 返回的完整过程。
需要深入某一模块时，进入下面的独立模块文档；其中的字段、控制流和边界均由当前固定 commit 源码确认。

需要把源码机制和真实性能现象对应起来时，继续阅读
[`../VLLM_SGLANG_SOURCE_LABS.md`](/posts/vllm-sglang-interview-source-labs/)：其中用 A100 实验深入验证 causal
prefill、Radix prefix、continuous batching、chunk/mixed 调度、CUDA Graph、NGRAM 和 decode retraction。

## 阅读方式：先主线，再模块

| 学习模块 | 它解决的核心问题 | 深入文档 |
| --- | --- | --- |
| 请求与输出管线 | Tokenizer/Scheduler/Detokenizer 多进程如何交接对象并隔离 CPU 抖动 | [`modules/REQUEST_PIPELINE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/REQUEST_PIPELINE.md) |
| Scheduler | Policy、PrefillAdder 和 batch state 如何共同完成多资源 admission | [`modules/SCHEDULER.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/SCHEDULER.md) |
| Radix Cache/Memory Pool | prefix tree、request row、physical KV slots 如何匹配、共享、锁定和淘汰 | [`modules/RADIX_CACHE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/RADIX_CACHE.md) |
| Overlap/ModelRunner | `ScheduleBatch` 如何变成 `ForwardBatch`，并安全跨流、跨轮重叠 | [`modules/MODEL_EXECUTION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/MODEL_EXECUTION.md) |
| Speculative Decoding | EAGLE V2 的 draft、verify、accepted-path compact 和 draft-extend 如何闭环 | [`modules/SPECULATIVE_DECODING.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/SPECULATIVE_DECODING.md) |

每个模块统一回答：

```text
为什么需要 -> 上游 Input/不变量 -> 源码实际步骤/数据结构
-> 给下游的 Output/新不变量 -> 改变了什么物理开销
-> 改善哪个指标 -> 代价/退化条件 -> 如何实验验证
```

源码链接不是装饰：模块文档依靠实际函数确认对象变形、资源 ownership、异步 lifetime 和 fallback，
避免把其他版本或另一个框架的设计套到当前 SGLang 上。

## 导航

| 主线 | 章节 |
| --- | --- |
| 基础模型 | [0. 指标](#0-阅读约定与指标)、[1. 状态所有权](#1-进程拓扑与状态所有权)、[2. 对象变形](#2-一个请求在源码中的对象变形) |
| 完整请求 | [3. 生命周期总览](#3-完整请求生命周期总览)、[4. Tokenizer/IPC](#4-t0-t3tokenizermanager-的入口和-ipc)、[5. DTO 到 Req](#5-t4scheduler-将-dto-变成-cache-aware-req)、[6. Req 状态](#6-req-的关键状态与长度语义) |
| 缓存与调度 | [7. RadixAttention](#7-t5radixattention-如何查找共享-prefix)、[8. Policy/Admission](#8-t5-t7schedulepolicy-与-prefilladder)、[9. Batch 状态机](#9-scheduler-的-batch-centric-状态机)、[10. Overlap](#10-普通-event-loop-与-overlap-event-loop) |
| GPU 与输出 | [11. Batch/内存池](#11-t7-t10schedulebatch内存池与-runner-输入)、[12. Runner](#12-t8-t12tpmodelworkermodelrunnergraph-与-sample)、[13. 输出/Detokenizer](#13-t13-t17提交输出与独立-detokenizer)、[14. 普通 decode](#14-一轮普通-decode-的精确时序) |
| 深入优化 | [15. 投机解码](#15-speculative-decodingsglang-eagle-v2-完整流程)、[16. 调度权衡](#16-scheduler-设计延伸性能目标与冲突)、[17. 优化总表](#17-常见优化手段总表)、[18. 并行/P-D](#18-并行和-pd-分离) |
| 验证与复习 | [19. 性能验证](#19-如何验证设计真的达到性能目标)、[20. 正确性不变量](#20-关键正确性不变量)、[21. 阅读顺序](#21-推荐源码阅读顺序) |

## 0. 阅读约定与指标

### 0.1 本文关注的主路径

- API：OpenAI-compatible chat/generate 最终进入 `TokenizerManager.generate_request()`。
- Runtime：SRT 多进程架构。
- Scheduler：普通 event loop 与 overlap event loop 都解释，重点是当前复杂的 overlap 路径。
- Model：decoder-only generation 为主，multimodal、structured output、LoRA、Mamba/Hybrid cache
  作为相同资源模型的扩展。
- Spec decode：重点解释当前 EAGLE V2 和 NGRAM 的 draft/verify/commit 路径。

### 0.2 用什么判断优化是否有效

| 指标 | 定义 | SGLang 中的主要影响模块 |
| --- | --- | --- |
| TTFT | 请求到第一个可见 token | Tokenizer、waiting queue、Radix hit、EXTEND |
| ITL | 相邻可见 token 间隔 | DECODE batch、overlap、kernel、detokenizer |
| TPOT | 平均每输出 token 时间 | target decode、spec acceptance、通信 |
| tokens/s | 实例 token 吞吐 | continuous batching、kernel、并行、batch shape |
| p95/p99 | 尾延迟 | 长 EXTEND、retraction、graph miss、队列和 IPC |
| goodput | 满足 TTFT/ITL SLO 的吞吐 | 调度、准入和执行共同决定 |
| cache hit | 可复用 prefix token 比例 | Radix policy、workload locality、page alignment |
| retraction | 因 decode KV 不足撤回的请求 | admission 预测与实际输出增长偏差 |

真实 HTTP 控制流验证见 [`RUNTIME_VALIDATION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/RUNTIME_VALIDATION.md)。为了观察源码路径，
验证环境使用了追踪友好的配置，不代表生产 benchmark。后文只给出能从设计推出和实验验证的性能
方向，不编造固定百分比。

## 1. 进程拓扑与状态所有权

```text
HTTP/API process
  HTTP/OpenAI adapter
    -> TokenizerManager
       - tokenizer / multimodal preprocessing
       - rid_to_state / asyncio Event
       - request dispatch and response wait
              | TokenizedGenerateReqInput
              v
Scheduler process per rank/group
  Scheduler
    -> waiting_queue / running_batch / chunked_req
    -> SchedulePolicy / PrefillAdder
    -> RadixCache / ReqToTokenPool / TokenToKV allocator
    -> TpModelWorker or speculative worker
              | BatchTokenIDOutput
              v
Detokenizer process
  DetokenizerManager
    -> incremental decode / stop trimming
              | BatchStrOutput
              v
TokenizerManager -> HTTP JSON/SSE
```

各模块的状态边界：

| 模块 | 主要状态 | 解决的问题 |
| --- | --- | --- |
| `TokenizerManager` | tokenizer、`rid_to_state`、async event、输入预处理 | 隔离 CPU 抖动并连接 HTTP coroutine |
| `Scheduler` | waiting/running/chunked、result queue、stream/future | 串行维护准入、batch 和设备资源一致性 |
| `SchedulePolicy` | waiting 排序、prefix locality | 在 cache reuse、公平性和优先级间选择 |
| `PrefillAdder` | 本轮 prefill/KV/future decode 预算 | 防止只看当前 chunk 导致后续 decode OOM |
| `RadixCache` | token prefix tree、node lock、evictable KV | 将共享 prefix 显式变成可复用状态 |
| memory pools | request row、token->physical KV slot、真实 K/V tensor | 连接逻辑 request 与物理显存 |
| `TpModelWorker` | `ScheduleBatch -> ForwardBatch`、rank 协调 | 隔离 Scheduler 与模型/并行实现 |
| `ModelRunner` | model、attention backend、graph、sampler | 执行 rank-local tensor 计算 |
| `DetokenizerManager` | per-rid decode offset、稳定 text suffix | 防止字符串工作阻塞 Scheduler |

SGLang 的显著特点是：Tokenizer 和 Detokenizer 都是 Scheduler 之外的独立工作单元，Scheduler
专注于 GPU launch、batch 和内存状态。

## 2. 一个请求在源码中的对象变形

| 阶段 | 对象 | 关键内容 |
| --- | --- | --- |
| HTTP 入站 | `GenerateReqInput` | text/messages 转换结果、stream、sampling、grammar |
| tokenize 后 | `TokenizedGenerateReqInput` | `rid`、input IDs/embeds、params、LoRA、priority |
| Scheduler 内 | `Req` | output IDs、prefix indices、tree node、KV length、finish state |
| 一轮计划 | `ScheduleBatch` | req list、ForwardMode、pool indices、seq lens、spec info |
| worker 输入 | `ForwardBatch` | packed tensor、positions、attention/sampling metadata |
| GPU 结果 | `GenerationBatchResult` | logits、next IDs、accept lens、graph flag、next draft input |
| Scheduler 出站 | `BatchTokenIDOutput` | per-rid token delta、finish reason、cache/spec/latency stats |
| Detokenizer 出站 | `BatchStrOutput` | 稳定增量 text、token IDs、meta |
| API 结果 | Python dict -> JSON/SSE | OpenAI/native response |

`rid` 贯穿全链路，但 overlap 下同一个 `ScheduleBatch` 对象会被下一轮 filter/merge/prepare 修改，
所以 result queue 必须保存用于结果处理的 snapshot；GPU tensor 还需要跨 stream 的生命周期保护。

## 3. 完整请求生命周期总览

```text
T0  HTTP request enters TokenizerManager.generate_request
T1  normalize arguments and create rid_to_state
T2  tokenize / preprocess multimodal input
T3  dispatch TokenizedGenerateReqInput over IPC
T4  Scheduler constructs Req and puts it in waiting_queue
T5  SchedulePolicy computes priority and Radix prefix match
T6  PrefillAdder performs admission and chooses EXTEND/chunk
T7  ScheduleBatch.prepare_for_extend builds pool/sequence metadata
T8  Scheduler.run_batch dispatches to TpModelWorker
T9  ForwardBatch.init_new converts scheduler state to runner input
T10 ModelRunner chooses graph/eager and runs attention/model
T11 sample one token, or speculative draft + target verify
T12 GenerationBatchResult is copied/relayed back
T13 Scheduler commits Req/KV/grammar/finish state
T14 OutputStreamer sends BatchTokenIDOutput
T15 Detokenizer emits stable BatchStrOutput
T16 TokenizerManager matches rid_to_state and wakes HTTP coroutine
T17 repeat DECODE or release resources and finish JSON/SSE
```

第 5 到 14 步按 generation iteration 重复。新请求能在旧请求 decode 期间进入下一批，这就是
continuous batching。overlap 模式还会让本轮 GPU forward 与上一轮 CPU 结果处理同时发生。

### 3.1 生命周期 I/O contract

| 流程 | 要解决的问题 | Input | 核心方法 | 传给下游的 Output | 主要性能维度 |
| --- | --- | --- | --- | --- | --- |
| T0-T3 Tokenizer/IPC | HTTP/text/MM 不能进入 Scheduler 热循环 | `GenerateReqInput`、tokenizer、MM payload | 先建 `rid_to_state`，tokenize/SHM wrap/dispatch | `TokenizedGenerateReqInput` | API CPU/TTFT、IPC、Scheduler 隔离 |
| T4 建立 `Req` | transport DTO 缺少 cache/batch 状态 | tokenized DTO、server/cache config | 构造 token/KV/priority/grammar/P-D state | waiting `Req` | queue TTFT、后续状态正确性 |
| T5 policy/prefix | waiting 顺序需兼顾公平与 locality | waiting reqs、Radix tree、priority | LPM/DFS/FCFS 等排序和 prefix match | ordered candidates + matched indices/node | cache hit、Scheduler CPU、公平性 |
| T6-T7 admission | 当前 suffix 和未来 decode共同争用多种 pool | candidates、KV/row/SWA/Mamba budgets | `PrefillAdder` ledger、lock double-check、chunk | execution-ready EXTEND `ScheduleBatch` | TTFT、retraction、KV capacity、ITL |
| T8-T9 worker contract | Scheduler 对象不能直接喂模型 | `ScheduleBatch`、pool mappings、future state | resolve future、`ForwardBatch.init_new()`、rank dispatch | tensor-level `ForwardBatch` | host gap、metadata/H2D、并行同步 |
| T10-T12 GPU/result | 计算 KV/logits/token并跨流返回 | forward tensor、weights/KV、sampling/spec info | graph/eager model、sample/verify、FutureMap、async D2H | `GenerationBatchResult` + next draft payload | kernel time、TPOT/ITL、overlap |
| T13 commit | allocated/乐观 state要变成 accepted truth | launch batch snapshot + result | update `Req`/KV/grammar/cache/finish | committed scheduler state | CPU launch gap、KV 回收、correctness |
| T14-T16 输出闭环 | token delta 要恢复稳定文本并路由 rid | committed `Req` delta、detok state | batch token IPC、incremental decode、event wake | `BatchStrOutput` -> JSON/SSE | perceived TTFT/ITL、CPU/IPC、backpressure |
| T17 循环/结束 | 继续 DECODE 或释放 row/KV/tree lock/state | updated `Req`、finish/abort | merge/filter/re-enter loop 或 cleanup | 下一 `ScheduleBatch` 或最终 response | capacity、cache reuse、泄漏/retraction风险 |

表中 Output 是下游真正消费的 contract。每个模块的字段、状态变化和 fallback 见顶部对应模块文档。

## 4. T0-T3：TokenizerManager 的入口和 IPC

### 4.1 `generate_request()` 先建立返回通道

[`TokenizerManager.generate_request()`][tokenizer] 的关键顺序：

1. 归一化 batch 和参数，设置默认 priority；
2. `_init_req_state()` 为每个 `rid` 建立 `ReqState`、event 和输出缓存；
3. 记录请求、等待 pause condition、在 reader lock 下校验 LoRA；
4. `_tokenize_one_request()` 处理 text、input IDs、input embeds、多模态 placeholder；
5. `_send_one_request()` 跨 IPC 发给 Scheduler；
6. `_wait_one_response()` 等待同一 `rid` 的结果并 yield。

与 vLLM 类似，先建立 `rid_to_state` 再发送请求可以避免快速结果先于 consumer 到达。若 tokenization
阶段异常，代码会主动清理尚未由正常返回路径删除的 state，防止长期字典泄漏。

### 4.2 `_send_one_request()` 的 transport contract

[`_send_one_request()`][tokenizer]：

- 标记 API dispatch 时间；
- multimodal shared-memory feature 用 `wrap_shm_features()` 包装；
- 不适合直接 transport 的字段先 pickle wrapper；
- `_dispatch_to_scheduler()` 发送 DTO；
- 恢复本地 time stats 引用。

**设计结果**：tokenizer 和多模态 CPU 工作不会进入 Scheduler 的每轮 launch 热路径；代价是 IPC、
序列化、shared-memory ownership 和 late response 的处理复杂度。

## 5. T4：Scheduler 将 DTO 变成 cache-aware `Req`

[`Scheduler.handle_generate_request()`][scheduler] 构造 [`Req`][schedule-batch]，字段包括：

- `origin_input_ids`、`sampling_params`、EOS/stop；
- stream/logprob/hidden-state 输出选项；
- LoRA、priority、routing key、`extra_key`；
- multimodal input、session、P/D disaggregation 参数；
- `req_pool_idx`、prefix/cache/KV length 等后续状态。

普通请求进入 `waiting_queue`。grammar 请求可能先等待 grammar 编译；P/D decode 请求可能等待 KV
transfer。此时 `Req` 尚未等于“可立即 forward”，它还要经过 prefix matching、request-row admission
和 physical KV 预算。

## 6. `Req` 的关键状态与长度语义

理解 SGLang 最容易出错的地方是把 token list、逻辑序列和物理 KV 当成同一个长度。

| 字段 | 含义 |
| --- | --- |
| `origin_input_ids` | 原始输入 token |
| `output_ids` | 已提交给请求语义的输出 token |
| `prefix_indices` | 已命中/暂存 KV 对应的 physical slot indices |
| `req_pool_idx` | `ReqToTokenPool` 中该请求的行号 |
| `kv_committed_len` | 已确认可作为真实序列状态的 KV 长度 |
| `kv_allocated_len` | 已占用物理 slot 的长度，spec 时可能大于 committed |
| `last_node` | 当前锁住的 Radix node |
| `cache_protected_len` | 受 tree ownership 保护且不能重复 free 的边界 |
| `extend_range` | 本轮 EXTEND 要计算的 token 区间 |
| `inflight_middle_chunks` | 尚未完成结果处理的中间 prefill chunk 数 |

投机解码尤其依赖 `allocated >= committed`：target 可以为整棵 draft tree 分配临时 KV，最后只提交
accepted path，剩余 overshoot 必须回收或覆盖。

## 7. T5：RadixAttention 如何查找共享 prefix

### 7.1 Radix Tree 表达的不是文本，而是模型状态等价类

[`RadixCache`][radix-cache] 用压缩 radix tree 映射：

```text
RadixKey(token IDs, extra_key) -> physical KV slot indices
```

`extra_key` 用来隔离不允许共享状态的请求，例如不同 LoRA、cache salt 或其他上下文版本。只有 token
相同且额外命名空间兼容，KV 才是可复用的。

### 7.2 `match_prefix()` 的实际动作

1. EAGLE 场景可把 key 转成 bigram view；
2. 按 `page_size` 向下对齐；
3. 从 root 依据每页首 key 进入 child；
4. `RadixKey.match()` 查找公共前缀；长公共段使用指数窗口加二分定位分歧，避免 Python per-token loop；
5. 若匹配终止在压缩边内部，`_split_node()` 暴露精确边界；
6. 拼接沿途 `value` 得到 physical KV indices，更新 access time。

返回的 `last_device_node` 之后会被 request lock。匹配可以改变 tree 结构，但不会复制物理 KV。

### 7.3 insert、lock、finish 与 eviction

| 操作 | 状态变化 |
| --- | --- |
| `insert()` | 将新 token path 合并到 tree，共享段只保留一份逻辑节点 |
| `inc_lock_ref()` | 从 leaf 到 root 增加 lock，节点从 evictable 变 protected |
| `cache_unfinished_req()` | 把 chunk/未完成请求的稳定 KV 插入 tree，更新 row mapping 与 lock |
| `cache_finished_req()` | 插入已 committed KV、free duplicate/unaligned tail、释放 request lock |
| `dec_lock_ref()` | lock 归零的节点重新变成 evictable |
| `evict()` | 按配置策略从未锁 leaf 开始释放 physical KV，父节点成为 leaf 后可继续 |

`cache_unfinished_req()` 中的 duplicate free 很关键：当新 path 与 tree 已有 prefix 合并后，请求原来
私有计算出来的重复 slot 必须归还，并把 `ReqToTokenPool` 行重写为 tree 中 canonical indices。

### 7.4 RadixAttention 的最终性能结果

若 prompt 长度 `P`，命中 `H` 个可复用且 page-aligned token，本轮实际 EXTEND suffix 约为 `P-H`。
高共享 system prompt、few-shot、agent history workload 中，结果应表现为：

- cached token 上升；
- prefill GPU token/FLOPs 与 TTFT 下降；
- 同一 prefix KV 只占一份 physical slot；
- cache-aware scheduling 还能让本 batch 新计算 prefix 被后续请求复用。

低复用随机 prompt 下，tree match/insert/split/lock/evict 是额外 CPU 成本。LPM 并不保证所有 workload
更快，必须把高/低 reuse 分开 benchmark。

## 8. T5-T7：SchedulePolicy 与 PrefillAdder

### 8.1 SchedulePolicy 决定先看谁

[`SchedulePolicy`][schedule-policy] 支持：

| 类型 | 策略 | 目标 |
| --- | --- | --- |
| cache-aware | LPM | 优先最长 prefix match，直接减少当前计算 |
| cache-aware | DFS-weight | 聚集到共享 prefix 子树，提高一批请求的 locality |
| cache-agnostic | FCFS | 简单、公平、低 CPU 开销 |
| cache-agnostic | LOF | 依据预期输出长度排序 |
| cache-agnostic | RANDOM | 打散顺序 |
| cache-agnostic | ROUTING_KEY | 与 running batch 中常见 routing key 聚类 |

LPM 在 waiting queue 大于源码阈值时退化为 FCFS，避免 prefix match/sort 自身成为 Scheduler 热点。
in-batch prefix caching 还用模拟 Radix Tree 识别等待请求之间的公共 prefix，先调度一个 producer，
暂时降低 siblings 的优先级，让后续能命中刚产生的 KV。

### 8.2 PrefillAdder 不是简单的 batch append

[`PrefillAdder`][schedule-policy] 同时维护多种预算：

```text
rem_input_tokens       单轮 max_prefill_tokens
rem_chunk_tokens       chunked prefill 剩余额度
rem_total_tokens       available + evictable - future reservations
cur_rem_tokens         当前立即可用/可淘汰空间
rem_swa_tokens         sliding-window 独立预算
rem_mamba_slots        hybrid state 独立预算
request-row capacity   ReqToTokenPool 可用行
```

`rem_total_token_offset` 不只扣本轮 input，还根据 running requests 的剩余输出长度和
`new_token_ratio` 预留未来 decode 空间。新增 request 还要计入：

- suffix EXTEND tokens；
- 预估 `max_new_tokens`；
- page alignment overhead；
- Mamba shared-gap/state slot；
- hybrid SWA window；
- host/hierarchical cache load-back。

这解决了一个常见错误：当前 prefill chunk 能放下，不代表请求进入 decode 后仍有空间持续增长。

### 8.3 Admission 的执行过程

`Scheduler._get_new_batch_prefill_raw()` 大致执行：

```text
refresh grammar / hierarchical cache events
calculate waiting priority and prefix matches
construct PrefillAdder from current pools and running batch
continue an existing chunked_req first

for req in waiting_queue:
    enforce request-row, LoRA, priority and delay constraints
    req.init_next_round_input(tree_cache)
    result = adder.add_one_req(req)
    if no token/resource: stop or preempt according to policy

remove admitted reqs from waiting_queue
create ScheduleBatch
prepare_for_extend()
```

`add_one_req()` 在持有 matched-node lock 前后各检查一次预算，因为 lock 会让原本 evictable 的 KV 变成
protected，实际可用空间会下降。这是并发资源账本中的典型 double-check。

### 8.4 Chunked prefill 的实现与结果

当 suffix 超过 `rem_chunk_tokens`：

1. chunk size 向 page size/确定性 split alignment 对齐；
2. `Req.extend_range` 只覆盖当前 chunk；
3. `new_chunked_req` 被单独保存；
4. 本轮不为尚未开始的输出预留正常采样结果；
5. chunk 完成后 `cache_unfinished_req()` 暂存已稳定 KV；
6. 下轮先继续该 chunked request，直到最后 chunk 才合入 running decode batch。

**结果**：把一次长 EXTEND GPU 时间上界从整段 prompt 降为 chunk 的时间，减少并发 decode 的
head-of-line blocking 和 ITL/p99；代价是长 prompt 自身多次调度/metadata/kernel，TTFT 可能增加。

## 9. Scheduler 的 batch-centric 状态机

SGLang 与 vLLM 的统一 token-debt 表达不同，它显式维护：

- `waiting_queue`：尚未准入或被 retract 的请求；
- `running_batch`：可以继续 DECODE 的请求；
- `last_batch`：上一轮执行计划，EXTEND 完成后要合并；
- `chunked_req`：尚未完成 prompt 的单独请求；
- `result_queue`：overlap 模式中已 launch、待 CPU commit 的 batch/result。

[`get_next_batch_to_run()`][scheduler] 的主要顺序：

1. 处理 timeout/abort；
2. 将上一轮完成的 EXTEND 请求合入 running batch，但排除未完成 chunk；
3. filter finished request；
4. 优先调用 `get_new_batch_prefill()` 尝试新 EXTEND；
5. 有新 prefill 就运行它；否则 `update_running_batch()` 后运行 DECODE；
6. DP attention/MLP sync、ngram embedding 等做最终计划修饰；
7. 返回 `NextBatchPlan(batch_to_run, running_batch)`。

默认“prefill first”有利于 waiting TTFT，但长 prefill 可能干扰 decode，所以必须结合 chunk、prefill
delayer、mixed chunk 和资源阈值理解，不能只看一个 if 分支。

## 10. 普通 event loop 与 overlap event loop

### 10.1 普通循环

[`event_loop_normal()`][scheduler] 是最容易理解的同步基线：

```text
receive requests
process input
get next batch
run batch
process same batch result
set last_batch
```

正确性简单，但 CPU schedule、GPU forward、D2H 和结果处理串行，任何一段空隙都会成为 GPU idle gap。

### 10.2 overlap 循环

[`event_loop_overlap()`][scheduler] 用 `result_queue` 将阶段流水化：

```text
receive new requests
apply WAR barrier
schedule current batch
launch current batch on forward stream
enqueue (batch snapshot, result)
process previous batch result on schedule/CPU side
launch delayed sample if dependency ready
```

所以 trace 中可能出现“下一轮 Step 10”早于“上一轮 Step 14”，这是预期 pipeline，而不是日志乱序。

### 10.3 WAR barrier 为什么存在

Scheduler 会改写 request-row、seq lens、block mapping 等共享 buffer，而上一轮 GPU forward 可能仍在读。
`_apply_war_barrier()`：

- 快路径等待 runner 发布的 `read_done_event`；
- 没有细粒度 event 时退化为 `schedule_stream.wait_stream(forward_stream)`。

对 spec V2，最后读取共享 buffer 的阶段是 `draft_extend`，不是 target verify，所以 barrier owner 指向
draft runner。过早写会产生 write-after-read race，结果可能不是 crash，而是静默错误 token/KV。

### 10.4 FutureMap 与两轮 tensor 生命周期

普通 decode 的 next token、spec decode 的 `next_draft_input` 通过 `FutureMap` relay 到下一轮，避免
Scheduler 为读取一个 token 强制 D2H 同步。`run_batch()` 还会：

- 为 spec worker 提供 `on_publish` callback；
- 在 verify 结束、draft_extend 开始前发布 next seq lens，让下一轮 schedule 尽早准备；
- 用 batch record ring 保留跨 stream tensor 引用约两轮；
- 在 copy stream 做 D2H，和下一次 forward 重叠。

**最终结果**：host scheduling、上一轮 D2H/result processing 与本轮 GPU 计算重叠，降低 launch gap、
提高 tokens/s。代价是 snapshot、fence、future version、tensor lifetime 和异常清理复杂度显著增加。

## 11. T7-T10：ScheduleBatch、内存池与 runner 输入

### 11.1 ScheduleBatch 是 Scheduler 和 worker 的边界对象

[`ScheduleBatch`][schedule-batch] 同时携带：

- `reqs` 和 `ForwardMode`；
- `req_pool_indices`、`seq_lens`、`out_cache_loc`；
- EXTEND 的 prefix/extend lengths；
- sampling info、grammar、multimodal inputs；
- speculative `spec_info`；
- DP/PP、chunk、hierarchical cache 和 metrics metadata。

`filter_batch()` 必须同步 filter req list、CPU/GPU seq lens、pool indices、sampling/grammar/spec state；
`merge_batch()` 必须按相同顺序 concat。少更新一个伴随 tensor 就会发生 request 与 token 错配。

### 11.2 两层 memory pool

[`ReqToTokenPool`][memory-pool] 是二维映射：

```text
req_pool_idx x logical token position -> physical KV slot
```

每个 active request 占一行；第 0 行保留给 CUDA Graph padding 的 dummy request。真正 K/V tensor 由
`TokenToKVPoolAllocator` 和具体 MHA/MLA/DSA/SWA/Mamba pool 管理。

Radix Tree 保存“哪些 token prefix 指向哪些 slot”；ReqToTokenPool 保存“当前 request 各位置指向哪些
slot”；KV pool 保存“slot 中真正的 tensor”。三层职责不能混为一谈。

### 11.3 EXTEND 与 DECODE allocation

`prepare_for_extend()` 为 suffix 分配连续的逻辑输出位置并更新 request row。普通
`prepare_for_decode()` 每请求分配一个 token slot，然后乐观增加 seq lens、`kv_committed_len` 和
`kv_allocated_len`。spec decode 则由 `spec_prepare_for_decode()` 按 tree/draft width 分配更多 headroom。

如果普通 decode 下一轮空间不足：

1. 先从 Radix Tree evict 未锁 cache；
2. 仍不足则 [`ScheduleBatch.retract_decode()`][schedule-batch] 按策略撤回请求；
3. 释放其 row/KV，不插入 tree 以立即得到空间；
4. request 回 waiting，之后可能重算；
5. 即使只剩一个请求仍放不下，则显式 abort，避免 Scheduler crash。

retraction 类似 vLLM preemption，是 OOM 保护，不是常态优化。它会增加 recompute、TTFT/ITL 和 p99。

## 12. T8-T12：TpModelWorker、ModelRunner、graph 与 sample

### 12.1 TpModelWorker 将 Scheduler contract 变成模型 contract

[`TpModelWorker.forward_batch_generation()`][tp-worker]：

1. `ForwardBatch.init_new(batch, model_runner)`；
2. 初始化/标记 attention metadata；
3. last PP rank 调用 `ModelRunner.forward()`；
4. normal generation 调用 sampler，spec target verify 则跳过普通 sampler；
5. 包装 `GenerationBatchResult`，协调 PP/TP 输出。

Scheduler 不需要知道具体模型 forward tensor 签名，runner 也不决定 waiting queue 公平性。

### 12.2 ModelRunner 的执行分派

[`ModelRunner._forward_raw()`][model-runner] 选择：

| 路径 | 条件 | 结果 |
| --- | --- | --- |
| decode CUDA Graph | mode/shape/backend 已 capture | replay 静态 graph，降低 host launch |
| prefill/piecewise graph | EXTEND/VERIFY 可由 runner 承载 | 部分捕获动态 prefill |
| split prefill | 特殊 layer/context split | 分段执行长 prefill |
| eager runner | graph 不适配或关闭 | 完整动态能力，launch 开销较高 |

graph eligibility 之前还要处理 DP/MLP-sync padding、attention metadata 和 Mamba deferred COW/clear。
CUDA Graph 只减少 host/driver overhead；大 GEMM/attention FLOPs 不会消失。静态 padding、capture pool
显存和多 shape capture 是代价。

### 12.3 sampling 与 structured output

`ModelRunner.sample()` 先更新 regex/grammar vocab mask 和 logits bias，再执行 sampler/logprob。overlap
模式可能延迟 sample，因为本轮 grammar state 依赖上一轮 CPU commit；`launch_batch_sample_if_needed()`
在依赖就绪后才执行，并立即清理 closure 持有的大 logits/mask，避免稳定 VRAM leak。

## 13. T13-T17：提交输出与独立 Detokenizer

### 13.1 Scheduler result processor

[`Scheduler.process_batch_result()`][scheduler] 根据 ForwardMode 分发到 prefill/decode/idle/prebuilt
处理器。普通 prefill：

- 中间 chunk 只更新 logprob/timing，不 stream pseudo next token；
- 最后 chunk append 首个 sampled token，更新 finish；
- 未完成 request 可 `cache_unfinished_req()` 并进入 running decode；
- 完成 request `release_kv_cache()`。

普通 decode：

- 将输出统一为 per-request `List[List[int]]`；
- spec path 一次可能 extend 多 token；
- 更新 `Req.output_ids`、finish、grammar、logprob、KV；
- 输出 spec acceptance/graph/cache metrics；
- `OutputStreamer` 发送 token delta。

### 13.2 为什么 detokenize 要独立进程

[`DetokenizerManager.event_loop()`][detokenizer] 接收 `BatchTokenIDOutput`。它维护每请求的 decode
offset，只对新增且文本稳定的 suffix 解码，再生成 `BatchStrOutput`。这样避免：

- Scheduler 每轮做 tokenizer 字符串工作；
- 每个 token 都全量 decode 导致接近 O(n^2) 的重复复制；
- UTF-8 不完整 byte、stop string、special token 处理阻塞下次 GPU launch。

代价是一次额外 IPC、detokenizer backlog 和进程 watchdog。拆进程提供并行机会，不保证任何负载下
单请求都更低延迟。

### 13.3 回到 HTTP coroutine

TokenizerManager 的 `handle_loop()` 收到 `BatchStrOutput`，`_handle_batch_output()` 用 `rid` 查
`rid_to_state`，合并 meta、logprob、cache/spec stats 并唤醒 event。慢流式客户端积压时，代码会合并
多个 delta chunk，限制 queue item 数；积压仍会抬高该请求 p99 ITL，因此需要 socket timeout 和
per-request buffer 上限。

## 14. 一轮普通 DECODE 的精确时序

假设 prompt 已完成并已有最后 token `y_t`：

1. `get_next_batch_to_run()` 没有更优先 EXTEND，选择 running batch。
2. `update_running_batch()` filter finished，并检查下轮 decode memory。
3. 空间不足先 evict cache，再 retract request。
4. `prepare_for_decode()` 为每请求分配一个 KV slot，seq lens 乐观 `+1`。
5. overlap 模式从 FutureMap 解析真实 `input_ids`；非 overlap 使用上一轮 token tensor。
6. attention 根据 request row 和 physical slot 读旧 KV、写 `y_t` 的新 KV。
7. logits -> grammar/bias -> sampler 得到 `y_{t+1}`。
8. result 通过 FutureMap 供下一轮 forward，同时 D2H 给上一轮 result processor。
9. CPU append `y_{t+1}`、检查 finish、stream；下一轮再处理它对应的 KV。

这说明 sampling token、logical seq lens、committed KV 和 visible text 的更新时间不同；overlap 只是把
它们流水化，不能省略 commit/fence。

## 15. Speculative Decoding：SGLang EAGLE V2 完整流程

### 15.1 支持的算法 contract

[`SpeculativeAlgorithm`][spec-info] 内置 EAGLE、EAGLE3、FROZEN_KV_MTP、STANDALONE、NGRAM、
DFLASH、DSPARK，并支持 plugin registry。不同实现共享 `spec_info`、draft worker、target verify 和
accepted-token commit contract。

核心目标与 vLLM 相同：让一次 target iteration 提交多个 token，降低 decode 的串行 target 次数。

### 15.2 首次 EXTEND：为 drafter 建立起点

[`EAGLEWorkerV2.forward_batch_generation()`][eagle-worker] 处理 EXTEND 时：

1. target worker 正常 prefill，并按算法捕获 full/last hidden states；
2. prefill output 给出首个 target token；
3. `on_publish(new_seq_lens)` 可先发布 target 完成边界；
4. `_draft_extend_for_prefill()` 用 target hidden states 和尾 token 跑 draft model；
5. 生成下一轮 `EagleDraftInput`：top-k probability/index、hidden states、bonus token。

EAGLE 的 draft state 与 target KV 同样需要正确 prefill，不能在第一次 decode 临时凭空构造。

### 15.3 每轮 decode 的三阶段流水

```text
Stage A: draft
  EagleDraftInput
    -> draft model runs speculative_num_steps
    -> top-k chain/tree candidates
    -> tree mask / positions / retrieve graph
    -> EagleVerifyInput

Stage B: target verify
  all draft tree nodes in one target forward
    -> target logits per node
    -> greedy or stochastic tree acceptance
    -> accepted path + bonus token
    -> commit/move accepted KV, discard overshoot

Stage C: draft_extend
  accepted target hidden states + last accepted token
    -> update draft KV/state
    -> next EagleDraftInput
```

在 overlap 路径，Stage B 后 `on_publish()` 就让 Scheduler 准备下一轮；Stage C 仍在 GPU 执行。下轮
真正覆盖共享 buffer 前，WAR barrier 必须等 Stage C 的 read-done event。

### 15.4 Stage A：链式与树式 proposal

[`EagleDraftWorker.draft()`][eagle-worker]：

1. `prepare_for_draft()` 构造 draft ForwardBatch 和临时 KV；
2. 多个 speculative step 递推 proposal；
3. `topk=1` 是线性 chain，可以跳过通用 tree sort/gather；
4. `topk>1` 保留多分支候选，根据累计 score 选固定 `num_draft_tokens`；
5. `build_tree_kernel_efficient()` 生成 tree attention mask、position、parent/sibling/retrieve index；
6. 返回 `EagleVerifyInput`。

树可以提高“至少存在一条 target 接受路径”的概率，但 target verify token 数、tree metadata、临时 KV
和 compaction 成本都更高。top-k 不是越大越快。

### 15.5 Stage B：target verify 与 acceptance

[`EAGLEWorkerV2.verify()`][eagle-worker]：

1. plan stream 调用 `eagle_prepare_for_verify()` 分配 target verify slots、构造 ForwardBatch；
2. target worker 以 `is_verify=True` 对所有 draft nodes forward，跳过普通单-token sampler；
3. structured output 为 tree 各路径生成 vocab mask；
4. [`eagle_sample()`][eagle-utils] 应用 penalty、grammar、temperature/top-k/top-p；
5. greedy 使用 tree match kernel；随机采样使用 target-only tree sampling 或 rejection sampling；
6. 得到 `predict`、`accept_lens`、`accept_index`；
7. 更新 `new_seq_lens = old_seq_lens + accept_lens`；
8. hybrid Mamba/DSV4 状态只提交 accepted path；
9. topk>1 将 accepted path 的 KV/hidden/predict compact 到每请求 block 前部；
10. 返回 `GenerationBatchResult`。

源码中 `accept_lens` 包含最后 bonus/target token；“正确 draft 数”是 `accept_lens - 1`。结果处理器
按每请求 acceptance length 从 padded verify 输出中恢复真实 token list。

### 15.6 Greedy 与 stochastic correctness

Greedy chain 沿路径接受连续相等 token，在第一个 mismatch 处提交 target token。随机采样时，若启用
rejection sampling，需要 drafter 提供与 target vocab 对齐的 proposal distribution；接受概率近似：

```text
min(1, p_target(draft_token) / p_draft(draft_token))
```

拒绝后按 target 与 draft 的残差分布采样。源码在 kernel 前验证 draft probability shape，TP 多 rank
还从 rank 0 broadcast `predict/accept_index/accept_len`，防止浮点微差让不同 rank 走不同 KV 分支。

### 15.7 Stage C：draft_extend 为什么不能省

`_draft_extend_for_decode()`：

- 根据 `accept_lens` 找每请求最后 accepted node；
- 用 accepted target hidden states 和 token 更新 draft model KV；
- 重新得到下轮 top-k probability/index 与 hidden state；
- 可用单独 CUDA Graph；
- 它是本轮最后读取共享 request/memory mapping 的阶段。

如果只做 target verify 而不更新 drafter，下一轮 proposal 基于陈旧上下文，接受率和正确性都会失效。

### 15.8 NGRAM 路径

[`NGRAMWorker`][ngram-worker] 从 prompt/output 的 n-gram corpus 查找续写，构造线性/tree verify input，
然后复用 target verify 和 acceptance。它没有 draft-model KV，因此 proposal 成本很低，适合重复文本、
代码和可预测 pattern；自然语言低重复场景命中短时，收益有限。

### 15.9 自适应 speculative steps

[`AdaptiveController`][adaptive-runtime] 负责原子切换整套 runtime state；其
[`AdaptiveSpeculativeParams`][adaptive-spec] 用 observed acceptance length 的 EMA 调整 steps，并可按
batch size 选择候选状态。设计动机是：小 batch/高接受率时多 draft，batch 变大或接受率低时减少甚至
关闭 draft，避免固定 `K` 在所有负载下产生相同 overshoot。

动态策略的代价是要预建或切换 attention/CUDA Graph runtime state；若频繁落入未 capture shape，控制
策略本身可能抵消收益。

### 15.10 速度模型与最终性能结果

令平均正确 draft 数为 `A`，每轮平均提交 `L=A+1` 个 token，普通 target decode 成本为 `C_t`，
tree verify、draft 和其他开销分别为 `C_v(K)`、`C_d(K)`、`C_o`：

```text
speedup ~= L * C_t / (C_v(K) + C_d(K) + C_o)
```

应观察到 target iteration 数约从 `N` 降到 `N/L`，高 acceptance 时 TPOT/ITL 下降。与此同时：

- target 一轮处理更多 verify nodes；
- draft/draft_extend 增加模型计算；
- allocated KV 大于 committed KV；
- topk tree 需要 mask、retrieve、compaction；
- graph capture 和 padding 集合扩大。

小 batch、target 大、draft 便宜、领域匹配时通常收益最好。高并发大 batch、temperature 高、tree 太宽、
draft TP 通信重或 graph fallback 多时可能持平/变慢。评价必须同时记录 mean acceptance length、各位置
accept histogram、draft/verify/draft_extend GPU time、KV headroom、TPOT 和 goodput。

## 16. Scheduler 设计延伸：性能目标与冲突

SGLang 的 Scheduler 同时在优化：

```text
cache locality + GPU occupancy + TTFT + ITL + fairness
subject to KV/request rows/parallel rank consistency
```

| 设计 | 解决的问题 | 最终可见结果 | 风险 |
| --- | --- | --- | --- |
| prefill first | waiting 请求尽快首 token | 低负载 TTFT 好 | 长 EXTEND 干扰 decode ITL |
| chunked prefill | 限制单轮长 prompt | p99 ITL 更稳 | prompt TTFT/launch 次数上升 |
| LPM/DFS | 提高 Radix reuse | prefill FLOPs 和 TTFT 降低 | miss 请求公平性、排序 CPU |
| future decode reservation | 防止过度 admission | retraction/OOM 减少 | 预测保守会降低并发 |
| retraction | 实际增长超预算时恢复 | 服务不 crash | 重算与 p99 抖动 |
| overlap | 隐藏 CPU/D2H gap | tokens/s 上升、ITL 下降 | race/lifetime/debug 复杂 |
| prefill delay/mixed chunk | 平衡 prefill 与 running decode | goodput/SLO 更稳 | 参数依赖 workload |

最优策略不是固定常数。低 QPS 下重点是 TTFT；饱和服务重点是 SLO 内 goodput；高 prefix reuse agent
流量可以更积极使用 LPM，而随机 prompt 流量可能更适合低开销 FCFS。

## 17. 常见优化手段总表

| 优化 | 源码实现思路 | 应看到的性能结果 | 代价/退化条件 |
| --- | --- | --- | --- |
| Continuous batching | 每轮 filter/merge waiting、running 和 last batch | GPU busy/tokens/s/并发提高 | 饱和时排队与 TTFT 上升 |
| Radix prefix cache | 压缩 tree + lock + physical slot reuse | 高 reuse 时 EXTEND tokens、TTFT、显存下降 | 低 reuse tree CPU 开销 |
| Cache-aware policy | LPM/DFS/in-batch producer-first | hit rate 与 locality 提高 | fairness、queue sort cost |
| Chunked prefill | page-aligned `extend_range` 分轮 | decode ITL/p99 改善 | prompt TTFT 和轮数增加 |
| Overlap schedule | result queue、FutureMap、streams、WAR fence | host gap 降低、tokens/s 提升 | 状态版本与 tensor lifetime 复杂 |
| CUDA Graph | decode/prefill/spec 多 runner capture | 小 batch launch overhead、TPOT 降低 | capture 显存、padding/fallback |
| Attention backend | FlashInfer/FlashAttention 等按 mode/layout 选择 | attention time/HBM traffic 下降 | 硬件、shape、dtype 依赖 |
| Quantization | 权重/KV/activation 低比特与专用 kernel | 显存/权重带宽下降，可增 batch | 精度、dequant、backend 覆盖 |
| Spec decoding | EAGLE/NGRAM 等 proposal + target verify | 高 acceptance 时 TPOT/ITL 降低 | draft/tree/临时 KV，低接受率退化 |
| Structured output | grammar state + vocab mask + delayed sample | 格式正确且无需 retry | grammar/mask 可能成 CPU/GPU 瓶颈 |
| DP attention | attention DP 与 MLP/EP 协作 | 长序列/MoE 并行扩展 | rank 同步、padding、负载不均 |
| EP/EPLB | expert 分布、all-to-all、动态负载均衡 | MoE 聚合吞吐提高 | routing 通信和 hot expert |
| Hierarchical cache | GPU/host/storage 多层 prefix | GPU KV 容量扩展、复用提升 | load-back latency、准入 reservation |
| P/D disaggregation | P 计算/发送 KV，D 加载后 decode | 阶段隔离、decode SLO 稳定 | KV 网络、路由、failure/backpressure |

## 18. 并行和 P/D 分离

### 18.1 TP/PP/DP/EP

- TP 切层内矩阵，需要 collective；单卡放不下时必要，但小 decode batch 可能通信主导。
- PP 切模型层并传 activation，需要 microbatch 填 bubble；单请求延迟不随 stage 数线性下降。
- DP 复制请求处理能力，routing 决定负载均衡；DP attention 会增加跨 rank mode/MLP sync 约束。
- EP 将 MoE expert 分布到 rank，性能取决于 token routing、all-to-all、expert imbalance 和 EPLB。

扩卡结果必须把 NCCL/all-to-all time、padding token 和 idle rank 计入，不能只看总 GPU utilization。

### 18.2 Prefill/Decode disaggregation

SGLang Scheduler 在 P 模式可提前发送 cached prefix KV，并与 suffix forward 重叠；D 模式接收 request
metadata/KV 后进入 decode。它解决 prefill compute-heavy 与 decode latency/bandwidth-sensitive 的资源
冲突，也允许两侧独立扩缩容。

性能是否成立取决于：

```text
saved queue interference
  > KV serialization + network + bootstrap + remote admission + failure overhead
```

长 prompt、稳定高速互联和明显阶段干扰时更有利；短 prompt 或跨节点带宽不足时可能增加 TTFT。
必须观测 KV transfer bytes/time、P/D queue、D-side ITL、cache hit、失败重算和端到端成本。

## 19. 如何验证设计真的达到性能目标

### 19.1 最小对照实验

| 实验 | A/B 配置 | 必看指标 |
| --- | --- | --- |
| Radix | 随机 prompt vs 高共享 prefix | cached tokens、EXTEND GPU time、TTFT、tree CPU |
| policy | FCFS vs LPM/DFS | hit rate、TTFT 分布、饥饿/公平性、Scheduler time |
| chunk | off vs 多个 chunk size | 长 prompt TTFT、并发 decode ITL/p99、tokens/s |
| overlap | disable vs enable | GPU launch gap、schedule/result CPU、tokens/s、correctness |
| graph | eager vs decode/prefill graph | host gap、graph hit、TPOT、capture memory |
| spec | off vs EAGLE/NGRAM/不同 K/topk | accept length、各 stage GPU time、TPOT、KV usage |
| retraction | 保守 vs 激进 admission | 并发、retraction count、重算、p99 |
| P/D | colocated vs disaggregated | TTFT、D ITL、KV transfer、goodput、网络成本 |

每组固定模型、dtype、attention backend、prompt/output length distribution、seed 和并发，完成 warmup 后
报告均值、p50/p95/p99 和 confidence interval。

### 19.2 用 trace 定位端到端时间

启用 `SGLANG_FLOW_TRACE=1`，按 `rid` 对齐：

- Step 8：tokenization/dispatch；
- Step 9：Scheduler admission 入口；
- Step 10：batch plan；
- Step 11-13：TP worker、runner、sampler；
- Step 14：Scheduler commit；
- Step 15：detokenizer；
- Step 16-17：API event 和 HTTP。

overlap 会跨 iteration 交错，必须结合 `forward_iter`、batch mode、rid 和 stream profiler 看因果。若 GPU
kernel 已很快但 ITL 高，应查 schedule/result/detokenizer；若 Radix hit 高但 TTFT 不降，应查 page
alignment、host load-back、queue；若 spec accept 高却不快，应拆 draft、verify、draft_extend、tree
compaction、graph fallback 和临时 KV。

### 19.3 性能结论的合格标准

1. 目标指标在代表性并发和长度分布上改善；
2. TTFT、ITL、p99、错误率与显存没有越过 SLO；
3. correctness、grammar、TP rank 和 stop behavior 保持一致；
4. CPU、GPU、IPC、网络与 cache eviction 成本全部计入；
5. 高 reuse、低 reuse、长 prompt、structured output 至少分场景报告。

## 20. 关键正确性不变量

1. `rid_to_state` 必须先于 request dispatch 建立。
2. `Req` 只有通过 PrefillAdder/pool admission 后才能进入 ScheduleBatch。
3. Radix node lock 非零时，其 physical KV 不可 eviction。
4. tree duplicate merge 后必须 free 私有重复 slot 并重写 request row。
5. `kv_allocated_len >= kv_committed_len`，未接受 spec overshoot 不可成为真实 prefix。
6. filter/merge batch 必须同步处理所有伴随 tensor 和 sampling/spec state。
7. overlap result 必须使用对应 batch snapshot，不能读取已被下一轮修改的对象。
8. Scheduler 写 shared buffer 前必须等待上一轮最后 reader 的 WAR fence。
9. FutureMap relay 必须带正确 iteration/request-row 版本。
10. TP/DP rank 对 accepted path、seq lens 和 sampled token 必须一致。
11. 中间 chunk 不能把 pseudo next token stream 给客户端。
12. abort/finish 的 row、KV、tree lock 和 `rid_to_state` 都必须形成释放闭环。

## 21. 推荐源码阅读顺序

1. [`FLOW_TRACE.md`][flow-trace]：先跑通 Step 1-17。
2. [`TokenizerManager`][tokenizer] 与 [`DetokenizerManager`][detokenizer]：闭合进出 runtime 的链路。
3. [`Scheduler.event_loop_*`][scheduler]：比较同步与 overlap。
4. [`get_next_batch_to_run()`][scheduler]、[`SchedulePolicy/PrefillAdder`][schedule-policy]：理解准入。
5. [`ScheduleBatch`][schedule-batch]：理解 EXTEND/DECODE、filter/merge/retract。
6. [`RadixCache`][radix-cache] 与 [`memory_pool.py`][memory-pool]：分清逻辑 cache 与物理 KV。
7. [`TpModelWorker`][tp-worker]、[`ModelRunner`][model-runner]：跟进 GPU contract。
8. [`EAGLEWorkerV2`][eagle-worker]、[`eagle_sample()`][eagle-utils]：跟进 draft/verify/commit。
9. 最后进入 hierarchical cache、DP attention、MoE/EP 和 disaggregation。

[flow-trace]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/FLOW_TRACE.md
[tokenizer]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py
[scheduler]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py
[schedule-policy]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py
[schedule-batch]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py
[radix-cache]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py
[memory-pool]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/memory_pool.py
[tp-worker]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tp_worker.py
[model-runner]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/model_runner.py
[detokenizer]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/detokenizer_manager.py
[spec-info]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/spec_info.py
[eagle-worker]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/eagle_worker_v2.py
[eagle-utils]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/eagle_utils.py
[ngram-worker]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/ngram_worker.py
[adaptive-runtime]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/adaptive_runtime_state.py
[adaptive-spec]: https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/adaptive_spec_params.py
