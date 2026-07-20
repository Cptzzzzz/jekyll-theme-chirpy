---
title: 'vLLM 请求处理全流程：从 HTTP 到 Scheduler、GPU 与流式输出'
author: Cptz
date: 2026-07-20 00:19:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, architecture, scheduler, kv cache, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_architecture_posts.py. -->

> 本文按固定源码版本逐段跟踪一个 vLLM 请求。先用 30 秒建立全局路径，再按模块解释问题、Input、实现、Output、性能收益、代价与正确性不变量。

[另一篇：SGLang 请求全流程](/posts/sglang-request-lifecycle-source-guide/) · [面试题系列](/posts/vllm-sglang-interview-series/) · [源码实验手册](/posts/vllm-sglang-interview-source-labs/)

## 30 秒全流程概览

```text
HTTP / OpenAI Chat Completion
  -> OpenAIServingChat：协议校验、chat template、tokenize
  -> AsyncLLM：建立 request ID 和本地输出 collector
  -> EngineCoreClient / IPC：把 EngineCoreRequest 送入 core 进程
  -> EngineCore + Scheduler：waiting/running、token budget、KV admission
  -> KVCacheManager / BlockPool：prefix lookup、paged KV 分配与回收
  -> Executor / GPUWorker / ModelRunner：packed input、attention、forward、sample
  -> Scheduler.update_from_output()：提交 accepted token、停止判断、释放资源
  -> OutputProcessor：增量 detokenize、按 request ID 投递结果
  -> JSON / SSE：客户端看到首 token 和后续 delta
```

一句话理解：API 进程拥有客户端语义，EngineCore/Scheduler 拥有逻辑请求与资源决策，rank-local Runner 拥有设备执行状态；`SchedulerOutput` 是一次执行计划，`ModelRunnerOutput` 是待提交结果，二者必须严格配对。

下面进入完整源码路径。文中的类、函数和状态均固定到文章开头声明的 fork commit；伪代码用于压缩控制流，不替代链接中的源码条件分支。

---

本文固定到已运行验证的 fork commit
[`d1d2f7535`](https://github.com/Cptzzzzz/vllm/commit/d1d2f7535686acdc39b034bf18ba52d07c21def5)。
该提交基于 `3b39fd284`，只增加默认关闭的 `VLLM_FLOW_TRACE` 注释与日志，不改变推理语义。

本文是生命周期骨架：沿一个 OpenAI Chat Completion 请求，跟踪它如何从 JSON 变成 token、内部请求、
调度计划、GPU tensor、采样结果、增量文本，最后释放 KV Cache。需要深入某一模块时，进入下面的独立
模块文档；模块文档按当前源码逐项说明问题、Input、实现、Output、性能因果、代价和验证方法。

需要把源码机制和真实性能现象对应起来时，继续阅读
[`../VLLM_SGLANG_SOURCE_LABS.md`](/posts/vllm-sglang-interview-source-labs/)：其中用 A100 实验深入验证 causal
prefill、prefix/Beam KV、continuous batching、chunked prefill、CUDA Graph、NGRAM 和 KV preemption。

## 阅读方式：先主线，再模块

| 学习模块 | 它解决的核心问题 | 深入文档 |
| --- | --- | --- |
| 请求与输出管线 | 外部协议如何变成稳定 engine contract，异步结果如何回到正确客户端 | [`modules/REQUEST_PIPELINE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/REQUEST_PIPELINE.md) |
| Scheduler | 变长请求如何共享 token/KV budget，并冻结为一轮可执行计划 | [`modules/SCHEDULER.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/SCHEDULER.md) |
| Paged KV/Prefix Cache | 动态 KV 如何分页、共享、淘汰和在部分命中时 CoW；V1 beam 如何复用 KV | [`modules/KV_CACHE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/KV_CACHE.md) |
| Executor/ModelRunner | `SchedulerOutput` 如何变成 packed tensor、attention metadata、forward 和 sample | [`modules/MODEL_EXECUTION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/MODEL_EXECUTION.md) |
| Speculative Decoding | proposal、target verify、acceptance、KV 回滚如何跨轮协作 | [`modules/SPECULATIVE_DECODING.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/SPECULATIVE_DECODING.md) |

每个模块统一回答：

```text
为什么需要 -> 上游 Input/不变量 -> 源码实际步骤/数据结构
-> 给下游的 Output/新不变量 -> 改变了什么物理开销
-> 改善哪个指标 -> 代价/退化条件 -> 如何实验验证
```

源码链接的作用是约束实现细节：字段、状态推进、异步边界和特殊条件均以固定 commit 的实际代码为准，
而不是用通用 LLM serving 概念推测当前版本行为。

## 导航

| 主线 | 章节 |
| --- | --- |
| 基础模型 | [0. 指标](#0-阅读约定与性能指标)、[1. 状态所有权](#1-先看全局进程对象和状态所有权)、[2. 对象变形](#2-一个请求在源码中的对象变形) |
| 完整请求 | [3. 生命周期总览](#3-完整生命周期总览)、[4. HTTP 到 EngineCoreRequest](#4-t0-t2http-请求如何变成-enginecorerequest)、[5. IPC 入队](#5-t3-t5先注册消费者再跨-ipc-入队)、[6. Request 状态](#6-request-状态机与关键-token-计数) |
| 核心执行 | [7. Scheduler](#7-t6-t7scheduler-的统一-token-debt-模型)、[8. 调度权衡](#8-scheduler-的设计延伸公平性吞吐与尾延迟)、[9. KV Cache](#9-t6paged-kv-cacheprefix-cache-与物理内存)、[10. GPU forward](#10-t8-t13从-scheduleroutput-到一次-gpu-forward)、[11. 输出提交](#11-t14-t17提交结果detokenizestream-与结束) |
| 深入优化 | [12. 普通 decode](#12-一轮普通-decode-的精确时序)、[13. 投机解码](#13-speculative-decoding从理论到-vllm-源码)、[14. 优化总表](#14-常见优化手段设计实现和可观察结果)、[15. 并行/P-D](#15-并行设计tpppdp-与-pd-分离) |
| 验证与复习 | [16. 性能验证](#16-如何从源码验证每个优化真的工作)、[17. 正确性不变量](#17-关键正确性不变量清单)、[18. 阅读顺序](#18-推荐的源码阅读顺序) |

## 0. 阅读约定与性能指标

### 0.1 本文讨论的运行路径

- API 层：OpenAI-compatible `/v1/chat/completions`。
- Engine：vLLM V1 engine，而不是已经移除或兼容保留的旧 engine 叙述。
- GPU 执行：优先解释本机验证实际选择的 V2 GPU ModelRunner；必要处指出 V1 runner 的同构职责。
- 普通 decoder-only generation 为主，同时说明 multimodal、structured output、LoRA、KV connector
  在哪里进入相同生命周期。

### 0.2 性能结果不能只写“更快”

后文使用以下指标区分不同优化的真实目标：

| 指标 | 定义 | 主要受谁影响 |
| --- | --- | --- |
| TTFT | 请求到第一个可见 token 的时间 | 排队、tokenize、prefix hit、prefill、首 token 回传 |
| ITL | 相邻输出 token 的时间间隔 | decode batch、调度间隔、kernel、通信、输出回压 |
| TPOT | 每个输出 token 的平均时间 | decode 与 speculative acceptance |
| tokens/s | 实例单位时间完成的 token 数 | continuous batching、并行度、kernel、batch shape |
| p95/p99 | 尾延迟 | 长 prefill、抢占、graph miss、队列、慢客户端 |
| goodput | 满足 SLO 的请求或 token 吞吐 | 吞吐和尾延迟的共同结果 |
| KV usage | KV block 使用率与可回收率 | page size、prefix reuse、序列长度、抢占 |

源码控制流已通过真实 HTTP 请求验证，详见
[`RUNTIME_VALIDATION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/RUNTIME_VALIDATION.md)。该验证使用 eager/小模型配置用于看清流程，
不是生产性能 benchmark；本文不会把一次功能验证伪装成固定加速比。

## 1. 先看全局：进程、对象和状态所有权

```text
API process
  FastAPI router
    -> OpenAIServingChat
    -> Renderer / tokenizer
    -> AsyncLLM
    -> OutputProcessor + per-request collector
            | EngineCoreRequest / EngineCoreOutputs
            v
EngineCore process
  EngineCoreProc
    -> EngineCore
    -> Scheduler
    -> KVCacheManager / BlockPool
    -> Executor client
            | SchedulerOutput / ModelRunnerOutput
            v
rank-local worker process(es)
  GPUWorker
    -> GPUModelRunner
    -> attention backend / model / sampler / speculator
```

这里最重要的不是类名，而是状态所有权：

| 状态 | 唯一或主要 owner | 为什么不能放到别处 |
| --- | --- | --- |
| HTTP、chat template、外部模型名 | serving/API process | 不应污染 GPU 调度热路径 |
| asyncio waiter、增量 detokenizer | `OutputProcessor` | 属于客户端语义，不是设备资源状态 |
| waiting/running、token progress | `Scheduler` | 所有 admission 和完成更新必须串行一致 |
| logical request -> KV blocks | `KVCacheManager` | 与调度准入、抢占和 prefix cache 强耦合 |
| CUDA buffer、model、attention metadata | rank-local ModelRunner | 不能跨进程共享普通 Python 可变状态 |
| SSE socket 与客户端回压 | API process | GPU kernel 已 launch 后不能依赖 socket 是否可写 |

这条边界解决了在线推理的核心冲突：协议是高并发且可能很慢，GPU 状态必须一致且持续前进。

## 2. 一个请求在源码中的对象变形

不要把全流程中的 `request` 当成同一个对象。它至少经过下面这些形态：

| 阶段 | 对象 | 关键字段或含义 |
| --- | --- | --- |
| HTTP 入站 | `ChatCompletionRequest` | messages、stream、tools、sampling 参数 |
| render 后 | `EngineInput` | prompt token IDs/embeds、multimodal features |
| API 到 core | `EngineCoreRequest` | request ID、prompt、params、priority、LoRA、trace |
| core 内部 | `Request` | status、output IDs、computed count、block hashes、spec IDs |
| 一轮调度 | `SchedulerOutput` | 本轮 request->token 数、KV blocks、spec IDs、encoder inputs |
| runner 输入 | `InputBatch` | packed input IDs、positions、block tables、slot mapping |
| GPU 输出 | `SamplerOutput` / `ModelRunnerOutput` | sampled IDs、rejected count、logprobs、connector output |
| core 输出 | `EngineCoreOutput(s)` | 单请求新增 token、finish reason、events、stats |
| API 输出 | `RequestOutput` | 增量/累计 token 与 text、usage、finished |
| 网络输出 | JSON 或 SSE chunk | OpenAI-compatible response |

`request_id` 是这些对象跨进程关联的主键，但不是全部正确性。异步执行时还必须让“这一轮的
`SchedulerOutput`”与“这一轮返回的 `ModelRunnerOutput`”配对，否则 token count 和 KV 状态会错位。

## 3. 完整生命周期总览

以下先给出单请求 happy path。第 10 到 16 步在 decode 中循环；continuous batching 会让其他请求
在任意一次循环进入或退出同一 batch。

```text
T0  POST /v1/chat/completions
T1  validate + render chat template + tokenize
T2  build SamplingParams and EngineCoreRequest
T3  register local output collector
T4  IPC enqueue to EngineCoreProc
T5  EngineCore converts to internal Request and enters waiting
T6  Scheduler prefix lookup + admission + KV allocation
T7  build SchedulerOutput
T8  Executor dispatches SchedulerOutput to GPU worker(s)
T9  ModelRunner updates resident request state and packs InputBatch
T10 build block table / slot mapping / attention metadata
T11 model forward writes KV and produces hidden states
T12 logits processing + sample or speculative rejection sampling
T13 async D2H; optionally draft proposals for next iteration
T14 Scheduler commits accepted output and repairs optimistic state
T15 EngineCoreOutputs cross IPC; OutputProcessor detokenizes
T16 generator yields RequestOutput; serving emits JSON/SSE
T17 repeat T6-T16 or finish, free ownership, close stream
```

### 3.1 生命周期 I/O contract

| 流程 | 要解决的问题 | Input | 核心方法 | 传给下游的 Output | 主要性能维度 |
| --- | --- | --- | --- | --- | --- |
| T0-T2 协议/输入 | messages 和外部参数不能直接调度 | HTTP JSON、tokenizer、model config | render/tokenize/校验；`InputProcessor` 统一模型输入 | `EngineCoreRequest` + cloned params | API CPU/TTFT；让 core 热路径不处理协议分支 |
| T3-T4 异步提交 | 快结果与多请求乱序不能丢失 | engine request、request ID | 先建 collector/parent mapping，再 IPC enqueue | core 收到 DTO；API 持有可等待的 collector | API 并发、IPC latency、正确性 |
| T5 core request | transport DTO 缺少调度状态 | `EngineCoreRequest`、block hasher、grammar/MM cache | 建立 `Request`、hash/counters/status | waiting `Request` | prefix lookup 前置 CPU、queue TTFT |
| T6-T7 调度/KV | 动态请求争用 token 和物理 slots | waiting/running state、budgets、cache/connector | token debt、prefix hit、allocation、preemption | `SchedulerOutput` | batch 吞吐、TTFT/ITL、公平性、KV capacity |
| T8-T10 执行准备 | Python request 不能直接喂 kernel | `SchedulerOutput`、resident state、block tables | rank dispatch、ragged pack、slot/attention metadata | `InputBatch` + forward context | host launch gap、padding、metadata/H2D |
| T11-T13 GPU/sample | 计算本轮状态并产生合法 token | packed tensors、weights/KV、sampling/grammar/spec state | graph/eager forward、selective logits、sample、async D2H | `ModelRunnerOutput`/future + next draft | kernel time、TPOT/ITL、spec acceptance |
| T14 core commit | 乐观状态要按真实结果提交/回滚 | 同一 `SchedulerOutput` + runner result | accept/reject、token/grammar/finish、free | per-request `EngineCoreOutput` | commit CPU、KV 回收速度、正确性 |
| T15-T16 输出 | token delta 不是稳定 OpenAI 文本 | core outputs、decode/parent state、socket | 增量 detokenize、聚合、JSON/SSE flush | 客户端可见 text/token | perceived TTFT/ITL、API CPU、网络回压 |
| T17 循环/结束 | 继续生成或归还所有 ownership | updated request/finish/abort | 重新进入 schedule，或 free KV/collector | 下一轮状态或最终 response | 并发 capacity、泄漏/preemption风险 |

表中 Output 是下游真正消费的 contract。每个模块的字段、状态变化和 fallback 见顶部对应模块文档。

下面逐段展开。

## 4. T0-T2：HTTP 请求如何变成 EngineCoreRequest

### 4.1 Router 只做协议边界

Chat router 将 Pydantic 请求交给 [`OpenAIServingChat.create_chat_completion()`][chat-serving]。
真正的生命周期入口在 `_create_chat_completion()`：

1. 选择 tokenizer、chat template 和 tool/reasoning parser。
2. `render_chat_request()` 把 messages 变成 conversation 与一个或多个 `EngineInput`。
3. 创建 `chatcmpl-*` request ID；批量 prompt 会派生 sub-request ID。
4. 根据 `max_model_len`、prompt length 和用户参数约束 `max_tokens`。
5. `request.to_sampling_params()` 将 OpenAI 参数归一化为 engine 的 `SamplingParams`。
6. 调用 `engine_client.generate()`，也就是 `AsyncLLM.generate()`。

设计思路是“在跨 EngineCore 进程前完成协议归一化”。Scheduler 不理解 messages、role、tool call
或 OpenAI JSON；它只处理已经确定的 token 和生成约束。

### 4.2 InputProcessor 固化 engine contract

[`AsyncLLM.add_request()`][async-llm] 调用 [`InputProcessor.process_inputs()`][input-processor]，
构造 `EngineCoreRequest`。这一层处理：

- text/token IDs/input embeds 的统一输入形式；
- sampling 或 pooling task；
- multimodal feature 与 placeholder；
- LoRA、priority、arrival time、DP rank、trace header；
- sampling/pooling params clone、generation config/default EOS 合并。

到这里以后，prompt 文本只在输出侧为了 detokenize/展示而保留；core 计算依赖 token IDs/embeds。
`InputProcessor` 返回后，`AsyncLLM.add_request()` 才分配内部唯一 request ID；`n > 1` 时也在这里建立
`ParentRequest` 并 fan-out child `EngineCoreRequest`，而不是在 `InputProcessor` 内展开。

**性能结果**：render/tokenize 从 GPU owner 进程移出，不会直接延长每轮 GPU launch gap；但超长文本、
慢 tokenizer 或多模态预处理仍会增加请求自身 TTFT，因此 API 侧需要独立 CPU profiling。

## 5. T3-T5：先注册消费者，再跨 IPC 入队

### 5.1 为什么 `OutputProcessor.add_request()` 必须先执行

[`AsyncLLM._add_request()`][async-llm] 的顺序是：

```python
self.output_processor.add_request(request, ... , queue)
await self.engine_core.add_request_async(request)
```

这是一个明确的并发正确性设计。小 prompt、小模型或 cache hit 请求可能很快完成；如果先通知
EngineCore，结果可能在 per-request collector 建立前返回，造成 late output 无法路由。

本地 `RequestOutputCollector` 建立后，`EngineCoreClient` 才把 `EngineCoreRequest` 发送给独立
`EngineCoreProc`。API coroutine 此后只等待自己的 collector，不阻塞共享 engine 输出循环。

### 5.2 EngineCore 将 transport DTO 变成调度对象

[`EngineCore.preprocess_add_request()`][engine-core] 调用
`Request.from_engine_core_request()`：

- 建立内部 [`Request`][request]；
- 初始化 `status`、prompt/output/spec token 列表和 token 计数；
- 使用 request block hasher 预计算 prefix block hashes；
- structured output 请求异步初始化 grammar；
- multimodal feature 从 receiver cache 恢复。

随后 `EngineCore.add_request()` 调用 `Scheduler.add_request()`，请求进入 waiting queue。此时它只是
可调度，并未拥有本轮计算所需的全部物理 KV slot。

## 6. Request 状态机与关键 token 计数

[`RequestStatus`][request] 至少区分 waiting、running、preempted、等待 grammar、等待 remote KV、
finished 等状态。理解调度必须先理解这些量：

| 字段 | 含义 |
| --- | --- |
| `num_tokens` | 当前已确定序列长度，通常是 prompt + 已提交 output |
| `num_computed_tokens` | target model 已实际计算并可依赖的 token 位置数 |
| `spec_token_ids` | drafter 提议但尚未被 target 验证的 token |
| `num_tokens_with_spec` | `num_tokens + len(spec_token_ids)` 对应的追赶目标 |
| `num_output_placeholders` | async scheduling 中为尚未回收结果预留的逻辑位置 |
| `num_in_flight_tokens` | 已调度但结果尚未应用的 token 数 |

普通 decode 可以理解为：新 output token 被提交后，`num_tokens` 增长，而它对应的 KV 尚未计算，
下一轮就产生一个 token debt。投机解码一次产生多个未验证位置，debt 变大；prefix hit 则让
`num_computed_tokens` 从 0 直接跳到命中长度附近。

这正是 vLLM Scheduler 能统一各种模式的基础。

## 7. T6-T7：Scheduler 的统一 token-debt 模型

### 7.1 源码中没有互斥的 prefill/decode phase

[`Scheduler.schedule()`][scheduler] 的核心注释直接说明：Scheduler 不维护互斥的“prefill phase”
和“decode phase”。它尝试让每个请求的 `num_computed_tokens` 追上 `num_tokens_with_spec`。

抽象成公式：

```text
debt(req) = num_tokens_with_spec
            + num_output_placeholders
            - num_computed_tokens

scheduled(req) = min(
    debt(req),
    per-request limits,
    remaining token budget,
    max_model_len headroom
)
```

同一个表达覆盖：

- 首次 prompt：debt 是未计算 prompt 长度；
- chunked prefill：一次只偿还 debt 的一部分；
- prefix cache：先提高 computed count，只计算 suffix；
- 普通 decode：每次通常欠一个 token；
- speculative decode：把 draft tokens 一起变成待 target 验证的 debt；
- async scheduling：placeholder 表达已经在 pipeline 中但输出尚未回收的位置。

### 7.2 每轮调度的实际顺序

可将源码压缩成以下伪代码：

```text
token_budget = max_num_batched_tokens
encoder_budget = max_encoder_tokens

for request in running:
    new_tokens = bounded_debt(request, token_budget)
    new_tokens = fit_encoder_and_model_len(request, new_tokens)
    blocks = kv_manager.allocate_slots(request, new_tokens, lookahead)
    while blocks is None:
        victim = choose_preemption_victim()
        preempt(victim)
        retry allocation
    if allocation succeeded:
        add request/token count/blocks/spec IDs to this iteration
        token_budget -= new_tokens

if no preemption happened:
    for request in waiting order:
        reject or skip blocked grammar/remote-KV/LoRA cases
        cached_blocks, cached_tokens = prefix_lookup(request)
        external_tokens = connector_lookup_if_enabled(request)
        new_tokens = bounded_remaining_sequence(request)
        blocks = allocate cached + external + new + lookahead
        if not fit: stop or skip according to reason
        move request to running and add it to this iteration

build SchedulerOutput
optimistically advance num_computed_tokens by scheduled count
```

“先 running 后 waiting”保护已在 decode 的请求，减少其 ITL 抖动；waiting 只有在没有刚发生的抢占时
才尝试 admission，避免一边抢占一边立刻塞入更多请求造成抖动。

### 7.3 SchedulerOutput 是一轮执行的不可变计划

[`SchedulerOutput`][scheduler-output] 主要携带：

- `scheduled_new_reqs`、`scheduled_resumed_reqs`、`scheduled_running_reqs`；
- `num_scheduled_tokens: request_id -> count`；
- request 对应的新 KV blocks 和 block copy；
- `scheduled_spec_decode_tokens`；
- multimodal encoder inputs、structured-output 标志、finished IDs；
- KV connector metadata。

Executor 和各 rank 必须消费同一逻辑计划。Scheduler 在 `_update_after_schedule()` 中乐观增加
`num_computed_tokens`，因为这允许 pipeline 继续组下一轮；如果投机 token 被拒绝，稍后
`update_from_output()` 必须精确回退。

### 7.4 抢占为什么是必要但昂贵的

`KVCacheManager.allocate_slots()` 返回 `None` 表示物理资源不足。Scheduler 按 FCFS 或 priority
选择 victim，`_preempt_request()` 会：

1. 释放该请求 KV block ownership；
2. 清理 encoder cache；
3. 状态改为 `PREEMPTED`；
4. `num_computed_tokens` 重置为 0，清空 spec token；
5. 放回 waiting queue，之后尽量通过 prefix cache 复用，否则重算。

**最终性能表现**：抢占让服务避免直接 OOM，并允许更高优先级请求取得资源；代价是重算、cache
污染和 p99 峰值。持续出现 preemption 不是“GPU 很忙”的好现象，而是 admission/KV 容量配置失败。

## 8. Scheduler 的设计延伸：公平性、吞吐与尾延迟

调度不是单目标优化。一个实用模型是：

```text
maximize useful_tokens_per_second
subject to TTFT/ITL SLO, KV capacity, model length, fairness, rank consistency
```

关键冲突如下：

| 决策 | 正向结果 | 负向结果 |
| --- | --- | --- |
| 大 token budget | 大 batch、较高吞吐 | 单轮更长，decode ITL/p99 可能升高 |
| running 优先 | decode 更平滑 | waiting TTFT 可能上升 |
| 小 prefill chunk | 限制 head-of-line blocking | prompt 分多轮，TTFT 和调度开销可能上升 |
| prefix-hit 优先 | 少算 FLOPs，整体吞吐高 | miss 请求可能不公平 |
| aggressive admission | GPU 更容易满载 | KV 紧张、抢占与尾延迟抖动 |
| priority | 业务 SLO 可表达 | 低优先级饥饿，需要 aging/配额 |

因此线上调参不能只最大化平均 tokens/s。正确做法是给定 workload 分布和 SLO，寻找最高 goodput。

## 9. T6：Paged KV Cache、Prefix Cache 与物理内存

### 9.1 两层结构

[`KVCacheManager`][kv-manager] 是 Scheduler 的逻辑接口；[`BlockPool`][block-pool] 管理物理
`KVCacheBlock`、free queue、hash map 和 ref count。不同 attention 类型由 coordinator/manager
适配，包括 full attention、sliding window、R-SWA、Mamba、cross attention 等。

逻辑上一个 request 拥有 block table：

```text
logical block 0  -> physical block 317
logical block 1  -> physical block  42
logical block 2  -> physical block 901
```

attention kernel 通过 block table/slot mapping 找到物理 KV，不要求整条序列连续分配。

### 9.2 `get_computed_blocks()` 做什么

prefix cache key 基于 token block hash 以及影响模型状态的额外信息。普通单 full-attention group 中，
hash 粒度等于物理 block size，因此只命中完整、已经计算的 cache blocks。当前源码在多 KV group 且
`prefix_match_unit` 小于某个物理 group block size 时还支持 fine-grained lookup；命中可落在物理 block
内部，该 shared tail 会在继续写入前转成 private CoW block。几个关键边界：

- prefix caching 关闭、prompt logprobs 等不允许读 cache 的请求直接 miss；
- 普通整 block 模式下，page/block 不完整尾部不能命中；fine-grained 模式需要单 block CoW copy；
- 即使 prompt 全命中，也要重新计算最后一个 token 才能得到下一 token logits；
- hybrid cache group 必须协调不同 layout 可共同复用的最长前缀；
- remote KV connector 的命中与本地命中分别记账。

若 prompt 长度为 `P`，本地/远端可安全命中 `H` 个 token，则 target prefill 的主要 token 计算量从
`P` 降到约 `P-H`，但要计入 block 对齐与最后 token 重算，不能简单声称等于零。

### 9.3 `allocate_slots()` 的完整账本

源码把 request token 区间分成：

```text
| computed | new local cached | external cached | new compute | lookahead |
```

分配过程同时处理：

1. 已不再需要的 sliding-window block；
2. 本地命中 block 的 ref count；
3. remote KV 即将加载的 slot；
4. 本轮要写入的 new token slot；
5. EAGLE 等 KV-aware drafter 的 lookahead slot；
6. fine-grained partial hit 的 private block 与 pending CoW copy；
7. watermark、full-sequence-fit 和 in-flight reservation。

投机 token 可能写入临时 KV，但只有已经验证、finalized 的 token 能进入可复用 prefix cache。否则另一
请求可能读取错误分支，造成静默语义错误。

### 9.4 Paged KV 最终解决了什么

它并不改变 attention 数学复杂度。它的主要结果是：

- 不为每个请求预留连续 `max_model_len`；
- 降低外部碎片和 over-reservation；
- request 可动态增长，完成后 block 可细粒度归还；
- 相同 full block 可通过 ref count 被多个请求共享；
- 同样显存通常能容纳更多有效 KV token，从而提高可并发序列数和吞吐。

代价是内部 page 尾部碎片、block table 寻址、hash/ref-count CPU 开销，以及 page size 对 kernel/cache
locality 的影响。最佳 block size 依赖模型 KV bytes/token、平均序列长度和 attention backend。

## 10. T8-T13：从 SchedulerOutput 到一次 GPU forward

### 10.1 EngineCore 与 Executor 边界

[`EngineCore.step()`][engine-core] 的顺序是：

```text
scheduler.schedule()
model_executor.execute_model(..., non_block=True)
scheduler.get_grammar_bitmask()
future.result() / sample_tokens()
process aborts that arrived while GPU was running
scheduler.update_from_output()
```

`non_block=True` 允许 CPU 准备 grammar 或其他工作。batch queue/async scheduling 路径还能让多轮执行
进入 pipeline，但每个 `SchedulerOutput` 必须与自己的 future 一起保存。

### 10.2 ModelRunner 的 resident state

V2 [`GPUModelRunner.execute_model()`][v2-runner] 先把动态调度变化应用到常驻状态：

- finish/free 已结束 request slot；
- add new/resumed request；
- update running request 和 block table；
- 应用 staged KV block writes/copies。

随后 `prepare_inputs()` 将 ragged request 打包为 `InputBatch`：

- request ID 排序与 `idx_mapping`；
- 每请求 scheduled token count；
- packed `input_ids` 与 `positions`；
- `query_start_loc` 前缀和；
- prefill/decode 判定；
- spec draft token 展开；
- 需要计算 logits 的位置索引。

这是“动态 CPU 对象”到“稳定 GPU buffer”的关键桥梁。

### 10.3 attention metadata

`prepare_attn()` 从 block table 构造：

- per-cache-group block tables；
- 每个 query token 的 slot mapping；
- sequence length、query start、causal mask 等 backend metadata。

模型层不需要理解 Scheduler，只通过 forward context 中的 attention metadata 读写正确 KV slot。

### 10.4 eager、piecewise graph 与 full graph

runner 根据 batch descriptor 选择：

| 路径 | 方法 | 适合场景 | 主要成本 |
| --- | --- | --- | --- |
| eager | `self.model(**model_inputs)` | 动态/调试/不支持 graph | Python 与 driver launch overhead |
| piecewise | `run_pw_graph()` | 部分 shape 可编译/捕获 | graph break 与多套 capture 管理 |
| full graph | `run_fullgraph()` | 稳定 decode shape | padding、capture 显存、fallback |

CUDA Graph 的效果是消除一部分 host launch 开销，不会减少 Transformer 的 FLOPs 或 HBM 读写。
小 batch decode 中 host gap 占比高时收益最明显；大 prefill 已由大 kernel 主导时相对收益较小。

### 10.5 forward、logits 与 sample

模型 forward 写入本轮 KV，并返回 hidden states。最后一个 PP rank：

1. 只 gather 需要 logits 的 hidden positions；
2. `compute_logits()`；
3. structured output grammar bitmask 先约束 logits；
4. 普通路径进入 sampler；有 draft token 时进入 rejection sampler；
5. `AsyncOutput` 在 copy stream 发起 D2H，尽量与后续 GPU 工作重叠。

PP 非末 rank 返回 `IntermediateTensors`，最终采样结果再广播给其他 rank，使各 rank 的 request/KV
状态保持一致。

## 11. T14-T17：提交结果、detokenize、stream 与结束

### 11.1 Scheduler 只提交被接受的状态

[`Scheduler.update_from_output()`][scheduler] 将 batch result 拆回 request：

- 降低 `num_in_flight_tokens`；
- 读取本请求 sampled token IDs；
- 若有 speculative tokens，计算 accepted/rejected 数并回退 rejected 的 computed count；
- 逐 token append output，检查 EOS、stop、length 与 grammar；
- 完成时释放 request ownership、encoder state 和 KV blocks；
- 生成 `EngineCoreOutput`，附带 events/stats/finish reason。

这里是 Scheduler 乐观状态与真实 GPU 结果的 commit point。

### 11.2 OutputProcessor 恢复客户端语义

[`AsyncLLM.output_handler()`][async-llm] 批量拉取 `EngineCoreOutputs`，分 chunk 调用
[`OutputProcessor.process_outputs()`][output-processor]，避免一次处理大量请求长期占用 asyncio loop。

`OutputProcessor` 负责：

- internal child ID 到 parent request 的聚合；
- 增量 detokenization、UTF-8 稳定边界、stop string；
- logprobs、usage、reasoning/tool parser；
- 将 `RequestOutput` 推入对应 collector；
- stop string 在 API 侧触发时反向 abort core。

`AsyncLLM.generate()` 消费 collector 并 `yield`；serving 再构造非流式 JSON 或 SSE chunk。

### 11.3 回压与取消的真实边界

- 已经 launch 的 CUDA kernel 通常不能因 HTTP disconnect 中途撤销；abort 阻止后续调度，并在结果回来
  后清理。
- `RequestOutputCollector` 可合并 delta，减少慢客户端导致的 queue item 爆炸。
- 生产环境仍应限制每请求缓存 byte/token、socket write timeout 和最大暂停时间。
- 慢客户端若继续占用 KV，会伤害其他请求；若直接 abort，则牺牲该请求完成率。需要显式策略。

## 12. 一轮普通 decode 的精确时序

假设 prompt 已完成，当前已提交序列为：

```text
[prompt tokens..., y0, y1]
```

下一轮大致发生：

1. `num_tokens` 已包含 `y1`，但 `y1` 对应的新 KV 位置需要 target 计算。
2. Scheduler 发现一个 token debt，分配一个 slot。
3. runner 输入通常是 `y1`，position 是当前末尾位置。
4. attention 读取旧 block table，并把 `y1` 的 K/V 写入新 slot。
5. hidden state -> logits -> sampler 得到 `y2`。
6. Scheduler 提交 `y2`，下轮再计算 `y2` 对应的 KV。

这解释了为什么“采样出 token”和“该 token 已写入可依赖 KV”不是同一时刻，也解释了各种
off-by-one、prefix 全命中仍需最后 token forward 的来源。

## 13. Speculative Decoding：从理论到 vLLM 源码

### 13.1 它解决什么问题

自回归 decode 每次 target forward 通常只产生一个新 token，尤其小 batch 时受权重读取、kernel
launch 和同步延迟限制。投机解码让便宜的 proposer 一次猜 `K` 个 token，再让 target 在一次更宽的
forward 中并行验证这些位置。如果连续接受多个 token，一次 target iteration 可提交多个输出。

它优化的是“每个 target iteration 的有效 token 数”，不是免费减少所有计算。

### 13.2 vLLM 支持的 proposer 形态

[`SpeculativeConfig`][spec-config] 与 [`vllm/v1/spec_decode`][spec-common] 包含多类方法：

- draft model：更小的独立模型；
- EAGLE/EAGLE3、MTP：利用 target hidden state 或多 token prediction head；
- Medusa：多 head proposal；
- n-gram/prompt lookup、suffix decoding：从已有 token 模式提议，几乎没有模型 forward 成本；
- DFlash、DSpark 等专用 proposer。

不同 proposer 共享“proposal -> target verify -> accept/repair”的 contract，但 draft 成本、KV 需求、
可用模型和接受率完全不同。

### 13.3 vLLM V2 的跨轮数据流

当前 [`GPUModelRunner.sample_tokens()`][v2-runner] 的关键顺序是：

```text
target sample / verify current iteration
  -> start async D2H copy
  -> postprocess accepted/rejected state
  -> speculator.propose(...) for the next iteration
  -> DraftTokensHandler stores draft IDs
  -> normal scheduling returns draft IDs to Scheduler;
     async scheduling keeps placeholder/worker state in its pipelined path
  -> next SchedulerOutput carries scheduled_spec_decode_tokens
```

下一轮 `prepare_inputs()`：

1. 读取每请求 draft token 数；
2. 将上一已采样 token和 draft tokens 组合成 packed input；
3. 展开 request mapping 与 logits positions；
4. target 一次计算 bonus position + draft positions；
5. [`RejectionSampler`][rejection-sampler] 决定接受前缀和替代 token。

proposal 放在 async D2H 启动之后，可以把 CPU copy 延迟与 drafter GPU 工作部分重叠。

### 13.4 greedy 验证

对于 greedy decode，沿 draft chain 从左到右：

```text
draft:   d1 d2 d3 d4
target:  d1 d2  x ...
accept:  d1 d2 x
```

接受连续相等的 `d1,d2`，第一个不匹配位置提交 target 自己的 `x`，后续 draft 作废。一次 verify
至少能得到一个 target token，最多得到 `K+1` 个 token（具体 bonus 定义依实现路径）。

### 13.5 sampling 时为什么需要 rejection sampling

若 target 分布为 `p(x)`，draft 分布为 `q(x)`，不能只比较采样 token 是否相等，否则会改变最终
分布。经典 acceptance probability 为：

```text
accept(d) = min(1, p(d) / q(d))
```

拒绝后从归一化残差 `max(p-q, 0)` 采样。vLLM 的 `RejectionSampler` 在 probabilistic draft 模式
接收 target logits、draft logits、temperature/seeds，并调用 GPU rejection-sampling kernel；这保证
正确配置下输出仍服从 target 分布，而不只是“看起来相近”。

### 13.6 KV 与计数如何修复

Scheduler 组 batch 时将 spec IDs 加入 `scheduled_spec_decode_tokens`，并乐观推进 computed count。
GPU 返回后：

```text
num_accepted = max(len(generated_token_ids) - normal_bonus_count, 0)
num_rejected = num_draft_tokens - num_accepted
num_computed_tokens -= num_rejected
```

同时 worker 的 request state、slot table 和 penalty state按 `num_sampled/num_rejected` 更新。未接受 token
对应的临时 slot 可以回收，不能 hash 成共享 prefix。EAGLE/Mamba 等 proposer 还需 lookahead block 或
专用 recurrent state，因此 KV 容量规划必须包含 speculative headroom。

### 13.7 速度模型与最终表现

令：

- `A`：每轮平均接受的 draft token 数；
- `L = A + 1`：每次 target verify 平均提交 token 数，`+1` 是 target/bonus token；
- `C_t`：普通一次 target decode 成本；
- `C_v(K)`：target 验证 `K` 个 draft 的成本；
- `C_d(K)`：draft/proposal 成本；
- `C_o`：tree、sampling、copy、状态修复等开销。

近似 speedup：

```text
speedup ~= L * C_t / (C_v(K) + C_d(K) + C_o)
```

因此源码设计最终应在指标上表现为：target forward 次数约从 `N` 降到 `N/L`，TPOT/ITL 下降；但
每次 target forward token 更宽、临时 KV 更多。以下场景通常收益好：

- target 大且 memory/launch bound，draft 显著便宜；
- acceptance length 高；
- batch 较小，普通 decode 没把 GPU 吃满；
- proposal/verify shape 能走 CUDA Graph。

以下场景可能持平或变慢：

- 高并发大 batch 已充分利用 target；
- domain mismatch、temperature 高导致接受率低；
- draft 模型不够小，或额外 TP 通信昂贵；
- `K` 太大，verify/KV/graph padding 成本超过多接受的 token；
- structured output/LoRA/backend 组合频繁 fallback。

必须同时记录 acceptance rate、mean acceptance length、draft tokens/s、accepted tokens/s、TPOT 和
端到端 goodput，不能只报 acceptance 百分比。

## 14. 常见优化手段：设计、实现和可观察结果

| 优化 | 核心思路与源码落点 | 应看到的结果 | 退化条件或代价 |
| --- | --- | --- | --- |
| Continuous batching | Scheduler 每轮重组 running/waiting，而非静态等待整批结束 | GPU busy、tokens/s、并发能力上升 | 高负载排队增加 TTFT；大 batch 拉高 ITL |
| Chunked prefill | 用 token debt 和 threshold 将长 prompt 分轮 | decode 不再被整段长 prefill 阻塞，ITL/p99 更稳 | 长 prompt 自身 TTFT、调度次数增加 |
| Prefix caching | block hash + ref count 复用 full computed blocks | 命中请求 prefill tokens/FLOPs 降低，TTFT 改善 | 低复用时 hash/tree 维护；block 尾部不能全用 |
| Paged KV | 非连续 block table 动态增长 | 同显存容纳更多有效 KV token，减少外部碎片 | page 尾部内部碎片、寻址/bookkeeping |
| CUDA Graph | 稳定 buffer + full/piecewise replay | 小 batch decode host gap、TPOT 下降 | capture 显存、padding、多 shape 与 fallback |
| `torch.compile`/fused op | 合并 Python/op dispatch，生成专用 kernel | launch 数和中间读写减少 | 编译时间、graph break、模型/backend 覆盖 |
| Optimized attention | FlashAttention/FlashInfer 等 backend 使用 paged metadata | attention kernel time 和 HBM traffic 降低 | shape、dtype、GPU 架构决定最优 backend |
| Quantization | 权重/KV/activation 降位宽 | 权重带宽与显存下降，可增 batch/KV | dequant kernel、精度、硬件支持；小模型未必更快 |
| Async output | copy stream D2H 与 proposal/下一轮工作重叠 | GPU/CPU gap 下降 | tensor 生命周期、stream fence 更复杂 |
| Structured output GPU mask | grammar state生成 mask，采样前应用 | 保证格式且避免无效重试 | grammar CPU 或 mask 更新成为瓶颈 |
| LoRA batching | 基座权重共享，runner 按 batch 激活 adapter | 多租户显存远低于完整模型复制 | adapter 数限制、动态 shape、额外 gather/kernel |
| Spec decode | draft 多 token + target 并行 verify | 高 acceptance 时 TPOT/ITL 下降 | draft/verify/临时 KV 开销，低接受率退化 |
| KV connector/P-D | 外部加载/发送 KV，prefill 与 decode 资源解耦 | 混合 workload 隔离，decode SLO 更稳 | KV 网络传输、路由、失败重算和背压 |

## 15. 并行设计：TP、PP、DP 与 P/D 分离

### 15.1 Tensor Parallel

TP 切分层内权重，rank 间用 all-reduce/reduce-scatter/all-gather 保持数学等价。它解决单卡放不下模型
或单卡算力不足；代价是每层通信。decode 小 batch 下计算量少，collective latency 可能吞掉扩卡收益。

### 15.2 Pipeline Parallel

PP 按层切 stage，传递 activation。vLLM 用 intermediate tensors 和 sampled-token broadcast 保持各
stage 状态一致。它降低单 rank 权重占用，但引入 pipeline bubble；需要 batch queue/microbatch 才能
提高利用率，单请求延迟通常不会随 stage 数线性下降。

### 15.3 Data Parallel 与 expert parallel

DP 复制模型并路由请求，天然扩展独立请求吞吐。MoE 场景还可用 EP/EPLB 分布 expert；最终性能取决于
token routing、expert imbalance 和 all-to-all，而不仅是 GPU utilization。

### 15.4 Prefill/Decode disaggregation

[`KVConnector`][kv-connector] 让 Scheduler 同时核算 local cached、external matched 和 in-flight KV。
P 节点计算 prompt KV，D 节点加载后继续 decode。它解决两种阶段资源特征不同的问题：prefill 更偏
compute，decode 更偏权重/KV 带宽与低延迟。

收益条件：长 prompt、足够高的 KV 传输带宽、稳定路由和可观的阶段干扰。短 prompt 下，序列化、网络
和 bootstrap 固定成本可能大于隔离收益。remote load 失败时源码会标 invalid block 并重算或报错，
这是 correctness 路径的一部分，不是边缘功能。

## 16. 如何从源码验证每个优化真的工作

### 16.1 最小实验矩阵

固定模型、硬件、dtype、seed、请求长度分布，再分别运行：

| 实验 | A | B | 必看指标 |
| --- | --- | --- | --- |
| chunked prefill | off | 多个 chunk size | 长 prompt TTFT、并发 decode ITL/p99、tokens/s |
| prefix cache | 冷且随机 | 高共享 prefix | cached tokens、prefill GPU time、TTFT、KV usage |
| CUDA Graph | eager | graph/compile | host gap、kernel 数、TPOT、graph hit/fallback |
| speculative | off | 不同 K/方法 | mean accept length、draft/accepted rate、TPOT、显存 |
| quantization | BF16 | 目标量化 | 精度、权重显存、tokens/s、kernel time |
| TP | 1 GPU | 2/4 GPU | compute time、NCCL time、端到端 goodput |
| P/D | colocated | disaggregated | P queue、D ITL、KV transfer time/failure、总成本 |

### 16.2 用 trace 对齐因果

启用 `VLLM_FLOW_TRACE=1` 后，按 `request_id` 追踪 Step 8-17；再结合 profiler/Nsight：

- Step 11 到 13：调度完成到 runner forward；
- Step 13 到 14：model 与 sampler；
- Step 14 到 15：GPU output 到 Scheduler commit；
- Step 15 到 17：output IPC、detokenize 与 HTTP。

若 GPU kernel 快但 ITL 仍高，问题可能在 schedule/IPC/output；若 cache hit 高但 TTFT 不降，需要检查
block 对齐、最后 token 重算、排队或 remote load；若 spec acceptance 高但 TPOT 不降，应拆出 draft、
verify、graph fallback 和临时 KV 成本。

### 16.3 判断“性能结果达到”的标准

一个优化只有同时满足以下条件才算成功：

1. 目标指标改善，例如 tokens/s 或 p99 ITL；
2. 非目标指标没有越过 SLO，例如 TTFT 没因过度 batching 失控；
3. 结果在 warmup 后、多个并发档位和足够样本上稳定；
4. correctness 与输出分布不变，量化近似除外但需单独评估；
5. GPU 显存、CPU、网络和失败恢复成本都计入。

## 17. 关键正确性不变量清单

1. collector 必须先于 core enqueue 注册。
2. request 只有在 KV allocation 成功后才能出现在本轮执行计划。
3. `SchedulerOutput` 与返回 future 必须一一对应。
4. `num_computed_tokens` 的乐观推进必须在 reject/failure 时回退。
5. 未验证 draft token 不能提交到共享 prefix cache。
6. block ref count 非零时不能物理淘汰。
7. full prompt cache hit 仍必须保留产生 logits 的计算边界。
8. PP/TP rank 必须看到一致 sampled token 和 finish 状态。
9. abort 不能假设已 launch kernel 被取消；late output 必须可安全丢弃。
10. async D2H/graph/shared buffer 的 tensor 生命周期必须覆盖消费 stream。

## 18. 推荐的源码阅读顺序

1. [`FLOW_TRACE.md`][flow-trace]：用真实日志建立 17 步地图。
2. [`OpenAIServingChat`][chat-serving] 与 [`AsyncLLM`][async-llm]：看协议到 engine 的边界。
3. [`EngineCore`][engine-core]、[`Request`][request]：看进程和状态机。
4. [`Scheduler.schedule()`][scheduler]：围绕 token debt 读 running、waiting、preemption。
5. [`KVCacheManager`][kv-manager]、[`BlockPool`][block-pool]：看 prefix、allocation、ref count。
6. [V2 ModelRunner][v2-runner]：从 `SchedulerOutput` 跟到 input、attention、forward、sample。
7. [`RejectionSampler`][rejection-sampler] 和各 proposer：读 speculative commit/rollback。
8. [`OutputProcessor`][output-processor]：闭合 detokenize、stop、abort 和 stream 生命周期。

[flow-trace]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/FLOW_TRACE.md
[chat-serving]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/openai/chat_completion/serving.py
[async-llm]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py
[input-processor]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/input_processor.py
[engine-core]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core.py
[request]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/request.py
[scheduler]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py
[scheduler-output]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/output.py
[kv-manager]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py
[block-pool]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/block_pool.py
[v2-runner]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/model_runner.py
[output-processor]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/output_processor.py
[spec-config]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/config/speculative.py
[spec-common]: https://github.com/Cptzzzzz/vllm/tree/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/spec_decode
[rejection-sampler]: https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
[kv-connector]: https://github.com/Cptzzzzz/vllm/tree/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/distributed/kv_transfer
