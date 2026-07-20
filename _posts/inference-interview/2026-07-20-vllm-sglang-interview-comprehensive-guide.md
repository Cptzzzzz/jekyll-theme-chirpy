---
title: 'vLLM / SGLang 深度面试：24 道完整答案与评分指南'
author: Cptz
date: 2026-07-20 00:17:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文保留 24 道综合题的长篇回答，适合模拟单题多轮深挖。按难度训练请从[60 题系列总索引](/posts/vllm-sglang-interview-series/)开始；需要实验数据与源码边界时继续阅读[源码实验手册](/posts/vllm-sglang-interview-source-labs/)。

---

> 需要先按难度自测或用于组织面试时，见
> [`VLLM_SGLANG_INTERVIEW_QUESTION_BANK.md`](/posts/vllm-sglang-interview-series/)：
> 60 道题按基础、生命周期、调度、KV/执行、高级优化和系统设计六档组织，
> 每题都提供回答思路、详细答案、主问题和追问的独立得分点。

本文基于 2026-07-15 已经实际启动和跟踪过的当前 fork 源码：

- vLLM trace commit [`d1d2f7535`](https://github.com/Cptzzzzz/vllm/commit/d1d2f7535686acdc39b034bf18ba52d07c21def5)，基于 `3b39fd284`
- SGLang trace commit [`9529c2e96`](https://github.com/Cptzzzzz/sglang/commit/9529c2e96351f3b1d5715eaa89989b20723d60b5)，基于 `a3194d358`
- 实际验证模型：`Qwen/Qwen3-0.6B`
- 已验证链路：OpenAI Chat Completion 请求从 HTTP 入口到调度、GPU forward、采样、反分词和 HTTP 返回

本文中的源码链接固定到上述 Git commit，不假设后续版本仍具有完全相同的类名或进程拓扑。
两个仓库包含用于观察流程的 `Step N: xxx` 注释和日志，开关默认关闭，只记录元数据，
没有改变推理语义。两边当前源码均返回 HTTP 200 和 `trace ok`，并完整输出 Step 1-17。

## 使用方法

每道题都按照以下结构组织：

- `完整答案`：面试时可以据此组织 3 到 8 分钟的回答。
- `源码结合`：指出当前版本中真正负责该机制的类、函数和状态。
- `得分点`：按 10 分制拆解，方便自测或面试评价。
- `常见失分点`：说明只背概念时容易忽略的边界。

建议先理解请求生命周期，再学习调度与 KV Cache，最后学习 CUDA Graph、分布式和 PD 分离。只记住模块名通常只能得到 3 到 4 分；能讲清因果、资源上界、失败路径和验证手段，才接近 8 分以上。

## 核心源码地图

| 主题 | vLLM | SGLang |
| --- | --- | --- |
| HTTP/API | [`api_server.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/openai/api_server.py)、[`chat_completion/serving.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/openai/chat_completion/serving.py) | [`http_server.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/entrypoints/http_server.py) |
| 异步请求状态 | [`async_llm.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py)、[`output_processor.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/output_processor.py) | [`tokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py) |
| 核心推理循环 | [`core.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core.py) | [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py) |
| 调度 | [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py) | [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)、[`schedule_batch.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py) |
| KV Cache | [`kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py)、[`block_pool.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/block_pool.py) | [`radix_cache.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py)、[`memory_pool.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/memory_pool.py) |
| GPU 执行 | [`gpu_worker.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu_worker.py)、[`gpu/model_runner.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/model_runner.py) | [`tp_worker.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tp_worker.py)、[`model_runner.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/model_runner.py) |
| 反分词 | [`output_processor.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/output_processor.py) | [`detokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/detokenizer_manager.py) |

---

## 1. 分别描述一个 Chat Completion 请求的完整生命周期

### 完整答案

**vLLM 请求链路**

1. HTTP 请求进入 OpenAI API router，协议层校验参数，`serving.py` 渲染 chat template 并得到 token IDs。
2. `AsyncLLM.generate()` 调用 `add_request()`。它先在 API 进程的 `OutputProcessor` 中注册请求状态和 `RequestOutputCollector`，再通过 `EngineCoreClient` 把同一个内部 request ID 发给后台 `EngineCore`。先注册再发送能够避免 GPU 很快返回时发生“结果找不到消费者”的竞态。
3. `EngineCore` 是串行化 GPU 推理循环的所有者。它把请求交给 `Scheduler.add_request()`，后者放入 waiting queue。
4. 每一轮 `Scheduler.schedule()` 在 `max_num_batched_tokens`、`max_num_seqs`、KV block、encoder budget 等限制下生成 `SchedulerOutput`。同一轮可以包含多个请求，每个请求可以推进不同数量的 token，所以本质上是 token 级 continuous batching。
5. Executor 将 `SchedulerOutput` 分发到 rank-local `GPUWorker`。ModelRunner 更新 input batch、构造 position、slot mapping 和 attention metadata，将 CPU 侧状态复制到稳定的 GPU buffer，选择 eager、编译或 CUDA Graph 路径，执行模型 forward 和 sampler。
6. sampled token IDs 返回 `Scheduler.update_from_output()`。调度器更新请求 token、停止条件、KV 状态、推测解码接受情况，并释放已结束请求的资源。
7. `EngineCoreOutputs` 通过 IPC 回到 API 进程。后台 `AsyncLLM.output_handler` 调用 `OutputProcessor.process_outputs()`，按 request ID 拆分批量结果，增量反分词、处理 stop string/logprob，并将结果放入对应 collector。
8. `generate()` 的异步生成器读取 collector。非流式请求等待 finished 后组装一次响应；流式请求持续产生 delta，HTTP 层编码为 SSE。
9. 客户端断开会取消异步生成器。`AsyncLLM.generate()` 捕获 `CancelledError` 或 `GeneratorExit`，先清理 API 进程状态，再向 `EngineCore` 发 abort，最终由 Scheduler 释放 KV block。

**SGLang 请求链路**

1. HTTP/OpenAI 层将请求交给主进程中的 `TokenizerManager.generate_request()`。它完成协议归一化、tokenization、多模态预处理，并为 `rid` 建立 `ReqState`。
2. `TokenizerManager` 将 `TokenizedGenerateReqInput` 通过 ZMQ PUSH 发给 Scheduler 进程。Scheduler 的 `handle_generate_request()` 把传输 DTO 转为包含 prefix、采样参数和缓存状态的 `Req`，加入 waiting queue。
3. `SchedulePolicy` 计算 FCFS、LPM、DFS-weight、priority 等顺序，`PrefillAdder` 根据可用 token、KV pool、chunked prefill 和运行中 Decode 的预留量完成准入。
4. Scheduler 创建 `ScheduleBatch`。首次或新增上下文是 `EXTEND`，逐 token 生成是 `DECODE`。`TpModelWorker` 将其转为 rank-local `ForwardBatch`，`ModelRunner` 选择 attention、quantization 和 graph backend，执行 forward 与 sampler。
5. 非 overlap 模式下 Scheduler 同步处理结果；overlap 模式下它用独立 stream、future map 和 result queue，让 CPU 准备下一轮，同时上一轮 GPU/D2H 尚在完成。
6. Scheduler 更新 `Req.output_ids`、停止条件、Radix Cache 锁与内存池状态，然后把 token 批次发送给独立 `DetokenizerManager`。
7. Detokenizer 增量生成稳定文本 delta，再通过 ZMQ 返回 `TokenizerManager`。后者用 `rid_to_state` 找到等待的 asyncio state，设置 event，HTTP coroutine 随后返回非流式结果或 SSE delta。
8. 断连时 `TokenizerManager._wait_one_response()` 检查 `request.is_disconnected()` 并发送 `AbortReq`。Scheduler 分别处理 waiting、running、grammar 和 PD transfer queue；已经提交的 GPU iteration 通常不能中断，但结果回来后会丢弃并释放资源。

两者共同原则是：协议和 GPU loop 解耦、请求用稳定 ID 关联、调度器拥有 KV 生命周期、GPU 已提交工作不能被 Python 真正撤销。主要区别是 vLLM 在 API 进程内完成输出处理和反分词，而 SGLang 显式设置了 TokenizerManager、Scheduler、Detokenizer 的 ZMQ 拓扑。

### 源码结合

- vLLM：`AsyncLLM.add_request()`、`AsyncLLM.generate()`、`EngineCore.step()`、`Scheduler.schedule()`、`Scheduler.update_from_output()`、`OutputProcessor.process_outputs()`。
- vLLM IPC：[`core_client.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core_client.py) 创建异步多进程 client；具体 worker 数量取决于 uniproc、multiproc、Ray 和 TP/PP 配置。
- SGLang：`TokenizerManager._send_one_request()`、`Scheduler.handle_generate_request()`、`Scheduler.get_next_batch_to_run()`、`Scheduler.run_batch()`、`Scheduler.process_batch_result()`、`DetokenizerManager.event_loop()`。
- 已实测的 Step 1 到 Step 17 映射见 [`vLLM FLOW_TRACE.md`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/FLOW_TRACE.md) 与 [`SGLang FLOW_TRACE.md`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/FLOW_TRACE.md)。

### 得分点（10 分）

- 2 分：能从 HTTP 一直讲到 GPU 和 HTTP 返回，而不是只讲 Scheduler。
- 2 分：能指出进程边界、IPC 和 request ID 关联状态。
- 2 分：能区分 Prefill/EXTEND 与 Decode，并说明多轮调度。
- 2 分：能讲清结果更新、增量反分词和流式输出。
- 2 分：能讲清取消只能阻止后续调度，通常不能撤销已经 launch 的 kernel。

### 常见失分点

把 API server、Scheduler 和 GPUWorker 当成一个同步函数；认为每个请求独占一个 batch；忽略 abort、输出关联和进程间结果乱序。

---

## 2. vLLM 为什么将 AsyncLLM、EngineCore 和 OutputProcessor 分开

### 完整答案

这三个模块分别拥有三类完全不同的状态和并发约束。

`AsyncLLM` 面向前端并发。它运行在 asyncio 环境中，负责输入处理、异步请求提交、每请求 collector、后台接收结果和取消传播。它不能让一次 CPU 反分词或慢客户端阻塞所有 HTTP coroutine。

`EngineCore` 面向设备一致性。Scheduler、KV block table、executor iteration 必须按确定顺序更新，否则会出现同一 block 被重复分配、结果应用到错误 iteration 或异步执行后过早释放 KV。将它放到独立后台进程，还能避免 API 的 Python GC、tokenizer 和 web server 抖动直接延迟 GPU launch。

`OutputProcessor` 面向“批量 token 输出到单请求协议输出”的转换。它保存增量 detokenizer、stop string、logprob、parent/child parallel sampling 状态和外部 ID 到内部 ID 的映射。这些是 API 语义，不应污染 GPU 调度器。当前 `RequestOutputCollector` 在消费者落后时合并 DELTA，而不是为每个 token 无界增加一个 asyncio queue item。

这种拆分的收益是故障域清晰、多个 API frontend 可以复用 engine、GPU loop 更少受到 GIL/GC 干扰、输出处理可以独立优化。代价是 ZMQ/序列化、状态复制、request ID 协议和更复杂的 shutdown/error propagation。

如果全部合并为单进程同步循环，低并发场景会减少 IPC 和上下文切换，调试也更直观。但 tokenizer、JSON、反分词、socket backpressure、Python GC 都可能延误下一次 GPU launch；一个 HTTP handler 异常也更容易破坏调度状态。合理的折中不是机械追求进程越多越好，而是让“必须串行一致的设备状态”和“高并发但可能抖动的协议状态”分离。

### 源码结合

- [`async_llm.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py)：`output_handler` 后台任务、`generate()`、`abort()`。
- [`core.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core.py)：`EngineCore.step()` 的 schedule/execute/update 所有权，以及用于 PP/异步执行的 batch queue。
- [`output_processor.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/output_processor.py)：`RequestOutputCollector`、`RequestState`、增量反分词、external/internal request ID 映射。
- [`core_client.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core_client.py)：API 进程与后台 EngineCore 的通信及死亡传播。

### 得分点（10 分）

- 3 分：分别说清三个模块的状态所有权，而不只是“解耦”。
- 2 分：指出 GPU loop 需要串行一致性，API 层是高并发异步模型。
- 2 分：说明进程隔离对 GIL、GC、tokenizer 和尾延迟的影响。
- 2 分：说明拆分带来的 IPC、错误传播和状态关联成本。
- 1 分：能给出适合合并模式的低并发或离线推理场景。

### 常见失分点

只回答“微服务更清晰”；认为 `OutputProcessor` 在 GPU worker 内；没有意识到 KV 状态和 HTTP 状态的生命周期不同。

---

## 3. SGLang 为什么拆分 tokenizer、scheduler 和 detokenizer

### 完整答案

SGLang 的拆分主要是在保护 Scheduler 主循环。Scheduler 不只是排队器，它同时维护 waiting/running batch、Radix Tree、request-to-token pool、token-to-KV pool、CUDA stream/future map 和 PD transfer 状态。它是单线程且处于每次 GPU launch 的关键路径，任何长时间 CPU 工作都会直接制造 GPU bubble。

Tokenizer 的工作量具有突发性和输入相关性。长文本分词、多模态图像预处理、chat template 和批量协议转换都可能占用 CPU。`TokenizerManager` 在 HTTP 主进程中先把异构输入转成统一 DTO，并用 `rid_to_state` 保存异步调用上下文。Scheduler 只接收已经规范化的数据。

Detokenizer 也可能是显著 CPU 开销。它要处理 UTF-8 边界、特殊 token、增量文本 offset 和 stop string。如果 Scheduler 每个 Decode iteration 都重建完整字符串，复杂度可能从线性退化到近似 O(n²)，并推迟下一轮调度。独立 `DetokenizerManager` 保存 per-request offset，只返回稳定 delta。

拆分后的反向链路是 Scheduler -> Detokenizer -> TokenizerManager，而不是直接返回 HTTP。这使 GPU 调度和文本处理可以流水化，也让 native API/OpenAI API 共享相同 runtime。

成本包括 ZMQ send/recv、Python 对象序列化、额外内存副本、进程启动与 watchdog、跨进程日志时钟不完全有序，以及必须通过 `rid` 正确处理 abort/late output。ZMQ 队列积压还会把瞬时 CPU 抖动变成长尾。因此拆分只是提供并行机会，仍需控制输出合并、socket HWM、队列长度和错误传播。

### 源码结合

- [`tokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py)：`init_ipc_channels()`、`rid_to_state`、`generate_request()`、`handle_loop()`。
- [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`event_loop_normal()`、`event_loop_overlap()`、waiting/running/Radix/KV 状态。
- [`detokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/detokenizer_manager.py)：独立 event loop 与增量 token-to-text 转换。
- `TokenizerManager._handle_batch_output()` 对文本 chunk 延迟拼接，避免每步复制完整输出前缀。

### 得分点（10 分）

- 3 分：指出核心目标是保护 Scheduler/GPU launch 关键路径。
- 2 分：说明 tokenizer 与 detokenizer 各自的 CPU 和状态特征。
- 2 分：准确描述正向和反向 ZMQ 链路以及 `rid` 关联。
- 2 分：说明序列化、排队、late output 和故障传播成本。
- 1 分：能讨论字符串重复拼接导致的 O(n²) 风险。

### 常见失分点

把 DetokenizerManager 当作无状态函数；认为进程拆分一定降低延迟；忽略 Scheduler 仍需要处理所有 rank 的一致状态。

---

## 4. 如何设计端到端 backpressure

### 完整答案

正确原则是：慢客户端只能减速或终止自己的请求，不能阻塞整个 engine 的结果处理线程或 Scheduler。端到端需要分层限流，而不是在一个位置简单调用阻塞式 `put()`。

**第一层：准入控制。** API gateway 根据并发请求数、waiting token、KV usage、估算完成时间和租户配额决定接收、排队或返回 429/503。仅限制请求数不够，因为一个 128K prompt 与一个 32-token prompt 的资源需求完全不同。

**第二层：每请求输出缓冲。** 流式 delta 应按请求合并，设置 token/byte 上限和最长不可写时间。vLLM 的 `RequestOutputCollector` 已经只保存一个待消费对象并合并 DELTA；SGLang 使用 `ReqState.out_list`，接收端会批量唤醒和合并 chunk。工程上还应记录 buffered bytes、oldest unsent age 和 slow-consumer count。

**第三层：网络写入。** 每个连接有独立 coroutine 等待 socket 可写，不能让一个连接阻塞共享 output handler。超过超时或 buffer 上限就主动断开并发送 abort。

**第四层：调度反馈。** 如果客户端只是短暂抖动，可允许最多生成 N 个未发送 token；超过阈值后将该请求标为 output-blocked，Scheduler 暂时不为它分配 Decode token，但保留 KV。暂停太久则 abort，因为长期保留 KV 会伤害其他租户。这个状态必须是 per-request，不能让 EngineCore 全局等待。

**第五层：取消闭环。** 断连后先使前端状态幂等完成，再把 abort 送到 Scheduler。Scheduler 从 waiting/running queue 移除请求，并在已提交 iteration 返回后安全释放 KV。需要接受最多浪费一轮 GPU 计算这一事实。

不能简单地在网络层阻塞整个输出管线，因为这会使其他请求结果无法消费；也不能无条件让 Scheduler 持续生成，因为慢客户端会积累内存和无效 GPU 工作。最合理的控制点是“前端独立缓冲 + per-request 调度暂停 + 超时取消”。

### 源码结合

- vLLM [`output_processor.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/output_processor.py)：`RequestOutputCollector.put()` 合并 DELTA，避免每 token queue item 无界增长。
- vLLM [`async_llm.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py)：共享 `output_handler` 必须持续 drain EngineCore，`generate()` 是每请求消费者。
- SGLang [`tokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py)：`ReqState.out_list`、`_wait_one_response()`、断连检测与 `abort_request()`。
- SGLang [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`abort_request()` 分别清理 waiting、running、grammar 和 PD queue。

### 得分点（10 分）

- 2 分：明确不能用慢客户端阻塞全局 output handler 或 Scheduler。
- 2 分：提出 token-aware admission，而不只是连接数限制。
- 2 分：提出 per-request 有界缓冲、delta 合并、超时和指标。
- 2 分：提出调度层 per-request pause/abort，并说明 KV 保留代价。
- 2 分：说明已提交 GPU iteration 不可撤销及幂等清理要求。

### 常见失分点

只说“设置队列长度”；让共享 Scheduler 等待 socket；只释放前端对象而没有向调度器发送 abort；忽略暂停请求长期占用 KV。

---

## 5. Continuous batching 为什么提升吞吐，它如何影响 TTFT、TPOT 和公平性

### 完整答案

传统 static batching 在一个 batch 开始后通常等待所有序列结束。序列输出长度不同，短请求结束后留下空槽，最后一个长请求形成明显的 tail effect。Continuous batching 每个 iteration 都重新检查完成请求和 waiting queue，及时移出完成项、加入新请求，因此 GPU batch 的有效工作密度更高。

Prefill 与 Decode 的硬件特征不同。Prefill 一次处理许多 prompt token，GEMM 形状较大，通常更容易达到 Tensor Core 的计算吞吐；但 attention 的工作量还与上下文长度、mask 和 backend 有关。Decode 每个请求通常只推进一个 token，却要读取权重和该请求历史 KV，容易受显存带宽、kernel launch 和小 GEMM 效率限制。把多个 Decode 请求持续拼成 batch，能摊薄权重读取与 launch 开销，是 continuous batching 的主要收益之一。

它并不保证所有延迟指标同时改善：

- 吞吐通常上升，因为 GPU 空槽减少。
- 低负载 TTFT 可能上升，因为新 Prefill 需要与运行中的 Decode 分享 token budget。
- TPOT 在 Decode batch 变大时可能先改善，但混入超长 Prefill 后会恶化，因为某一轮 GPU 时间过长。
- 公平性可能变差。持续到来的短请求、缓存命中请求或高优先级请求可能反复占用预算，使长 Prefill 等待；相反，过大的长 Prefill chunk 也会阻塞所有 Decode。

因此 production scheduler 一般使用 token budget、chunked prefill、max sequence count、优先级和 KV admission 联合约束。关键不是“每轮塞入最多请求”，而是在一个可控 iteration 时长内最大化有效 token，并满足 TTFT/TPOT SLO。

### 源码结合

- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：`schedule()` 每轮重新构造 `num_scheduled_tokens`，优先推进 running，再从 waiting 准入；结束请求在 `update_from_output()` 后移除。
- vLLM [`config/scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/config/scheduler.py)：`max_num_batched_tokens`、`max_num_seqs`、`enable_chunked_prefill` 和 `long_prefill_token_threshold`。
- SGLang [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`get_next_batch_to_run()`、`get_new_batch_prefill()`、`update_running_batch()` 在 `EXTEND` 与 `DECODE` 间连续组批。
- SGLang [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)：`PrefillAdder` 同时核算当前 chunk、预计新 token 和 KV 容量。

### 得分点（10 分）

- 2 分：解释 static batch 的 tail effect 和 continuous admission。
- 2 分：正确区分 Prefill 的大矩阵计算与 Decode 的带宽/launch 特征。
- 2 分：分别讨论吞吐、TTFT 和 TPOT，不声称全部必然改善。
- 2 分：指出长 Prefill 与持续短请求造成的双向公平性问题。
- 2 分：将解决方案落到 token budget、chunked prefill 和 iteration 时长。

### 常见失分点

认为 continuous batching 等于动态改变 PyTorch batch dimension；认为 Prefill 和 Decode 每 token 成本相同；只谈吞吐不谈尾延迟。

---

## 6. 8K Prefill 与大量 Decode 共存时如何设计调度

### 完整答案

目标是限制单次 GPU iteration 的最大执行时间，同时保证 Prefill 最终取得进展。可以把每轮总预算记为 `B` 个逻辑 token，把运行中 Decode 的刚性预算记为 `D`。普通自回归下 `D` 约等于运行请求数；推测解码下还要加 draft/verify lookahead。该轮给 Prefill 的预算最多为 `B - D - R`，其中 `R` 是 KV、图 padding 或紧急 Decode 的安全余量。

一个 8K prompt 不应一次性吃完 `B`，而应切成大小为 `C` 的 chunk。`C` 需要通过实测选择，使单轮 Prefill kernel 时间不超过 TPOT 预算的一定比例，例如 30% 到 50%。如果 TPOT p95 超标，动态缩小 `C`；如果 GPU 利用率低且 Decode 很少，则扩大 `C`。不能把 chunk 切得过小，否则调度、attention metadata、H2D 和 kernel launch 开销会吞掉收益。

具体策略可以是：

1. 先为所有活跃 Decode 分配一个 token，避免它们被长 Prefill 完全阻塞。
2. 按 deadline/age 选取 Prefill，请求每轮最多拿一个 chunk，避免单个长请求独占剩余 budget。
3. 为 Prefill 保留最小进度，例如连续 `N` 轮未被选中后强制分配 `C_min`，防止 Decode 流量永久饿死长 prompt。
4. KV admission 按“完成当前 chunk 后的 block”与未来 Decode headroom 同时核算。只看当前 chunk 是否能放下，会过度准入，随后不断抢占。
5. Prefix cache 命中的 token 不需要 forward，但仍要正确映射已有 block；外部 KV 异步加载不能和本地计算预算混为一谈。

vLLM 当前通过 `token_budget`、`long_prefill_token_threshold` 和 `allocate_slots()` 实现这些约束的一部分。SGLang 使用 `chunked_prefill_size`、`max_prefill_tokens`、`PrefillAdder` 和 `new_token_ratio` 估计 Decode 的未来 KV 需求。两者都需要以真实 TPOT 曲线调参，而不是仅看 GPU utilization。

### 源码结合

- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：running request 先消耗 `token_budget`；waiting request 的 `num_new_tokens` 受 threshold 与剩余预算裁剪。
- vLLM [`kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py)：`allocate_slots()` 支持 `full_sequence_must_fit`、watermark、reserved blocks 和 lookahead。
- SGLang [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`_get_new_batch_prefill_raw()`、动态 `chunked_prefill_size`、与 running batch 混合逻辑。
- SGLang [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)：`PrefillAdder.rem_total_tokens`、`rem_chunk_tokens` 和 `add_chunked_req()`。

### 得分点（10 分）

- 2 分：先给 Decode 刚性预算，再使用剩余 token budget。
- 2 分：能解释 chunk 大小对 TPOT 与 launch 开销的双向影响。
- 2 分：有 Prefill 防饿死机制，而不是永久 Decode-first。
- 2 分：将 KV headroom、推测 lookahead 和缓存命中纳入准入。
- 2 分：提出用 TPOT/TTFT 反馈动态调节，而不只靠静态参数。

### 常见失分点

固定把 8K prompt 一次 prefill；永远优先 Decode 导致 Prefill 饿死；只计算计算预算而不计算 KV；认为缓存命中 token 仍需执行完整 Prefill。

---

## 7. 按 token 调度相比按请求数调度有什么优势，何时仍会不公平

### 完整答案

请求数不是可靠的资源单位。两个请求可能分别需要推进 1 个 Decode token 和 4K 个 Prefill token，也可能一个命中 95% prefix cache、另一个完全 miss。按“两个请求”计费看起来相同，但 GPU 时间、KV 增量和 H2D metadata 都完全不同。

vLLM 的 `SchedulerOutput.num_scheduled_tokens` 为每个 request ID 指定本轮推进量，总和受 `max_num_batched_tokens` 限制。这样可以直接控制本轮输入 token 数、KV slot 增量和大致的 GPU iteration 工作量，也自然支持 chunked prefill、普通 Decode 和 speculative verify 共存。

但 token 也只是代理指标，不是精确成本：

- Prefill token 与 Decode token 的 kernel 路径和单位成本不同。
- Attention 成本与上下文长度、GQA/MLA、sliding window 和 backend 有关。
- MoE token 的 expert 分布会改变计算与 all-to-all 开销。
- 多模态 encoder token、结构化输出 mask、logprob 和 LoRA 会增加额外成本。
- CUDA Graph bucket padding 可能使 33 个请求按 64 的图执行，逻辑 token 数没有体现 padding。
- Prefix hit token 占逻辑序列长度，但不一定执行 forward。

公平性也不能只靠 token budget保证。若队列只按 FCFS，一个超长请求可能持续占用大 chunk；若总偏爱短请求，长请求可能饿死；按 token 平均分配又会让已经接近 deadline 的请求得不到足够份额。更完善的做法是使用加权 token cost，例如 `cost = a*prefill_tokens + b*decode_tokens + c*kv_read_bytes + d*comm_bytes`，再结合 age、priority 和 deadline 形成调度分数。

### 源码结合

- [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：`token_budget`、`num_scheduled_tokens` 和 `total_num_scheduled_tokens` 是每轮核心资源账本。
- [`sched/output.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/output.py)：`SchedulerOutput` 将 per-request token 计划传给 worker。
- [`gpu/model_runner.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/model_runner.py)：逻辑计划最终转换为 input batch、slot mapping、graph descriptor 和实际张量形状。
- SGLang 的对应资源估计在 [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py) 的 `PrefillAdder` 和 [`schedule_batch.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py) 中。

### 得分点（10 分）

- 3 分：用 Prefill/Decode 例子说明请求数为什么失真。
- 2 分：说清 per-request token map 同时约束计算进度和 KV 增量。
- 3 分：至少指出上下文、MoE、graph padding、cache hit 中三个非线性因素。
- 2 分：提出 token cost 与 priority/age/deadline 联合建模。

### 常见失分点

把 token 数当成精确 FLOPs；把 prompt 长度等同于本轮实际计算 token；没有区分逻辑 token、缓存命中 token 和 graph padding token。

---

## 8. 为什么 SGLang 下一轮调度日志可能早于上一轮结果日志

### 完整答案

这是 overlap scheduling 的预期时序，不代表因果倒置。普通同步循环是 `schedule(n) -> GPU(n) -> D2H(n) -> process(n) -> schedule(n+1)`。Overlap 模式试图变成：

```text
schedule stream:  schedule(n) ---- prepare(n+1) ---- schedule(n+2)
forward stream:          GPU(n) -------- GPU(n+1) -------- GPU(n+2)
copy stream:                    D2H(n) -------- D2H(n+1)
CPU result:                         process(n) ----- process(n+1)
```

因此记录在组批处的下一次 `Step 10: DECODE`，可能发生在上一轮 D2H 完成和 `Step 14: process result` 之前。

困难在于第 `n+1` 轮的输入 token 恰好来自第 `n` 轮 sampler。SGLang 不能在 CPU 尚未拿到 token 时直接读取它，于是通过 GPU 侧 relay/future 机制保存下一轮依赖。`_relay_forward_payload()` 将 sampled token 或 draft payload 放入 `future_map`，`resolve_forward_inputs()` 在真正执行下一轮前解析。Scheduler 还要保存 batch 属性快照，因为 overlap 期间同一 Python 对象的 `seq_lens`、`spec_info`、logprob ranges 等可能已经被下一轮修改。

正确性依赖四件事：

1. stream event 保证 schedule buffer 写入先于 forward 读取，D2H 读取又晚于 forward 写入。
2. 静态 buffer 在消费者完成前不能被复用或被 Python GC 回收。
3. 结果必须按 iteration/request 对应，abort 后到达的 stale result 不能重新激活请求。
4. KV slot allocation、sampled token 和 sequence length 必须对同一个 batch snapshot 生效。

如果这些依赖出错，轻则输出 token 错位，重则使用错误 slot mapping 覆盖其他请求 KV、读取已释放 tensor、出现非法访存或只有高并发时才发生的随机错误。分析日志时必须同时看 `rid`、iteration/batch、forward mode 和 CUDA event，不能只按跨进程时间戳排序。

### 源码结合

- [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`event_loop_overlap()`、`record_batch_in_overlap()`、`run_batch()`、`process_batch_result()`。
- 同文件 `_relay_forward_payload()` 与 `future_map` 保存下一轮 token 依赖，`batch_record_buf` 保存批次属性和 tensor 引用。
- `copy_stream.wait_stream(forward_stream)` 与 `copy_done` event 允许 D2H 和下一轮 forward 重叠。
- vLLM [`core.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core.py) 的 batch queue/async scheduling 也可能造成相邻 iteration 日志交错，但实现边界不同。

### 得分点（10 分）

- 2 分：画出 schedule、forward、D2H、process 的重叠时间线。
- 3 分：指出下一 token 依赖，并说明 future/relay 的作用。
- 2 分：说明 stream event、buffer lifetime 和 batch snapshot。
- 2 分：能列举错误 slot mapping、stale result 或非法复用等后果。
- 1 分：知道日志应按 rid/iteration 建因果关系，而非只看时间。

### 常见失分点

认为“CPU 异步”足以解释一切；忽略下一轮输入依赖上一轮 sampled token；认为多个 CUDA stream 会自动保证顺序。

---

## 9. 如何满足高优先级 p99，同时避免低优先级饿死

### 完整答案

只使用 strict priority queue 无法满足“低优先级不能饿死”。更合适的是 priority、deadline、aging 和 token quota 的组合。

可以为每个请求维护 `base_priority`、`arrival_time`、`deadline`、`served_tokens` 和 `preemption_count`。以 vLLM“数值越小优先级越高”为例，定义：

```text
effective_priority = base_priority - floor(wait_time / aging_interval)
slack = deadline - now - estimated_remaining_time
```

先按是否即将 miss deadline 分组，再按 `effective_priority` 和 arrival 排序。Aging 使等待足够久的低优请求逐渐晋级。`estimated_remaining_time` 应按剩余 Prefill、Decode 长度分布、prefix hit 和当前 batch 性能估计，而不是仅看请求数。

每轮 token budget 可以为高优池预留 70% 到 90%，低优池保留最小 10% 到 30%；任一池空闲时允许另一池借用。对于每个租户再使用 deficit round robin token credits，避免单一高优租户垄断。Decode 通常先保证一个 token，长 Prefill 则按 chunk 运行。

抢占只应发生在 iteration 边界。优先抢占低优、刚开始、recompute 成本低且没有外部 KV transfer 的请求。加入 hysteresis 和最小运行 quantum，避免两个优先级接近的请求反复抢占。被抢占请求的 age/credit 必须保留，否则它回到 waiting 后会永远处于劣势。

验收不能只看高优 p99。至少同时记录按 priority/tenant 分组的 TTFT、TPOT、deadline miss ratio、starvation age、served token share、preemption count、recompute token 和 KV usage。低优“不能饿死”应转成明确指标，例如在可行负载下最大排队时间小于 30 秒。

当前 vLLM 有 FCFS/priority queue，优先级相同时按 arrival time；SGLang 支持可配置方向的 priority sorting 和 threshold-based priority preemption。但 deadline、aging、租户 token credit 仍需要新增策略状态。

### 源码结合

- vLLM [`request_queue.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/request_queue.py)：`PriorityRequestQueue` 按 `(priority, arrival_time, request_id)` 排序。
- vLLM [`request.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/request.py)：`priority`、`arrival_time`、`num_preemptions` 和 `RequestStatus`。
- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：`_preempt_request()` 释放 KV 并把请求重新放回等待路径。
- SGLang [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)：priority/FCFS 排序、`priority_scheduling_preemption_threshold` 与 running request preemption。

### 得分点（10 分）

- 2 分：指出 strict priority 会饿死，并给出 aging。
- 2 分：将 deadline 转为 slack，而不只比较绝对 deadline。
- 2 分：设计可借用的高低优 token quota 或 DRR credits。
- 2 分：抢占发生在 iteration 边界，有 hysteresis 和成本选择。
- 2 分：给出分优先级的 SLO、starvation 和 preemption 验收指标。

### 常见失分点

只有一个 priority heap；每来一个高优请求就立即打断 GPU kernel；被抢占后清零等待年龄；只证明高优更快而没有证明低优有进展。

---

## 10. Paged KV Cache 解决什么问题，block size 如何权衡

### 完整答案

传统连续 KV Cache 如果在请求开始时按 `max_seq_len` 预留，会让短请求占用大量永远不会使用的空间；如果按实际长度动态扩容，又会产生拷贝、外部碎片和地址移动。不同序列长度还让 continuous batching 很难高效复用被释放空间。

Paged KV Cache 将 token 的 K/V 按固定大小 block/page 存储。每个请求只维护逻辑 token 位置到物理 block ID 的表，物理块可以来自显存池任意位置。序列增长时追加新 block，结束时把 block 归还池中，不需要移动其他请求。固定大小分配基本消除了可变长度对象造成的外部碎片，也使 prefix cache 可以让多个请求引用同一物理 block。

设 block size 为 `B`：

- 内部碎片：每个活动序列最后一块平均浪费接近 `B/2` token，短请求占比高时更明显。
- 元数据：block 数约为总 token 数除以 `B`。`B` 越小，block table、hash、refcount、调度和 slot mapping 越多。
- Prefix 粒度：通常只有 page-aligned 或完整 block 的前缀能直接复用。`B` 越小，分叉点附近少重算，但查找和管理开销更高。
- Kernel 访问：较大 block 更连续，page table 间接寻址更少；过大则降低并发容量与缓存复用粒度。最佳值依赖 attention kernel 的 page-size 支持和 head layout。
- 抢占与淘汰：小 block 可以更细粒度释放；大 block 一次释放更多，但容易因为少量尾部 token 保留整块。

KV 每 token 的基础显存可以近似为：

```text
bytes_per_token_per_rank =
    local_layers * 2(K+V) * local_kv_heads * head_dim * kv_dtype_bytes
```

实际还要考虑 KV head 复制、MLA latent cache、hybrid attention、多 cache group、对齐和 allocator metadata。容量最终向 block size 取整。

### 源码结合

- vLLM [`block_pool.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/block_pool.py)：`BlockPool`、`KVCacheBlock`、hash map 和 fixed block free queue。
- vLLM [`kv_cache_utils.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_utils.py)：`FreeKVCacheBlockQueue` 使用侵入式双向链表，支持 O(1) 移除中间命中块。
- vLLM [`kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py)：`allocate_slots()` 将 computed、external、new 和 lookahead token 转成物理块。
- SGLang [`memory_pool.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/memory_pool.py)：request-to-token 与 token-to-KV 两级内存池；[`radix_cache.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py) 将 token 前缀映射到 KV indices。

### 得分点（10 分）

- 2 分：解释连续预留的浪费、扩容拷贝和外部碎片。
- 2 分：讲清逻辑 block table 到物理 block 的间接映射。
- 3 分：同时讨论内部碎片、metadata、prefix 粒度和 kernel 访问。
- 2 分：能写出标准 MHA/GQA 的 KV bytes/token 公式。
- 1 分：指出实际模型可能有 MLA、hybrid cache 或 head replication 修正。

### 常见失分点

声称 paging 消除了所有碎片；把 block size 当作 CUDA thread block；忽略每个活动请求的尾块浪费和 kernel 支持约束。

---

## 11. 比较 vLLM prefix caching 与 SGLang Radix Cache

### 完整答案

两者都复用已经计算的 KV，但索引结构不同。

vLLM 把 prompt 按 hash block size 切分，当前 block hash 由前序 hash、当前 token block 和额外 namespace 信息链式构造。`BlockHashToBlockMap` 从 hash 找到一个或多个物理 `KVCacheBlock`，`find_longest_cache_hit()` 从序列开头连续匹配，任一 cache group miss 就在相应边界停止。命中 block 通过 `touch()` 增加引用并从 free/LRU 队列移除。完整 block 在释放后仍可保留 hash，refcount 为 0 时成为可淘汰候选。为保持 request block table append-only，当前实现允许同一 hash 对应多个物理 block，而不强制去重。

SGLang 的 `RadixCache` 是压缩前缀树。树边 `TreeNode.key` 可以保存一段 token，`value` 保存对应 KV indices。查找在边内部结束时会 split node，从而显式表示新的分叉边界。相同系统 prompt、多轮会话和工具调用模板天然共享祖先路径。`extra_key` 用于隔离 LoRA、cache salt 等逻辑 namespace；请求通过 `lock_ref` 保护从叶子到根的路径，淘汰只从未锁定的叶子开始。SchedulePolicy 还能使用 LPM 或 DFS-weight 按共享前缀组织 waiting queue，并支持 in-batch prefix caching。

负载判断如下：

- 多轮 Agent、树状分支、长系统 prompt：二者都能获益，SGLang 的树结构和 cache-aware policy 更直接表达共享关系。
- 大量完全相同、block-aligned 的长前缀：vLLM 的链式 hash 查找简单直接，Radix Tree 不必然更快。
- 完全随机 prompt：命中率接近零，二者都只增加 hash/tree 管理开销，应测量后考虑关闭。
- 分叉点频繁且不对齐 page：较小 page 或 Radix split 能提高复用粒度，但最终仍受 attention backend/page alignment 约束。

还要注意：即使整个 prompt 命中，目标模型通常仍需重新计算最后一个 token 才能得到下一个 token logits。vLLM 的 `get_computed_blocks()` 因此把最大命中长度限制为 `num_tokens - 1`，并可能因 block 对齐而重算不止一个 token。

### 源码结合

- vLLM [`block_pool.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/block_pool.py)：`BlockHashToBlockMap`、`cache_full_blocks()`、`touch()`、`free_blocks()`。
- vLLM [`single_type_kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/single_type_kv_cache_manager.py)：最长连续 block hit 与 running request 的 cached block tracking。
- SGLang [`radix_cache.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py)：`match_prefix()`、`insert()`、`_split_node()`、`cache_unfinished_req()`。
- SGLang [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)：LPM、DFS-weight、in-batch prefix matching。

### 得分点（10 分）

- 3 分：准确说出 vLLM 是链式 block hash，不把它描述成 Radix Tree。
- 3 分：准确说出 SGLang 压缩树的边分裂、KV indices 和路径锁。
- 2 分：能按多轮、相同前缀、随机 prompt 分析负载。
- 1 分：提到 LoRA/cache salt 等 namespace 隔离。
- 1 分：说明全命中仍需 logits 边界计算和 page alignment。

### 常见失分点

只说“SGLang prefix cache 更强”；认为相同 token 一定可以跨 LoRA/模型权重共享；忽略树维护和 page alignment 成本。

---

## 12. Prefix Cache 中正在被请求引用的节点能否淘汰

### 完整答案

正在被执行请求引用的 KV 物理存储不能被重新分配，否则该请求下一轮 attention 会读取其他序列写入的数据。可以从“可命中索引”中删除其 hash/tree metadata，使新请求不再命中，但只要 refcount/lock 仍大于零，底层 block 就不能进入 allocator 的可分配集合。

vLLM 的 block 有 `ref_cnt`。Prefix hit 时 `touch()` 增加引用；如果该 block 原先 refcount 为 0 且位于 free queue，会先从队列 O(1) 移除。请求完成时 `free_blocks()` 减引用；只有变成 0 才进入 free/LRU queue。`evict_blocks()` 可以清除仍在使用 block 的 hash，但不会因 refcount 大于零就把它作为自由物理块分配。

SGLang 对 Radix node 使用路径 `lock_ref`。`inc_lock_ref(leaf)` 沿祖先递增，节点从 evictable 转为 protected；`dec_lock_ref()` 在引用归零时恢复可淘汰状态。`evict()` 从 `evictable_leaves` 按策略选叶子，释放 node value，并在父节点成为未锁叶时继续向上。

一个稳健的并发方案应包含：

1. Scheduler 单写所有权，或对 refcount 使用原子操作；查找成功与加引用必须是一个不可分割事务。
2. 只有 `refcount == 0` 且所有相关 CUDA/transfer event 已完成的块才能放入 free allocator。
3. LRU 只决定“零引用候选”的顺序，不能越过引用保护。
4. block descriptor 使用 `(block_id, generation)`。物理 block 每次重新分配时 generation 加一；异步结果携带旧 generation 就被拒绝，防止 ABA。
5. Reset、权重更新、PD connector 和 abort 都要走同一个 release protocol，且 release 幂等。

ABA 场景是：请求 A 保存 block 17，A 被异步 abort；block 17 释放后分给 B；A 的迟到回调再次释放或写 block 17。只有 refcount 不足以区分“同一 ID 的新生命周期”，因此 generation/fence 很重要。

### 源码结合

- vLLM [`block_pool.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/block_pool.py)：`touch()`、`free_blocks()`、`_maybe_evict_cached_block()`。
- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：async scheduling 下的 deferred free/fence，防止 GPU 尚在写时复用块。
- SGLang [`radix_cache.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py)：`inc_lock_ref()`、`dec_lock_ref()`、`evict()` 与 `evictable_leaves`。
- SGLang overlap 路径还通过 CUDA event、batch record buffer 和 tensor keep-alive 保护跨 stream 生命周期。

### 得分点（10 分）

- 2 分：区分删除 cache 索引与回收物理存储。
- 2 分：讲清 refcount/lock 与 LRU 的关系。
- 2 分：查找加引用具有原子性或 scheduler 单写所有权。
- 2 分：考虑 CUDA/transfer 尚未完成时的 deferred free。
- 2 分：能描述 ABA，并给出 generation/fence 或等价方案。

### 常见失分点

认为 LRU 最旧节点可无条件淘汰；只在 Python 对象层计数而忽略 GPU 异步写；abort 和正常完成走两套不一致释放逻辑。

---

## 13. 为什么设置 0.9 的 GPU memory utilization 仍会 OOM

### 完整答案

`gpu_memory_utilization` 或 `mem_fraction_static` 通常用于决定启动时为权重和 KV pool 预留多少空间，不是 CUDA allocator 的硬隔离上限。运行时还有许多不在简单 KV 公式中的分配，而且它们的峰值未必出现在启动 profile 形状中。

应按以下账本核算：

```text
GPU peak =
    model weights + quant scales/metadata
  + KV pools + encoder/Mamba/MLA auxiliary cache
  + eager activations and attention/GEMM workspace
  + CUDA Graph private pool and static input/output buffers
  + NCCL/custom all-reduce/all-to-all buffers
  + sampler/logprob/grammar/speculative temporary tensors
  + LoRA adapters and multimodal encoder buffers
  + allocator fragmentation and framework context
  + overlap 中同时存活的多 iteration buffers
```

常见 OOM 原因包括：CUDA Graph capture 为多个 bucket 保留内存；某 attention backend 在最长序列或最大 batch 请求更大 workspace；TP/NCCL 首次通信延迟分配 buffer；返回大 logprobs 或结构化 mask；EAGLE 同时存在 draft/target KV；异步 overlap 让两轮临时 tensor 同时存活；其他进程在 profile 后占用 GPU；PyTorch reserved memory 中存在不能满足大连续分配的碎片。

诊断时要区分 `allocated`、`reserved`、driver free 和各内存池逻辑使用量。在模型加载、KV 初始化、graph capture、首次 Prefill、最大 Decode batch 和功能特性开启后分别记录 snapshot。使用 memory history 或 allocator snapshot 找到 peak stack，不能只看 `nvidia-smi`。

治理顺序通常是：为非 KV 分配留真实 headroom；减少 graph capture bucket/max batch；降低 `max_num_batched_tokens`/`max_num_seqs`；关闭不需要的 logprob/LoRA/spec 功能；使用更小 KV dtype；最后才是简单降低 KV fraction。`empty_cache()` 只能归还空闲 cached segment，不能释放仍有 tensor/graph 引用的内存。

### 源码结合

- vLLM [`gpu_worker.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu_worker.py)：profile available memory 并初始化 KV cache。
- vLLM [`cudagraph_utils.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/cudagraph_utils.py) 与 [`cuda_graph.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/compilation/cuda_graph.py)：多个 descriptor、静态地址和 graph pool。
- SGLang [`model_runner.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/model_runner.py)：内存池、attention backend、graph 的分阶段初始化。
- SGLang [`decode_cuda_graph_runner.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)：按最大 capture batch 创建静态 buffers；Scheduler 源码中也明确提示 memory profile 需考虑 graph memory。

### 得分点（10 分）

- 2 分：明确 fraction 不是硬显存隔离。
- 3 分：至少列出权重、KV、workspace、graph、通信和碎片六类账本。
- 2 分：指出 overlap/spec/logprob 等功能改变运行峰值。
- 2 分：给出分阶段 snapshot 和 allocated/reserved/free 的诊断方法。
- 1 分：说明 `empty_cache()` 的能力边界。

### 常见失分点

只回答“内存碎片”；把 `nvidia-smi used` 当作 PyTorch live tensor；降低 KV fraction 后不检查 graph/communication 峰值。

---

## 14. 比较 recompute、swap、抢占和拒绝请求

### 完整答案

这四个词处于不同层次。抢占是 Scheduler 的策略动作，抢占后可以选择 recompute、swap 或直接终止；拒绝则发生在准入前或无法继续时。

**Recompute**：释放被抢占请求的 KV，保留 token IDs，重新获准时再次 Prefill 已生成上下文。它不需要 CPU KV 空间和 PCIe 传输，但浪费 GPU 计算。短上下文、prefix cache 可重新命中、PCIe 较慢时较合适；超长上下文会显著恶化恢复后的 TTFT/TPOT，并污染总体吞吐。

**Swap/offload**：把 KV 转移到 CPU、NVMe 或远程 cache，恢复时再搬回。它保留已做计算，但需要额外内存、带宽、DMA buffer 和传输状态机。只有 `transfer_time < recompute_time` 且传输能与其他计算重叠时才有价值。PCIe 拥塞或大量请求同时恢复时，会造成新的尾延迟和 head-of-line blocking。

**抢占**：在资源不足或高优请求到来时，选择 victim，并在安全 iteration 边界停止后续调度。victim 选择应考虑 priority、已计算 token、缓存可恢复比例、transfer 状态、deadline 和 preemption count。反复抢占同一请求会形成 livelock，因此需要最小 quantum、aging 和抢占次数上限。

**拒绝请求**：在 admission 阶段预测无法满足容量/SLO，直接返回 429/503。它不浪费模型计算，也不会伤害已经接收请求的 SLO，是过载保护中最确定的方案；代价是成功率降低，需要上游 retry budget、排队或降级实例。

对高优在线服务，推荐“准入拒绝优先，少量 iteration-boundary 抢占兜底；短上下文 recompute，超长且高速互联可 swap”。评价要同时看有效 tokens/s、recompute tokens、swap bytes、preemption count、deadline miss、恢复延迟和用户可见错误率。

### 源码结合

- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：`_preempt_request()` 当前释放 request blocks、增加 `num_preemptions` 并重新排队，表现为 recompute 型恢复。
- vLLM [`kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py)：watermark、full-sequence admission 和 external KV connector 可减少过度准入。
- SGLang [`schedule_batch.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py)：`retract_decode()` 选择并撤回请求。
- SGLang [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`update_running_batch()` 在 KV 不足时 retraction，并更新 `new_token_ratio`；HiCache/PD 路径提供更复杂的外部缓存与传输选择。

### 得分点（10 分）

- 2 分：区分“抢占策略”与“抢占后的恢复机制”。
- 2 分：正确比较 recompute 的 GPU 成本与 swap 的传输成本。
- 2 分：victim 选择考虑 work done、priority、deadline 和恢复成本。
- 2 分：说明 admission rejection 对已接收请求 SLO 的保护作用。
- 2 分：给出量化选择条件和完整指标，而非固定选某一种。

### 常见失分点

把 preemption 与 swap 当成同义词；假设 CPU swap 一定比 recompute 快；过载时继续接收所有请求；没有防止同一请求反复抢占。

---

## 15. 描述一次 Decode iteration，并定位 CPU-GPU 同步点

### 完整答案

以普通自回归 Decode 为例，一轮并不只是调用一次 `model(input_ids)`：

1. Scheduler 为每个 running request 安排一个 token，确定 request order、KV slot、block table 和可选 speculative/grammar 元数据。
2. Worker/ModelRunner 将新请求加入持久 input batch，移除完成请求，并把 token ID、position、sequence length、slot mapping、sampling params 等写入 CPU staging 或 GPU static buffer。
3. Attention backend 根据 block table 构造 metadata。Graph 路径把实际 batch pad 到 capture bucket 并原地更新静态 buffer；eager 路径按实际形状执行。
4. 模型运行 embedding、逐层 attention、MLP/MoE 和 TP collective。每层 attention 读取历史 KV，并把当前 token K/V 写入 Scheduler 分配的 slot。
5. LM head 得到 logits。Sampler 应用 temperature、penalty、top-k/top-p、logit bias、grammar mask 等，生成 token ID；speculative 模式还要执行 target verification/rejection sampling。
6. sampled IDs、必要 logprob 和状态通过 D2H 或 GPU relay 返回。Scheduler 更新 token 数、停止状态和 KV ownership，开始下一轮。

潜在 CPU-GPU 同步点包括：对 CUDA tensor 调用 `.item()`、`.tolist()`、同步 `.cpu()`；显式 `torch.cuda.synchronize()`；CPU 在 event 上阻塞；需要从 GPU 结果决定 Python 控制流；动态 allocation 触发 allocator 同步；某些 profiler/log 读取 tensor；D2H 后立刻访问内存；异常检查；首次 JIT/kernel load。NCCL collective 会形成 GPU stream 依赖，但不一定同步 CPU，不能把所有 collective 都误判为 host sync。

定位步骤：

1. 用框架 Step/rid/batch 日志确认是 schedule、prepare、forward、sample 还是 result 阶段。
2. 用 NVTX range 或 PyTorch profiler 标记 CPU op、CUDA kernel、memcpy 和 collective。
3. 用 Nsight Systems 看 GPU kernel 之间的空白，检查空白前 CPU thread 在做什么，以及是否有 `cudaStreamSynchronize`/`cudaEventSynchronize`。
4. 对可疑 `.item()` 或 D2H 改为 async copy + event，比较 TPOT；`CUDA_LAUNCH_BLOCKING=1` 只用于定位错误，不能作为性能基准。
5. 再用 Nsight Compute 分析 kernel 内部性能，不能用它替代系统级同步分析。

### 源码结合

- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py) 生成 `SchedulerOutput`，随后 [`gpu_worker.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu_worker.py) 调用 ModelRunner。
- vLLM [`gpu/model_runner.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/model_runner.py) 与 [`gpu_model_runner.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu_model_runner.py) 是当前并存的 V2/V1 runner，负责 batch update、input prepare、forward 和 sampler。
- SGLang [`tp_worker.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tp_worker.py) 创建 `ForwardBatch`，[`model_runner.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/model_runner.py) 负责 backend dispatch、forward 和 sample。
- SGLang overlap 路径用 forward/copy stream 和 `copy_done` event 将 D2H 与下一轮 forward 重叠。

### 得分点（10 分）

- 3 分：从 SchedulerOutput/Batch 讲到 attention、sampler 和结果更新。
- 2 分：能指出 KV read/write、slot mapping 与 block table。
- 2 分：列出 `.item()`、D2H、event、allocator 等真实同步来源。
- 2 分：正确区分 GPU collective 依赖和 CPU host synchronization。
- 1 分：能给出 Nsight Systems 优先、Nsight Compute 后续的定位顺序。

### 常见失分点

把 Decode 简化成一个 PyTorch forward；看到 GPU 空白就认定 kernel 太慢；用 `CUDA_LAUNCH_BLOCKING=1` 的结果评价生产性能。

---

## 16. CUDA Graph 为什么降低 Decode 延迟，动态特性带来什么挑战

### 完整答案

普通 eager 执行每轮都由 CPU 依次 launch 大量小 kernel。Decode 的单 kernel 工作量较小，Python、dispatcher 和 CUDA launch 的固定开销占比很高。CUDA Graph 在 capture 时记录 kernel、依赖、参数地址和内存操作，replay 时用一次较轻量的提交重放整段 DAG，从而减少 CPU launch 开销和 kernel 间空洞。

Graph 的基本约束是地址、拓扑和许多 shape 必须稳定。框架通常预分配静态 input/output buffer，并把运行 batch 向上 pad 到最小 capture bucket。例如真实 batch 33 可能使用 batch 64 的图。这样提高命中率，但浪费 31 个 padding slot 的算力和静态显存。

动态特性的挑战包括：

- 动态 batch/sequence：使用 bucket、slot mapping 和原地 metadata update；超过最大 bucket 或不支持的 attention shape 时回退 eager。
- LoRA：活跃 adapter 数量和权重指针会变化。需要捕获不同 LoRA case，或使用稳定的 dense metadata/buffer；超出 capture case 时回退或 pad。
- MoE：每 token 路由 expert 和 all-to-all size 动态，需要固定容量、padding、graph-compatible communicator 或 piecewise break。
- 结构化输出：FSM 每步改变 vocab mask。mask tensor 地址可固定、内容原地更新，但 grammar 准备和 CPU bitmask 生成不能阻塞 replay；延迟 sample closure 还要及时释放 logits/mask 引用，避免 graph 外显存泄漏。
- Speculative decoding：每请求 verify token 数、tree shape 和 acceptance 动态，通常需要独立 TARGET_VERIFY graph 和 padding 策略。
- 多模态/encoder：图中是否包含 cross-attention、encoder length 和不同 modality 必须在 descriptor 中表达。

vLLM 区分 FULL 与 PIECEWISE graph，使用 batch descriptor、capture sizes 和稳定地址；SGLang 有 decode/prefill phase runner，并可选择 full、breakable、TC piecewise 等 backend。Graph 不是免费优化：它增加启动 capture 时间、静态 buffer/graph pool 显存和 bucket padding。性能比较必须同时报告 graph hit rate、padding ratio、capture memory 和 eager fallback 次数。

### 源码结合

- vLLM [`cuda_graph.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/compilation/cuda_graph.py)：`CUDAGraphWrapper` 根据 runtime mode 与 `BatchDescriptor` capture/replay，并在 debug 下校验 tensor address。
- vLLM [`cudagraph_utils.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/cudagraph_utils.py)：`CudaGraphManager` 管理 FULL/PIECEWISE、capture sizes 和 LoRA capture cases。
- SGLang [`base_cuda_graph_runner.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py)：bucket 选择、static buffer 和 backend 抽象。
- SGLang [`decode_cuda_graph_runner.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)：DECODE/TARGET_VERIFY 模式、LoRA batch info 和静态 `DecodeInputBuffers`。

### 得分点（10 分）

- 2 分：解释减少的是 host launch overhead 和 kernel gap。
- 2 分：讲清静态地址、shape bucket 和 padding。
- 3 分：至少深入分析 LoRA、MoE、grammar、spec 中三个动态问题。
- 2 分：知道 graph miss 需要 eager/piecewise fallback，不能强行 replay。
- 1 分：同时评价 hit rate、padding 与额外显存。

### 常见失分点

认为 CUDA Graph 会让单个 kernel 算得更快；忽略静态地址；只开 graph 不统计 padding 和 fallback；把 Torch Compile 与 CUDA Graph 当成同一件事。

---

## 17. Tensor Parallel 中 Attention/MLP 需要哪些通信

### 完整答案

经典 Megatron 风格 TP 将线性层分为 column parallel 和 row parallel。

Attention 中，QKV projection 通常是 column-parallel：每个 rank 持有部分输出 head，计算本地 Q/K/V，本地 attention 不需要立即 gather。后续 output projection 是 row-parallel：每个 rank 对自己的一部分输入和权重计算 partial output，最后通过 all-reduce 求和，使下一层得到完整 hidden state。GQA 中 KV head 数小于 TP size 时可能复制 KV head，具体切分不能只用 `num_kv_heads / tp` 粗暴计算。

MLP 中 gate/up projection 通常 column-parallel，每 rank 持有中间维的一部分，激活函数在本地执行；down projection 是 row-parallel，产生 partial hidden，随后 all-reduce。MoE 则可能另外按 expert parallel 做 all-to-all，不能只看 TP all-reduce。

三个 collective 的语义：

- all-reduce：各 rank 对同形状 partial tensor 求和，并让每 rank 都得到完整结果。RowParallelLinear 的默认输出常用它。
- all-gather：拼接每 rank 的 shard，使每 rank 获得完整 tensor。ColumnParallelLinear 只有下游确实需要完整输出时才开启，否则会浪费带宽。
- reduce-scatter：先求和再把结果分片。若下一算子能继续消费 shard，它比 all-reduce 后再切分更省通信和显存，也便于和 GEMM 融合。

通信是否值得取决于模型能否放入单卡和互联。NVLink/NVSwitch 提供高带宽、低延迟，较大 TP 可以换取模型容量与更小本地 GEMM；PCIe 上每层多次 collective，尤其 Decode 的小消息延迟，会很快盖过计算收益。跨节点 TP 还受 NIC、拓扑和 NCCL algorithm 影响，通常优先节点内 TP、节点间 PP/DP/EP。选择 TP size 应通过每层 compute/communication overlap 和 p99 测量，而不是默认 GPU 数越多越快。

### 源码结合

- vLLM [`linear.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/model_executor/layers/linear.py)：`ColumnParallelLinear` 可选 `gather_output`；`RowParallelLinear` 默认 `tensor_model_parallel_all_reduce()`。
- vLLM [`communication_op.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/distributed/communication_op.py) 与 [`parallel_state.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/distributed/parallel_state.py)：TP group 的 all-reduce/all-gather/reduce-scatter。
- vLLM [`cuda_communicator.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/distributed/device_communicators/cuda_communicator.py)：在 custom all-reduce、symmetric memory、PyNCCL 等实现间 dispatch。
- SGLang [`linear.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/layers/linear.py) 和 [`communication_op.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/distributed/communication_op.py) 采用相同的 column/row contract，并对 attention/MoE 有专用通信入口。

### 得分点（10 分）

- 3 分：正确描述 QKV/attention output 与 MLP up/down 的切分。
- 2 分：准确区分 all-reduce、all-gather 和 reduce-scatter。
- 2 分：说明为何能保持 shard 时不应过早 all-gather。
- 2 分：分析 NVLink、PCIe、跨节点对 TP size 的影响。
- 1 分：提到 GQA KV head replication 或 MoE all-to-all 的特殊性。

### 常见失分点

认为每个 Linear 后都 all-reduce；混淆数据并行的梯度 all-reduce；忽略 Decode 小消息的 latency-bound 特征和物理拓扑。

---

## 18. Speculative decoding 的收益公式与失效场景

### 完整答案

推测解码让较便宜的 draft 路径先提出 `K` 个候选 token，target model 在一次并行 forward 中验证这些位置。验证必须使用 rejection sampling 或等价的概率修正，保证最终分布仍等价于 target model；简单地“相同就接受、不同就取 target argmax”只在特定 greedy 条件下正确。

设单个普通 target Decode step 成本为 `C_t(1)`；产生 K 个 draft 的总成本为 `C_d(K)`；target 并行验证 K 个位置的成本为 `C_v(K)`；调度、采样和 padding 额外开销为 `O`。如果每个 draft 条件接受概率近似为 `p` 且相互独立，一轮期望输出 token 数为：

```text
E[L] = 1 + p + p^2 + ... + p^K
     = (1 - p^(K+1)) / (1 - p)

speedup ~= E[L] * C_t(1) / (C_d(K) + C_v(K) + O)
```

其中最后的 `1` 是第一个拒绝位置由 target 重新采样的 token，或全接受后的 bonus token。真实 EAGLE/tree speculation 不满足独立链假设，应直接使用 accepted length 分布计算。

低接受率时，`E[L]` 接近 1，但仍支付 draft、verify、额外 KV slot、sampler 和 graph padding，因此比普通 Decode 慢。其他失效场景包括：draft model 太大；target verify 对 batch/上下文不够高效；温度高或 domain shift 降低接受率；小 batch 下 draft launch 不能摊薄；TP 通信翻倍；结构化 grammar 拒绝大量 draft；K 太大造成 KV/graph padding和显存压力。

Scheduler 必须为 lookahead token 预分配 KV。验证后，被拒绝位置不能算作 target 已提交上下文，需要修正 `num_computed_tokens`、输出 placeholder 和 block 状态。Grammar 还要先过滤非法 draft。vLLM 在 SchedulerOutput 中携带 `scheduled_spec_decode_tokens`，`update_from_output()` 根据生成长度计算 accepted/rejected 并回退状态；SGLang 将 draft worker、TARGET_VERIFY ForwardMode、draft KV 和 target result processor组成完整状态机。

### 源码结合

- vLLM [`config/speculative.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/config/speculative.py)：spec method、draft/num token 等配置。
- vLLM [`v1/spec_decode`](https://github.com/Cptzzzzz/vllm/tree/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/spec_decode) 与 [`worker/gpu/spec_decode`](https://github.com/Cptzzzzz/vllm/tree/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/spec_decode)：draft proposer、EAGLE/MTP 与 rejection sampler。
- vLLM [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：lookahead allocation、grammar validation、accepted/rejected 统计及 computed token 回退。
- SGLang [`speculative`](https://github.com/Cptzzzzz/sglang/tree/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative) 中的 `eagle_worker_v2.py`、`eagle_utils.py` 和独立 draft CUDA Graph runner 实现 draft/verify/accept pipeline。

### 得分点（10 分）

- 2 分：解释 draft、target verify、accept/reject 和分布正确性。
- 3 分：写出期望输出长度与近似 speedup 公式。
- 2 分：指出低接受率仍支付 draft/verify/overhead，可能减速。
- 2 分：说明 Scheduler 的 lookahead KV 与 rejection rollback。
- 1 分：能扩展到 grammar、tree speculation 或 TP 通信影响。

### 常见失分点

把吞吐提升直接写成 K 倍；忽略 bonus/residual token；只测 acceptance rate 不测 draft 与 verify 成本；拒绝 token 后没有回滚 KV/sequence state。

---

## 19. Prefill/Decode 分离如何设计，何时会得不偿失

### 完整答案

Prefill 处理长 prompt，计算密集、耗时方差大，主要影响 TTFT；Decode 每步工作小但持续读取大量权重/KV，主要影响 TPOT。共置时，一个长 Prefill iteration 会阻塞所有 Decode。PD 分离让 P 节点用大 token batch 优化 Prefill，让 D 节点用稳定小 iteration 优化 Decode，并分别扩缩容。

一个完整流程是：

1. Router 根据 P/D 负载、模型版本、prefix locality 和租户选择节点，生成全局 request/transfer ID。
2. D 节点预留 request slot 与 KV 目标地址，建立 receiver/bootstrap room；预分配失败应在 P 做昂贵 Prefill 前返回。
3. P 节点进行本地/外部 prefix match，只计算未命中 suffix。KV 按 layer/block 通过 RDMA、NIXL、Mooncake 或其他 connector 发送。
4. 传输可以与 suffix Prefill 重叠，但 D 只有在相关 KV range 完整、版本和 layout 校验通过后才能 Decode。
5. P/D 交换完成 ACK，Router 将流式输出绑定到 D；正常完成或 abort 后双方幂等释放 transfer 和 KV 资源。

故障协议至少要包含 model/weight version、dtype、block layout、TP rank mapping、token range、checksum/generation 和 timeout。P 失败可在另一 P 重算；D 失败需要重新选择 D 并重传或重算。两端都必须能处理 late ACK、重复 abort 和 partial transfer，不能依赖“连接关闭自然清理”。

KV 传输量近似为 `prompt_tokens * bytes_per_token`。只有满足以下近似条件时分离才有收益：

```text
saved_decode_interference + better_P_batching + better_D_batching
    > bootstrap + KV_transfer + synchronization + extra_queueing
```

短 prompt、低并发、慢 TCP/PCIe、大 KV、跨节点 TP layout 重排、D 节点本来已经有 prefix cache 时，传输成本可能超过收益。还要警惕 P 快而 D 慢导致 KV inflight queue 膨胀。

核心指标包括 P queue/TTFT、D queue/TPOT、KV bytes 与有效带宽、bootstrap latency、transfer overlap ratio、D preallocation failure、orphan allocation、重试率和端到端 p99。

### 源码结合

- vLLM [`distributed/kv_transfer`](https://github.com/Cptzzzzz/vllm/tree/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/distributed/kv_transfer) 定义 connector、scheduler/worker 角色和 NIXL、Mooncake、LMCache 等实现；[`config/kv_transfer.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/config/kv_transfer.py) 定义角色配置。
- vLLM Scheduler 的 external matched tokens、async KV load 与 delayed block free 将远端 KV 纳入正常 token/block 账本。
- SGLang [`disaggregation`](https://github.com/Cptzzzzz/sglang/tree/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/disaggregation) 提供 prefill、decode 和 transfer backend；Scheduler 有独立 bootstrap、prealloc、transfer、inflight queue。
- SGLang [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py) 在 PD 模式选择专用 event loop，支持 cached prefix 早发、suffix forward 和传输重叠，并在 abort 中逐队列清理 sender/receiver。

### 得分点（10 分）

- 2 分：从 Prefill/Decode 硬件和 SLO 差异解释分离动机。
- 2 分：讲清 D 预分配、P 计算、KV 传输、ACK 和 Decode 顺序。
- 2 分：包含 model version、layout、幂等和 partial failure 协议。
- 2 分：给出收益不等式和短 prompt/慢网络等反例。
- 2 分：提出 KV 带宽、overlap、orphan、P/D queue 等指标。

### 常见失分点

只部署两个服务就称为 PD 分离；先 Prefill 再发现 D 无空间；不校验模型版本和 TP layout；只看 P/D 单机吞吐，不看 KV 网络和端到端 p99。

---

## 20. 吞吐正常但周期性 TPOT p99 暴涨，如何定位

### 完整答案

先确认指标定义和范围。TPOT 应是首 token 之后相邻输出 token 的间隔，不能把 queue wait 或 Prefill 混入。周期性 p99 表示平均吞吐掩盖了少量长 iteration，优先寻找与周期一致的事件，而不是一开始就调 kernel。

推荐按以下层次建立因果链：

1. **请求层**：把 spike 时刻与输入长度、输出长度、streaming、logprob、grammar、LoRA、prefix share 和租户相关联。如果只在某类请求出现，先复现该 shape。
2. **调度层**：查看每轮 Prefill/Decode token、batch size、iteration duration、waiting/running、KV usage、cache hit、preemption/retraction。周期性长 Prefill chunk、cache eviction 后重算或抢占恢复，都会造成 TPOT 尖峰。
3. **CPU 层**：用 `py-spy`/`perf` 分进程采样。检查 Python GC、tokenizer/detokenizer、hash/Radix 遍历、grammar bitmask、日志格式化、Prometheus scrape、ZMQ 序列化和 CPU 被 cgroup throttling。vLLM 会 freeze startup heap 来减少老年代 GC，但运行期新对象仍可能触发停顿。
4. **CUDA Graph/编译层**：记录 graph descriptor、hit/miss、padding bucket、eager fallback、首次 shape 编译/JIT 和 recapture。若 spike 只发生在新 batch shape，应先固定 shape 或预热验证。
5. **GPU 层**：Nsight Systems 对齐 spike 窗口。GPU 空白且 CPU thread 忙，说明 launch/host bottleneck；连续长 kernel 则看 attention/GEMM shape；出现 D2H/stream wait 则检查隐藏同步。
6. **分布式层**：记录每个 rank 的 collective 起止和 straggler。NCCL p99 可能来自某 rank CPU launch 落后、PCIe/NVLink 竞争、跨节点网络、后台传输或 collective shape 改变。一个 rank 慢会让所有 rank 看起来都卡在 NCCL。
7. **系统层**：检查 GPU clock/功耗、ECC/Xid、CPU steal/throttle、NUMA、磁盘/JIT cache 和同机进程。周期几十秒也可能对应 metrics scrape、日志 flush、cache maintenance 或 checkpoint/adapter 操作。

验证要一次只做一个 ablation：关闭 prefix cache、固定全 Decode、禁用 graph、关闭详细日志、用 token IDs 绕过 tokenizer、单卡去掉 NCCL。哪个操作消除周期，才能缩小根因。不能因为关闭某功能后 p99 好转就立即断言该功能有 bug，还要确认负载和 batch shape没有同步改变。

### 源码结合

- vLLM [`metrics/stats.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/metrics/stats.py) 与 [`metrics/loggers.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/metrics/loggers.py)：TTFT、inter-token latency、queue/prefill/decode time、KV usage、prefix hit 和 preemption。
- vLLM Step 11 到 16 日志可将 Scheduler、GPU、result 和 API 输出对齐；源码见 [`FLOW_TRACE.md`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/FLOW_TRACE.md)。
- SGLang [`metrics_reporter.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler_components/metrics_reporter.py)：batch、cache hit、retraction、new-token ratio 等调度状态。
- SGLang Step 10 到 16 加 `rid`/mode 可以区分 overlap 日志交错与真实 stall；见 [`FLOW_TRACE.md`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/FLOW_TRACE.md)。

### 得分点（10 分）

- 2 分：先验证 TPOT 定义并按 spike 时间关联请求/iteration。
- 2 分：覆盖 Prefill chunk、cache eviction、preemption 等调度原因。
- 2 分：能区分 CPU launch gap、GPU long kernel 和 D2H sync。
- 2 分：考虑 graph miss/JIT 和 NCCL straggler，不只看平均 GPU util。
- 2 分：提出逐项 ablation、跨层时间线和可证伪假设。

### 常见失分点

直接说“Python GC”；只看一分钟平均 GPU utilization；只 profile 正常窗口；同时修改多个参数导致无法归因。

---

## 21. GPU 利用率低但 CPU 单核满载，如何区分瓶颈

### 完整答案

单核满载通常意味着 GPU 的提交线程或某个单线程状态机跟不上。首先按 PID/TID 分辨是哪一个进程，而不是看整机 CPU 总量。

**Tokenizer 瓶颈**：CPU flame graph 集中在 tokenizer、chat template 或 multimodal processor；请求到达 HTTP 后，进入 Scheduler 的速率低。用预先生成的 `input_ids`、关闭 chat template 或增加 tokenizer worker 后 GPU 利用率明显上升，即可确认。要注意增加 tokenizer 并不会解决单个超长 prompt 的串行 tokenizer 算法。

**Scheduler 瓶颈**：GPU kernel 之间有规则空洞，Scheduler 线程集中在 queue sorting、prefix matching、Radix/hash、batch tensor metadata 或 Python 循环。vLLM 可观察 schedule duration；SGLang 可切换 FCFS/关闭 Radix 或比较 overlap。SGLang 源码特别提醒，TP rank 上重复的大型 CPU 多模态处理会推迟 kernel launch。

**IPC/序列化瓶颈**：CPU 集中在 msgspec/pickle/ZMQ send/recv、memory copy 或 syscall；socket queue 增长，Scheduler 本身 compute 时间不高。减小返回 metadata/logprob、批量消息、使用共享内存 tensor transport或提高 IPC batch 后改善。

**Sampler/grammar 瓶颈**：forward kernel 已结束，但下一轮迟迟不 launch；CPU/GPU trace 在 top-k/top-p、penalty、grammar bitmask、detokenizer 或大 logprob 排序。用 greedy、关闭 logprob/structured output 做对照。

**日志/指标瓶颈**：flame graph 在字符串格式化、JSON、stdout lock、Prometheus label 或磁盘写；降低 log level 后直接改善。异步日志仍可能因队列/锁反压主线程。

**CPU launch bottleneck**：flame graph 主要是 PyTorch dispatcher/CUDA launch，单个 kernel 很短且密集。开启并命中 CUDA Graph 后 GPU gap 减少，说明不是 tokenizer，而是 eager launch overhead。

最终应同时给出三个证据：CPU flame graph 的函数、Nsight 中 GPU gap 前的 host activity、以及一个最小 ablation。只有其中一个证据通常不足以定因。

### 源码结合

- vLLM [`async_llm.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py) 与 [`core.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core.py) 分离 API CPU 与 Scheduler CPU，便于按进程采样。
- vLLM [`gpu/model_runner.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu/model_runner.py) 的 input prepare/graph dispatch 可区分 host launch 与 GPU compute。
- SGLang [`tokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py)、[`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)、[`detokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/detokenizer_manager.py) 是三个应分别 profile 的进程。
- SGLang [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py) 在大 waiting queue 时会将昂贵 LPM 调整为 FCFS，正是控制 CPU 调度成本的例子。

### 得分点（10 分）

- 2 分：先定位具体进程/线程，而不是泛称 Python 慢。
- 2 分：给出 tokenizer 输入 IDs 对照实验。
- 2 分：用 GPU gap 与 Scheduler/launch trace 区分调度和 eager launch。
- 2 分：覆盖 IPC、sampler/grammar、日志中的至少两类。
- 2 分：要求 flame graph、系统 trace、ablation 三类证据闭环。

### 常见失分点

看到单核 100% 就盲目增加 API worker；在同一 GPU 上增加多个独立 Scheduler 导致争用；只优化 tokenizer 而请求早已在 Scheduler 排队。

---

## 22. 如何设计公平的 vLLM 与 SGLang benchmark

### 完整答案

公平 benchmark 需要区分两种目标：同功能同配置的机制对比，以及各框架最佳实践的 production 对比。两组都应该做，不能用一个框架默认开启优化、另一个强制 eager 后宣称框架胜负。本机之前的 trace 为了读清控制流关闭了 compile/CUDA Graph，它只能证明可运行，不能用于性能结论。

**环境控制**：固定 GPU 型号与数量、拓扑、功耗/时钟、driver、CUDA/PyTorch、CPU quota/NUMA、容器内存和后台进程。两个服务顺序运行，每轮间冷却并确认 GPU 空闲。

**模型控制**：相同模型权重/revision、tokenizer、chat template、dtype、quantization、KV dtype、max context、TP/PP/DP 和 attention backend能力。若 backend 名称不同，应验证语义/精度等价并分别报告。

**容量控制**：不要只把 memory fraction 都设为 0.9。应报告启动后可用 KV token/block 数、最大 sequence 数、graph pool 和权重显存，使实际并发容量接近。

**运行时控制**：分别做 eager 等功能隔离组，以及各自推荐 graph/compile/warmup 组。统一 structured output、prefix caching、chunked prefill、spec decode、stream interval 和 logprob；不用的功能全部关闭。先完成足够 warmup，另做 cold-start/JIT 测试。

**工作负载控制**：固定输入/输出 token 分布而非只固定平均值；覆盖短短、长短、长长、burst、Poisson open-loop 和 closed-loop concurrency。明确 prefix share ratio、multi-turn 分支、cache salt 和 cache warm/cold。使用相同 token IDs 可隔离 tokenizer，使用真实文本可评价端到端服务。

**测量与判定**：报告请求吞吐、output tokens/s、总 tokens/s、TTFT/TPOT/E2E 的 p50/p95/p99、错误率、preemption、cache hit、显存和功耗。最重要的是 SLO goodput，例如同时满足 `TTFT < 500ms` 且 `TPOT < 50ms` 的完成请求数/秒。到达率必须逐级提升直到 SLO 拐点，而不是只比较饱和吞吐。

使用固定随机种子、原始逐请求结果、至少多轮重复和置信区间。Prefix benchmark 每轮必须明确是否 reset cache；否则运行顺序会决定赢家。

### 源码结合

- vLLM [`benchmarks/benchmark_serving.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/benchmarks/benchmark_serving.py)、`benchmark_latency.py`、`benchmark_throughput.py` 和 `benchmark_prefix_caching.py` 提供不同层次工具。
- SGLang 的 [`benchmark`](https://github.com/Cptzzzzz/sglang/tree/9529c2e96351f3b1d5715eaa89989b20723d60b5/benchmark) 目录包含 serving、multi-turn、HiCache、structured JSON、speculative 和 kernel 场景，应选择与目标负载一致的 driver。
- 两框架的启动配置与功能 trace 见 [`RUNTIME_VALIDATION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/RUNTIME_VALIDATION.md)，其中明确 trace run 禁用了 graph/compile。

### 得分点（10 分）

- 2 分：区分 iso-feature 与 best-tuned 两组实验。
- 2 分：控制模型、精度、backend、parallelism 和硬件拓扑。
- 2 分：用实际 KV capacity 而非只对齐 memory fraction。
- 2 分：工作负载包含长度分布、arrival model 和 prefix share。
- 2 分：以 SLO goodput、p99、错误率和置信区间判定。

### 常见失分点

只测单请求 latency 或饱和 tokens/s；不同 chat template 导致 token 数不同；缓存未 reset；一个框架 warm、另一个冷；忽略失败请求后吞吐看起来更高。

---

## 23. 不同业务场景如何选择框架

### 完整答案

选型不能脱离固定版本、模型和 workload benchmark。以下是基于当前源码架构的默认倾向，而不是永久结论。

| 场景 | 默认倾向 | 决定性依据 |
| --- | --- | --- |
| 标准 OpenAI 推理服务 | vLLM | API/EngineCore 边界清晰，通用模型、协议、executor 和生态集成面广；团队更容易把它作为标准模型 serving engine。仍需对目标模型实测。 |
| 长共享前缀、多轮 Agent | SGLang | Radix Tree、LPM/DFS-weight、in-batch prefix、session/HiCache 与 runtime 调度结合紧，能显式利用树状共享。vLLM block-hash prefix cache 仍可能足够。 |
| 严格结构化生成 | 条件选择，常偏 SGLang | 两者都有 grammar/bitmask 和 speculative 兼容处理。SGLang 在结构化程序与 runtime 优化上更一体化；vLLM 的 OpenAI structured output 接口更适合通用服务。必须测 schema 编译、首 token 和长 JSON Decode。 |
| 异构 GPU 集群 | 先做同构资源池，而不是直接二选一 | 不应把性能不同的 GPU 放进同一个同步 TP group，最慢 rank 会拖住全组。按 GPU/model 建同构 pool，由 router 做能力和负载路由。vLLM 可偏通用 serving pool，SGLang 可偏 PD/EP/长前缀 pool。 |
| 深度自定义调度 | 看改动边界 | vLLM 有 `SchedulerInterface`、custom scheduler class 和相对清晰的 EngineCore contract；SGLang 的 `SchedulePolicy`、`PrefillAdder`、Radix 和 overlap 状态更直接，但修改时必须维护更多 runtime 不变量。 |

进一步的判断维度包括：目标模型是否 MLA/MoE、多卡互联、quant/backend 支持、PD 分离、结构化输出、prefix sharing、运维指标、升级成本和团队对复杂状态机的掌握程度。框架在某个公开 benchmark 上领先，不代表你的长度分布、缓存率和 SLO 下领先。

一个务实架构是统一上层 OpenAI/router/metrics 接口，把不同框架作为可替换后端。先按 workload 路由，再持续用 canary 和 shadow traffic 比较 SLO goodput。这样选型是数据驱动的部署决策，而不是一次不可逆的框架绑定。

### 源码结合

- vLLM 的通用边界见 [`api_server.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/openai/api_server.py)、[`EngineCore`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/core.py) 和 [`SchedulerInterface`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/interface.py)。
- SGLang 的长前缀能力见 [`radix_cache.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py) 与 [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)。
- 两者结构化输出分别见 vLLM [`structured_output`](https://github.com/Cptzzzzz/vllm/tree/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/structured_output) 与 SGLang [`constrained`](https://github.com/Cptzzzzz/sglang/tree/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/constrained)。
- 两者都具有多种 attention、quantization、speculative 和 KV transfer backend，必须检查目标模型的实际代码路径，不能只看功能列表。

### 得分点（10 分）

- 2 分：给出倾向但保留以 workload benchmark 验证的条件。
- 2 分：准确使用 Radix/cache-aware scheduling 解释 Agent 场景。
- 2 分：知道两者都支持结构化输出，不做绝对化宣传。
- 2 分：异构集群优先拆同构 pool，而非跨异构卡做同步 TP。
- 2 分：自定义 Scheduler 能比较接口清晰度与状态机复杂度。

### 常见失分点

一句话宣布某框架全面更快；把功能“支持”当成目标模型上的成熟性能；异构卡直接组成 TP；忽略团队维护和版本升级成本。

---

## 24. 如何在现有源码中增加 deadline-aware scheduling

### 完整答案

这项改动不能只把 priority 改成 deadline。Deadline 会随时间接近，静态 heap key 会过期；Scheduler 还需要估计剩余工作、控制抢占并保持 KV/取消语义。

**数据模型与入口**

API 接收相对 deadline 或绝对时间，并在入口转换为本进程 `monotonic_deadline`。跨机器只传“剩余预算 + ingress timestamp/trace”，避免直接比较不同机器的 monotonic clock。请求结构增加 `deadline_ts`、`original_deadline_ms`、`estimated_remaining_cost`、`last_scheduled_ts` 和 `deadline_class`。

vLLM 的传递链为 OpenAI protocol/serving -> `AsyncLLM`/`InputProcessor` -> `EngineCoreRequest` -> `Request`。SGLang 为 HTTP input -> `GenerateReqInput`/`TokenizedGenerateReqInput` -> `Req`。所有 copy、parallel sampling child、streaming update 和 PD DTO 都必须保留 deadline。

**调度策略**

每轮计算：

```text
remaining_cost =
    estimated_uncached_prefill_ms
  + expected_remaining_decode_tokens * estimated_tpot_ms
  + transfer_or_recompute_ms

slack = deadline_ts - now - remaining_cost
score = (deadline_miss_risk, slack, effective_priority, arrival_time)
```

负 slack 请求要按策略选择 best-effort、立即拒绝或降级，不能继续占用大量资源却注定失败。运行中 Decode 获得最小 token quantum；Prefill 继续 chunk 化。低优无 deadline 请求使用 aging/DRR 保证进展。抢占只在 iteration 边界发生，并对 recompute/transfer 成本设置 hysteresis。

动态 slack 不适合直接复用永不 re-key 的静态 heap。简单可靠的第一版是在每次 admission/schedule 时对 waiting candidates 重新计算 score；规模很大时可使用 deadline bucket/timing wheel，加一个 indexed heap 处理 bucket 内 priority，并定期重算 cost。

**vLLM 修改点**

- [`request.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/request.py)：新增 deadline/cost 状态，但不要把 deadline 放入 prefix hash key。
- [`request_queue.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/request_queue.py)：新增 deadline queue 或让 Scheduler 每轮动态选择，避免当前 `Request.__lt__` 静态排序失效。
- [`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)：token budget、chunked prefill、victim selection、miss policy 和统计。
- [`async_llm.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py) 与 OpenAI protocol：参数传播、前端 timeout/abort 和结果 finish reason。

**SGLang 修改点**

- [`io_struct.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/io_struct.py) 与 [`schedule_batch.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py)：在 DTO/`Req` 保存 deadline。
- [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py)：增加 EDF/slack policy，并与现有 priority、LPM/DFS prefix policy 组合。可以先按 deadline risk 分桶，再在桶内使用 cache-aware ordering。
- [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)：`PrefillAdder` admission、priority preemption、retraction、grammar/PD queue timeout 和 abort。
- [`tokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py)：HTTP deadline、断连和 Scheduler abort 的最终闭环。

**不破坏现有语义的证明思路**

1. Deadline 只改变排序、准入和停止决策，不进入 token hash、Radix `extra_key`、KV layout 或 sampler，因此相同 token 的缓存语义不变。
2. 所有抢占/超时仍调用原有 `finish_requests()`、`abort_request()` 或 `release_kv_cache()`，不创建第二套释放路径。
3. 已提交 GPU batch 的结果可迟到，但 finished/aborted request 不再进入后续 `SchedulerOutput`；迟到结果只完成 fence，不能产生第二次 HTTP finish。
4. Continuous batching 的 token/KV budget 不变量不变：每轮 scheduled token 不超过预算，每个 KV block 只有有效 ref/lock owner。
5. Prefix hit 只降低 `remaining_cost`，不能绕过 block allocation、lock 或 full-sequence admission。

**测试与指标**

单测使用 fake clock 验证 EDF、aging、动态 re-key、同 deadline tie-break、负 slack policy。属性测试验证任意 add/schedule/preempt/abort 序列下 refcount 不为负、同一 block 不重复分配、每请求恰好一次 finish。集成测试覆盖 waiting/running/grammar/overlap/PD transfer 各阶段超时，以及 high/low priority 混合长度负载。

新增指标至少包括 admission slack、first-schedule slack、finish slack、deadline miss ratio、deadline rejection、deadline preemption、recompute tokens、按 class 的 TTFT/TPOT 和 starvation age。只有证明 deadline miss 降低且低优 goodput 有下界，改动才算成功。

### 源码结合

- vLLM 的入口与内部 DTO 链路位于 [`chat_completion/protocol.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/openai/chat_completion/protocol.py)、[`async_llm.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/async_llm.py) 和 [`engine/__init__.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/engine/__init__.py)。
- vLLM 的动态排序与资源不变量落在 [`request_queue.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/request_queue.py)、[`scheduler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py) 和 [`kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py)。
- SGLang 的入口与 Req 状态位于 [`io_struct.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/io_struct.py)、[`tokenizer_manager.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/tokenizer_manager.py) 和 [`schedule_batch.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py)。
- SGLang 的策略、准入、抢占和多队列释放位于 [`schedule_policy.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_policy.py) 与 [`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)。

### 得分点（10 分）

- 2 分：deadline 从 API 到内部 Request/Req 的完整传播设计。
- 2 分：用 remaining cost 与 slack，而不是只按绝对 deadline 排序。
- 2 分：识别动态 key 不能直接依赖静态 heap，并给出实现方案。
- 2 分：改动复用原 KV/abort 释放路径，列出关键不变量。
- 2 分：测试覆盖 fake clock、属性不变量、overlap/PD 和 SLO 指标。

### 常见失分点

仅新增一个 `deadline` 字段并按它排序；使用跨机器 wall clock 却不处理偏差；超时直接删除 Python request 而不释放 KV；为了 cache locality 无限制延误即将 miss deadline 的请求。

---

## 总体评分建议

这 24 题不适合在一次面试中全部提问。推荐按岗位级别抽取：

- 初中级推理工程师：1、5、10、13、15、22。
- 高级推理工程师：6、8、11、12、16、18、20。
- AI Infra Tech Lead：4、9、17、19、23、24。

单题 10 分的总体解释：

| 分数 | 表现 |
| --- | --- |
| 0-2 | 只知道术语，关键机制或方向错误。 |
| 3-4 | 能描述一般原理，但无法对应源码与状态所有权。 |
| 5-6 | 能讲清主流程和常见权衡，缺少失败路径或量化方法。 |
| 7-8 | 能结合源码、资源账本、并发边界和验证指标。 |
| 9-10 | 能推导公式、识别隐藏不变量，并提出可实施且可证伪的设计。 |

## 配套材料

- 环境与版本结论：[`RUNTIME_VALIDATION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/RUNTIME_VALIDATION.md)
- vLLM 架构笔记：[`vllm/ARCHITECTURE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/ARCHITECTURE.md)
- SGLang 架构笔记：[`sglang/ARCHITECTURE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/ARCHITECTURE.md)
- vLLM 实际执行链路：[`FLOW_TRACE.md`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/FLOW_TRACE.md)
- SGLang 实际执行链路：[`FLOW_TRACE.md`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/FLOW_TRACE.md)

完整 Step 日志属于可复现的运行产物，不提交到笔记仓库；使用两个 `FLOW_TRACE.md` 中的命令生成。
