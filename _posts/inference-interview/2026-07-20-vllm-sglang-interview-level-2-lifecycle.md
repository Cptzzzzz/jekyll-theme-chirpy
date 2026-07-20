---
title: 'vLLM / SGLang 面试题（二）：请求生命周期与模块契约'
author: Cptz
date: 2026-07-20 00:12:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)第 2 卷，覆盖第 11-20 题。重点是HTTP 到 GPU 再到流式返回、IPC、状态所有权、取消与异步边界。

[上一篇：推理服务基础](/posts/vllm-sglang-interview-level-1-foundations/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：调度、准入与资源预算](/posts/vllm-sglang-interview-level-3-scheduling/)

---

## Level 2：请求生命周期与模块契约

目标：确认候选人能跟着真实对象和进程边界，从 HTTP 请求走到流式返回与资源释放。

### 11. 分别画出 vLLM 和 SGLang 的一个 Chat Completion 请求完整生命周期。

- **递进追问**：逐段给出 Input、Output、状态所有者、IPC 和取消路径；两者在哪里 tokenization、detokenization、schedule、sample？
- **面试官观察点**：能否覆盖 HTTP -> engine -> scheduler -> worker -> output，而不是只描述 GPU forward。
- **源码锚点**：[`vllm/modules/REQUEST_PIPELINE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/REQUEST_PIPELINE.md)；[`sglang/modules/REQUEST_PIPELINE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/REQUEST_PIPELINE.md)；完整答案指南第 1 题。

**回答思路**：分别画出控制面、设备面和输出面，沿每次对象变形说明 Input/Output，并在图上标出进程、IPC、状态所有者和 abort 回路。

**详细答案**：vLLM 中，OpenAI serving 层验证协议、渲染 chat template并得到 engine input；`AsyncLLM.add_request()` 经 `InputProcessor` 构造 `EngineCoreRequest`，先在 API 进程的 `OutputProcessor` 注册 collector，再通过 `EngineCoreClient` 送入后台。`EngineCore` 将 DTO 转为内部 `Request`，`Scheduler.schedule()` 在 token、sequence、KV等预算下产生 `SchedulerOutput`。Executor/`GPUWorker`/`ModelRunner` 更新 resident state、打包 ragged input、构造 attention metadata，forward并 sample。`Scheduler.update_from_output()` 提交 token、stop和资源状态，`EngineCoreOutputs` 回到 API进程，`OutputProcessor` 增量 detokenize并按 request ID投递 collector，HTTP层生成一次性响应或 SSE。

SGLang 中，HTTP请求进入 `TokenizerManager.generate_request()`，先建立 `ReqState`，完成 tokenization/MM preprocessing后，以 `TokenizedGenerateReqInput` 经 ZMQ 发给 Scheduler。Scheduler构造 `Req`，`SchedulePolicy`排序，`PrefillAdder`准入，形成 EXTEND/DECODE `ScheduleBatch`；`TpModelWorker` 转成 `ForwardBatch`，`ModelRunner` forward/sample。Scheduler提交 `Req` 和 KV/Radix状态，将 token IDs发给独立 `DetokenizerManager`；文本 delta再经 ZMQ回到 `TokenizerManager.handle_loop()`，由 `rid_to_state` 唤醒对应 HTTP coroutine。两边取消都只能阻止后续调度并释放 ownership，通常不能撤销已 launch kernel。

**主问题得分点（10 分）**：

- 2 分：两边都覆盖 HTTP、tokenize、schedule、GPU、sample、detokenize和返回。
- 2 分：准确指出主要进程边界和 IPC。
- 2 分：说出关键对象变形与 request ID关联。
- 2 分：说明 Scheduler拥有 KV和 commit/finish状态。
- 2 分：覆盖 streaming、abort和迟到结果。

**追问与答案（5 分）**：两框架最显著的输出拓扑差异是什么？vLLM 当前主路径在 API进程的 `OutputProcessor` 做增量输出语义；SGLang显式设置 Scheduler -> Detokenizer -> TokenizerManager进程链。前者减少一段 IPC并便于聚合 API语义，后者隔离 tokenizer CPU/GIL，但增加消息关联和部署状态。得分点：拓扑 2 分，ownership 1 分，权衡 2 分。

### 12. vLLM 的外部请求在进入 GPU 前经历了哪些对象变形？

- **递进追问**：`ChatCompletionRequest`、`EngineInput`、`EngineCoreRequest`、内部 `Request` 各自应该拥有和隐藏哪些字段？为什么不让 Scheduler 直接依赖 HTTP DTO？
- **面试官观察点**：是否理解边界 DTO、验证责任和稳定 core contract。
- **源码锚点**：`OpenAIServingChat._create_chat_completion()`、`InputProcessor.process_inputs()`、`Request.from_engine_core_request()`。

**回答思路**：不要只列类名，要对每个对象回答“属于哪一层、已经完成哪些验证、允许下游依赖哪些字段”。

**详细答案**：`ChatCompletionRequest` 是外部协议对象，包含 messages、tools、stream、OpenAI兼容采样字段等。Serving层渲染模板、处理多模态和工具语义，得到规范化 `EngineInput` 与 `SamplingParams`。`InputProcessor.process_inputs()` 验证 task/长度/模型能力，将不稳定输入形态收敛为 core可消费字段，构造 transport友好的 `EngineCoreRequest`，其中包含 token IDs、sampling/pooling、LoRA、priority、MM references等，但不包含 Scheduler运行态。

后台 `EngineCore.preprocess_add_request()` 通过 `Request.from_engine_core_request()` 建立长生命周期内部 `Request`，生成 block hashes、token counters、status以及 grammar/MM receiver等 core state，再交给 Scheduler。这样 API schema变化不会传播到核心调度，IPC contract可测试，Scheduler也不需要理解 messages、HTTP或chat template。

**主问题得分点（10 分）**：

- 3 分：四类对象的层次与主要字段正确。
- 2 分：明确 validation/normalization责任逐层收敛。
- 2 分：区分 transport DTO与 Scheduler运行态。
- 2 分：说明解耦 API演进、IPC和 core测试的价值。
- 1 分：指出 block hash/counters在内部 `Request`建立。

**追问与答案（5 分）**：为什么不直接把 Python tokenizer对象或完整 MM tensor塞进 EngineCoreRequest？它们可能不可序列化、生命周期不清、跨进程复制昂贵，也把 API/runtime实现泄漏到 core。应传稳定 IDs、metadata或有显式 ownership的 shared-memory/receiver handle。得分点：序列化 1 分，复制/lifetime 2 分，稳定 contract 2 分。

### 13. vLLM 为什么要先在 OutputProcessor 注册请求，再把请求发给 EngineCore？

- **递进追问**：如果顺序反过来，会出现什么竞态？`RequestOutputCollector` 如何处理生产者快于消费者？失败或取消时两端状态如何清理？
- **面试官观察点**：能否识别“极快结果先到但没有 consumer”的并发 bug，并讨论有界缓冲与 delta 合并。
- **源码锚点**：`AsyncLLM._add_request()`、`OutputProcessor.add_request()`、`RequestOutputCollector`。

**回答思路**：把提交请求与接收结果看成两个并发 actor，构造“core极快返回”的最坏时序即可解释注册顺序。

**详细答案**：`AsyncLLM._add_request()` 先调用 `OutputProcessor.add_request()` 建立 request ID到 state/collector/增量 detokenizer的映射，然后才经 client向 EngineCore enqueue。若顺序反过来，小模型、cache hit或同进程模式下 core可能在 API侧映射创建前返回，output handler找不到消费者，导致丢结果、unknown request或请求永久等待。

collector还解决生产和消费速度不一致：对 DELTA输出可合并相邻更新，避免每 token无限增加 asyncio queue item；最终状态必须保留。enqueue失败时要删除已注册 state，取消时先使 API消费者终止，再向 core传播 abort，并让迟到 output按“已不存在/已终止 request”安全丢弃。这里的关键不变量是同一个内部 ID在任何可见结果到达前已有唯一 owner。

**主问题得分点（10 分）**：

- 3 分：能画出 enqueue-first导致的具体竞态。
- 2 分：说明注册内容包括 collector、detokenizer和 ID路由。
- 2 分：说明 delta合并/有界内存目的。
- 2 分：覆盖 enqueue失败、cancel和迟到结果清理。
- 1 分：提出并发测试方法。

**追问与答案（5 分）**：如何写测试稳定复现这个竞态？使用 fake EngineCoreClient，在 `add_request` 内同步或立即回调 output handler，再断言 collector已存在；另注入 enqueue异常和 cancel/late-output交错，检查无泄漏且 future终止。得分点：可控同步回调 2 分，异常/取消场景 2 分，资源断言 1 分。

### 14. `SamplingParams.n > 1` 在 vLLM 中在哪里 fan-out，结果又如何聚合？

- **递进追问**：外部 request ID、parent request 和 child request 的映射如何维护？单个 child 提前结束或 abort 会怎样？为什么这个逻辑不应进入 Scheduler？
- **面试官观察点**：是否能区分 API 语义的逻辑并行与 core 中独立请求的资源所有权。
- **源码锚点**：`AsyncLLM.add_request()`、`ParentRequest`、`OutputProcessor.process_outputs()`。

**回答思路**：区分外部“一次请求要 n 个候选”的 API语义与 core视角的 n 个独立序列，再追踪 parent/child ID和资源结束条件。

**详细答案**：`SamplingParams.n > 1` 时，`AsyncLLM.add_request()` 在 API/engine边界建立 `ParentRequest`，为每个 child生成唯一内部 request ID，并把每个 child作为普通 core request提交。Scheduler只看到独立序列，因此每个 child可独立分配 KV、被 continuous batch调度和完成，不需要理解 OpenAI choices聚合。

`OutputProcessor` 保存 parent到children映射，处理每个 child的 token、logprob、finish，按 output kind决定返回增量或聚合结果；只有满足协议所需的 child状态后 parent才完成。取消 parent必须传播到所有仍活跃 children，单个 child提前 EOS则只释放其 core资源。将该逻辑留在输出/API层可避免 Scheduler同时承担协议排序、choice index和 best-of语义。

**主问题得分点（10 分）**：

- 2 分：指出 fan-out发生在 `AsyncLLM.add_request()`附近而非 Scheduler。
- 2 分：每个 child有唯一内部 ID和独立 KV/调度状态。
- 2 分：`ParentRequest`/`OutputProcessor`负责 choices聚合。
- 2 分：覆盖 child提前结束与 parent cancel。
- 2 分：解释为什么属于 API语义边界。

**追问与答案（5 分）**：`n=8` 是否等价于 batch size增加 8？不完全等价。它增加8个逻辑 sequences和KV增长，可共享 prompt prefix但各自采样/输出独立；受 `max_num_seqs`、token budget、KV容量和sampling开销共同约束，未必同轮执行。得分点：sequence/KV 2 分，prefix sharing 1 分，多资源约束 2 分。

### 15. 客户端断开后，框架能不能立即停止已经 launch 的 GPU kernel？

- **递进追问**：API state、waiting/running request、已提交 iteration、异步 D2H 和 KV block 分别何时释放？迟到结果怎样丢弃？
- **面试官观察点**：是否理解 cancel 是跨进程协议，不等于撤销设备工作；能否避免 use-after-free 和 double-free。
- **源码锚点**：vLLM `AsyncLLM.generate()/abort()`；SGLang `_wait_one_response()`、`AbortReq`、Scheduler abort handler。

**回答思路**：按 API、Scheduler、GPU stream、output四个时间域回答，明确取消能改变的最早边界和资源的 last-use event。

**详细答案**：通常不能立即停止已 launch kernel。客户端断开首先取消 HTTP coroutine/async generator，API层移除或标记 per-request state并发送 abort消息。Scheduler收到后可从 waiting queue删除请求；对 running请求，阻止其进入下一次 `SchedulerOutput`并在设备不再引用后释放 KV、encoder和其他 ownership。

已经提交到 CUDA stream的 forward/sample/D2H通常继续执行。其 buffers和KV不能在 kernel或copy event完成前复用，否则发生 use-after-free；结果回来后依据 aborted/unknown ID丢弃，不再写入客户端 state。SGLang还需处理 grammar、chunked、PD transfer和overlap result queue，vLLM需让 core与 API两个状态表最终收敛。正确目标是“停止未来工作并及时回收”，不是假设 Python cancel能撤回设备命令。

**主问题得分点（10 分）**：

- 2 分：明确不能普遍撤销已 launch GPU工作。
- 2 分：区分 waiting、running、in-flight处理。
- 2 分：说明 KV/tensor释放要等 last device use。
- 2 分：说明迟到结果按 ID/状态丢弃。
- 2 分：覆盖跨进程 abort最终一致与泄漏监控。

**追问与答案（5 分）**：为什么收到 abort后立刻把 block放回 free pool很危险？旧 kernel可能仍通过 block table读写它，新请求若复用同一 block会发生跨请求数据竞争和KV污染。必须以调度/stream event确认最后使用结束。得分点：指出 use-after-free 2 分，描述复用后果 2 分，提出 event/iteration安全边界 1 分。

### 16. SGLang 请求从 `GenerateReqInput` 到 `Req` 经历了什么？

- **递进追问**：tokenization、多模态 feature、shared memory ownership、ZMQ DTO、sampling/priority/session/PD metadata 分别在哪一层建立？
- **面试官观察点**：能否说清 API 进程与 Scheduler 进程之间的 transport contract。
- **源码锚点**：`TokenizerManager._tokenize_one_request()`、`_send_one_request()`、`Scheduler.handle_generate_request()`。

**回答思路**：沿 TokenizerManager与Scheduler的ZMQ边界，把外部语义、传输字段和执行状态分开。

**详细答案**：HTTP/OpenAI adapter先把外部schema归一为 native `GenerateReqInput`。`TokenizerManager.generate_request()`建立 rid和等待状态，`_tokenize_one_request()`接受 text、input IDs或embeddings，完成tokenization、长度校验、多模态feature/placeholder处理与sampling参数规范化。`_send_one_request()`把大feature按配置包装为shared-memory handle，附加LoRA、priority、session、grammar和P/D metadata，形成可传输 `TokenizedGenerateReqInput`，经ZMQ发给目标Scheduler。

`Scheduler.handle_generate_request()`才构造内部 `Req`，初始化origin/input IDs、output、prefix/cache fields、allocated/committed counters、sampling和状态，将其放入waiting queue。物理request row、KV slots和Radix lock要到prefix match/admission阶段才获得。该分层避免Tokenizer进程修改Scheduler-owned cache state，也使transport DTO可重试和观测。

**主问题得分点（10 分）**：

- 2 分：外部请求先归一为 `GenerateReqInput`。
- 2 分：tokenization/MM和校验在TokenizerManager完成。
- 2 分：说明shared-memory feature的显式ownership。
- 2 分：列出transport DTO中的关键metadata。
- 2 分：说明内部 `Req`与物理资源到Scheduler才建立。

**追问与答案（5 分）**：shared-memory MM feature何时可以释放？必须有明确producer/consumer协议：Scheduler/worker完成映射或拷贝并确认后才由owner unlink；abort、send失败和进程崩溃需有回收路径。只在ZMQ send返回后释放不安全，因为send完成不等于consumer已读取。得分点：区分send与consume 2 分，正常ack 1 分，异常回收 2 分。

### 17. SGLang 为什么在发请求前先建立 `rid_to_state`？

- **递进追问**：输出乱序时怎样唤醒正确 coroutine？batch request、streaming、超时和 abort 如何改变 state 生命周期？
- **面试官观察点**：能否把稳定 request ID、event 和输出路由理解为异步正确性的基础。
- **源码锚点**：`TokenizerManager.generate_request()`、`_init_req_state()`、`handle_loop()`。

**回答思路**：与vLLM注册竞态相同，构造Scheduler极快返回且多个请求乱序到达的时序，说明rid是跨进程join key。

**详细答案**：`_init_req_state()` 在dispatch前为rid建立event、output buffers、stream位置、tokenizer/grammar相关状态。Scheduler和Detokenizer是独立进程，输出会按batch和运行时间乱序返回；`TokenizerManager.handle_loop()`只能依靠rid查找正确 `ReqState`，写入delta并set event唤醒 `_wait_one_response()`。

如果先dispatch再注册，极快响应可能成为orphan。batch请求需要每个rid独立state并在外层聚合；streaming state会经历多次event清除/唤醒；timeout/断连发送abort，但state不能在可能仍需识别迟到消息前以不安全方式复用rid。完成后删除映射并记录终态，防止内存泄漏。

**主问题得分点（10 分）**：

- 3 分：解释dispatch前注册防止orphan response。
- 2 分：说明rid用于乱序跨进程输出关联。
- 2 分：覆盖event的streaming多次唤醒。
- 2 分：覆盖batch、timeout/abort和迟到结果。
- 1 分：说明ID不可过早复用/状态需清理。

**追问与答案（5 分）**：怎样处理同一rid重复或重放请求？入口必须保证rid唯一或实现幂等表；活跃rid冲突应拒绝，已完成rid若允许重放需返回缓存结果或生成新内部rid，不能覆盖旧state。得分点：活跃冲突 2 分，幂等/新ID策略 2 分，禁止覆盖 1 分。

### 18. SGLang 为什么把 Detokenizer 做成独立进程，而 vLLM 当前主路径放在 API 侧 OutputProcessor？

- **递进追问**：两种拓扑对 GIL、故障隔离、IPC、文本状态 ownership、stream latency 和部署复杂度有什么影响？
- **面试官观察点**：是否能基于 workload 做权衡，而不是简单判定哪种设计更先进。
- **源码锚点**：SGLang `DetokenizerManager.event_loop()`；vLLM `OutputProcessor.process_outputs()`。

**回答思路**：从CPU隔离与并行度、IPC成本、状态归属、故障域四个维度比较，不把拓扑选择等同于算法优劣。

**详细答案**：SGLang把Detokenizer设为独立进程，Scheduler只发送token batch，Detokenizer维护每个rid的decode offset/state并输出文本，再交给TokenizerManager。这可以把CPU tokenizer、Unicode/string和GIL抖动隔离出GPU调度进程，多个组件可独立profile；代价是额外ZMQ序列化、消息复制/排队、进程健康管理，以及跨两段IPC维持rid、finish和abort一致性。

vLLM当前主路径由API进程的 `OutputProcessor` 批量接收core outputs并增量detokenize、聚合parallel children和投递collector，少一段专用输出IPC，API语义集中；但高输出速率、复杂tokenizer或大量连接会和web event loop竞争CPU，需要批量处理和delta合并。选择取决于CPU核数、tokenizer成本、消息速率、故障隔离和部署复杂度，应以event-loop lag、ITL和IPC占比验证。

**主问题得分点（10 分）**：

- 2 分：准确描述两种输出拓扑。
- 2 分：说明SGLang的CPU/GIL与故障隔离收益。
- 2 分：说明额外IPC和状态一致性代价。
- 2 分：说明vLLM集中API语义的收益与event loop风险。
- 2 分：给出基于workload的指标和选择方法。

**追问与答案（5 分）**：什么时候独立Detokenizer反而变慢？tokenizer很轻、输出batch很小、IPC/调度延迟占主导，或进程CPU绑核/NUMA位置不佳时，额外hop会提高ITL。得分点：轻workload 1 分，IPC小消息 2 分，CPU/NUMA与实测 2 分。

### 19. SGLang 的 `Req`、`ScheduleBatch` 和 `ForwardBatch` 分别解决什么问题？

- **递进追问**：哪个对象是长生命周期逻辑状态，哪个冻结一轮资源计划，哪个是模型 tensor contract？为什么不能合成一个“大对象”？
- **面试官观察点**：能否明确 policy state、iteration plan 和 rank-local execution metadata 的边界。
- **源码锚点**：`managers/schedule_batch.py`、`model_executor/forward_batch_info.py`；[`sglang/modules/MODEL_EXECUTION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/MODEL_EXECUTION.md)。

**回答思路**：以生命周期和owner分类：请求级真值、iteration级资源计划、一次forward的tensor contract。

**详细答案**：`Req` 是跨多轮存在的逻辑请求状态，持有input/output IDs、sampling、prefix match、request row、KV committed/allocated lengths、finish和timing等。`ScheduleBatch` 是Scheduler/worker边界的一轮计划：冻结本轮requests、EXTEND/DECODE mode、seq/prefix/extend lengths、pool indices、sampling/spec info，并在prepare阶段获得本轮物理资源。

`ForwardBatch.init_new()` 将 `ScheduleBatch` 转成rank-local模型contract，包括device tensors、positions、attention metadata、forward mode、DP/TP信息和one-shot overrides，供 `ModelRunner.forward()` 消费。三者分开使policy/cache逻辑不依赖具体kernel tensor layout，也防止worker修改长生命周期Scheduler state；代价是必须维护字段一致性和snapshot lifetime。

**主问题得分点（10 分）**：

- 3 分：三类对象的生命周期和owner准确。
- 2 分：给出各自关键Input/Output字段。
- 2 分：说明prepare阶段冻结物理资源。
- 2 分：说明解耦policy与tensor/backend的价值。
- 1 分：指出转换一致性/snapshot代价。

**追问与答案（5 分）**：为什么overlap模式不能把同一个可变 `ScheduleBatch` 引用直接放进result queue？Scheduler会继续filter、merge或改sampling字段，GPU结果回来时引用已不代表launch时request order，造成错配。必须copy/snapshot并保活其tensor。得分点：可变状态 2 分，错配后果 2 分，snapshot方案 1 分。

### 20. 慢流式客户端会怎样反向影响推理服务？

- **递进追问**：每 token 一个无界 queue item 会发生什么？输出合并、有界队列、暂停准入或取消请求分别损失什么语义？如何给 backpressure 打指标？
- **面试官观察点**：是否能跨越 GPU、engine IPC、event loop 和 socket 四层分析内存增长与尾延迟。
- **源码锚点**：vLLM `RequestOutputCollector`；SGLang `ReqState`/TokenizerManager output loop；完整答案指南第 4 题。

**回答思路**：沿“GPU生产 token -> engine输出 -> API queue -> socket发送”找出每层buffer和速率不匹配，最后给出有界策略与语义取舍。

**详细答案**：慢客户端不会立刻让GPU变慢。如果每个token在API queue中形成独立对象且无界积累，首先表现为API进程RSS、GC和event-loop lag增长；IPC接收若被阻塞，core输出buffer继续增长，最终才可能反向影响engine step或全局服务。一个慢连接因此可能拖累其他请求的TTFT/ITL。

vLLM collector可合并DELTA，降低queue item数量但不能消除文本本身的未发送字节；SGLang也需管理rid state与ZMQ水位。生产方案应限制每请求和全局pending bytes/items，记录consumer lag、oldest output age、merge ratio和socket blocked time。达到阈值可合并delta、暂停该请求后续调度、超时取消或断开；不能静默丢token，否则破坏stream语义。backpressure必须传播到请求粒度，避免阻塞整个output handler。

**主问题得分点（10 分）**：

- 2 分：画出四层生产/消费链和buffer。
- 2 分：解释无界queue导致RSS/GC/event-loop问题。
- 2 分：说明delta合并的收益和边界。
- 2 分：给出有界水位及request级降级策略。
- 2 分：给出consumer lag/pending bytes等指标并保护其他租户。

**追问与答案（5 分）**：暂停一个慢请求的GPU调度是否免费？不是。它仍占有KV，延长生命周期并降低可并发容量；若释放后重算又增加计算。应在短暂停、取消和preempt之间按KV占用、预计恢复和SLO决策。得分点：KV占用 2 分，重算权衡 1 分，策略与指标 2 分。

---

[上一篇：推理服务基础](/posts/vllm-sglang-interview-level-1-foundations/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：调度、准入与资源预算](/posts/vllm-sglang-interview-level-3-scheduling/)
