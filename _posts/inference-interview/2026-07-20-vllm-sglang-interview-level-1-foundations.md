---
title: 'vLLM / SGLang 面试题（一）：推理服务基础'
author: Cptz
date: 2026-07-20 00:11:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)第 1 卷，覆盖第 1-10 题。重点是Prefill/Decode、在线指标、批处理、采样、并行与基础容量分析。

[系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：请求生命周期与模块契约](/posts/vllm-sglang-interview-level-2-lifecycle/)

---

## Level 1：推理服务基础

目标：确认候选人理解在线自回归推理的成本、指标和基本资源，而不是只会调用 API。

### 1. Prefill 和 Decode 分别在做什么，为什么它们的瓶颈通常不同？

- **递进追问**：长 prompt 与大 batch 分别如何改变算术强度？为什么不能简单说“Prefill 一定 compute-bound，Decode 一定 memory-bound”？混合批次会怎样影响判断？
- **面试官观察点**：能否从 token 依赖、矩阵形状、权重/KV 访存和硬件利用率建立因果链。
- **源码锚点**：vLLM `Scheduler.schedule()` 的 `num_scheduled_tokens`、FlashAttention/Triton backend 的 `causal` 参数；SGLang `ForwardMode.EXTEND/DECODE` 与 EXTEND attention backend；两边 `ModelRunner.forward()`。

**回答思路**：先从自回归依赖解释两个阶段能并行的 token 数，再用算子形状和 Roofline 模型判断瓶颈，最后补充动态 batch 和长上下文带来的例外。

**详细答案**：Prefill 接收整个 prompt，一层内可并行处理多个 query token，线性层通常形成 `tokens x hidden` 的较大 GEMM，容易提高 Tensor Core 利用率；同时要为所有 prompt positions 写 KV。Decode 每轮每个请求只产生一个新的 query position，输出 token 之间严格串行，每轮既要读取模型权重，也要读取该请求历史 KV，矩阵较“瘦”，因此小 batch 下常受 HBM 带宽、kernel launch 和同步开销限制。

Decoder-only Prefill 确实使用 causal mask。对从零开始、长度为 `N` 的完整 prompt，第 `i` 个 query 只看前 `i` 个 key，有效 Q-K pairs 为 `N(N+1)/2`，约是无因果 `N^2` 的一半；但渐进复杂度仍是 `Theta(N^2)`。支持 causal tiling 的 FlashAttention/Triton kernel 可以跳过完全位于上三角的 tiles，naive `QK^T` 后再 mask 的实现则可能仍做接近 `N^2` 的乘法。若已有 cached prefix `P`，本轮 EXTEND `Q` 个 tokens，有效 pairs 是 `P*Q + Q(Q+1)/2`；当 `P >> Q` 时，读取旧 prefix 的 `P*Q` 占主导，不能再说节省一半。这一结论还只针对 attention 的 QK/PV 部分，QKV/O projection 和 MLP 仍主要按 token 数线性增长。

这不是绝对分类。很短的 prefill 可能仍被 launch/CPU 开销主导；长上下文 prefill 的 attention 也可能受内存与二次增长的 attention work 限制；decode batch 足够大时，权重可被更多 token 摊薄，GEMM 会逐渐接近 compute-bound。vLLM 用每请求 `num_scheduled_tokens` 把不同长度工作打包，SGLang 用 EXTEND/DECODE mode 构造不同 metadata；真实瓶颈取决于本轮 token shape、上下文长度、量化、attention backend 和硬件。

**主问题得分点（10 分）**：

- 2 分：说明 Prefill 可并行处理 prompt，Decode 在输出维度串行。
- 2 分：给出 causal attention 的三角 pair 公式，并说明系数减半但仍是二次复杂度。
- 2 分：指出长上下文、小 prefill、大 decode batch 等反例。
- 2 分：能映射到 `num_scheduled_tokens` 或 EXTEND/DECODE 源码对象。
- 2 分：提出用算术强度、HBM 吞吐、SM 利用率和 batch sweep 验证。

**追问与答案（5 分）**：如何证明某个线上 Decode workload 真的是 memory-bound？不能只看 GPU utilization。应同时采集 kernel timeline、DRAM throughput、Tensor Core/SM active、GEMM shape、launch gap，并改变 batch size或模型量化做对照。如果 batch 增大后每 token 权重读取被摊薄、tokens/s 上升且 HBM 接近平台上限，而算力仍远低于峰值，证据才较完整。得分点：指标选择 2 分，对照实验 2 分，避免只凭单一 utilization 下结论 1 分。

### 2. TTFT、ITL、TPOT、E2E latency、throughput 和 goodput 有什么区别？

- **递进追问**：流式接口怎样打点？排队时间属于哪个指标？为什么平均 tokens/s 提升可能伴随 p99 SLO 退化？
- **面试官观察点**：能否将用户体验、调度阶段和服务容量对应起来，并明确 percentile 与 workload。
- **源码锚点**：vLLM `RequestState.stats`/`EngineCoreEvent`；SGLang `Req` timing fields、scheduler metrics；[`VLLM_VS_SGLANG.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/VLLM_VS_SGLANG.md)。

**回答思路**：先统一时间戳定义，再区分单请求体验、单 token 节奏和系统容量，最后强调分位数必须绑定 workload 与 SLO。

**详细答案**：TTFT 是请求到达服务端到首个可见输出 token/delta 的时间，包含排队、tokenization、prefix lookup、prefill、首次 sample 和输出链路。ITL 是相邻可见 token 之间的间隔，应报告分布而非只报均值。TPOT 常定义为首 token 后生成阶段耗时除以后续 token 数，即 `(finish-first)/(output_tokens-1)`；如果实现采用其他口径必须明确。E2E latency 是请求到完整响应结束的总时间。

Throughput 是单位时间完成的 request 或 token 数，但必须注明是 input、output 还是 total token。Goodput 是满足 TTFT/ITL/E2E 等 SLO 的有效吞吐，更能反映在线服务容量。平均 tokens/s 可以通过塞入大 batch 提高，却同时延长 waiting queue、Prefill blocking 和每轮 latency，使 p99 TTFT/ITL 变差。因此比较必须固定 prompt/output 分布、arrival process、并发和错误率。

**主问题得分点（10 分）**：

- 2 分：准确定义 TTFT、ITL/TPOT 和 E2E。
- 2 分：区分 request throughput、input/output/total token throughput。
- 2 分：说明 goodput 必须绑定 SLO。
- 2 分：指出排队属于 TTFT/E2E，并解释吞吐与尾延迟冲突。
- 2 分：给出服务端单调时钟、流式 flush 和 workload 口径。

**追问与答案（5 分）**：流式接口应在哪里打点？服务端至少记录 admission、schedule、first GPU result、first detokenized delta、socket write/flush、finish；客户端记录 request send、first byte/semantic token、每个 delta 和结束。两端时间不可直接相减，除非校时；服务端 TTFT 与客户端 TTFT 的差可用于估计网络/代理影响。得分点：阶段时间戳 2 分，区分 first GPU token 与 first visible token 1 分，时钟/flush 边界 2 分。

### 3. Continuous batching 相比静态 batching 解决了什么问题？

- **递进追问**：为什么请求长度不齐会产生浪费？iteration-level batching 的准入单位是 request 还是 token？它给 KV 管理和输出关联增加了什么复杂度？
- **面试官观察点**：能否讲清“每轮重组执行计划”，而不是笼统地说动态加请求。
- **源码锚点**：vLLM `SchedulerOutput`；SGLang `Scheduler.get_next_batch_to_run()` 与 `ScheduleBatch`。

**回答思路**：用长度不齐的固定 batch 说明空洞，再描述每个 iteration 的 admission、finish 和重新打包，最后说明代价转移到 Scheduler、KV 与输出关联。

**详细答案**：静态 batch 在一批请求形成后通常不能及时补入新请求。短请求完成后，剩余请求继续 Decode，空出的 batch slot 被浪费；若用 padding 保持固定 shape，还会执行无效工作。Continuous batching 在每个或若干模型 iteration 重新选择 active requests：完成请求退出，waiting request 在资源允许时进入，同一轮可包含多个请求且每个请求推进的 token 数不同。

因此核心准入单位既不是纯 request，也不是任意 token：Scheduler 用 token budget限制本轮计算量，同时受 sequence count、KV slots、encoder 和其他资源约束。收益是减少 batch 空洞、提高 GPU busy time 和 goodput；代价是必须维护 request 到 block table/slot 的动态映射、严格配对每轮结果、处理 preemption/abort，并把乱序批量输出还原到单请求 stream。

**主问题得分点（10 分）**：

- 2 分：说明静态 batch 的长度不齐与 slot 空洞。
- 2 分：说明 active set 在 iteration 边界动态变化。
- 2 分：同时提到 token budget 与 request/KV 等约束。
- 2 分：映射到 `SchedulerOutput` 或 `ScheduleBatch`。
- 2 分：讲清 KV ownership、结果关联和调度 CPU 等新增复杂度。

**追问与答案（5 分）**：什么情况下静态 batching 仍可能合适？离线吞吐任务、输入输出长度相近、无流式 SLO、shape 固定且 graph/compiler 对固定 batch收益很大时，静态 batch 简化状态并可能获得更稳定 kernel。得分点：给出 workload 前提 2 分，说明 graph/编译收益 1 分，明确不适合低延迟动态流量 2 分。

### 4. 请推导一个 Transformer 请求每 token 的 KV Cache 显存占用。

- **递进追问**：MHA、GQA、MQA、MLA 分别改变公式中的哪一项？TP 是否一定按比例降低每卡 KV？block/page 尾部碎片怎样计入容量规划？
- **面试官观察点**：是否能从层数、KV heads、head dimension、K/V 两份和 dtype 推导，而不是背模型容量表。
- **源码锚点**：vLLM `KVCacheConfig/KVCacheSpec`；SGLang `TokenToKVPoolAllocator` 与各 attention KV pool。

**回答思路**：先写单层单 token 的 K/V 元素数，再乘层数、dtype、序列和并发，最后加入 TP layout 与 page 分配修正项。

**详细答案**：对标准 attention，单个 token 的逻辑 KV 字节数近似为：

```text
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

前面的 `2` 表示 K 和 V。一个请求再乘其可缓存 token 数，批量容量为所有请求之和。例：32 层、8 个 KV heads、head dimension 128、BF16 2 bytes，单 token 是 `2*32*8*128*2 = 131072` bytes，即 128 KiB；8192 tokens 约 1 GiB，不含 page 尾部、allocator metadata、临时 lookahead 和其他状态。

MHA 的 KV heads 通常等于 query heads；GQA 减少 KV heads；MQA 可降到 1；MLA 保存压缩 latent/rope 等模型特定状态，不能生搬公式。TP 若按 KV heads切分，单卡可近似除以 TP degree，但 KV replication、head 数不可整除、不同 backend 和 MLA layout 会改变结论。物理规划还要向 block/page size取整，并预留 speculative、fragmentation 和安全水位。

**主问题得分点（10 分）**：

- 3 分：公式包含 K/V、层数、KV heads、head dimension、dtype。
- 2 分：能正确做一个数值例子。
- 2 分：区分 MHA/GQA/MQA/MLA。
- 1 分：说明 TP 是否切分取决于实际 layout。
- 2 分：加入 page 对齐、lookahead 和 allocator headroom。

**追问与答案（5 分）**：为什么不能直接用“显存除以公式”得到最大并发？还要扣除模型权重、CUDA graph capture、workspace、activations、通信 buffer、draft model，并考虑每请求长度分布与 page 向上取整；此外框架通常设置利用率水位以避免运行时峰值 OOM。得分点：非 KV 显存 2 分，长度/碎片 1 分，临时峰值和安全水位 2 分。

### 5. 为什么 vLLM、SGLang 都不为每个请求分配一段连续且固定大小的 KV？

- **递进追问**：外部碎片、内部碎片和 over-allocation 分别是什么？block size 变大或变小的收益与代价是什么？
- **面试官观察点**：是否理解逻辑序列到物理 slot 的间接映射，以及它优化的是容量/复用而非 attention 理论复杂度。
- **源码锚点**：[`vllm/modules/KV_CACHE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/KV_CACHE.md)；[`sglang/modules/RADIX_CACHE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/RADIX_CACHE.md)。

**回答思路**：比较“按最大长度连续预留”和“按需分页”的物理占用，再分别分析外部碎片、尾部碎片与间接寻址成本。

**详细答案**：请求长度和存活时间不可预知。如果每个请求按 `max_model_len` 连续预留，会产生巨大的 over-allocation；若按当前长度连续扩容，又需要搬迁或寻找更大连续区间，产生外部碎片。Paged KV 把逻辑 token 区间映射到固定大小物理 block/slot，请求增长时按需追加任意空闲 block，结束后独立回收，因此不要求每个序列物理连续，也便于多个请求共享 prefix blocks。

代价是 block table/slot mapping 查找、allocator 和 metadata CPU 开销，以及最后一个未满 block 的内部碎片。block 大时 table 短、分配和 kernel metadata 较简单，但尾部浪费与 prefix 对齐损失更大；block 小时复用粒度细、尾部浪费少，但 table、hash、allocation 和 kernel indirection 增加。最终要按长度分布、cache reuse 和 backend 做实验选择。

**主问题得分点（10 分）**：

- 2 分：解释最大长度预留造成 over-allocation。
- 2 分：区分连续扩容的外部碎片和分页尾部内部碎片。
- 2 分：说明逻辑到物理的 block/slot 映射。
- 2 分：说明 prefix sharing 与独立回收收益。
- 2 分：完整给出 block size 双向权衡。

**追问与答案（5 分）**：Paged KV 会降低 attention 的理论复杂度吗？不会。它改变存储布局、容量利用和复用方式，kernel 仍需读取语义上可见的 KV；甚至间接寻址有额外开销。它可能通过容纳更大 batch 间接提高吞吐。得分点：明确不改变理论 work 2 分，指出 indirection 1 分，说明容量带来的间接性能 2 分。

### 6. Prefix Cache 能跳过哪些计算，不能跳过哪些计算？

- **递进追问**：为什么完整 prompt 命中时仍常保留一个 forward 边界？cache key 除 token IDs 外还应包含什么？命中率高为何不一定显著降低 TPOT？
- **面试官观察点**：是否区分 prefill 复用、KV 容量共享和 decode 性能，并能说明错误共享会直接破坏语义。
- **源码锚点**：vLLM `KVCacheManager.get_computed_blocks()`；SGLang `RadixCache.match_prefix()`。

**回答思路**：把命中的 committed prefix 看作已计算模型状态，明确它只能替代对应 Prefill work，然后检查产生 next-token logits 的边界和 cache identity。

**详细答案**：Prefix Cache 保存共享 token prefix 已经产生的 KV。新请求命中后，Scheduler 可把这些位置计入已计算状态，只为未命中的 suffix 分配和执行 EXTEND/Prefill，从而减少 Prefill FLOPs、KV 写入和 TTFT；物理 blocks/slots 还可通过引用共享节省容量。它不能跳过后续 Decode，也不能复用 sampling RNG、stop state 或最终文本。

即使 prompt 全部命中，框架仍需要获得“基于完整上下文的 next-token logits”。当前 vLLM 会把安全 hit 限制在 `num_tokens-1` 等边界，确保有一次有效 forward 产生 logits。cache identity 必须覆盖完整链式 token prefix，以及影响 KV 数值的 cache salt、模型/adapter/LoRA、多模态输入等状态。只有已提交且内容稳定的 KV 能进入共享 cache，未验证 speculative token 不能发布。

**主问题得分点（10 分）**：

- 2 分：说明跳过的是命中 prefix 的 Prefill 计算。
- 2 分：说明物理 KV sharing 与 TTFT/容量收益。
- 2 分：说明 next-token logits 的最后 forward 边界。
- 2 分：cache key 包含 token chain 与所有模型状态命名空间。
- 2 分：指出 Decode/采样不被复用及 speculative committed 边界。

**追问与答案（5 分）**：为什么 cache hit rate 很高但端到端延迟几乎不变？可能命中的是短 prompt 或本来 Prefill 占比很小；block/page 对齐后实际跳过 token 少；服务受排队、Decode、tokenizer、网络或 cache lookup CPU 限制；也可能 cache hit 指标按逻辑 token计数却发生 remote load。得分点：区分指标与实际 skipped work 2 分，列出非 Prefill 瓶颈 2 分，提出分阶段 trace 1 分。

### 7. 为什么在线服务通常需要增量 detokenization，而不能对全部 output IDs 每次重新 decode？

- **递进追问**：UTF-8 边界、byte fallback、stop string 跨 token、stream delta 和 logprob 对齐如何处理？
- **面试官观察点**：能否意识到 tokenizer state 也是有生命周期的 per-request state，并会影响 API CPU 与 ITL。
- **源码锚点**：vLLM `OutputProcessor`；SGLang `DetokenizerManager.handle_batch_token_id_out()`。

**回答思路**：先指出全量重复 decode 的时间复杂度，再解释 tokenizer 并非总能逐 token独立映射字符串，最后落到 per-request decode state 与稳定 delta。

**详细答案**：若第 `i` 个输出都重新 decode 前 `i` 个 IDs，总 CPU 工作近似随输出长度二次增长，还会重复分配字符串。增量 detokenizer 保存 token buffer、文本偏移和已确认边界，只处理新 token及少量重叠窗口，输出客户端尚未见过的稳定 delta。

不能简单对每个 token单独 `decode()`：BPE/SentencePiece、byte fallback、Unicode replacement 和前后 token 合并可能使字符边界跨 token；stop string 也可能横跨多个 token。框架需要保留足够尾部，确认不会被后续 token改变后再发送，并让 token IDs、logprobs、文本 offsets、finish reason 对齐。vLLM 在 API 侧 `OutputProcessor` 保存状态，SGLang 在独立 Detokenizer 进程按 rid 保存状态，拓扑不同但不变量相同。

**主问题得分点（10 分）**：

- 2 分：指出全量重复 decode 的二次 CPU/分配成本。
- 2 分：说明 token 到文本并非逐 token独立。
- 2 分：覆盖 UTF-8/byte fallback 和跨 token stop string。
- 2 分：说明稳定 delta、offset/logprob 对齐。
- 2 分：映射到两框架的 per-request detokenizer ownership。

**追问与答案（5 分）**：客户端要求 stop string 不出现在输出中，应怎样流式处理？服务端要保留最多与 stop pattern 前缀相关的未确认尾部，只有确定不可能组成 stop string 才 flush；匹配后截断未发送/当前 buffer，并正确设置 finish reason。得分点：尾部保留 2 分，跨 delta 匹配 1 分，截断与 finish 语义 2 分。

### 8. 从 logits 到一个可提交 token，采样链路通常包含哪些步骤？

- **递进追问**：temperature、top-k/top-p、repetition penalty、grammar mask、logprob 和 RNG 的顺序为何重要？greedy 与 stochastic 路径的正确性测试有何不同？
- **面试官观察点**：是否把 sampling 当成影响分布与可复现性的核心计算，而非一个 `argmax`。
- **源码锚点**：vLLM `GPUModelRunner.sample_tokens()`；SGLang `ModelRunner.sample()`。

**回答思路**：按“原始 logits -> processors/masks -> 概率变换 -> sample -> commit/stop”回答，并强调具体顺序定义了 API 语义与输出分布。

**详细答案**：模型先产生需要位置的 logits。框架按配置应用 logits processors，例如 repetition/frequency/presence penalty、bad words、allowed tokens 和 grammar vocab mask；再根据 temperature 与 top-k/top-p 等规则得到候选分布，使用 per-request RNG/seed 采样，或在 greedy 路径取 argmax。随后生成所需 logprob，更新 token history、penalty 和 grammar state，检查 EOS、stop token/string、最大长度，再决定哪些 token/文本可提交给客户端。

不同框架与 API 对 logprob 是变换前还是最终采样分布、penalty 与 mask顺序可能有明确约定，不能随意交换：先截断再加 penalty 与先 penalty 再截断会得到不同 support。TP ranks 还必须保证采样结果和 RNG 状态一致。Spec Decode 的 stochastic acceptance需要 target/draft probability，而非两个独立 sample相等。

**主问题得分点（10 分）**：

- 2 分：覆盖 logits processor、mask、temperature、truncation 和 sample。
- 2 分：说明顺序会改变分布/support。
- 2 分：覆盖 RNG、seed 和跨 rank一致性。
- 2 分：说明 sample 后的 history/grammar/stop commit。
- 2 分：区分 greedy 与 stochastic/spec correctness。

**追问与答案（5 分）**：怎样测试采样实现没有被优化改坏？Greedy 用确定性 token-by-token oracle；随机采样固定 seed做回归，同时对大量样本做分布/频率检验；分别覆盖 mask、top-p边界、极端 temperature、TP 和 batch order变化。得分点：确定性测试 1 分，统计测试 2 分，组合边界与并行一致性 2 分。

### 9. TP、PP、DP、EP 分别切分什么，主要通信和瓶颈是什么？

- **递进追问**：小 batch decode 为什么可能无法摊薄 TP collective？PP bubble 怎样产生？MoE 的 EP 为什么会把问题转成 all-to-all 与负载均衡？
- **面试官观察点**：能否根据模型结构、单卡容量、互联拓扑和流量目标选并行方式。
- **源码锚点**：两边 `distributed/parallel_state.py`、`communication_op.py`；SGLang `layers/dp_attention.py` 与 EP MoE 实现。

**回答思路**：逐个回答“切什么、通信什么、解决什么约束”，再基于容量、互联和请求形状说明组合选择。

**详细答案**：TP 在单层内部切分权重/heads/channels，各 rank共同执行同一个 batch，层间通常需要 all-reduce、all-gather 或 reduce-scatter；它解决单层容量和算力，但频繁 collective 对小 batch Decode 很敏感。PP 按层切 stage，传输激活，解决整模型容量；吞吐依赖 microbatch 流水，batch 小或不均衡会产生 bubble。DP 复制模型，每个 replica处理不同请求，扩展 aggregate throughput且通信较少，但单副本仍需装下模型；SGLang DP attention等设计可让 attention request分片并与 MoE/其他并行组合。

EP 将 MoE experts分布到 ranks，token 经过路由后发生 all-to-all，容量和专家计算可扩展，但热门 expert 会造成负载不均、网络拥塞和 padding。生产部署通常先用 DP 横向扩吞吐，在模型单卡放不下时按高速互联域内使用 TP，跨节点再考虑 PP；MoE 模型结合 EP。最终选择取决于权重/KV容量、batch、NVLink/PCIe/IB拓扑和 latency SLO。

**主问题得分点（10 分）**：

- 4 分：TP/PP/DP/EP 各自切分和通信均正确。
- 2 分：解释 TP 小 batch与 PP bubble。
- 1 分：解释 EP all-to-all与负载不均。
- 2 分：能结合容量和互联拓扑选择组合。
- 1 分：提出 compute/communication/bubble 分解验证。

**追问与答案（5 分）**：为什么跨节点 TP 往往比节点内 TP 风险大？TP collective 基本每层发生，Decode 每步串行，跨节点延迟和带宽难以隐藏；PP 通信次数少但传激活且有 bubble。应比较每 token collective 时间占比、链路拓扑与 stage平衡，不是绝对禁止跨节点 TP。得分点：频率/串行依赖 2 分，网络特性 1 分，能与 PP量化比较 2 分。

### 10. CUDA Graph 在 LLM serving 中解决什么问题，为什么不能覆盖所有请求形状？

- **递进追问**：capture 对地址、shape 和控制流有什么要求？padding、capture 显存、graph 数量和 fallback 的成本是什么？
- **面试官观察点**：是否知道 graph 优化的是 host launch/dispatch，收益取决于 batch、kernel 时长和命中率。
- **源码锚点**：vLLM `BatchExecutionDescriptor` 与 full/piecewise graph；SGLang `CudaGraphRunner`、`ModelRunner.forward()`。

**回答思路**：先定位 eager iteration 的 host launch gap，再说明 graph replay 的静态约束，最后以命中率和 padding后的端到端成本判断收益。

**详细答案**：LLM Decode 常由许多短 kernel组成，小 batch时 Python/C++ dispatch、CUDA launch和 CPU-GPU同步会在每轮形成明显空档。CUDA Graph 在 capture 时记录固定操作图和内存地址，replay 可一次提交整图，降低 host overhead并稳定 iteration latency。vLLM 根据 `BatchExecutionDescriptor` 选择 eager、piecewise 或 full graph；SGLang 为 Decode及投机解码不同阶段维护 graph runner。

限制来自动态 shape、数据依赖控制流、动态 LoRA/grammar/backend、地址稳定性和不支持 capture的算子。框架常把 batch pad 到已 capture shape，但 padding增加无效计算；capture多种 shape占显存并增加 warmup，未覆盖组合会 fallback。因而要看 graph hit/fallback、replay后 host gap、padding ratio、capture memory 和最终 TPOT，而不是只看开关已启用。

**主问题得分点（10 分）**：

- 2 分：说明优化目标是 host dispatch/launch gap。
- 2 分：说明 capture/replay 与稳定地址、shape、控制流约束。
- 2 分：区分 eager、部分图和完整图。
- 2 分：覆盖 padding、shape数量、显存和 fallback代价。
- 2 分：提出 hit rate、timeline、TPOT 和 memory验证。

**追问与答案（5 分）**：graph hit率 95%但性能没提升，可能为什么？命中的 kernel本来很长而 host gap占比低；padding工作抵消 launch收益；5% fallback恰好是长尾请求；graph前后仍有 sampling/D2H同步；capture占用显存降低 batch容量。得分点：至少三类原因 3 分，要求端到端 timeline与消融 2 分。

---

[系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：请求生命周期与模块契约](/posts/vllm-sglang-interview-level-2-lifecycle/)
