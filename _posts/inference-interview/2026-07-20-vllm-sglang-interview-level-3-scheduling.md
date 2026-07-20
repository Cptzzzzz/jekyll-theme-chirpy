---
title: 'vLLM / SGLang 面试题（三）：调度、准入与资源预算'
author: Cptz
date: 2026-07-20 00:13:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)第 3 卷，覆盖第 21-30 题。重点是Continuous batching、token budget、chunked prefill、公平性与背压。

[上一篇：请求生命周期与模块契约](/posts/vllm-sglang-interview-level-2-lifecycle/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：KV Cache 与 GPU 执行细节](/posts/vllm-sglang-interview-level-4-kv-runner/)

---

## Level 3：调度、准入与资源预算

目标：确认候选人能解释 Scheduler 为什么正确、为什么快，以及过载时为什么会退化。

### 21. vLLM 如何用统一的 token debt 表示 Prefill、Decode 和 Spec Decode 工作？

- **递进追问**：`num_tokens_with_spec - num_computed_tokens` 表示什么？为什么一次可调度多个 token？encoder token、lookahead slot 怎样进入额外预算？
- **面试官观察点**：是否能摆脱“waiting 是 prefill、running 是 decode”的过度简化模型。
- **源码锚点**：vLLM `Request` counters、`Scheduler.schedule()`；[`vllm/modules/SCHEDULER.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/SCHEDULER.md)。

**回答思路**：不要按“请求处于Prefill还是Decode”建两个互斥状态，而要回答target模型距离逻辑token truth还欠多少位置的计算。

**详细答案**：当前V1把prompt、已提交output和draft proposal放入统一token序列视图。核心近似为：

```text
token_debt = num_tokens_with_spec
             + num_output_placeholders
             - num_computed_tokens
```

首次Prefill的debt是未命中prompt suffix；chunked prefill每轮偿还一部分；普通Decode中，新采样token已经成为逻辑truth但尚未为其计算下一位置所需状态，通常形成一个token debt；Spec Decode把draft IDs加入待target verify的debt；async scheduling用placeholder/in-flight counters表示已进入流水但尚未commit的位置。

`Scheduler.schedule()`在本轮token budget内截断debt，并额外受max model length、encoder input、KV blocks和lookahead slots约束。统一计数避免多个phase flag组合爆炸，但乐观推进 `num_computed_tokens` 后必须由同轮output按accepted/rejected结果修复。

**主问题得分点（10 分）**：

- 3 分：给出包含spec/placeholder的debt语义。
- 2 分：正确映射首次/分块Prefill和普通Decode。
- 2 分：说明Spec verify如何形成多token debt。
- 1 分：指出token预算外还有encoder/KV/lookahead约束。
- 2 分：说明乐观推进与reject/commit回滚。

**追问与答案（5 分）**：为什么“output token数减computed token数”容易出现off-by-one？采样出的token已经是逻辑输出，但其KV通常要在下一次forward才写入；同时prompt最后位置的forward产生首个logits。必须按框架对“已知token”和“已计算位置”的定义推导，不能只比较列表长度。得分点：区分known与computed 2 分，首token边界 2 分，拒绝猜长度 1 分。

### 22. 为什么 `schedule()` 与 `update_from_output()` 必须是两个不同阶段？

- **递进追问**：计划阶段可以修改哪些状态，GPU 完成后才能提交哪些状态？`SchedulerOutput` 如何与返回结果配对？异步执行时怎样防止把结果应用到下一轮 batch？
- **面试官观察点**：是否理解 plan/execute/commit transaction，以及 launch-time snapshot 的必要性。
- **源码锚点**：vLLM `SchedulerOutput`、`Scheduler.update_from_output()`；SGLang `NextBatchPlan` 和 result queue。

**回答思路**：把一次iteration视作小型事务：plan阶段预留资源并生成不可歧义的执行描述，execute阶段异步运行，commit阶段应用真实结果或回滚。

**详细答案**：`schedule()`只能根据当前状态决定本轮谁运行、运行多少token，并提前分配KV/encoder/lookahead资源，生成包含request order、token counts、new blocks/copies、spec IDs等的 `SchedulerOutput`。为支持CPU/GPU overlap，它可能乐观增加computed/in-flight计数，但此时还不知道sample、accept length、EOS或设备失败。

`update_from_output()`必须消费与该plan严格配对的runner result，减少in-flight，提交sampled/accepted IDs，回退rejected positions，推进grammar/stop，释放finished资源并生成API输出。若future/result与plan错位，token ID即使合法也会提交给错误request并污染KV。SGLang overlap同样依靠launch-time batch snapshot和result queue延迟commit。分开两阶段还能明确异常回滚和device buffer last-use。

**主问题得分点（10 分）**：

- 2 分：说清plan阶段的资源预留和输出contract。
- 2 分：说明execute可能异步且结果未知。
- 2 分：说清commit的token/stop/free职责。
- 2 分：解释乐观状态为何需要rollback。
- 2 分：说明plan-result pairing和snapshot不变量。

**追问与答案（5 分）**：可以给每个request单独commit而不保留整批snapshot吗？只有结果携带足够稳定的request/iteration/position identity且共享资源更新可原子化才可能；现实中batch order、tensor slices、block changes和TP collective共同定义结果，整批snapshot更容易保持一致。得分点：条件 2 分，共享状态风险 2 分，合理取舍 1 分。

### 23. vLLM 一轮调度至少要同时满足哪些预算？

- **递进追问**：`max_num_batched_tokens`、`max_num_seqs`、KV blocks、encoder budget、grammar/LoRA、remote KV 和 speculative lookahead 谁先约束谁？失败应等待、抢占还是拒绝？
- **面试官观察点**：是否把 Scheduler 当作多资源 admission controller，而不是队列排序器。
- **源码锚点**：`vllm/v1/core/sched/scheduler.py::Scheduler.schedule`、`KVCacheManager.allocate_slots()`。

**回答思路**：按“本轮计算预算、长期存储预算、执行兼容约束、异步依赖”四类列账，并说明只有全部成功才可进入plan。

**详细答案**：首先有 `max_num_batched_tokens` 限制本轮target work，`max_num_seqs/max_num_running_reqs` 限制resident请求数。其次 `KVCacheManager.allocate_slots()`必须同时容纳命中/新分配blocks和spec lookahead，encoder manager还要核算多模态/encoder compute与cache。结构化输出grammar可能未ready；LoRA受同时resident adapters限制；remote KV connector可能仍在load；max model length和per-request限制也会截断token数。

Scheduler对running request通常优先偿还debt，allocation失败时抢占低优先级/末端victim并重试；waiting request只有prefix/external match、所有资源分配和依赖都成功后才从queue移入running并写入 `SchedulerOutput`。暂时依赖未ready应skip/wait，容量压力可preempt或不准入，不可恢复的输入/长度错误才reject。把“排到”与“可执行”混为一谈会生成worker无法落地的计划。

**主问题得分点（10 分）**：

- 2 分：token和sequence预算。
- 2 分：KV/new/lookahead及encoder资源。
- 2 分：grammar、LoRA、remote KV等兼容/异步约束。
- 2 分：区分running preempt、waiting defer和invalid reject。
- 2 分：强调成功分配后才能写入plan。

**追问与答案（5 分）**：为什么token budget不能精确代表GPU时间？Prefill与Decode token的attention长度和GEMM shape不同，MM encoder、spec verify、不同dtype/backend成本也不同。token budget是低成本近似，若做cost-aware调度需在线估计且避免模型误差破坏公平性。得分点：非等价成本 2 分，举出两个因素 1 分，估计与公平权衡 2 分。

### 24. Chunked Prefill 要解决什么问题，它一定会降低 TTFT 吗？

- **递进追问**：chunk 大小怎样影响 decode ITL、长 prompt TTFT、GPU occupancy 和 kernel 次数？如何避免 chunk 中间状态被错误当作可 decode/cache 的完整请求？
- **面试官观察点**：能否说明 chunking 是延迟隔离与批处理效率的权衡，不是无条件加速。
- **源码锚点**：两边 Scheduler 模块的 Chunked Prefill 章节；完整答案指南第 6 题。

**回答思路**：先构造一个长Prefill阻塞并发Decode的head-of-line场景，再比较chunk变小后单轮上界、总工作和launch次数。

**详细答案**：长prompt若一次进入forward，会占用较长GPU时间，期间运行中的Decode请求不能得到下一token，造成ITL/p99尖峰。Chunked Prefill把未命中suffix按本轮token/chunk budget截断，只分配和计算当前区间，保留请求未完成状态，下轮继续。它限制单轮EXTEND work上界并允许与Decode更细粒度交错，主要目标是延迟隔离和SLO goodput。

它不保证降低该prompt的TTFT。chunk越小，长prompt要经历更多schedule、metadata、kernel launch和中间cache操作，GPU GEMM也可能变小，TTFT反而增加。中间chunk只能提交已真实计算的KV，不能作为正常Decode请求合入running；SGLang保留 `chunked_req/extend_range`，vLLM用computed debt推进。选择chunk size要同时看长prompt TTFT、并发Decode p99 ITL、GPU occupancy和launch count。

**主问题得分点（10 分）**：

- 2 分：解释长Prefill造成Decode HoL blocking。
- 2 分：说明按budget只推进suffix区间。
- 2 分：明确优化目标是ITL/p99/SLO隔离。
- 2 分：说明TTFT、GEMM和launch代价。
- 2 分：说明中间chunk的状态/缓存不变量和实验指标。

**追问与答案（5 分）**：如何自动选择chunk size？可用目标最大GPU占用时间/ITL budget，结合近期token shape到kernel时间模型动态截断，同时设置page/graph对齐和最小高效GEMM尺寸；以p99 ITL约束下最大goodput调参。得分点：SLO目标 2 分，成本模型与对齐 2 分，反馈稳定性 1 分。

### 25. vLLM 在 KV 不足时如何选择和处理 preemption victim？

- **递进追问**：为什么当前常见路径是释放后重算，而不是必然 swap？抢占时哪些 worker resident state、scheduled tokens 和 KV refs 必须同步移除？
- **面试官观察点**：是否能讲清可用性、重算成本、cache churn 和 p99 的因果关系。
- **源码锚点**：`Scheduler.schedule()` 的 allocation failure/preempt path、`KVCacheManager.free()`。

**回答思路**：从“运行中序列还会继续长，首次准入无法保证未来容量”出发，追踪victim从本轮计划撤销到waiting重建的每个状态。

**详细答案**：running request申请本轮slots失败时，vLLM循环选择victim。普通策略从running尾部取请求；priority策略按源码选择最低服务优先级/较晚到达者。如果victim已经加入本轮 `scheduled_running_reqs`，要撤销其token budget、new blocks、spec tokens和encoder budget。随后 `_preempt_request()`释放该request的KV/encoder ownership，标为PREEMPTED，重置可重建的computed/spec/in-flight状态并放回waiting；如果victim就是当前请求且仍无法分配，本轮停止。

恢复时请求重新走prefix lookup/admission，仍存活的hashed prefix可能减少重算，否则需要重新Prefill。当前这条常见路径的语义是release + recompute，不应笼统描述为一定swap到CPU。Preemption保证不OOM和高优先级可进展，但持续发生会增加recompute、cache churn和p99，说明admission预留、KV容量或负载配置有问题。

**主问题得分点（10 分）**：

- 2 分：说明未来KV增长使preemption必要。
- 2 分：准确描述普通/priority victim选择原则。
- 2 分：撤销本轮token/block/spec/encoder预算。
- 2 分：释放、重置、回waiting并依赖prefix重建。
- 2 分：区分recompute与swap并说明性能代价。

**追问与答案（5 分）**：为什么不总抢占KV占用最大的请求？它释放空间多，但可能已生成很长、重算代价最高，也可能高优先级且接近完成。更合理的score综合可释放blocks、预计剩余时间、重算成本、priority/deadline和cache可恢复性。得分点：反例 2 分，多因素score 2 分，避免饥饿 1 分。

### 26. SGLang 为什么把 `SchedulePolicy` 与 `PrefillAdder` 分开？

- **递进追问**：排序得分与硬资源准入分别输入什么、输出什么？LPM/DFS/priority 改变执行顺序时，为什么不能绕过 grammar、LoRA、PD readiness 和容量约束？
- **面试官观察点**：是否区分 policy preference 与 feasibility，并理解模块化对扩展新策略的价值。
- **源码锚点**：`schedule_policy.py`、`Scheduler.get_new_batch_prefill()`、`PrefillAdder`。

**回答思路**：用数据库查询的“排序”和“约束检查”类比：Policy给出尝试顺序，Adder维护资源ledger并决定能否真正执行。

**详细答案**：`SchedulePolicy` 输入waiting requests及prefix/cache信息，根据FCFS、priority、LPM或DFS-weight等策略计算顺序，输出ordered candidates和必要的match state。它优化TTFT、公平性或cache locality，但不拥有物理allocation，也不能保证候选可运行。

`PrefillAdder` 接收该顺序、running batch和free/evictable KV、token/chunk、request row、SWA/Mamba、future decode等预算。它对每个request lock prefix、double-check容量、选择full或chunked EXTEND，成功才加入 `can_run_list`并扣账。分离后可替换排序策略而复用复杂正确性约束，也避免cache-aware policy绕过grammar、LoRA、PD readiness和容量硬条件。

**主问题得分点（10 分）**：

- 3 分：Policy输入/输出及只决定顺序。
- 3 分：Adder的多资源ledger与准入输出。
- 2 分：说明硬约束不可被策略绕过。
- 1 分：说明可插拔和测试价值。
- 1 分：指出排序/匹配的CPU成本。

**追问与答案（5 分）**：LPM一定提升吞吐吗？共享prefix高时可减少Prefill work并改善locality；低复用或waiting很大时，match/sort CPU可能超过收益，还可能让短无共享请求饥饿。得分点：收益条件 2 分，CPU/公平代价 2 分，需workload实验 1 分。

### 27. SGLang 的 PrefillAdder 为什么需要为未来 Decode 预留资源？

- **递进追问**：只按本轮 EXTEND 所需 slot 准入会导致什么？`new_token_ratio` 类估计过保守或过激分别怎样影响 batch size、retraction 和 goodput？
- **面试官观察点**：是否理解 admission 必须估计请求未来增长，而不是只做瞬时 free-space check。
- **源码锚点**：`PrefillAdder` 的 token/KV ledger；[`sglang/modules/SCHEDULER.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/SCHEDULER.md)。

**回答思路**：构造“本轮prompt刚好装下，但所有请求下一token都要新增KV”的反例，说明瞬时容量检查不是稳定admission。

**详细答案**：一个request的suffix当前能放下，只代表本轮EXTEND可执行。准入后它会进入Decode并持续增长；若同时准入很多“刚好放下”的request，下一轮所有running请求申请slot时就会耗尽KV，只能大规模retract甚至abort。`PrefillAdder` 因此从free + evictable容量中扣除running及新request的未来Decode估计，使用 `new_token_ratio`、剩余 `max_new_tokens` 等形成 `rem_total_tokens`，并分别核算input/chunk、SWA、Mamba和request rows。

ratio太大时保留过多headroom，batch变小、GPU利用和throughput下降；太小时over-admission，retraction、recompute和p99增加。它是风险估计而非正确输出长度预测，运行时还需根据实际完成/retraction更新。`ignore_eos` 等请求可能要求更保守估计。

**主问题得分点（10 分）**：

- 3 分：能给出瞬时准入导致下一轮OOM的场景。
- 2 分：说明future reservation的估计输入。
- 2 分：覆盖full/SWA/Mamba/request row等多ledger。
- 2 分：分析过保守与过激的双向代价。
- 1 分：提出用retraction/goodput反馈调参。

**追问与答案（5 分）**：为什么不能简单为每个请求预留 `max_new_tokens`？用户上限通常远大于实际输出，全部预留会严重降低并发；但完全按均值又忽略长尾。可按tenant/model历史分布、priority和风险分位数估计，并保留全局emergency headroom。得分点：上限浪费 2 分，长尾风险 1 分，分层概率预留 2 分。

### 28. SGLang 命中 Radix prefix 后为什么要 lock，并重新检查可淘汰容量？

- **递进追问**：命中但未锁定时该 node 属于 free/evictable 还是 used？把同一批容量同时算作“prefix 命中”和“可分配空间”会造成什么错误？
- **面试官观察点**：能否从 ownership 变化解释 double counting，而不是只记住 lock API。
- **源码锚点**：`RadixCache.inc_lock_ref()`、`PrefillAdder` lock double-check、`RadixCache.evict()`。

**回答思路**：区分“内容在cache里”和“物理slot受活跃request保护”，计算一次lock前后的free + evictable集合变化。

**详细答案**：Radix node未被活跃请求引用时，其KV内容可继续命中，但物理slots属于evictable容量；`PrefillAdder`初始可把它计入潜在可用空间。新请求match该node后必须 `inc_lock_ref()`，确保从调度到GPU消费期间不会被eviction覆盖。lock使这部分容量从evictable变为protected，同时请求又希望复用它。

如果只在lock前检查，账本可能一边把这些slots当命中prefix，一边仍把它们当作可evict空间分配给suffix，形成double counting，最终allocation失败或覆盖活跃KV。因此源码在lock后重新计算/检查 `cur_rem_tokens/rem_total_tokens`，失败则释放lock并不准入。eviction只能选择无锁候选，通常从可淘汰leaf释放。

**主问题得分点（10 分）**：

- 3 分：区分cached-evictable与locked-protected状态。
- 2 分：说明lock覆盖schedule到device使用窗口。
- 2 分：准确解释容量double counting。
- 2 分：说明lock后double-check和失败解锁。
- 1 分：指出eviction只能处理未锁节点。

**追问与答案（5 分）**：lock是否意味着该node永远不能淘汰？不是。它是引用计数，活跃request完成、retract或转移ownership后 `dec_lock_ref()`；ref归零后node重新成为evictable，内容可保留到真正覆盖。得分点：引用计数 2 分，释放路径 2 分，lazy cache语义 1 分。

### 29. SGLang Decode retraction 的触发条件、状态回滚和后续代价是什么？

- **递进追问**：被 retract 的请求回到哪里？private KV、cache prefix、request row、allocated/committed length 如何处理？怎样选 victim 才能降低重算与 starvation？
- **面试官观察点**：是否能完整描述紧急释放、状态一致性和再次准入，而非只说“显存不足就踢请求”。
- **源码锚点**：`Scheduler.update_running_batch()`、`ScheduleBatch.retract_decode()`。

**回答思路**：按trigger、victim order、释放内容、返回队列、再次准入五步回答，并区分allocated和committed状态。

**详细答案**：running batch准备下一次Decode/spec所需slots时，如果allocator空间不足，`ScheduleBatch.retract_decode()` 按policy/配置计算retraction顺序，逐个移除request直到剩余batch可分配。被retract请求释放private/uncommitted KV和request运行态，已安全插入Radix的committed prefix可保留为cache；其 `Req` 回到waiting并带有retracted标记，之后重新match prefix和EXTEND。batch被filter后更新 `new_token_ratio` 估计。

如果撤掉其他请求后最后一个仍无法运行，源码可将其标为OOM abort，避免无限循环。victim选择要考虑priority、已完成长度、可释放资源和重算成本；不允许被retract请求继续引用已归还slots。频繁retraction不是吞吐优化，而是future reservation或容量预测失效的信号，应观测retracted requests、recompute tokens、cache churn和p99。

**主问题得分点（10 分）**：

- 2 分：准确说明decode allocation failure触发。
- 2 分：说明排序移除到容量恢复。
- 2 分：区分private/uncommitted释放与committed cache保留。
- 2 分：回waiting、重新match并更新估计。
- 2 分：覆盖最后请求abort和频繁retraction指标。

**追问与答案（5 分）**：retraction与vLLM preemption的共同不变量是什么？都必须在worker不再使用后释放物理资源，把逻辑request退回可重建边界，撤销本轮plan状态，且不能丢失已提交output语义。得分点：设备lifetime 1 分，逻辑/物理边界 2 分，plan回滚与output保留 2 分。

### 30. 设计一个同时支持 priority、deadline 和公平性的调度策略。

- **递进追问**：如何避免低优先级 starvation？prefill 与 decode 用同一个分数是否合理？cache locality 应作为主目标还是 tie-breaker？SLO 违约时怎样降级？
- **面试官观察点**：能否给出明确 objective、状态、复杂度、不变量与 trace-driven 实验，而不只提出“加权排序”。
- **源码锚点**：vLLM Scheduler waiting/running queues；SGLang `SchedulePolicy`；完整答案指南第 9、24 题。

**回答思路**：先定义优化目标和硬约束，再设计score、两阶段准入与aging；最后给复杂度、过载降级和trace replay验证，而不是只写一个排序公式。

**详细答案**：可为每个请求维护arrival、priority、deadline、预测prefill/decode cost、remaining tokens、cache hit和tenant service。硬约束先过滤grammar/LoRA/PD readiness和资源不可行项。waiting score可组合deadline slack、priority和aging，例如优先最小 `slack = deadline-now-estimated_remaining_service`，但为每个tenant维护virtual runtime/deficit防止独占；cache locality只作为相近slack下的tie-breaker。running Decode需保留最小服务份额，避免新Prefill持续插入导致ITL违约。

Scheduler仍按plan/commit分离：score只给顺序，allocator决定可行性；高优先级抢占要计入recompute penalty，低优先级随等待时间提升有效priority。过载时先拒绝预计已无法满足deadline的新请求，而不是接收后无限排队。实现可用heap/bucket避免每轮全量复杂排序，并用真实arrival trace回放比较SLO goodput、starvation、preemption和Scheduler CPU。

**主问题得分点（10 分）**：

- 2 分：明确SLO goodput/fairness目标和hard constraints。
- 2 分：定义slack/priority/aging或等价可实现score。
- 2 分：区分waiting Prefill与running Decode服务保障。
- 2 分：纳入preemption成本、tenant公平和overload reject。
- 2 分：给出数据结构、trace实验和消融指标。

**追问与答案（5 分）**：cache locality为什么不应直接成为最高优先级？它可降低总计算，但会持续偏爱热门prefix，造成冷请求/租户饥饿，甚至热门实例排队更长。应将其转成预计service-time节省或有限tie-breaker，并受deadline/deficit约束。得分点：局部收益 1 分，公平/排队反例 2 分，受约束整合方案 2 分。

---

[上一篇：请求生命周期与模块契约](/posts/vllm-sglang-interview-level-2-lifecycle/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：KV Cache 与 GPU 执行细节](/posts/vllm-sglang-interview-level-4-kv-runner/)
