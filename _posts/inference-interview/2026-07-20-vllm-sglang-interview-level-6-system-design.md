---
title: 'vLLM / SGLang 面试题（六）：Staff 级系统设计与现场推演'
author: Cptz
date: 2026-07-20 00:16:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)第 6 卷，覆盖第 51-60 题。重点是SLO、容量、故障恢复、正确性不变量、扩展点与演进路线。

[上一篇：高级优化、分布式与性能诊断](/posts/vllm-sglang-interview-level-5-advanced-distributed/) · [系列总索引](/posts/vllm-sglang-interview-series/)

---

## Level 6：Staff 级系统设计与现场推演

目标：确认候选人能组合前述机制，在不确定 workload、故障和多租户约束下做完整工程决策。

### 51. 设计一个 deadline-aware continuous batching Scheduler。

- **递进追问**：输入状态、目标函数、waiting/running 决策、prefill/decode budget、preemption、aging、复杂度和不变量是什么？怎样接入 vLLM 或 SGLang 现有边界？
- **面试官观察点**：答案必须落到可实现的 plan/commit 状态机，并有 trace replay、消融实验和 SLO goodput 判据。
- **源码锚点**：vLLM `Scheduler.schedule()`；SGLang `SchedulePolicy/PrefillAdder`；完整答案指南第 24 题。

**回答思路**：按“需求/SLO -> 状态与估价 -> 排序 -> 多资源准入 -> plan/commit -> 过载与验证”回答，先保证正确性，再讨论score最优性。

**详细答案**：入口为request记录arrival、TTFT/ITL/E2E deadline、priority、tenant、prompt/cache hit和最大输出。成本模型分别预测remaining Prefill、下一Decode iteration、KV增长和transfer；计算 `slack = deadline-now-estimated_remaining_service`。waiting queue可按slack bucket/heap排序，在同bucket加入priority和aging；running Decode保留每轮最低token份额，避免Prefill让ITL违约。tenant deficit/virtual runtime限制长期占用。

score只决定尝试顺序，现有KV/encoder/LoRA/grammar allocator仍做硬准入。preempt只有当预计释放收益大于recompute且能改善更紧deadline时发生；所有决策生成iteration plan，结果到达后commit/rollback。预计已不可能满足SLO的新请求应早拒绝或降级，而不是排队到超时。实现使用deadline buckets减少每轮全量排序，定期校准cost model；通过trace replay和shadow mode比较SLO goodput、deadline miss、starvation、preempt/recompute与Scheduler CPU。

**主问题得分点（10 分）**：

- 2 分：定义deadline/slack、阶段成本和目标函数。
- 2 分：waiting排序、running Decode保障和aging。
- 2 分：tenant公平与硬资源准入分离。
- 2 分：preemption/early reject及plan/commit不变量。
- 2 分：数据结构、模型校准、trace/shadow验证。

**追问与答案（5 分）**：成本预测错误会不会让EDF类策略更差？会。系统应给估计加入置信区间/安全裕量，在线校准并限制单次决策影响；未知请求退化为size class或FCFS+aging。用shadow记录反事实而非直接全量上线。得分点：误差意识 1 分，稳健估计/回退 2 分，渐进发布 2 分。

### 52. 设计一个支持多租户与动态 LoRA 的 serving 系统。

- **递进追问**：tenant quota、priority、adapter residency、batch compatibility、cache namespace、热租户保护和冷租户 starvation 如何处理？LoRA 切换成本怎样进入调度分数？
- **面试官观察点**：是否同时覆盖公平性、隔离、GPU 容量、正确 cache key 和可观测性。
- **源码锚点**：两边 request LoRA fields、Scheduler LoRA constraints、vLLM block hash extra keys、SGLang `RadixKey.extra_key`。

**回答思路**：把租户隔离、调度公平、adapter residency和cache correctness分成四个子系统，再说明它们在admission时如何联合决策。

**详细答案**：入口为tenant配置token/request rate、concurrency、priority和SLO，使用token bucket加per-tenant waiting queue；Scheduler以deficit round robin/virtual time分配token work，并设全局紧急容量，避免一个长prompt或慢client独占。LoRA manager维护GPU/CPU adapter residency、ref count、load future和LRU；请求只有adapter ready且本轮active LoRA数不超backend限制时才可执行，batch尽量聚合同adapter但受aging约束。

KV cache key必须包含base model revision、LoRA/adapter identity与版本，必要时包含tenant salt；adapter更新后不能命中旧KV。cache容量可按tenant软配额/加权淘汰，防止热门tenant挤出所有缓存。取消/evict adapter前等待in-flight kernel last use。指标按tenant报告queue、TTFT/ITL、GPU tokens、KV/adapter bytes、cache hit、throttle和starvation；压测覆盖热/冷adapter交替和恶意大请求。

**主问题得分点（10 分）**：

- 2 分：tenant rate/concurrency与token级公平队列。
- 2 分：adapter load/residency/ref count和batch兼容。
- 2 分：cache key包含adapter版本/tenant namespace。
- 2 分：cache/adapter容量隔离与aging。
- 2 分：in-flight lifetime、按tenant指标和压力验证。

**追问与答案（5 分）**：总是按LoRA分组能提高吞吐，为什么还需限制？持续热门adapter会让冷adapter请求永远等不到load/batch。设置最大聚合窗口、deadline/aging提升和tenant deficit，只有在不破坏SLO时使用adapter locality。得分点：locality收益 1 分，starvation 2 分，受限聚合策略 2 分。

### 53. 设计一个跨副本的 prefix-aware router 与分布式 KV Cache。

- **递进追问**：router 如何获知 cache locality 又不维护过期全量状态？一致性、eviction event、租户隔离、热点复制、失败 fallback 和 load balance 怎样权衡？
- **面试官观察点**：能否识别“最高命中率”可能导致热点实例过载，并用预期节省 prefill 成本减去排队/传输成本决策。
- **源码锚点**：vLLM KV events/connectors；SGLang Radix cache、disaggregation KV events；两边 prefix 模块。

**回答思路**：Router不需要复制每个block table，只需持有带epoch的prefix存在性摘要和实例负载，用“预计节省成本减去等待/传输成本”选路。

**详细答案**：每个replica发布model/adapter/tenant namespace、prefix digest区间或Bloom/近似索引、可复用token数、cache epoch以及queue/KV水位；evict/finish事件异步更新Router，摘要允许false positive但不能导致错误复用。请求先计算同样的链式prefix identity，候选实例分数可写为 `saved_prefill_time(hit) - predicted_queue_delay - remote_transfer_cost - overload_penalty`，而不是只取最长hit。

到达实例后必须用本地精确cache lookup再次验证；stale摘要miss/false hit安全退化为Prefill。model/LoRA/version和tenant salt保证正确性隔离。热门prefix可预热/复制，但要限流避免所有请求形成热点；一致性用immutable content hash + epoch/event最终收敛，不要求对eviction做昂贵强一致。跨实例拉KV还需connector completion/ownership协议。故障时移除实例、重试带幂等ID并防止重复提交。

**主问题得分点（10 分）**：

- 2 分：设计可扩展prefix摘要/epoch/event传播。
- 2 分：路由score包含命中节省、排队和transfer。
- 2 分：实例本地精确验证与安全fallback。
- 2 分：版本/adapter/tenant namespace和故障幂等。
- 2 分：热点复制、过载平衡和观测验证。

**追问与答案（5 分）**：Bloom filter false positive会破坏生成正确性吗？不应。它只用于路由提示，目标实例必须用完整hash chain和本地metadata验证；false positive只增加一次错误选路/重算成本。得分点：提示非权威 2 分，本地验证 2 分，性能而非语义后果 1 分。

### 54. 设计过载保护与端到端 backpressure。

- **递进追问**：在 HTTP、tokenization、engine ingress、waiting queue、KV admission、output queue 各设置什么界限？何时 reject、shed、degrade、cancel？怎样避免重试风暴？
- **面试官观察点**：是否能用 Little's Law、SLO budget 和资源水位建立闭环，而不是无限排队。
- **源码锚点**：vLLM `AsyncLLM/EngineCoreClient/Scheduler` 边界；SGLang `TokenizerManager/Scheduler/Detokenizer` ZMQ 边界。

**回答思路**：列出每一层有限资源和最大允许排队时间，让上游只在下游有credit时生产；然后定义reject、degrade和cancel的顺序。

**详细答案**：HTTP层限制连接、请求bytes、prompt tokens和tenant rate；tokenizer层有有界work queue/CPU semaphore；engine ingress与Scheduler waiting按预计token work/KV而非request数设上限；GPU admission预留KV headroom；output层按request/global pending bytes和consumer lag限流。相邻层通过credits或awaitable queue传播水位，不能让ZMQ/asyncio默认无界buffer隐藏压力。

Admission估计 `queue_delay + service_time`，若已超SLO或容量上限，尽早返回429/503与带抖动Retry-After。接近过载可限制 `n`/max_tokens、关闭昂贵logprobs/spec或把低priority流量导向batch服务，但需显式协议；运行中只在收益大于recompute时preempt。慢输出超时取消，不能阻塞全局handler。监控每层queue age/bytes、reject reason、retry rate、KV水位、preempt和SLO goodput，使用open-loop阶梯与突发流量验证无内存增长和无重试风暴。

**主问题得分点（10 分）**：

- 2 分：HTTP/tokenizer/engine/Scheduler/output均有有界水位。
- 2 分：token/KV/bytes度量而非只数请求。
- 2 分：credit传播和早拒绝。
- 2 分：明确degrade/preempt/cancel语义与重试抑制。
- 2 分：完整指标、open-loop overload和稳定性验证。

**追问与答案（5 分）**：为什么增大waiting queue不能提高容量？容量由最慢服务阶段决定；更长queue只增加Little's Law中的等待和内存，使更多请求在开始前已错过SLO。得分点：瓶颈容量 2 分，Little's Law/SLO 2 分，早拒绝 1 分。

### 55. 为长 prompt、短 output 和短 prompt、长 output 混合流量设计 P/D 分离集群。

- **递进追问**：P:D replica 比例、routing、KV transfer、decode admission、故障恢复和 autoscaling 信号怎么选？在哪些 workload 下应回退到 colocated serving？
- **面试官观察点**：能否用阶段服务时间、网络带宽、KV bytes 和队列稳定性给出容量模型。
- **源码锚点**：两边 P/D/KV connector 路径；完整答案指南第 19 题。

**回答思路**：分别计算两类流量在Prefill、transfer和Decode三个service centers的工作量，按利用率与SLO配置P/D比例，并保留colocated fallback。

**详细答案**：从trace得到每类请求prompt `P`、output `O`、KV bytes/token和各硬件上 `S_p(P)`、`S_d(O,batch)`、`S_x(P)`。到达率为 `lambda_i` 时，P池需求近似 `sum lambda_i*S_p_i`，D池需求为 `sum lambda_i*S_d_i`，网络/staging需求为 `sum lambda_i*KVBytes(P_i)`；各池利用率需留p99与故障headroom，不能只按平均tokens配比。长prompt短output主要压P和网络，短prompt长output主要压D/KV存活。

Router按请求特征、queue和cache hit选择P节点，transfer完成后D admission才接管；D端对长output设置KV/tenant配额。P/D分别autoscale：P看prefill queue/TTFT和compute，D看running tokens、ITL/KV水位，网络看transfer age/credits。短prompt、transfer setup占比高、网络拥塞或规模小时直接colocated更快；故障时可在D重算Prefill或路由colocated，但需幂等防止双响应。最终按SLO goodput而非各池单独utilization选比例。

**主问题得分点（10 分）**：

- 3 分：写出P、D、transfer三池容量/字节模型。
- 2 分：区分两类流量的资源主导项。
- 2 分：routing、completion后handoff和D admission。
- 1 分：独立autoscaling信号。
- 2 分：colocated/failure fallback与SLO验证。

**追问与答案（5 分）**：网络带宽够就一定适合P/D吗？不一定。还受单次transfer latency、注册/setup、NUMA/PCIe路径、queue同步、目标slot等待和短prompt占比影响；应比较完整 `S_x` 与隔离节省。得分点：带宽外延迟 2 分，控制/slot队列 2 分，端到端对比 1 分。

### 56. 设计一个按负载动态调节 speculative width/steps 的控制器。

- **递进追问**：观测 accept-length histogram、stage cost、KV headroom、graph shape、queue pressure 的哪些统计？如何避免控制震荡和切换时 stale state？
- **面试官观察点**：是否以“预测单位 committed token 成本”为目标，并给出禁用 speculation 的明确阈值与回退路径。
- **源码锚点**：SGLang adaptive spec state；两边 speculative performance model。

**回答思路**：把每个可用spec配置视为离散action，在线估计每个action的单位committed token成本和容量代价，用带滞回的控制器选择。

**详细答案**：观测按model/workload bucket统计accept-length histogram、draft/verify/extend/compact阶段时间、graph hit/padding、batch size、queue pressure、KV headroom和普通Decode基线。对每个已预注册且graph/buffer已准备的steps/top-k action，估计：

```text
cost(action) = total_iteration_cost / expected_committed_tokens
```

并加入临时KV导致的concurrency penalty与SLO风险。低并发、高接受且proposal便宜时增steps；高batch、低接受、graph miss或KV紧张时降steps，必要时0。使用EWMA、最小驻留时间、上下阈值和有限步长避免震荡；只在iteration/plan安全边界切换，清理旧draft/overshoot state并让Scheduler、target、drafter使用同一配置。shadow预测与逐级canary比较TPOT和goodput，保留立即回退普通Decode。

**主问题得分点（10 分）**：

- 2 分：完整观测accept与各阶段成本。
- 2 分：以单位committed token成本为目标。
- 2 分：加入queue、KV和graph约束。
- 2 分：滞回/EWMA/驻留防震荡。
- 2 分：安全切换、state清理、canary和fallback。

**追问与答案（5 分）**：steps=0时为什么某些实现仍保留draft-extend？可保持drafter KV/state warm，低负载恢复spec时无需重建；但高负载若该成本不划算可由明确开关跳过。得分点：warm state收益 2 分，额外成本 1 分，受控策略/一致性 2 分。

### 57. 如何给异步推理引擎设计低扰动的 tracing 与日志？

- **递进追问**：怎样关联 request、iteration、batch、rank、CUDA stream 和 KV allocation？同步打印为什么会改变被测系统？采样、ring buffer、NVTX/CUPTI 和离线聚合怎样分工？
- **面试官观察点**：是否能区分逻辑因果时序与日志输出顺序，并量化 instrumentation overhead。
- **源码锚点**：两个 fork 的 `FLOW_TRACE.md` 与 `Step 1-17` 日志；SGLang overlap result snapshot；vLLM EngineCore events。

**回答思路**：先定义跨层唯一关联键与事件schema，再让热路径只做无阻塞写入，离线重建因果图，并单独校准观测开销。

**详细答案**：每个event至少带trace/request ID、internal child ID、iteration/plan ID、batch ID、rank、process/thread、CUDA stream、monotonic timestamp、phase和关键计数；KV事件再带logical block/physical slot及ownership transition。API注入trace context，经IPC传播，Scheduler为每轮生成稳定plan ID，runner以NVTX/CUPTI标记GPU spans，result回传同一ID。不能依赖打印先后推断异步因果。

热路径默认只记录固定大小结构化metadata到per-thread ring buffer，按请求/异常/尾延迟采样，后台批量flush；禁止每token同步格式化、全局锁或CUDA synchronize。敏感prompt/token不写日志，只写长度/hash且受tenant策略约束。离线按parent/plan/event dependencies重建critical path，并将trace开关A/B比较CPU、ITL、tokens/s、drop rate；buffer满时丢低优先级trace而不能阻塞推理。

**主问题得分点（10 分）**：

- 2 分：完整request/iteration/batch/rank/stream关联键。
- 2 分：结构化事件与KV ownership transition。
- 2 分：ring buffer、sampling和异步flush低扰动。
- 2 分：NVTX/CUPTI与因果图而非日志顺序。
- 2 分：隐私、drop策略和overhead A/B校准。

**追问与答案（5 分）**：为什么在日志前后加 `torch.cuda.synchronize()` 会得到更“准确”却错误的性能结论？它改变原有异步执行和overlap，把GPU等待强加给CPU，测到的是被instrument后的串行系统。应用CUDA events/NVTX异步计时。得分点：同步扰动 3 分，替代手段 2 分。

### 58. 你会怎样验证一个 Scheduler/KV 优化没有破坏正确性？

- **递进追问**：设计哪些 property/invariant test、随机状态机、压力测试和 fault injection？如何覆盖 abort、preempt、prefix collision、partial CoW、spec reject、overlap 和 rank failure？
- **面试官观察点**：是否把 ref count、committed/allocated boundary、request-result pairing 和 target distribution 转成可执行断言。
- **源码锚点**：十个模块文档中的“正确性不变量”和“如何验证”章节。

**回答思路**：先把设计文档中的ownership和计数关系写成断言，再分层做纯状态机、差分语义、并发压力与故障注入。

**详细答案**：核心properties包括：ref/lock大于0的KV不可覆盖；physical slot最多一个writer且只释放一次；`allocated >= committed`，未验证spec positions不进cache；plan/result request order与iteration ID一致；abort/preempt后worker不再引用已释放资源；随机acceptance保持target分布；TP/PP ranks进度一致。纯CPU allocator/Scheduler模型用随机add/schedule/finish/abort/preempt序列，每步检查free+owned总量与引用。

语义层与无cache/eager/无spec reference逐token差分，覆盖prefix collision/salt/LoRA/MM、partial CoW、chunk边界、stop和固定seed；stochastic用统计检验。并发层强制调整CUDA stream延迟、D2H乱序、overlap filter和late abort；fault injection覆盖OOM allocation失败、connector timeout、进程/rank失败并检查fallback/幂等。长稳测试监控VRAM/RSS、orphan blocks、重复ID和p99；任何性能优化先shadow/canary并保留invariant telemetry。

**主问题得分点（10 分）**：

- 2 分：列出KV ref/allocated-committed/单writer不变量。
- 2 分：plan-result/abort/跨rank不变量。
- 2 分：随机状态机与资源守恒property test。
- 2 分：reference差分和stochastic统计测试。
- 2 分：异步/fault injection、长稳与canary。

**追问与答案（5 分）**：哈希碰撞测试怎样做而不等真实digest碰撞？依赖注入一个故意低位/constant hash函数，构造token不同但hash相同的blocks，验证实现是否有namespace/key校验或明确接受的碰撞模型，至少确保不会double-free。得分点：可控hash注入 2 分，语义/ownership断言 2 分，模型边界 1 分。

### 59. 一个业务应该选择 vLLM 还是 SGLang？请给出可审计的决策过程。

- **递进追问**：如何根据模型支持、结构化生成、prefix workload、spec、P/D、并行、硬件、运维成熟度和团队调试能力加权？如何避免用单次 benchmark 过拟合选型？
- **面试官观察点**：是否能把源码成熟度、功能边界、性能证据和迁移风险写成 decision matrix，而不是站队。
- **源码锚点**：[`VLLM_VS_SGLANG.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/VLLM_VS_SGLANG.md)；完整答案指南第 23 题。

**回答思路**：把选型变成可重复的decision matrix：先列硬门槛，再按业务trace衡量性能，最后加入运维、团队和迁移风险，不以社区印象代替证据。

**详细答案**：先筛硬约束：目标模型/量化/硬件backend、OpenAI语义、structured output/MM、LoRA、并行、spec算法、P/D或外部KV connector是否在固定commit可用且经过验证。再用真实trace在同配置下比较TTFT/ITL SLO goodput、VRAM、prefix hit、preemption/retraction、CPU和长稳错误。vLLM与SGLang都快速演进，应评估所需路径的代码owner、测试覆盖、release节奏和当前版本，不做永久标签。

运营维度包括部署拓扑、metrics/tracing、滚动升级、故障恢复、调试复杂度、团队已有patch和上游合入成本。给每项权重、证据链接、风险owner和退出条件；对关键未知做两周POC/影子流量。迁移采用兼容API、golden outputs、双写/镜像和分流canary，明确cache不可跨不等价版本复用。最终结论应是“在该workload和约束下选择X”，不是框架排名。

**主问题得分点（10 分）**：

- 2 分：硬功能/模型/硬件门槛。
- 2 分：真实trace下SLO goodput和资源证据。
- 2 分：版本固定、源码成熟度与测试owner。
- 2 分：运维/团队/迁移成本。
- 2 分：加权matrix、POC、canary和退出条件。

**追问与答案（5 分）**：某框架benchmark快15%就应直接选吗？不应。确认差异是否来自配置/语义、是否在目标SLO负载持续、关键功能和稳定性是否等价；15%还要与维护patch、故障和迁移成本比较。得分点：可比性 2 分，SLO/稳定性 2 分，总成本 1 分。

### 60. 现场设计：实现一个下一代统一推理 runtime，你保留和重构两框架的哪些部分？

- **递进追问**：请定义 API/core contract、Scheduler transaction、KV abstraction、runner tensor contract、async execution、spec plugin、distributed connector、observability 和 compatibility strategy；指出三个最危险的不变量。
- **面试官观察点**：高分答案不会罗列功能，而会定义 ownership、Input/Output、fast path、fallback、性能模型、验证计划和分阶段交付边界。
- **源码锚点**：[`vllm/ARCHITECTURE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/ARCHITECTURE.md)、[`sglang/ARCHITECTURE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/ARCHITECTURE.md) 及十个模块文档。

**回答思路**：先定义少量稳定contracts和owner，再把优化做成可插拔plan/execute/commit组件；给出fast path、fallback、不变量、观测和分阶段交付。

**详细答案**：保留清晰的API/Core隔离：frontend只做协议、tokenize和per-request output state，向core提交稳定 `CoreRequest`；core的Scheduler是逻辑token、资源和lifecycle唯一owner，每轮输出不可歧义 `ExecutionPlan`，runner只消费plan并返回带plan ID的 `ExecutionResult`。KV层以logical ranges、group specs和physical handles抽象paged/full/SWA/Mamba/remote states，统一 `lookup/reserve/commit/rollback/release` transaction；不能向policy泄漏backend tensor细节。

Runner维护resident buffers，将ragged request转换为 `ForwardBatch`，执行backend选择、graph、attention和sampling；async执行显式future/event与snapshot lifetime。Spec作为plugin实现 `propose/verify-result/repair` contract，必须声明临时KV和概率信息。Distributed connector实现带completion/abort/credits的KV/weight协议。Policy与admission分离，可插拔deadline/cache/fairness但共享allocator不变量。所有层输出结构化events并支持eager/no-cache/no-spec reference fallback。

三个最危险不变量是：plan/result与request/iteration严格配对；任何物理state在last device use前不释放且只释放一次；只有target确认的committed path能对下游/cache可见。交付顺序是单卡eager正确性 -> paged KV/continuous batch -> graph/overlap -> spec -> distributed，每阶段用golden差分、property test、故障注入和SLO benchmark设门槛。

**主问题得分点（10 分）**：

- 2 分：API/Core、Scheduler plan/commit和runner contracts清晰。
- 2 分：统一KV transaction与多state/remote抽象。
- 2 分：async snapshot/lifetime和spec plugin contract。
- 1 分：distributed connector/backpressure与policy/admission分离。
- 2 分：明确三个核心不变量及reference fallback。
- 1 分：给出可验证的分阶段交付。

**追问与答案（5 分）**：最先不应该抽象什么？不要一开始统一所有model/backend算子或为未知优化设计巨型字段集合；先稳定ownership与plan/result/KV事务，backend差异通过窄capability和metadata扩展。得分点：反对过度抽象 1 分，指出应先稳定的边界 3 分，可演进机制 1 分。

---

## 推荐面试组合

不要连续问同一模块的记忆题。一次面试应至少包含一题完整控制流、一题资源/正确性和一题性能验证。

| 目标级别 | 60 分钟组合 | 重点判断 |
| --- | --- | --- |
| 初级/校招 | 1、2、4、11、38 | Transformer 成本、指标、端到端意识、tensor 基础 |
| 中级推理工程师 | 3、6、13、21、24、31、49 | continuous batching、并发、调度、KV、定位方法 |
| 高级推理工程师 | 20、22、28、33、40、41、44、48 | ownership、异步正确性、spec、分布式性能模型 |
| Staff/架构师 | 30、51、53、54、56、58、60 | objective、不变量、容量模型、故障闭环、落地路径 |

### 自测方式

1. 第一遍每题只讲 90 秒，检查能否说出“问题、Input、方法、Output、收益、代价”。
2. 第二遍选择源码锚点，画出对象和 ownership 变化，不看笔记复述主路径。
3. 第三遍回答递进追问，必须给公式、反例或实验，不能只给结论。
4. 最后对照完整答案指南和模块文档补缺；源码版本不同的地方必须以当前 checkout 为准。

---

[上一篇：高级优化、分布式与性能诊断](/posts/vllm-sglang-interview-level-5-advanced-distributed/) · [系列总索引](/posts/vllm-sglang-interview-series/)
