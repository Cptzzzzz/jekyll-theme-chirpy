---
title: 'vLLM / SGLang 面试题（五）：高级优化、分布式与性能诊断'
author: Cptz
date: 2026-07-20 00:15:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)第 5 卷，覆盖第 41-50 题。重点是投机解码、CUDA Graph、TP/EP、P/D 分离、量化与 benchmark。

[上一篇：KV Cache 与 GPU 执行细节](/posts/vllm-sglang-interview-level-4-kv-runner/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：Staff 级系统设计与现场推演](/posts/vllm-sglang-interview-level-6-system-design/)

---

## Level 5：高级优化、分布式与性能诊断

目标：确认候选人不仅懂机制，还能判断优化何时生效、何时变慢、怎样证明。

### 41. Speculative Decoding 为什么可以减少 target model iteration，又不改变目标分布？

- **递进追问**：greedy acceptance 与 stochastic rejection sampling 分别怎样保证正确？为什么“draft sample 等于 target sample才接受”在随机采样下是错的？
- **面试官观察点**：是否能同时讲清性能动机和分布正确性，而不是只背“draft 多猜几个 token”。
- **源码锚点**：vLLM rejection sampler；SGLang `speculative/reject_sampling.py`；两边 Spec 模块的 acceptance 章节。

**回答思路**：先把普通Decode的串行target iterations作为瓶颈，再说明draft只提出候选、target一次并行verify决定唯一可提交前缀，最后分别证明greedy和stochastic正确性。

**详细答案**：普通Decode每次target forward通常只提交一个token。Spec Decode先用更便宜的draft model、EAGLE/MTP或n-gram提出K个候选，target将这些positions组成宽query一次计算条件分布。若平均接受A个draft tokens，并再提交一个target/bonus token，一次昂贵target iteration平均前进 `A+1` 个tokens，因此可能降低TPOT。

Greedy时沿draft chain/tree从root逐位置比较target argmax，接受连续相同前缀；首个mismatch提交target token，后续候选因条件上下文已变必须全部作废。随机采样时draft分布为 `q`、target为 `p`，对draft token `d` 以 `min(1,p(d)/q(d))` 接受；拒绝后从归一化残差 `max(p-q,0)`采样替代，才能保持最终边缘分布等于target。只比较两个独立sample是否相同会引入偏差。任何未接受KV/hidden都不能进入committed/cache path。

**主问题得分点（10 分）**：

- 2 分：说明便宜proposal + 一次宽target verify减少串行iterations。
- 2 分：正确描述greedy连续前缀和mismatch。
- 3 分：给出stochastic acceptance与残差采样。
- 1 分：解释独立sample相等规则错误。
- 2 分：说明accepted path和未接受state的commit边界。

**追问与答案（5 分）**：为什么不能跳过一个mismatch后继续接受后面的draft token？后续draft概率以错误候选作为条件，而target真实路径已换成mismatch处的target token；条件分布不再对应。得分点：自回归条件依赖 3 分，连续accepted prefix不变量 2 分。

### 42. vLLM V2 speculative decoding 如何跨两轮维护 draft、verify、KV slots 和 token counts？

- **递进追问**：上一轮 proposal 如何进入下一轮 `SchedulerOutput`？reject 后 scheduled/computed/output token 数怎样修复？临时 lookahead slots 何时释放，哪些 token 才能进入共享 cache？
- **面试官观察点**：是否能以“proposal -> target verify -> commit/repair -> next proposal”描述跨轮 contract。
- **源码锚点**：`GPUModelRunner.sample_tokens()/postprocess_sampled()`、`Speculator.propose()`；[`vllm/modules/SPECULATIVE_DECODING.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/SPECULATIVE_DECODING.md)。

**回答思路**：以iteration N的target result和iteration N+1的verify input为主线，分别追踪token truth、computed count、draft metadata和临时KV ownership。

**详细答案**：本轮 `sample_tokens()` 对普通请求采一个token，对带scheduled draft的请求运行accept/rejection sampler，得到sampled IDs、accepted/rejected数量。runner先启动异步D2H，再由 `postprocess_sampled()` 用真实接受结果修复resident token arrays、penalty/sampler state、computed boundary和模型特定state。随后把target hidden/aux state与最新committed token交给 `Speculator.propose()`，产生下一轮draft IDs/metadata。

normal路径将proposal返回Scheduler，下一轮通过 `SchedulerOutput.scheduled_spec_decode_tokens` 再送runner；async路径还要用placeholder/in-flight保证pipeline位置。下一轮 `prepare_inputs()` 展开“最新committed token + draft positions”，Scheduler预留lookahead slots，target一次verify。若draft共K个、接受A个，未接受 `K-A` 对应的乐观computed progress必须回退；临时slots可覆盖/释放，但不能hash成共享cache。D2H、proposal和旧tensor生命周期通过event/future配对。

**主问题得分点（10 分）**：

- 2 分：完整描述verify/sample到下一proposal的跨轮顺序。
- 2 分：说明normal/async proposal如何进入下一plan。
- 2 分：说明packed verify input与lookahead slots。
- 2 分：说明reject后的computed/token/KV修复。
- 2 分：覆盖D2H overlap和plan/result lifetime。

**追问与答案（5 分）**：为什么proposal通常在sample D2H完成前就可启动？proposal需要GPU上的accepted/target state，runner可在device侧postprocess后运行drafter；host只需最终IDs用于Scheduler/API。先发起copy可与drafter部分重叠，但所有consumer必须等待各自event。得分点：device依赖 2 分，重叠机会 2 分，event正确性 1 分。

### 43. SGLang EAGLE V2 的 Draft、Target Verify、Draft Extend 为什么缺一不可？

- **递进追问**：tree proposal 如何组织？accepted path 为什么要 compact？Draft Extend 为下一轮修复了哪一份 hidden/token/KV truth？grammar mask 怎样与 tree node 对齐？
- **面试官观察点**：是否理解 target 与 drafter 是两套相关但不同的状态机，以及 accepted path 是唯一可提交事实。
- **源码锚点**：`EAGLEWorkerV2.forward_batch_generation()/verify()`、`eagle_info.py`；[`sglang/modules/SPECULATIVE_DECODING.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/SPECULATIVE_DECODING.md)。

**回答思路**：把target与drafter视为两套状态机，说明Draft产生候选树、Verify确定target truth、Draft Extend让drafter重新对齐该truth。

**详细答案**：Draft阶段消费上轮bonus/committed token、target hidden和draft KV，多步运行EAGLE drafter。`topk=1`形成chain，`topk>1`形成共享parent的tree；输出 `EagleVerifyInput`，显式携带draft tokens、scores/probs、parent/retrieve indices、tree mask和positions。Target Verify构造宽 `ForwardBatch`，target forward后 `eagle_sample()` 得到 `accept_lens/accept_index` 与predict，唯一可提交的是从root开始的accepted path。

tree nodes在临时layout中不连续，`_finalize_accept_tree_path()` 按同一accept index compact target KV、tokens和hidden到每请求canonical前部，释放overshoot。此时drafter先前展开的是整棵候选树，并不等于target accepted chain；Draft Extend使用target accepted hidden/token推进或重建draft KV，生成下一轮 `EagleDraftInput`。缺少它会让下一轮proposal基于stale draft state。Grammar mask还必须按每个tree node真实history重新生成。

**主问题得分点（10 分）**：

- 2 分：Draft输入、tree/chain和显式索引输出。
- 2 分：Verify由target确定accept length/index。
- 2 分：解释tree accepted path为何compact。
- 2 分：解释Draft Extend对齐drafter state。
- 2 分：覆盖grammar和overshoot/committed边界。

**追问与答案（5 分）**：为何token、KV和hidden必须用同一 `accept_index` compact？它们共同描述同一accepted path；任一使用不同索引都会让后续token历史、attention state和drafter feature错位，产生静默错误。得分点：三类state一致 3 分，静默错误后果 2 分。

### 44. 接受率很高时，Spec Decode 为什么仍可能比普通 Decode 慢？

- **递进追问**：请写出包含 proposal、宽 verify、draft-extend、tree/compaction、D2H、graph fallback 和额外 KV 的完整分母；并发升高时结论为何可能反转？
- **面试官观察点**：能否从 accepted tokens/iteration 进展到单位 committed token 成本和服务 goodput。
- **源码锚点**：两边 Spec 模块“性能模型/如何验证”；完整答案指南第 18 题。

**回答思路**：不要用acceptance rate直接推speedup，要写“每次verify提交多少token”除以“proposal到commit的完整阶段成本”，再加入并发容量影响。

**详细答案**：设K为draft宽度，A为平均接受draft数，`L=A+1` 为每次target verify平均提交tokens。普通target decode成本为 `C_t`，spec一次循环包括proposal `C_d(K)`、宽target verify `C_v(K)`、draft extend、tree/compact/rejection sampling/D2H/通信等 `C_o(K)`，近似：

```text
speedup ~= L * C_t / (C_d(K) + C_v(K) + C_o(K))
```

高接受率只增大分子。drafter过重、verify shape导致graph miss或padding、tree sampling/compact昂贵、TP通信增加、临时lookahead KV压低最大并发，都可能让分母或服务容量损失更大。高并发时普通Decode batch已能高效摊薄权重/launch，减少iteration的边际收益下降；单请求加速也可能不转化为SLO goodput。应报告accept-length histogram、每阶段GPU/CPU时间、target forwards/output token、graph hit、临时KV和不同并发下TPOT/goodput。

**主问题得分点（10 分）**：

- 3 分：写出包含L和完整阶段成本的模型。
- 2 分：区分acceptance rate与accept length。
- 2 分：列出graph/padding/communication/overhead退化。
- 1 分：说明临时KV降低并发。
- 2 分：说明并发改变基线并给出分阶段指标。

**追问与答案（5 分）**：如何选择K？逐K测量或在线预测 `cost_per_committed_token=(C_d+C_v+C_o)/L`，同时约束KV headroom与graph覆盖；选择低于普通 `C_t` 且goodput最高的K，高负载可降K或关闭。得分点：正确目标 2 分，资源/graph约束 2 分，动态回退 1 分。

### 45. 如何在 eager、piecewise graph 和 full CUDA Graph 之间做运行时选择？

- **递进追问**：哪些动态特征会破坏 capture？graph hit 率高却不加速有哪些可能？如何评估 padding work、capture memory、shape explosion 和 compile warmup？
- **面试官观察点**：是否会同时看 host gap、kernel time、命中/回退和显存，而不是只看“graph enabled”。
- **源码锚点**：vLLM `BatchExecutionDescriptor`；SGLang graph runner 与各 forward stage；完整答案指南第 16 题。

**回答思路**：为一个iteration建立feature/shape descriptor，用它先判断正确性可capture，再比较full、piecewise和eager的估计成本，而不是全局固定开关。

**详细答案**：Eager支持任意动态shape、Python控制流和backend，fallback最稳但host launch最大。Full graph要求整个forward路径、buffer地址和descriptor落在已capture集合，replay开销最低；piecewise只capture稳定子图，把动态sampling、某些attention/backend或graph break留在eager，收益与灵活性居中。vLLM的descriptor综合batch/token shape和功能选择，SGLang对Decode、EAGLE draft/verify/extend等阶段分别维护runner。

运行时先检查模型/backend、LoRA、grammar、MM、PP/DP、spec stage等兼容性，再找最小不小于真实shape的capture size并估算padding。如果padding FLOPs、capture显存或罕见shape编译成本高于host gap，应选择piecewise/eager。必须预热capture，记录按descriptor的hit/fallback、padding、replay和非图阶段时间；动态特征变化时安全fallback，不能复用错误graph state。

**主问题得分点（10 分）**：

- 2 分：三种模式及适用边界。
- 2 分：descriptor包含shape和动态feature兼容性。
- 2 分：说明padding/capture memory/shape explosion。
- 2 分：说明spec多阶段可有不同graph选择。
- 2 分：给出运行时决策、fallback和观测指标。

**追问与答案（5 分）**：为何不能为所有batch size精确capture一个graph？shape组合会乘上LoRA/spec/backend等feature形成爆炸，capture耗时与静态buffer显存增长；用有限bucket + padding通常更划算。得分点：组合爆炸 2 分，显存/warmup 2 分，bucket策略 1 分。

### 46. Tensor Parallel 从 1 卡扩到 8 卡后，为什么延迟和吞吐不一定线性改善？

- **递进追问**：请建立 compute、all-reduce/all-gather、launch、同步、拓扑和小矩阵效率的模型；跨节点时怎样选择 TP/PP？
- **面试官观察点**：能否写出扩展效率并用 profiler/collective trace 拆分，而非笼统归因 NCCL。
- **源码锚点**：两边 `distributed/communication_op.py`、device communicators、parallel state；完整答案指南第 17 题。

**回答思路**：写出单步时间由每卡compute、collective、launch/sync和串行部分组成，再用拓扑与矩阵效率解释非线性。

**详细答案**：理想情况下TP把线性层计算近似除以N，但每层的row/column parallel需要all-reduce、all-gather或reduce-scatter。单步可粗略写成：

```text
T(N) = T_compute(N) + T_collective(N, bytes, topology)
       + T_launch_sync + T_serial
```

N增大时每rank GEMM变小，Tensor Core效率可能下降；collective latency和同步不会按N缩小，小batch Decode又无法用足够计算隐藏通信。跨NVLink域或跨节点后带宽/latency阶跃恶化，straggler让所有ranks等待。吞吐还受batch/KV capacity、CPU dispatch和PP/DP组合影响。因此要报告speedup、parallel efficiency、每层collective timeline、link bandwidth、GEMM shape和端到端goodput。

**主问题得分点（10 分）**：

- 2 分：正确拆分compute/collective/serial时间。
- 2 分：说明小GEMM效率下降。
- 2 分：说明collective latency、同步和拓扑。
- 2 分：解释小batch Decode最难摊薄。
- 2 分：给出parallel efficiency与profiler验证。

**追问与答案（5 分）**：TP=8比TP=4慢，先查什么？固定请求shape比较每层GEMM与collective时间，确认是否跨互联域、算法选择/拓扑、rank straggler和CPU launch；再检查TP=8是否改变KV heads replication或graph shape。得分点：compute/comm拆分 2 分，拓扑/straggler 2 分，layout/graph 1 分。

### 47. Pipeline Parallel 中非末 rank 为什么也需要准确更新 token 和 sequence state？

- **递进追问**：sample 在哪里发生，结果如何广播给其他 stage？stage 间激活、microbatch、bubble 和 request order 如何保持一致？
- **面试官观察点**：是否理解只有末 rank 采样不代表其他 rank 可以缺少下一轮输入状态。
- **源码锚点**：vLLM `GPUModelRunner.update_pp_decode_requests()`；两边 PP group/broadcast 路径。

**回答思路**：沿一个sampled token从末stage返回下一iteration输入的闭环，说明所有stage虽不采样，却必须共享同一逻辑sequence进度。

**详细答案**：PP按层切stage，前stage输出activations给后stage，末stage计算logits并sampling。下一轮所有stage都要处理同一批request的新token/position：前stage需要token embedding和正确position，所有stage的本地KV block table与sequence length也必须推进。因此末rank的sampled/accepted IDs和finish/length信息要通过PP group广播或返回控制面，vLLM非末rank在 `update_pp_decode_requests()` 应用之前的sample结果。

若某stage少推进一个token、request order不同或提前释放KV，下一轮activation与local KV不再代表同一序列。microbatch流水可提高吞吐，但stage计算不均、少量microbatches和Decode串行依赖产生bubble；还要确保abort/preempt在所有stage一致。PP优化必须同时测stage时间、activation通信、bubble和端到端latency。

**主问题得分点（10 分）**：

- 2 分：末stage logits/sample的职责。
- 2 分：说明非末stage下一轮仍需token/position。
- 2 分：说明所有stage本地KV/seq state一致。
- 2 分：覆盖broadcast、request order、finish/preempt。
- 2 分：解释microbatch、stage imbalance和bubble。

**追问与答案（5 分）**：增加microbatch总能降低PP bubble吗？通常能提高流水填充，但会增加调度、通信、buffer与可能的单microbatch低效率，也受在线请求数量和latency SLO限制。得分点：填充收益 2 分，三类代价 2 分，SLO约束 1 分。

### 48. Prefill/Decode 分离系统中的 KV transfer 是什么协议问题，而不仅是一次 memcpy？

- **递进追问**：如何描述源/目的 slot、ready/completion、失败重算、backpressure、取消、超时和版本一致性？网络传输何时抵消分离收益？
- **面试官观察点**：能否把 ownership 和状态机放在带宽计算之前，并建立 prefill/decode/transfer 三阶段流水模型。
- **源码锚点**：vLLM `distributed/kv_transfer` 与 KV connector mixin；SGLang `srt/disaggregation`；完整答案指南第 19 题。

**回答思路**：把KV transfer建模为带ownership的分布式状态机：announce/allocate、write/read、completion、commit、abort/fallback，而不是只算网络带宽。

**详细答案**：Prefill节点先为prompt计算KV，Decode节点必须为同一model/version、request prefix和各layer/group建立目标slots。控制面要交换request/transfer ID、token boundary、source/destination rank和TP mapping；数据面通过NIXL/Mooncake/NCCL等搬运。Decode Scheduler只能在所有必需KV对目标worker可见且completion确认后把相应positions标为computed并运行，不能用“已发起”代替“已完成”。

取消、超时、源/目的故障需要幂等清理与fallback重算；destination slots在失败前后只能有一个owner，迟到completion不能提交到已复用request。transfer queue要有credits/backpressure，防止P端无限生产占满staging/Decode KV。收益条件是Prefill/Decode隔离节省的排队和设备适配大于KV bytes/网络带宽、setup和额外队列延迟；短prompt或慢网络可能更差。

**主问题得分点（10 分）**：

- 2 分：model/prefix/rank/slot metadata协议。
- 2 分：区分发起、完成可见和computed commit。
- 2 分：说明ownership、幂等与迟到completion。
- 2 分：覆盖backpressure、cancel、failure和recompute。
- 2 分：建立计算节省与transfer/queue成本模型。

**追问与答案（5 分）**：TP degree不同的P/D实例能直接按block ID传吗？通常不能。block ID是本地allocator身份，TP还改变head/shard layout；connector必须定义逻辑token/layer到source/destination shard的mapping并可能重排。得分点：本地ID不可移植 2 分，TP layout 2 分，显式mapping 1 分。

### 49. 线上出现“GPU utilization 低，但一个 CPU core 满载”，你怎样定位？

- **递进追问**：如何依次排查 tokenizer/detokenizer、Scheduler sort/hash、Python object churn、input packing、graph miss、IPC、小 batch 和同步点？每个假设需要什么证据？
- **面试官观察点**：是否坚持 timeline + CPU profile + batch/queue metrics 的证据链，并能把优化映射到 TTFT/ITL。
- **源码锚点**：两边 request、scheduler、runner 模块的“如何验证”；完整答案指南第 21 题。

**回答思路**：先用端到端timeline判断GPU空档是在请求前处理、两次kernel之间还是结果后处理，再用CPU profile和队列指标逐层缩小，不先猜某个Python函数。

**详细答案**：同时采集GPU timeline/NVTX、CPU sampling profile、各阶段queue depth和batch shape。若GPU长时间无work且waiting非空，检查Scheduler step、prefix hash/sort、grammar、input packing、IPC和锁；若waiting也空，检查HTTP/tokenization或arrival。若kernel很碎且间隔大，检查graph miss、CPU launch、TP同步和小batch。若GPU完成后stream迟到，检查D2H、detokenization、OutputProcessor、event-loop lag和socket backpressure。

CPU profile进一步定位单核热点：tokenizer/string、Radix match、Python object allocation/GC、metadata construction、serialization或busy polling。每个优化都要做对照：增大batch/启graph、缓存tokenization、换policy、批量IPC、绑核/多进程等，并观察TTFT/ITL而非只看CPU下降。也要排除GPU utilization统计窗口和memory-stall kernel造成的假象。

**主问题得分点（10 分）**：

- 2 分：timeline + CPU profile + queue/batch三类证据。
- 2 分：按请求前、iteration间、结果后分段。
- 2 分：覆盖Scheduler/hash/packing/IPC/graph。
- 2 分：覆盖detokenize/event loop/backpressure。
- 2 分：提出单变量对照并映射TTFT/ITL。

**追问与答案（5 分）**：waiting queue很长但GPU仍有周期性空洞，最可能说明什么？供给不缺，问题在Scheduler/CPU准备、同步或graph fallback；应对齐每个gap前最后CPU span和CUDA API。也可能batch因硬约束不可执行，需看skip reason。得分点：排除arrival 1 分，CPU/sync定位 2 分，硬约束skip 2 分。

### 50. 你会如何公平比较 vLLM 与 SGLang，而不是跑一个 tokens/s 数字？

- **递进追问**：怎样固定模型、精度、采样、prompt/output 分布、并发/arrival process、SLO、warmup、graph、cache、并行和错误率？如何分别报告 TTFT、ITL、goodput、VRAM 与稳定性？
- **面试官观察点**：是否能设计可复现、可归因、覆盖 steady-state 与 overload 的 benchmark matrix。
- **源码锚点**：[`VLLM_VS_SGLANG.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/VLLM_VS_SGLANG.md)；完整答案指南第 22、23 题。

**回答思路**：先固定语义和资源，再用同一workload发生器覆盖负载曲线，报告SLO、容量与稳定性，并对每个功能开关做消融。

**详细答案**：两边使用同一模型checkpoint、revision、dtype/quant、max length、sampling/seed、chat template、GPU/驱动和TP/PP拓扑；确认输出语义、stop、logprob和错误率一致。workload保存真实prompt/output长度、prefix共享比例、structured/MM比例，使用相同open-loop arrival process和并发阶梯，而不是只跑固定并发闭环。分别预热compile/graph/cache，并明确冷/热cache测试。

报告p50/p95/p99 TTFT、ITL/TPOT、E2E、request/output-token throughput、SLO goodput、GPU/CPU/VRAM、preemption/retraction、graph hit和失败率。持续运行检查OOM、RSS/VRAM漂移和尾延迟。随后做prefix、chunk、graph、spec等单变量消融，并公开配置、commit、容器与原始trace。只有在相同SLO下比较最大稳定goodput，才能避免用大batch平均tokens/s掩盖用户延迟。

**主问题得分点（10 分）**：

- 2 分：固定模型语义、精度、硬件与并行。
- 2 分：真实长度/prefix/arrival和负载曲线。
- 2 分：冷暖状态、预热与配置对齐。
- 2 分：延迟分位、goodput、资源和错误完整指标。
- 2 分：消融、长稳测试和可复现材料。

**追问与答案（5 分）**：为什么只用闭环固定并发可能误导？客户端会等请求完成再发下一个，服务变慢会自动降低arrival rate，隐藏过载排队；open-loop可控制到达率并观察稳定边界。闭环仍适合交互并发场景，但要明确。得分点：协调遗漏效应 2 分，open-loop意义 2 分，适用边界 1 分。

---

[上一篇：KV Cache 与 GPU 执行细节](/posts/vllm-sglang-interview-level-4-kv-runner/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：Staff 级系统设计与现场推演](/posts/vllm-sglang-interview-level-6-system-design/)
