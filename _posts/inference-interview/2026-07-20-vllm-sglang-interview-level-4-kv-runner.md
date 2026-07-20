---
title: 'vLLM / SGLang 面试题（四）：KV Cache 与 GPU 执行'
author: Cptz
date: 2026-07-20 00:14:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)第 4 卷，覆盖第 31-40 题。重点是Paged KV、Radix Cache、Beam Search、slot mapping、attention backend 与显存。

[上一篇：调度、准入与资源预算](/posts/vllm-sglang-interview-level-3-scheduling/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：高级优化、分布式与性能诊断](/posts/vllm-sglang-interview-level-5-advanced-distributed/)

---

## Level 4：KV Cache 与 GPU 执行细节

目标：确认候选人能从逻辑 token 追到物理 slot，并处理异步 GPU 执行中的生命周期问题。

### 31. vLLM 的 request block table、BlockPool、free queue、hash map 和 ref count 如何协作？

- **递进追问**：一个 finished 且 hashed 的 block 为什么可以同时在 free queue 和 hash map 中？cache hit 的 `touch()` 做什么？何时真正 eviction？
- **面试官观察点**：是否区分“无 owner、可覆盖”和“内容仍可命中”，并能说明 lazy eviction。
- **源码锚点**：`KVCacheManager`、`SingleTypeKVCacheManager`、`BlockPool`。

**回答思路**：分别画逻辑ownership、内容索引和可回收顺序，重点解释 `ref_cnt==0` 不等于内容立刻失效。

**详细答案**：每个request的block table保存逻辑block到physical block ID的映射；`BlockPool`拥有固定数量physical blocks及其 `ref_cnt`、block hash、free queue位置。内容寻址hash map从block hash找到仍保存对应KV的physical block。新请求cache hit时，manager把命中blocks挂到request，`touch()` 将其从free queue移出并增加引用，防止被覆盖。

request结束或preempt时，`free_blocks()`递减引用；降到0的block加入free queue，成为可分配/可淘汰，但若有有效hash，其内容和hash map entry可保留。因此下一请求在它被覆盖前仍可命中。真正需要该physical block存新内容时，pool从free queue取出，移除旧hash身份、重置metadata并分配；引用大于0的block绝不能进入可覆盖集合。free queue顺序同时近似承担lazy eviction/LRU策略。

**主问题得分点（10 分）**：

- 2 分：request block table到physical ID映射。
- 2 分：ref count表示活跃ownership。
- 2 分：hash map表示内容身份而非ownership。
- 2 分：解释ref=0 block可同时在free queue和hash map。
- 2 分：说明touch、真正覆盖和旧hash清理顺序。

**追问与答案（5 分）**：一个ref=0 hashed block算“空闲”还是“cache占用”？从allocator角度是空闲、可立即覆盖；从cache角度内容仍有机会命中。指标应分别报告free/evictable、protected和cache-resident，不能把resident bytes都当不可分配。得分点：双重语义 2 分，覆盖优先级 1 分，正确指标 2 分。

### 32. vLLM 的 prefix block hash 为什么需要链式身份和额外命名空间？

- **递进追问**：如果只 hash 当前 block token 会发生什么？cache salt、LoRA、multimodal input、adapter 或模型状态应怎样进入 key？哈希碰撞如何考虑？
- **面试官观察点**：是否把 cache key 看成“完整模型状态等价证明”，而非普通字符串缓存。
- **源码锚点**：request block hashing、`get_computed_blocks()`；[`vllm/modules/KV_CACHE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/KV_CACHE.md)。

**回答思路**：把每个block的hash当作从序列起点到该block结束的状态承诺，而不是只标识局部token片段。

**详细答案**：如果只hash当前block token，同一局部片段出现在不同前文时会错误命中，但Transformer KV依赖整个前缀。链式hash把上一block hash与当前block token等共同作为输入，使第 `i` 个block身份代表完整prefix到该边界。`get_computed_blocks()`只能从起点连续匹配，不能跳洞复用中间block。

身份还必须加入所有会改变KV数值的命名空间，例如cache salt/租户隔离、LoRA/adapter、multimodal feature或其稳定hash、模型/版本和必要的position/attention配置。实现可使用足够强digest或完整key equality降低碰撞风险；一旦错误命中，后果不是普通cache stale，而是静默生成错误结果。只有完整、已提交且满足group对齐的block可发布普通prefix hash。

**主问题得分点（10 分）**：

- 3 分：解释局部token hash为何错误及链式hash作用。
- 2 分：说明最长命中必须从序列起点连续。
- 2 分：列出LoRA/MM/salt/model等状态命名空间。
- 1 分：讨论碰撞与key验证。
- 2 分：说明完整/committed/group alignment发布边界。

**追问与答案（5 分）**：为什么tenant salt既是正确性问题也是安全问题？不同tenant可能token相同但adapter/权限/数据隔离要求不同；无salt的共享会泄露cache存在性/timing，错误namespace还可能复用不等价状态。得分点：模型状态正确性 2 分，侧信道/隔离 2 分，salt进入链式key 1 分。

### 33. 部分物理 block 命中后，为什么可能需要 Copy-on-Write？

- **递进追问**：source、destination、request block table 和 GPU copy 的生命周期是什么？为什么只 copy 冲突 block，而不是整条 KV？不支持 fine-grained hit 时如何退化？
- **面试官观察点**：能否识别共享尾块继续写造成的跨请求数据破坏，并给出最小复制范围。
- **源码锚点**：vLLM `_apply_cow()`、`KVCacheBlockCopy`、`copy_kv_cache_blocks_inplace()`。

**回答思路**：构造两个请求共享同一未满physical block且其中一个继续append的场景，找出最小需要私有化的写冲突范围。

**详细答案**：full-block prefix命中只共享不会再写的完整block，通常无需CoW。fine-grained/hybrid模式可能让请求命中physical block的一部分；若它直接在同一block剩余slots写新KV，会修改仍被其他request/cache identity共享的物理内容。manager因此记录partial hit的source block，为当前request分配private destination，将block table对应entry切到destination，并输出 `KVCacheBlockCopy(src,dst)`。

Runner在forward写入前调用GPU copy把source有效内容复制到destination，随后只向private block追加。source与destination及copy descriptor必须保活到copy event结束；失败时不能先释放source。复制范围是发生写冲突的单个physical block，不是整条prefix，因此成本受block size和partial hit频率影响。不支持该layout时应退化到较短的full-block hit并重算tail。

**主问题得分点（10 分）**：

- 3 分：准确构造共享尾块写冲突。
- 2 分：source/private destination/block table切换。
- 2 分：说明GPU copy必须在forward写入前完成。
- 1 分：说明只复制冲突block。
- 2 分：覆盖lifetime和不支持时的安全退化。

**追问与答案（5 分）**：CoW一定比重算tail快吗？不一定。copy读取/写入整block，若有效tail很短或copy破坏overlap，重算少量token可能更便宜；应比较copy bytes/kernel latency与tail forward cost，并考虑更多命中带来的后续共享。得分点：成本两端 2 分，workload因素 1 分，实验决策 2 分。

### 34. 当前 vLLM V1 做 beam search 时会不会立即 fork 出 K 份 KV Cache block？

- **递进追问**：每轮 beam 如何变成普通请求？完整 token prefix、APC、full-block 对齐和 fine-grained CoW 如何共同决定物理共享？关闭 prefix cache 会怎样？
- **面试官观察点**：是否基于当前源码回答“逻辑分叉 + 新 request + prefix sharing”，并主动区分已经移除的 V0 fork 设计。
- **源码锚点**：`entrypoints/generate/beam_search/online.py::BeamSearchOnlineMixin.beam_search()`；[`vllm/modules/KV_CACHE.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/KV_CACHE.md) 第 7 节。

**回答思路**：明确限定“当前V1在线beam实现”，先追API层每轮生成请求，再说明APC如何产生物理共享，最后与旧V0 fork区分。

**详细答案**：当前V1在线beam search不是Scheduler内部调用 `Sequence.fork()` 并立即复制K份block table/KV。`BeamSearchOnlineMixin.beam_search()` 在CPU保存每个存活beam的完整token IDs；每轮为各beam创建新的普通request ID，以该完整prefix调用 `generate(max_tokens=1, logprobs=...)`，再按累计logprob选择下一轮K条序列。

物理KV复用依赖automatic prefix caching。共同prefix的完整hashed blocks通过ref count共享，不会复制K份；未满尾block和分支suffix通常各自重算/写private blocks。若开启支持的fine-grained partial hit，只有共享尾块发生写冲突时才可能做单block CoW。关闭APC不改变beam正确性，但每轮新request会重新计算更长上下文，性能显著退化。旧V0的block-table fork + CoW是另一套已移除控制流，不能套用到当前源码。

**主问题得分点（10 分）**：

- 3 分：明确回答不会在分叉点eager复制K份KV。
- 2 分：说明CPU beam列表、新request ID和单token generate循环。
- 2 分：说明APC按完整block/ref count共享。
- 1 分：说明尾部private/recompute与可选单block CoW。
- 2 分：区分APC关闭和旧V0实现。

**追问与答案（5 分）**：block size=16、公共prefix=100、beam width=4时，普通full-block cache如何共享？可稳定共享前96 tokens的6个完整blocks；剩余4-token tail及新分支通常各request私有/重算。不能直接说共享100 tokens。得分点：96/6计算 2 分，tail处理 2 分，ref共享非复制 1 分。

### 35. SGLang 为什么同时需要 `ReqToTokenPool` 和 Token-to-KV pool？

- **递进追问**：request row 与 physical KV slot 的生命周期为什么不同？attention 如何从 request/token position 间接定位 KV？只用一个 allocator 会混淆哪些 ownership？
- **面试官观察点**：能否画出 `Req -> row -> slot index -> K/V tensor` 的两级映射。
- **源码锚点**：`mem_cache/memory_pool.py`、`ScheduleBatch.prepare_for_extend/decode()`。

**回答思路**：画出两级地址翻译：`Req -> req_pool_idx row -> token position对应physical slot -> K/V tensor`，再比较两种资源的生命周期。

**详细答案**：`ReqToTokenPool` 分配一行request mapping，行内每个逻辑token位置存其physical KV slot index。它解决“给定请求和position，attention去哪里找KV”；row数量限制同时resident的requests。Token-to-KV pool/allocator管理真正的K/V tensors和空闲physical slots，容量按总tokens而不是request数计算，可支持MHA/MLA、page-major、量化或hybrid layout。

一个request可能释放private slots但保留prefix tree中的canonical slots；finished后request row可释放，而cached KV仍驻留。反之，有free KV slots也可能因无request row不能准入。`ScheduleBatch.prepare_for_extend/decode()` 必须先获得row并把本轮分配slots写入mapping，`ForwardBatch`/attention才可消费。合并两个allocator会混淆row ownership与cache slot ownership，容易double-free或泄漏。

**主问题得分点（10 分）**：

- 3 分：准确画出两级映射。
- 2 分：区分row capacity与token capacity。
- 2 分：说明row和cached KV生命周期不同。
- 2 分：说明prepare阶段写mapping给attention。
- 1 分：指出不同KV tensor layout由第二层隐藏。

**追问与答案（5 分）**：为什么有大量free KV仍可能无法准入？`ReqToTokenPool` row耗尽、LoRA/grammar/PD未ready、token budget或max running限制都可能先到。得分点：row限制 2 分，另外两个约束 2 分，多资源结论 1 分。

### 36. SGLang 把 unfinished/finished request 放入 Radix Cache 时为什么需要 canonicalization？

- **递进追问**：insert 后为何可能同时存在 cached canonical slots 与 request private duplicate slots？哪些 slots、locks 和 rows 在 `cache_unfinished_req()`、`cache_finished_req()`、外层 release 中释放？
- **面试官观察点**：是否能追踪 duplicate physical ownership，避免 double-free、泄漏和错误共享。
- **源码锚点**：`RadixCache.cache_unfinished_req()`、`cache_finished_req()`、`release_kv_cache()`。

**回答思路**：区分request当前private路径与Radix tree中canonical路径，分析insert遇到已存在prefix时哪些physical slots重复。

**详细答案**：request执行时，其row可能指向新分配private slots。把已committed token路径插入Radix tree时，树中某些prefix可能已由其他请求插入，insert返回的canonical slots与该request对应private slots不同，但语义内容相同。如果两份都保留，会浪费KV；如果释放错一份，会破坏其他引用。

`cache_unfinished_req()` 将稳定部分插入树，释放duplicate private indices，更新request row指向canonical indices，并调整last node/lock，使下一chunk或Decode继续使用唯一物理路径。`cache_finished_req()` 插入可缓存committed path并处理duplicate、未对齐和private tail；外层 `release_kv_cache()` 再释放spec overshoot/Mamba等额外状态，最后释放request row。`kv_allocated_len >= kv_committed_len`，未提交位置绝不能发布到共享树，所有ownership只能释放一次。

**主问题得分点（10 分）**：

- 3 分：解释duplicate slots如何由并发/已有tree path产生。
- 2 分：说明canonicalize后row改指canonical slots。
- 2 分：区分unfinished继续持有lock与finished发布/释放。
- 2 分：说明外层再释放overshoot/state和request row。
- 1 分：给出allocated/committed不变量。

**追问与答案（5 分）**：为什么request row不应在 `cache_finished_req()` 开头释放？方法仍需读取row中的slot mapping完成insert、duplicate比较和tail处理；过早释放可能被新request覆盖。得分点：依赖mapping 2 分，复用竞态 2 分，正确外层释放顺序 1 分。

### 37. Page alignment、滑窗注意力、Mamba state 等混合模型怎样改变 cache 复用？

- **递进追问**：为什么 full-attention 的最长 token prefix 不能无条件外推到所有 KV group？不同 group block size、可见窗口和 recurrent state 如何共同限制 shared prefix？
- **面试官观察点**：是否知道“token 命中长度”最终必须映射为每种状态都安全的物理边界。
- **源码锚点**：vLLM `KVCacheSpec/KVCacheGroupSpec`、fine-grained manager；SGLang hybrid memory pool 与 Radix page alignment。

**回答思路**：把“相同token prefix”拆成每个state group各自可复用的物理边界，最终共享长度取所有必要状态的安全交集。

**详细答案**：full attention每个新token依赖全部历史KV，通常按完整block/page发布和匹配。Sliding-window attention只需最近窗口，可提前释放旧blocks，但其physical mapping与full attention不同；Mamba/recurrent layer保存压缩state而非逐token KV，state只在特定chunk/boundary可安全复用。混合模型还可能让不同group拥有不同block size、page alignment和retention规则。

因此逻辑token最长公共prefix不能直接作为所有group共同hit。vLLM的KV cache specs/groups与single-type managers分别计算命中并取可安全共享边界，fine-grained模式必要时CoW；SGLang分别核算full/SWA/Mamba pools、alignment和gap reserve。错误地按full-attention规则外推会让某个group读取缺失或来自错误边界的state。

**主问题得分点（10 分）**：

- 2 分：full attention block语义。
- 2 分：SWA窗口回收与映射差异。
- 2 分：Mamba/recurrent state边界差异。
- 2 分：说明多group安全交集与alignment。
- 2 分：映射到两框架hybrid manager/ledger与错误后果。

**追问与答案（5 分）**：为什么取各group命中长度的最大值通常不安全？target forward需要所有相关层在同一position前都有正确state；任一group缺失都要从更早位置重算，因此通常受最短安全边界限制，除非实现能对group做独立计算计划。得分点：跨层一致性 2 分，最短边界 2 分，独立计划例外 1 分。

### 38. Ragged batch 如何转换成模型和 attention kernel 可消费的 packed tensors？

- **递进追问**：input IDs、positions、`query_start_loc`、sequence lengths、block tables、slot mapping 和 logits locations 各解决什么问题？padding 位置如何防止写入真实 KV？
- **面试官观察点**：能否从 Scheduler 的 per-request map 追到连续 tensor view 和 attention metadata。
- **源码锚点**：vLLM `GPUModelRunner.prepare_inputs()/prepare_attn()`；SGLang `ForwardBatch.init_new()`。

**回答思路**：从不同请求本轮token数不相等出发，把嵌套序列flatten，并为kernel提供恢复边界、位置和物理KV地址的metadata。

**详细答案**：Runner按本轮request order把各请求要计算的token IDs和positions拼成一维packed query。`query_start_loc`/prefix sums标识每个请求在packed tensor中的起止；sequence/context lengths告诉attention可见历史范围；block table或req-to-token row给出逻辑历史到physical KV pages；slot mapping给出本轮每个query写入哪个physical location；logits locations只选择需要采样/返回的hidden positions，避免为所有Prefill tokens做大词表projection。

vLLM从resident arrays和 `num_scheduled_tokens` 构造这些views，SGLang由 `ForwardBatch.init_new()` 消费 `ScheduleBatch`。graph padding/idle entries必须有显式mask和dummy slots，既不能读取越界，也不能把padding写成真实request KV或提交token。所有metadata必须与request order和TP/PP ranks一致。

**主问题得分点（10 分）**：

- 2 分：说明flatten/packed query和prefix sums。
- 2 分：IDs、positions、seq/context lengths作用。
- 2 分：block table与slot mapping读写KV作用。
- 2 分：selective logits减少工作。
- 2 分：padding和跨rank一致性不变量。

**追问与答案（5 分）**：为什么Prefill通常只需最后或指定positions的logits？生成下一token只依赖每个序列最后有效position；若API请求prompt logprobs等才需更多positions。selective gather可减少vocab GEMM和中间tensor。得分点：生成语义 2 分，例外 1 分，性能收益 2 分。

### 39. vLLM ModelRunner 为什么维护 resident request state，并采用 staged block-table writes？

- **递进追问**：每轮全量重建 Python/tensor 状态有什么成本？add/update/finish/preempt 的顺序为什么重要？异步 output 尚未完成时哪些 tensor 不能复用？
- **面试官观察点**：是否理解 CPU preparation、稳定 GPU buffer、增量更新和 lifetime safety 的共同目标。
- **源码锚点**：`GPUModelRunner.execute_model()`、`InputBatch`、`block_tables.apply_staged_writes()`；[`vllm/modules/MODEL_EXECUTION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/vllm/modules/MODEL_EXECUTION.md)。

**回答思路**：比较每轮重建全部request tensor与维护稳定slot的CPU/graph代价，再说明批量提交block table更新为何能维持一致快照。

**详细答案**：continuous batching每轮大部分requests不变。若每次从Python `Request` 全量构造sampling、LoRA、token和block table tensors，会产生对象遍历、allocation、H2D和地址变化，破坏CUDA Graph稳定性。ModelRunner因此为active request维护resident index和host/device buffers：先应用PP sample，移除finished/preempted与附属state，再add新request、update已有request的computed count/new blocks，最后准备inputs。

block table变更先staged，待本轮所有add/update/remove完成后 `apply_staged_writes()` 一次性提交，避免attention metadata看到半更新表，也便于合并H2D。异步sample/D2H和future仍引用旧tensor时，buffer不能立即覆盖；runner用events、keep-alive和固定buffer窗口管理last use。收益是降低Scheduler-to-GPU CPU gap、稳定地址和graph replay，代价是复杂增量一致性。

**主问题得分点（10 分）**：

- 2 分：解释全量重建的CPU/H2D/地址成本。
- 2 分：说明resident request index与稳定buffers。
- 2 分：正确给出finish/add/update顺序意义。
- 2 分：说明staged writes避免半更新snapshot。
- 2 分：覆盖async tensor lifetime和graph收益。

**追问与答案（5 分）**：为什么先add再finish可能出错？resident slot看似已满，无法复用本轮刚结束请求的位置；更严重时新请求可能写入仍被旧request metadata引用的slot。应先确认旧device work结束并finish/free，再分配新resident index。得分点：容量 1 分，旧引用竞态 2 分，正确顺序/lifetime 2 分。

### 40. SGLang overlap event loop 为什么比 normal loop 难很多？

- **递进追问**：launch snapshot、`FutureMap`、两轮 tensor lifetime、copy stream 和 WAR barrier 分别防止什么错误？为什么 `batch.copy()` 不是可选优化？
- **面试官观察点**：能否说明 overlap 的真实收益来自 CPU/GPU/D2H 重叠，同时列出 stale state、buffer overwrite 和 use-after-free 风险。
- **源码锚点**：`Scheduler.event_loop_overlap()`、`record_batch_in_overlap()`、`_apply_war_barrier()`；[`sglang/modules/MODEL_EXECUTION.md`](https://github.com/Cptzzzzz/infra-codes/blob/1f8930ce3f206cdaac9ec24954f5a428e8c890f8/inference/sglang/modules/MODEL_EXECUTION.md)。

**回答思路**：先画normal串行timeline，再把CPU commit/next-plan、GPU forward和D2H并行展开，逐一识别跨iteration的RAW/WAR与对象变异风险。

**详细答案**：normal loop按“取batch -> forward/sample -> 等结果 -> 更新Req/cache -> 下一轮”串行，容易在GPU前后留下CPU空档。overlap loop让当前GPU/D2H运行时，CPU处理上一轮result并准备下一轮；`FutureMap` 可发布/解析设备侧next-token和seq length，减少host round-trip，copy stream与forward stream重叠输出传输。

复杂性来自状态跨轮。result必须绑定launch时 `ScheduleBatch` snapshot，所以入queue前 `batch.copy()` 并保活tensor；否则后续filter/merge会改变order。下一轮schedule stream覆盖共享input buffer前，必须等待上一轮最后读者，`_apply_war_barrier()` 优先等待精确read-done event，缺失时退化为stream wait。async D2H event完成前CPU不能读取，FutureMap key在filter后仍须对应正确rid，keep-alive通常覆盖两轮。收益要以timeline中CPU/GPU/D2H重叠和ITL改善证明。

**主问题得分点（10 分）**：

- 2 分：画出normal与overlap timeline差异。
- 2 分：说明batch snapshot防止request/result错配。
- 2 分：说明FutureMap减少host依赖。
- 2 分：解释WAR barrier和精确event/fallback。
- 2 分：覆盖D2H/两轮tensor lifetime及性能验证。

**追问与答案（5 分）**：日志显示Step N顺序变化是否证明执行错序？不一定。不同CPU线程、CUDA streams和异步flush的日志输出顺序不等于依赖完成顺序。应检查event依赖、iteration ID、batch snapshot和NVTX timeline。得分点：区分日志与因果 2 分，正确关联键 1 分，event/timeline证据 2 分。

---

[上一篇：调度、准入与资源预算](/posts/vllm-sglang-interview-level-3-scheduling/) · [系列总索引](/posts/vllm-sglang-interview-series/) · [下一篇：高级优化、分布式与性能诊断](/posts/vllm-sglang-interview-level-5-advanced-distributed/)
