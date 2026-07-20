---
title: 'vLLM / SGLang 面试源码深挖：优化机制与实验'
author: Cptz
date: 2026-07-20 00:18:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit 1f8930ce3f206cdaac9ec24954f5a428e8c890f8 by tools/sync_inference_interview_posts.py. -->

> 本文是[面试题系列](/posts/vllm-sglang-interview-series/)的源码实验卷。它不重复罗列术语，而是用固定版本源码和 A100 实验回答可以被继续追问的工程细节。

---

本文不是框架功能清单，而是一组可以被继续追问的工程问题。每节固定回答：模块为什么存在、Input
是什么、状态如何变化、Output 给谁、优化改变哪个性能维度、代价是什么，以及当前源码和实测结果是否
支持这个判断。

实验日期为 2026-07-17，环境是单张 A100 PCIe 40 GB 和 `Qwen/Qwen3-0.6B`。固定源码：

- vLLM [`d1d2f7535`](https://github.com/Cptzzzzz/vllm/commit/d1d2f7535686acdc39b034bf18ba52d07c21def5)
- SGLang [`9529c2e96`](https://github.com/Cptzzzzz/sglang/commit/9529c2e96351f3b1d5715eaa89989b20723d60b5)
- 聚合实验数据：[`experiments/2026-07-17-a100-qwen3-0.6b.json`](/assets/files/vllm-sglang-interview/2026-07-17-a100-qwen3-0.6b.json)

单卡小模型数据用来验证机制和因果，不应被当成生产模型排行榜。尤其是 graph capture bucket、attention
backend、sampling backend 并不完全相同，绝对数字不能直接用于框架选型。

## 1. 深入回答的统一方法

一道推理框架问题至少要讲清五层：

1. **语义层**：请求最终承诺什么，例如 target sampling 分布不变、stop/cancel 生效。
2. **逻辑状态层**：prompt、generated token、computed token、committed token 分别走到哪里。
3. **物理资源层**：KV page/block、request row、graph buffer、CUDA stream 由谁拥有。
4. **执行层**：Scheduler 生成什么计划，Runner launch 什么，结果何时 commit。
5. **性能层**：减少的是 FLOPs、HBM bytes、CPU launch、同步、排队还是碎片；代价落在哪个指标。

高分回答不是罗列 `Scheduler`、`PagedAttention`、`RadixCache`，而是给出 contract：

```text
Input + 当前状态
  -> admission / schedule plan
  -> allocate physical resources
  -> execute without changing user-visible truth
  -> validate/sample
  -> commit accepted result
  -> release or cache remaining resources
  -> Output to the next owner
```

下面每节同时给出主问题、追问、完整回答和评分点，可直接用于面试训练。

---

## 2. Prefill 有 causal mask，为什么仍称 O(N^2)

### 主问题

Prefill 的 attention 有 causal mask，只计算下三角，是否意味着计算量不再是 `N^2`？FlashAttention
又改变了什么？

### 要解决的问题

长度为 `N` 的 prompt 中，第 `i` 个 query 只能读取 `0..i` 的 K/V。框架既要保证因果正确性，又要让
GPU 以高吞吐的 tile/GEMM 方式执行，且不能显式保存巨大的 `N x N` score matrix。

### Input -> 方法 -> Output

| 项目 | 内容 |
| --- | --- |
| Input | Q/K/V、每个 sequence 的长度、position、causal/window mask 信息 |
| 方法 | causal tile pruning、online softmax、Q/K/V 和 MLP 的批量 GEMM |
| Output | 每个 prompt position 的 attention output，以及写入 paged KV 的 K/V |
| 下游 | 最后位置 logits 用于首 token；全部 K/V 被后续 decode 读取 |

严格的有效 attention pair 数是：

```text
1 + 2 + ... + N = N(N + 1) / 2
```

相对没有 mask 的 `N^2`，主项常数大约减半，但渐进复杂度仍是 `Theta(N^2)`。因此“少算约一半”是对
pair count 的描述，“不再是 N 平方”则不正确。

FlashAttention 的主要收益不是把 dense causal attention 的理论复杂度变成线性，而是：

- 只在 SRAM/register 中分块保存 score，避免把 `N x N` score/probability 写回 HBM；
- 用 online softmax 合并 tile，减少 HBM traffic；
- 对完全位于 causal 上三角的 tile 直接跳过；边界 tile 用 mask；
- 让矩阵计算保持足够大的 tile，提高 Tensor Core 利用率。

总层时间也不等于 attention pair count。QKV projection、output projection、MLP 对 token 数近似线性；
短序列时 launch、Python/IPC、采样和低占用率占比较大。长序列才逐渐由 attention 主导。

### 源码结合

- vLLM 的 Runner 将 query positions、KV slot mapping 和 attention metadata 交给所选 backend，核心入口见
  [`gpu_model_runner.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/worker/gpu_model_runner.py)。
- SGLang 在 `ForwardBatch` 中区分 EXTEND/DECODE/TARGET_VERIFY，并由 backend 消费 sequence length、positions
  和 page table，见
  [`forward_batch_info.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/model_executor/forward_batch_info.py)
  与 [`flashattention_backend.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/layers/attention/flashattention_backend.py)。

### 实验

并发 1、只生成 1 token，使用唯一随机 prompt，TTFT 如下：

| 框架/模式 | 256 | 1024 | 4096 |
| --- | ---: | ---: | ---: |
| vLLM eager | 41.66 ms | 60.23 ms | 114.43 ms |
| SGLang eager | 61.42 ms | 73.02 ms | 105.07 ms |
| SGLang graph | 24.27 ms | 37.56 ms | 105.45 ms |

长度从 256 增加到 4096 是 16 倍，端到端 TTFT 只增加约 1.7 到 4.3 倍。这不否认 attention 的
`Theta(N^2)`：小模型、线性层、固定开销、tile 利用率和 graph launch reduction 混在端到端指标里。
更有辨识度的现象是 SGLang graph 在 256/1024 明显减少 launch 开销，到 4096 与 eager 几乎相同，说明
长 prefill 已转为计算主导。

### 得分点（10 分）

- 2 分：给出 `N(N+1)/2`，并说明仍是 `Theta(N^2)`。
- 2 分：区分 FLOPs 减半与复杂度阶数改变。
- 2 分：说明 FlashAttention 主要优化 memory IO，并不普遍改变理论阶数。
- 2 分：把 QKV/MLP、launch 和硬件利用率纳入端到端模型。
- 2 分：提出用 kernel trace/FLOP counter、不同长度和 warm graph 实验验证，而不只看一次 TTFT。

### 追问

**为什么 profiler 统计的 FLOPs 可能不是精确一半？** Kernel 可能以固定 tile 计算 causal 边界、padding
或 graph bucket 中的无效位置；不同工具统计 executed instruction、理论 GEMM FLOPs 或有效 FLOPs，口径不同。

**什么时候 attention 近似线性？** Sliding window 将每个 query 的可见 K/V 限为 `W` 时是
`O(NW)`；真正的线性 attention/SSM 是不同模型算法，不是给 dense attention 加 causal mask。

---

## 3. Prefix cache 命中了 99%，为什么 TTFT 可能只快一点

### 主问题

Prefix caching 的 Input/Output 是什么？vLLM 与 SGLang 的匹配粒度为何不同？命中率是否能直接预测
TTFT speedup？

### 要解决的问题

多个请求共享 system prompt、few-shot 或对话历史时，重复 prefill 会浪费 attention/MLP 计算。框架需要
从 token prefix 找到仍有效的物理 KV，并确保 eviction、并发引用和尾部写入不会破坏它。

### vLLM contract

| 阶段 | Input | 方法 | Output |
| --- | --- | --- | --- |
| 查找 | token IDs、cache salt/extra keys、KV block size | 对完整 block 建链式 hash，在 `BlockPool` 查最长连续 hit | cached block IDs、`num_computed_tokens` |
| 分配 | 已命中 blocks、本轮 scheduled tokens、free queue | 给缺失 token 分配新 blocks，必要时 evict 无引用 cached blocks | request block table |
| 执行 | block table、slot mapping、未命中 suffix | 只 forward suffix，写新 KV | 新 K/V 和 logits |
| 提交 | 已完成完整 blocks、request finish | cache full blocks；释放 request 引用但可保留 cache 身份 | 下一个请求可复用的 blocks |

核心源码是
[`kv_cache_manager.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/kv_cache_manager.py)
和 [`block_pool.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/block_pool.py)。
默认实验 block size 为 16，所以 2080-token 重复输入命中 2064，尾部一个 block 边界必须重新 forward。

### SGLang contract

SGLang `RadixCache` 以 token sequence 的 radix edge 做逻辑匹配，value 指向 token-to-KV pool slots。默认实验
`page_size=1`，相同输入命中 2079 token。最后一个 token 仍需运行，因为 causal LM 需要该位置的 logits
来产生下一个 token。核心见
[`radix_cache.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/mem_cache/radix_cache.py)。

Radix node lock 很关键：Scheduler 根据“free + evictable”做初步 admission 后，锁住 matched node 会让一部分
evictable KV 变成 protected，因此 `PrefillAdder` 必须再次核算容量。否则同一份容量会被重复承诺。

### 实验

| 模式 | Cold | Warm | 命中 |
| --- | ---: | ---: | ---: |
| vLLM eager server prefill | 47.15 ms | 41.78 ms | 2064/2080 |
| vLLM graph server prefill | 44.52 ms | 13.78 ms | 2064/2080 |
| SGLang eager server E2E | 56.84 ms | 49.59 ms | 2079/2080 |
| SGLang graph server E2E | 44.66 ms | 18.17 ms | 2079/2080 |

Eager 小模型中即使计算 token 几乎全部省掉，仍要做一次尾部 forward、metadata、kernel launch、sampling、IPC
和 output，因此命中率不能等比例变成延迟收益。Graph 将固定 launch 成本压低后，省掉 prefill 的收益才更
明显。

### 得分点（10 分）

- 2 分：说明 cache key 是 token/额外上下文身份，不是字符串模糊相似。
- 2 分：解释 vLLM full-block hash 与 SGLang radix/page 粒度。
- 2 分：指出最后一个 token/未对齐 block 仍需 forward。
- 2 分：讲清引用、lock、eviction 与 request-private 尾部。
- 2 分：知道 hit rate 是 work-reduction 指标，不是 latency speedup。

### 追问

**为什么不能缓存任意半个 vLLM block？** 物理 attention/page table、hash 身份、分配和 eviction 都以 block
为单位；只把完整 block 暴露为共享 cache entry 可以保持下游 contract 简单。代价是最多一个 block 的
内部碎片/重算。

**cache salt 有什么用？** 它进入 cache identity，使不应跨 tenant/request 复用的相同 token prefix 分离；
否则 prefix cache 可能成为信息侧信道。

---

## 4. vLLM Beam Search 会不会 fork KV block

### 结论

对当前源码的 online/offline Beam Search，答案不是“在 Scheduler 内将父 sequence 的 block table 原地 fork
并对最后 block 做 CoW”。它在 serving/LLM 上层维护 `BeamSearchSequence`：每一轮把每条 beam 的完整 token
前缀重新提交为独立 engine request，设置 `max_tokens=1` 和 `logprobs=2*beam_width`，收集候选后在 CPU 排序，
保留 top K，再进入下一轮。

源码证据：

- [`online.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/generate/beam_search/online.py)
  为每条 beam 构造 `request_id-...-beam-i` 并调用 `engine_client.generate()`。
- [`offline.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/generate/beam_search/offline.py)
  每轮 `_render_and_run_requests()`，同样只生成一个 token。
- [`utils.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/entrypoints/generate/beam_search/utils.py)
  的 `get_prompt()` 用当前 beam 完整 token list 重建输入。

### 实际 KV 行为

```text
父 beam token prefix
  -> 新 engine request
  -> prefix cache 查找完整 hashed blocks
  -> 已缓存完整 blocks 可被新 request 引用
  -> 未对齐尾部/新增 token 重新计算并获得自己的 block table
  -> 一 token后请求结束，full blocks可继续成为 prefix cache
```

因此确实存在“多个逻辑 beam 共享相同物理 cached prefix block”的效果，但它通过跨请求 prefix caching
实现，不是传统单请求 sequence fork API。每条 beam 都有独立 request 生命周期和 block table；共享的是
命中的 immutable cached blocks。

### 为什么这样设计

优点：复用正常 generate、scheduler、abort、structured output 和 prefix cache，不需要在 core 中维护复杂的
beam parent/child 状态，也不会让 beam 语义侵入通用 sampling request。

代价：每个 output token 都经历新请求构造、调度、prefix lookup 和结果收集；尾部块可能反复重算；CPU
top-K 和 barrier 使所有 beams 按轮同步，难以获得普通 continuous decode 的效率。Beam Search 不是当前 vLLM
最理想的吞吐路径。

### 得分点（10 分）

- 3 分：基于当前实现指出“上层多次独立 generate”，而不是背旧版 sequence fork。
- 2 分：说明 full cached prefix blocks 仍可物理复用。
- 2 分：说明尾部 block 对齐和重新 forward。
- 2 分：说明每轮 CPU top-K/barrier/新 request 的成本。
- 1 分：区分逻辑 beam fork、block-table fork、物理 KV copy 三个概念。

### 追问

**如果原生实现 block-table fork，应怎样做？** 父子先共享 block ID 并增加 refcount；后续 append 若落在
已共享且可变的最后 block，给子分配新 block 并复制有效 KV，之后各自写入。完整 immutable block 不需要
复制。还必须处理 beam prune 时 refcount、prefix hash 身份和异步 in-flight kernel 的生命周期。

---

## 5. Continuous batching 提升吞吐，为什么 TTFT/TPOT 会变差

### 主问题

Scheduler 每轮真正调度的单位是什么？为什么提高 concurrency 同时会增加 TTFT、TPOT 或 p99 ITL？

### vLLM

[`Scheduler.schedule()`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/core/sched/scheduler.py)
的核心预算是 token budget 与 sequence budget，而不是“一个请求等于一个槽”：

1. 先处理 running requests，按 `num_computed_tokens` 与需要推进的 token 数分配本轮 token；
2. 按可用 KV blocks 调整实际 scheduled token；
3. 剩余 `token_budget` 再从 waiting queue 准入新请求；
4. Output `SchedulerOutput` 冻结本轮 request IDs、scheduled token counts、new/resumed/preempted state 和 block tables；
5. Runner 执行，Scheduler 再按 request/iteration 将 sampled result commit。

### SGLang

[`Scheduler`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)
维护 waiting/running/last/result queue。`SchedulePolicy` 先排序，`PrefillAdder` 再同时核算 KV、prefill token、
request row、future decode、SWA/Mamba/spec 等预算，Output 是 EXTEND/DECODE/MIXED `ScheduleBatch`。

### 实验

16 个请求，每个 256 input + 32 output：

| 模式 | 并发 | Output tok/s | Mean TTFT | Mean TPOT | p99 ITL |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM graph | 1 | 280.79 | 40.04 ms | 2.38 ms | 5.06 ms |
| vLLM graph | 4 | 798.27 | 69.44 ms | 2.90 ms | 4.77 ms |
| vLLM graph | 16 | 2078.31 | 128.10 ms | 3.58 ms | 10.58 ms |
| SGLang graph | 1 | 370.83 | 25.82 ms | 1.94 ms | 3.19 ms |
| SGLang graph | 4 | 1012.52 | 42.46 ms | 2.68 ms | 6.75 ms |
| SGLang graph | 16 | 2316.93 | 88.34 ms | 4.06 ms | 31.99 ms |

更大的 batch 摊薄权重读取、launch 和同步，吞吐上升；但单步 kernel 更长、请求在 queue/batch boundary 等待，
新 prefill 还会和 running decode 竞争 token/SM/HBM，所以延迟不是免费收益。生产调参目标应是满足 TTFT/ITL
SLO 下的 goodput，而不是最大离线 tok/s。

### 得分点（10 分）

- 2 分：指出动态 token-level schedule，而不是静态 request batch。
- 2 分：讲清 waiting、running、finish 后补入新请求。
- 2 分：解释吞吐收益来自硬件利用和固定成本摊薄。
- 2 分：解释 TTFT/ITL 损失来自 queue、step duration 和 prefill interference。
- 2 分：用 SLO goodput 而非峰值 tok/s 作为调参目标。

---

## 6. Chunked prefill 为什么有时越切 p99 越差

### 主问题

Chunking 是否天然能保护 decode ITL？vLLM 和 SGLang 的 interleave 方式有什么实质差异？

### 要解决的问题

一个 7K token prefill 可以独占一次长 forward，让已经 streaming 的请求长时间拿不到下一 token。Chunking
限制每轮 prefill work，但只有在 chunk 之间允许 decode 前进时，才真正提供 latency isolation。

### vLLM 方法

`Scheduler.schedule()` 先给 running 请求分配当轮 token，再用剩余 token budget 接纳 waiting prefill。
将 `max_num_batched_tokens` 从 8192 降到 512，会让大 prompt 分多轮推进，同时 running decode 每轮仍在同一个
计划中获得预算，不需要另一个 mixed flag。

### SGLang 方法

默认 `enable_mixed_chunk=False` 时，一个 chunked prefill request 可以连续产生多个 EXTEND batch；chunk 只控制
单批容量，不保证每个 chunk 与 decode 混合。开启 mixed 后，Scheduler 调用
`new_batch.mix_with_running(running_batch)`，形成 MIXED batch，decode 才能在长 prefill 期间持续推进。源码见
[`scheduler.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler.py)
和 [`schedule_batch.py`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py)。

### 实验

一个请求 streaming 256 token，第 20 个 stream event 后注入 7168-token prefill，重复五次：

| 配置 | 基线 event gap | 重叠期最大 gap | 长 prefill 客户端延迟 |
| --- | ---: | ---: | ---: |
| SGLang chunk 8192，无 mixed | 1.97 ms | 119.67 ms | 153.94 ms |
| SGLang chunk 512，无 mixed | 2.00 ms | 221.63 ms | 251.40 ms |
| SGLang chunk 512，有 mixed | 1.95 ms | 23.87 ms | 273.50 ms |
| vLLM budget 8192 | 2.36 ms | 87.25 ms | 142.83 ms |
| vLLM budget 512 | 2.34 ms | 22.18 ms | 302.11 ms |

SGLang 只把 chunk 改成 512 反而更差，因为日志显示 14 个 512-token EXTEND 连续执行，decode 没有前进；
增加了调度/metadata/launch 次数，却没有获得 interleave。mixed 后最大 gap 从 221.6 降到 23.9 ms，但长
prefill 延迟增加。vLLM 也展示同一 trade-off：小 budget 保护 streaming gap，同时把长 prefill 延迟翻倍。

注意 OpenAI SSE 可能合并 detokenized 片段，本实验测的是客户端可见 event gap，不是精确的单 token kernel
ITL；但对用户 SLO 仍有直接意义。

### 得分点（10 分）

- 2 分：区分 work chunking 与 scheduling interleave。
- 2 分：说明 vLLM running-first token budget 的行为。
- 2 分：说明 SGLang mixed batch 的显式开关和 merge。
- 2 分：讲清 ITL 改善与 prefill TTFT/launch overhead 的交换。
- 2 分：知道 SSE event gap 的测量口径和限制。

### 追问

**chunk size 如何选？** 用最长允许 non-preemptible GPU step 反推，而不是只看 prompt 长度。需要联合观察
decode p99 ITL、long-prompt TTFT、graph bucket hit、batch size 和 GPU utilization；不同模型/TP/backend 的最优值
不同。

---

## 7. CUDA Graph 为什么小 batch 提升 6 到 9 倍，长 prefill 几乎无效

### 要解决的问题

Decode 的 shape 小、iteration 多，CPU launch、Python、driver 和跨 rank launch latency 可能接近 kernel 时间。
CUDA Graph 预先捕获固定地址和 shape 的执行 DAG，replay 一次提交多个 kernel。

### Input -> 方法 -> Output

| Input | 方法 | Output |
| --- | --- | --- |
| batch/token shape、稳定 GPU buffers、backend graph support | 选择不小于真实 shape 的 captured bucket，padding 后 replay | logits/sampled token，与 eager 语义一致 |

Graph 优化的是 launch/scheduling overhead，不会减少模型理论 FLOPs，也不会消除 queueing。它还需要稳定地址、
预分配 buffer 和可 capture op；动态图、未捕获 shape、某些 LoRA/grammar/spec 路径会 fallback 或走另一套 graph。

### 当前实现差异

- vLLM graph server 使用 compile 的 full/piecewise graph，并捕获多个 batch buckets；启动时首次 compile 约
  31 s，graph capture 约 1 s，额外占用约 0.12 GiB，KV capacity 从 245312 降为 244288 token。
- SGLang 当前默认对 decode 使用 full graph，对 prefill 使用 breakable graph。实验捕获 58 个 4..8192 prefill
  token buckets，约 10.14 s/0.66 GB；decode 捕获 8 个 batch buckets，约 1.37 s/0.07 GB。
- SGLang NGRAM 将 decode graph 变成 `TARGET_VERIFY` graph，`num_tokens_per_req=8`，捕获 bucket 数和临时 buffer
  也随之变化。这说明“开启 graph”不是一个布尔值，而是一组 phase/shape-specific runners。

### 实验

| 框架，并发 1 | Eager output tok/s | Graph output tok/s | 倍数 |
| --- | ---: | ---: | ---: |
| vLLM | 43.14 | 280.79 | 6.51x |
| SGLang | 40.46 | 370.83 | 9.17x |

SGLang 256-token prefill 从 61.42 ms 降到 24.27 ms，4096-token prefill却是 105.07 vs 105.45 ms。短
workload 的 launch 固定成本可被 graph 消除；长 attention 已计算主导，graph 无法改变 FLOPs/HBM 主项。

### 得分点（10 分）

- 2 分：指出 graph 优化 launch，不是模型计算复杂度。
- 2 分：说明稳定地址、bucket、padding 和 fallback。
- 2 分：区分 decode/prefill/verify graph。
- 2 分：说出 capture 时间、显存和 KV capacity 成本。
- 2 分：用短/长 workload 的数据解释收益边界。

---

## 8. Speculative decoding 的 draft/verify/commit 到底做什么

### 主问题

投机解码为何能保持 target 输出分布不变？`accept_rate`、`accept_length` 和 speedup 的关系是什么？

### 要解决的问题

普通 decode 每个 target forward 只提交一个 token。小 batch 时 target 权重读取、kernel launch 和通信没有被
充分摊薄。Speculation 用便宜 proposer 产生 K 个候选，让 target 一次宽 forward 验证多个位置，再只提交合法
连续前缀。

### 三阶段 contract

| 阶段 | Input | Output | 不变量 |
| --- | --- | --- | --- |
| Draft | committed tokens、可选 target hidden、draft KV/corpus、K/top-k | chain/tree token IDs、parent/mask、可选 draft probs | proposal 不是用户可见 truth |
| Verify | target KV、draft tree、positions/mask、sampling params | accepted path、mismatch recovery/bonus、accept length | 输出分布与 target 原采样一致 |
| Commit | accept indices、临时 KV slots、hidden/draft state | 连续 committed tokens/KV、下一轮 draft seed | rejected branch 不可进入后续上下文 |

Greedy 情况沿 root 开始逐个比较 target argmax，首个 mismatch 后不能继续接受更深 token，因为条件上下文已经
不同。随机采样要用 acceptance/rejection correction；vLLM 的
[`rejection_sampler.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/sample/rejection_sampler.py)
明确区分 accepted、recovered 和 bonus token，并按论文算法保持 target distribution。

### 成本模型

设一轮平均提交 `A` 个 token：

```text
spec time per committed token ~= (T_draft(K) + T_verify(K) + T_commit(K)) / A
baseline time per token       ~= T_target_decode(1)
```

只有左侧更小时才加速。增加 K 可能提升 A，也会扩大 verify attention/logits、临时 KV、graph padding 和 draft
成本。高并发时 baseline target decode 已能形成大 batch，spec 的额外 work 可能不再划算。因此接受率不是
唯一目标，adaptive K/batch-size policy 是合理设计。

### vLLM NGRAM

[`ngram_proposer.py`](https://github.com/Cptzzzzz/vllm/blob/d1d2f7535686acdc39b034bf18ba52d07c21def5/vllm/v1/spec_decode/ngram_proposer.py)
在当前 token history 中找最长后缀 n-gram 的历史匹配，并取其后的 K 个 token。没有 draft model weights，
proposal 可为空，target rejection sampler 仍负责正确性。

### SGLang NGRAM

[`NGRAMWorker`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/speculative/ngram_worker.py)
持有 CPU corpus/SAM，没有 draft model。每轮：

1. 将 overlap 中尚未 commit 到 `Req.output_ids` 的上一轮 accepted tokens 拼入查询尾部；
2. `batch_get()` 产生固定 node budget 的 irregular tree 与 mask；
3. 将 batch 改为 `TARGET_VERIFY`，为 K 个节点分配临时 KV slots；
4. target 宽 forward 后 `eagle_sample()` 得到 accepted index/length；
5. `move_accept_tokens_to_target_kvcache()` 移动/保留 accepted KV，释放 rejected slots；
6. 将最近 token 插回 corpus，形成下一轮 proposal 来源。

`accept_lens` 包含每轮至少一个 target/bonus token；metrics 中：

```text
accept_rate   = correct draft tokens / proposed draft tokens
accept_length = completion tokens / verify count
```

两者分母和是否包含 bonus 不同，不能互换。

### 实验

SGLang 默认 graph，NGRAM verify width 8，生成 256 tokens，每类 prompt 测三次：

| Prompt | Baseline E2E | NGRAM E2E | Speedup | Draft accept rate | Accept length | Verify 次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alpha beta gamma delta` 重复序列 | 504.47 ms | 187.35 ms | 2.69x | 96.97% | 7.76 | 33 |
| 普通 compiler 说明文 | 503.38 ms | 331.92 ms | 1.52x | 42.21% | 3.88 | 66 |

重复序列几乎每次接受满 7 个 draft，只需 33 次 verify；普通文本仍通过格式/局部重复获得平均 3.88 token，
但 speedup 明显下降。`7.76x` 的 iteration reduction 只变成 `2.69x` E2E speedup，直接证明 verify width、
proposal、KV commit 和 prefill/HTTP 固定成本不可忽略。

### 得分点（10 分）

- 2 分：讲清 draft/verify/commit 三阶段和状态所有权。
- 2 分：说明首 mismatch 后更深候选无效，以及 stochastic correction。
- 2 分：区分 accept rate、accept length、speedup。
- 2 分：给出包含 draft/verify/commit 的成本模型。
- 2 分：讲清 rejected KV 清理、accepted path compaction/commit。

### 追问

**EAGLE 为什么还要 draft extend？** Drafter 展开的是候选 tree；verify 后最终 truth 可能是若干 accepted draft
加一个 target token。下一轮 proposal 前必须用 accepted target hidden/token 将 drafter KV/state 对齐，否则 drafter
会基于 rejected branch 继续生成。

**为什么 NGRAM 可能优于小 draft model？** 它没有额外权重显存和 draft forward，对代码、JSON、重复模板等
workload proposal 极便宜；自然语言低重复或 corpus domain mismatch 时命中/接受低，小模型 drafter可能更稳。

**为什么高 batch 可能关闭 speculation？** Baseline decode 已经充分利用 GPU，而 verify 扩宽 token 数、临时 KV
和 sampling work；每省一次 launch 的边际收益下降。应该按 batch size/acceptance/profile 自适应 K。

---

## 9. KV 不够时 vLLM preempt 与 SGLang retract 有什么不同

### 主问题

两者如何避免 KV OOM？为什么同样 4096 token 容量，vLLM 出现 preemption，而 SGLang 默认没有 retraction？

### vLLM

vLLM 按本轮需要增量分配 blocks。running request 分配失败时，Scheduler 从候选中选择 victim，调用
`_preempt_request()`，释放其 KV、清空 computed progress，并放回 waiting 走 recompute。它优先保证本轮计划可执行，
但 preempted request 的 TTFT/p99 和重复 prefill work 上升。

### SGLang

SGLang `PrefillAdder` 对新请求不仅看 prompt suffix，还用 `new_token_ratio`/max_new_tokens 预测 future decode 空间。
保守 admission 可以一开始只让较少 request running，从而避免 retraction；代价是更多请求停在 waiting queue。

当 active decode 的真实增长超过预测、allocation 失败时，
[`retract_decode()`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/schedule_batch.py)
选择 request 回退、释放 private KV、重新排队；
[`NewTokenRatioTracker`](https://github.com/Cptzzzzz/sglang/blob/9529c2e96351f3b1d5715eaa89989b20723d60b5/python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py)
根据这次失败提高 future reservation，降低继续 over-admit 的概率。

### 实验

4096 KV tokens、8 个并发请求，每个 1024 input + 512 forced output：

| 配置 | Preempt/retract | Mean TTFT | p99 TTFT | Output tok/s | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | 3 preemptions | 1765 ms | 3589 ms | 780.08 | 3.41 ms |
| SGLang 默认，无 mixed | 0 | 2205 ms | 4325 ms | 719.77 | 2.69 ms |
| SGLang mixed | 1 retraction | 1989 ms | 4438 ms | 707.89 | 3.21 ms |

SGLang 默认日志显示仅 2 个 request running，其余 waiting，因此没有内存危机；这不是 retraction 更强，而是
admission 更保守。开启 mixed 后 active batch 增到 3、KV usage 达 0.99，随后 retract 一个已经生成 338 token
的请求，释放 1024 input + 338 output 对应状态，并将 `new_token_ratio` 从 0.098 提高到 0.7063。

### 指标陷阱

实验后 `sglang:num_retracted_reqs` gauge 又回到 0，因为它表示当前/最近 iteration，不是累计事件。累计事实应
看 `num_retracted_input_tokens_total`、`num_retracted_output_tokens_total` 或日志。面试中只截图一个 gauge 很容易
得出错误结论。

### 得分点（10 分）

- 2 分：区分 admission reservation 与运行期 recovery。
- 2 分：说明 vLLM incremental block allocation + recompute preemption。
- 2 分：说明 SGLang future token prediction + retraction + ratio feedback。
- 2 分：解释保守 admission 降 retraction 但升 queue TTFT。
- 2 分：识别 gauge 与 cumulative counter 的指标语义。

### 追问

**频繁 preemption/retraction 应如何排查？** 同时看 KV usage/free blocks、running/waiting、prompt/output length
分布、chunk/mixed、max running、preemption tokens、queue time 和 prefix hit。先判断容量真的不足，还是 future
reservation/调度过激；只扩大 max running 往往让 p99 更差。

---

## 10. Overlap/异步执行为什么需要 plan-execute-commit

### 主问题

为什么不能在 Scheduler launch GPU 后立即把 sampled token、KV length 当成已提交？

### 要解决的问题

为了隐藏 CPU scheduling、metadata build、D2H 和 detokenization，框架会让 iteration `t+1` 的准备与 `t`
的 GPU/result 处理重叠。但 GPU 输出尚未完成时，CPU 不能把乐观状态暴露为用户 truth，也不能释放它仍在读写的
buffer/KV。

### 通用状态机

```text
Plan(t):
  snapshot request IDs, scheduled token counts, block tables, input positions

Execute(t):
  kernels read snapshot and write preallocated output/temporary KV

Plan(t+1) in parallel:
  may use published shape/length hints, but not unverified user-visible tokens

Commit(t):
  match result to launch snapshot
  validate sample/spec acceptance
  update logical tokens and committed KV boundary
  release rejected/finished resources
```

### vLLM

EngineCore/Scheduler 通过 `SchedulerOutput` 冻结计划，Runner 返回 model output 后
`Scheduler.update_from_output()` 才更新 request。异步 scheduling 还需要避免下一轮覆盖仍被 graph/kernel 使用的
stable input buffers。

### SGLang

overlap scheduler 使用 future map/result queue 与独立 stream。NGRAM 源码是最直观的证据：上一轮 accepted
tokens 在 overlap 模式下尚未写入 `req.output_ids`，下一轮 `_prepare_draft_tokens()` 必须从 `spec_info` 拼入，
同步模式则不能拼，否则会重复 token。Spec verify 还区分 `kv_allocated_len` 和 `kv_committed_len`，因为临时 tree
slots 已分配但不一定被接受。

### 正确性不变量

1. result 必须应用到原 launch snapshot，不可按当前 batch index 猜 request；
2. `allocated >= committed`，rejected overshoot 必须只释放一次；
3. 已 publish length 不等于 token 已对外提交；
4. 下一轮覆盖 buffer 前，所有 reader stream 必须记录 event/完成；
5. abort 可以阻止未来 schedule，通常不能撤销已 launch kernel，回来后应丢弃结果并清理资源。

### 得分点（10 分）

- 3 分：清楚区分 planned、in-flight、committed state。
- 2 分：说明 launch snapshot/request identity。
- 2 分：说明 allocated/committed KV 边界。
- 2 分：说明 stream event、buffer lifetime、WAR hazard。
- 1 分：把 cancel/abort 放进同一生命周期模型。

---

## 11. 如何从实验结果反推瓶颈

面试中给出一张 latency/throughput 表时，可以按以下顺序定位：

| 现象 | 首要假设 | 下一步证据 |
| --- | --- | --- |
| graph 对短 decode 提升巨大 | launch/CPU overhead 主导 | Nsight kernel gap、CPU submit、graph hit/fallback |
| graph 对长 prefill无提升 | attention/GEMM compute 主导 | SM/Tensor Core occupancy、HBM、length sweep |
| prefix hit 高但 TTFT 变化小 | fixed tail forward/IPC/sampling 主导 | server phase timer、尾部 token/block alignment |
| 并发升高 tok/s 涨、TTFT/ITL 涨 | batch efficiency换 queue/step time | queue time、batch size、per-step duration |
| chunk 变小 p99 反而涨 | 没有 interleave或 launch 次数过多 | 每轮 batch mode、decode progress、mixed flag |
| retraction 为 0 但 TTFT 高 | conservative admission | running/waiting、future reservation、KV usage |
| spec accept rate 高但没加速 | verify/draft/commit成本或 baseline batch 已饱和 | accept length、phase time、verify width、batch size |
| p99 ITL 异常但 mean TPOT正常 | prefill interference、retraction、GC/IPC或 graph fallback | timeline 与请求级 histogram，不只平均值 |

不要从一个 aggregate 指标直接推实现。例如“无 retraction”可能是容量足，也可能是请求根本没被 admit；
“cache hit 99%”可能只省掉一部分 GPU compute，用户延迟仍被尾部和排队主导。

---

## 12. 还需要多 GPU 验证的问题

本轮单卡没有伪造 TP/EP/P-D 性能结论。以下是下一阶段应实际做的实验：

### Tensor Parallel

**问题**：增加 TP 为什么可能降低单请求 latency，却降低小 batch efficiency？

**实验**：固定模型和输出，TP=1/2/4，分别扫 batch=1/8/32；记录每层 collective 时间、compute/communication
overlap、graph bucket 和 NCCL bytes。需要区分 GEMM 变小带来的低占用与 all-reduce 固定延迟。

### Expert Parallel

**问题**：MoE 的瓶颈为什么从权重 GEMM 变成 token dispatch/all-to-all 和 load imbalance？

**实验**：控制 token routing skew，记录每 rank expert token 数、all-to-all、padding/drop、EPLB 前后 p99。平均
expert load 相同不代表 straggler 相同。

### Prefill/Decode disaggregation

**问题**：P/D 分离何时能减少 interference，何时 KV transfer 让 TTFT/ITL 更差？

**实验**：扫 prompt/output ratio 与 KV transfer backend，拆分 prefill compute、KV serialization/transfer、decode
admission 三段 latency；同时测 replica failure/cancel 后 KV ownership 清理。只报告端到端 tok/s 无法判断收益来源。

### 高分回答模板

```text
先声明 workload 和 SLO。
再确定逻辑状态与物理资源的 owner。
沿 Input -> Plan -> Allocate -> Execute -> Commit -> Output 讲源码。
指出优化直接减少的成本，以及新增的成本。
最后给出能证伪自己判断的指标与对照实验。
```

做到这一步，回答才从“知道框架名词”进入“能设计、调试和评审推理系统”的层次。
