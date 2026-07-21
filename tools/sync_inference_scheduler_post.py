#!/usr/bin/env python3
"""Publish a source-based vLLM and SGLang scheduler deep dive."""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path


BLOG_ROOT = Path(__file__).resolve().parents[1]
INFRA_ROOT = Path(os.environ.get("INFRA_CODES_ROOT", "/root/infra-codes"))
INFERENCE_ROOT = INFRA_ROOT / "inference"
POST_PATH = (
    BLOG_ROOT
    / "_posts"
    / "inference-architecture"
    / "2026-07-21-vllm-sglang-scheduler-source-deep-dive.md"
)
INFRA_COMMIT = "1f8930ce3f206cdaac9ec24954f5a428e8c890f8"
VLLM_COMMIT = "d1d2f7535686acdc39b034bf18ba52d07c21def5"
SGLANG_COMMIT = "9529c2e96351f3b1d5715eaa89989b20723d60b5"
INFRA_BLOB = f"https://github.com/Cptzzzzz/infra-codes/blob/{INFRA_COMMIT}/inference"

SCHEDULER_URL = "/posts/vllm-sglang-scheduler-source-deep-dive/"
VLLM_LIFECYCLE_URL = "/posts/vllm-request-lifecycle-source-guide/"
SGLANG_LIFECYCLE_URL = "/posts/sglang-request-lifecycle-source-guide/"
INTERVIEW_URL = "/posts/vllm-sglang-interview-series/"


def read_module(path: str) -> str:
    source = INFERENCE_ROOT / path
    if not source.is_file():
        raise FileNotFoundError(f"missing scheduler note: {source}")
    return source.read_text(encoding="utf-8")


def prepare_module(text: str, prefix: str, source_dir: str) -> str:
    """Strip the source H1, demote headings, namespace refs, and fix links."""

    text = re.sub(r"\A# .+?\n+", "", text, count=1).strip()
    text = re.sub(r"(?m)^(#{2,5}) ", lambda match: f"#{match.group(1)} ", text)
    text = re.sub(
        r"\]\[([^]]+)\]",
        lambda match: f"][{prefix}-{match.group(1)}]",
        text,
    )
    text = re.sub(
        r"(?m)^\[([^]]+)\]:",
        lambda match: f"[{prefix}-{match.group(1)}]:",
        text,
    )

    def rewrite_inline(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith(("http://", "https://", "/", "#", "mailto:")):
            return match.group(0)
        path, marker, fragment = target.partition("#")
        normalized = posixpath.normpath(f"{source_dir}/{path}")
        rewritten = f"{INFRA_BLOB}/{normalized}"
        if marker:
            rewritten = f"{rewritten}#{fragment}"
        return f"]({rewritten})"

    return re.sub(r"\]\(([^)]+)\)", rewrite_inline, text)


def front_matter() -> str:
    return """---
title: 'vLLM 与 SGLang 调度器源码详解：排序、准入、KV 与提交'
author: Cptz
date: 2026-07-21 00:05:00 +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, scheduler, kv cache, continuous batching, chunked prefill]
math: true
mermaid: true
---

"""


def introduction() -> str:
    return f"""<!-- Generated from infra-codes commit {INFRA_COMMIT} by tools/sync_inference_scheduler_post.py. -->

> 本文固定到实际运行验证过的 vLLM 与 SGLang fork，从源码对象和状态变化解释调度。目标不是背策略名称，而是理解一轮计划如何在 token、KV、请求槽位和 SLO 约束下成立。

[vLLM 请求全流程]({VLLM_LIFECYCLE_URL}) · [SGLang 请求全流程]({SGLANG_LIFECYCLE_URL}) · [面试题系列]({INTERVIEW_URL})

## 先用 30 秒理解调度

在线推理调度不是简单地“从队列取几个请求组成 batch”。它至少包含四个不同阶段：

```text
排序 Ordering
  哪些请求先尝试，解决优先级、公平性和 cache locality

准入 Admission
  检查 token、KV、request row、encoder、LoRA 等资源是否真的足够

计划 Planning
  冻结本轮每个请求计算多少 token、使用哪些物理 slot、采用什么 forward mode

提交 Commit
  GPU 返回后提交 accepted token、回滚 speculative overshoot，并释放结束请求资源
```

一个请求“排在前面”不等于“已经进入 GPU batch”。只有完成物理资源分配，它才有资格进入 `SchedulerOutput` 或 `ScheduleBatch`。

```mermaid
flowchart LR
    A[Waiting / Running State] --> B[Order]
    B --> C[Admission Ledger]
    C -->|资源不足| D[Wait / Preempt / Retract]
    C -->|资源落实| E[Execution Plan]
    E --> F[GPU Forward + Sample]
    F --> G[Commit / Rollback]
    G --> A
```

## 源码与讨论边界

- vLLM 固定到 [`{VLLM_COMMIT[:9]}`](https://github.com/Cptzzzzz/vllm/commit/{VLLM_COMMIT})，核心入口是 `Scheduler.schedule()` 与 `update_from_output()`。
- SGLang 固定到 [`{SGLANG_COMMIT[:9]}`](https://github.com/Cptzzzzz/sglang/commit/{SGLANG_COMMIT})，核心入口是 `get_next_batch_to_run()`、`SchedulePolicy` 和 `PrefillAdder`。
- 主路径是 decoder-only generation；multimodal、LoRA、P/D、SWA/Mamba 和 speculative decoding 作为资源约束进入同一调度框架。
- 伪代码用于压缩控制流，分支细节以链接中的固定 commit 为准。

核心函数地图：

| 框架 | 调度环节 | 固定源码入口 |
| --- | --- | --- |
| vLLM | 一轮计划 | [`Scheduler.schedule()`](https://github.com/Cptzzzzz/vllm/blob/{VLLM_COMMIT}/vllm/v1/core/sched/scheduler.py#L421) |
| vLLM | Prefix 命中与 KV 准入 | [`get_computed_blocks()` / `allocate_slots()`](https://github.com/Cptzzzzz/vllm/blob/{VLLM_COMMIT}/vllm/v1/core/kv_cache_manager.py#L207) |
| vLLM | 抢占 | [`_preempt_request()`](https://github.com/Cptzzzzz/vllm/blob/{VLLM_COMMIT}/vllm/v1/core/sched/scheduler.py#L1186) |
| vLLM | 结果提交/回滚 | [`update_from_output()`](https://github.com/Cptzzzzz/vllm/blob/{VLLM_COMMIT}/vllm/v1/core/sched/scheduler.py#L1546) |
| SGLang | Batch 状态机 | [`get_next_batch_to_run()`](https://github.com/Cptzzzzz/sglang/blob/{SGLANG_COMMIT}/python/sglang/srt/managers/scheduler.py#L2649) |
| SGLang | 候选排序 | [`SchedulePolicy.calc_priority()`](https://github.com/Cptzzzzz/sglang/blob/{SGLANG_COMMIT}/python/sglang/srt/managers/schedule_policy.py#L184) |
| SGLang | Prefill 准入账本 | [`PrefillAdder.add_one_req()`](https://github.com/Cptzzzzz/sglang/blob/{SGLANG_COMMIT}/python/sglang/srt/managers/schedule_policy.py#L976) |
| SGLang | Decode 更新 | [`update_running_batch()`](https://github.com/Cptzzzzz/sglang/blob/{SGLANG_COMMIT}/python/sglang/srt/managers/scheduler.py#L3117) |
| SGLang | KV 不足时撤回 | [`retract_decode()`](https://github.com/Cptzzzzz/sglang/blob/{SGLANG_COMMIT}/python/sglang/srt/managers/schedule_batch.py#L2591) |
| SGLang | CPU/GPU overlap | [`event_loop_overlap()`](https://github.com/Cptzzzzz/sglang/blob/{SGLANG_COMMIT}/python/sglang/srt/managers/scheduler.py#L1542) |

## 两套调度器共同求解的问题

可以把目标写成：

```text
maximize goodput
subject to:
  per-iteration token budget
  physical KV capacity
  request/model length limits
  TTFT and ITL SLO
  priority and fairness
  rank-consistent execution plan
```

关键状态不能混淆：

| 状态 | 含义 | 混淆后的后果 |
| --- | --- | --- |
| 逻辑 token | 用户已提交的 prompt/output 真值 | 可能把 rejected draft 暴露给用户 |
| computed token | target model 已计算、可被后续位置依赖 | attention 读取未生成的 KV |
| allocated KV | 已占用的物理位置，可能包含临时或未来位置 | 泄漏或错误共享 speculative KV |
| committed KV | 已确认属于真实序列的 KV 边界 | Prefix Cache 污染 |
| in-flight plan | 已 launch 但结果尚未提交的一轮工作 | 跨轮 plan/result 错配 |

下面先逐行理解 vLLM，再理解 SGLang。

---

## Part I：vLLM 的统一 Token-Debt 调度
"""


def comparison() -> str:
    return f"""

---

## Part III：把两套设计放在同一张图里

### 1. 核心设计差异

| 维度 | vLLM | SGLang |
| --- | --- | --- |
| 核心抽象 | 每请求还欠 target model 多少 token | `EXTEND/DECODE` batch 状态机 |
| Prefill/Decode | 用 token debt 统一表达 | 用 `ForwardMode` 显式表达 |
| 排序 | FCFS/Priority request queue 为主 | FCFS、LPM、DFS-weight、LOF、Routing Key |
| Prefix 感知 | waiting admission 时查 block hash/cache | Policy 排序时结合 Radix prefix match |
| 准入实现 | `schedule()` 直接调用 KV/encoder manager | `SchedulePolicy` 排序，`PrefillAdder` 独立记账 |
| 本轮输出 | `SchedulerOutput` | `NextBatchPlan` + `ScheduleBatch` |
| KV 压力恢复 | preempt running request，回 waiting 重建 | retract decode request，释放私有 slot 后回 waiting |
| 异步状态 | token count、placeholder、future 严格配对 | batch snapshot、result queue、FutureMap、WAR barrier |
| 默认延迟倾向 | running-first 保护 Decode ITL | prefill-first 改善 waiting TTFT |
| Cache 数据结构 | hash block table + ref count | Radix Tree + request row + physical KV pool |

两者不是“谁更先进”的简单关系。vLLM 用统一 token 进度减少 phase 状态分叉；SGLang 把 prefix locality、准入账本和 batch mode 显式化，方便围绕 Radix/agent workload 做策略扩展。

### 2. 同一组请求会怎样被调度

假设当前有：

```text
A、B：正在 Decode，每个下一轮需要 1 token KV
C：8K prompt，其中 6K prefix 已缓存
D：8K prompt，没有缓存命中
本轮 token budget：2048
```

**vLLM**：先为 A、B 偿还各自的 Decode debt；只要没有发生 preemption，再扫描 waiting。C 先将 cached computed boundary 推进到约 6K，只为 suffix 核算 token/KV，剩余预算不足时形成 chunk。D 只有在剩余 token/KV 仍满足时才进入计划，否则继续 waiting。

**SGLang**：`SchedulePolicy` 若采用 LPM，会把 C 排在 D 前面；`PrefillAdder` 在锁住 C 的 Radix node 后重新计算可淘汰空间，并为 suffix、page overhead 和未来 Decode 预留资源。只要产生 EXTEND batch，默认先执行它；A、B 继续留在 `running_batch`，下一轮没有合适 Prefill 时再 Decode。mixed chunk 配置可以把部分 Decode 合入 EXTEND batch。

因此，同一 workload 下常见倾向是：vLLM 更直接保护正在运行请求的 ITL，SGLang 更积极利用新请求的 prefix locality和 TTFT；最终表现仍由 chunk、预算、priority、delayer 与 Cache 压力决定。

### 3. Preemption 与 Retraction 的共同本质

两者都是 OOM 防线，不是免费优化：

```text
资源预测失败或序列增长超出预留
  -> 选择 victim
  -> 释放 victim 的私有物理资源
  -> 将逻辑请求退回可重建状态
  -> 后续依靠 Prefix Cache 命中或重新计算
```

如果 preemption/retraction 持续发生，应检查 admission、KV 容量和 workload，而不是继续提高 batch 上限。它们通常表现为 recompute 增加、Cache churn、TTFT/ITL p99 峰值和 goodput 下降。

## Part IV：如何调参和验证

### 1. 不要从最大吞吐开始

先固定 workload：prompt/output 长度分布、prefix 复用率、到达过程、并发、模型/dtype/backend。然后定义 TTFT、ITL 和错误率 SLO，在满足 SLO 的结果中比较最高 goodput。

### 2. 建议的实验顺序

| 步骤 | vLLM 重点 | SGLang 重点 | 观测指标 |
| --- | --- | --- | --- |
| 基线 | eager、无 spec、稳定 token budget | FCFS、无 overlap、稳定 prefill budget | Scheduler CPU、GPU gap、TTFT/ITL |
| Chunk | `max_num_scheduled_tokens`、prefill threshold | chunk size、mixed chunk、delayer | 长 Prompt TTFT、Decode p99 ITL |
| Cache | Prefix Cache on/off、hit length | FCFS/LPM/DFS、Radix hit | cached tokens、Prefill GPU time、公平性 |
| 压力 | KV utilization、max sequences | future reservation、max running | preempt/retract、recompute、p99 |
| 异步 | batch queue/async scheduling | overlap on/off | plan/commit CPU、launch gap、正确性 |
| Spec | draft K、acceptance | EAGLE/NGRAM、tree width | accepted length、临时 KV、TPOT |

### 3. 调度器必须保持的共同不变量

1. 只有物理资源落实的请求才能进入本轮执行计划。
2. 每份计划只能与对应 iteration 的 GPU 结果提交。
3. 未验证 speculative token 不能进入用户输出或共享 Prefix Cache。
4. KV 的 allocation、lock/refcount、commit 和 free 必须形成单一 ownership 闭环。
5. Abort 不能假设已 launch kernel 被撤销，late result 必须能安全丢弃。
6. TP/PP/DP 各 rank 必须看到相同 request order、token count 和 accepted path。
7. 优先级和 Cache locality 不能绕过 model length、grammar、LoRA、P/D readiness 等硬约束。

## 最终记忆模型

```text
vLLM：计算每个请求还欠 target model 多少 token，
      然后在 token/KV 等预算内偿还 debt，并在结果返回后修正乐观状态。

SGLang：先按 Priority/Radix locality 排序，
        再由 PrefillAdder 做多资源准入，
        最后在 EXTEND 与 DECODE batch 状态机中选择本轮计划。
```

调度优化真正减少的不是 Transformer 理论 FLOPs，而是排队、GPU 空洞、无效 Prefill、资源碎片和重算。最终评价标准应是满足 TTFT/ITL SLO 后的 goodput，而不是单独看平均 tokens/s。

[返回 vLLM 请求全流程]({VLLM_LIFECYCLE_URL}) · [返回 SGLang 请求全流程]({SGLANG_LIFECYCLE_URL}) · [进入面试题系列]({INTERVIEW_URL})
"""


def validate(text: str) -> None:
    required = [
        "## 先用 30 秒理解调度",
        "## Part I：vLLM 的统一 Token-Debt 调度",
        "### 3. 核心状态：统一 token debt",
        "### 4. `schedule()` 的实际实现流程",
        "### 5. `update_from_output()`：真正的提交点",
        "## Part II：SGLang 的 Policy、Admission 与 Batch 状态机",
        "### 4. SchedulePolicy",
        "### 5. PrefillAdder：多资源 admission ledger",
        "### 6. `get_next_batch_to_run()` 状态机",
        "## Part III：把两套设计放在同一张图里",
        "## Part IV：如何调参和验证",
    ]
    missing = [heading for heading in required if heading not in text]
    if missing:
        raise ValueError(f"scheduler post is missing sections: {missing}")

    for commit in (INFRA_COMMIT, VLLM_COMMIT, SGLANG_COMMIT):
        if commit not in text:
            raise ValueError(f"scheduler post is missing commit {commit}")

    if text.count("```") % 2:
        raise ValueError("scheduler post has unbalanced code fences")

    definitions = re.findall(r"(?m)^\[([^]]+)\]:\s+(\S+)", text)
    labels = [label for label, _ in definitions]
    if len(labels) != len(set(labels)):
        raise ValueError("scheduler post has duplicate reference labels")
    used = set(re.findall(r"\]\[([^]]+)\]", text))
    undefined = sorted(used - set(labels))
    if undefined:
        raise ValueError(f"scheduler post has undefined references: {undefined}")
    invalid_targets = [
        target
        for _, target in definitions
        if not target.startswith(("http://", "https://", "/", "#"))
    ]
    if invalid_targets:
        raise ValueError(f"scheduler post has invalid reference targets: {invalid_targets}")

    unresolved = [
        target
        for target in re.findall(r"\]\(([^)]+)\)", text)
        if not target.startswith(("http://", "https://", "/", "#", "mailto:"))
    ]
    if unresolved:
        raise ValueError(f"scheduler post has unresolved inline links: {unresolved}")


def main() -> None:
    vllm = prepare_module(
        read_module("vllm/modules/SCHEDULER.md"), "vllm", "vllm/modules"
    )
    sglang = prepare_module(
        read_module("sglang/modules/SCHEDULER.md"), "sglang", "sglang/modules"
    )
    body = (
        front_matter()
        + introduction()
        + vllm
        + "\n\n---\n\n## Part II：SGLang 的 Policy、Admission 与 Batch 状态机\n\n"
        + sglang
        + comparison()
    )
    validate(body)
    POST_PATH.parent.mkdir(parents=True, exist_ok=True)
    POST_PATH.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"generated and validated scheduler post: {POST_PATH}")


if __name__ == "__main__":
    main()
