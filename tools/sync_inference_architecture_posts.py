#!/usr/bin/env python3
"""Publish fixed-commit vLLM and SGLang request lifecycle notes."""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path


BLOG_ROOT = Path(__file__).resolve().parents[1]
INFRA_ROOT = Path(os.environ.get("INFRA_CODES_ROOT", "/root/infra-codes"))
INFERENCE_ROOT = INFRA_ROOT / "inference"
POST_ROOT = BLOG_ROOT / "_posts" / "inference-architecture"
INFRA_COMMIT = "1f8930ce3f206cdaac9ec24954f5a428e8c890f8"
INFRA_BLOB = f"https://github.com/Cptzzzzz/infra-codes/blob/{INFRA_COMMIT}/inference"

VLLM_URL = "/posts/vllm-request-lifecycle-source-guide/"
SGLANG_URL = "/posts/sglang-request-lifecycle-source-guide/"
INTERVIEW_URL = "/posts/vllm-sglang-interview-series/"
LABS_URL = "/posts/vllm-sglang-interview-source-labs/"

POSTS = [
    {
        "framework": "vLLM",
        "source": "vllm/ARCHITECTURE.md",
        "filename": "2026-07-20-vllm-request-lifecycle-source-guide.md",
        "title": "vLLM 请求处理全流程：从 HTTP 到 Scheduler、GPU 与流式输出",
        "date": "00:19:00",
        "framework_commit": "d1d2f7535686acdc39b034bf18ba52d07c21def5",
        "peer_name": "SGLang",
        "peer_url": SGLANG_URL,
        "overview": """```text
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

一句话理解：API 进程拥有客户端语义，EngineCore/Scheduler 拥有逻辑请求与资源决策，rank-local Runner 拥有设备执行状态；`SchedulerOutput` 是一次执行计划，`ModelRunnerOutput` 是待提交结果，二者必须严格配对。""",
    },
    {
        "framework": "SGLang",
        "source": "sglang/ARCHITECTURE.md",
        "filename": "2026-07-20-sglang-request-lifecycle-source-guide.md",
        "title": "SGLang 请求处理全流程：从 Tokenizer 到 Scheduler、Radix 与输出",
        "date": "00:20:00",
        "framework_commit": "9529c2e96351f3b1d5715eaa89989b20723d60b5",
        "peer_name": "vLLM",
        "peer_url": VLLM_URL,
        "overview": """```text
HTTP / OpenAI Chat Completion
  -> TokenizerManager：建立 ReqState、tokenize、多模态预处理
  -> ZMQ transport：发送 TokenizedGenerateReqInput
  -> Scheduler：DTO 转 Req，管理 waiting/running 与 batch 状态机
  -> RadixCache + SchedulePolicy：prefix match、cache-aware 排序
  -> PrefillAdder：token/KV/request-row 多资源 admission
  -> ScheduleBatch / TpModelWorker / ModelRunner：EXTEND 或 DECODE forward、sample
  -> Scheduler result processor：提交 token、Radix/KV 状态与停止条件
  -> DetokenizerManager：独立进程增量 detokenize
  -> TokenizerManager：按 rid 唤醒 HTTP coroutine，返回 JSON / SSE
```

一句话理解：SGLang 用 Tokenizer、Scheduler、Detokenizer 多进程隔离输入输出 CPU 工作，把 Radix prefix reuse、准入与 batch 状态集中在 Scheduler；overlap 模式再用 batch snapshot、FutureMap 和 CUDA event 保证跨轮异步执行仍能正确提交。""",
    },
]


def strip_h1(text: str) -> str:
    return re.sub(r"\A# .+?\n+", "", text, count=1).strip()


def rewrite_links(text: str, source_dir: str) -> str:
    """Convert infra-codes-relative Markdown links to stable public links."""

    routes = {
        "../VLLM_SGLANG_SOURCE_LABS.md": LABS_URL,
    }

    def replace_inline(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith(("http://", "https://", "/", "#", "mailto:")):
            return match.group(0)

        path, marker, fragment = target.partition("#")
        if path in routes:
            rewritten = routes[path]
        else:
            normalized = posixpath.normpath(f"{source_dir}/{path}")
            rewritten = f"{INFRA_BLOB}/{normalized}"
        if marker:
            rewritten = f"{rewritten}#{fragment}"
        return f"]({rewritten})"

    return re.sub(r"\]\(([^)]+)\)", replace_inline, text)


def front_matter(title: str, date: str) -> str:
    escaped_title = title.replace("'", "''")
    return f"""---
title: '{escaped_title}'
author: Cptz
date: 2026-07-20 {date} +0800
categories: [AI Infra, Inference]
tags: [vllm, sglang, llm inference, architecture, scheduler, kv cache, cuda]
math: true
mermaid: true
---

<!-- Generated from infra-codes commit {INFRA_COMMIT} by tools/sync_inference_architecture_posts.py. -->

"""


def build_intro(post: dict[str, str]) -> str:
    return f"""> 本文按固定源码版本逐段跟踪一个 {post['framework']} 请求。先用 30 秒建立全局路径，再按模块解释问题、Input、实现、Output、性能收益、代价与正确性不变量。

[另一篇：{post['peer_name']} 请求全流程]({post['peer_url']}) · [面试题系列]({INTERVIEW_URL}) · [源码实验手册]({LABS_URL})

## 30 秒全流程概览

{post['overview']}

下面进入完整源码路径。文中的类、函数和状态均固定到文章开头声明的 fork commit；伪代码用于压缩控制流，不替代链接中的源码条件分支。

---

"""


def validate(posts: list[Path]) -> None:
    required = {
        "vLLM": [
            "## 3. 完整生命周期总览",
            "## 7. T6-T7：Scheduler 的统一 token-debt 模型",
            "## 9. T6：Paged KV Cache、Prefix Cache 与物理内存",
            "## 10. T8-T13：从 SchedulerOutput 到一次 GPU forward",
            "## 11. T14-T17：提交结果、detokenize、stream 与结束",
            "## 13. Speculative Decoding：从理论到 vLLM 源码",
            "## 17. 关键正确性不变量清单",
        ],
        "SGLang": [
            "## 3. 完整请求生命周期总览",
            "## 7. T5：RadixAttention 如何查找共享 prefix",
            "## 8. T5-T7：SchedulePolicy 与 PrefillAdder",
            "## 10. 普通 event loop 与 overlap event loop",
            "## 12. T8-T12：TpModelWorker、ModelRunner、graph 与 sample",
            "## 15. Speculative Decoding：SGLang EAGLE V2 完整流程",
            "## 20. 关键正确性不变量",
        ],
    }

    post_by_filename = {post["filename"]: post for post in POSTS}
    for path in posts:
        text = path.read_text(encoding="utf-8")
        framework = "SGLang" if "sglang-" in path.name else "vLLM"
        post = post_by_filename[path.name]
        missing = [heading for heading in required[framework] if heading not in text]
        if missing:
            raise ValueError(f"{path.name} is missing sections: {missing}")
        if text.count("```text") < 5:
            raise ValueError(f"{path.name} does not contain enough algorithm pseudocode")
        if text.count("```") % 2:
            raise ValueError(f"{path.name} has unbalanced code fences")
        if INFRA_COMMIT not in text:
            raise ValueError(f"{path.name} does not identify its source note commit")
        if post["framework_commit"] not in text:
            raise ValueError(f"{path.name} does not identify its framework commit")

        unresolved = []
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("http://", "https://", "/", "#", "mailto:")):
                unresolved.append(target)
        if unresolved:
            raise ValueError(f"{path.name} has unresolved links: {unresolved}")

        reference_targets = re.findall(r"(?m)^\[[^]]+\]:\s+(\S+)", text)
        invalid_references = [
            target
            for target in reference_targets
            if not target.startswith(("http://", "https://", "/", "#"))
        ]
        if invalid_references:
            raise ValueError(
                f"{path.name} has unresolved reference links: {invalid_references}"
            )


def main() -> None:
    POST_ROOT.mkdir(parents=True, exist_ok=True)
    generated = []
    for post in POSTS:
        source_path = INFERENCE_ROOT / post["source"]
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source document: {source_path}")
        source_body = strip_h1(source_path.read_text(encoding="utf-8"))
        source_body = rewrite_links(source_body, str(Path(post["source"]).parent))
        output_path = POST_ROOT / post["filename"]
        output_path.write_text(
            front_matter(post["title"], post["date"])
            + build_intro(post)
            + source_body.rstrip()
            + "\n",
            encoding="utf-8",
        )
        generated.append(output_path)

    validate(generated)
    print(f"generated and validated {len(generated)} architecture posts")


if __name__ == "__main__":
    main()
