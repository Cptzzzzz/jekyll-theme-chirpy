#!/usr/bin/env python3
"""Publish the vLLM/SGLang interview notes from infra-codes as a blog series."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


BLOG_ROOT = Path(__file__).resolve().parents[1]
INFRA_ROOT = Path(os.environ.get("INFRA_CODES_ROOT", "/root/infra-codes"))
INFERENCE_ROOT = INFRA_ROOT / "inference"
POST_ROOT = BLOG_ROOT / "_posts" / "inference-interview"
ASSET_ROOT = BLOG_ROOT / "assets" / "files" / "vllm-sglang-interview"
INFRA_COMMIT = "1f8930ce3f206cdaac9ec24954f5a428e8c890f8"
INFRA_BLOB = f"https://github.com/Cptzzzzz/infra-codes/blob/{INFRA_COMMIT}/inference"
INFRA_TREE = f"https://github.com/Cptzzzzz/infra-codes/tree/{INFRA_COMMIT}/inference"

SERIES_URL = "/posts/vllm-sglang-interview-series/"
GUIDE_URL = "/posts/vllm-sglang-interview-comprehensive-guide/"
LABS_URL = "/posts/vllm-sglang-interview-source-labs/"

LEVELS = [
    {
        "number": 1,
        "range": "1-10",
        "slug": "vllm-sglang-interview-level-1-foundations",
        "title": "vLLM / SGLang 面试题（一）：推理服务基础",
        "short": "推理服务基础",
        "focus": "Prefill/Decode、在线指标、批处理、采样、并行与基础容量分析",
    },
    {
        "number": 2,
        "range": "11-20",
        "slug": "vllm-sglang-interview-level-2-lifecycle",
        "title": "vLLM / SGLang 面试题（二）：请求生命周期与模块契约",
        "short": "请求生命周期与模块契约",
        "focus": "HTTP 到 GPU 再到流式返回、IPC、状态所有权、取消与异步边界",
    },
    {
        "number": 3,
        "range": "21-30",
        "slug": "vllm-sglang-interview-level-3-scheduling",
        "title": "vLLM / SGLang 面试题（三）：调度、准入与资源预算",
        "short": "调度、准入与资源预算",
        "focus": "Continuous batching、token budget、chunked prefill、公平性与背压",
    },
    {
        "number": 4,
        "range": "31-40",
        "slug": "vllm-sglang-interview-level-4-kv-runner",
        "title": "vLLM / SGLang 面试题（四）：KV Cache 与 GPU 执行",
        "short": "KV Cache 与 GPU 执行细节",
        "focus": "Paged KV、Radix Cache、Beam Search、slot mapping、attention backend 与显存",
    },
    {
        "number": 5,
        "range": "41-50",
        "slug": "vllm-sglang-interview-level-5-advanced-distributed",
        "title": "vLLM / SGLang 面试题（五）：高级优化、分布式与性能诊断",
        "short": "高级优化、分布式与性能诊断",
        "focus": "投机解码、CUDA Graph、TP/EP、P/D 分离、量化与 benchmark",
    },
    {
        "number": 6,
        "range": "51-60",
        "slug": "vllm-sglang-interview-level-6-system-design",
        "title": "vLLM / SGLang 面试题（六）：Staff 级系统设计与现场推演",
        "short": "Staff 级系统设计与现场推演",
        "focus": "SLO、容量、故障恢复、正确性不变量、扩展点与演进路线",
    },
]


def read_source(name: str) -> str:
    path = INFERENCE_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"missing source document: {path}")
    return path.read_text(encoding="utf-8")


def strip_h1(text: str) -> str:
    return re.sub(r"\A# .+?\n+", "", text, count=1).strip()


def rewrite_links(text: str) -> str:
    """Rewrite infra-codes-relative links for the independently deployed blog."""

    blog_routes = {
        "VLLM_SGLANG_INTERVIEW_QUESTION_BANK.md": SERIES_URL,
        "VLLM_SGLANG_INTERVIEW_GUIDE.md": GUIDE_URL,
        "VLLM_SGLANG_SOURCE_LABS.md": LABS_URL,
        "experiments/2026-07-17-a100-qwen3-0.6b.json": (
            "/assets/files/vllm-sglang-interview/2026-07-17-a100-qwen3-0.6b.json"
        ),
        "experiments/README.md": (
            "/assets/files/vllm-sglang-interview/EXPERIMENTS.md"
        ),
    }

    def replace(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith(("http://", "https://", "/", "#", "mailto:")):
            return match.group(0)

        path, marker, fragment = target.partition("#")
        if path in blog_routes:
            rewritten = blog_routes[path]
        elif path.endswith(".md"):
            rewritten = f"{INFRA_BLOB}/{path}"
        elif path.startswith(("vllm/", "sglang/", "experiments/")):
            rewritten = f"{INFRA_TREE}/{path}"
        else:
            return match.group(0)

        if marker:
            rewritten = f"{rewritten}#{fragment}"
        return f"]({rewritten})"

    return re.sub(r"\]\(([^)]+)\)", replace, text)


def front_matter(title: str, date: str, *, pin: bool = False) -> str:
    pin_line = "pin: true\n" if pin else ""
    return (
        "---\n"
        f"title: '{title.replace(chr(39), chr(39) * 2)}'\n"
        "author: Cptz\n"
        f"date: 2026-07-20 {date}:00 +0800\n"
        "categories: [AI Infra, Inference]\n"
        "tags: [vllm, sglang, llm inference, interview, kv cache, scheduler, cuda]\n"
        "math: true\n"
        "mermaid: true\n"
        f"{pin_line}"
        "---\n\n"
        f"<!-- Generated from infra-codes commit {INFRA_COMMIT} by "
        "tools/sync_inference_interview_posts.py. -->\n\n"
    )


def write_post(filename: str, title: str, date: str, body: str, *, pin: bool = False) -> None:
    path = POST_ROOT / filename
    path.write_text(front_matter(title, date, pin=pin) + body.rstrip() + "\n", encoding="utf-8")


def level_navigation(index: int) -> str:
    links = []
    if index > 0:
        previous = LEVELS[index - 1]
        links.append(f"[上一篇：{previous['short']}](/posts/{previous['slug']}/)")
    links.append(f"[系列总索引]({SERIES_URL})")
    if index < len(LEVELS) - 1:
        following = LEVELS[index + 1]
        links.append(f"[下一篇：{following['short']}](/posts/{following['slug']}/)")
    return " · ".join(links)


def build_index(bank_intro: str) -> str:
    rows = "\n".join(
        f"| Level {level['number']} | {level['range']} | "
        f"[{level['short']}](/posts/{level['slug']}/) | {level['focus']} |"
        for level in LEVELS
    )
    return f"""## 系列说明

这套材料基于 vLLM 与 SGLang 的固定源码版本和真实运行实验整理。主线题库共 **60 题**，从推理基础递进到 Staff 级系统设计；另保留 **24 道长篇综合题**和一册源码实验手册。部分主题有意重复，因为分卷题库用于系统训练，综合题解用于模拟一题多轮深挖，回答结构和考察角度不同。

```mermaid
flowchart LR
    A[推理基础] --> B[请求生命周期]
    B --> C[调度与资源预算]
    C --> D[KV Cache 与 GPU 执行]
    D --> E[高级优化与分布式]
    E --> F[系统设计与现场推演]
    D --> G[源码实验]
    E --> G
```

## 六级题库

| 分卷 | 题号 | 文章 | 核心范围 |
| --- | --- | --- | --- |
{rows}

## 配套长文

- [vLLM 请求处理全流程](/posts/vllm-request-lifecycle-source-guide/)：从 HTTP、EngineCore、Scheduler、Paged KV、GPU Runner 一直跟到流式输出和释放。
- [SGLang 请求处理全流程](/posts/sglang-request-lifecycle-source-guide/)：从 Tokenizer、Scheduler、Radix Cache、EXTEND/DECODE 一直跟到 Detokenizer 和 HTTP coroutine。
- [24 道长篇综合题：完整答案与评分指南]({GUIDE_URL})：适合模拟 3 到 8 分钟的完整面试回答。
- [源码深挖与实验手册]({LABS_URL})：用 A100 实验进一步追问 Prefill、Prefix Cache、Beam Search KV、Continuous Batching、CUDA Graph、投机解码和 KV 压力。
- [实验聚合数据](/assets/files/vllm-sglang-interview/2026-07-17-a100-qwen3-0.6b.json)与[复现实验说明](/assets/files/vllm-sglang-interview/EXPERIMENTS.md)：用于检查结论的适用范围，而不是把单卡小模型数字当作框架排行榜。

## 阅读与作答方法

1. 第一遍每题限时 90 秒，按“问题、Input、方法、Output、收益、代价”组织回答。
2. 第二遍沿源码锚点画对象转换、进程边界和资源 ownership，不看答案复述完整链路。
3. 第三遍回答追问，必须给出公式、反例或可证伪实验，不能只复述框架术语。
4. 最后以固定 commit 的源码为准核对类、函数与行为；不要把新旧版本实现混在一个答案中。

{bank_intro}

## 固定源码版本

- vLLM trace commit [`d1d2f7535`](https://github.com/Cptzzzzz/vllm/commit/d1d2f7535686acdc39b034bf18ba52d07c21def5)，上游基线 `3b39fd284`
- SGLang trace commit [`9529c2e96`](https://github.com/Cptzzzzz/sglang/commit/9529c2e96351f3b1d5715eaa89989b20723d60b5)，上游基线 `a3194d358`
- 笔记来源 commit [`{INFRA_COMMIT[:9]}`](https://github.com/Cptzzzzz/infra-codes/commit/{INFRA_COMMIT})
"""


def validate_generated() -> None:
    level_text = "\n".join(
        (POST_ROOT / f"2026-07-20-{level['slug']}.md").read_text(encoding="utf-8")
        for level in LEVELS
    )
    question_numbers = [int(number) for number in re.findall(r"(?m)^### (\d+)\.", level_text)]
    if question_numbers != list(range(1, 61)):
        raise ValueError(f"expected ordered questions 1..60, got {question_numbers}")

    guide = (POST_ROOT / "2026-07-20-vllm-sglang-interview-comprehensive-guide.md").read_text(
        encoding="utf-8"
    )
    guide_questions = [int(number) for number in re.findall(r"(?m)^## (\d+)\.", guide)]
    if guide_questions != list(range(1, 25)):
        raise ValueError(f"expected ordered guide questions 1..24, got {guide_questions}")

    unresolved = []
    for path in sorted(POST_ROOT.glob("*.md")):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.endswith(".md") or ".md#" in target:
                if not target.startswith(("https://", "/assets/")):
                    unresolved.append(f"{path.name}: {target}")
    if unresolved:
        raise ValueError("unresolved relative Markdown links:\n" + "\n".join(unresolved))


def main() -> None:
    bank = read_source("VLLM_SGLANG_INTERVIEW_QUESTION_BANK.md")
    guide = read_source("VLLM_SGLANG_INTERVIEW_GUIDE.md")
    labs = read_source("VLLM_SGLANG_SOURCE_LABS.md")

    POST_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    level_matches = list(re.finditer(r"(?m)^## Level ([1-6])[:：].*$", bank))
    if len(level_matches) != 6:
        raise ValueError(f"expected six level sections, got {len(level_matches)}")

    intro_start = bank.index("\n", bank.index("# ")) + 1
    bank_intro = bank[intro_start : level_matches[0].start()].strip()
    bank_intro = re.sub(r"\n---\s*$", "", bank_intro).strip()
    bank_intro = rewrite_links(bank_intro)

    write_post(
        "2026-07-20-vllm-sglang-interview-series.md",
        "vLLM / SGLang 源码面试题系列：从基础到 Staff",
        "00:10",
        build_index(bank_intro),
        pin=True,
    )

    for index, (level, match) in enumerate(zip(LEVELS, level_matches)):
        end = level_matches[index + 1].start() if index + 1 < len(level_matches) else len(bank)
        section = bank[match.start() : end].strip()
        section = re.sub(r"\n---\s*$", "", section).strip()
        section = rewrite_links(section)
        body = f"""> 本文是[vLLM / SGLang 源码面试题系列]({SERIES_URL})第 {level['number']} 卷，覆盖第 {level['range']} 题。重点是{level['focus']}。

{level_navigation(index)}

---

{section}

---

{level_navigation(index)}
"""
        write_post(
            f"2026-07-20-{level['slug']}.md",
            level["title"],
            f"00:{11 + index:02d}",
            body,
        )

    guide_body = rewrite_links(strip_h1(guide))
    guide_intro = f"""> 本文保留 24 道综合题的长篇回答，适合模拟单题多轮深挖。按难度训练请从[60 题系列总索引]({SERIES_URL})开始；需要实验数据与源码边界时继续阅读[源码实验手册]({LABS_URL})。

---

"""
    write_post(
        "2026-07-20-vllm-sglang-interview-comprehensive-guide.md",
        "vLLM / SGLang 深度面试：24 道完整答案与评分指南",
        "00:17",
        guide_intro + guide_body,
    )

    labs_body = rewrite_links(strip_h1(labs))
    labs_intro = f"""> 本文是[面试题系列]({SERIES_URL})的源码实验卷。它不重复罗列术语，而是用固定版本源码和 A100 实验回答可以被继续追问的工程细节。

---

"""
    write_post(
        "2026-07-20-vllm-sglang-interview-source-labs.md",
        "vLLM / SGLang 面试源码深挖：优化机制与实验",
        "00:18",
        labs_intro + labs_body,
    )

    shutil.copy2(
        INFERENCE_ROOT / "experiments" / "2026-07-17-a100-qwen3-0.6b.json",
        ASSET_ROOT / "2026-07-17-a100-qwen3-0.6b.json",
    )
    shutil.copy2(
        INFERENCE_ROOT / "experiments" / "README.md",
        ASSET_ROOT / "EXPERIMENTS.md",
    )
    (ASSET_ROOT / "SOURCE_MANIFEST.md").write_text(
        "# vLLM / SGLang interview series source manifest\n\n"
        f"- infra-codes commit: `{INFRA_COMMIT}`\n"
        "- main question bank: `inference/VLLM_SGLANG_INTERVIEW_QUESTION_BANK.md` (60 questions)\n"
        "- comprehensive guide: `inference/VLLM_SGLANG_INTERVIEW_GUIDE.md` (24 questions)\n"
        "- source labs: `inference/VLLM_SGLANG_SOURCE_LABS.md`\n"
        "- experiment: `inference/experiments/2026-07-17-a100-qwen3-0.6b.json`\n",
        encoding="utf-8",
    )

    validate_generated()
    print(f"generated {len(list(POST_ROOT.glob('*.md')))} posts in {POST_ROOT}")
    print("validated 60 level questions and 24 comprehensive questions")


if __name__ == "__main__":
    main()
