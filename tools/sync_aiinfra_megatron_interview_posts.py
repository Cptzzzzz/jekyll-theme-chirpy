#!/usr/bin/env python3
"""Publish the V100 AIInfra and Megatron interview banks as blog series."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path


BLOG_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = Path(os.environ.get("V100_LLM_LAB_ROOT", "/root/v100-llm-lab"))
REPORT_ROOT = LAB_ROOT / "report"
POST_DATE = "2026-07-20"

AI_POST_ROOT = BLOG_ROOT / "_posts" / "aiinfra-interview"
MEGATRON_POST_ROOT = BLOG_ROOT / "_posts" / "megatron-interview"
ASSET_ROOT = BLOG_ROOT / "assets" / "files" / "v100-aiinfra-interview"

AI_SOURCE = REPORT_ROOT / "aiinfra-interview-question-bank.md"
MEGATRON_SOURCE = REPORT_ROOT / "megatron-interview-question-bank.md"
EXPERIMENT_REPORT = REPORT_ROOT / "experiment-report.md"
ENGINEERING_REPORT = REPORT_ROOT / "engineering-deep-dive-report.md"
VALIDATION_RESULT = LAB_ROOT / "results" / "summary" / "validation.json"

AI_SERIES_URL = "/posts/aiinfra-interview-series/"
MEGATRON_SERIES_URL = "/posts/megatron-interview-series/"
ASSET_URL = "/assets/files/v100-aiinfra-interview"

AI_LEVELS = [
    {
        "section": 2,
        "range": "Q01-Q12",
        "slug": "aiinfra-interview-level-1-gpu-performance",
        "title": "AI Infra 面试题（一）：GPU、通信与性能基础",
        "short": "GPU 与性能基础",
        "focus": "GPU 瓶颈、显存、Roofline、拓扑、collective、精度与 benchmark",
    },
    {
        "section": 3,
        "range": "Q13-Q26",
        "slug": "aiinfra-interview-level-2-training-runtime",
        "title": "AI Infra 面试题（二）：DDP、FSDP 与训练运行时",
        "short": "DDP、FSDP 与训练运行时",
        "focus": "梯度同步、bucket overlap、ZeRO/FSDP、checkpoint 与数值诊断",
    },
    {
        "section": 4,
        "range": "Q27-Q40",
        "slug": "aiinfra-interview-level-3-model-parallel",
        "title": "AI Infra 面试题（三）：大模型多维并行",
        "short": "TP、SP、CP、PP 与 EP",
        "focus": "Tensor、Sequence、Context、Pipeline、Expert Parallel 及组合选型",
    },
    {
        "section": 5,
        "range": "Q41-Q54",
        "slug": "aiinfra-interview-level-4-inference-serving",
        "title": "AI Infra 面试题（四）：LLM 推理引擎与在线服务",
        "short": "推理引擎与在线服务",
        "focus": "KV Cache、PagedAttention、调度、SLO、P/D 分离、投机解码与量化",
    },
    {
        "section": 6,
        "range": "Q55-Q64",
        "slug": "aiinfra-interview-level-5-kernel-compiler",
        "title": "AI Infra 面试题（五）：GPU Kernel、编译与性能工程",
        "short": "Kernel、编译与性能工程",
        "focus": "Profiling、算术强度、FlashAttention、Triton、occupancy 与图编译",
    },
    {
        "section": 7,
        "range": "Q65-Q74",
        "slug": "aiinfra-interview-level-6-production-reliability",
        "title": "AI Infra 面试题（六）：生产集群、可靠性与 RL Infra",
        "short": "集群、可靠性与 RL Infra",
        "focus": "可观测性、hang/straggler/OOM、恢复、数据、放置、过载与 RL 系统",
    },
    {
        "section": 8,
        "range": "Q75-Q84",
        "slug": "aiinfra-interview-level-7-project-defense",
        "title": "AI Infra 面试题（七）：8 卡 V100 项目答辩",
        "short": "项目答辩与高压追问",
        "focus": "实验数字、证据边界、反直觉结果和项目价值的现场防守",
    },
    {
        "section": 9,
        "range": "Q85-Q90",
        "slug": "aiinfra-interview-level-8-system-design",
        "title": "AI Infra 面试题（八）：训练与推理系统设计",
        "short": "系统设计",
        "focus": "70B/MoE 训练、在线推理、百万上下文、checkpoint 与自动并行搜索",
    },
    {
        "section": 10,
        "range": "Q91-Q115",
        "slug": "aiinfra-interview-level-9-engineering-deep-dive",
        "title": "AI Infra 面试题（九）：工程实现与实验深挖",
        "short": "工程实现与实验深挖",
        "focus": "DDP/NCCL/MoE/checkpoint/Triton/vLLM 实验的实现与归因细节",
    },
    {
        "section": 11,
        "range": "Q116-Q145",
        "slug": "aiinfra-interview-level-10-runtime-contracts",
        "title": "AI Infra 面试题（十）：框架运行时与底层契约",
        "short": "运行时与底层契约",
        "focus": "Process group、stream、FSDP 状态机、KV 抢占和 kernel 资源证据",
    },
]

MEGATRON_LEVELS = [
    {
        "section": 2,
        "level": 0,
        "range": "M001-M012",
        "slug": "megatron-interview-level-0-architecture",
        "title": "Megatron 面试题（零）：全局架构与核心抽象",
        "short": "全局架构",
        "focus": "Megatron-LM/MCore/TE 边界、package、parallel state、配置与训练 iteration",
    },
    {
        "section": 3,
        "level": 1,
        "range": "M013-M024",
        "slug": "megatron-interview-level-1-model-dataflow",
        "title": "Megatron 面试题（一）：模型抽象与训练数据流",
        "short": "模型抽象与数据流",
        "focus": "Transformer layer、QKV、GQA、RoPE、SwiGLU、vocab parallel 与精度状态",
    },
    {
        "section": 4,
        "level": 2,
        "range": "M025-M040",
        "slug": "megatron-interview-level-2-tensor-sequence-parallel",
        "title": "Megatron 面试题（二）：Tensor 与 Sequence Parallel",
        "short": "TP 与 SP",
        "focus": "Column/Row Parallel、mapping autograd、通信量、切分约束与 V100 实测",
    },
    {
        "section": 5,
        "level": 3,
        "range": "M041-M054",
        "slug": "megatron-interview-level-3-pipeline-parallel",
        "title": "Megatron 面试题（三）：Pipeline Parallel 调度",
        "short": "PP 调度与负载均衡",
        "focus": "GPipe/1F1B、bubble、microbatch、P2P、VPP 和 stage imbalance",
    },
    {
        "section": 6,
        "level": 4,
        "range": "M055-M068",
        "slug": "megatron-interview-level-4-distributed-optimizer",
        "title": "Megatron 面试题（四）：DDP 与 Distributed Optimizer",
        "short": "DDP、分布式优化器与 FSDP",
        "focus": "Main grad、contiguous buffer、通信 overlap、状态分片、norm 与 overflow",
    },
    {
        "section": 7,
        "level": 5,
        "range": "M069-M080",
        "slug": "megatron-interview-level-5-context-long-sequence",
        "title": "Megatron 面试题（五）：Context Parallel 与长序列",
        "short": "CP、重计算与长序列",
        "focus": "SP/CP 边界、attention 通信、recompute、offload、packing 与 batch 联动",
    },
    {
        "section": 8,
        "level": 6,
        "range": "M081-M096",
        "slug": "megatron-interview-level-6-moe-expert-parallel",
        "title": "Megatron 面试题（六）：MoE 与 Expert Parallel",
        "short": "MoE 与 EP",
        "focus": "Router、容量、token dispatch、All-to-All、GroupedGEMM 与负载均衡",
    },
    {
        "section": 9,
        "level": 7,
        "range": "M097-M110",
        "slug": "megatron-interview-level-7-performance-precision",
        "title": "Megatron 面试题（七）：性能、融合、精度与通信重叠",
        "short": "性能、精度与 overlap",
        "focus": "MFU、FLOPs、Transformer Engine、fusion、CUDA Graph、FP8 与拓扑映射",
    },
    {
        "section": 10,
        "level": 8,
        "range": "M111-M124",
        "slug": "megatron-interview-level-8-checkpoint-reliability",
        "title": "Megatron 面试题（八）：Checkpoint、数据与可靠性",
        "short": "Checkpoint 与可靠性",
        "focus": "Sharded state、reshard、原子提交、可复现性、hang/NaN、数据与回归 CI",
    },
    {
        "section": 11,
        "level": 9,
        "range": "M125-M140",
        "slug": "megatron-interview-level-9-expert-systems",
        "title": "Megatron 面试题（九）：专家系统设计、源码与事故题",
        "short": "专家系统设计与源码",
        "focus": "70B/405B/MoE/128K 方案、OOM/性能/数值事故、源码追踪与专家表达",
    },
]


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing source: {path}")
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_numbered_sections(text: str) -> dict[int, str]:
    matches = list(re.finditer(r"(?m)^## (\d+)\. .+$", text))
    if not matches:
        raise ValueError("source has no numbered H2 sections")

    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[number] = text[match.start() : end].strip()
    return sections


def rewrite_links(text: str) -> str:
    routes = {
        "megatron-interview-question-bank.md": MEGATRON_SERIES_URL,
        "aiinfra-interview-question-bank.md": AI_SERIES_URL,
        "experiment-report.md": f"{ASSET_URL}/EXPERIMENT_REPORT.md",
        "engineering-deep-dive-report.md": f"{ASSET_URL}/ENGINEERING_DEEP_DIVE_REPORT.md",
    }

    def replace(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "/", "#", "mailto:")):
            return match.group(0)
        path, marker, fragment = target.partition("#")
        if path not in routes:
            return match.group(0)
        rewritten = routes[path]
        if marker:
            rewritten = f"{rewritten}#{fragment}"
        return f"[{label}]({rewritten})"

    return re.sub(r"\[([^]]+)\]\(([^)]+)\)", replace, text)


def front_matter(title: str, minute: int, kind: str, *, pin: bool = False) -> str:
    if kind == "aiinfra":
        categories = "[AI Infra, Interview]"
        tags = "[ai infra, distributed training, llm inference, gpu, nccl, interview]"
    else:
        categories = "[AI Infra, Megatron]"
        tags = "[megatron, megatron core, distributed training, model parallel, interview]"
    pin_line = "pin: true\n" if pin else ""
    return (
        "---\n"
        f"title: '{title.replace(chr(39), chr(39) * 2)}'\n"
        "author: Cptz\n"
        f"date: {POST_DATE} 16:{minute:02d}:00 +0800\n"
        f"categories: {categories}\n"
        f"tags: {tags}\n"
        "math: true\n"
        "mermaid: true\n"
        f"{pin_line}"
        "---\n\n"
        "<!-- Generated by tools/sync_aiinfra_megatron_interview_posts.py. -->\n\n"
    )


def write_post(root: Path, slug: str, title: str, minute: int, body: str, kind: str, *, pin: bool = False) -> Path:
    path = root / f"{POST_DATE}-{slug}.md"
    path.write_text(front_matter(title, minute, kind, pin=pin) + body.rstrip() + "\n", encoding="utf-8")
    return path


def navigation(levels: list[dict[str, object]], index: int, series_url: str) -> str:
    links = []
    if index > 0:
        previous = levels[index - 1]
        links.append(f"[上一篇：{previous['short']}](/posts/{previous['slug']}/)")
    links.append(f"[系列总索引]({series_url})")
    if index < len(levels) - 1:
        following = levels[index + 1]
        links.append(f"[下一篇：{following['short']}](/posts/{following['slug']}/)")
    return " · ".join(links)


def series_table(levels: list[dict[str, object]]) -> str:
    return "\n".join(
        f"| Level {level.get('level', index + 1)} | {level['range']} | "
        f"[{level['short']}](/posts/{level['slug']}/) | {level['focus']} |"
        for index, level in enumerate(levels)
    )


def ai_index(section_one: str) -> str:
    return f"""## 系列说明

这套题库以 8 张 Tesla V100 PCIe 的真实实验为证据底座，共包含 **145 道核心题**和 **8 道编码/上机题**。内容从 GPU 基础、训练运行时、多维并行和推理引擎，逐步深入到 kernel、生产可靠性、系统设计和框架底层契约。

```mermaid
flowchart LR
    A[GPU 与通信] --> B[训练运行时]
    B --> C[多维并行]
    C --> D[推理引擎]
    D --> E[Kernel 与编译]
    E --> F[生产可靠性]
    F --> G[系统设计]
    G --> H[底层契约]
```

## 十级题库

| 分卷 | 题号 | 文章 | 核心范围 |
| --- | --- | --- | --- |
{series_table(AI_LEVELS)}

## 配套内容

- [编码题、岗位选题、六周训练计划与评分表](/posts/aiinfra-interview-practical-guide/)
- [Megatron M001-M140 专家题库]({MEGATRON_SERIES_URL})
- [vLLM / SGLang 源码面试题系列](/posts/vllm-sglang-interview-series/)
- [NCCL 面试题与评分指南](/posts/nccl-interview-guide/)
- [V100 实验报告]({ASSET_URL}/EXPERIMENT_REPORT.md)
- [工程深挖补充报告]({ASSET_URL}/ENGINEERING_DEEP_DIVE_REPORT.md)
- [题库与实验来源清单]({ASSET_URL}/SOURCE_MANIFEST.md)

{section_one}
"""


def megatron_index(section_one: str) -> str:
    return f"""## 系列说明

这套专项题库包含 **M001-M140 共 140 道 Megatron 面试题**。每道题都提供前置知识、逐步分析、10 分主问题评分、完整追问回答和 5 分追问评分，目标是从会使用配置推进到能解释张量、通信、状态、源码和生产决策。

```mermaid
flowchart LR
    A[MCore 架构] --> B[模型数据流]
    B --> C[TP 与 SP]
    C --> D[PP 调度]
    D --> E[分布式优化器]
    E --> F[CP 与长序列]
    F --> G[MoE 与 EP]
    G --> H[性能与精度]
    H --> I[可靠性]
    I --> J[专家系统设计]
```

## 十级题库

| 分卷 | 题号 | 文章 | 核心范围 |
| --- | --- | --- | --- |
{series_table(MEGATRON_LEVELS)}

## 配套内容

- [十二组实验、连续深挖链、四周计划与源码阅读路径](/posts/megatron-interview-study-labs/)
- [综合 AI Infra Q01-Q145 题库]({AI_SERIES_URL})
- [V100 实验报告]({ASSET_URL}/EXPERIMENT_REPORT.md)
- [题库与实验来源清单]({ASSET_URL}/SOURCE_MANIFEST.md)

{section_one}
"""


def render_level(
    section: str,
    levels: list[dict[str, object]],
    index: int,
    series_url: str,
    series_name: str,
) -> str:
    level = levels[index]
    nav = navigation(levels, index, series_url)
    return f"""> 本文是[{series_name}]({series_url})的 Level {level.get('level', index + 1)}，覆盖 {level['range']}。重点是{level['focus']}。

{nav}

---

{section}

---

{nav}
"""


def publish_ai(source: str) -> list[Path]:
    sections = split_numbered_sections(source)
    if sorted(sections) != list(range(1, 19)):
        raise ValueError(f"expected AIInfra sections 1..18, got {sorted(sections)}")

    paths = [
        write_post(
            AI_POST_ROOT,
            "aiinfra-interview-series",
            "AI Infra 面试题系列：从 GPU 基础到生产系统",
            0,
            ai_index(rewrite_links(sections[1])),
            "aiinfra",
            pin=True,
        )
    ]
    for index, level in enumerate(AI_LEVELS):
        section = rewrite_links(sections[int(level["section"])])
        paths.append(
            write_post(
                AI_POST_ROOT,
                str(level["slug"]),
                str(level["title"]),
                index + 1,
                render_level(section, AI_LEVELS, index, AI_SERIES_URL, "AI Infra 面试题系列"),
                "aiinfra",
            )
        )

    appendix = "\n\n".join(rewrite_links(sections[number]) for number in range(12, 19))
    paths.append(
        write_post(
            AI_POST_ROOT,
            "aiinfra-interview-practical-guide",
            "AI Infra 面试实战：上机题、岗位选题与六周训练计划",
            11,
            f"""> 本文是[AI Infra 面试题系列]({AI_SERIES_URL})的实践附录，包含 C01-C08 上机题、岗位选题、训练计划、评分表、证据表达和最终自检。

[系列总索引]({AI_SERIES_URL})

---

{appendix}
""",
            "aiinfra",
        )
    )
    return paths


def publish_megatron(source: str) -> list[Path]:
    sections = split_numbered_sections(source)
    if sorted(sections) != list(range(1, 18)):
        raise ValueError(f"expected Megatron sections 1..17, got {sorted(sections)}")

    paths = [
        write_post(
            MEGATRON_POST_ROOT,
            "megatron-interview-series",
            "Megatron 专家面试题系列：架构、并行与大规模训练",
            12,
            megatron_index(rewrite_links(sections[1])),
            "megatron",
            pin=True,
        )
    ]
    for index, level in enumerate(MEGATRON_LEVELS):
        section = rewrite_links(sections[int(level["section"])])
        paths.append(
            write_post(
                MEGATRON_POST_ROOT,
                str(level["slug"]),
                str(level["title"]),
                index + 13,
                render_level(
                    section,
                    MEGATRON_LEVELS,
                    index,
                    MEGATRON_SERIES_URL,
                    "Megatron 专家面试题系列",
                ),
                "megatron",
            )
        )

    appendix = "\n\n".join(rewrite_links(sections[number]) for number in range(12, 18))
    paths.append(
        write_post(
            MEGATRON_POST_ROOT,
            "megatron-interview-study-labs",
            "Megatron 专家训练：实验、深挖链与源码阅读路径",
            23,
            f"""> 本文是[Megatron 专家面试题系列]({MEGATRON_SERIES_URL})的训练附录，包含十二组实验、六条连续深挖链、四周计划、源码阅读路径和最终自检。

[系列总索引]({MEGATRON_SERIES_URL})

---

{appendix}
""",
            "megatron",
        )
    )
    return paths


def copy_assets() -> None:
    copies = {
        AI_SOURCE: ASSET_ROOT / "AIINFRA_INTERVIEW_QUESTION_BANK.md",
        MEGATRON_SOURCE: ASSET_ROOT / "MEGATRON_INTERVIEW_QUESTION_BANK.md",
        EXPERIMENT_REPORT: ASSET_ROOT / "EXPERIMENT_REPORT.md",
        ENGINEERING_REPORT: ASSET_ROOT / "ENGINEERING_DEEP_DIVE_REPORT.md",
        VALIDATION_RESULT: ASSET_ROOT / "VALIDATION.json",
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing asset source: {source}")
        shutil.copy2(source, destination)

    manifest_lines = [
        "# V100 AIInfra interview series source manifest",
        "",
        "Generated from `/root/v100-llm-lab` on 2026-07-20.",
        "",
    ]
    for source, destination in copies.items():
        manifest_lines.append(
            f"- `{destination.name}`: source `{source.relative_to(LAB_ROOT)}`, sha256 `{sha256(source)}`"
        )
    (ASSET_ROOT / "SOURCE_MANIFEST.md").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )


def validate_generated(paths: list[Path]) -> None:
    if len(paths) != 24:
        raise ValueError(f"expected 24 generated posts, got {len(paths)}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("generated duplicate post filenames")

    ai_text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.parent == AI_POST_ROOT)
    megatron_text = "\n".join(
        path.read_text(encoding="utf-8") for path in paths if path.parent == MEGATRON_POST_ROOT
    )

    ai_questions = [int(number) for number in re.findall(r"(?m)^### Q(\d{2,3})\.", ai_text)]
    if ai_questions != list(range(1, 146)):
        raise ValueError(f"expected ordered AIInfra questions 1..145, got {ai_questions}")
    coding_questions = [int(number) for number in re.findall(r"(?m)^### C(\d{2})\.", ai_text)]
    if coding_questions != list(range(1, 9)):
        raise ValueError(f"expected ordered coding questions 1..8, got {coding_questions}")

    megatron_questions = [int(number) for number in re.findall(r"(?m)^### M(\d{3})\.", megatron_text)]
    if megatron_questions != list(range(1, 141)):
        raise ValueError(f"expected ordered Megatron questions 1..140, got {megatron_questions}")

    detailed_markers = (
        "**前置知识与分析过程**",
        "**主问题得分点（10分）**",
        "**追问分析与参考回答**",
        "**追问得分点（5分）**",
    )
    question_sections = re.split(r"(?m)^### M\d{3}\.", megatron_text)[1:]
    for marker in detailed_markers:
        count = sum(marker in section for section in question_sections)
        if count != 140:
            raise ValueError(f"expected {marker} in all 140 Megatron questions, got {count}")

    known_routes = {f"/posts/{path.name[len(POST_DATE) + 1:-3]}/" for path in paths}
    known_routes.update(
        {
            "/posts/vllm-sglang-interview-series/",
            "/posts/vllm-request-lifecycle-source-guide/",
            "/posts/sglang-request-lifecycle-source-guide/",
            "/posts/nccl-interview-guide/",
        }
    )
    unresolved = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            unresolved.append(f"{path.name}: missing front matter")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            clean_target = target.split("#", 1)[0]
            if clean_target.startswith("/posts/") and clean_target not in known_routes:
                unresolved.append(f"{path.name}: unknown post route {clean_target}")
            if clean_target.endswith(".md") and not clean_target.startswith(("https://", ASSET_URL)):
                unresolved.append(f"{path.name}: unresolved Markdown link {target}")
    if unresolved:
        raise ValueError("generated content validation failed:\n" + "\n".join(unresolved))


def main() -> None:
    AI_POST_ROOT.mkdir(parents=True, exist_ok=True)
    MEGATRON_POST_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    ai_source = read(AI_SOURCE)
    megatron_source = read(MEGATRON_SOURCE)
    paths = publish_ai(ai_source) + publish_megatron(megatron_source)
    copy_assets()
    validate_generated(paths)

    print(f"generated and validated {len(paths)} posts")
    print("AIInfra coverage: Q01-Q145 and C01-C08")
    print("Megatron coverage: M001-M140 with detailed answers and follow-up scoring")


if __name__ == "__main__":
    main()
