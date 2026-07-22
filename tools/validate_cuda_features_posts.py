#!/usr/bin/env python3
"""Validate the 20-post CUDA Programming Guide chapter 4 series."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POST_ROOT = ROOT / "_posts" / "cuda-features"

POSTS = [
    (1, "cuda-features-01-unified-memory"),
    (2, "cuda-features-02-cuda-graphs"),
    (3, "cuda-features-03-stream-ordered-memory-allocator"),
    (4, "cuda-features-04-cooperative-groups"),
    (5, "cuda-features-05-programmatic-dependent-launch"),
    (6, "cuda-features-06-green-contexts"),
    (7, "cuda-features-07-lazy-loading"),
    (8, "cuda-features-08-error-log-management"),
    (9, "cuda-features-09-asynchronous-barriers"),
    (10, "cuda-features-10-pipelines"),
    (11, "cuda-features-11-asynchronous-data-copies"),
    (12, "cuda-features-12-cluster-launch-control"),
    (13, "cuda-features-13-l2-cache-control"),
    (14, "cuda-features-14-memory-synchronization-domains"),
    (15, "cuda-features-15-interprocess-communication"),
    (16, "cuda-features-16-virtual-memory-management"),
    (17, "cuda-features-17-extended-gpu-memory"),
    (18, "cuda-features-18-dynamic-parallelism"),
    (19, "cuda-features-19-graphics-interop"),
    (20, "cuda-features-20-driver-entry-point-access"),
]


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:
    expected_paths = [POST_ROOT / f"2026-07-22-{slug}.md" for _, slug in POSTS]
    actual_paths = sorted(POST_ROOT.glob("*.md"))
    if actual_paths != sorted(expected_paths):
        fail(f"expected exactly 20 CUDA feature posts, got {[path.name for path in actual_paths]}")

    routes = {f"/posts/{slug}/" for _, slug in POSTS}
    total_bytes = 0

    for index, ((section, slug), path) in enumerate(zip(POSTS, expected_paths)):
        text = path.read_text(encoding="utf-8")
        total_bytes += len(text.encode("utf-8"))

        if not text.startswith("---\n") or text.count("\n---\n", 0, 500) != 1:
            fail(f"{path.name}: invalid front matter")
        required = (
            f"title: 'CUDA Features 4.{section}",
            "categories: [AI Infra, CUDA Features]",
            "docs.nvidia.com/cuda/cuda-programming-guide/",
            "**总结：",
        )
        for marker in required:
            if marker not in text:
                fail(f"{path.name}: missing {marker}")

        if not re.search(rf"(?m)^## 4\.{section}(?:\.|\s)", text):
            fail(f"{path.name}: missing section 4.{section} content")
        if text.count("```") % 2:
            fail(f"{path.name}: unbalanced fenced code block")
        if len(text.encode("utf-8")) < 2_500:
            fail(f"{path.name}: content unexpectedly short")

        if index > 0:
            previous_route = f"/posts/{POSTS[index - 1][1]}/"
            if previous_route not in text:
                fail(f"{path.name}: missing previous navigation")
        if index < len(POSTS) - 1:
            next_route = f"/posts/{POSTS[index + 1][1]}/"
            if next_route not in text:
                fail(f"{path.name}: missing next navigation")

        for target in re.findall(r"\]\((/posts/[^)]+/)\)", text):
            if target.startswith("/posts/cuda-features-") and target not in routes:
                fail(f"{path.name}: unknown CUDA series route {target}")

    first = expected_paths[0].read_text(encoding="utf-8")
    if "pin: true" not in first or "## CUDA Features 中文系列目录" not in first:
        fail("first post must be the pinned series index")
    for route in routes:
        if route not in first:
            fail(f"series index missing {route}")

    final = expected_paths[-1].read_text(encoding="utf-8")
    if "## 全章知识图谱" not in final:
        fail("final post must include the chapter knowledge map")

    print(f"validated {len(expected_paths)} CUDA feature posts")
    print("chapter coverage: 4.1 through 4.20")
    print(f"content bytes: {total_bytes}")


if __name__ == "__main__":
    main()
