# A100 单卡源码实验复现

本目录保存 2026-07-17 深挖实验的聚合结果。完整结论见
[`../VLLM_SGLANG_SOURCE_LABS.md`](../VLLM_SGLANG_SOURCE_LABS.md)。原始 benchmark JSON 在实验机
`/tmp/aiinfra-deep-lab/`，只将去除请求文本后的聚合数据提交到仓库。

## 1. 环境前置

本机存在 `/root/vllm/.venv`、`/root/sglang/.venv` 和已缓存的 `Qwen/Qwen3-0.6B`。localhost 请求与
SGLang self-warmup 必须绕过环境代理：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

直接传本地 snapshot path 可避免框架为量化配置等可选文件发起 Hugging Face HEAD 请求。服务端仍用
`--served-model-name Qwen/Qwen3-0.6B` 暴露稳定 API model name。

## 2. Server 配置

vLLM eager：

```bash
cd /root/vllm
.venv/bin/vllm serve "$MODEL" \
  --served-model-name Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 18000 \
  --gpu-memory-utilization .70 --max-model-len 8192 \
  --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --enforce-eager
```

vLLM graph 使用相同命令，移除 `--enforce-eager`。Chunk 实验只将
`--max-num-batched-tokens` 分别设为 8192 和 512。

SGLang eager：

```bash
cd /root/sglang
.venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 18001 \
  --mem-fraction-static .70 --context-length 8192 \
  --max-running-requests 32 --chunked-prefill-size 8192 \
  --attention-backend triton --enable-metrics \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled
```

SGLang graph 使用相同命令，移除两个 `disabled` 参数。Chunk/mixed 对照分别使用：

```text
--chunked-prefill-size 8192
--chunked-prefill-size 512
--chunked-prefill-size 512 --enable-mixed-chunk
```

NGRAM 对照在 graph server 上增加：

```text
--speculative-algorithm NGRAM --speculative-num-draft-tokens 8
```

当前源码会为 NGRAM 自动关闭 mixed chunk，并将 decode graph 替换为每请求 8 token 的 target-verify graph。

## 3. 基准负载

Prefill sweep：20 个唯一随机 prompt，并发 1，input length 为 256/1024/4096，只生成 1 token。

Decode sweep：16 个唯一随机 prompt，256 input + 32 output，并发为 1/4/16。服务使用 OpenAI-compatible
endpoint，数据由 vLLM benchmark client 的 `random` dataset 产生。

Prefix 对照：先调用 flush-cache endpoint，再发两个直接 token-ID 请求；query length 为 2080，第二个请求与
第一个完全相同。必须同时读取 framework server metrics 中的 cached/matched tokens，不能只按客户端延迟猜
cache hit。

Chunk interference：先建立一个 256-output-token SSE stream，收到 20 个客户端 event 后注入 7168-token、
1-output-token 请求。重复五次，记录注入前 event gap 中位数、重叠期间最大 gap 和长 prefill E2E。SSE 可能
合并文本片段，因此该值定义为 client-visible event gap。

KV pressure：将总 KV capacity 限为 4096 tokens、context length 限为 2048；并发提交 8 个 1024-input、
512-output 的请求。vLLM 使用 `--num-gpu-blocks-override 256`（block size 16）；SGLang 使用
`--max-total-tokens 4096`。同时采集 queue time、preemption/retraction counter 和日志。

NGRAM：每次请求前 flush cache，greedy、强制生成 256 token，每个 prompt 三次。分别使用明显重复的
`alpha beta gamma delta` 序列和普通 compiler 说明文；读取响应 `meta_info` 中：

```text
spec_accept_rate
spec_accept_length
spec_num_correct_drafts
spec_num_proposed_drafts
spec_verify_ct
e2e_latency
```

`spec_accept_length` 包含每轮的 target/bonus token；`spec_accept_rate` 只统计正确 draft，指标定义不同。

## 4. 验证规则

1. eager/graph 对照除 graph 配置外保持一致，并在测量前完成 warmup。
2. prefix/NGRAM 每个独立样本前确认 cache flush 成功，不能在 running request 存在时假定 flush 已生效。
3. server sequentially 使用同一张 GPU；切换框架前确认 `nvidia-smi` 无残留 compute process。
4. 每条延迟结论同时报告 workload、concurrency、input/output length 和统计口径。
5. 单卡小模型只用于验证机制；TP/EP/P-D 结论必须另做多卡实验。
