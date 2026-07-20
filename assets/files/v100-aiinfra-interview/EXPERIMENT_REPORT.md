# 8x V100 大模型并行训练与推理框架实测报告

实验日期：2026-07-15 至 2026-07-17（UTC）  
实验机器：8x NVIDIA Tesla V100 PCIe 32 GB，单机、无 NVLink  
代码与结果目录：`/root/v100-llm-lab`

## 1. 报告摘要

### 1.1 最终结论

这台机器不是“因为 GPU 老而什么大模型框架都不能学”。更准确的结论是：

1. **训练并行完全能学，而且很适合学习通信代价。** PyTorch DDP、FSDP，以及固定到 `Megatron-Core 0.9.0` 的 TP、SP、PP 都已经在 8 张 V100 上真实跑通。由于没有 NVLink，很多并行策略的通信代价会被放大，反而更容易看清算法机制。
2. **vLLM 可以跑，但需要兼容版本。** 隔离环境中的 `vLLM 0.19.1 + PyTorch 2.10.0/cu128` 已成功识别 `sm_70`、启动 7B 模型结构、运行 continuous batching、TP/DP，并加载真实 0.5B 权重完成 OpenAI 兼容 API 请求。它不是当前最新版本，不能据此推断以后所有 vLLM 版本都继续支持 V100。
3. **V100 失去的是一部分新内核，不是 CUDA 推理能力。** vLLM 明确拒绝 FlashAttention 2，自动回退到 `TRITON_ATTN`；新的 `SymmMemCommunicator` 也不支持 SM70。框架仍能工作，但性能路径与 Ampere/Hopper 不同。
4. **在这台 PCIe 机器上，TP 首先是容量工具，不是加速工具。** Megatron 训练中，用 TP 替换 DP 后吞吐持续下降；vLLM 7B 推理中，TP=2/4 也比单卡慢，但 KV cache 容量分别扩大到约 3.19/7.57 倍。
5. **能单卡放下的推理模型优先做副本，而不是 TP。** vLLM 的 4 副本在高负载下比单副本快 46%，方向正确；但本地 API/连接负载均衡只有约 66% 的估算副本效率，仍需生产级外部负载均衡。
6. **FSDP 的关键不是“越彻底分片越好”。** 367.9M 参数实验中，`SHARD_GRAD_OP` 同时取得最高吞吐和 76% 显存下降；`FULL_SHARD` 显存最低，但因逐层 all-gather/reshard 比前者慢约 23%。
7. **PP 是否有效由 microbatch 数和 stage balance 共同决定。** PP=8、M=1 时理论气泡 87.5%，实测比单卡还慢；M=16 时达到 2.59x 加速，但只有 32.4% 并行效率，而且首尾 stage 的参数量相差 2.76 倍。
8. **DDP 的关键自变量是每次同步前的计算量。** microbatch 从 1 增到 8 后，每 token 梯度 payload 从 838.8 KiB 降到 104.8 KiB，8 卡弱扩展效率从 31.9% 提高到 51.2%；profiler 看到的 13 个 AllReduce kernel 总时长则基本保持在 104-106 ms。
9. **长序列不保证 SP 自动变得划算。** 在本 Megatron 实现和模型上，`seq=4096` 时 SP 只额外节省 1.3%-3.8% 显存；full activation recompute 才稳定节省 77.6%-78.1%，代价是约 27%-30% 吞吐。
10. **prefix cache 可以改变服务容量，但收益由复用率决定。** 1024-token 高复用前缀 workload 中，显式开启并预热缓存相对关闭缓存把吞吐提高 4.32 倍、mean TTFT 降低 84.8%；这是高复用上界，不是普通聊天流量的默认收益。
11. **推理容量应按 SLO goodput，而不是完成吞吐定。** Poisson 负载从 5 增到 25 QPS 时，完成吞吐从 4.65 增到 16.46 request/s，但同时满足 `TTFT<250 ms`、`TPOT<40 ms` 的 goodput 从 4.65 降到 0.40 request/s。

### 1.2 最值得保留的十个数字

| 观察 | 实测值 | 含义 |
|---|---:|---|
| 同 PCIe switch 两卡 AllReduce | 11.74 GB/s | 比 PHB 路径高 66%，rank 映射很重要 |
| DDP 8 卡强扩展 | 2.82x / 35.3% 效率 | 无 NVLink 时小模型很快被梯度通信限制 |
| FSDP `SHARD_GRAD_OP` | 10.01k token/s，1.55 GiB/GPU | 本机该模型下的最佳吞吐/显存平衡 |
| PP=8、M=16 | 42.52k token/s，2.59x | microbatch 能摊薄气泡，但不能消除通信和 stage 不均衡 |
| vLLM 单 V100、C=32 | 686.6 output token/s | continuous batching 相对 C=1 提高约 19.8 倍 |
| NCCL 8 卡 AllGather / ReduceScatter | 7.45 / 7.45 GB/s bus BW | FSDP/SP 的关键原语和 AllReduce 一样受 PCIe 限制 |
| DDP microbatch=8 | 227.4k token/s，51.2% 弱扩展效率 | 增加每次同步前的计算可摊薄固定梯度通信 |
| TP2、seq=4096 full recompute | 26.28 -> 5.77 GiB/GPU | 省 78.1% 显存，吞吐代价 29.7% |
| vLLM prefix cache warm | 13.82k total token/s，165 ms mean TTFT | 高复用前缀下缓存比关闭时快 4.32 倍 |
| vLLM 5 QPS Poisson | 4.65 request/s goodput | 本 SLO 下的稳定点；更高 offered load 反而降低合格吞吐 |

## 2. 实验边界与可信度

本报告主要回答系统问题：通信、并行度、显存、调度和吞吐如何互相影响。它不是模型质量或收敛报告。

- 训练数据是预先放在 GPU 上的合成 token，避免磁盘和 DataLoader 把通信结论污染。
- 训练模型执行真实 Transformer forward、cross entropy、backward 和 AdamW step，不是只测矩阵乘法。
- NCCL、DDP、FSDP、Megatron TP/SP/PP 的主结果均做 3 次独立进程启动，表格报告均值；原始 JSON 保留单次结果。
- 补充的 DDP microbatch 与 Megatron 长序列实验也各做 3 次独立进程启动；vLLM prefix 和 Poisson QPS 点做 2 次。NCCL 每个消息档位由工具内部做 10 次 warmup、30 次测量。
- DDP profiler 只记录 rank 0 的一个额外 step。NCCL kernel duration 可能与计算重叠，因此它用于证明通信工作量近似固定，不能与其他 kernel 时间直接相加当作 wall-time 分解。
- vLLM 的 7B 性能实验使用 `--load-format dummy`：模型结构、权重尺寸、KV cache、attention 和调度路径都是真实的，但权重值随机，因此不能评估生成质量。
- 另外加载了真实 `Qwen2.5-0.5B-Instruct` 权重，验证 tokenizer、safetensors、采样和 HTTP API 全链路。
- vLLM 第一轮 C=16 出现一次性低吞吐，第二轮恢复。它很可能是首次 Triton 形状/JIT 或调度预热开销，但日志不足以唯一证明原因；因此报告把第一轮作为冷形状数据保留，稳态比较使用第二轮。
- prefix cache workload 只有 4 个 1024-token 前缀供 48 个请求复用，属于刻意构造的高命中场景；它回答“缓存机制能带来什么”，不代表真实业务命中率。
- 功耗、能效、真实数据加载、多机网络、训练收敛、checkpoint 恢复和故障容错不在本轮范围。

## 3. 机器与软件环境

### 3.1 硬件

| 项目 | 配置 |
|---|---|
| GPU | 8x Tesla V100-PCIE-32GB，SM70，32 GiB/GPU |
| GPU 链路 | PCIe Gen3 x16，无 NVLink |
| 拓扑 | GPU 0-3 属于 NUMA 0，GPU 4-7 属于 NUMA 1；每两个相邻 GPU 为 PIX |
| CPU | 2x Intel Xeon E5-2680 v4，名义 56 logical CPU，本环境 33 online |
| 主机内存 | 64 GiB，无 swap |
| 磁盘 | 实验前约 898 GiB 可用 |
| OS | Ubuntu 22.04.5 LTS，Linux 5.15 |

关键拓扑如下：同组的 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)` 为 PIX；跨 NUMA（例如 0 到 4）为 SYS。后续 NCCL 结果证明，这个差异会直接影响 TP/DP 进程组的代价。

### 3.2 系统训练环境

| 组件 | 版本 |
|---|---|
| NVIDIA driver | 580.159.03 |
| `nvidia-smi` CUDA compatibility | 13.0 |
| CUDA toolkit / nvcc | 12.6.77 |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0+nv24.10，CUDA 12.6 |
| NCCL（PyTorch） | 2.22.3 |
| Apex | 0.1 |
| Transformer Engine | 1.11.0 |
| flash-attn | 2.4.2 |
| Megatron-Core | tag `core_v0.9.0`, commit `1afee592e85ac7994887eb5f4ef3998f76384333` |

### 3.3 隔离推理环境

为了不破坏系统已有的 NVIDIA PyTorch/Apex/TE 组合，vLLM 安装在 `.venv-vllm019`：

| 组件 | 版本/结果 |
|---|---|
| vLLM | 0.19.1 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| wheel 架构列表 | 包含 `sm_70` |
| `pip check` | 无 broken requirements |
| 隔离环境磁盘占用 | 约 11 GiB |
| V100 FP16 matmul smoke test | 有限值，通过 |

系统原本缺少 `ensurepip`，实验中安装了 Ubuntu 的 `python3.10-venv` 组件后创建隔离环境。这是环境准备问题，不是 GPU 兼容性问题。

## 4. 实验一：NCCL 拓扑与集合通信

![NCCL topology](figures/01-nccl-topology.png)

### 4.1 要回答的问题

1. PIX、PHB、SYS 路径的真实带宽差多少？
2. 4 卡和 8 卡 AllReduce 的有效总线带宽是多少？
3. Ring、Tree 和 NCCL 自动选择在该拓扑上有什么区别？

### 4.2 设置

- 工具：`nccl-tests/all_reduce_perf`
- 消息范围：1 KiB 到 256 MiB，每档 4 倍增长
- 每档：10 次 warmup，30 次测量
- 每进程 1 GPU
- 组合：PIX `(0,1)`、PHB `(0,2)`、SYS `(0,4)`、NUMA 内 4 卡、跨 NUMA 4 卡、全部 8 卡
- 8 卡额外强制 `NCCL_ALGO=Ring` 与 `Tree`

运行入口：`bash scripts/run_nccl_bench.sh`

### 4.3 指标

- 1 KiB latency：小梯度/控制消息的固定通信开销。
- 256 MiB algorithm bandwidth：从调用者看到的数据吞吐。
- 256 MiB bus bandwidth：按 collective 数据移动量归一化，更适合横向比较不同 world size。

### 4.4 结果

| 组合 | 1 KiB latency (us) | 256 MiB alg BW (GB/s) | 256 MiB bus BW (GB/s) |
|---|---:|---:|---:|
| 2 卡 PIX `(0,1)` | 15.06 | 11.74 | 11.74 |
| 2 卡 PHB `(0,2)` | 14.97 | 7.07 | 7.07 |
| 2 卡 SYS `(0,4)` | 14.79 | 6.93 | 6.93 |
| 4 卡 NUMA 0 | 21.92 | 4.92 | 7.38 |
| 4 卡跨 NUMA | 31.18 | 4.93 | 7.39 |
| 8 卡 auto | 48.65 | 4.34 | 7.60 |
| 8 卡 Ring | 45.58 | 4.33 | 7.59 |
| 8 卡 Tree | 58.29 | 4.32 | 7.56 |

### 4.5 解释

- PIX 大消息带宽比 PHB 高 **66.1%**，比 SYS 高 **69.4%**。因此 TP=2 时优先放在 `(0,1)` 这类同 switch pair，而不是随意选择两张卡。
- PHB 与 SYS 的 256 MiB 带宽接近，但跨 NUMA 的 4 卡小消息延迟从 21.92 us 增至 31.18 us，说明 NUMA 对频繁小 collective 更敏感。
- 8 卡 Ring/Tree 在 256 MiB 上几乎相同；Tree 的 1 KiB 延迟反而更高。不能只靠“Tree 小消息更好”的一般经验，必须在具体拓扑上测。
- 8 卡 bus BW 约 7.6 GB/s，是后续梯度 AllReduce、TP activation collective 的共同物理背景。

### 4.6 能学到什么

- 如何从 `nvidia-smi topo -m` 推导 process-group placement。
- algorithm bandwidth 与 bus bandwidth 的区别。
- 为什么没有 NVLink 时，TP、SP 和 ZeRO/FSDP 的通信更容易成为主导项。
- 为什么 rank mapping、NUMA 绑定和通信 bucket 是模型并行实验的前置条件，而不是事后调参。

### 4.7 补充：AllGather、ReduceScatter 与 SendRecv

![NCCL primitives](figures/09-nccl-primitives.png)

原报告只测了 AllReduce，但这不足以覆盖后续框架的真实通信：FSDP 参数分片依赖 AllGather，梯度分片依赖 ReduceScatter；Megatron SP 在层边界使用 AllGather/ReduceScatter；PP stage 之间则是 Send/Recv。补充实验保持与 AllReduce 相同的 1 KiB-256 MiB 范围、10 warmup、30 measured iterations，并复用 PIX/PHB/SYS/8 卡放置。

| 原语 | PIX 256 MiB bus BW | PHB | SYS | 8 卡 | 8 卡 1 KiB latency |
|---|---:|---:|---:|---:|---:|
| AllGather | 11.66 GB/s | 6.23 | 6.08 | 7.45 | 58.86 us |
| ReduceScatter | 9.47 GB/s | 6.14 | 6.07 | 7.45 | 59.40 us |
| SendRecv | 12.16 GB/s | 8.43 | 8.10 | 6.91 | 62.91 us |

所有 correctness check 都是 0 wrong。这里的 8 卡 SendRecv 是 nccl-tests 同时运行的多 rank 模式，反映共享 PCIe 路径下的 aggregate pattern，不等同于只测一对相邻 PP stage。

补充结论：

- AllGather/ReduceScatter 的 8 卡 bus BW 都是 7.45 GB/s，与 AllReduce 的 7.60 GB/s 接近，验证了 FSDP/SP 同样受该节点 PCIe 总线约束。
- 同一 PIX pair 上，AllGather 为 11.66 GB/s，ReduceScatter 只有 9.47 GB/s。即使拓扑相同，不同 collective 的实现与数据流也不能假定完全对称。
- PIX 相对 SYS 的大消息优势在三类原语上都存在；1 KiB 的单点延迟顺序有测量抖动，不能用 11-18 us 的微小差异推翻大消息拓扑结论。
- PP 的 P2P 不会自动绕开物理瓶颈。8 卡同时 SendRecv 只有 6.91 GB/s，而且 1 KiB 延迟约 63 us；microbatch 太小会同时暴露 pipeline bubble 与消息固定开销。

这组实验补上了“collective 语义 -> 拓扑路径 -> 框架行为”的映射：以后分析 FSDP/SP/PP 时，应测它实际调用的原语，而不是统一拿 AllReduce 带宽代替。

## 5. 实验二：DDP 强扩展、弱扩展与 bucket

![DDP scaling](figures/02-ddp-scaling.png)

### 5.1 模型与训练设置

| 项目 | 值 |
|---|---:|
| 模型 | 自定义 decoder-only GPT |
| 参数量 | 109,942,272 |
| 层数 / hidden / heads | 12 / 768 / 12 |
| vocab / sequence | 32,000 / 512 |
| microbatch / GPU | 2 |
| 参数/优化器 | FP32 参数，fused AdamW，lr=1e-4 |
| 计算精度 | FP16 autocast + GradScaler |
| DDP | NCCL，`gradient_as_bucket_view=True` |
| warmup / 测量 | 3 / 10 optimizer steps |
| 重复 | 3 次进程启动 |

强扩展固定 global batch=16：GPU 数 1/2/4/8 对应梯度累积 8/4/2/1。  
弱扩展固定每卡 local batch=2：global batch 随 GPU 数增长。  
bucket 对照固定 8 卡，比较 4/25/100 MiB。

### 5.2 指标

- aggregate token/s 与最慢 rank 的 step time。
- speedup 与强扩展效率：`speedup / GPU 数`。
- 弱扩展效率：实测 aggregate throughput / 理想线性 aggregate throughput。
- 每 rank peak allocated GPU memory。

### 5.3 强扩展结果

| GPU | Grad accumulation | token/s | Step (s) | Speedup | 效率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 23,815 | 0.3447 | 1.00x | 100.0% |
| 2 | 4 | 52,145 | 0.1572 | 2.19x | 109.5% |
| 4 | 2 | 57,838 | 0.1416 | 2.43x | 60.7% |
| 8 | 1 | 67,180 | 0.1219 | 2.82x | 35.3% |

2 卡“超线性”不是免费算力：从 1 卡到 2 卡时梯度累积循环从 8 次降到 4 次，Python/launch、缓存和同步结构都发生变化，所以该点不能解释为纯硬件 109.5% 效率。4/8 卡之后，梯度通信的代价开始明显压过计算分摊收益。

### 5.4 弱扩展结果

| GPU | Global batch | Aggregate token/s | 理想 token/s | 弱扩展效率 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 31,809 | 31,809 | 100.0% |
| 2 | 4 | 32,875 | 63,619 | 51.7% |
| 4 | 8 | 37,370 | 127,238 | 29.4% |
| 8 | 16 | 67,377 | 254,475 | 26.5% |

弱扩展下每卡计算量不变，理想情况 step time 应接近不变；实际 8 卡 step 从 32.2 ms 增到 121.6 ms。这个实验比强扩展更直接地显示了 AllReduce 固定成本和 PCIe 带宽瓶颈。

### 5.5 Bucket 结果

| Bucket | 8 卡 token/s | 相对 25 MiB |
|---:|---:|---:|
| 4 MiB | 65,950 | -1.83% |
| 25 MiB | 67,180 | 基准 |
| 100 MiB | 65,752 | -2.13% |

这组原始 sweep 使用 microbatch=2。4 MiB 会产生更多 collective，100 MiB 又减少了梯度计算与通信重叠机会；25 MiB 在这个配置上最好，但差距仅约 2%。工程深挖中的 microbatch=1 对照反而由 4 MiB 最快，进一步证明最优 bucket 会随 backward 计算窗口和 overlap 改变，不能推广为固定答案。

### 5.6 能学到什么

- 强扩展与弱扩展回答的是不同问题；只报“8 卡比 1 卡快几倍”不够。
- 梯度累积既改变 global batch，也改变通信频率和 Python/kernel 开销。
- DDP bucket 是 overlap 调参，不是越大越好。
- 对约 110M 的模型，8 张无 NVLink V100 已明显通信受限；更大的每卡 microbatch 或更大的模型通常能提高算术强度。

### 5.7 补充：microbatch 转折与 step profiler

![DDP microbatch and profiler](figures/10-ddp-microbatch-profile.png)

旧实验只固定 `microbatch=2`。补充实验保持模型、seq=512、一次 optimizer step 一次梯度同步不变，在 1 卡和 8 卡分别扫描每卡 microbatch `1/2/4/8`；每点 5 warmup、20 measured steps、3 次独立启动。模型参数为 FP32，因此每 rank 每次同步前形成约 `109,942,272 x 4 B = 419.4 MiB` 梯度 payload。

| Microbatch/GPU | 梯度 KiB/token | 1 GPU token/s | 8 GPU token/s | 1->8 speedup | 弱扩展效率 | 8 GPU peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 838.8 | 13,553 | 34,589 | 2.55x | 31.9% | 2.23 GiB |
| 2 | 419.4 | 27,564 | 67,306 | 2.44x | 30.5% | 2.58 GiB |
| 4 | 209.7 | 44,285 | 126,696 | 2.86x | 35.8% | 3.25 GiB |
| 8 | 104.8 | 55,566 | 227,430 | 4.09x | 51.2% | 4.63 GiB |

`m=1` 到 `m=2` 的效率并非严格单调，说明 kernel shape、launch 和测量噪声仍会影响小 batch；但 `m=8` 的提升远大于三次重复方差，主趋势明确。增加 microbatch 没有减少每 step 的梯度张量，只是用更多 forward/backward 计算摊薄同一份同步成本，同时以 activation 显存换效率。

为了直接检查通信归因，额外对 `m=1/8` 各记录 1 卡和 8 卡的 rank 0 单步 PyTorch CUDA trace：

| GPU | Microbatch | 相邻非 profiler step | NCCL kernel 累计时长 | AllReduce kernel 数 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 33.0 ms | 0 ms | 0 |
| 1 | 8 | 73.7 ms | 0 ms | 0 |
| 8 | 1 | 119.1 ms | 103.9 ms | 13 |
| 8 | 8 | 144.2 ms | 105.7 ms | 13 |

8 卡两种 microbatch 都调用 13 个 `ncclDevKernel_AllReduce_Sum_f32_RING_LL`，累计 GPU duration 只差约 1.8 ms，而非 profiler step 增加了约 25 ms 计算。NCCL kernel 可与 backward 计算重叠，所以 103.9 ms 不能从 119.1 ms 中直接相减；它表达的是通信 stream 的忙碌时间，不是未重叠阻塞时间。Chrome trace 已保留在 `results/raw/ddp_profile/`，可在 Perfetto 中继续观察 bucket ready 顺序和 overlap。

这组实验补出的学习要点是：

- DDP 是否值得扩卡取决于 `compute per synchronization`，而不只是参数量或 GPU 数。
- gradient accumulation 用 `no_sync()` 时也能增加每次通信前的计算，但它与增大 microbatch 的 activation 显存和 kernel shape 影响不同，应分别扫描。
- profiler 验证了“固定梯度 collective 被更多 token 摊薄”，比只从吞吐曲线反推瓶颈更可靠。
- `m=8` 仍只有 51.2% 弱扩展效率；调大 batch 能缓解 PCIe 瓶颈，不能消除它。

## 6. 实验三：FSDP / ZeRO 风格分片

![FSDP tradeoff](figures/03-fsdp-tradeoff.png)

### 6.1 设置

| 项目 | 值 |
|---|---:|
| 参数量 | 367,888,384 |
| 层数 / hidden / heads | 24 / 1024 / 16 |
| vocab / sequence | 32,000 / 256 |
| world size | 8 |
| microbatch | 1/GPU，global batch=8 |
| optimizer | fused AdamW，lr=1e-4 |
| FSDP wrap | 每个 `GPTBlock` 自动 wrap |
| warmup / 测量 / 重复 | 2 / 5 / 3 |

比较四种模式：

- `DDP`：FP32 参数和默认 FP32 梯度通信。
- `DDP + FP16 comm hook`：仅把梯度通信压缩到 FP16，作为通信字节数公平对照。
- `SHARD_GRAD_OP`：近似 ZeRO-2，分片梯度和 optimizer state，forward 期间保留完整参数。
- `FULL_SHARD`：近似 ZeRO-3，参数、梯度和 optimizer state 都分片，并逐层 all-gather/reshard。

FSDP 使用 `param_dtype/reduce_dtype/buffer_dtype=float16`、`limit_all_gathers=True`、`use_orig_params=True`。

### 6.2 结果

| 模式 | token/s | Step (s) | GPU allocated (GiB) | Host RSS/rank (GiB) |
|---|---:|---:|---:|---:|
| DDP FP32 comm | 5,274 | 0.3883 | 6.49 | 1.94 |
| DDP FP16 comm | 8,280 | 0.2474 | 6.49 | 2.01 |
| SHARD_GRAD_OP | **10,009** | **0.2046** | 1.55 | 2.11 |
| FULL_SHARD | 7,704 | 0.2659 | **1.06** | 2.11 |

### 6.3 解释

- 只改变 DDP 梯度通信精度，就把吞吐提高 **57.0%**，显存基本不变。这直接证明本机主要矛盾之一是 PCIe 上移动的字节数。
- `SHARD_GRAD_OP` 相对默认 DDP：显存下降 **76.1%**，吞吐提高 **89.8%**；相对 FP16-comm DDP 仍快约 20.9%。
- `FULL_SHARD` 相对默认 DDP：显存下降 **83.7%**；但相对 `SHARD_GRAD_OP` 慢约 23.0%，原因是每层参数 all-gather 和 reshard。
- `FULL_SHARD` 只比 FP16-comm DDP 慢约 7.0%，却把每卡 allocated memory 从 6.49 GiB 降至 1.06 GiB；模型放不下时，这个交换通常值得。
- FSDP 的主机 RSS 略高，因为元数据、flat parameter 和分片管理也有 CPU 成本。

### 6.4 能学到什么

- ZeRO/FSDP 的 stage 本质是“显存换通信”，不是单调性能等级。
- 公平比较必须控制 reduce dtype；否则会把 FP16 通信收益错误归因给分片算法。
- `SHARD_GRAD_OP` 适合模型参数能放下、optimizer/grad state 压力较大的情况。
- `FULL_SHARD` 适合容量优先；在纯 PCIe 节点上尤其要关注 all-gather 次数和 wrap 粒度。

## 7. 实验四：Megatron-Core TP 与 SP

![Megatron TP and SP](figures/04-megatron-tp-sp.png)

### 7.1 设置

- 真实 `Megatron-Core 0.9.0` `GPTModel`，使用 `get_gpt_layer_local_spec()`。
- 12 层、hidden 768、16 heads、vocab 32k、seq 512，约 110M 参数。
- FP16 参数、Apex `FusedAdam(capturable=True, master_weights=True)`、FP32 master weights、动态 loss scale（初值 1024）。
- 固定使用 8 张 GPU、固定 global batch=16、microbatch=2。
- `(TP, DP, grad accumulation)` 为 `(1,8,1)`、`(2,4,2)`、`(4,2,4)`、`(8,1,8)`。
- TP=2/4 额外比较 sequence parallel on/off。
- 3 warmup、10 measured steps、3 repeats。

注意：这是在固定 8 卡预算下，把 DP degree 逐步换成 TP degree 的实验，不是“固定 DP 再增加 GPU”的扩展实验。SP 参数梯度由脚本对带 `sequence_parallel` 标记的参数显式在 TP group 内归约。

### 7.2 指标与结果

| TP | DP | SP | token/s | 相对 TP1 | 每卡参数上限 | Peak allocated/GPU |
|---:|---:|:---:|---:|---:|---:|---:|
| 1 | 8 | off | 84,626 | 100.0% | 110.0M | 2.85 GiB |
| 2 | 4 | off | 51,322 | 60.6% | 55.2M | 1.61 GiB |
| 2 | 4 | on | 43,155 | 51.0% | 55.2M | 1.56 GiB |
| 4 | 2 | off | 29,053 | 34.3% | 27.8M | 0.88 GiB |
| 4 | 2 | on | 23,455 | 27.7% | 27.8M | 0.80 GiB |
| 8 | 1 | off | 14,667 | 17.3% | 14.1M | 0.47 GiB |

### 7.3 解释

- TP 从 1 增到 8 时，每卡参数和显存近似按比例下降，但每个 Transformer layer 都增加 TP collective；在 7.6 GB/s 左右的 8 卡 NCCL 总线带宽下，吞吐持续下降。
- TP=8 的每卡显存比 TP=1 少 **83.4%**，但吞吐也只剩 **17.3%**。所以 TP 的第一职责是让更大模型放得下。
- SP 在 TP=2 时再省 3.0% allocated memory、损失 15.9% 吞吐；TP=4 时省 9.3%、损失 19.3%。本实验 seq=512、模型较小，activation 不是主要显存项，SP 的额外通信不划算。
- 18 个正式运行的 final loss 均为有限值。普通 AdamW 直接更新 FP16 参数曾导致非有限 loss，最终结果改用 FP32 master weights 后全部重跑；这也是混合精度训练中 master-weight optimizer 的实际意义。
- 当序列更长、microbatch 更大或 activation 占主导时，SP 的显存收益会变大；不能从当前结果得出“SP 永远没用”。

### 7.4 能学到什么

- TP 的 row/column parallel linear 如何把参数和 GEMM 分片，又为何每层需要通信。
- 为什么高速 GPU 互联是 TP 的核心基础设施。
- SP 分的是 activation 序列维度，解决的问题与 TP 参数分片不同。
- 并行度规划应遵循：先使用高效 DP，容量不足时再引入最小必要 TP/SP。

### 7.5 补充：长序列 SP 与 full activation recompute

![Megatron long sequence and recompute](figures/11-megatron-longseq-recompute.png)

为验证“序列更长时 SP 会更值得”这一原假设，补充实验固定 8 卡、global batch=16、microbatch=2，扫描 TP=2/4、seq=512/2048/4096、SP on/off。`gradient_accumulation=TP`，因此不同 TP 的 global batch 保持一致。`seq=4096` 额外启用 Megatron `recompute_granularity=full`、`recompute_method=uniform`、`recompute_num_layers=1`，即 backward 时逐层重算；每点仍为 3 warmup、10 measured steps、3 次独立启动。

#### SP 结果

| TP | Seq | SP off token/s | SP off memory | SP memory saving | SP throughput change |
|---:|---:|---:|---:|---:|---:|
| 2 | 512 | 51,465 | 1.61 GiB | 3.1% | -17.3% |
| 2 | 2048 | 82,531 | 7.70 GiB | 2.3% | -6.6% |
| 2 | 4096 | 54,134 | 26.28 GiB | 1.3% | -3.7% |
| 4 | 512 | 28,923 | 0.88 GiB | 9.1% | -16.2% |
| 4 | 2048 | 53,437 | 4.06 GiB | 6.5% | -9.8% |
| 4 | 4096 | 40,209 | 13.55 GiB | 3.8% | -6.8% |

结果否定了“只要 seq 变长，SP 的相对显存收益就会单调增加”。在该 Megatron-Core 路径里，SP 主要分片 layernorm/dropout 等可 sequence-parallel 的 activation，并在 column/row parallel 边界加入 AllGather/ReduceScatter；self-attention 仍要处理完整序列上的 attention 结构。seq 增长后，未被 SP 等比例切分的 attention 和中间张量增长更快，反而把 SP 的相对节省从 TP2 的 3.1% 稀释到 1.3%。

吞吐从 seq512 到 2048 先上升，是更大 kernel 提高了 GPU 利用率；到 4096 后 attention 的二次复杂度超过这个收益，token/s 再下降。samples/s 会把这种序列长度差异完全隐藏，因此长上下文训练至少要同时报告 token/s、step time 和峰值显存。

#### Full recompute 结果（seq=4096）

| TP | SP | Baseline memory | Recompute memory | Memory saved | Baseline token/s | Recompute token/s | Throughput cost |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 2 | off | 26.28 GiB | 5.77 GiB | 78.1% | 54,134 | 38,040 | 29.7% |
| 2 | on | 25.95 GiB | 5.68 GiB | 78.1% | 52,125 | 36,876 | 29.3% |
| 4 | off | 13.55 GiB | 3.04 GiB | 77.6% | 40,209 | 28,646 | 28.8% |
| 4 | on | 13.03 GiB | 2.89 GiB | 77.8% | 37,470 | 27,287 | 27.2% |

Full recompute 的收益在四种组合中非常稳定：约 78% activation memory 换约 28%-30% 额外计算。recompute 后再开 SP 只从 5.77 降到 5.68 GiB（TP2），或从 3.04 降到 2.89 GiB（TP4）；在本模型上两者不是两个同量级、可简单相乘的容量杠杆。

能从补充实验学到：

- 应先把显存拆成参数/optimizer、线性 activation、attention 相关 activation 和临时 workspace，再选择 TP、SP、FSDP 或 recompute；“activation OOM”不是一种单一对象。
- SP 是通信换特定 activation 分片，recompute 是计算换保存 activation；两者优化的张量集合不同。
- 无 NVLink 时，SP 的 AllGather/ReduceScatter 成本可以从 512 序列的约 16%-17% 降到 4096 序列的约 4%-7%，因为计算增长更快，但它是否值得仍由绝对显存收益决定。
- 如果目标只是把本例 `seq=4096` 放下，TP2+recompute 比进一步增大 TP 或叠加 SP 更直接；如果模型参数本身也放不下，再组合 TP/FSDP/PP。

## 8. 实验五：Megatron-Core 1F1B 流水线并行

![Megatron PP](figures/05-megatron-pp.png)

### 8.1 设置

- 使用 Megatron-Core 自带 `get_forward_backward_func()` 和非交错 1F1B schedule，不是自写流水线模拟器。
- 16 层、hidden 768、16 heads、vocab 32k、seq 512，总参数约 162.95M。
- FP16 参数、fused AdamW、静态 loss scale 1024。
- TP=1、DP=1；PP=1/2/4/8 时分别使用 1/2/4/8 张 GPU。
- microbatch size=2；num microbatches `M=1/4/16`，对应 global batch 2/8/32。
- 每个点 1 warmup、3 measured optimizer steps、3 repeats。
- `share_embeddings_and_output_weights=False`，便于独立测量首尾 stage 参数不均衡。

同一个 M 内比较不同 PP 是严格相同 global batch；跨 M 的吞吐变化同时包含更充分流水线填充和更大的 global batch。

### 8.2 理论指标

非交错、stage 时间完全均衡时，简化气泡比例为：

```text
bubble = (PP - 1) / (M + PP - 1)
```

实际性能还受 stage 计算不均衡、P2P 通信、optimizer step 和端点 embedding/output layer 影响。

### 8.3 结果

| PP | M | 理论气泡 | token/s | 相对 PP1 speedup | 并行效率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.0% | 11,924 | 1.00x | 100.0% |
| 2 | 1 | 50.0% | 11,304 | 0.95x | 47.4% |
| 4 | 1 | 75.0% | 10,456 | 0.88x | 21.9% |
| 8 | 1 | 87.5% | 9,627 | 0.81x | 10.1% |
| 1 | 4 | 0.0% | 14,645 | 1.00x | 100.0% |
| 2 | 4 | 20.0% | 18,211 | 1.24x | 62.2% |
| 4 | 4 | 42.9% | 20,916 | 1.43x | 35.7% |
| 8 | 4 | 63.6% | 25,137 | 1.72x | 21.5% |
| 1 | 16 | 0.0% | 16,387 | 1.00x | 100.0% |
| 2 | 16 | 5.9% | 19,912 | 1.22x | 60.8% |
| 4 | 16 | 15.8% | 27,128 | 1.66x | 41.4% |
| 8 | 16 | 30.4% | 42,522 | 2.59x | 32.4% |

### 8.4 Stage 不均衡

| PP | 最小 stage 参数 | 最大 stage 参数 | Max/Min |
|---:|---:|---:|---:|
| 2 | 81.3M | 81.7M | 1.00x |
| 4 | 28.4M | 53.3M | 1.88x |
| 8 | 14.2M | 39.1M | 2.76x |

层数虽然等分，embedding 和输出投影仍集中在端点。PP=8 时端点 stage 的参数量是中间 stage 的 2.76 倍，慢 stage 会把整条流水线的节拍拉长。

### 8.5 能学到什么

- warmup、steady-state 1F1B 和 cooldown 三个阶段怎样产生气泡。
- `M >= PP` 只是起点，不等于已经高效；本实验 PP=8、M=16 仍有 30.4% 理论气泡。
- 切层不能只按 layer count，需要按 profile 后的 stage time 或参数/算子成本平衡。
- PP 适合深模型的容量扩展；小模型、少 microbatch 时会比单卡更慢。
- interleaved 1F1B、virtual pipeline stage 可以继续降低气泡，但会提高调度与通信复杂度。

## 9. 实验六：vLLM 兼容性、真实模型与内核回退

### 9.1 安装与架构验证

安装命令位于隔离环境，未替换系统 PyTorch。验证结果：

```json
{
  "vllm": "0.19.1",
  "torch": "2.10.0+cu128",
  "device": "Tesla V100-PCIE-32GB",
  "capability": [7, 0],
  "arch_list_contains": "sm_70",
  "matmul_finite": true
}
```

### 9.2 7B dummy 引擎

统一配置：

- 模型结构：`Qwen/Qwen2.5-7B-Instruct`
- `dtype=half`、`max_model_len=2048`、`gpu_memory_utilization=0.85`
- `load_format=dummy`，随机权重但完整 7B 结构
- vLLM V1 engine，默认启用 prefix caching 和 chunked prefill

| 模式 | 权重/GPU | 可用 KV/GPU | 框架报告 KV token | Engine init |
|---|---:|---:|---:|---:|
| TP1 eager | 14.25 GiB | 12.03 GiB | 225,200 | 12.04 s |
| TP1 compiled | 14.25 GiB | 11.55 GiB | 216,304 | 45.95 s |
| TP2 eager | 7.12 GiB | 19.19 GiB | 718,640 | 14.03 s |
| TP4 eager | 3.55 GiB | 22.78 GiB | 1,704,736 | 15.19 s |

内核日志给出的事实：

- FlashAttention 2 因 compute capability `< 8.0` 被拒绝。
- vLLM 自动选择 `TRITON_ATTN`，引擎正常工作。
- TP=2/4 时 `SymmMemCommunicator` 因 SM70 不可用。
- 超过两张纯 PCIe GPU 时，vLLM custom all-reduce 被显式关闭，回退普通 NCCL。

### 9.3 真实权重全链路

真实模型：`Qwen/Qwen2.5-0.5B-Instruct`，FP16、max length 2048、eager、GPU utilization 0.50。

| 观察 | 结果 |
|---|---:|
| Hugging Face 权重缓存 | 约 954 MiB |
| 下载耗时 | 39.87 s |
| 模型显存 | 0.93 GiB |
| KV cache | 1,244,880 token |
| API 状态 | HTTP 200 |
| 请求 token | prompt 43，completion 59 |
| 请求总耗时 | 3.03 s |

该输出只用于证明真实权重、tokenizer、采样、detokenize 和 OpenAI API 成功，不用于评价 0.5B 模型回答质量。

### 9.4 能学到什么

- “wheel 能 import”与“模型 kernel 能跑”是两级不同的兼容性验证。
- GPU 架构不支持最快 backend 时，成熟框架会选择 fallback；应从日志确认实际 backend，不能只看配置名。
- `load_format=dummy` 很适合做调度和容量实验，但必须补一个真实权重 smoke test。
- V100 上可学习 vLLM scheduler、paged KV cache、continuous batching、prefix caching、chunked prefill、TP/DP；不能把 FlashAttention 2/3、FP8 等新卡路径当作本机实验目标。

## 10. 实验七：vLLM continuous batching 与 compile

![vLLM concurrency](figures/06-vllm-concurrency.png)

### 10.1 在线负载设置

- 随机输入长度固定 128 token，输出固定 32 token，`ignore_eos=True`。
- 48 个请求，request rate 为无限（一次性施压）。
- 并发上限 C=1/4/16/32。
- 每点 4 个 warmup request；两轮正式测试。
- 指标：request/s、input+output token/s、output token/s、TTFT、TPOT、ITL、失败数。
- 稳态表使用第 2 轮；第 1 轮完整保留在原始结果。

### 10.2 Eager 稳态结果

| C | Request/s | Output token/s | Total token/s | Mean TTFT | P95 TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.09 | 34.8 | 173.8 | 67 ms | 72 ms | 27.5 ms |
| 4 | 4.12 | 131.8 | 659.0 | 94 ms | 109 ms | 28.3 ms |
| 16 | 15.20 | 486.4 | 2,431.9 | 153 ms | 172 ms | 28.8 ms |
| 32 | 21.46 | 686.6 | 3,433.1 | 204 ms | 276 ms | 30.2 ms |

从 C=1 到 C=32，output throughput 提高 **19.8 倍**，而 TPOT 只增加约 9.8%；主要延迟代价体现在排队和 prefill 带来的 TTFT 增长。这就是 continuous batching 的核心：牺牲一部分首 token 延迟，提高 GPU 批处理效率和 aggregate throughput。

### 10.3 Eager 与默认 compile/CUDA Graph

默认优化路径额外产生：

- `torch.compile`：19.83 s
- CUDA Graph capture：12 s
- CUDA Graph pool：0.44 GiB
- Engine init：45.95 s，对比 eager 12.04 s
- KV token：216,304，对比 eager 225,200，减少约 4.0%

| C | Eager output tok/s | Compiled output tok/s | Compiled 增益 |
|---:|---:|---:|---:|
| 1 | 34.8 | 41.3 | +18.8% |
| 4 | 131.8 | 148.9 | +13.0% |
| 16 | 486.4 | 511.9 | +5.3% |
| 32 | 686.6 | 699.9 | +1.9% |

低并发下 kernel launch/graph 优化占比高，因此收益更明显；高并发已经由大 batch 摊薄 launch 开销，收益趋小。常驻服务通常可以接受额外启动时间；短作业、频繁弹性伸缩或 KV 容量优先时，eager 更有吸引力。

### 10.4 长 prefill 干扰

负载改为 input=1024、output=16、C=8、32 requests：

| 模式 | Total token/s | Output token/s | Mean TTFT | P95 TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| Eager | 3,468.8 | 53.4 | 1,023 ms | 1,592 ms | 91.0 ms |
| Compiled | 3,672.3 | 56.5 | 1,167 ms | 1,849 ms | 72.6 ms |

总 token/s 很高不代表用户体验更好：长 prefill 把 TTFT 提高到约 1 秒，并干扰同批 decode。推理报告必须至少同时给出 input throughput、output throughput、TTFT 和 TPOT。

### 10.5 能学到什么

- continuous batching 如何把并发请求拼成动态 batch。
- TTFT、TPOT、ITL、throughput 分别对应排队/prefill、decode 速度和流式稳定性。
- compile/CUDA Graph 是启动时间、显存和稳态速度三者的交换。
- prefill-heavy 与 decode-heavy workload 应分开压测；一个“tokens/s”不能描述服务质量。

## 11. 实验八：vLLM TP 容量与 DP 副本

### 11.1 TP：容量提高，性能下降

![vLLM TP](figures/07-vllm-tp.png)

相同 7B dummy 模型、eager 模式、128 input/32 output，使用第二轮稳态数据：

| TP | C=1 output tok/s | C=32 output tok/s | 相对 TP1 (C=32) | KV token capacity |
|---:|---:|---:|---:|---:|
| 1 | 34.8 | 686.6 | 基准 | 225,200 |
| 2 | 24.0 | 575.0 | -16.3% | 718,640 |
| 4 | 26.2 | 562.1 | -18.1% | 1,704,736 |

单请求下 TP=2/4 也分别比 TP1 慢 30.9%/24.6%。模型每层的通信发生在每个 decode token 上，PCIe/NCCL 延迟无法被小 batch 的计算充分隐藏。

KV 容量却显著提高：权重分片后，每卡空闲显存更多；同时 KV head 也随 TP 分片。TP=4 的框架报告容量是 TP1 的约 7.57 倍。需要超长上下文、极高并发 KV 或模型本身超过 32 GiB 时，这正是 TP 的价值。

### 11.2 DP：需要足够负载和正确负载均衡

![vLLM DP](figures/08-vllm-dp.png)

为避免短测试的启动/排空占比过高，C=32 使用 128 requests，C=128 使用 256 requests。表格使用第 2 轮：

| 模式 | C=32 output tok/s | C=128 output tok/s | C=128 Mean TTFT | C=128 Mean TPOT |
|---|---:|---:|---:|---:|
| 单副本 | 853.5 | 1,543.3 | 500 ms | 67.0 ms |
| DP4，默认 4 API | 855.4 | **2,255.9** | 374 ms | 40.4 ms |
| DP4，强制 1 API | 812.6 | 2,159.8 | 543 ms | 39.6 ms |

解释：

- 总 C=32 时，4 副本平均只有约 8 并发/副本，且连接可能粘在少数 API process 上，所以 aggregate throughput 几乎没有提升。
- C=128 时，默认 DP4 比单副本提高 **46.2%**，同时 TTFT 和 TPOT 都下降，说明副本开始被有效利用。
- 若把单副本 C=32 的 853.5 token/s 看作每副本目标，4 副本理想值约 3,414 token/s；实测 2,256，估算效率约 **66.1%**。
- 强制一个 API process 没有改善，默认 4 API 略快。瓶颈不只是 API process 数，还包括连接复用、请求分配、DP coordinator 和负载不均衡。

生产部署应使用外部负载均衡器、足够多客户端连接，并分别观察每副本 queue depth/GPU utilization。对于能在单 V100 放下的模型，副本仍然是正确的吞吐扩展方向；本实验说明的是内置本地 LB 也需要压测，而不是“DP 无效”。

### 11.3 能学到什么

- 推理 TP 与训练 TP 一样，以容量扩展为第一目标。
- DP replica 不产生逐 token 跨 GPU collective，更适合吞吐横向扩展。
- 并行策略不能脱离 frontend、连接池和负载均衡讨论。
- 同样占 4 张 GPU，TP4、DP4 解决的是完全不同的问题：前者让一个请求/模型跨卡，后者让多个独立请求分散到副本。

## 12. 实验九：vLLM prefix cache 与 SLO goodput

### 12.1 Automatic prefix caching

![vLLM prefix cache](figures/12-vllm-prefix-cache.png)

服务端分别显式使用 `--enable-prefix-caching` 和 `--no-enable-prefix-caching`，避免依赖版本默认值。每个正式 repeat 都重新启动 7B dummy eager server；一次服务内先跑 cold，再用相同 seed 和 prompt 运行 warm：

- 48 requests，4 个前缀，理论 request reuse 为 `1 - 4/48 = 91.7%`。
- 每个 prompt 为 1024-token shared prefix + 32-token unique suffix，固定输出 16 token。
- `max_concurrency=8`、request rate=inf、`ignore_eos=True`。
- on/off 各 2 次独立服务启动，所有请求零失败。

| Prefix cache | Phase | Request/s | Total token/s | Mean TTFT | P95 TTFT | Mean TPOT |
|---|---|---:|---:|---:|---:|---:|
| off | cold | 2.98 | 3,194 | 1,175 ms | 2,337 ms | 99.5 ms |
| off | repeat | 2.99 | 3,202 | 1,088 ms | 1,873 ms | 104.8 ms |
| on | cold | 9.09 | 9,742 | 381 ms | 1,472 ms | 32.6 ms |
| on | warm | **12.90** | **13,824** | **165 ms** | **194 ms** | **29.9 ms** |

关闭缓存后 repeat 与 cold 几乎相同，说明普通 engine/kernel 预热不能解释 on 的增益。on warm 相对 off repeat：吞吐提高 **4.32x**，mean TTFT 降低 **84.8%**，P95 TTFT 降低 **89.7%**。on warm 相对 on cold 仍快 41.9%，因为 cold 内每个前缀首次出现仍需 prefill。

两次 on server 日志最终都报告 `93.7%` prefix cache hit rate。该值按缓存 block/token 统计，与按请求计算的 91.7% 理论复用率口径不同，不能直接相减。TPOT 也显著改善并不表示缓存了输出 token，而是大量 prefill 被跳过后，decode 受到的调度和算力干扰减少。

这是一个刻意构造的高复用上界。真实系统还要扫描 prefix 长度、前缀分布、缓存容量、eviction、租户隔离和版本化 prompt；低命中率流量不会得到 4.32x 收益。

### 12.2 Poisson request rate 与 goodput

![vLLM arrival SLO](figures/13-vllm-arrival-slo.png)

旧 benchmark 使用 request rate=inf，只能测饱和 throughput，不能回答“线上可以接多少 QPS”。补充实验固定随机输入 128、输出 32、64 prompts、`max_concurrency=64`，以 burstiness=1 的 Poisson 过程发送 `2/5/10/15/20/25 QPS`。每点 2 次，goodput 定义为单请求同时满足：

```text
TTFT < 250 ms  and  TPOT < 40 ms
```

| Offered QPS | Completed request/s | SLO goodput | P95 TTFT | P95 TPOT | 判断 |
|---:|---:|---:|---:|---:|---|
| 2 | 1.94 | 1.94 | 121 ms | 33.3 ms | 全部合格 |
| 5 | 4.65 | 4.65 | 157 ms | 36.4 ms | 全部合格 |
| 10 | 8.63 | 5.06 | 205 ms | 50.5 ms | TPOT 开始越界 |
| 15 | 11.93 | 4.38 | 220 ms | 54.6 ms | 完成吞吐升、goodput 降 |
| 20 | 14.41 | 1.13 | 271 ms | 68.6 ms | TTFT/TPOT 都越界 |
| 25 | 16.46 | 0.40 | 313 ms | 73.2 ms | 接近饱和，几乎无合格请求 |

`10 QPS` 是转折区：两轮 goodput 标准差为 1.62 request/s，说明靠近容量边界时短测的尾延迟很不稳定；`25 QPS` 的 goodput 也有 0.56 标准差。生产容量规划需要更长持续时间、更多重复和真实长度分布。本轮 64-request 结果足以定位机制和大致拐点，但不是生产 SLA 认证。

两种“容量”必须区分：

- 如果要求本轮所有请求满足该 SLO，保守运行点是约 5 offered QPS。
- 如果只追求最大绝对 SLO goodput，10 QPS 的均值略高（5.06），但约 41% 完成请求已经不合格且方差很大。
- 25 QPS 时服务器仍完成 16.46 request/s、没有报错，但 goodput 只有 0.40；“无失败”和“服务可用”不是同一件事。
- inf-rate/C=32 的 686.6 output token/s 是饱和吞吐指标，不能直接换算成线上安全 QPS。

### 12.3 能学到什么

- prefix cache 利用跨请求的 prefill 复用，continuous batching 利用同时在场请求的 batch 复用；两者是正交机制。
- cache hit rate 必须带统计口径，平均吞吐必须带 workload 的 prefix 分布，否则数字不可迁移。
- open-loop offered load、completed throughput、tail latency 和 SLO goodput 是四个不同指标。
- 线上过载常先表现为 TPOT/TTFT 退化，而不是 HTTP failure；容量控制、排队和 admission control 应在错误出现前介入。

## 13. 综合建议：这台机器适合怎样学习

### 13.1 推荐学习顺序

1. **NCCL 与拓扑**：先读 PIX/PHB/SYS，再分别测 AllReduce、AllGather、ReduceScatter 和 SendRecv，建立“框架操作对应哪种原语”的习惯。
2. **DDP**：画 strong/weak scaling，扫描 microbatch、accumulation 和 bucket，再打开 trace 验证通信次数、时长与 overlap。
3. **FSDP**：控制 reduce dtype 后比较 `SHARD_GRAD_OP` 与 `FULL_SHARD`，理解参数、梯度、optimizer state 三类内存和不同 collective。
4. **长序列 activation**：同时扫描 sequence length、SP 和 full recompute，区分线性 activation、attention 张量与保存/重算策略。
5. **Megatron TP/SP**：保持总 GPU=8，改变 TP/DP 组合，观察容量收益与逐层 collective 代价。
6. **Megatron PP**：固定 PP 扫 microbatch 数，计算理论 bubble，再用 stage profile 指导切层，而不是只按 layer count。
7. **vLLM scheduler**：依次学习 continuous batching、prefill/decode、compile、prefix cache、TP/replica、Poisson arrival 和 SLO goodput。
8. **端到端训练/服务**：最后加入真实 dataset、checkpoint、validation、MFU、真实请求分布和外部负载均衡，验证微基准结论能否迁移。

### 13.2 并行策略选择原则

| 目标 | 本机优先策略 | 原因 |
|---|---|---|
| 小/中模型训练吞吐 | DP + 足够大 microbatch | 减少逐层 TP 通信 |
| optimizer/grad 显存不足 | FSDP `SHARD_GRAD_OP` | 本轮吞吐/显存平衡最好 |
| 参数本身放不下 | `FULL_SHARD` 或最小必要 TP/PP | 以容量换通信 |
| 长序列 activation OOM | 先 full recompute，再按实测决定 SP | 本例 recompute 省约 78%，SP 只额外省 1%-4% |
| 深模型跨卡 | PP + 足够多 microbatch + stage balance | 避免 TP 每层高频 collective |
| 推理模型单卡能放下 | 多 replica | 避免逐 token TP 通信 |
| 推理模型/单请求 KV 放不下 | TP | 容量优先，接受吞吐损失 |
| 高复用长前缀 | prefix cache | 跳过重复 prefill；必须先测真实命中率 |
| 在线容量规划 | Poisson/open-loop + goodput | 饱和 token/s 不能代表 SLO 安全 QPS |

### 13.3 不建议在本机作为主目标的内容

- FlashAttention 2/3、FP8、Transformer Engine 新版 Blackwell/Hopper 路径。
- 把 SGLang 最新主线作为本机主性能目标。本轮没有安装或运行 SGLang，因此报告不对“能否启动某个具体版本”给出实验结论；若要研究其调度，应单独固定一个保留 SM70 路径的版本再验证。
- 最新 Megatron-Core 的所有特性。这里验证的是固定版本 0.9.0，不能把结果外推到当前 main branch。
- 用 8 卡 TP 跑一个本来单卡就能放下的小模型并期待线性加速。

## 14. 复现方法

### 14.1 训练与通信

```bash
cd /root/v100-llm-lab

bash scripts/collect_env.sh
bash scripts/run_nccl_bench.sh
python3 scripts/parse_nccl.py
bash scripts/run_nccl_primitives.sh
python3 scripts/parse_nccl_primitives.py

bash scripts/run_ddp_bench.sh
python3 scripts/parse_training_results.py
bash scripts/run_ddp_microbatch_bench.sh
bash scripts/run_ddp_profile.sh
python3 scripts/parse_ddp_supplement.py

bash scripts/run_fsdp_bench.sh
python3 scripts/parse_fsdp.py

bash scripts/run_megatron_tp_bench.sh
python3 scripts/parse_megatron.py
bash scripts/run_megatron_longseq_bench.sh
python3 scripts/parse_megatron_longseq.py

bash scripts/run_megatron_pp_bench.sh
python3 scripts/parse_megatron_pp.py

# 工程深挖：DDP/NCCL/MoE/checkpoint/failure/Triton/process group/FSDP runtime
bash scripts/run_ddp_engineering_bench.sh
bash scripts/run_nccl_engineering_bench.sh
bash scripts/run_moe_alltoall_bench.sh
bash scripts/run_resilience_bench.sh
bash scripts/run_triton_rmsnorm_bench.sh
bash scripts/run_parallel_group_runtime_bench.sh
bash scripts/run_fsdp_runtime_bench.sh
bash scripts/run_triton_resource_bench.sh
python3 scripts/parse_engineering_deep_dive.py
```

Megatron checkout 已固定在 `vendor/Megatron-LM`。所有 runner 都会把 stdout/stderr 写到 `results/raw/`。长序列 sweep 会连续占用全部 8 张 GPU，运行前应确认没有其他 CUDA 进程。

### 14.2 vLLM

启动一个测试服务：

```bash
cd /root/v100-llm-lab
bash scripts/start_vllm_server.sh tp1_eager
```

可选 mode：`tp1_eager`、`tp1_eager_prefix_on`、`tp1_eager_prefix_off`、`tp1_eager_prefix_observed`、`tp1_eager_prefix_small_cache`、`tp1_eager_scheduler_pressure`、`tp1_compiled`、`tp2_eager`、`tp4_eager`、`dp4_eager`、`dp4_asc1_eager`、`real_0.5b`。

另一个终端运行对应客户端：

```bash
VLLM_ENGINE_LABEL=tp1_eager bash scripts/run_vllm_client_bench.sh
VLLM_ENGINE_LABEL=tp2_eager bash scripts/run_vllm_tp_compare_client.sh
VLLM_ENGINE_LABEL=dp4_eager bash scripts/run_vllm_dp_bench.sh

python3 scripts/parse_vllm.py
```

补充的 prefix 与 arrival runner 会自行启动、健康检查并清理 server：

```bash
bash scripts/run_vllm_prefix_bench.sh
bash scripts/run_vllm_arrival_bench.sh
bash scripts/run_vllm_cache_eviction_bench.sh
bash scripts/run_vllm_preemption_bench.sh
python3 scripts/parse_vllm_supplement.py
python3 scripts/parse_engineering_deep_dive.py

python3 scripts/generate_report_assets.py
python3 scripts/validate_results.py
```

### 14.3 结果目录

- `results/raw/environment.txt`：完整机器快照。
- `results/raw/nccl/`、`nccl_primitives/`：AllReduce 与补充 collective 原始日志。
- `results/raw/ddp/`、`ddp_microbatch/`、`ddp_profile/`：DDP 原始结果、事件汇总和 Chrome trace。
- `results/raw/ddp_engineering/`、`nccl_engineering/`：reducer/accumulation trace，以及算法、协议、channel 和并发拓扑原始日志。
- `results/raw/fsdp/`、`megatron_tp/`、`megatron_pp/`、`megatron_longseq/`：分片/模型并行单次 JSON 和日志。
- `results/raw/moe_alltoall/`、`resilience/`、`triton_rmsnorm/`：MoE 路由、checkpoint/failure injection 和自定义算子结果。
- `results/raw/parallel_group_runtime/`、`fsdp_runtime/`、`triton_resources/`：3D process-group、FSDP 状态机 trace 和 kernel 编译资源结果。
- `results/raw/vllm/`、`vllm_supplement/`：安装、server、benchmark、prefix/SLO 和真实请求日志。
- `results/raw/vllm_cache_eviction/`：默认/1 GiB KV cache 的分阶段污染与驱逐证据。
- `results/raw/vllm_preemption/`：1/4 GiB KV 压力下的请求结果、250 ms metrics time series 和 server 日志。
- `results/summary/`：标准化 JSON/CSV。
- `results/summary/engineering_deep_dive_summary.json`：十类工程深挖实验的统一汇总。
- `results/summary/validation.json`：结果数量、有限 loss、vLLM 失败数与图表完整性检查。
- `report/figures/`：本报告使用的图。

## 15. 证据审计与结论修正

### 15.1 本轮真正实测了什么

- 硬件、拓扑、driver/CUDA/NCCL 和框架版本来自本机命令与日志，不是按 GPU 型号推测。
- PyTorch DDP/FSDP、Megatron-Core 0.9.0 TP/SP/PP/recompute 均执行真实 forward、loss、backward、optimizer step；所有正式训练 JSON 的 `loss_is_finite=true`。
- NCCL 四类通信原语均由 `nccl-tests` 实测并通过 correctness check。
- vLLM 结论限定在 0.19.1：7B dummy 覆盖真实结构/计算/调度/显存路径，0.5B real weights 覆盖 tokenizer、加载、采样和 HTTP 全链路。
- prefix cache 和 Poisson goodput 各做两次；20 个补充 vLLM benchmark 全部 `failed=0`。
- DDP 工程补充包含 21 个稳态 run 和 7 个 rank 0 trace，读取实际 reducer bucket，并对 CUDA 区间做 union/intersection 计算 exposed NCCL。
- NCCL 算法/协议/channel 与四对 GPU 并发实验均通过 `wrong=0`；MoE 六次 variable-split All-to-All 均通过 token-id 回环校验。
- checkpoint 正常版本均发布带 SHA256 的原子 manifest，缺 rank 文件的 partial 版本均不可见；rank exit 与 collective timeout 均非零退出且无残留进程。
- Triton RMSNorm 所有 shape 均通过 FP32 reference；cache eviction 的 20 个阶段请求均 `completed=48, failed=0`。
- `TP2 x PP2 x DP2` process-group 的 6 个 mapping run 全部通过 group-sum oracle；collective 顺序错配对照则在无报错时复现静默数据串接。
- FSDP runtime 包含 15 个稳态 run 和 5 个 rank 0 trace；block wrap 的 AllGather/ReduceScatter 计数、prefetch overlap 与显存峰值均由 artifact 重建。
- vLLM preemption 六轮均 `completed=16, failed=0`；1 GiB/C=16 两次都出现 4 次抢占。Triton 资源 sweep 的 24 个点全部通过 FP32 oracle。

### 15.2 补充实验修正了哪些旧认识

| 旧认识或证据缺口 | 新证据 | 修正后的结论 |
|---|---|---|
| 用 AllReduce 代表所有并行通信 | AllGather/ReduceScatter/SendRecv 拓扑 sweep | 应按 FSDP/SP/PP 实际原语分析，带宽与延迟不完全相同 |
| “更大 microbatch 应该能提高 DDP 效率”只来自推理 | 1/8 卡 sweep + rank 0 trace | m=8 效率升到 51.2%，而 13 个 NCCL kernel 时长近似固定 |
| “长序列时 SP 会更值” | seq512/2048/4096 三轮 sweep | 本实现中 SP 相对显存收益反而缩小；recompute 才是主容量工具 |
| inf-rate throughput 可代表服务容量 | 2-25 QPS Poisson + per-request SLO | completed throughput 上升时 goodput 可崩溃，安全点约为 5 QPS |
| prefix caching 只是可学习概念 | 显式 on/off、cold/warm、两次重启 | 93.7% 高命中场景有 4.32x 吞吐收益，但它是 workload 条件结论 |
| bucket 越大、collective 越少越快 | 4 个 cap 的实际 bucket 日志与 trace | 100 MiB 仅 4 个 NCCL kernel，但 overlap 更低，吞吐比 4 MiB 低约 8.1% |
| 等 effective batch 的梯度累积等价于大 microbatch | m8a1、m1a8 no-sync、m1a8 每次同步 | `no_sync()` 避免 8 倍通信；大 GEMM/少 forward 仍让 m8a1 比 no-sync 快 2.52x |
| 静态拓扑足以预测并发 | 四组 PIX/SYS pair 单独和并发 | PIX aggregate 基本不变；SYS aggregate 从 27.45 降到 13.21 GB/s |
| checkpoint 恢复 model/optimizer 就可逐 bit 复现 | RNG/data cursor/hash/reducer warm 因果对照 | 保存点状态精确不代表 bitwise replay；本例还受 DDP reducer 重建顺序影响 |
| 进程级累计 cache hit rate 能代表体验 | 默认/1 GiB 工作集污染 + phase counter delta | 两者累计命中只差 2 点，小 cache 的 revisit p95 TTFT 却为 warm 的 5.87x |

### 15.3 仍然只是机制解释或有限推断的部分

- FSDP `FULL_SHARD` 已补 rank 0 timeline，并量化 AllGather/ReduceScatter 与 exposed time；但仍不是跨 rank 全局 critical path，也没有覆盖 FSDP2、CPU offload 或真实大模型 auto-wrap。
- PP stage 参数量 2.76x 不平衡是明确事实，但参数量只是 stage time 的 proxy；要证明它贡献了多少吞吐损失，仍需逐 stage profile 后重切层对照。
- vLLM 第一轮某些 shape 变慢可能来自 Triton/JIT 预热，日志不足以唯一归因。
- DP4 的 66.1% 是基于单副本 C=32 的估算效率，不是严格的负载均衡分解；缺少每副本 queue depth 与外部 LB 对照。
- DDP rank 0 Chrome trace 已用区间交集估算 exposed communication，但它仍不是跨 rank critical path，也缺少 Nsight Systems 的 CPU/CUDA/NCCL 全局对齐。
- 本报告没有安装 SGLang，也没有运行 Megatron/vLLM 当前 main branch。因此只能证明固定版本可用，不能把“最新版本一定能/不能跑”写成实验结论。
- MoE 实验是每 rank 单 expert 的数据路径 toy，没有 router loss、capacity/drop、grouped GEMM 或真实 Megatron EP。
- checkpoint 是同 world-size、rank 0 完整 state 的教学协议；没有验证 FSDP/DCP sharded state、异步保存或跨并行配置 reshard。
- Triton 已有 forward correctness、性能、register/stack resource usage 与理论 occupancy 上界；仍没有 backward、autotune，容器也因 `ERR_NVGPUCTRPERM` 无法取得 Nsight Compute hardware counter。
- 3D process-group 实验只验证 communicator、rank mapping 和隔离通信 phase，不等价于已实现端到端 TP layer、1F1B pipeline 与 DP optimizer step。
- 合成 token、短 step 和 dummy weights 不回答收敛、真实质量、数据管线、MFU 或长期稳定性。

## 16. 工程深挖补充结论

完整设置、结果表、实现审计和面试追问见 [工程深挖补充报告](engineering-deep-dive-report.md)。最值得保留的结论是：

1. DDP `bucket_cap_mb` 是调优目标而非任意参数的硬切分上限；collective 次数减少不保证 exposed communication 下降。
2. global batch 相同不等于执行成本相同。`no_sync()` 把 accumulation 的 NCCL kernel 从 104 降到 13，而 m8a1 又依靠更大 GEMM 和更少 forward 比 no-sync 快 2.52x。
3. NCCL 最优算法/协议随消息大小 crossover；256 MiB 强制 2 channel 比 4/8 channel 更快。调优变量应先用于实验和诊断，不应无验证固化到生产。
4. MoE 热点不仅拖慢热点 rank，还会让其他 rank 的等待显现在 combine collective；collective duration 包含 peer wait，不是纯网络时间。
5. 原子 manifest 能阻止 partial checkpoint 被恢复；保存 model/optimizer/RNG/data cursor 能语义恢复，但 DDP reducer 生命周期仍可能改变浮点归约顺序。
6. prefix cache 必须按工作集和时间窗口观测。生命周期累计 hit rate 几乎相同，不能排除关键业务阶段 p95 TTFT 退化近 6 倍。
7. `TP2 x PP2 x DP2` 中把高频 TP 从 PIX 映射到 SYS 后，TP phase 慢 3.05x；rank mapping 要按 logical dimension 的调用频率和 critical path 设计。
8. communicator 按 collective sequence 匹配而不是 tensor identity；同 shape 的跨 rank 调用顺序错配可能不 hang，而是静默把 A/B 归约串接。
9. FSDP block wrap 用更细分片把 allocated peak 降到 root wrap 的约 55%，代价是 25 次 AllGather/13 次 ReduceScatter；`BACKWARD_PRE` 通过 overlap 提升吞吐。
10. vLLM runtime preemption 需要联合判断：1 GiB/C=16 的吞吐只有 4 GiB 的 66.0%，TPOT 更差但 TTFT 略低。Triton 理论 occupancy 同样不能替代硬件 counter。

## 17. 局限与下一轮实验

本轮补齐了最重要的通信、算术强度、activation 策略、缓存和排队证据，但还不能回答完整训练或生产部署问题。后续优先级是：

1. **端到端 3D parallel step**：把已验证的 `TP2 x PP2 x DP2` communicator 接入真实 TP layer、1F1B schedule 和 DP gradient sync，验证 loss、通信依赖与吞吐。
2. **PP stage profile 与重切层**：把 embedding/output 端点成本计入，用真实每 stage F/B/P2P 时间替代参数量 proxy。
3. **FSDP2/DCP reshard**：8 rank sharded save、4 rank restore，测读放大、峰值、optimizer state 和 loss 连续性。
4. **端到端训练**：真实 packed dataset、DataLoader、validation、MFU、checkpoint pause 和数小时 soak，确认微基准能否迁移到 time-to-quality。
5. **生产式推理入口**：四个单卡 server + least-connections/cache-aware LB，补每副本 queue depth、cache 和 30-60 分钟 SLO。
6. **完整 MoE/EP**：top-k router、capacity/drop、aux loss、grouped GEMM、dispatch overlap 与 expert load balancing。
7. **真实 7B 与版本边界**：补真实 7B 权重加载/质量，并分别固定一个可验证的 SGLang 版本、Megatron/vLLM 较新分支做 SM70 兼容矩阵。
8. **系统与能效 profiling**：在开放 GPU performance-counter 权限后，用 Nsight Systems/Compute 标注 NCCL/GEMM/bubble、HBM 事务、achieved occupancy，并加入 token/Joule。

## 18. 参考资料

- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron-Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [vLLM 0.19.1 GPU Installation](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)
- [vLLM 0.19.1 Architecture](https://docs.vllm.ai/en/v0.19.1/design/arch_overview.html)
- [vLLM 0.19.1 Optimization and Tuning](https://docs.vllm.ai/en/v0.19.1/configuration/optimization/)
- [vLLM 0.19.1 Automatic Prefix Caching](https://docs.vllm.ai/en/v0.19.1/design/prefix_caching/)
- [Triton Layer Normalization Tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
