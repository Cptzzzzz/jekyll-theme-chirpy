# 8x V100 AI Infra 工程深挖补充报告

> 实验日期：2026-07-17。机器：单机 8x Tesla V100 PCIe 32 GiB，GPU 间无 NVLink。
>
> 本报告补充主报告中容易被资深面试官继续追问的实现证据。完整原始数据位于 `results/raw/`，统一汇总位于 `results/summary/engineering_deep_dive_summary.json`。

## 1. 为什么要做这轮补充

原实验已经能说明 DDP/FSDP/TP/PP、NCCL 和 vLLM 的基本性能现象，但“跑过 benchmark”和“真正理解工程实现”之间仍有明显差距。深入面试通常会沿下面五条线验证经历真实性：

1. **计量是否正确**：有没有同步 GPU、取最慢 rank、排除 warmup、区分累计 kernel 时间与关键路径时间。
2. **对照是否公平**：global batch、输入 token、模型、精度和请求序列是否一致；是否只改变一个变量。
3. **正确性是否闭环**：collective、路由、恢复、算子和服务是否有独立 correctness oracle。
4. **失败是否可解释**：能否从日志中的 rank、collective sequence number、transport、bucket 和 cache counter 定位问题。
5. **结论边界是否清楚**：能否区分单机 V100 上的实测、受控 toy model 和需要多机/新 GPU 才能验证的外推。

两轮共选择十组高价值实验，而不是继续堆叠相似吞吐数字。第二轮专门下钻到 process-group 契约、FSDP runtime、推理抢占和 kernel 资源占用。

| 方向 | 原证据缺口 | 新增工程证据 | 最重要结论 |
|---|---|---|---|
| DDP | bucket 只看端到端时间；没有 accumulation 对照 | reducer bucket 日志、rank 0 Chrome trace、等 effective batch | bucket 少不等于更快；`no_sync()` 把 104 次 NCCL kernel 降为 13 次 |
| NCCL | 只有默认算法和孤立 pair | Ring/Tree x Simple/LL/LL128、channel、四对 GPU 并发、DEBUG 日志 | 最优算法随消息大小变化；跨 NUMA 四对并发带宽约减半 |
| MoE | 没有 EP/All-to-All 实测 | variable-split dispatch/combine、热点 expert、token-id 回环校验 | max/mean=6 的热点让吞吐下降 66.4%，等待会显现在非热点 rank |
| 可靠性 | 没有 checkpoint/failure injection | 原子 manifest、checksum、RNG/data cursor、rank exit、collective timeout | 状态能语义恢复不代表逐 bit 复现；DDP reducer 生命周期也会影响归约顺序 |
| Kernel | 没有自写 Triton kernel | 单行 FP16 RMSNorm、FP32 reference、shape sweep | 大 shape 约 4.2x 快于 native RMSNorm；小 shape 有约 0.05 ms 启动下限 |
| 推理缓存 | 只有 warm hit，没有驱逐与 counter 分段 | 默认/1 GiB KV、工作集污染、Prometheus phase delta | 全局命中率会掩盖驱逐；小 cache 的 revisit p95 TTFT 变为 warm 的 5.87x |
| 进程组 | 只会列 TP/PP/DP 概念，没有 communicator 级证据 | `TP2 x PP2 x DP2` 分组、rank mapping、并发 collective、顺序错配 | TP 放到 SYS 链路后该 phase 慢 3.05x；同 shape 顺序错配可静默串数据 |
| FSDP runtime | 只有端到端 FSDP 对比 | root/block wrap、PRE/POST/NONE、limiter、rank 0 trace | block wrap 产生 25 次 AllGather/13 次 ReduceScatter；PRE 提升 overlap，但 root wrap 更快且更占显存 |
| 推理调度 | 没有主动触发运行时抢占 | 1/4 GiB KV、C=8/16、250 ms metrics time series | 1 GiB/C=16 每轮 4 次抢占，吞吐仅为 4 GiB 的 66.0%；TTFT 却略低，证明单指标会误导 |
| Kernel 资源 | 没有 register/occupancy 证据 | `num_warps` sweep、cubin、`cuobjdump`、Nsight Compute 权限探针 | hidden=8192 时 1 warp 使用 255 registers/thread 且慢 1.72x；理论 occupancy 不是硬件实测 |

## 2. 统一实验规范

### 2.1 运行与计时

- 分布式训练与通信均通过 `torchrun --standalone --nproc-per-node=8` 或 `nccl-tests` 启动。
- 每个 CUDA wall-time 测量在边界执行 `torch.cuda.synchronize()`；分布式吞吐使用各 rank step time 的最大值。
- 正式 DDP 结果做 5 step warmup、12 step 计量、3 次独立重复。MoE 做 3 step warmup、10 step 计量、3 次重复。
- profiler 只用于分析事件结构；绝对吞吐来自非 profiler 稳态 run，避免 instrumentation overhead 混入结论。
- vLLM 每种 cache 容量重启两次 server，每次严格执行 cold、warm、pollution、revisit、rewarm 五个阶段。

### 2.2 正确性 oracle

| 实验 | 正确性检查 |
|---|---|
| DDP | 最终 loss 有限；每 rank 参数同步由 DDP 语义保证；记录 reducer 日志 |
| NCCL | `nccl-tests` 的 `wrong=0` |
| MoE | 全局唯一 token id 经 dispatch/combine 后逐元素等于原序列 |
| checkpoint | SHA256 manifest、文件长度、模型/optimizer hash、每步 gradient/model hash、各 rank 模型 hash |
| Triton | 与 FP32 RMSNorm reference 比较；所有 shape 通过容差 |
| vLLM | 每阶段 `completed=48`、`failed=0`；Prometheus query/hit token counter 做差分 |

### 2.3 证据等级

- **直接实测**：表格中的时间、吞吐、显存、计数、hash 和日志字段。
- **因果对照支持**：只改变目标变量且结果重复出现，例如 cache 容量和 reducer warm control。
- **机制一致但未唯一归因**：观测符合实现机制，但还缺硬件 counter 或更细粒度 trace。
- **未在本机证明**：多机 RDMA、真实 Megatron EP、跨 world-size reshard、H100/B200 上的绝对性能。

## 3. DDP reducer、bucket 与梯度累积

### 3.1 面试问题与实验假设

需要回答三个工程问题：

1. `bucket_cap_mb` 是严格 bucket 大小上限吗？
2. bucket 数更少、collective 次数更少，是否必然有更高吞吐？
3. global/effective batch 相同时，大 microbatch 和梯度累积是否等价？`no_sync()` 到底消除了什么？

假设是：小 bucket 可以更早发起 AllReduce，但会增加 launch/collective 次数；大 bucket 减少次数但推迟通信。梯度累积若每个 microstep 都同步，会重复归约整套梯度；正确使用 `no_sync()` 后只在最后一次 backward 触发 reducer。

### 3.2 精确设置

- 模型：12 层 decoder-only TinyGPT，hidden 768、12 heads、vocab 32,000、sequence 512。
- 精度：autocast FP16，FP32 参数，GradScaler，fused AdamW；TF32 关闭。
- GPU：8 卡 DDP，`gradient_as_bucket_view=True`。
- bucket sweep：`1/4/25/100 MiB`，microbatch 1，accumulation 1。
- 等 effective batch：每 rank 8 samples，global batch 64。
  - `m8a1`：microbatch 8，accumulation 1。
  - `m1a8 no_sync`：8 个不同输入切片，前 7 次 forward/backward 在 `model.no_sync()` 内。
  - `m1a8 sync every`：8 次 backward 均执行 reducer 同步。
- loss 在每个 microstep 除以 accumulation，避免梯度被放大；输入使用同一个 accumulated batch 的不同 slice，而不是重复消费一条样本。

### 3.3 指标定义

- `tokens/s = global_batch * sequence_length / max_rank_step_seconds`。
- `actual_bucket_count` 和 `actual_bucket_sizes` 来自 `DDP._get_ddp_logging_data()`，不是根据 cap 推测。
- trace 中先对 compute/NCCL kernel 区间分别做 union，再计算交集：
  - `NCCL overlap fraction = |NCCL union ∩ compute union| / |NCCL union|`
  - `exposed NCCL = |NCCL union| - |intersection|`
- 解析器中的 `compute` 精确定义为“所有非 NCCL CUDA kernel”，包含 forward、backward、loss 和 optimizer 等，不等价于只统计 GEMM FLOPs。
- Chrome trace 只采 rank 0 的一个 step，因此用于相对事件结构；端到端均值看三次稳态 run。

### 3.4 Bucket 结果

| cap | 实际 bucket 数 | 最大 bucket | tokens/s，均值 | step，均值 | NCCL kernel | NCCL union | overlap | exposed NCCL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 MiB | 50 | 93.75 MiB | 32,235 | 127.10 ms | 50 | 107.54 ms | 18.85% | 87.27 ms |
| 4 MiB | 37 | 95.26 MiB | 32,815 | 124.88 ms | 37 | 106.84 ms | 20.44% | 85.01 ms |
| 25 MiB | 13 | 95.26 MiB | 31,448 | 130.97 ms | 13 | 109.07 ms | 18.99% | 88.35 ms |
| 100 MiB | 4 | 108.05 MiB | 30,149 | 135.91 ms | 4 | 104.02 ms | 15.05% | 88.36 ms |

关键解释：

- cap 不是把任意参数 tensor 切成固定大小分片的硬上限。该模型有约 93.75 MiB 的大参数，reducer bucket 不能据此任意切开，因此 `cap=1 MiB` 仍出现 93.75 MiB bucket。
- 100 MiB 配置只剩 4 个 NCCL kernel，但吞吐比 4 MiB 配置低约 8.1%。它的 NCCL 总时长略短，overlap 却从 20.44% 降到 15.05%。减少 collective 次数同时推迟了首批通信，关键路径没有缩短。
- 25 MiB 三次结果方差较高，不能据 3 次短 run 宣称 4 MiB 是全局最优。可以可靠陈述的是“本机该 shape 上，单调增大 bucket 没有单调收益”。

### 3.5 等 effective batch 结果

| 配置 | NCCL kernel/step | NCCL union | exposed NCCL | tokens/s，均值 | 相对 `sync every` |
|---|---:|---:|---:|---:|---:|
| m8a1 | 13 | 105.73 ms | 59.46 ms | 218,065 | 6.54x |
| m1a8 + `no_sync()` | 13 | 108.30 ms | 83.73 ms | 86,420 | 2.59x |
| m1a8 + 每次同步 | 104 | 919.15 ms | 717.70 ms | 33,339 | 1.00x |

这组结果分离出两个不同收益来源：

- `no_sync()` 将 8 次 backward 的 reducer 通信从 `8 x 13 = 104` 个 NCCL kernel 降到 13 个，吞吐提高 2.59x。
- 即使通信次数相同，m8a1 仍比 m1a8+`no_sync()` 快 2.52x。原因是 m8a1 只做一次 forward/backward，GEMM 更大，且 launch、LayerNorm、attention 和 Python/框架开销更少。
- effective batch 相同只保证优化语义接近，不保证 kernel shape、执行图、显存峰值和浮点归约顺序相同。

### 3.6 能证明实际经验的工程细节

- `no_sync()` 必须包住相应的 **forward 和 backward**，因为 DDP 在 forward 中准备 reducer 状态；只包 backward 不是可靠用法。
- 不能把 profiler 的 NCCL duration 直接加到 compute duration上。不同 CUDA stream 可重叠，必须对时间区间做 union/intersection。
- 测量分布式 step 要取最慢 rank；只看 rank 0 会掩盖 straggler。
- cap sweep 后必须读取实际 bucket 日志。仅复述命令行参数无法证明真正发送了多少个 collective。

### 3.7 面试追问及高分回答骨架

**问：为什么 bucket 越小不一定越好？**

答：更早 launch 能增加 overlap，但会提高 collective/launch 数、协议延迟和 stream 调度成本；还受大参数不可切分限制。应联合看 actual bucket、首个 NCCL 发起时间、NCCL union、overlap、exposed time 和 step wall-time。

**问：m1a8 和 m8a1 的 loss 很接近，能否证明数值完全等价？**

答：不能。样本和 loss scale 可对齐，但 GEMM shape、梯度累加顺序和 AMP 行为不同。严格验证要从同一初始化运行多 step，比较每步梯度/参数误差和最终质量，而不只比较一个标量 loss。

## 4. NCCL 算法、协议、channel 与拓扑争用

### 4.1 实验问题

默认 NCCL 会自动选择算法和协议。深入面试不要求背某台机器的固定阈值，但会要求解释：

- Ring 与 Tree 分别在哪类消息上可能占优；
- LL、LL128、Simple 的延迟/带宽 trade-off；
- channel 越多为什么可能更慢；
- 两卡 pair 带宽为何不能直接预测 8 卡 collective 或多 job 并发。

### 4.2 精确设置

- `nccl-tests all_reduce_perf`，8 GPU，1 KiB 到 256 MiB，factor 4，10 warmup、30 iterations。
- 表格统一读取 `nccl-tests` 的 out-of-place time/algbw/busbw/wrong 列，避免混用 in-place 口径。
- 强制矩阵：`Ring/Tree x Simple/LL/LL128`，共 6 组；每个 size 均检查 `wrong=0`。
- 256 MiB Ring+Simple 固定 channel 为 `1/2/4/8`，10 warmup、40 iterations。
- 拓扑争用：
  - 本地 PIX pairs：`(0,1) (2,3) (4,5) (6,7)`。
  - 跨 socket/SYS pairs：`(0,4) (1,5) (2,6) (3,7)`。
  - 先分别单独跑，再让四对同时跑，比较 aggregate busbw 和每 pair slowdown。
- 另用 `NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING` 保存 transport/channel 决策。

### 4.3 算法与协议 crossover

| 消息大小 | 本机最快组合 | time | busbw |
|---:|---|---:|---:|
| 1 KiB | Tree + LL | 51.05 us | 0.04 GB/s |
| 4 KiB | Tree + LL | 50.61 us | 0.14 GB/s |
| 16 KiB | Tree + LL | 52.37 us | 0.55 GB/s |
| 64 KiB | Ring + LL | 56.27 us | 2.04 GB/s |
| 256 KiB | Ring + LL128 | 142.5 us | 3.22 GB/s |
| 1 MiB | Ring + Simple | 293.6 us | 6.25 GB/s |
| 4 MiB | Ring + Simple | 1,028.9 us | 7.13 GB/s |
| 16 MiB | Ring + Simple | 4,119.5 us | 7.13 GB/s |
| 64 MiB | Tree + Simple | 16,606 us | 7.07 GB/s |
| 256 MiB | Tree + Simple | 64,811 us | 7.25 GB/s |

这不是应背诵的静态阈值。阈值由 NCCL 版本、拓扑、GPU 架构、channel、消息大小、并发和协议支持共同决定。实验价值在于证明候选人知道如何校准 cost model，而不是假设 Ring 永远适合大消息、Tree 永远只适合小消息。

### 4.4 Channel 与并发争用

| 固定 channel | 256 MiB Ring+Simple busbw |
|---:|---:|
| 1 | 6.92 GB/s |
| 2 | 6.98 GB/s |
| 4 | 6.15 GB/s |
| 8 | 6.40 GB/s |

更多 channel 会占用更多 CUDA blocks/SM 资源，也会增加调度和链路竞争；本机 2 channel 最快，8 channel 并没有更高带宽。当前 NCCL 官方文档也把这类强制变量定位为调试/实验手段，并在新版本中推荐 CTA 相关变量替代旧 `MIN/MAX_NCHANNELS`。本机安装的 NCCL 2.22.3 仍接受旧变量，但不应把强制配置直接复制到生产。

| pair family | 单独运行 aggregate | 四对并发 aggregate | 平均每 pair slowdown | 最大 slowdown |
|---|---:|---:|---:|---:|
| 四组 PIX | 47.15 GB/s | 47.14 GB/s | 1.000x | 1.001x |
| 四组 SYS | 27.45 GB/s | 13.21 GB/s | 2.080x | 2.100x |

DEBUG 日志显示 PIX pair 使用 P2P/direct；部分跨域路径使用 SHM/direct。四组本地 pair 几乎互不影响，四组跨 socket pair 并发时共享 host/PCIe/NUMA 路径，aggregate 下降 51.9%。这比一张静态 `nvidia-smi topo -m` 更接近真实多 process-group 或多 job 的竞争行为。

### 4.5 工程面试细节

- `algbw` 与 `busbw` 口径不同；比较算法时必须确认使用相同 size、rank 数和指标定义。
- 强制 `NCCL_PROTO=LL128` 在不支持的拓扑/硬件上可能不只是变慢，还可能不被支持；生产应优先让 NCCL 自动调优并用日志验证。
- topology-aware placement 不能只把 TP 放在最快 pair，还要考虑多个 group 同时工作时是否共享 switch、CPU socket 或 NIC。
- 单机只能证明 PCIe/NUMA 争用，不能把 SHM/P2P 结果外推为 InfiniBand/RoCE、GPUDirect RDMA 或交换网络拥塞。

## 5. MoE variable-split All-to-All 与热点 expert

### 5.1 精确设置与数据流

- 8 ranks，每 rank 4,096 tokens，总计 32,768 tokens；hidden 1,024，expert FFN hidden 4,096，FP16。
- 每个 rank 持有一个本地 expert，执行 `hidden -> 4x hidden -> hidden` 两次 GEMM 和 GELU。
- dispatch 前先用一次 `all_to_all_single` 交换每个 destination 的 token count，形成 variable `input_split_sizes/output_split_sizes`。
- dispatch 后本地 expert 计算，再交换相反 split 完成 combine。
- balanced：每个 source 向每个 expert 发 512 tokens。
- hot expert：每个 source 的 75% token 发往 expert 0，其余均分。
- 正确性：token id 使用同样 split 做 dispatch/combine，最终必须逐元素恢复原顺序。

### 5.2 结果

| 路由 | 每 rank received tokens | max/mean | dispatch max | expert compute max | combine max | step max | global tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced | `[4096 x 8]` | 1.0 | 1.712 ms | 1.258 ms | 1.788 ms | 4.693 ms | 6.988M |
| 75% hot | `[24576, 1176, 1176, 1168, ...]` | 6.0 | 4.704 ms | 5.401 ms | 9.701 ms | 13.953 ms | 2.350M |

热点使吞吐下降约 66.4%。更重要的工程现象是：

- expert 0 收到 24,576 tokens，计算时间显著增加。
- 非热点 rank 会较早到达 combine collective，但必须等待热点 rank 完成计算。于是 profile 中长时间可能记在 **combine/collective** 上，而根因是上游 expert compute imbalance。
- collective duration 包含等待 peer 的时间，不等于纯链路传输时间。只看 NCCL kernel 长度可能误判为网络退化。

### 5.3 这还不是真实 Megatron EP

这个 toy benchmark 实测了 variable-split All-to-All、token permute 语义、expert GEMM 和热点等待，但没有实现：top-k router、capacity factor、token drop、aux/z-loss、grouped GEMM、shared expert、dispatch/combine overlap、expert replication 或 EPLB。因此可陈述“做过 MoE 数据路径与负载不均衡实验”，不能声称“完成了生产 Megatron MoE 优化”。

### 5.4 深入追问

**问：平均 token 数很均匀，step 仍有长尾，还看什么？**

答：看每 expert p95/max token、token shape 导致的 GEMM efficiency、expert 所在拓扑、不同 source-destination 流量矩阵、grouped GEMM padding、router skew 的时间相关性，以及是否有某 rank 在 dispatch 前就晚到。

**问：capacity factor 如何影响系统？**

答：更大 capacity 降低 drop、提高质量稳定性，但增加 padding、显存和最坏通信/计算；更小 capacity 限制尾部成本但会 drop/reroute token。应同时观察 dropped-token rate、expert max/mean、tokens/s、loss/aux loss 和质量。

## 6. Distributed checkpoint、位级恢复与故障注入

### 6.1 Checkpoint 协议实现

实验使用一个教学型原子协议：

1. rank 0 将完整 model/optimizer/step 写到临时文件，再 `os.replace` 为正式文件。
2. 每个 rank 独立写 CPU RNG、CUDA RNG、data generator state、step/data cursor。
3. barrier 后由 rank 0 检查所有预期文件，计算 bytes 和 SHA256。
4. 只有文件全部存在时，最后原子发布 `manifest.json`，它是 checkpoint 可见性的 commit point。
5. load 必须先看到 manifest，再逐个验证文件长度与 checksum；world size 不匹配直接拒绝。

模型是 hidden 1,024 的两层 MLP，含 GELU/Dropout；batch 32，FP32 DDP，AdamW。连续运行 6 step，对照在 step 3 save，重启后恢复 step 4-6。每一步记录 rank 0 gradient SHA256 和更新后的 model SHA256。

这套协议的目的不是替代 PyTorch Distributed Checkpoint，而是把一致性条件做成可检查的最小系统。它保存的是 rank 0 完整 model/optimizer，不支持 sharded state 或新 world size reshard。

### 6.2 原子提交结果

- 5 组 optimizer/determinism/reducer 对照的正常 checkpoint 均 `checkpoint_committed=true`。
- 每组 partial case 都让 rank 3 跳过文件写入，所有 case 均未生成 manifest，`partial_checkpoint_not_committed=true`。
- 约 50 MB model/optimizer 加少量 rank-local 文件，五组正常 save 为 0.206-0.257 s，rank 0 load 为 0.211-0.710 s。短 run 的 load 波动明显；该数字包含本地文件系统和 checksum，不代表远端对象存储吞吐。
- load 后保存点的 model SHA 和 optimizer SHA 均与 save 端完全一致。

### 6.3 为什么状态完全加载后仍不逐 bit 一致

| 对照 | 保存点 model/optimizer 精确加载 | 首个 gradient mismatch | 最终最大参数绝对差 | 不同元素数 | 最终 bitwise exact |
|---|---|---:|---:|---:|---|
| fused AdamW | 是 | step 4 | 3.725e-9 | 213,194 / 4,197,376 | 否 |
| single-tensor AdamW | 是 | step 4 | 3.725e-9 | 210,883 / 4,197,376 | 否 |
| deterministic algorithms | 是 | step 4 | 3.725e-9 | 210,883 / 4,197,376 | 否 |
| `static_graph=True` | 是 | step 4 | 5.588e-9 | 215,216 / 4,197,376 | 否 |
| resume 前预热 reducer 3 次 | 是 | 无 | 0 | 0 | 是 |

排查顺序很关键：

1. 先验证 model、optimizer、CPU/CUDA RNG、data generator 和 cursor 均已恢复。
2. fused optimizer 改为 single-tensor，差异仍在，排除 fused AdamW 是唯一原因。
3. 启用 deterministic algorithms，差异仍在，说明问题不只是常见 nondeterministic kernel。
4. 启用 `static_graph=True`，差异仍在。
5. 对比 `_get_ddp_logging_data()`：连续 run 在后续迭代已 `has_rebuilt_buckets=1`，新建 DDP 后直接 load 的 reducer 生命周期不同。
6. resume 前执行 3 次 dummy backward，使 reducer 先重建到同样的 bucket/参数顺序，再加载 model/optimizer/RNG。此时 step 4 的 gradient hash、每步 model hash、loss sequence 和最终参数全部逐 bit 一致。

这是一组强因果对照，支持“本实验差异来自 DDP reducer/bucket 生命周期改变了浮点归约顺序”。它不意味着生产 checkpoint 必须序列化 PyTorch reducer 内部对象，也不意味着一般训练必须追求逐 bit 一致。生产更常见的验收层次是：状态完整、数据不重不漏、loss 连续、短窗口误差有界、长期质量一致。位级复现主要用于回归测试和缩小故障范围。

### 6.4 故障注入

| 注入 | 设置 | 观测 | 退出/清理 |
|---|---|---|---|
| rank exit | rank 3 在 step 2 `os._exit(17)` | torch elastic 报 `ChildFailedError` 并标出 rank/exitcode | torchrun 约 9.06 s 非零退出；无残留子进程 |
| straggler timeout | rank 3 睡眠 11 s；PG timeout 6 s | watchdog 标出 `SeqNum=3`、`ALLREDUCE`、`NumelIn=1048576` | 外层约 15.56 s 非零退出；所有 rank 终止 |

工程上要区分：

- rank 主动退出由 launcher/elastic agent 首先观测；
- rank 仍存活但不进入 collective，其他 rank 只能通过 process-group watchdog/timeout 发现；
- 日志中的 collective sequence number、op type、numel 和最后完成序号是对齐各 rank 现场的重要线索；
- 只确认主进程失败不够，还要验证 child、共享内存和 GPU context 被清理，否则重试会受到污染。

### 6.5 与生产 DCP 的边界

PyTorch Distributed Checkpoint 支持多 rank 并行写分片、load-time reshard 和异步保存等能力。本实验没有覆盖：FSDP/DTensor state dict、不同 world size 恢复、远端对象存储、异步 snapshot 一致性、增量 checkpoint、限速和 GC。面试时应主动把“我实现了原子 manifest 原理验证”和“我做过生产级 DCP”区分开。

## 7. Triton RMSNorm：从 correctness 到性能模型

### 7.1 Kernel 映射

- 输入和 weight 为 FP16；每个 Triton program 处理一行 hidden vector。
- `BLOCK_SIZE = next_power_of_2(hidden_size)`，越界元素由 mask 屏蔽。
- 将输入转 FP32，计算 `sum(x*x)`、`rsqrt(mean + eps)`，乘 weight 后写回 FP16。
- kernel 将归约、归一化和 scale 融合在一次 program 中；wrapper 每次调用先执行 `torch.empty_like`，因此当前计时包含输出分配。
- 对照为 PyTorch `F.rms_norm` 和显式 composite 实现；正确性 reference 全程 FP32。

### 7.2 结果

| rows | hidden | Triton | PyTorch native | 对 native 加速 | 对 composite 加速 | 最小字节模型有效带宽 | max abs error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 768 | 0.0500 ms | 0.0663 ms | 1.33x | 1.96x | 63.0 GB/s | 0.001953 |
| 1,024 | 4,096 | 0.0501 ms | 0.1167 ms | 2.33x | 4.46x | 335.5 GB/s | 0.001953 |
| 1,024 | 8,192 | 0.0500 ms | 0.2029 ms | 4.06x | 8.10x | 671.5 GB/s | 0.003906 |
| 8,192 | 768 | 0.0492 ms | 0.1578 ms | 3.21x | 6.37x | 512.0 GB/s | 0.001953 |
| 8,192 | 4,096 | 0.1685 ms | 0.7233 ms | 4.29x | 8.92x | 796.4 GB/s | 0.003906 |
| 8,192 | 8,192 | 0.3333 ms | 1.4007 ms | 4.20x | 8.87x | 805.5 GB/s | 0.003906 |

### 7.3 如何正确解释

- rows 1,024 的三个 shape 都接近 0.05 ms，说明 launch/scheduling 固定开销主导，不能用该点拟合内存带宽。
- 大 shape 的约 4.2x native 加速是本环境、该 PyTorch/Triton 版本和 shape 的结果。需要 profiler 检查 native 是否包含额外 kernel/中间访存，才能精确归因。
- 表中“有效带宽”使用最小字节模型：读 input、写 output、读一次 weight。它不是 Nsight Compute 的 HBM hardware counter，也没有计入 cache line、重复 weight 读取或其他事务。
- 正确性不能只与 FP16 native 比较，否则两个实现可能共享同类误差。这里先用 FP32 reference 定义 oracle，再报告 max abs error。

### 7.4 面试可继续深挖

- hidden 超过单 program 可承受的 shared/register 规模时，为什么要做 two-pass reduction 或分块？
- power-of-two padding 会浪费多少 lane/读请求，何时应提供多个 kernel specialization？
- occupancy、register spilling、vectorized load、L2 hit 和 memory coalescing 如何验证？
- backward 需要保存什么，如何融合 weight gradient，如何做 autotune 与 shape cache？

## 8. vLLM prefix cache 驱逐与工作集实验

### 8.1 为什么只看 warm benchmark 不够

一次 cold/warm 对照只能证明存在复用收益，不能回答 cache 容量不足、多租户污染和 LRU 驱逐后的尾延迟。生产面试会继续问：工作集有多大、cache 能容纳多少 block、污染后是否仍命中、指标是瞬时还是累计。

### 8.2 设置

- vLLM 0.19.1，单卡 eager，Qwen2.5-7B dummy weights，automatic prefix caching 开启。
- 默认 cache：225,200 KV tokens；按 max model length 2,048 计算的日志上限约 109.96x concurrency。
- 小 cache：`--kv-cache-memory-bytes 1G`，18,720 KV tokens，上限约 9.14x。
- 每个阶段 48 requests，max concurrency 8，prefix 1,024、suffix 32、output 16、`ignore-eos`。
- anchor 使用 4 个共享 prefix；pollution 使用 48 个不同 prefix。
- 固定顺序：anchor cold -> 同 anchor warm -> pollution -> anchor revisit -> 同 anchor rewarm。
- 每阶段结束抓 `/metrics`；因为 Prometheus counter 是累计值，解析器使用相邻快照做差分。

### 8.3 结果

| cache | cold mean TTFT | warm mean TTFT | revisit mean TTFT | rewarm mean TTFT | revisit/warm mean | revisit/warm p95 | revisit hit tokens | revisit throughput/warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 415.68 ms | 173.12 ms | 167.95 ms | 169.90 ms | 0.97x | 0.94x | 49,920/50,688 = 98.48% | 1.014x |
| 1 GiB | 385.50 ms | 159.81 ms | 321.58 ms | 161.37 ms | 2.01x | 5.87x | 45,056/50,688 = 88.89% | 0.787x |

默认 cache 经污染仍保留 anchor；1 GiB cache 的 revisit 表现退回接近 cold 阶段，随后 rewarm 恢复。88.89% 不是“仍保留了大部分 anchor”的充分证据，因为同一批 48 请求内部也会互相填充并复用。它恰好与 cold phase 相同，结合 TTFT 和下一阶段恢复，支持 anchor 在 pollution 后已被驱逐。

### 8.4 为什么全局命中率会骗人

server 结束时日志中的累计 prefix hit rate：默认 76.9%，小 cache 74.9%，只差 2 个百分点；但小 cache 的 revisit p95 TTFT 是 warm 的 5.87x。累计值混合了 cold、warm、无复用 pollution 和 rewarm，掩盖了业务关键阶段。生产告警至少要按模型、租户、route 和时间窗口观察：

- phase/window hit token rate，而不是进程生命周期累计值；
- cache used/free blocks、eviction/recompute rate、可容纳 tokens；
- TTFT p50/p95/p99、SLO goodput、prefill tokens/s；
- prefix 长度/复用距离/租户工作集，及 cache-aware routing 的命中收益和负载偏斜。

### 8.5 服务生命周期细节

runner 用独立 process group 启动 server，循环健康检查 `/health`，每阶段保存 benchmark 与 metrics；退出时先向整个 process group 发 TERM，超时后 KILL 并 `wait` 回收。这个细节用于避免 engine worker 残留占用 GPU。仅看到客户端 `failed=0` 仍不够，server 端 OOM、重计算、取消和尾延迟必须一起看。

## 9. 对原实验结论的整体修正

### 9.1 已由补充实验闭环的内容

| 原结论 | 当前更准确的表述 |
|---|---|
| 小 microbatch 的 DDP 通信占比高 | 固定梯度 bytes 时 NCCL 时间近似固定；大 microbatch 同时提高 GEMM 效率和 overlap。梯度累积只有配合 `no_sync()` 才避免重复同步 |
| bucket 需要调优 | cap 是 soft target 且受大参数约束；应从 actual bucket 和 exposed communication 判断，不看 collective 数量单指标 |
| PCIe 拓扑影响带宽 | 不仅静态 pair 不同，并发流量还会让跨 socket aggregate 从 27.45 降到 13.21 GB/s |
| MoE 会受负载不均衡影响 | variable-split 数据路径实测 max/mean=6 使吞吐下降 66.4%，且等待可能被记在下游 collective |
| checkpoint 要保存 model/optimizer/RNG | 还要保存 data cursor、做原子 commit/checksum，并明确 framework runtime state 对 bitwise replay 的影响 |
| prefix cache 高命中有收益 | cache 容量和复用距离决定 anchor 是否存活；全局累计 hit rate 不能代表关键阶段 SLO |
| 应学习 Triton | 已完成一个带 FP32 oracle、shape sweep 和带宽模型的融合 RMSNorm，但尚未做 backward/硬件 counter |

### 9.2 仍然缺失、且值得下一轮做的实验

按面试价值和本机可完成性排序：

1. **端到端 `TP2 x PP2 x DP2` 模型**：本轮完成了三类 process group、rank mapping 和通信阶段，但还没有把真实 TP layer、1F1B pipeline、DP gradient sync 组合到同一个训练 step。
2. **PP per-stage trace 与重切层**：当前 2.76x 参数不均衡只是 proxy；应记录每 stage F/B/P2P 时间，做非均匀 partition 后验证 bubble 是否下降。
3. **PyTorch DCP/FSDP2 跨 world-size reshard**：例如 8 rank 保存、4 rank 恢复，检查峰值、读放大、optimizer state 和 loss 连续性。
4. **真实数据管线与 MFU**：加入 packed dataset、DataLoader、H2D、validation、checkpoint pause 和至少数小时 soak；当前合成静态 input 不能证明 time-to-quality。
5. **服务副本可观测性**：四个单卡 server + least-connections/cache-aware LB，记录每副本 queue depth、cache hit、GPU busy 和 30-60 分钟 SLO。
6. **完整 MoE router/EP**：top-k、capacity/drop、aux loss、grouped GEMM、dispatch overlap 和 expert load balancing。本轮 toy 只覆盖核心数据路径。
7. **context parallel/长上下文**：在可承受模型上验证 ring/AllGather KV 数据流、通信字节和 attention correctness。
8. **硬件 counter**：当前容器无 GPU performance-counter 权限；需要管理员按 NVIDIA 指引开放后，用 Nsight Compute 验证实际 occupancy、dram bytes、L2 hit 和 stall reason。

本机无法可信补齐的内容包括：多节点 RDMA、NIC rail/SHARP、交换网络拥塞、大规模 elastic membership，以及 H100/B200 的 NVLink/NVSwitch、BF16/FP8 和新 attention kernel。对此应准备设计、脚本和指标，不应伪造本地实测结论。

## 10. 工程证据审计清单

面试前可用下面的问题检查每一条简历描述：

1. 是否能说出完整命令、模型 shape、dtype、global batch 和 warmup/计量步数？
2. 吞吐分母用 rank 0、mean rank 还是 max rank，为什么？
3. GPU 异步执行在哪些位置 synchronize，barrier 是否属于计量窗口？
4. profiler 看到的累计 kernel duration 是否做了区间去重和 overlap？
5. 是否有三次以上重复、方差和异常 run，结论是否依赖单点？
6. 每个实验的 correctness oracle 是什么，为什么 `loss finite` 或 HTTP 200 仍可能不够？
7. 是否保存原始日志、结构化 JSON、版本、环境变量和实际 runtime 决策？
8. 失败时能否指出第一个异常 rank、collective、sequence number、输入规模和清理结果？
9. 哪些结论是直接实测，哪些是因果对照，哪些只是机制推断？
10. 如果换 H100/NVSwitch 或多机 RDMA，哪些数字必须重新校准，哪些分析方法仍成立？

## 11. 复现命令与产物

```bash
# DDP reducer、bucket、gradient accumulation 和 rank 0 trace
bash scripts/run_ddp_engineering_bench.sh

# NCCL algorithm/protocol/channel 和并发 pair
bash scripts/run_nccl_engineering_bench.sh

# MoE variable-split All-to-All
bash scripts/run_moe_alltoall_bench.sh

# 原子 checkpoint、resume、rank exit 和 straggler timeout
bash scripts/run_resilience_bench.sh

# Triton RMSNorm
bash scripts/run_triton_rmsnorm_bench.sh

# 第二轮：process group、FSDP runtime 与 Triton 编译资源
bash scripts/run_parallel_group_runtime_bench.sh
bash scripts/run_fsdp_runtime_bench.sh
bash scripts/run_triton_resource_bench.sh

# vLLM cache eviction；runner 自行管理 server 生命周期
bash scripts/run_vllm_cache_eviction_bench.sh

# vLLM KV 压力与 runtime preemption
bash scripts/run_vllm_preemption_bench.sh

# 汇总与全仓库结果校验
python3 scripts/parse_engineering_deep_dive.py
python3 scripts/validate_results.py
```

主要产物：

- `results/raw/ddp_engineering/`：21 个稳态 run、7 个 profile metrics/events/Chrome trace。
- `results/raw/nccl_engineering/`：算法协议矩阵、channel、pair 并发和 DEBUG 日志。
- `results/raw/moe_alltoall/`：balanced/hot expert 三次重复。
- `results/raw/resilience/`：5 组恢复对照、state snapshots、manifest、故障日志/status。
- `results/raw/triton_rmsnorm/`：3 次 shape sweep。
- `results/raw/vllm_cache_eviction/`：20 个 phase benchmark、20 个 metrics snapshot、4 份 server log。
- `results/raw/parallel_group_runtime/`：6 个 mapping run 和 collective 顺序错配对照。
- `results/raw/fsdp_runtime/`：15 个稳态 run、5 组 profile metrics/events/Chrome trace。
- `results/raw/triton_resources/`：3 次资源 sweep、cubin resource usage 和 `ncu` 权限探针。
- `results/raw/vllm_preemption/`：6 个请求 benchmark、6 组 metrics time series 和 4 份 server log。
- `results/summary/engineering_deep_dive_summary.json`：统一机器可读汇总。
- `results/summary/*engineering*.csv`、`parallel_group_runtime_runs.csv`、`fsdp_runtime_runs.csv`、`fsdp_runtime_trace_summary.csv`、`triton_resource_runs.csv`、`vllm_preemption_runs.csv`：面向表格分析的明细。

## 12. Process group：rank mapping、调用顺序与 communicator 契约

### 12.1 设置和分组

这组实验不实现完整 Transformer，而是先隔离 3D parallel 最容易出错的 runtime 层。8 个 rank 显式建立 `TP=2, PP=2, DP=2` 三维网格；所有 rank 都按同一全局顺序调用每一个 `new_group()`，包括自己不属于的 group。每次传输 64 MiB，TP phase 连续做 4 次 AllReduce，PP/DP phase 各做 1 次，3 次独立重复。

| mapping | TP groups | PP groups | DP groups | 目的 |
|---|---|---|---|---|
| `tp_local` | `(0,1),(2,3),(4,5),(6,7)`，PIX | `(0,2),(1,3),(4,6),(5,7)`，PHB | `(0,4),(1,5),(2,6),(3,7)`，SYS | 高频 TP 留在最短链路 |
| `tp_cross` | `(0,4),(1,5),(2,6),(3,7)`，SYS | PIX pairs | PHB pairs | 只交换 logical dimension 与物理链路的对应关系 |

每个 group 的正确性 oracle 是 rank 值求和后的解析期望，不只检查 collective 是否返回。另做一组独立 pair communicator 对照：逻辑 tensor A/B shape 相同，一侧按 A 后 B，另一侧按 B 后 A 调用 AllReduce。

### 12.2 结果

| mapping | TP phase，4 次 | PP phase | DP phase | 串行总计 | TP+PP+DP 各一次并发 |
|---|---:|---:|---:|---:|---:|
| `tp_local` | 25.55 ms | 17.54 ms | 20.18 ms | 62.99 ms | 37.46 ms |
| `tp_cross` | 77.92 ms | 6.39 ms | 17.93 ms | 102.07 ms | 37.42 ms |

- 把高频 TP 从 PIX 映射到 SYS 后，TP phase 慢 **3.05x**，整个串行工作负载慢 **1.62x**。这直接说明 rank mapping 不是命名问题，而是 critical-path placement。
- 两种 mapping 的“一次 TP + 一次 PP + 一次 DP 并发”都约 37.4 ms，因为每种方案实际都同时包含一组 PIX、PHB、SYS 流，只是 logical label 互换。它不能推翻 TP mapping 结论，反而说明必须按调用频率和依赖路径加权，而不是只列 group 拓扑。
- 全部 group-sum correctness 通过。该实验仍不是完整 3D 训练：没有 TP layer、pipeline schedule、microbatch dependency 或 DP gradient bucket。

### 12.3 顺序错配为何可能静默错误

ordered 对照得到 rank 0 的 `A=2001, B=4001`。把 rank 1 上 A/B 的调用顺序反转后，两次 NCCL collective 都完成，但两侧最终都得到 `A=3001, B=3001`，逻辑正确性失败且没有 hang。

底层原因是 communicator 按 collective sequence 匹配，不认识 Python 变量名或业务 tensor identity。同 communicator、同 shape/op 的顺序错配可以把 A 与 B 配成第一轮、B 与 A 配成第二轮，形成**静默数据损坏**；shape/op 不兼容时才更可能表现为报错或 hang。因此工程上需要：

1. 所有 rank 使用一致的 communicator 创建顺序与 collective 调用顺序。
2. 把 logical step、group id、collective sequence、shape/dtype 写入可观测日志。
3. 调试时加入小规模语义 checksum/oracle；“没有超时”不是 correctness。
4. 多 communicator 并发时保存 `Work` 生命周期，并明确在哪个 stream/event 上建立依赖。

## 13. FSDP runtime：wrap、prefetch、limiter 与退出生命周期

### 13.1 公平设置

- 同一 12 层 TinyGPT：hidden 768、12 heads、vocab 32,000、sequence 256、microbatch 1，8 卡。
- 所有配置使用 `FULL_SHARD`、FP16 parameter/reduce/buffer dtype、`use_orig_params=True`、fused AdamW。
- `root` 只有 1 个 FSDP unit；`block` 用 `GPTBlock` transformer auto-wrap，共 13 个 unit（12 个 block 加外层 root unit）。
- 扫描 `BACKWARD_PRE`、`BACKWARD_POST`、无 prefetch，以及 `limit_all_gathers=True/False`。
- 每个配置 5 step warmup、8 step 计量、3 次重复；另外 profile 一个 rank 0 step。显存是重置 peak stats 后的本进程 CUDA allocated/reserved 峰值。

### 13.2 稳态性能与显存

| case | units | tokens/s | step | allocated peak | reserved peak |
|---|---:|---:|---:|---:|---:|
| root + PRE | 1 | 26,102 | 78.50 ms | 0.981 GiB | 2.277 GiB |
| block + PRE | 13 | 21,030 | 97.38 ms | 0.539 GiB | 1.174 GiB |
| block + POST | 13 | 19,325 | 105.98 ms | 0.539 GiB | 1.160 GiB |
| block + NONE | 13 | 19,855 | 103.16 ms | 0.539 GiB | 1.156 GiB |
| block + PRE + no limiter | 13 | 21,553 | 95.03 ms | 0.539 GiB | 1.269 GiB |

root wrap 比 block+PRE 快 **1.24x**，但 allocated peak 是 **1.82x**。block+PRE 比 POST 快 8.8%、比 NONE 快 5.9%。关闭 all-gather limiter 只提升 2.5%，reserved peak 却增加 8.1%，说明 limiter 的价值是抑制预取造成的 in-flight allocation 压力，不保证每个小模型都更快。

### 13.3 Trace 如何对应状态机

| case | AllGather kernels | ReduceScatter kernels | NCCL union | overlap | exposed NCCL |
|---|---:|---:|---:|---:|---:|
| root + PRE | 1 | 1 | 63.14 ms | 0% | 63.14 ms |
| block + NONE | 25 | 13 | 86.12 ms | 0% | 86.12 ms |
| block + POST | 25 | 13 | 84.74 ms | 0% | 84.74 ms |
| block + PRE | 25 | 13 | 85.53 ms | 13.08% | 74.34 ms |
| block + PRE + no limiter | 25 | 13 | 85.64 ms | 14.15% | 73.52 ms |

- root unit 在 forward 前一次 AllGather，backward 后一次 ReduceScatter；通信次数少，但完整参数驻留窗口大。
- block wrap 观测到 13 次 forward AllGather、12 次 backward 参数重聚合和 13 次 gradient ReduceScatter，即 25/13。这个计数来自当前 graph/wrap 的 trace，不应背成所有 FSDP 模型的固定公式。
- PRE 没有减少 NCCL union，收益来自约 11.19 ms AllGather 与 compute 重叠；POST/NONE 在这个 rank 0 trace 中为 0。ReduceScatter 没有观测到 overlap。
- profiler step 有 instrumentation overhead，绝对吞吐取三次非 profiler 稳态结果。`compute` 仍定义为非 NCCL CUDA kernel union，不是只算 GEMM，也不是跨 rank 全局 critical path。

### 13.4 一个真实的退出错误

第一次 smoke run 的数值和 JSON 都成功，但进程退出时出现 NCCL proxy `Close` error。根因不是训练计算，而是末尾 default process-group GPU `all_gather` 的生命周期没有在所有 rank 销毁 communicator 前完全闭环。修复是在写结果后执行 `torch.cuda.synchronize()` 和带 `device_ids` 的最终 barrier，再统一 `destroy_process_group()`；复跑日志干净。

这类问题很适合面试判断真实经验：Python 调用返回、`async_op` 的 `Work` 完成、CUDA stream 完成、所有 peer 离开 collective 和 communicator 可销毁不是同一个时刻。退出阶段也需要同步协议，不能因为结果文件已经写出就认为 run 完整成功。

## 14. 推理抢占与 Triton 编译资源

### 14.1 vLLM runtime preemption

服务使用 vLLM 0.19.1、单卡 V100、Qwen2.5-7B dummy、eager、prefix cache 关闭，固定 `max_num_seqs=32`、`max_num_batched_tokens=2048`。每轮 16 个随机请求，prompt 1,024、output 512、`ignore_eos`，比较 1 GiB/C=8、1 GiB/C=16、4 GiB/C=16，各重复两次。runner 每 250 ms 抓一次 `/metrics`，保存 counter delta 与 gauge 峰值。

| KV / concurrency | cache tokens | preemptions | peak KV | peak running/waiting | total tok/s | mean TTFT | mean TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB / 8 | 18,720 | 0 | 65.7% | 8 / 3 | 660.7 | 1,588 ms | 33.27 ms |
| 1 GiB / 16 | 18,720 | 4 | 99.9% | 16 / 13 | 742.7 | 2,601 ms | 40.86 ms |
| 4 GiB / 16 | 74,896 | 0 | 32.8% | 16 / 11 | 1,125.4 | 2,931 ms | 36.83 ms |

16 个 prompt 初始占约 `16 x 1024 = 16,384` tokens，能放入 18,720-token cache；decode 继续增长后越过容量，才在运行中发生 4 次 preemption/recompute。C=8 工作集没有越界，4 GiB 则有足够余量。

1 GiB/C=16 的总 token 吞吐只有 4 GiB 的 **66.0%**，mean TPOT 慢 11.0%，但 mean TTFT 反而低 11.3%。合理解释是更小 cache 和抢占形成了隐式并发节流，降低部分请求 prefill 的同时竞争，却用重计算和更长 decode 换取代价。这里不能用 TTFT 单指标宣布小 cache 更优；应联合看 TTFT、TPOT、throughput、waiting、KV usage 和 preemption counter。metrics 是 250 ms 采样，gauge 峰值仍可能漏掉更短瞬态，counter delta 更适合统计事件数。

### 14.2 `num_warps`、register 与 occupancy 上界

资源实验改为预分配输出，只计 RMSNorm kernel；rows=8,192，hidden=4,096/8,192，`num_warps=1/2/4/8`，每点 20 warmup、100 iterations、3 次重复。Triton 编译 cubin 用 `cuobjdump --dump-resource-usage` 读取 registers/thread、stack/static shared/local memory；occupancy 仅按 V100 的 65,536 registers、2,048 threads、64 warps、32 blocks 和 96 KiB shared-memory 上限推算。

| hidden | warps | registers/thread | stack | 理论 occupancy 上界 | kernel | 最小字节有效带宽 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 | 168 | 0 B | 18.75% | 0.1737 ms | 772.6 GB/s |
| 4,096 | 2 | 99 | 0 B | 31.25% | 0.1694 ms | 792.2 GB/s |
| 4,096 | 4 | 52 | 0 B | 56.25% | 0.1687 ms | 795.9 GB/s |
| 4,096 | 8 | 32 | 0 B | 100.00% | 0.1682 ms | 797.8 GB/s |
| 8,192 | 1 | 255 | 384 B | 12.50% | 0.5707 ms | 470.4 GB/s |
| 8,192 | 2 | 172 | 0 B | 15.63% | 0.3391 ms | 791.7 GB/s |
| 8,192 | 4 | 99 | 0 B | 31.25% | 0.3331 ms | 805.8 GB/s |
| 8,192 | 8 | 54 | 0 B | 50.00% | 0.3323 ms | 808.0 GB/s |

hidden=8192 时，1 warp 的 register/thread 是 8 warps 的 4.72x，并出现 stack 使用，kernel 慢 **1.72x**。这支持“并行 reduction 映射改变 live value/register pressure，限制 resident warps”的机制，但不能声称 occupancy 已被硬件 counter 直接测量。hidden=4096 从 2 到 8 warps 性能几乎饱和，也说明 occupancy 上界提高不等于吞吐同比提高。

Nsight Compute `ncu --query-metrics` 返回文本 `ERR_NVGPUCTRPERM`，但进程退出码仍为 0。自动化必须解析诊断文本，不能只看 shell status。由于容器没有 performance-counter 权限，本报告不提供 dram bytes、L2 hit、achieved occupancy 或 stall reason；`cuobjdump` 回答的是编译资源，Nsight Compute 才能回答运行时硬件计数。

## 15. 官方资料

- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch `new_group` and process-group ordering](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)
- [PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [NVIDIA NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA GPU performance-counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/v0.19.1/design/prefix_caching/)
- [vLLM 0.19.1 optimization and tuning](https://docs.vllm.ai/en/v0.19.1/configuration/optimization/)
- [Triton Layer Normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
