# 基模组 AI Infra 面试题库与模拟面试指南

> 基于本仓库 8x Tesla V100 PCIe 实验整理。版本日期：2026-07-17。
>
> 这不是任何公司的内部题库。题目按公开岗位能力要求、主流训练/推理框架原理，以及本仓库的实测结果组织，目标是训练可迁移的系统分析能力。
>
> Megatron 的 M001-M140 专项训练、源码阅读路径和上机实验另见 [Megatron 专家面试题库](megatron-interview-question-bank.md)。

## 1. 这份题库怎么用

### 1.1 对应的岗位方向

当前基模组的 AI Infra 岗位通常不是一个单一工种，可以拆成五类能力画像：

1. **训练系统与性能**：分布式并行、通信、显存、MFU、数据与 checkpoint 路径、稳定性。
2. **推理引擎与服务**：KV cache、调度、并行部署、量化、吞吐/时延/SLO、资源隔离。
3. **GPU 内核与编译器**：CUDA/Triton、算子融合、访存、occupancy、图编译、CUDA Graph。
4. **集群与可靠性**：多机网络、拓扑放置、故障检测、容错恢复、可观测性、容量与成本。
5. **后训练与 RL Infra**：训练/推理协同、异步 rollout、权重同步、数据新鲜度和集群调度。

这种划分和公开岗位要求一致。例如 OpenAI 的 Training Performance Engineer 明确覆盖计算、通信、存储、GPU kernel、collective、sharding、MFU 与 checkpoint；其 RL Training Infra 和 Platform Systems 岗位还强调训练/推理编排、故障处理与可观测性。Anthropic 的公开岗位列表也分别设置 GPU Performance、Inference Systems、Pretraining Scaling、ML Networking 与 RL Systems 等方向。参考文档见文末。

### 1.2 本项目已经能证明什么

| 能力 | 本机实验论据 | 面试中应如何表达 |
|---|---|---|
| 拓扑与通信 | PIX/PHB/SYS 配对，AllReduce/AllGather/ReduceScatter/SendRecv，8 卡 collective | 不只报带宽；解释链路、算法字节量、慢链路和并发竞争 |
| 实验设计 | warmup、重复、控制变量、correctness、原始日志、归一化结果、自动校验 | 区分观测、推断和未验证假设 |
| 数据并行 | DDP 强/弱扩展、实际 reducer bucket、gradient accumulation、CUDA 区间 overlap | 从计算/通信比、bucket 发起时机和 exposed communication 解释扩展效率 |
| 参数分片 | FSDP `SHARD_GRAD_OP`/`FULL_SHARD`，root/block wrap，prefetch/limiter trace | 从 unit 状态机解释显存、AllGather/ReduceScatter、overlap 和吞吐 |
| 模型并行 | Megatron TP/SP/PP、长序列、重计算；显式 `TP2 x PP2 x DP2` process group | 用切分维度、rank mapping 和 communicator 契约推导，不背结论 |
| MoE 数据路径 | variable-split All-to-All、热点 expert、dispatch/combine 回环校验 | 区分网络传输、peer wait、路由偏斜与 expert GEMM |
| 推理服务 | continuous batching、compile、TP/DP、prefix 驱逐、Poisson SLO、KV preemption | 分开准入、抢占、容量、吞吐、TTFT、TPOT、工作集和 goodput |
| Kernel | Triton RMSNorm、FP32 oracle、shape/warps sweep、cubin resources | 解释融合、launch floor、register、理论 occupancy 与硬件 counter 边界 |
| 可靠性 | 原子 manifest、checksum、RNG/data cursor、bitwise resume、rank/collective failure | 区分语义恢复、逐 bit 回放、故障检测和进程清理 |
| 兼容性诊断 | SM70 wheel、attention backend 回退、custom all-reduce 禁用 | 能沿硬件架构、二进制、kernel backend 和运行日志定位 |

### 1.3 当前证据不能证明什么

面试时主动说明边界比夸大项目更专业。本项目尚未实测：

- 多机 InfiniBand/RoCE、GPUDirect RDMA、交换网络拥塞和跨机容错；
- 真实框架 MoE expert parallel、top-k router、capacity/drop、grouped GEMM 与 expert load balancing；
- context parallel 和超长上下文 attention 通信；
- FSDP2/DTensor、异步 distributed checkpoint 和跨 world-size reshard；
- Triton backward、autotune 和 Nsight Compute 级硬件 counter；
- 真实 70B/百亿级训练的端到端 MFU、数据管道和长时间稳定性；
- 大规模 RL rollout、权重同步与训练/推理解耦。

这些内容仍然会出现在后半部分，因为它们决定候选人能否从“做过单机实验”上升到“能设计生产系统”。回答时要明确哪些是实测，哪些是根据机制推导。

### 1.4 统一评分标准

每题按 0-4 分评价：

| 分数 | 表现 |
|---:|---|
| 0 | 不知道概念，或核心方向错误 |
| 1 | 能给定义，但不能解释机制或边界 |
| 2 | 能画数据流、给关键公式，结论基本正确 |
| 3 | 能讨论 trade-off、瓶颈和应观测指标，并连接实测经验 |
| 4 | 能处理反例、故障和规模变化，提出可验证实验或生产设计 |

推荐每个回答使用同一骨架：

1. 先澄清硬件、模型、精度、batch、序列长度和 SLO。
2. 画出 tensor/请求的数据流，写出显存、FLOPs 或通信字节量。
3. 找关键路径，判断 compute-bound、memory-bound、communication-bound 还是 queueing-bound。
4. 给出指标、profiler 事件、对照组和预期现象。
5. 最后讲例外、失败模式，以及本项目证据是否足够。

### 1.5 推荐模拟面试轮次

| 轮次 | 时长 | 题目范围 | 通过标准 |
|---|---:|---|---|
| 项目深挖 | 60 分钟 | Q75-Q84、Q91-Q115 中抽 8-10 题 | 能从原始证据解释实现、计量、失败和局限 |
| 运行时底层 | 60 分钟 | Q116-Q145 中抽 8-10 题 | 能落到 communicator、stream、分片/调度状态机和 profiler 证据 |
| 分布式训练 | 60 分钟 | Q13-Q40 | 能从切分方式推导通信和显存 |
| 推理系统 | 60 分钟 | Q41-Q54、Q87 | 能建立排队、KV 和调度模型 |
| 性能/内核 | 60 分钟 | Q01-Q12、Q55-Q64 | 会用 profiler/roofline 缩小问题 |
| 系统设计 | 60 分钟 | Q65-Q74、Q85-Q90 | 先定约束，再做容量、可靠性和观测设计 |
| 编码实践 | 60-90 分钟 | C01-C08 | 正确性、复杂度、测试和可观测性完整 |

---

## 2. Level 1：GPU 与性能基础

### Q01. 一个 GPU workload 为什么可能算力利用率很低？

**高分回答要点**

- 先区分 SM active、Tensor Core 利用率和端到端 GPU utilization；`nvidia-smi` 的 busy 百分比不等于有效 FLOPs。
- 可能受 HBM/显存带宽、L2/共享内存、PCIe/NVLink、kernel launch、CPU dispatch、同步、依赖链、低 occupancy 或小矩阵限制。
- 用 roofline 的 arithmetic intensity 判断理论上偏 compute-bound 还是 bandwidth-bound，再用 profiler 验证，而不是只看一个 utilization 数字。
- 对小 batch decode，矩阵形状小、逐 token 串行和权重反复读取通常比峰值 FLOPs 更关键；大 batch prefill 更容易把 GEMM 做大。

**追问**：V100 的 Tensor Core 可用，为什么某个 FP16 模型仍可能跑不到 Tensor Core 峰值？应检查 dtype、矩阵维度、kernel 选择、对齐、融合和实际 Tensor Core 指令计数。

**实验锚点**：本项目中 TP 把单卡矩阵切小后吞吐下降，是“峰值算力相加不等于有效算力相加”的直接例子。

### Q02. 如何估算一个模型训练时的显存？

**高分回答要点**

- 分开计算参数、梯度、optimizer state、master weight、activation、通信 bucket、临时 workspace 和 allocator 碎片。
- 对 AdamW，不应死记“每参数 16 bytes”；应按实现逐项算。例如 FP16 参数 2B、FP16 梯度 2B、FP32 master weight 4B、两个 FP32 moment 8B，合计常见为 16B/参数，但梯度 dtype、master weight 和 fused optimizer 会改变结果。
- activation 与 microbatch、sequence length、hidden size、layer 数和保存策略相关；attention 中某些中间量随序列平方增长，FlashAttention 会改变保存与 IO 方式。
- FSDP/ZeRO 只能分片特定状态；AllGather 临时参数、prefetch、未被 wrap 的模块和通信 buffer 决定峰值，不是简单除以 world size。

**追问**：为什么 `FULL_SHARD` 实测峰值不是 DDP 的八分之一？

**实验锚点**：约 368M 参数模型中，DDP 约 6.49 GiB，`SHARD_GRAD_OP` 约 1.55 GiB，`FULL_SHARD` 约 1.06 GiB；这个结果适合用组件法反推，而不是套比例。

### Q03. 什么是 roofline model？它如何指导优化？

**高分回答要点**

- 横轴是 arithmetic intensity，即 FLOPs/搬运字节；纵轴是达到的 FLOP/s。上界为 `min(peak_compute, bandwidth * intensity)`。
- 位于斜线附近说明受带宽限制，优先减少 HBM 流量、融合 kernel、提高数据复用；位于水平屋顶附近才值得优化指令吞吐、Tensor Core 和并行度。
- roofline 是上界模型，不包含 launch latency、依赖、通信和调度，因此小 kernel 或分布式 workload 还需时间线分析。
- 必须说明字节统计在哪一级存储：HBM roofline、L2 roofline 和 shared-memory roofline结论可能不同。

**追问**：为什么 FlashAttention 不改变 attention 数学结果，却能显著加速？核心是 tiling 和 online softmax 减少 HBM 中间张量读写，而不是减少所有计算。

### Q04. 如何读一张 8 GPU 拓扑图？

**高分回答要点**

- 识别 GPU 到 PCIe switch、CPU socket/NUMA、NIC 的路径，区分 PIX、PXB、PHB、SYS，以及是否存在 NVLink/NVSwitch。
- `SYS` 往往经过 CPU 互连，时延更高、共享链路更多；但实际 collective 性能还取决于 NCCL 拓扑建图、消息尺寸和并发流量。
- 进程绑核、内存 NUMA 位置、NIC affinity 会改变 host staging、控制面和跨机路径。
- placement 应把高频通信组映射到快链路，例如同一 TP group 尽量局部，DP 可以承受更慢但带宽稳定的跨节点链路。

**实验锚点**：本机 512 MiB AllReduce 中 PIX 约 11.74 GB/s，PHB 约 7.07 GB/s，SYS 约 6.93 GB/s，是拓扑影响而非 GPU 型号差异。

### Q05. AllReduce、ReduceScatter、AllGather、All-to-All 和 Send/Recv 分别做什么？

**高分回答要点**

- AllReduce：每个 rank 获得所有输入按操作归约后的完整结果，典型用于 DDP 梯度。
- ReduceScatter：先归约再把结果分片到各 rank；AllGather：把各 rank 分片拼成完整 tensor。两者组合可实现 AllReduce，也对应 FSDP 梯度分片和参数聚合。
- All-to-All：每个 rank 向每个 rank 发送不同分片，典型用于 MoE token dispatch/combine，流量和热点对负载不均衡敏感。
- Send/Recv 是点对点通信，pipeline stage activation/gradient 传递常用它。
- 要进一步说明 tensor shape、group 大小、每 rank 输入/输出布局和是否原地，避免只背名字。

**追问**：为什么不能用“项目里测了 AllReduce”代表掌握所有通信？不同 primitive 的消息模式、并发连接数和慢链路敏感性不同。

### Q06. Ring AllReduce 的通信量和时间模型是什么？

**高分回答要点**

- 对 `N` 个 rank、每 rank tensor 大小 `S`，ring 通常有 `N-1` 步 ReduceScatter 和 `N-1` 步 AllGather；每步传 `S/N`。
- 每 rank 总发送/接收量约 `2(N-1)S/N`。简化时间模型为 `2(N-1)alpha + 2(N-1)S/(N beta)`，其中 `alpha` 是每步延迟，`beta` 是有效链路带宽。
- 大消息时 bandwidth 项主导；小消息时 2(N-1) 次启动/同步使 latency 主导，tree 往往更有优势。
- 实际 NCCL 会按拓扑使用多 channel、协议和混合算法，不能把单环模型当完整实现。

**追问**：为什么 8 卡 ring 的性能可能由最慢路径或共享上行链路限制，而不是两卡最快带宽？

### Q07. `algbw` 和 `busbw` 有什么区别？

**高分回答要点**

- `algbw` 通常是逻辑 payload 大小除以操作时间，便于从应用视角比较。
- `busbw` 按 collective 算法实际搬运量做归一化，便于比较底层互连利用。Ring AllReduce 常用修正系数 `2(N-1)/N`；AllGather/ReduceScatter 常用 `(N-1)/N`。
- 两个值都不是任意一条物理链路的精确瞬时速率；多 channel、双向流量和拓扑共享会使解释更复杂。
- 面试中必须确认工具定义和单位，不能把 GB/s、GiB/s、单向和双向带宽混用。

**实验锚点**：报告同时保留 nccl-tests 原始输出和归一化表，可以现场解释为什么比较拓扑时优先保持消息大小、rank 数和指标定义一致。

### Q08. rank、local rank、world size 和 process group 是什么？

**高分回答要点**

- global rank 唯一标识整个 job 的进程；local rank 标识节点内设备位置；world size 是默认 group 的进程数。
- process group 定义 collective 的参与者与顺序。同一进程可属于 DP、TP、PP、CP、EP 等多个正交 group。
- group 构造错误会导致 silent wrong result 或 hang；所有 rank 必须以一致顺序进入匹配 collective。
- `CUDA_VISIBLE_DEVICES` 后的 local rank 是逻辑设备序号，不一定等于物理 PCI bus id，做拓扑放置时要显式核对。

**追问**：给定 `TP=4, PP=2, DP=8`，总 world size 是多少？如何为一个 rank 推导其三个坐标？

### Q09. FP16、BF16、TF32 和 FP32 在训练中如何选择？

**高分回答要点**

- FP16 exponent 范围小，通常需要 loss scaling；BF16 与 FP32 exponent 范围相同，训练更稳但尾数更短；TF32 是 NVIDIA 对 FP32 matmul 的加速路径，不是通用存储 dtype。
- 参数存储、计算、梯度归约、optimizer state 可以使用不同 dtype。所谓“FP16 训练”必须拆解具体路径。
- V100 支持 FP16 Tensor Core，但不原生支持 BF16 Tensor Core；这会直接影响可复现配置和最新框架兼容性。
- 通信压缩能减少字节，但可能引入 cast、精度损失或额外 kernel，必须把算法公平性和性能收益分开报告。

**实验锚点**：FSDP 对比中 FP16 通信提高吞吐，但它不是与 FP32 通信完全相同的数值配置；报告明确把它作为独立变量。

### Q10. 怎样设计一个可信的 GPU benchmark？

**高分回答要点**

- 固定软件版本、GPU 时钟/功耗策略、模型配置、精度、seed、输入分布、并行配置和环境变量。
- 区分冷启动、warmup 和 steady state；异步 CUDA 必须正确同步，服务 benchmark 还要排除模型加载与客户端瓶颈。
- 至少报告中位数和尾部/离散程度，保留原始样本；吞吐测试要验证输出正确，不能只求快。
- 一次只改变一个主变量，同时记录可能的混杂项，如 compile、cache、通信 dtype、batch token 数和 OOM 回退。
- 写明观测事实、机制推断和未验证假设；通过脚本、原始日志、解析器和 artifact 校验保证可复现。

**追问**：为什么单次最高值不可信？为什么服务端“发送完成”不等于客户端“请求成功”？

### Q11. 训练性能应该报告哪些指标？

**高分回答要点**

- 核心是 global tokens/s 或 samples/s、step time 分布、每 GPU 吞吐、扩展效率、峰值/保留显存、MFU/HFU。
- 分解 data loading、forward、backward、optimizer、collective、checkpoint 时间，并看计算通信 overlap。
- 稳定性指标包括 loss、梯度 norm、overflow/skip step、straggler、hang、重试和有效训练 token。
- MFU 分母必须说明硬件峰值和精度，分子 FLOPs 模型必须说明是否计入 attention、embedding、recompute 和稀疏专家。
- 对长任务还要报告 uptime、故障间隔、恢复时间和 checkpoint 开销；短 microbenchmark 不能代表这些指标。

### Q12. 推理服务应该报告哪些指标？

**高分回答要点**

- 区分 TTFT、TPOT/ITL、端到端 latency、output tokens/s、request throughput、并发和队列等待。
- 报告 p50/p90/p95/p99，而不是只有平均值；按输入/输出长度分桶，否则 workload mix 会掩盖结果。
- 把原始吞吐与 SLO goodput 分开：只有满足所有目标的请求才计入 goodput。
- 记录 KV cache 使用率/命中率、preemption、batch token、prefix reuse、GPU/CPU 利用率和错误率。
- 明确 open-loop arrival rate、closed-loop concurrency 或 trace replay；三者回答的问题不同。

**实验锚点**：本项目在 25 QPS 时 completed throughput 仍约 16.46 req/s，但满足 TTFT/TPOT SLO 的 goodput 只有约 0.40 req/s，说明“服务器很忙”不等于“服务可用”。

---

## 3. Level 2：DDP、FSDP 与训练运行时

### Q13. DDP 从 forward 到参数更新经历了什么？

**高分回答要点**

- 初始化时各 rank 拥有完整模型，参数应保持一致；每 rank 处理不同数据分片。
- autograd 生成梯度后，DDP hook 在 bucket ready 时启动 AllReduce；归约后每个 rank 获得等价的平均梯度。
- bucket 允许后层梯度生成时就通信，与更早层 backward 计算重叠；optimizer 在所有所需梯度就绪后本地更新。
- unused parameter、动态图、梯度 view、bucket rebuild 和不同 rank 控制流会影响正确性或性能。

**追问**：为什么只在 step 末尾手工对整个 gradient buffer 做一次 AllReduce，通常 overlap 更差？

### Q14. 强扩展和弱扩展怎样定义？

**高分回答要点**

- 强扩展固定全局问题规模，增加 GPU；理想 step time 约除以 `N`。效率可写为 `throughput_N/(N * throughput_1)`。
- 弱扩展固定每 GPU 工作量，全局规模随 GPU 数增加；理想每 GPU 吞吐不变，step time 近似不变。
- 对训练必须明确固定的是 samples、tokens 还是 sequence；global batch 改变还会影响优化超参数和收敛，系统 scaling 与 time-to-quality 不是一回事。
- 小模型/小 microbatch 下固定通信和 launch 成本占比高，强扩展很快失效。

**实验锚点**：约 110M 模型的 8 卡 DDP 强扩展仅 2.82x、效率 35.3%，适合解释为什么单机 GPU 数增加不保证线性速度。

### Q15. microbatch 为什么影响 DDP 扩展效率？

**高分回答要点**

- 对固定模型，每个 optimizer step 的梯度 payload 基本与参数量相关，而不是与 microbatch 成正比；计算量则随 token 数增长。
- 因而增大 microbatch 提高 compute/communication ratio，也通常产生更大的 GEMM、减少 launch 占比。
- 但 activation 显存随 microbatch 增长，最终受到 OOM、数据分布和全局 batch/收敛约束。
- 梯度累积可以增加每次同步前的本地计算；使用 `no_sync` 避免每个 microstep 都 AllReduce。

**实验锚点**：本项目 microbatch 1→8 时，每 token 梯度 payload 约从 838.8 KiB 降到 104.8 KiB，8 卡弱扩展效率从约 31.9% 提升到 51.2%。

### Q16. DDP bucket size 应该怎样调？

**高分回答要点**

- 小 bucket 更早 ready、潜在 overlap 更好，但 collective 数量多、latency 与 launch overhead 高。
- 大 bucket 带宽效率高，却要等更多梯度生成，尾部 exposed communication 可能变长。
- 最优值取决于 layer backward 时间、网络 latency/bandwidth、参数分布和是否有大 embedding；不能给通用常数。
- 用 timeline 看每个 bucket ready、NCCL 起止、compute stream 空洞和最后一个 collective，而不是只看总 NCCL kernel 时长。

**追问**：如果总 AllReduce 时间不变，但 step time 下降，发生了什么？如果总时间下降而 step 不变，又可能是什么？

### Q17. 如何证明计算和通信发生了 overlap？

**高分回答要点**

- 在 Nsight Systems/PyTorch Profiler 时间线上同时观察 compute stream 与 NCCL stream，检查真实时间区间重叠。
- 比较 step wall time 与 critical path；不能把所有 kernel duration 直接相加，因为不同 stream 可能并发。
- 做因果对照：禁用异步/改变 bucket、插入同步或改变 microbatch，观察 exposed communication 是否变化。
- 还要检查 GPU 间 skew：一个 rank 较慢会让其他 rank 的 collective kernel 表现为等待。

**实验锚点**：本项目 m1 与 m8 均看到约 13 个 FP32 Ring AllReduce kernel，累计约 104-106 ms；仅凭这个累计数不能断言它们全部暴露在 step 关键路径上。

### Q18. 为什么 profile 中 NCCL kernel 很长，不一定说明网络带宽差？

**高分回答要点**

- GPU collective kernel 生命周期可能包含等待 peer、等待数据 ready、同步和传输，不等于纯链路搬运时间。
- 某个 rank 的前序 compute、CPU launch 或数据加载慢，会让其他 rank 在 collective 内等待，形成 straggler 放大。
- 多 stream 并发时累计 kernel duration 可大于 wall time；工具时钟、采样和 trace 范围也要核对。
- 应结合 nccl-tests、每 rank timeline、消息大小、算法/协议、拓扑和系统 counters 定位。

### Q19. 梯度累积与增大全局 batch 有什么系统和算法区别？

**高分回答要点**

- 系统上，累积 `K` 个 microstep、只同步一次，可摊薄通信和 optimizer 成本，但总 forward/backward 计算不减少。
- 显存主要由单个 microbatch activation 决定，梯度 buffer 需要保留；可用它突破单步 microbatch 显存限制。
- 算法上 global batch 变大可能需要学习率、warmup、数据顺序调整，并可能改变泛化和达到目标 loss 的 token 数。
- BatchNorm、dropout RNG、loss normalization、gradient clipping 和 mixed-precision overflow 都可能让“数学等价”失效。

### Q20. ZeRO-1/2/3 与 FSDP 的核心关系是什么？

**高分回答要点**

- ZeRO-1 分片 optimizer state，ZeRO-2 再分片 gradient，ZeRO-3 再分片 parameter。
- PyTorch FSDP `FULL_SHARD` 在概念上接近 ZeRO-3：计算前 AllGather 所需参数，backward 后 ReduceScatter 梯度，并让参数/梯度/optimizer state 分片驻留。
- `SHARD_GRAD_OP` 更接近只分片 gradient 与 optimizer state 的折中，通常保留更多参数驻留以减少重新聚合。
- 真正实现差异还包括 flatten/per-parameter 表示、prefetch、通信调度、offload、checkpoint 和混合精度，不能只按 ZeRO stage 贴标签。

### Q21. `SHARD_GRAD_OP` 为什么可能比 `FULL_SHARD` 更快却更占显存？

**高分回答要点**

- 它减少参数反复 reshard/AllGather 的频率或驻留转换，在 backward 前保留 unsharded 参数，通信和同步更少。
- 代价是完整参数保持更久，峰值显存高于 `FULL_SHARD`。
- `FULL_SHARD` 容量更强，但参数 AllGather、ReduceScatter、prefetch buffer 和 allocator 峰值都可能进入关键路径。
- 选择应由“模型能否放下、目标 batch、网络带宽和吞吐”共同决定，不是 stage 越高越好。

**实验锚点**：本机 `SHARD_GRAD_OP` 约 10,009 tok/s、1.55 GiB，`FULL_SHARD` 约 7,704 tok/s、1.06 GiB，正好展示容量/吞吐 Pareto frontier。

### Q22. 如何公平比较 DDP、FSDP 和通信压缩？

**高分回答要点**

- 固定模型、初始化、数据、global tokens/step、优化器语义、计算精度、warmup 和测量窗口。
- 把参数/activation 精度与 collective dtype 分开；FP16 通信不是纯粹“换并行策略”。
- 同时报吞吐、峰值显存、loss/梯度误差和通信字节；容量配置应再比较可达到的最大 batch 或模型规模。
- 校验每 rank 参数更新一致性和 loss 轨迹，避免因错误少算通信获得虚假加速。

**实验锚点**：报告把 DDP FP32 通信、DDP FP16 通信和两种 FSDP 配置分列，面试时应主动解释这一公平性边界。

### Q23. FSDP wrap policy 为什么会影响性能和峰值显存？

**高分回答要点**

- wrap unit 决定参数 AllGather/ReduceScatter 的粒度，以及可否与相邻层计算 prefetch/overlap。
- 粒度过大，临时完整参数峰值大且通信启动晚；粒度过小，collective 和元数据/launch 开销过多。
- shared parameter、embedding、root module 未 wrap、模块执行顺序与动态控制流会破坏简单估算。
- 应画出两三个 layer 的 forward/backward 时间线，检查当前、prefetch 和待释放的参数同时驻留数量。

### Q24. FSDP1 与 FSDP2/`fully_shard` 有哪些值得面试讨论的变化？

**高分回答要点**

- FSDP2 使用基于 DTensor 的 per-parameter sharding 表示，而不是主要依赖 flat parameter；更容易与 tensor parallel 等组合并检查 placement。
- 参数在计算前 unshard、计算后 reshard，仍需考虑 AllGather、ReduceScatter、prefetch 和 mixed precision，基本物理成本没有消失。
- API 与 state dict/checkpoint 行为不同，迁移不能只替换类名；要验证 optimizer state、shared weights、compile 和 checkpoint。
- 本项目只实测 FSDP1，因此对 FSDP2 应给官方机制和计划实验，不能宣称有性能结论。

### Q25. distributed checkpoint 需要解决哪些问题？

**高分回答要点**

- 保存模型、optimizer、scheduler、RNG、data loader/样本游标、loss scaler 和训练元数据，保证恢复后的语义连续。
- 每 rank 分片并行写入，使用 manifest/完成标记实现原子提交；故障时不能把半写 checkpoint 当可恢复版本。
- 支持不同 DP/TP/PP 配置恢复需要全局 tensor 元数据和 reshard，而不是把 rank 文件机械重命名。
- 控制 checkpoint pause、异步 staging、存储带宽、保留策略和恢复时间目标；定期做真实 restore 演练。

**追问**：一个 rank 写失败、其余 rank 成功时怎么办？怎样避免重启后重复或跳过训练样本？

### Q26. 训练中出现 loss spike/NaN，你如何定位是数值问题还是分布式错误？

**高分回答要点**

- 先确定首次异常 step/rank/layer，保存输入、loss scale、梯度 norm、激活统计和 optimizer state；不要只在最终 NaN 后排查。
- 单卡 FP32 或更稳定精度复现，关闭 fused kernel、compile、通信压缩和并行维度，做逐步二分。
- 检查不同 rank 的参数 checksum、梯度归约、数据重复/坏样本、overflow skip 是否一致，以及 collective 顺序。
- 对照最近代码、驱动、kernel、拓扑和数据变更；建立小规模 deterministic replay，但承认某些异步算法无法 bitwise 重现。

**评分上限提示**：只回答“调低学习率/做 gradient clipping”最多 1 分，因为它没有定位根因。

---

## 4. Level 3：Tensor、Sequence、Context、Pipeline 与 Expert Parallel

### Q27. 以两层 MLP 为例，Megatron tensor parallel 如何切分？

**高分回答要点**

- 设输入 `X:[B,S,H]`，第一层权重 `W1:[H,4H]`。Column Parallel 把 `W1` 的输出维切到 `T` 个 rank，每 rank 得到 `[B,S,4H/T]`，通常不需先合并即可做局部 activation。
- 第二层 `W2:[4H,H]` 用 Row Parallel 按输入维切，每 rank 对局部分片做 matmul，产生 `[B,S,H]` 的 partial sum，再通过 AllReduce 得到完整输出。
- attention 中 QKV projection 和 output projection 使用相似的列切/行切；head 数和 hidden size 必须满足切分约束。
- 要说明 bias、dropout、LayerNorm 和 residual 在哪个 rank 上执行，以及是否引入 sequence parallel，否则数据流不完整。

**追问**：每层 TP 的通信字节与参数量还是 activation shape 更相关？为什么 sequence length 增大会加重 TP 通信？

### Q28. 为什么 TP 在 PCIe V100 上更像容量手段，而不是加速手段？

**高分回答要点**

- TP 降低每 rank 参数和局部 GEMM 大小，但每层引入 activation collective；通信位于层级关键路径，难以像 DDP 梯度那样跨多层充分 overlap。
- PCIe 带宽低于 NVLink/NVSwitch，且本机跨 PHB/SYS 路径更慢；切小后的 GEMM 也可能降低 Tensor Core 效率。
- 若单卡本就能容纳模型，TP 节省的计算时间可能小于 collective、同步和 kernel launch 增量。
- 评价 TP 需同时看单请求延迟、总 tokens/s、每 GPU 效率和可容纳模型/KV 容量，不能只看 aggregate GPU 数。

**实验锚点**：TP1/DP8 约 84,626 tok/s，而 TP8/DP1 约 14,667 tok/s；TP8 单 rank 显存降到约 0.47 GiB，但吞吐显著下降。

### Q29. Megatron sequence parallel 做了什么？

**高分回答要点**

- SP 通常与 TP 绑定，把原本在 TP rank 间复制的 sequence 维 activation 分片，让 LayerNorm、dropout 等逐 token 操作只处理本地序列片段。
- TP 中某些 AllReduce 可重写为 ReduceScatter 与 AllGather，使通信总量近似不变，但中间 activation 不必一直在每个 TP rank 完整驻留。
- 它不是把整个 attention 的 context 维彻底切开；attention 仍需满足 Q/K/V 依赖，不能把 SP 等同于超长上下文 context parallel。
- 收益取决于可被 SP 分片的 activation 在总峰值中的占比，以及通信实现和 allocator 峰值。

**追问**：为什么 TP+EP 的一些现代配置要求启用 SP？候选人应从 tensor layout 和避免 TP 区域重复 token 的角度解释，再核对具体框架约束。

### Q30. 序列长度到 4096，为什么 SP 的显存仍可能只节省几个百分点？

**高分回答要点**

- “长序列”不自动意味着峰值由 SP 可切分 activation 主导；参数、optimizer、attention workspace、未切分张量、通信 buffer 和 caching allocator 都可能占主导。
- 要看测量点是 forward peak、backward peak 还是进程保留显存，以及 FlashAttention/重计算是否已经减少中间 activation。
- TP degree、microbatch、层数和 hidden size 会改变组成；若模型较小或 microbatch 很小，固定开销占比更高。
- 正确后续实验是做显存快照/按 allocation 分类，并扫描 sequence、microbatch、TP，而不是只把 sequence 再加长然后猜测。

**实验锚点**：本项目 seq=4096 时 SP 只省约 1.3%-3.8%，这个负结果反而能证明候选人会修正“长序列必然让 SP 有效”的假设。

### Q31. sequence parallel 和 context parallel 的区别是什么？

**高分回答要点**

- SP 主要分片 TP 区域中可按 token 独立计算的 activation，降低 LayerNorm/dropout 等复制；一般不独立解决完整 attention context 的内存扩展。
- CP 把整个 sequence/context 分到多个 rank，并设计 K/V 交换或 attention 归约，使每个 rank 能得到其 query 对全上下文的正确 attention。
- CP 会影响 attention 通信、causal mask、position encoding、load balance 和 KV layout；实现可能采用 all-gather、ring P2P 或分层算法。
- 选择 CP 要比较 activation/KV 节省与随 context 增长的通信，且应把 CP group 映射到高带宽链路。

**追问**：对 causal attention，为什么不同 context shard 的计算量可能不均？如何用 zig-zag/ring 调度改善？

### Q32. activation recomputation 为什么省显存但增加计算？

**高分回答要点**

- 正常 backward 需要保存 forward activation；checkpointing 只保留边界，backward 前重做部分 forward 来恢复中间量。
- 粗粒度 full-layer recompute 节省更大但重复 FLOPs 多；selective recompute 只重做便宜或高存储占比的算子，能形成更好 Pareto 点。
- 峰值节省取决于参数状态、通信 buffer 和不可重算张量占比，不是 activation 理论大小的简单相减。
- 还需处理 dropout RNG、随机算子、autocast 和副作用，保证重算语义一致。

**实验锚点**：seq=4096 的 full recompute 节省约 77.6%-78.1% 显存，同时吞吐下降约 27%-30%，比本机 SP 的节省明显得多。

### Q33. 长上下文训练的主要复杂度和优化手段有哪些？

**高分回答要点**

- 标准 attention score 计算复杂度约 `O(B*S^2*H)`，QKV/MLP 约 `O(B*S*H^2)`；哪个占主导取决于 `S/H`、架构和 kernel。
- 朴素 attention 中间矩阵显存随 `S^2`，FlashAttention 通过 IO-aware tiling 避免完整 materialization，但数学计算仍近似平方。
- 系统手段包括 FlashAttention、activation recompute、CP、SP、sequence packing、变长 attention 和合适的 microbatch。
- 算法手段还包括 sparse/sliding-window/linear attention，但这会改变模型，不应与系统无损优化混为一谈。

### Q34. GPipe、1F1B 和 interleaved pipeline 的差异是什么？

**高分回答要点**

- GPipe 先完成所有 microbatch forward，再全部 backward，调度简单但需要保存更多 activation，flush bubble 明显。
- 1F1B 在 warmup 后交替执行一个 forward 和一个 backward，降低 activation 驻留并保持相似 flush bubble 量级。
- interleaved 让每个物理 rank 持有多个 virtual stage，缩短逻辑 stage 间隔，可降低 bubble，但增加通信次数和调度复杂度。
- pipeline 还要考虑 forward activation 与 backward gradient 的 P2P、权重版本、loss 所在 stage 和 tied embedding。

### Q35. pipeline bubble 怎样估算？

**高分回答要点**

- 对均衡、无 interleave、同步 flush schedule，`p` 个 stage、`m` 个 microbatch 的理想利用率近似 `m/(m+p-1)`，bubble fraction 近似 `(p-1)/(m+p-1)`。
- `p=8,m=1` 时 bubble 约 87.5%；`m=16` 时理想 bubble 约 30.4%。这只是均衡 stage、忽略通信和其他开销的上界模型。
- 增大 `m` 可摊薄 bubble，但可能改变 microbatch 大小、global batch、调度 overhead 和 activation 内存。
- 实际吞吐还受最慢 stage、P2P、数据 stage、embedding/loss 和同步影响。

**实验锚点**：本项目 PP8/M1 的理论 bubble 为 87.5%；M16 相对 M1 吞吐约 2.59x，但扩展效率仍仅约 32.4%。

### Q36. 如何发现并解决 pipeline stage imbalance？

**高分回答要点**

- 分 rank 测 stage forward/backward wall time、kernel time、参数量、activation bytes 和 P2P wait；吞吐由最慢 stage 限制。
- 参数量只是代理，attention/MLP FLOPs、embedding、vocab projection、MoE token 数和通信位置都会造成不均衡。
- 可用非均匀 layer partition、virtual stage/interleave、把 embedding/loss 单独计价、调整 MoE expert placement 或复制轻量 stage。
- 重分区后要验证显存峰值和通信；把计算均衡但让跨慢链路流量增大，可能得不偿失。

**实验锚点**：本项目各 PP stage 参数量最大/最小约 2.76x，能解释为什么实际结果低于只看 bubble 公式的理想值。

### Q37. pipeline 的 microbatch 数越多越好吗？

**高分回答要点**

- 更多 microbatch 降低 flush bubble，但单个 microbatch 可能变小，使 GEMM 效率下降、launch 数上升。
- 若保持单 microbatch 不变，global batch 会增大，影响优化语义和数据并行组合；若保持 global batch 不变，局部计算形状改变。
- activation memory 取决于 schedule 中同时 in-flight 的 microbatch 数，1F1B 与 GPipe 不同。
- 最优点需扫描 throughput、峰值显存、bubble、P2P 和 time-to-quality，而不是只最大化 `m`。

### Q38. 怎样构造 DP/TP/PP/CP 的 rank mapping？

**高分回答要点**

- 先满足 `world_size = DP * TP * PP * CP`；有 EP 时要明确 EP 与 DP/TP 的 group 关系，不能盲目全部相乘。
- 给 rank 定义多维坐标，再按框架规定的维度顺序建立 process groups；用小 tensor collective 做成员校验。
- placement 原则是通信频繁、位于关键路径的 TP/CP 优先放 NVLink/NVSwitch 域内；PP 相邻 stage 应有稳定带宽；DP 可跨节点但要考虑梯度流量和 NIC。
- 同时配置 GPU-NIC affinity、NUMA 和容错域，避免一个节点故障摧毁过多复制或完整 pipeline。

**追问**：给 64 GPU、每节点 8 GPU，设计 `TP=4, PP=4` 的 group，并说明 DP group 为什么跨哪些节点。

### Q39. MoE expert parallel 的通信和负载问题是什么？

**高分回答要点**

- router 为 token 选择 top-k expert；若 expert 分布在不同 rank，需要 All-to-All dispatch token，再在 expert MLP 后 All-to-All combine。
- 平均 FLOPs 稀疏，但通信、router、排序/permute、padding、grouped GEMM 和最慢 expert 决定实际性能。
- token 分布不均会造成 rank straggler 或超过 capacity；辅助负载均衡 loss、capacity factor、token drop、expert replication 和动态重放置各有质量/系统代价。
- 指标应包括每 expert token 数分布、max/mean、drop rate、All-to-All bytes/time、GEMM shape、overlap 和端到端 MFU。

**追问**：为什么单纯提高 capacity factor 可能降低 drop 却恶化吞吐和显存？

### Q40. 给定模型和集群，如何选择并行组合？

**高分回答要点**

- 先用参数/optimizer/activation/KV 预算判断最低必要分片：优先使用吞吐损失较小的 DP/FSDP；单模型放不下才逐步引入 TP/PP/CP/EP。
- TP/CP 放高带宽域，PP 用于跨节点扩展层数，DP 利用剩余副本；长 context 可能迫使 CP，MoE 架构迫使 EP。
- 枚举合法组合，建立每层 FLOPs、collective/P2P 字节、bubble、峰值显存和 topology-aware bandwidth 模型，再用小规模 sweep 校准。
- 最终目标不是单步最快，而是满足显存、稳定性和目标 global batch 下的 tokens/s、MFU、成本及恢复要求。

**评分上限提示**：只说“8 卡就 TP8”或“越多并行越快”是 0-1 分答案。

---

## 5. Level 4：LLM 推理引擎与在线服务

### Q41. prefill 和 decode 的瓶颈为什么不同？

**高分回答要点**

- Prefill 并行处理输入序列，GEMM 较大、算术强度高，通常更容易 compute-bound；同时创建每层 K/V。
- Decode 每步只为每个请求生成少量 token，但要读取权重和已有 KV，矩阵较小且逐 token 有依赖，常偏 memory-bandwidth/launch-bound。
- batching 可以复用权重读取、增大 GEMM，但 decode batch 受请求到达、长度差异和 SLO 限制。
- 不能把两个阶段用一个平均 tokens/s 概括；应分别看 TTFT、TPOT、prefill tokens/s 和 decode batch efficiency。

### Q42. 如何估算 KV cache 显存？

**高分回答要点**

- 对每请求长度 `S`、层数 `L`、KV head 数 `H_kv`、head dim `D`、每元素 `b` bytes，KV 约为 `2 * L * S * H_kv * D * b`；2 来自 K 和 V。
- MHA 中 `H_kv` 等于 query head 数；GQA/MQA 显著减少 KV，但不同比例取决于架构。
- 多请求按实际 token 总数求和，再加 block metadata、对齐、碎片、prefix 引用和临时 workspace。
- TP 后每 rank KV 是否按 head 分片取决于实现；容量不能简单按 GPU 数倍增，需检查复制项与 block manager。

**追问**：给出一个 70B 模型配置和 128k context，现场计算单请求 KV；候选人应先向面试官索要 `L/H_kv/D/dtype`。

### Q43. PagedAttention 解决了什么问题？

**高分回答要点**

- 把逻辑连续的 KV 序列映射到固定大小物理 block，避免为最大长度预留连续大块显存，降低外部碎片并支持动态增长。
- block table 让 attention kernel 按映射读取 KV；copy-on-write/reference counting 可支持共享 prefix。
- block 太大产生内部碎片，太小增加 metadata、寻址和 kernel overhead；block size 是容量与效率权衡。
- 它改善内存管理和可批处理性，但不会消除 KV 本身的线性容量，也不会自动解决调度和尾延迟。

### Q44. continuous batching 与静态 batching 有什么区别？

**高分回答要点**

- 静态 batch 等整批请求完成，短请求会等待长请求，设备可能在 batch 边界空闲。
- continuous batching 在迭代/token 边界加入新请求、移除完成请求，保持活跃 token batch，提高利用率并减少 head-of-line blocking。
- scheduler 要在 prefill 与 decode、最大 batched tokens、序列数、KV 容量和优先级之间决策。
- 高吞吐策略可能延迟新请求 prefill 或已有请求 decode，因此必须用 TTFT/TPOT SLO 约束，不只是最大化 tokens/s。

**实验锚点**：本项目 eager 模式并发 1→32 时 output throughput 约 34.8→686.6 tok/s，体现 batching 对权重复用和 GPU 占用的价值。

### Q45. 推理 scheduler 在 KV 不足时有哪些选择？

**高分回答要点**

- admission control：在进入 GPU 前拒绝、排队或降级，避免过载后所有请求一起违约。
- preemption：暂停低优先级序列；可丢弃并重算 KV，或 swap 到 CPU/远端，两者分别付计算或传输代价。
- chunked prefill：把长 prompt 分块，限制单轮 token budget，避免长 prefill 长时间阻塞 decode。
- 公平性策略可按请求、token、租户或 deadline；必须防止长请求饥饿和单租户占满 cache。
- 观测 queue time、preemption/recompute、KV occupancy、eviction、每轮 prefill/decode token 和 SLO miss 原因。

### Q46. chunked prefill 为什么可能改善尾延迟，也可能降低效率？

**高分回答要点**

- 把长 prefill 分成多个调度 quantum，可在块之间插入 decode，降低已有请求的 TPOT 抖动和新请求 head-of-line blocking。
- 块过小会增加调度、kernel launch、状态管理并让 GEMM 变小；块过大又失去抢占效果。
- 某些实现可把 compute-heavy prefill 与 memory-heavy decode 混在一轮以提高资源互补，但 kernel/内存竞争也可能抵消收益。
- 应扫描 chunk size 与 token budget，分别观察 TTFT、TPOT p99、throughput、batch composition 和 GPU counters。

### Q47. prefix caching 的正确收益边界是什么？

**高分回答要点**

- 命中后复用已有 prefix KV，直接跳过对应 prefill 计算，因此主要改善共享长 prompt 请求的 TTFT 和 prefill 吞吐。
- 它不减少新 output token 的自回归 decode 计算；TPOT 若改善，通常来自队列和资源竞争下降，是间接效果。
- 无共享 prefix、prefix 很短、cache 容量不足或频繁 eviction 时收益小；hash lookup 和 block 管理也有成本。
- benchmark 要区分 cold、warm、cache-off，固定 prompt/output，并验证真正命中 token 数。

**实验锚点**：高复用场景命中约 93.7%，warm-cache 相对关闭约 4.32x 吞吐、TTFT 降约 84.8%；这不能外推到随机 prompt。

### Q48. prefix cache 如何保证正确性和租户隔离？

**高分回答要点**

- cache key 不能只含 token ids；还应覆盖模型/adapter、position、相关 sampling/attention 配置，以及会改变 KV 的多模态或模板输入。
- block hash collision 必须可接受地检测/规避；引用计数保证共享 block 不被提前回收，copy-on-write 防止后续 token 修改共享状态。
- 多租户需在 key 中加入隔离域或显式禁止跨租户复用，避免 timing/存在性侧信道和数据泄露。
- 模型热更新、LoRA 切换和 tokenizer/template 变更要使旧 cache 失效；指标应能按租户查看命中与 eviction，但不能泄漏 prefix 内容。

### Q49. 推理部署时如何选择 TP 还是 DP replicas？

**高分回答要点**

- 模型单卡可放下时，DP replica 没有层内 collective，通常 aggregate throughput 和故障隔离更好；TP 可能降低单 rank 参数/KV 压力，但增加每层通信。
- 模型放不下或单请求延迟需要聚合算力时使用 TP；前提是互连足够快且本地 GEMM 不被切得过小。
- DP 需要路由与负载均衡；cache-aware routing、请求长度不均和 replica warm state 会让简单 round-robin 失效。
- 应比较相同 GPU 总数下 capacity、单请求 latency、aggregate goodput、每 GPU tokens/s 和故障域。

**实验锚点**：本机 TP 在 PCIe 上主要换容量；而 4 个 DP replica 在高负载下相对单 replica 只提高约 46%，说明副本扩展还受客户端、路由、CPU、到达分布和每副本 batch 破碎影响。

### Q50. `torch.compile`、CUDA Graph 和 eager 各有什么取舍？

**高分回答要点**

- eager 灵活、冷启动低、调试容易，但 Python dispatch、launch 和未融合算子开销更大。
- 图编译可做 fusion、layout/constant 优化和生成专用 kernel，但有 compile time、graph break、shape specialization、缓存与额外内存成本。
- CUDA Graph 通过 capture/replay 降低 launch overhead，但要求地址和控制流较稳定，动态 batch/sequence 需 padding、分桶或多图管理。
- 必须分别测首次启动、首次请求、warm steady state、不同 shape 命中率和峰值显存；只报 warm throughput 会掩盖生产代价。

**实验锚点**：本项目 compile 配置显示启动与显存代价；在 V100 上还受到 backend/架构支持边界影响，不能把新卡结论直接复制过来。

### Q51. open-loop、closed-loop 与 SLO goodput 如何设计？

**高分回答要点**

- closed-loop 固定并发，前一个请求结束后才发下一个；它容易形成自节流，适合容量扫描但掩盖真实排队。
- open-loop 按外生到达过程发请求；Poisson workload 的 inter-arrival time 服从指数分布，更适合观察过载和排队拐点。
- completed throughput 是单位时间完成数；goodput 只统计满足 TTFT、TPOT、E2E 等 SLO 的请求。
- 对每个 arrival rate 保持足够时长和独立 seed，报告 queue length、p99、错误、未完成请求，并确认 client 不是瓶颈。

**实验锚点**：本机在设定 TTFT<250 ms、TPOT<40 ms 时稳定区约为 5 QPS；超过拐点后 completed throughput 可继续升，goodput 却崩溃。

### Q52. 怎样用排队论解释推理尾延迟？

**高分回答要点**

- 基础关系 `L = lambda W` 可检查平均在途请求、到达率和响应时间是否自洽；但 LLM service time 由输入/输出长度和 batching 决定，不满足简单常数服务时间。
- 当利用率接近 1，微小 service-time 抖动、长请求和 burst 会导致队列及尾延迟非线性增长。
- dynamic batching 又让单请求 service time 依赖其他请求，需用真实 trace/离散事件模型或实测校准，而不能只套 M/M/1。
- admission、优先级、deadline、长度分桶和容量冗余用于控制尾延迟；目标是最大化受 SLO 约束的 goodput。

### Q53. prefill-decode disaggregation 何时有收益？

**高分回答要点**

- prefill 偏 compute-heavy，decode 偏 memory/latency-sensitive；分池可独立扩缩容和批处理，避免长 prefill 干扰 decode TPOT。
- 代价是 KV 从 prefill worker 传到 decode worker的网络传输、序列调度、失败恢复和额外排队。
- 只有阶段干扰/独立扩缩收益大于 KV transfer 与空转成本时才值得；短 prompt、小模型或慢网络可能更差。
- 设计需计算每请求 KV bytes、NIC 带宽/并发、transfer latency 与可隐藏窗口，并观测 prefill/decode utilization、KV transfer p99 和跨池 backpressure。

**追问**：如何为 prompt 长度设置 colocated 与 disaggregated 的动态路由阈值？

### Q54. speculative decoding 和量化分别如何加速推理？

**高分回答要点**

- speculative decoding 用便宜 draft 一次提出多个 token，由 target 并行验证；输出分布可保持正确，但速度取决于接受率、draft 成本、验证 batch、额外 KV 和调度交互。
- 理论每次 target step 接受更多 token 不等于端到端等比例加速；低接受率、强 batching 或内存压力会抵消收益。
- 量化通过减少权重/KV bytes、提高可用低精度吞吐来加速；区分 weight-only、W8A8/FP8、INT4、KV quant，它们影响的瓶颈不同。
- 必须验证硬件是否有高效 kernel、量化/反量化开销、精度/任务质量、校准集、长上下文稳定性与实际 batch shape。

**追问**：decode 明显带宽受限时，weight-only INT4 为什么可能有效？prefill compute-bound 时为什么收益比例可能不同？

---

## 6. Level 5：GPU Kernel、编译与性能工程

### Q55. 接到“训练吞吐下降 20%”的任务，你的 profiling 流程是什么？

**高分回答要点**

- 先确认回归真实存在：同硬件、同输入、同 commit/镜像、同 warmup，查看分布而非单点，并定位首次坏版本。
- 自上而下分解 step：data、H2D、forward、backward、optimizer、collective、checkpoint；同时查看 CPU launch、GPU timeline 和各 rank skew。
- 找 critical path 上的最大变化，再进入 kernel：shape、dtype、调用次数、实现、FLOPs、bytes、occupancy、Tensor Core 和 memory throughput。
- 用最小复现与单变量 A/B 验证因果；优化后同时检查 correctness、loss、显存、尾部和其他 shape，防止局部 benchmark 变快而端到端变慢。
- 最后固化 benchmark、trace marker 和回归阈值，避免同类问题重复发生。

**评分上限提示**：直接打开 profiler 找“最慢 kernel”，但不先确认 workload 和 critical path，最多 2 分。

### Q56. 怎样为一个 kernel 估算 arithmetic intensity？

**高分回答要点**

- 明确计算次数和从目标存储层实际读取/写入的 bytes；不能只按 tensor 理论大小，cache reuse、写回和中间量都要考虑。
- 例如大 GEMM 的每个元素可被 tile 多次复用，intensity 高；elementwise op 通常每元素做少量 FLOPs、至少一读一写，偏带宽受限。
- decode 中权重若每 token/batch 只复用有限次，权重 bytes 主导；batch 增大提高复用和 intensity。
- 将估算点放到对应 GPU 的 HBM/compute roofline，再用 profiler counters 检查实际 DRAM bytes 与 FLOPs。

**追问**：为什么把多个 elementwise op 融合后 FLOPs 几乎不变，却可能明显加速？

### Q57. FlashAttention 的系统原理是什么？

**高分回答要点**

- 标准实现若 materialize `S x S` score/probability，会产生大量 HBM 读写；FlashAttention 将 Q/K/V 分块放入更快的片上存储。
- 用 online softmax 维护每行最大值与归一化和，逐块得到与标准 attention 等价的结果，无需保存完整 attention matrix。
- backward 可重算部分统计/score，以更多计算换 HBM IO 和 activation 显存。
- 性能受 head dim、sequence、causal/varlen、dtype、GPU 架构、shared memory/register 和 kernel 实现影响；不是所有 shape 都同样快。

**实验锚点**：V100 的 SM70 不满足本项目 vLLM FlashAttention-2 backend 要求，运行时回退到兼容 backend；这属于支持矩阵问题，不代表 attention 原理不可运行。

### Q58. kernel fusion 的收益和风险是什么？

**高分回答要点**

- 收益是减少 kernel launch、避免中间 tensor 写回 HBM、提升 producer-consumer locality，例如 bias+activation、RMSNorm、optimizer update。
- 风险是融合后 register/shared-memory 压力上升、occupancy 降低、代码生成膨胀、动态 shape specialization 过多，甚至失去更优库 kernel。
- reduction 与 matmul 融合要考虑跨 block 同步限制；不是把所有算子放进一个 kernel 就最好。
- 用端到端 timeline、HBM bytes、launch 数、occupancy 和多 shape benchmark 判断，不只比较单个融合 kernel。

### Q59. 让你用 Triton 写 RMSNorm，你会怎样设计和验证？

**高分回答要点**

- 每行计算平方和、`rsqrt(mean(x^2)+eps)`，再乘输入和权重；先明确 shape、dtype、是否保存统计量和 backward 需求。
- 一行或分块映射到 program，使用向量化 load、mask 非 2 次幂长度，在 FP32 中累积 reduction，再 cast 输出。
- hidden 很大时单 program 的 register 压力可能过高，需要分段 reduction 或两阶段 kernel；权重读取可利用 cache。
- correctness 要覆盖非对齐 shape、极值、不同 dtype，与高精度 reference 比误差；性能扫描 block size/warp 数，并与 PyTorch/已有 fused kernel 在多 shape 比较。
- backward 不能只依赖数值看起来接近；应做 gradient check 和训练片段验证。

### Q60. occupancy 越高，kernel 一定越快吗？

**高分回答要点**

- occupancy 是活跃 warps 相对上限的比例，受 registers/thread、shared memory/block、threads/block 和硬件限制。
- 足够 warps 可隐藏 memory/指令 latency，但超过所需后更高 occupancy 不再提升；降低寄存器以追求 occupancy 可能引发 spill 到 local memory。
- compute-bound kernel 可能靠更高 ILP 和每线程 tile 在较低 occupancy 下更快；Tensor Core pipeline 也不能只用 occupancy 判断。
- 应结合 eligible warps、stall reasons、register spill、memory throughput 和 achieved FLOPs 判断。

### Q61. CUDA stream/event 如何实现计算通信 overlap？

**高分回答要点**

- 不同 stream 的工作只有在数据依赖、硬件资源和 runtime 允许时才可能并发；默认 stream/隐式同步会破坏 overlap。
- producer compute 完成后记录 event，communication stream wait event；consumer 再等待 collective 完成，形成最小依赖而非全设备同步。
- compute 与 NCCL 同时争用 SM、HBM、copy engine 或互连，因此时间区间重叠不等于两者都保持独立峰值。
- tensor 生命周期必须用 stream-aware allocator/`record_stream` 等机制保护，否则内存可能被过早复用。

**追问**：为什么插入 `cudaDeviceSynchronize()` 便于计时，却会改变你要测的 overlap？

### Q62. PyTorch 2 图编译链路中，graph break 和 dynamic shape 怎样影响性能？

**高分回答要点**

- 大体链路包括 Python graph capture、autograd graph/分解、后端代码生成；候选人应说明自己讨论的是哪个层级，而非把 `torch.compile` 当单一黑盒。
- Python side effect、数据依赖控制流、不支持算子和某些 mutation 会 graph break，导致频繁往返 eager、失去跨算子优化。
- dynamic shape 可通过 guards/symbolic shape 支持，但 shape 分布太散会 recompilation、缓存膨胀；服务端常做 bucket/padding 稳定 shape。
- 排查要记录 compile time、graph 数、break reason、recompile/guard failure、生成 kernel、warm latency 和内存。

### Q63. NCCL 在多机上如何选择路径？遇到带宽异常看什么？

**高分回答要点**

- 数据路径可能是 GPU-NVLink/PCIe-NIC，经 GPUDirect RDMA 走 InfiniBand/RoCE；若条件不满足可能经 host staging，性能和 CPU 占用显著变化。
- 检查 GPU/NIC/NUMA 拓扑、驱动/CUDA/NCCL、HCA 与 GID、MTU、RDMA 能力、交换网络、链路速率、rail 绑定和容器设备映射。
- 从小到大分层测试：单 GPU kernel/内存、节点内 pair/all-rank、节点间 point-to-point、collective；用同消息尺寸和并发数复现训练模式。
- 查看 NCCL 拓扑/algorithm/protocol 日志、NIC counters、丢包/ECN/PFC、PCIe 降速和 rank skew；环境变量只应用于验证假设，不是永久“调参玄学”。

**当前证据边界**：本项目只有单机 PCIe/NCCL 数据。面试时应明确提出上述多机实验计划，而非假装做过 RDMA。

### Q64. 怎样建设性能回归测试，而不让结果充满噪声？

**高分回答要点**

- 分层建立 kernel microbench、单机模型 step、分布式 scaling 和在线 trace replay；每层都有 correctness gate。
- 固定硬件池、时钟/功耗、软件镜像和 workload，做 warmup、多次重复、稳健统计与环境健康检查。
- 阈值同时考虑相对回归、绝对影响和方差；关键指标包括吞吐、p99、显存、compile/startup、错误与数值质量。
- 自动保存 commit、配置、trace 摘要和原始样本；回归发生时可二分版本，并区分代码回归、基础设施噪声和 workload 漂移。
- 不应只守一个平均 tokens/s；例如优化吞吐但让 TTFT p99 或 OOM 率恶化，仍应阻止发布。

---

## 7. Level 6：生产集群、可靠性与 RL Infra

### Q65. 如何为千卡训练任务设计可观测性？

**高分回答要点**

- 训练层：step/phase time、tokens/s、MFU、loss、gradient norm、loss scale、data cursor、checkpoint 状态。
- GPU/rank 层：每 rank kernel/collective 时间、显存、温度/功耗/时钟、ECC/Xid、heartbeat、straggler score。
- 网络/存储层：每 NIC/rail throughput、拥塞/重传、collective size、数据与 checkpoint IO、metadata latency。
- 所有指标带 job/rank/node/parallel-group/step 标签和统一时钟；保留低开销连续指标，并支持触发式抓取 trace。
- 告警围绕用户影响和自动诊断：step 卡住、rank skew、loss 异常、SLO/MFU 持续下降、checkpoint 过期，而不是单纯 GPU busy 低。

**追问**：怎样控制高基数标签和 trace 成本？如何在一个 rank 变慢时快速定位其同组 peer？

### Q66. 一个大训练任务卡住，所有 GPU utilization 看起来仍为 100%，怎么查 NCCL hang？

**高分回答要点**

- 先冻结现场：记录最后完成 step、各 rank stack/heartbeat、NCCL log、GPU Xid、网络 counters，避免直接重启丢证据。
- 判断是真 deadlock、极慢 collective 还是某 rank 未进入 collective；比较各 rank 当前 op、sequence number、tensor shape 和 process group。
- 常见根因包括不同控制流/collective 顺序、一个 rank OOM/异常退出、网络链路故障、数据 loader 卡住导致 peer 等待，以及超时配置掩盖。
- 用 watchdog/async error handling 与独立健康通道检测，超时后协调终止整个 job，避免残留 rank 占卡；从最近一致 checkpoint 恢复。
- 最小化到具体 group/rank/node 后做替换节点、pair test 和代码 replay，区分硬件与逻辑错误。

### Q67. 训练 step 出现长尾 straggler，如何定位？

**高分回答要点**

- 先按 step 计算每 rank phase duration 和 barrier/collective wait，把“被迫等待的 rank”和“真正先变慢的 rank”分开。
- 检查 GPU 时钟降频/ECC、CPU 抢占/NUMA、data shard、文件系统、网络拥塞、kernel shape、MoE token imbalance 和 Python GC。
- 关联同节点、同 NIC、同 TP/DP group 是否共同异常，利用拓扑缩小故障域。
- 短暂噪声可通过调度隔离、数据预取缓冲缓解；持续硬件慢节点应驱逐，算法不均衡则需 repartition/load balance。
- 报告 p50/p99 step time 和有效 MFU，平均值会掩盖同步系统的尾部成本。

### Q68. 训练或推理发生间歇性 OOM，你如何区分原因？

**高分回答要点**

- 区分 allocated、reserved、active、inactive split 和非框架分配；查看峰值发生的 phase 与 allocation stack。
- 训练可能来自变长 batch、prefetch/FSDP AllGather 重叠、临时 workspace、未释放 graph、梯度累积或 checkpoint 保存。
- 推理可能来自 prompt/output 长度尾部、KV block 碎片、过量并发、prefix 引用未释放、compile/CUDA Graph pool 或 preemption 峰值。
- 先构造触发输入并记录 memory snapshot，再调整单一变量；`empty_cache` 可能缓解 reserved memory，但不是修复泄漏。
- 生产方案包括 token budget/admission、显存水位、长度限制、分片/重计算和内存泄漏回归测试。

### Q69. 千卡任务的 checkpoint 与故障恢复策略怎么定？

**高分回答要点**

- 根据故障率、checkpoint 写入时间和可接受丢失工作量选择间隔；规模越大，任一组件故障概率越高，不能沿用单机频率。
- 分片并行写、异步 staging、限流避免压垮训练网络/存储；manifest 原子提交并保留至少一个已验证版本。
- 自动检测故障、释放旧进程、替换坏节点、重建 groups，再从 checkpoint 恢复；记录恢复时间 RTO 和丢失训练量 RPO。
- 定期验证 checksum 与真实 restore，并测试不同 world size/并行配置 reshard；checkpoint“写成功”不代表“能恢复”。
- 数据游标、RNG 和 optimizer 语义必须一起恢复，否则 loss 能继续下降也可能重复/漏训练数据。

### Q70. 数据管道如何把 GPU 持续喂满，同时保证可恢复性？

**高分回答要点**

- 分解 object store/文件系统读取、解压/解析、tokenize、shuffle、packing、host memory、H2D；对每级设置队列和 backpressure 指标。
- 使用分片、顺序大块读取、异步预取、缓存、pinned memory 和 worker 并行，但避免所有 rank 同时打热点对象。
- global shuffle、去重、样本权重与 packing 要有确定语义；每 rank shard 必须避免无意重复，并能记录可恢复 cursor。
- 数据坏样本/超长样本需可观测和隔离，不能让一个 rank 跳过而其他 rank 不跳过导致 collective 不匹配。
- benchmark 应测 steady-state、冷 cache、存储故障和 checkpoint restore 后的数据连续性。

### Q71. 怎样做 topology-aware placement 和故障域设计？

**高分回答要点**

- 收集 GPU-NVLink/NVSwitch、PCIe/NUMA、NIC/rail、机架/交换机和电源故障域，形成可供 scheduler 使用的资源图。
- 把 TP/CP/EP 等高频 critical-path group 放在高带宽低延迟域；PP 相邻 stage 优先近邻；DP 跨更大故障域获得副本扩展。
- 不能只追求通信最短：同一机架放完所有 pipeline 可提高性能，却让单交换机/电源故障损失整个 job。
- 通过 placement 候选的成本模型和实际 collective calibration 选择，并持续检查硬件降级后拓扑是否改变。

**实验锚点**：本机 PIX/PHB/SYS 差异提供了单节点校准方法；生产扩展需要增加 GPU-NIC 和交换网络层级。

### Q72. 在线推理过载时，怎样保护 SLO 和多租户公平性？

**高分回答要点**

- 在入口按租户做 token-aware rate limit、并发上限、长度/预算校验和优先级；请求成本不能只按 req/s 估算。
- scheduler 做 deadline/priority、prefill chunk、decode reservation 和 KV 水位保护；超过能力时应尽早排队/拒绝，避免完成吞吐尚高但 goodput 为零。
- 资源池可按模型/优先级隔离，cache key 与指标按租户隔离；防止 noisy neighbor 占满 KV 或发送超长 prompt。
- autoscaling 要使用 queue、token arrival、KV 和 SLO 预测，考虑模型加载/compile/warm cache 延迟，不能只看 GPU utilization。
- 定义降级策略：降低 max output、切换量化/小模型、关闭低价值功能，且对调用方返回明确状态。

### Q73. RL/后训练 Infra 为什么比普通 SFT 更复杂？

**高分回答要点**

- 同时存在 rollout inference、reward/critic、训练和数据处理，资源特征不同且要编排；生成端常受 decode 限制，训练端偏大 batch compute/communication。
- actor 权重持续更新，rollout worker 需要版本同步；必须定义 policy staleness、样本版本和 on-policy/off-policy 容忍度。
- colocate 可减少权重/KV 传输和空闲，但训练与推理争显存；disaggregate 易独立扩缩，却增加权重广播、队列和故障状态。
- 需要处理变长 episode、过滤/奖励、数据 lineage、可重复性、partial failure 和 backpressure。
- 指标包括 rollout tokens/s、训练 tokens/s、GPU idle、样本年龄、权重同步时间、队列深度、有效样本比例和最终 reward/time-to-quality。

**追问**：训练比 rollout 快或慢时，分别怎样动态分配 GPU？权重更新途中 worker 失败怎么保证版本可追踪？

### Q74. 一次框架/驱动升级后性能和数值都发生变化，你如何组织发布？

**高分回答要点**

- 使用兼容矩阵和锁定镜像，先跑 kernel correctness、确定性/误差、单机性能、分布式 scaling、checkpoint restore 与真实 workload canary。
- 性能变化按 kernel selection、compile graph、collective algorithm、allocator 和数据路径分解；数值变化按 dtype、reduction 顺序、随机数、fused kernel 定位。
- 采用小规模/少数 job 灰度，保留自动 rollback 和旧 checkpoint 可读性；记录每个 artifact 的软件/硬件元数据。
- 若性能改善但 loss/质量不可比，不可发布；若平均快但尾延迟/故障率恶化，也要按用户目标决定。
- 事故后把最小复现、监控缺口和回归测试加入发布门禁，而不只写一次性复盘。

---

## 8. Level 7：本项目答辩与高压追问

### Q75. 请用两分钟介绍这个 8xV100 项目，不要按实验目录流水账。

**参考回答骨架**

> 我在一台 8x V100 PCIe 32GB、无 NVLink 的机器上搭建了一套可复现的 LLM systems lab，目标不是追求绝对性能，而是用控制变量验证分布式训练和在线推理中的关键 trade-off。
>
> 第一层是通信基线。我按 PIX/PHB/SYS 拆分 GPU pair，并测 AllReduce、AllGather、ReduceScatter 和 P2P，发现 512 MiB AllReduce 的 PIX 带宽约 11.74 GB/s，跨 host bridge 或 socket 约 7 GB/s，这为后续并行放置提供了物理解释。
>
> 第二层是训练。我对 DDP 做强/弱扩展、microbatch sweep 和 CUDA trace；8 卡强扩展只有 2.82x，而 microbatch 增大后效率改善，说明关键是每 token 固定梯度通信和 overlap。FSDP 实验则得到明确的容量/吞吐边界：`SHARD_GRAD_OP` 约 10k tok/s、1.55 GiB，`FULL_SHARD` 约 7.7k tok/s、1.06 GiB。Megatron 实验覆盖 TP/SP/PP 和重计算，最重要的结论是 PCIe 上 TP 主要解决容量，SP 在这个 workload 上只省少量显存，而 full recompute 省约 78% 但损失约 30% 吞吐。
>
> 第三层是推理。我在 SM70 上完成 vLLM backend 兼容性诊断，验证 continuous batching、TP/DP、prefix cache 和 Poisson 到达。prefix 高复用时 warm cache 吞吐约为关闭缓存的 4.32x；但在过载实验里 completed throughput 继续升，SLO goodput 却接近归零，说明容量决策必须以尾延迟约束下的 goodput 为目标。
>
> 最后一层是工程证据。我进一步量化 DDP reducer 与 `no_sync()`，做 NCCL 算法/协议和跨 NUMA 并发、MoE variable-split All-to-All、原子 checkpoint/failure injection、自写 Triton RMSNorm，以及 prefix cache 驱逐。随后又下钻 runtime：`TP2 x PP2 x DP2` 中 SYS TP 比 PIX TP phase 慢 3.05x，同 shape collective 顺序错配会静默串数据；FSDP block wrap trace 是 25 次 AllGather/13 次 ReduceScatter；1 GiB KV 在 C=16 时每轮 4 次抢占；Triton hidden8192/1warp 使用 255 registers/thread 并慢 1.72x。所有方向都有独立 correctness 和原始 artifact。边界也很明确：它仍未覆盖多机 RDMA、端到端 3D model、真实 Megatron EP/CP、跨 world-size DCP、Triton backward 和长期真实训练。

**面试官在听什么**

- 是否先讲研究问题和结论，而不是脚本数量；
- 是否能给 3-5 个准确数字并解释机制；
- 是否主动区分实测与外推；
- 是否把负结果和假设修正讲成能力，而不是回避。

### Q76. V100 太老，这个项目对今天的 AI Infra 岗位还有什么价值？

**高分回答要点**

- 不声称 V100 绝对性能代表 H100/B200；架构缺少 BF16/FP8、新 attention backend、NVLink/NVSwitch 等会改变最优配置。
- 仍然可验证跨架构稳定的机制：collective 数据流、拓扑、计算通信比、分片峰值、pipeline bubble、KV 容量、排队和 benchmark 方法。
- 老硬件会放大通信和兼容性问题，适合训练故障定位；例如 backend 回退和 TP 在 PCIe 上变慢都需要读日志和建立成本模型。
- 要迁移结论，应在新硬件重新校准 peak FLOPs/HBM/互连、kernel 支持和 workload shape，而不是复制具体吞吐数字。

**失分点**：回答“原理都一样，所以完全不影响”忽略了硬件能力会改变瓶颈和并行最优点。

### Q77. 为什么 PIX pair 是 11.74 GB/s，而 8 卡 AllReduce bus bandwidth 只有约 7.6 GB/s？

**高分回答要点**

- 两卡 pair 是局部路径；8 卡 collective 必须覆盖更多 PCIe switch/host bridge/NUMA 路径，并共享上行链路。
- Ring 或 NCCL 选择的通道包含慢边，整体由拓扑嵌入、共享竞争和同步共同限制，不能取最快 pair 平均。
- `busbw` 与 pair 报告的逻辑/物理口径也要核对，不能只比数字；应保持消息大小并查看 NCCL algorithm/channel。
- 后续可显式重排 `CUDA_VISIBLE_DEVICES`、限制 rings/channels、并发 pair traffic，验证拓扑映射与共享瓶颈。

### Q78. DDP profile 中 m1 和 m8 的 AllReduce 累计都约 105 ms，为什么 m8 扩展更好？

**高分回答要点**

- 固定模型每次同步的 gradient bytes 相近，因此 AllReduce kernel 数和累计时间不随 microbatch 线性增长。
- m8 每 step 计算/token 更多，通信占比下降；更大的 GEMM 也可能提高计算效率。
- 累计 NCCL duration 不等于 exposed critical-path time，m8 还可能有不同 overlap；需要 timeline 而不是相加作结论。
- 应用吞吐是总 tokens/step 除以 wall time，因此即使通信时间相近，每 token 摊销也显著下降。

### Q79. FSDP `SHARD_GRAD_OP` 比 DDP 更省显存且更快，是否说明 FSDP 全面优于 DDP？

**高分回答要点**

- 不能。这个观测只对特定模型、batch、FP32/FP16 通信设置和单机 PCIe 成立；通信 dtype 本身就是重要变量。
- FSDP 可能因 ReduceScatter、内存压力降低或实现细节获得收益，也会因 parameter AllGather、wrap 粒度和同步变慢。
- DDP 在模型可放下且网络合适时通常路径更简单，可能有更高吞吐；FSDP 的主要价值是容量及更大可行 batch/model。
- 公平结论应画 Pareto frontier，并补 loss/数值一致性、不同模型规模、batch、wrap 和网络的 sweep。

### Q80. SP 节省很少，而 recompute 节省约 78%，你会因此完全放弃 SP 吗？

**高分回答要点**

- 不会把一个配置的结果推广到所有模型。应先用 memory snapshot 判断峰值组成，验证可被 SP 分片的 activation 是否本来占比很小。
- 更大 hidden/microbatch/层数、不同 attention kernel、TP degree 和更长序列可能改变收益；SP 还可能是某些 TP+EP layout 的必要条件。
- recompute 的 27%-30% 吞吐成本很高；生产配置可能组合 selective recompute 与 SP，而不是只选极端 full recompute。
- 下一实验应扫描 microbatch×sequence×TP，增加 selective recompute，并同时看峰值、throughput 和通信时间。

### Q81. PP8/M16 比 M1 快 2.59x，为什么不是 16x？

**高分回答要点**

- M1 的理论利用率仅 1/8；M16 理论利用率为 `16/(16+7)=69.6%`，相对上界约 5.57x，不是 16x。
- 实际还有 stage 参数/计算不均衡（本项目约 2.76x）、P2P、调度、较小 microbatch kernel 效率和同步。
- 吞吐比较还要确认 global batch 和计量单位；增加 microbatch 数不是凭空减少总计算。
- 可用 per-stage timeline 分离 bubble、imbalance 和通信，再尝试非均匀 partition/interleave。

### Q82. 用 dummy 7B 权重做 vLLM 实验是否有意义？

**高分回答要点**

- 有限但有意义：它能验证模型结构规模下的参数/KV 分配、engine 启动、调度、并行通信、cache 和请求路径，且避免下载/加载真实大权重成为实验阻塞。
- 它不能验证真实权重的输出质量、量化误差、真实 checkpoint IO，也未必复现所有权重分布相关 kernel 或业务输入行为。
- 因此项目另用真实 Qwen2.5-0.5B 权重验证端到端生成正确性；两类实验回答不同问题。
- 面试中应称为 synthetic/systems benchmark，不应把它包装成 7B 模型质量验证。

### Q83. 如何证明 prefix cache 的 4.32x 不是一个有偏 benchmark？

**高分回答要点**

- 这个数字只代表刻意构造的高共享长 prefix 场景，报告必须同时给 prefix 长度、复用比例、输入/输出和并发。
- 设置 cache-off、cold-cache、warm-cache 三组，启动配置与请求完全一致；确认 hit token/率，而不是只看到 latency 下降。
- 多次重启或清空 cache，随机化请求顺序，排除 compile warmup、模型加载和客户端连接复用。
- 本项目已补默认/1 GiB cache 的 pollution/revisit：累计 hit rate 只差 2 点，小 cache 的 revisit p95 TTFT 却退化 5.87x，说明必须看工作集和阶段 counter delta。
- 下一步再扫 prefix 长度、共享比例、复用距离和租户隔离；随机 prompt 上不应承诺 4.32x。

### Q84. 如果再给你两周独占这台 8 卡机器，最优先补什么？

**高分回答要点**

- 第一优先是把已验证的 `TP2 x PP2 x DP2` process group 接入真实 TP layer、1F1B pipeline 与 DP gradient sync，固定 global batch，检查 loss、collective 序列和 critical path。
- 第二优先是 PP per-stage trace 与重切层；当前 2.76x 参数不均衡仍只是 proxy，需要真实 F/B/P2P 时间验证非均匀 partition。
- 第三优先是 PyTorch DCP/FSDP2 的 8-rank save、4-rank reshard restore，以及真实 packed DataLoader、MFU、checkpoint pause 和数小时 soak。
- 推理侧做四个单卡副本的 least-connections/cache-aware LB，补每副本 queue/cache 指标和 30-60 分钟 mixed-length/mixed-prefix SLO。
- 多机 RDMA 无法在这台单机证明，应准备脚本/指标和成本模型，但不消耗时间制造伪多机结论。

**4 分答案特征**：有优先级、每项有可证伪假设、指标和停止条件，而不是罗列热门名词。

---

## 9. Level 8：系统设计题

系统设计题没有唯一配置。面试开始的前五分钟应主动索要模型架构、GPU/网络、精度、序列长度、global batch、SLO、容错和成本约束。未经估算直接报 `TP=8, PP=16` 通常是失分项。

### Q85. 设计一个 256xH100 的 70B dense 模型预训练系统

**必须覆盖**

- 容量表：参数、梯度、Adam state、master weight、activation、临时 buffer；说明 BF16/FP8 和 optimizer 分片假设。
- 枚举 DP/FSDP、TP、PP、CP 候选，给 `world size` 与 global batch 约束；TP/CP 放 NVLink 域，跨节点通信说明 NIC 路径。
- 用 `6*N*tokens` 作为 dense Transformer 粗略训练 FLOPs 起点，再说明 attention、embedding、recompute 会修正；给 MFU 与目标 tokens/s 的估算方法。
- pipeline microbatch/bubble、activation recompute、数据 packing、distributed optimizer 和 checkpoint 策略。
- 观测 MFU、step 分解、collective、straggler、loss/overflow、数据等待、checkpoint pause；故障注入和 restore 验证。

**高压追问**

- 模型单卡能否放下不是唯一问题，为什么仍可能使用 TP？
- global batch 被优化团队固定后，如何选择 PP microbatch 数？
- 某个机架链路降级 30%，job 是否继续、重排还是重启？

### Q86. 设计一个大规模 MoE 预训练系统

**必须覆盖**

- 区分总参数与每 token active 参数，估算 expert 权重容量、router/attention dense 部分和每 token FLOPs。
- 明确 TP/EP/DP/PP group 关系、expert placement 和 All-to-All 路径；说明为什么 EP 对网络和拓扑更敏感。
- router top-k、capacity factor、token drop、auxiliary loss、expert replication/dynamic placement。
- grouped GEMM、token permute、dispatch/combine overlap，以及每 expert token 分布的 p99/max-to-mean。
- checkpoint 如何保存/reshard experts；节点故障后如何保持 expert 集合完整。

**高压追问**：平均每 expert token 很均匀，但 step 仍有长尾，你还会看什么？

### Q87. 用 64 张 GPU 部署 70B chat 模型，要求 TTFT p99<1s、TPOT p99<50ms

**必须覆盖**

- 先索要输入/输出长度分布、arrival trace、上下文上限、精度/质量、可用 GPU 型号和 availability target。
- 计算参数与单请求 KV，选择最小可行 TP；剩余 GPU 做 replicas，设计 topology-aware placement 和 cache-aware routing。
- scheduler 的 max batched tokens、chunked prefill、decode reservation、admission/preemption；用 open-loop sweep 找 SLO knee。
- prefix cache、量化、compile/CUDA Graph 的收益边界；模型加载、warmup 和 cache 冷启动。
- 多租户隔离、健康检查、滚动升级、失败请求重试与容量冗余；主指标是 SLO goodput/GPU，不只是 output tok/s。

**高压追问**：流量突然 3x 时，为什么立刻扩容也可能来不及？什么请求应先拒绝？

### Q88. 设计支持 1M context 的推理服务

**必须覆盖**

- 先算 KV cache；即便 GQA/量化，单请求容量也可能决定整个部署，不能先谈 scheduler。
- prefill 的 attention 计算、TTFT 与显存；考虑 CP、ring attention、chunked prefill、KV offload/分层存储和算法级稀疏方案的语义差异。
- 超长请求 admission、异步进度、取消、租户配额和对短请求的隔离，避免 head-of-line blocking。
- prefix reuse、prompt ingestion、KV transfer 和 PD disaggregation 的网络预算。
- benchmark 使用长度分桶和真实 needle/retrieval 质量，性能优化不能破坏长上下文正确性。

### Q89. 设计可跨并行配置恢复的 distributed checkpoint 服务

**必须覆盖**

- 全局 tensor identity、shape、dtype、placement/shard metadata，与 rank-local chunk 文件解耦。
- rank 并行上传、checksum、manifest、两阶段/原子完成标记、垃圾回收和版本兼容。
- restore planner 根据新 mesh 读取并重分片，控制读放大和内存峰值；optimizer/RNG/data cursor 一并恢复。
- storage 限流、训练网络隔离、异步 snapshot 一致性、加密与访问控制。
- 测试 partial write、rank/node loss、坏 chunk、不同 world size、旧版本迁移和真实 loss 连续性。

### Q90. 设计一套自动寻找最佳并行配置的系统

**必须覆盖**

- 输入模型图、shape/dtype、GPU 拓扑/显存/峰值、collective calibration、global batch 与约束。
- cost model 估算每算子 FLOPs/bytes、activation/状态峰值、DP/TP/PP/CP/EP 通信、pipeline bubble 和 overlap。
- 搜索合法 mesh/config，先做静态剪枝，再运行少量 calibration/profile 校正模型；输出 Pareto frontier 而非单点。
- 目标函数可为 time-to-train、tokens/s、SLO goodput、成本或能耗，并对 OOM/故障域设置硬约束。
- 线上反馈修正带宽、kernel 和 straggler 分布；配置变化后必须做 correctness/收敛 gate。

---

## 10. Level 9：工程实现与现场深挖

这一层不再考“知道术语”，而是检查候选人是否真的写过 runner、读过 runtime 日志、定义过 correctness，并能解释反常结果。回答应主动给出代码路径、同步边界、指标口径和证据限制。

### Q91. `bucket_cap_mb=1`，为什么实际最大 DDP bucket 仍有 93.75 MiB？

**高分回答要点**

- `bucket_cap_mb` 是 reducer 组 bucket 的目标容量，不是把每个 parameter tensor 任意切片的硬上限；单个大参数通常作为完整梯度进入一个 bucket。
- 本实验 vocab 32,000、hidden 768，embedding/output 相关大矩阵形成约 93.75 MiB 参数，因此 cap 1/4/25 MiB 时最大 bucket 仍接近 94-95 MiB。
- 不应从命令行 cap 推断 collective 数和 bytes；应读 `_get_ddp_logging_data()` 的 `bucket_sizes`、重建后的 bucket 和 parameter ready order。
- reducer 首轮可能使用初始布局，观察 steady state 要先 warmup，确认 `has_rebuilt_buckets` 和实际迭代阶段。

**高压追问**：如果大 embedding 破坏 overlap，能否手工拆 parameter？这会影响权重绑定、optimizer state、checkpoint 命名和 kernel，不应只为 bucket 数粗暴改模型；可先考虑分片 embedding、通信 hook 或参数注册/执行顺序。

**实验锚点**：cap 1/4/25/100 MiB 分别得到 50/37/13/4 个 bucket，最大 bucket 仍为 93.75/95.26/95.26/108.05 MiB。

### Q92. 100 MiB 配置只有 4 个 NCCL kernel，为什么比 4 MiB 的 37 个更慢？

**高分回答要点**

- collective 数减少只降低固定 launch/协议开销；更大的 bucket 要等更多梯度 ready，首个 AllReduce 更晚，compute/communication overlap 可能下降。
- 本实验 100 MiB 的 NCCL union 约 104.02 ms，确实略短于 4 MiB 的 106.84 ms；但 overlap 从 20.44% 降到 15.05%，exposed NCCL 仍约 88.36 ms。
- 端到端吞吐应由 max-rank step time 判定：100 MiB 约 30.1K tok/s，4 MiB 约 32.8K tok/s。
- 25 MiB 结果方差较大，因此结论应是“数量不单调决定性能”，而不是宣布 4 MiB 永远最优。

**追问**：如何进一步验证首发时间？在 trace 中定位第一个 NCCL kernel 相对 backward 起点，并按 bucket 关联 gradient-ready event；更完整时用 Nsight Systems 做多 rank 对齐。

### Q93. DDP 梯度累积中，`no_sync()` 应该包哪里？为什么？

**高分回答要点**

- 对前 `A-1` 个 microstep，`with model.no_sync():` 应覆盖 forward 和 backward；最后一个 microstep 正常 forward/backward 触发一次 reducer 同步。
- DDP 文档和实现要求 forward 也在 context 内，因为 forward 会准备 reducer/unused-parameter 等迭代状态；只包 backward 容易与预期不同。
- 每个 microstep 的 loss 要除以 accumulation，或在 optimizer step 前等价缩放 gradient；否则 effective learning rate 被放大 A 倍。
- 实验输入应使用 accumulated batch 的不同 slice。重复同一 microbatch 虽能测性能，却不能作为等 effective batch 的数值语义对照。
- 只有最后一次同步时，13 个 bucket 产生 13 个 NCCL kernel；每次都同步则是 `8 x 13 = 104`。

**实验锚点**：m1a8 no-sync 为 86.4K tok/s，sync-every 为 33.3K tok/s，前者快 2.59x。

### Q94. global batch 都是 64，为什么 m8a1 比 m1a8+`no_sync()` 快 2.52x？

**高分回答要点**

- global/effective batch 只约束一次 optimizer update 消费的样本数，不约束执行图成本。
- m8a1 只执行一次 forward/backward；m1a8 执行 8 次，重复支付 attention、LayerNorm、launch、autocast、Python 和框架调度开销。
- m8 的 GEMM M 维更大，更容易提高 Tensor Core/SM 利用率；m1 的小矩阵效率低。
- 两者都只有 13 个 NCCL kernel，但 m8 trace 的 NCCL overlap 为 43.76%，no-sync 约 22.69%，说明计算形状也改变 overlap。
- 数值上也不保证逐 bit 相同，因为 GEMM/梯度累加顺序不同；应比较梯度、参数误差和训练质量。

**失分点**：只回答“m8 并行度更高”，却说不出 forward 次数、GEMM shape、通信次数和数值顺序。

### Q95. 你是怎样从 Chrome trace 计算 exposed communication 的？

**高分回答要点**

- 解析 CUDA kernel event，按时间戳形成 `[start, end)` 区间；分别识别 NCCL kernel 和所有非 NCCL GPU kernel。后者只是本解析器的 `compute` 分类，不等于纯 GEMM/FLOPs。
- 先对每一类区间做 union，防止嵌套/并发事件重复计时；再求两个 union 的 intersection。
- `overlap fraction = intersection / NCCL union`，`exposed NCCL = NCCL union - intersection`。不能把 key-averages 的 duration 直接相加。
- rank 0 单步 trace 只近似本 rank 的事件暴露量，不是全 job critical path；绝对性能仍用无 profiler 的多次稳态 wall-time。
- 更严谨分析还需处理多 stream、CPU launch、step 边界、跨 rank 时钟对齐和 profiler overhead。

**上机追问**：给一组重叠区间，要求现场写 union/intersection，并为相邻、包含、空列表和跨 step 事件写单测。

### Q96. 为什么 NCCL 的最快算法/协议会随消息大小变化？

**高分回答要点**

- 小消息主要受 latency、同步轮次和协议开销控制；Tree 深度较低，LL 针对低延迟。
- 大消息更看稳态带宽、chunk/pipeline 和拓扑映射；Simple 常有更高 payload efficiency，Ring/Tree 的具体胜负由路径和 channel 决定。
- LL128 处于延迟和带宽之间，但支持条件受硬件/拓扑影响，不能假设所有机器都可强制使用。
- 本机 crossover 是校准数据，不是普遍阈值：1-16 KiB Tree+LL，64 KiB Ring+LL，256 KiB Ring+LL128，1-16 MiB Ring+Simple，64-256 MiB Tree+Simple 略胜。
- 生产通常让 NCCL 自动选择；只有发现稳定回归时才用强制矩阵做诊断，并保留 correctness check。

### Q97. 为什么把 NCCL channel 从 2 增加到 8 反而变慢？

**高分回答要点**

- channel 增加并行 chunk/路径机会，也增加 GPU blocks、SM 占用、launch/调度和共享链路竞争；收益不是单调的。
- 256 MiB Ring+Simple 本机 1/2/4/8 channel 为 6.92/6.98/6.15/6.40 GB/s，2 最快。
- 训练中 NCCL 与 GEMM 争 SM 时，孤立 collective 的最优 channel 也未必是端到端最优；需要同时测 exposed time 和 step time。
- 当前 NCCL 文档把强制调优变量定位为实验/调试手段，并在较新版本用 CTA 变量替代旧 channel 变量；版本兼容要从本机日志确认。

**追问**：如何防止一个环境变量污染全部作业？把它限定在 runner 进程环境，结果中持久化 env，生产通过 canary 和版本化配置发布，不写进全局 shell profile。

### Q98. 四组跨 socket pair 并发为什么总带宽约减半，而四组 PIX pair 没有？

**高分回答要点**

- PIX pair 分布在各自较近的 PCIe 路径，四对可近似使用独立局部资源；单独与并发 aggregate 均约 47.15 GB/s。
- SYS pair 跨 CPU/NUMA/host bridge，共享上行链路、内存/SHM transport 或 socket interconnect；并发 aggregate 从 27.45 降到 13.21 GB/s。
- `nvidia-smi topo -m` 只给静态关系，不能表达多个 flow 同时争用；必须做 alone-vs-concurrent 对照。
- `NCCL_DEBUG` 中 PIX 使用 P2P/direct，跨域路径可看到 SHM/direct 等 transport 选择，日志和性能现象相互印证。
- 这仍不能证明多机 NIC/交换机行为；RDMA 需要单独测 rail、GDR、拥塞和路由。

**系统设计追问**：TP、DP、EP group 同时存在时，placement 目标函数应包含哪些并发边，而不只是最快 pair？

### Q99. 一个 NCCL 性能回归，你会保存和比较哪些现场？

**高分回答要点**

- 硬件/拓扑：GPU/NIC/CPU NUMA、PCIe link width/speed、P2P capability、affinity 和进程 placement。
- 软件：driver、CUDA、NCCL、framework、container/插件版本和全部 `NCCL_*` 环境变量。
- workload：collective、dtype、count/bytes、rank/world size、in-place、warmup/iterations、并发 group。
- runtime 日志：algorithm/protocol/channel/CTA、ring/tree、transport、chunk size、GDR/P2P/SHM 决策。
- 指标：algbw/busbw、p50/p95、correctness、GPU clocks/power、端到端 exposed time；与已知 good run 做同条件 A/B。

**失分点**：第一步就强制 `NCCL_ALGO=Ring`，没有先保存自动选择和版本现场。

### Q100. variable-split `all_to_all_single` 的 split size 如何得到，combine 时为什么要反过来？

**高分回答要点**

- router/发送端先得到发往每个 destination 的 `send_splits`，其和等于本 rank routed token 数。
- 各 rank 先用一次小型 All-to-All 交换 count，得到“每个 source 将发给我多少”的 `recv_splits`；其和决定接收 buffer 大小。
- dispatch 调用的 input splits 是 `send_splits`，output splits 是 `recv_splits`。
- combine 要把 expert output 发回原 source，因此本地 input splits 变成 `recv_splits`，output splits 变成原 `send_splits`。
- split、token permutation 和 inverse permutation 任一错位都可能 shape 合法但语义错误，所以实验用全局唯一 token id 做回环 oracle。

**编码追问**：某 rank 的某个 split 为 0 时是否正确？如何验证所有 rank 的 send sum 与对应 recv sum 全局守恒？

### Q101. 为什么非热点 rank 的 combine 也可能很慢？

**高分回答要点**

- collective 完成取决于所有 peer 进入并推进操作。非热点 rank 先完成少量 expert compute 后到达 combine，会等待仍在计算 24,576 tokens 的热点 rank。
- profiler 将等待呈现在 collective/stream 上，不代表网络传 1,176 tokens 本身需要那么久。
- 要做跨 rank timeline，比较 dispatch 结束、expert compute 结束和 combine launch/finish；collective 前的 arrival skew 是关键。
- 可以插 barrier 做诊断以分离 arrival skew 和纯 collective，但 barrier 会改变真实关键路径，不能作为最终优化。
- 优化可能在 router balance、capacity、expert replication/EPLB、token drop/reroute 或 compute/communication overlap，而不只是换 NCCL 参数。

**实验锚点**：hot case max compute 5.40 ms，max combine 9.70 ms；balanced 分别 1.26/1.79 ms。

### Q102. 评估 MoE 负载均衡，为什么 `max/mean` 一个指标不够？

**高分回答要点**

- 还要看每 expert token 的 p95/p99、方差、Gini/熵、跨 step 时间相关性和 source-destination 流量矩阵。
- 相同 token 数下，不同序列/shape、padding 和 grouped GEMM packing 可能有不同 compute efficiency。
- 系统指标包括 dispatch/combine bytes/time、arrival skew、expert compute、drop/reroute、capacity utilization、峰值显存和 step tail。
- 模型指标包括 router auxiliary loss、z-loss、token drop 对主 loss/质量的影响；系统最均匀不一定是模型最优。
- 应按 layer 观测，因为少数层热点即可决定 step critical path，模型级平均会掩盖它。

### Q103. 如何把本项目的 MoE toy 扩展为更接近生产的实验？

**高分回答要点**

- 实现 top-k router、stable token permutation/inverse permutation、capacity factor、drop/reroute 和 aux/z-loss。
- 将多个 expert 放在每 rank，使用 grouped GEMM；比较 padding、group size 和 expert token shape。
- 引入 TP/EP/DP process group，明确 expert 权重如何分片/复制以及 All-to-All 穿过哪些拓扑。
- 尝试 dispatch/combine 与 dense/attention 或 expert compute overlap，使用跨 rank timeline 验证 critical path。
- 加入 skew trace、动态 expert placement/replication、checkpoint/reshard 和数值对照。

**边界表达**：当前可证明“variable-split All-to-All 与热点等待”，不能证明真实 Megatron EP 的吞吐或收敛。

### Q104. 为什么 checkpoint 的 manifest 必须最后发布？

**高分回答要点**

- 多 rank 文件不是同时完成；如果目录存在就视为有效，恢复端可能读取半写版本。
- 每个数据文件先写临时名并原子 rename；所有预期 shard 存在、长度和 checksum 确认后，最后原子发布 manifest 作为 commit point。
- 恢复只枚举已提交 manifest，不根据“最新目录名”猜测；partial checkpoint 没有 manifest，因此不可见。
- manifest 应包含格式版本、step、world/mesh metadata、tensor/shard identity、bytes/checksum；生产还要考虑对象存储没有 POSIX rename 的语义。
- GC 必须先判断没有 reader/引用，再删未提交临时文件和过期版本，避免与慢 writer 竞态。

**实验锚点**：rank 3 跳过 rank-local 文件时，五组 partial case 均不发布 manifest。

### Q105. 为了“可继续训练”，checkpoint 到底要保存哪些状态？

**高分回答要点**

- model parameters/buffers、optimizer state、LR scheduler、GradScaler、global step/microstep 和精度/版本 metadata。
- 每 rank CPU RNG、CUDA RNG，以及 dropout、采样或并行 RNG tracker；不能只保存一个全局 seed。
- data sampler/generator、shuffle epoch、packed dataset cursor、已消费 token/sample 数，保证不重不漏。
- 并行/shard metadata、tensor identity 和 checkpoint schema；跨 mesh 恢复需要 planner 重新分片。
- 生产还可能需要 EMA、MoE router/expert state、FP8 scaling history、compile/runtime cache 的重建规则，而不是盲目序列化所有内部对象。

**追问**：DataLoader 有 prefetch 时，一个 cursor 是否足够？通常不够，需要定义 snapshot 边界，处理已取未消费 batch 和 worker RNG。

### Q106. model、optimizer、RNG 和 data cursor 都精确加载，为什么恢复后 step 4 仍不同？

**高分回答要点**

- “持久化状态相同”不保证 runtime 执行状态和浮点操作顺序相同；浮点加法不满足结合律。
- 本实验首先验证保存点 model/optimizer SHA 精确，再用每步 gradient hash 定位第一个差异就在 step 4，而不是几步后 optimizer 才漂移。
- fused 改 single-tensor、启用 deterministic algorithms 仍有约 `3.725e-9` 最大参数差，排除了两个常见单一原因。
- DDP 连续 run 的 reducer 已按 gradient-ready order 重建 bucket；新进程直接 load 时 reducer 生命周期不同，归约分组/顺序可造成 ULP 级变化。
- 生产验收应区分 bitwise、数值误差有界、loss continuity 和最终质量等层级。

**失分点**：直接说“GPU 本来就随机”，没有逐层排除 RNG、输入、optimizer 和 reducer。

### Q107. reducer warm control 为什么是强证据，又为什么不能过度外推？

**高分回答要点**

- 对照只在 resume 前增加 3 次 dummy backward，让 reducer 达到与连续 run 相同的 rebuilt bucket/parameter order；随后再加载 checkpoint，避免 dummy 更新污染模型和 optimizer。
- 该组 gradient/model SHA、loss sequence 和最终参数全部 exact，最大差为 0；其他控制仍在 step 4 首次分叉。
- 这比“机制上可能是 reducer”更强，因为它操纵了怀疑变量并消除了结果差异。
- 仍不能推导所有 DDP 版本/模型的恢复差异都来自 reducer，也不能要求生产序列化私有 reducer 对象。
- 可迁移结论是：位级回放测试必须控制 framework runtime lifecycle，并用 first-divergence fingerprint 定位。

### Q108. rank exit 和 straggler timeout 的检测路径有何不同？

**高分回答要点**

- rank `os._exit(17)` 会被 elastic launcher 通过子进程退出状态发现，最终 `ChildFailedError` 指出 rank 和 exit code。
- straggler 仍存活但不进入 collective，launcher 看不到进程退出；其他 rank 的 ProcessGroupNCCL watchdog 在超时后报告 collective sequence、op 和 numel。
- 本实验 timeout 6 s，日志定位 `SeqNum=3`、`ALLREDUCE`、1,048,576 elements；外层 torchrun 总退出约 15.56 s，还包含传播和清理时间。
- `TORCH_NCCL_ASYNC_ERROR_HANDLING` 等设置会影响错误传播/abort 行为，必须与框架版本一起记录。
- runner 最后用 `pgrep` 检查无子进程残留；生产还要清理 rendezvous、共享内存、临时 checkpoint 和 scheduler allocation。

**追问**：为什么 timeout 不能设得极短？正常大 collective、编译、checkpoint 或系统抖动会造成误杀；要结合 heartbeat、progress 和分层超时。

### Q109. 这个 Triton RMSNorm kernel 是怎样映射到 GPU 的？

**高分回答要点**

- 一个 Triton program 处理一行，grid 大小等于 rows；hidden 维用 `next_power_of_2` block 和 mask 加载。
- FP16 input 转 FP32，做平方和 reduction、mean、`rsqrt`，再乘 FP16 weight 并写 FP16 output。
- 融合避免显式 composite 的中间 tensor 和多次 HBM round trip；每行独立，无跨 program 同步。
- hidden 很大时 power-of-two block、register/shared memory 和 occupancy 可能成为限制，不能无限沿用单行单 program。
- wrapper 中输出分配也会影响端到端 API 时间；benchmark 要说明计时是否包含 allocation。

**正确性追问**：为什么 reference 用 FP32？避免只与另一个 FP16 实现互相验证；同时报告 max abs/relative error 和极端输入。

### Q110. RMSNorm 的“805 GB/s 有效带宽”是怎么计算的，为什么不能当 HBM 实测？

**高分回答要点**

- 本实验最小字节模型为 `2 * rows * hidden * element_size + hidden * weight_element_size`，即读一次 input、写一次 output、读一次 weight。
- 用该 bytes 除以 kernel time 得约 805 GB/s，是算法字节口径，方便 shape 间比较。
- cache line、transaction、weight 是否每个 program 重读、ECC、写分配和内部临时流量都未计入；因此不是 DRAM bytes counter。
- 要声称 HBM 利用率，应使用 Nsight Compute 的 DRAM throughput/bytes，并核对 L2 hit、load efficiency 和时钟。
- 不能把“有效带宽高于某个直觉值”立即判为错误，先确认分母是否是最小 bytes 且 weight 命中 cache。

### Q111. 为什么多个小 shape 的 Triton 时间都卡在约 0.05 ms？如何验证？

**高分回答要点**

- 工作量不足时，launch、Python/driver dispatch、事件计时精度和固定调度开销主导，计算/内存 bytes 增长尚未反映到 wall-time。
- 本实验 rows=1,024 时 hidden 768/4096/8192 均约 0.05 ms，而 rows=8,192、hidden 4096/8192 才随 bytes 增到 0.169/0.333 ms。
- 用 CUDA Graph 或批量重复 kernel 减少 launch 影响；用 profiler 看 kernel 本体和 CPU gap；扩大 rows/hidden 检查线性区间。
- benchmark 要 warmup JIT，并把首次 compile 与 steady state 分离；使用 CUDA event/同步，不能只用未同步的 CPU timer。
- 小 shape 的 speedup 可能主要来自减少 launch 数，而不是单 kernel 带宽提高，两种价值都可以但要分开表述。

### Q112. 把这个 RMSNorm 变成可生产使用的 kernel，还缺什么？

**高分回答要点**

- backward：input/weight gradients、FP32 accumulation、数值误差和额外 reduction；验证 autograd/gradcheck 与基线训练。
- shape/dtype：BF16、FP32、非连续 stride、不同 hidden、batch/sequence layout、极端值和 epsilon。
- performance：num warps/stages autotune、two-pass 大 hidden、vectorized/coalesced load、register spill、occupancy 和 graph capture。
- integration：dispatcher/autograd wrapper、fallback、compile cache、版本/SM capability gate、stream 和错误处理。
- regression：跨 GPU/driver/Triton 版本的 correctness/perf CI，端到端模型收益，而不只微基准。

**边界表达**：当前证明的是 V100/SM70 上 forward shape sweep，不是完整 fused training kernel。

### Q113. 如何构造能证明 prefix cache 驱逐的 workload？

**高分回答要点**

- 先用固定 anchor prefixes 跑 cold 和 warm，建立未命中/命中基线；随后用超过 cache 工作集的互异长 prefix 做 pollution。
- 再访问完全相同的 anchor 做 revisit，最后立即重跑一次 rewarm；revisit 退化、rewarm 恢复可区分驱逐与永久服务回归。
- 设置默认和受限 cache 容量对照，保持模型、prompt、并发、seed 和请求顺序不变。
- 每阶段抓 cache query/hit token counter 差分，并看 TTFT/throughput；仅看客户端延迟不能唯一证明 cache state。
- 本实验默认 225,200 tokens 能保留 anchor；1 GiB/18,720 tokens 在 48 个独立 1,024-token prefix 污染后，revisit mean/p95 TTFT 为 warm 的 2.01x/5.87x。

**追问**：为什么 cold 阶段也有 88.89% hit？同一阶段 48 请求只使用 4 个 prefix，前面的请求填 cache 后，后续请求发生批内复用；所以必须结合阶段顺序理解 counter。

### Q114. 为什么进程生命周期累计 prefix hit rate 会掩盖 SLO 回归？

**高分回答要点**

- counter 混合了 cold、warm、无共享 pollution、revisit、rewarm，业务关键窗口在总量中占比有限。
- 本实验最终累计 hit rate 76.9% 与 74.9% 只差 2 点，但小 cache 的 revisit p95 TTFT 退化 5.87x。
- 应按相邻 Prometheus snapshot 做 delta，并按模型、租户、route、prefix class 和时间窗口聚合。
- 同时看 cache capacity/used blocks、eviction/recompute、TTFT tail、prefill tokens/s、SLO goodput 和每副本 queue depth。
- cache-aware routing 会提高命中但可能制造副本热点，所以命中率必须和负载偏斜、排队及拒绝率联动。

**服务工程追问**：runner 如何避免残留 worker？独立 process group 启动，health check 后运行，退出对整个 group 先 TERM、超时 KILL、`wait` 回收，并最终检查 GPU/进程。

### Q115. 让你现场证明这些数字不是手工写进报告的，你会怎样从原始产物重建结论？

**高分回答要点**

- 先给实验命令和版本环境，指出 raw log/JSON/trace/metrics 的一一对应关系，不从最终 Markdown 反向抄数。
- 运行 `parse_engineering_deep_dive.py`，说明每个 parser 的字段来源、单位转换、分组键、重复均值/方差和 derived ratio。
- 抽查 correctness：NCCL `wrong=0`、MoE token-id roundtrip、checkpoint manifest SHA、Triton FP32 oracle、vLLM `failed=0`。
- 对一个结论手工复算，例如 `13.21/27.45` 的跨域 aggregate、`321.58/159.81` 的 revisit TTFT，或 trace interval union/intersection。
- 运行 `validate_results.py` 检查期望文件数、JSON、有限值、失败状态和报告存在性；再检查没有残留进程/GPU 占用。
- 最后主动说明短 run、单 rank trace、toy/dummy model 和单机 PCIe 边界。可复现不等于可无限外推。

**4 分答案特征**：能从任意一条 claim 一路追溯到命令、原始事件、parser 公式和 correctness，而不是只展示一张图。

---

## 11. Level 10：框架运行时与底层契约

这一层不再满足于“知道某框架用了哪个 collective”。回答必须落到 communicator、rank、sequence、stream、内存生命周期、调度状态机和 profiler 证据。每题都可以沿“为什么没有报错”“哪个 rank 先异常”“哪一个指标能证伪”继续追问。

### Q116. 如何从 `TP=2, PP=2, DP=2` 推导每个 rank 的坐标和三类 group？

**高分回答要点**

- 先声明 rank layout。本实验 `tp_local` 使用 `rank=((dp*PP)+pp)*TP+tp`，所以 rank 0-7 坐标依次为 `(dp,pp,tp)`，TP 是最内层连续维度。
- 固定另外两维枚举 TP 得到 `(0,1),(2,3),(4,5),(6,7)`；固定 DP/TP 枚举 PP 得到 `(0,2),(1,3),(4,6),(5,7)`；固定 PP/TP 枚举 DP 得到 `(0,4),(1,5),(2,6),(3,7)`。
- 必须校验 `TP*PP*DP=world_size`、每个 rank 在每一维恰好属于一个 group、group 无重复 rank，并给 planner 确定性排序。
- group 逻辑正确不代表 placement 最优；还要把 rank 映射到 PCIe/NUMA/NIC 拓扑，按通信频率和关键路径计算成本。

**追问**：若 world size 不能整除三维乘积怎么办？生产 planner 应拒绝非法配置，或明确引入 standby/replica 维，不能悄悄遗漏 rank。

### Q117. 为什么通常优先把 TP group 放在最快链路？本机证据是什么？

**高分回答要点**

- TP collective 位于每层 forward/backward 关键路径，调用频率通常远高于每 step 一次的 DP gradient sync；PP P2P 又可能随 schedule 与 compute 交叠。
- 本实验每 phase 64 MiB，TP 做 4 次 AllReduce。TP 用 PIX 时 25.55 ms；换到 SYS pair 后 77.92 ms，慢 3.05x，串行总工作负载慢 1.62x。
- placement 目标不能只最小化单次 collective latency，应使用 `frequency x bytes x exposed_fraction`，再考虑共享 PCIe root/NIC contention。
- 在 NVSwitch 节点、跨节点 TP 或大 EP All-to-All 中优先级可能改变，必须按真实通信图重算。

**追问**：为什么不是永远把 DP 放最慢链路？DP bucket 也可能很大且 exposed；当 PP/TP 有更好 overlap、DP world size 很大时，DP 可能成为主瓶颈。

### Q118. 为什么两种 mapping 的三类 collective 并发时间几乎相同？

**高分回答要点**

- `tp_local` 与 `tp_cross` 只是把 TP/PP/DP 标签在 PIX/PHB/SYS 三类 pair 间置换；一次三类各一个并发时，物理流量集合基本相同。
- 因此两者分别是 37.46 与 37.42 ms，不能据此说 mapping 无影响。真实训练中三类通信次数、bytes、依赖点和可重叠窗口并不相同。
- 并发 wall time 也不是三个 phase 时间的最小值；共享 PCIe/CPU root、NCCL stream 排队和最慢 peer 都会影响 critical path。
- 正确实验需要构造与真实 schedule 接近的调用频率，记录每个 communicator 的 interval 和依赖，而不是只做对称的一次并发。

**陷阱**：只比较 aggregate bandwidth 会丢失 logical critical path。面试时应主动解释“同物理流量 multiset”这个控制变量。

### Q119. `new_group()` 为什么要求所有进程按一致顺序创建 group？

**高分回答要点**

- group 创建涉及全局 rendezvous 和 communicator 唯一标识分配；即使某 rank 不属于目标 group，也应进入调用并按同一全局顺序执行。
- 若不同 rank 用本地条件分支创建不同顺序，可能让 communicator identity 不一致，表现为初始化 hang、后续 collective 匹配错误或难以复现的并发问题。
- 本实验先生成 canonical TP、PP、DP group 列表，再让所有 8 rank 完整遍历，只有返回 handle 的使用按 membership 决定。
- 多线程创建或多个 overlapping process group 并发 collective 时，还要保证全局一致的执行顺序；业务层 mutex 只能解决本进程线程竞争，不能自动解决跨 rank 顺序。

**现场题**：给一段 `if rank in ranks: new_group(ranks)` 的代码，要求指出为什么危险，并改成所有 rank 都调用、成员只保存 handle。

### Q120. 为什么 collective 调用顺序错配可能静默错误，而不一定 hang？

**高分回答要点**

- communicator 按 sequence 匹配 collective，不知道 Python tensor 叫 A 还是 B；只要 op、shape、dtype 等兼容，两侧第 N 次调用就会彼此配对。
- 本实验 ordered 期望 rank 0 得到 `A=2001,B=4001`；rank 1 反转顺序后两次调用都完成，却得到 `A=3001,B=3001`。
- 这比 hang 更危险，因为 timeout、HTTP 状态和 `loss_is_finite` 都可能正常；需要 logical checksum、sequence logging 或框架层 collective fingerprint。
- 根因定位应比较各 rank 的 group id、sequence number、op、numel、dtype、调用栈和 logical step，而不是先调大 timeout。

**追问**：为什么测试必须让 A/B shape 相同？这是为了隔离“逻辑身份错配”，避免 shape mismatch 先触发显式错误。

### Q121. 如果错配 collective 的 shape 或 op 不同，系统会怎样？

**高分回答要点**

- 可能在 PyTorch/NCCL 参数检查阶段报 mismatch，可能某些 rank 等待超时，也可能因不同 count 导致未定义结果；具体表现取决于 backend、debug mode 和调用组合。
- 不应把“会 hang”背成唯一答案。先说明兼容错配可静默损坏，不兼容错配更容易显式失败，但仍需以日志为准。
- 开启 distributed debug/detail、NCCL debug 和 flight recorder，收集 sequence/fingerprint；在最小 world size 上缩小到第一个分歧调用。
- 修复应回到控制流一致性：相同数据依赖、异常/empty batch 分支、gradient hook 顺序和 communicator 使用，而不是依赖 watchdog 兜底。

**工程追问**：某 rank 因 NaN 跳过 backward，其他 rank 继续 AllReduce，会发生什么？应先全局同步 skip 决策，不能让 rank-local 分支改变 collective 序列。

### Q122. `async_op=True` 返回后，collective 到底完成了吗？

**高分回答要点**

- 返回的是 `Work` handle，通常只表示操作已排入 backend/stream，不表示 GPU 数据可被任意 stream、CPU 或下一个 communicator 安全消费。
- `work.wait()`、CUDA event/stream wait、`torch.cuda.synchronize()` 和跨 rank barrier 的作用不同；应按数据依赖建立最小必要同步。
- `Work` 过早丢失、输出 tensor 被复用、默认 stream 未等待通信 stream，可能造成竞态；过度全局同步又会消灭 overlap。
- benchmark 必须把要测的异步工作真正包含进计时窗口，退出前必须保证所有 peer 的 GPU work 与 communicator 生命周期闭环。

**追问**：为什么 `barrier()` 不能自动替代所有 stream dependency？它解决进程组同步，不等价于为任意用户 CUDA stream 上的数据读写建立正确 event 依赖。

### Q123. FSDP smoke run 已写出正确 JSON，为什么退出时仍会有 NCCL proxy `Close` error？

**高分回答要点**

- 结果正确只证明主训练路径完成，不证明最后一个 default-group GPU collective 已在所有 rank/stream 上完成，也不证明 peer 同时进入 communicator 销毁。
- 本实验末尾收集结果后，非 rank 0 可能更早 `destroy_process_group()`，导致代理连接关闭时仍有 peer work/lifecycle 未闭环。
- 修复为结果写出后 `torch.cuda.synchronize()`，再执行带 `device_ids` 的最终 barrier，最后所有 rank 统一 destroy；复跑日志干净。
- 生产 runner 还要 `wait` 子进程、检查退出码和日志，不可只凭 artifact 存在判定 job 成功。

**4 分答案特征**：能区分 host API return、GPU completion、peer completion、barrier 和 communicator teardown 五个时刻。

### Q124. 多个 process group 并发 collective 时，为什么 stream 顺序也属于正确性协议？

**高分回答要点**

- 每个 communicator 可能使用独立 NCCL stream；同一 rank 的 Python 发起顺序、CUDA stream dependency 与其他 rank 的顺序必须形成一致的 happens-before。
- 跨 communicator 没有天然全局序列。若 overlapping groups 的 rank 以不同顺序发起 blocking collective，可能形成循环等待；async 发起则可能暴露 tensor 生命周期竞态。
- 应定义 deterministic schedule，用 event 连接 producer/collective/consumer，保留 `Work`，并在日志记录 communicator id 和 logical sequence。
- performance 上要区分“允许并发”与“实际重叠”；共享 PCIe 链路可能让并发更慢，应从 trace interval 和端到端 critical path验证。

**追问**：为什么单 rank trace 不足以证明无死锁风险？死锁和最慢 peer 依赖跨 rank 相对顺序，需要分布式对齐日志或多 rank trace。

### Q125. 你会怎样实现一个 topology-aware rank planner 的成本函数？

**高分回答要点**

- 输入包含 GPU-GPU/NIC 带宽与延迟矩阵、NUMA、故障域，以及每类 edge 的 bytes、频率、collective 算法和 exposed fraction。
- 目标可近似为关键路径通信成本加共享链路 contention penalty，再加跨节点/故障域约束；TP、PP、DP、EP 不能用同一个常数权重。
- 先用启发式把高频紧耦合 group 放局部，再局部搜索/模拟；输出必须 deterministic，并附合法性和预估误差。
- 用本机 `tp_local/tp_cross` 做回归 oracle：planner 至少应偏好 PIX TP，并预测 SYS TP 显著更慢；再用真实 run 校准模型。

**边界**：collective microbenchmark 是链路先验，不包含完整 compute overlap 和多流竞争，planner 必须持续用运行 telemetry 校准。

### Q126. FSDP `FULL_SHARD` 的一个 unit 在一次 iteration 中经历哪些状态？

**高分回答要点**

- 稳态存 sharded parameter；forward 前 AllGather 得到完整参数，执行计算，按 reshard policy 释放/重新分片。
- backward 需要参数时可能再次 AllGather，计算参数梯度后 ReduceScatter，把完整 gradient 归约并只保留本 rank shard。
- optimizer 更新本地 shard；临时 full parameter、prefetched next unit、gradient 和 allocator reserved 决定峰值，不是简单 `参数/world_size`。
- root 与 nested unit 的 forward/backward hook 顺序、prefetch policy 和 `use_orig_params` 会改变具体事件数与驻留窗口。

**追问**：为什么 AllGather 与 ReduceScatter bytes 相同也不代表耗时相同？算法、链路方向、并发 peer wait、overlap 窗口和 kernel protocol 都可能不同。

### Q127. 13 个 FSDP unit 为什么 trace 中是 25 次 AllGather 和 13 次 ReduceScatter？

**高分回答要点**

- 本模型 auto-wrap 12 个 `GPTBlock`，外层 root 还管理 embedding/head，共 13 units。
- trace 观察到 forward 路径 13 次参数 AllGather；backward 中 12 个已 reshard block 再聚合参数，所以总 AllGather 为 `13+12=25`。
- 每个 unit 的 gradient 最终做一次 ReduceScatter，因此是 13 次。root-only wrap 则只有 1 次 AllGather/1 次 ReduceScatter。
- 这是当前 graph、wrap policy 和版本的观测，不是 FSDP 永恒公式；activation checkpoint、no-reshard、shared params 或 FSDP2 都可能改变计数。

**现场验证**：要求从 Chrome trace 按 kernel name 分类，并把事件顺序映射回 module pre/post-forward/backward hook。

### Q128. `BACKWARD_PRE`、`BACKWARD_POST` 和 NONE 的差异如何从 trace 证明？

**高分回答要点**

- PRE 在当前 unit backward compute 期间预取下一个 unit 参数，目标是让下一次 AllGather 与 compute 重叠；POST 等当前 compute 后再发，NONE 不主动预取。
- block+PRE 的 NCCL union 85.53 ms，与 POST/NONE 近似，但 overlap 13.08%、exposed 74.34 ms；POST/NONE 本 trace overlap 为 0。
- 稳态吞吐 PRE 比 POST 高 8.8%、比 NONE 高 5.9%，支持“重叠而非通信字节减少”的解释。
- 不能只看 profiler step wall time，因为 instrumentation overhead 很大；绝对吞吐使用非 profiler 三次重复，trace 用于事件结构。

**追问**：PRE 为什么可能更占显存？被预取 unit 的 full params 与当前 unit 的 params/grad/activation 同时驻留，扩大 live set。

### Q129. `limit_all_gathers` 限制的是什么？为什么关闭后可能略快但更占 reserved memory？

**高分回答要点**

- 它是 host-side rate limiter，限制过多 AllGather 提前发出导致 full-parameter allocation 堆积；不是减少模型所需的逻辑 AllGather 次数。
- 本实验关闭后 25/13 次计数不变，吞吐提高 2.5%，reserved peak 从 1.174 增到 1.269 GiB，约 8.1%。
- 小模型/短 run 上调度更激进可能略快；大模型接近容量边界时，额外峰值可能直接 OOM，trade-off 完全不同。
- 观测要同时看 allocated、reserved、inactive split、cudaMalloc retry/OOM 和 timeline 中 in-flight AllGather，不能只看 `nvidia-smi`。

**追问**：为什么 allocated 不变而 reserved 增加？caching allocator 保留了更大的 segment/high-watermark；reserved 不是当前活跃 tensor bytes。

### Q130. root wrap 为什么更快却更占显存？什么时候仍会选 block wrap？

**高分回答要点**

- root wrap 每 step 只有 1 次 AllGather/1 次 ReduceScatter，launch 和 communicator 调度更少；本机 26.1k tok/s，是 block+PRE 的 1.24x。
- 代价是整模型 full params 的驻留窗口更大，allocated peak 0.981 GiB，是 block+PRE 的 1.82x；模型放大后可能不可行。
- block wrap 让 full params 逐层 materialize/reshard，降低峰值并提供 prefetch overlap，但增加 collective 数和小消息/launch 开销。
- 选择依据是容量硬约束、unit 大小、计算窗口、链路、prefetch 峰值和吞吐，不是“越细分片越好”。

**设计追问**：应按 Transformer block、若干 block 一组还是 size-based wrap？做候选 unit 大小 sweep，联合看峰值、collective size distribution 与 exposed time。

### Q131. 怎样正确测 FSDP 峰值显存并解释 allocated/reserved？

**高分回答要点**

- 在初始化、warmup 后明确 reset peak，计量窗口结束同步 GPU，再读取每 rank `max_memory_allocated/reserved`，最后取 max rank。
- allocated 是活跃 tensor allocation 高水位；reserved 是 caching allocator 向 CUDA 保留的 segment 高水位，两者都不含全部 driver/context/NCCL 外部内存。
- 模型初始化、`sync_module_states`、optimizer state 首次 materialization 和 profiler buffer 是否在窗口内，会显著影响口径。
- 生产 OOM 调试还需 memory snapshot、allocation stack、fragmentation、non-releasable memory 和瞬态 rank skew。

**陷阱**：把 `nvidia-smi used` 减去模型参数当 activation，或把 rank 0 峰值当全局峰值，均不可靠。

### Q132. `use_orig_params=True` 改变了什么？为什么 optimizer 构造顺序重要？

**高分回答要点**

- 它向用户暴露原始 parameter 对象视图，支持更灵活的 per-parameter hyperparameter、冻结和某些编译路径；底层仍由 FSDP 管理 sharded storage/view。
- 参数在不同阶段可能呈现 sharded/full view，用户不应缓存不受管理的 data pointer 或在 hook 中假设 shape 永久不变。
- optimizer 应基于 FSDP 包装后暴露的参数构造，确保 state 与 FSDP 管理对象对应；state-dict 保存/加载也要使用对应 FSDP API/上下文。
- 这不自动解决 shared parameter、mixed frozen params 或跨 world-size optimizer reshard，需要按版本文档验证限制。

**追问**：为什么不能只检查 optimizer parameter 数量？对象 identity、flatten/view、state key 和 shard ownership 都可能出错。

### Q133. `sync_module_states=True` 与 device 初始化有哪些底层要求？

**高分回答要点**

- 它从 rank 0 广播 module parameters/buffers，确保各 rank 初始状态一致；通信需要 module 已在 GPU，或通过 `device_id`/`param_init_fn` 正确 materialize。
- 若先在每 rank 用不同 seed 初始化且不同步，loss 有限仍可能训练不同模型；DDP/FSDP 后续 gradient collective 不能修复第一次 forward 的状态差异。
- 大模型常用 meta-device 初始化避免每 rank CPU 完整副本，再由 FSDP materialize/shard；此时初始化顺序、tied weights 和 buffer 都需测试。
- oracle 应比较初始化后跨 rank parameter/buffer checksum，而不是等到 loss 发散才发现。

**追问**：为什么 broadcast 也可能制造启动峰值？rank 0 完整参数、materialization、flatten/shard 和通信 buffer 的生命周期可能重叠。

### Q134. FSDP checkpoint 为什么不能直接沿用单 rank `torch.save(model.state_dict())`？

**高分回答要点**

- FSDP 参数与 optimizer state 是分片/flatten 管理的；full、sharded、local state dict 语义和内存/IO 成本不同。
- full state 在 rank 0 聚合可能造成 CPU/GPU 峰值与串行 IO；sharded state 便于并行写，但恢复到不同 world size 需要 reshard planner。
- checkpoint 还要保存 scheduler/scaler/RNG/data cursor/framework metadata，使用原子 manifest/checksum 防止 partial version 可见。
- 当前实验只做同 world-size 教学协议，没有证明 FSDP/DCP sharded state 或 8-to-4 reshard，面试必须主动说明。

**追问**：optimizer state reshard 为什么比参数更复杂？Adam 有多个 state tensor、参数 key 映射、dtype/flatten metadata 与 ownership 重分配。

### Q135. FSDP rank 0 trace 能证明什么，不能证明什么？

**高分回答要点**

- 能证明该 rank 一个 profiled step 的 NCCL kernel 类别/次数、stream interval，以及按非 NCCL kernel union 定义的局部 overlap/exposed time。
- 不能直接证明跨 8 rank 全局 critical path、网络纯传输时间或整轮稳态吞吐；collective duration 包含 peer wait，profiler 也有 overhead。
- 本实验用三次非 profiler run 报 tokens/s，用 trace 解释 PRE/POST 事件结构；两类证据不能互相替代。
- 更完整做法是多 rank 时间对齐、NVTX 标注 FSDP unit、CPU/CUDA/NCCL trace 与 rank skew，再关联 step max-rank wall time。

**追问**：为什么 `sum(kernel duration)` 也不可靠？同 stream/跨 stream 可能重叠，必须先做 interval union，再算与 compute 的 intersection。

### Q136. 1 GiB KV cache 为什么在 C=16 时不是一开始就 OOM，而是在 decode 中发生抢占？

**高分回答要点**

- cache 容量是 18,720 tokens；16 个 prompt 各 1,024，初始约 16,384 tokens，仍能 admission。
- decode 每轮继续为 active sequence 增加 KV，工作集越过容量后 scheduler 需要释放受害 sequence 的 blocks，并在恢复时重计算其上下文。
- C=8 峰值 KV 65.7% 且无抢占；4 GiB/C=16 容量 74,896、峰值 32.8%，也无抢占；这个双对照支持容量因果。
- OOM、waiting 和 preemption 是不同状态：调度器可通过 admission/抢占避免 allocator OOM，但请求时延和吞吐仍受损。

**手算追问**：忽略 block rounding，`18,720-16,384=2,336` 只够每个请求平均再增长约 146 tokens，远小于目标 512。

### Q137. vLLM preemption 的 recompute、swap 和 reject/admission control 有什么区别？

**高分回答要点**

- recompute 释放某 sequence KV，恢复时重做 prefill/已生成上下文计算，省主机传输但消耗 GPU compute。
- swap 把 KV 移到 CPU/其他层级，恢复时传回，代价受 PCIe/内存容量和实现支持影响；不等价于 prefix cache eviction。
- admission/reject 在请求进入或运行前限制工作集，保护已接收请求 SLO；preemption 是已运行 sequence 的运行时资源回收。
- 比较策略要看 preemption 次数、recomputed/swapped tokens、TTFT/TPOT/E2E、throughput、KV waterline、waiting 和 OOM/reject。

**边界**：本实验直接观测 preemption counter 和性能，未单独解析每个受害请求或重计算 token 数，不能宣称精确的 victim policy。

### Q138. 为什么发生抢占的 1 GiB/C=16 TTFT 反而比 4 GiB/C=16 低？

**高分回答要点**

- 1 GiB mean TTFT 2,601 ms，4 GiB 为 2,931 ms；与此同时 1 GiB throughput 只有 66.0%，mean TPOT 慢 11.0%。
- 小 cache/抢占限制了实际可持续并发，部分 prefill 面临的同时 GPU 竞争更少，可能让首 token 更早；代价转移到被暂停请求和 decode/recompute。
- TTFT 是分布统计，不能说明同一请求 cohort 谁受益；应看 per-request timeline、victim 标记和 p95/p99，而不是只解释均值。
- 正确结论是“单指标有 trade-off 且需要更细 trace”，不是“小 cache 能优化 TTFT”这一因果宣称。

**面试陷阱**：若候选人只用 TTFT 选 1 GiB，说明没有把用户体验、完成率、吞吐和 TPOT 联合建模。

### Q139. 为什么要每 250 ms 抓 metrics time series，而不能只保存 server 结束时快照？

**高分回答要点**

- `requests_running/waiting`、KV usage 是 gauge，结束后通常回到 0；只看 final snapshot 会丢失峰值和饱和过程。
- preemption 是 counter，应对起止或相邻样本做 delta；本实验两次 1 GiB/C=16 都增加 4，其他 case 为 0。
- 250 ms 采样仍可能漏掉更短 gauge spike，因此“观测峰值”是下界；counter 不会因短事件漏计，但要处理 server restart/reset。
- 应保存采样时间、标签、原始 exposition text 和 benchmark 时间边界，避免 parser 把不同 server/case 混在一起。

**追问**：若 scrape 本身影响服务怎么办？降低频率、独立 exporter、测 overhead，并用 counter 与 trace 交叉验证。

### Q140. `max_num_seqs`、`max_num_batched_tokens`、KV 容量和 waiting queue 各控制什么？

**高分回答要点**

- `max_num_seqs` 限制一个调度批次/engine 可并行管理的 sequence 数；它不是 KV 的 token 容量。
- `max_num_batched_tokens` 限制一次 iteration 调度的 token budget，影响 chunked prefill 与 decode 共存、TTFT/TPOT trade-off。
- KV blocks 是跨 iteration 的状态容量；即使 sequence/token budget 合法，长期 decode 增长仍可能触发抢占。
- waiting 表示已接收但当前未运行的请求，来源可以是 sequence/token budget、KV pressure、优先级或调度策略；需要结合 running、KV 和 preemption 才能归因。

**设计追问**：怎样避免 C=16 抢占？增大 KV、降低 admission concurrency、缩短 max output、采用更小 KV dtype/模型并行布局，或基于预计总 tokens 做准入。

### Q141. `num_warps` 为什么会显著改变同一个 Triton RMSNorm 的 registers/thread？

**高分回答要点**

- 一个 program 处理整行 reduction。warp 数改变 lane 间工作划分、部分和数量、shuffle/shared reduction 路径及每线程 live values。
- hidden=8192 时 1 warp 每线程承担更多元素，编译结果 255 registers/thread、384 B stack；8 warps 为 54 registers、无 stack。
- 1 warp kernel 0.5707 ms，8 warps 0.3323 ms，慢 1.72x；这支持 register pressure/并行度机制，但仍需硬件 counter 区分 spill、latency hiding 和 memory stalls。
- hidden=4096 的 2/4/8 warps 性能只差约 0.7%，说明提高理论 occupancy 不保证线性收益，带宽/指令路径可能已饱和。

**追问**：为什么 255 registers 是警示值？它接近架构每线程上限并限制每 SM resident blocks/warps，还伴随 stack，需检查 local-memory spill 指令和实际 occupancy。

### Q142. “理论 occupancy 上界”是怎样算的，为什么不能当 achieved occupancy？

**高分回答要点**

- 用每 block threads/registers/shared memory 分别除 V100 每 SM 的 2,048 threads、65,536 registers、96 KiB shared，再受 32 blocks/64 warps 上限约束，取最小 resident blocks。
- occupancy 上界是 `resident_blocks * warps_per_block / 64`；hidden8192/1warp 为 12.5%，8warps 为 50%。
- 这是静态资源容量模型，不含动态调度、依赖 stall、实际 active cycles、编译器隐藏资源、cache miss 或 block 分布尾部。
- achieved occupancy、eligible warps、stall reason 需要 Nsight Compute hardware counter；即使 achieved occupancy 高，memory-bound kernel 也未必更快。

**现场题**：给 registers/thread、threads/block、shared/block，要求手算三个资源限制并指出哪个是 bottleneck。

### Q143. `cuobjdump` 和 Nsight Compute 分别能回答什么？

**高分回答要点**

- `cuobjdump --dump-resource-usage` 对 cubin 做静态分析，给函数 registers、stack、shared、local 等编译资源；适合无性能计数权限时做资源回归。
- 反汇编还能检查 load/store、local memory、Tensor Core 等指令，但不能告诉执行次数、cache hit、dram bytes 或 stall 占比。
- Nsight Compute 采运行时硬件 counter，可看 achieved occupancy、memory throughput、L2、warp stall、instruction mix；采集有权限和 replay overhead。
- 端到端 trace/Nsight Systems 负责 CPU/CUDA/NCCL 时间线。三类工具回答不同层级，不能互相替代。

**追问**：为什么本实验的 808 GB/s 也不是 Nsight dram throughput？它是按 input read+output write+weight read 一次的最小字节模型除以时间。

### Q144. `ncu --query-metrics` 文本报 `ERR_NVGPUCTRPERM`，退出码却是 0，自动化应怎么处理？

**高分回答要点**

- CLI exit code 只说明命令进程完成，不保证请求的 capability 可用；必须保存 stdout/stderr 并匹配结构化错误或已知诊断文本。
- runner 同时写 `.status` 和完整 log，parser 以 `ERR_NVGPUCTRPERM` 判定 counter 权限失败，不能把 status 0 误记成 profiler 成功。
- CI 应验证预期 metric 列和非空 kernel report；缺 counter 时标记 unsupported/blocked，而不是生成 0 值继续计算。
- 修复需要管理员按 NVIDIA 指引开放 profiling 权限；容器内无权时应诚实降级到 static resource + wall time，并记录证据边界。

**工程追问**：类似问题还会出现在什么地方？编译器 warning、benchmark 跳过全部 case、HTTP 200 内业务 error、测试框架 zero tests collected 都不能只看退出码。

### Q145. 如何设计一套能防守底层追问的 kernel/profile 实验？

**高分回答要点**

- correctness 先行：独立 FP32 oracle、多 shape/dtype/stride/极值、有限值和误差分布；异步 kernel 必须同步后检查。
- 性能控制：预分配输出、warmup/JIT 分离、固定 clocks/负载、CUDA event、足够 iterations/repeats，并报告方差和 launch floor。
- 机制证据：最小 bytes/FLOPs 模型、编译资源、SASS/PTX、hardware counter 和 timeline 分层收集；每个指标注明“估算、静态、实测”之一。
- 反例与边界：warp/stage/shape sweep、资源拐点、无权限/unsupported path、端到端集成收益；不能从单一最快点推广。
- artifact 链：命令与版本 -> raw JSON/log/cubin/trace -> parser -> validation -> report，任何数字都能手工复算。

**4 分答案特征**：能用 hidden8192 的 255 registers、12.5% 理论上界、0.5707 ms 和 `ERR_NVGPUCTRPERM` 同时说明“我们知道什么、为什么这样解释、还缺什么”。

---

## 12. 编码与上机题

### C01. 写一个 topology-aware process-group planner

**要求**：输入 GPU 间带宽矩阵、节点/NIC 信息和 `TP/PP/DP`，输出 rank placement 与各 group。

**考察点**：合法性校验、稳定/确定性输出、把 TP 放快链路的目标函数、非整齐拓扑的错误处理、复杂度与测试。高级追问是加入故障域惩罚和 EP All-to-All 成本。

### C02. 写一个训练显存与通信估算器

**要求**：输入 Transformer 配置、dtype、microbatch、sequence 和并行策略，输出每 rank 参数/梯度/optimizer/activation/KV、collective bytes 和可行性。

**考察点**：组件化公式、GQA、ZeRO/FSDP stage、TP/PP layer 分配、recompute、临时峰值和单位测试。结果应给假设与上下界，不能伪装成精确 allocator 模拟。

### C03. 实现并验证一个 PyTorch DDP communication hook

**要求**：实现 FP16/BF16 压缩或 PowerSGD 风格 hook，正确处理 future、stream、bucket 和误差反馈。

**考察点**：归约后除 world size、dtype 恢复、异步生命周期、跨 rank correctness、数值误差和 benchmark。测试必须比较多 step 参数与 baseline，而不只检查程序不报错。

### C04. 写一个 profiler trace 分析器

**要求**：读取 Chrome trace/结构化事件，按 rank/stream/step 汇总 compute、NCCL、overlap、空洞和 straggler。

**考察点**：区间 union/intersection，不能直接相加并发 duration；时钟对齐、缺失事件、嵌套事件、critical path 近似和 CSV/图表输出。

### C05. 实现一个简化 KV block manager

**要求**：支持 allocate/free、逻辑到物理 block table、prefix 引用计数、copy-on-write 和 LRU eviction。

**考察点**：重复释放、引用泄漏、hash collision、部分 block、OOM 原子性和多线程安全。属性测试应验证“已分配+空闲=总块数”及共享块不被提前回收。

### C06. 实现 continuous batching scheduler 模拟器

**要求**：输入带 arrival time、prompt/output length、priority 的请求，模拟 token budget、chunked prefill、decode、KV 水位和 preemption。

**考察点**：状态机正确、无饥饿、TTFT/TPOT/E2E、吞吐与 goodput；比较 FCFS、decode-priority 和 deadline-aware 策略。高级追问是校准真实 GPU service-time model。

### C07. 写一个 open-loop Poisson 负载发生器

**要求**：按指数 inter-arrival 发送异步请求，记录客户端排队、服务流式 token 时间戳、错误和未完成请求。

**考察点**：不能被 response completion 自节流；seed、clock、连接池、超时、取消、percentile 和 SLO goodput 定义正确。验证实际 arrival distribution 与目标 QPS。

### C08. 实现原子 distributed checkpoint manifest

**要求**：模拟多 rank 分片写入、checksum、失败注入、完成 manifest 与恢复选择。

**考察点**：半写版本不可见、幂等重试、重复 rank、坏 checksum、版本兼容、垃圾回收竞态。高级追问是新 world size 的 reshard plan 与最小读放大。

---

## 13. 按岗位选择题目

不要试图一次背完全部题。先完成通用核心，再按岗位集中训练。

### 13.1 通用核心，所有 AI Infra 候选人

`Q01-Q12`、`Q13-Q18`、`Q27-Q28`、`Q34-Q35`、`Q41-Q52`、`Q55`、`Q64-Q69`、`Q75-Q84`、`Q91-Q99`、`Q104-Q108`、`Q113-Q123`、`Q128`、`Q136`、`Q138-Q144`。

目标：任何一道都能在 2 分钟给结构化回答，5-10 分钟完成公式、图和追问；Q75 的项目介绍应稳定控制在 110-130 秒。

### 13.2 训练系统/Pretraining Performance

重点：`Q13-Q40`、`Q55-Q71`、`Q85-Q86`、`Q89-Q108`、`Q116-Q135`、`C01-C04`、`C08`。

额外准备：optimizer/fused optimizer、distributed checkpoint、FSDP2/DTensor、CP、MoE/EP、多机 RDMA，以及真实训练 loss/MFU 诊断。

### 13.3 推理引擎/Serving

重点：`Q41-Q64`、`Q68`、`Q72`、`Q87-Q88`、`Q90`、`Q109-Q115`、`Q136-Q145`、`C04-C07`。

额外准备：attention backend、quantization kernel、speculative decoding、PD disaggregation、multi-LoRA、cache-aware routing 和容量规划。

### 13.4 Kernel/Compiler/Performance

重点：`Q01-Q12`、`Q55-Q64`、`Q74`、`Q90`、`Q95-Q99`、`Q109-Q112`、`Q141-Q145`、`C03-C04`，并现场写至少一个 Triton reduction/fused op。

额外准备：CUDA programming model、memory coalescing、bank conflict、register spill、warp primitive、PTX/SASS 基础、CUTLASS 和编译器 graph break。

### 13.5 集群平台/Reliability

重点：`Q04-Q08`、`Q25-Q26`、`Q63-Q74`、`Q85-Q90`、`Q96-Q108`、`Q114-Q125`、`Q134`、`Q139`、`Q144`、`C01`、`C04`、`C08`。

额外准备：Kubernetes/批调度、RDMA/RoCE、存储、镜像发布、故障域、elastic recovery、指标/日志/trace 和事故响应。

### 13.6 RL/后训练 Infra

重点：训练与推理两套通用核心，加 `Q25`、`Q41-Q54`、`Q65-Q74`、`Q73`、`Q87`、`Q116-Q140`、`C06-C08`。

额外准备：PPO/GRPO 等算法的数据依赖只需达到能设计系统接口的程度，重点讲 rollout、样本版本、policy staleness、权重同步和异步 backpressure。

---

## 14. 六周面试训练计划

### 第 0 天：建立基线

- 闭卷回答 Q75，并录音；限制 2 分钟。
- 随机抽 20 道通用核心题，每题 3 分钟，按 0-4 分评分。
- 完成 C02 或 C07 的 60 分钟限时版本。
- 记录四个基线：概念正确率、能独立推导比例、引用实测比例、追问后改口/失守比例。

### 第 1 周：硬件、通信与项目叙事

- 完成 Q01-Q12、Q75-Q78；手画 V100 拓扑、ring AllReduce 和 roofline。
- 不看报告复算 PIX/PHB/SYS、ring bytes、DDP efficiency 和 memory accounting。
- 用本仓库原始 CSV/日志随机抽一个数字，要求 5 分钟内追溯到命令、配置和结论。
- 周末模拟：30 分钟项目深挖，目标 Q75≥3 分，其余平均≥2.5。

### 第 2 周：DDP、FSDP 与性能分析

- 完成 Q13-Q26、Q55-Q56、C03-C04。
- 为 DDP 画 backward/bucket/NCCL 两条 stream 的时间线；能解释“累计 NCCL 时间不等于 exposed 时间”。
- 对 7B/70B 各做一张 AdamW 显存表，并给 DDP、ZeRO-2、FULL_SHARD 下界/峰值来源。
- 周末模拟：45 分钟训练轮，目标所有公式题≥3 分，故障题至少能给二分实验。

### 第 3 周：模型并行与大规模训练设计

- 完成 Q27-Q40、Q79-Q81、Q85-Q86、C01-C02。
- 对一个 Transformer block 手推 TP 前后 tensor shape、每层 collective bytes。
- 给 64/256 GPU 各做一个 topology-aware rank mesh；算 PP bubble 并写出 stage imbalance 观测方案。
- 补读 CP、MoE EP 和 distributed optimizer；明确哪些是官方机制、哪些尚未实测。
- 周末模拟：60 分钟训练系统设计，目标先问约束、15 分钟内给出第一版容量表。

### 第 4 周：推理引擎、调度与 SLO

- 完成 Q41-Q54、Q82-Q83、Q87-Q88、C05-C07。
- 对给定模型现场计算 KV bytes；分别为 prefill/decode 画 roofline 与调度关键路径。
- 用纸面或模拟器生成 arrival-rate→TTFT/TPOT/goodput 曲线，能解释 knee 而不只描述图形。
- 设计 mixed-prefix、mixed-length benchmark，列出 cold/warm/off 和 cache isolation 对照。
- 周末模拟：60 分钟推理设计，目标容量估算误差可解释、所有 SLO 都有观测点。

### 第 5 周：Kernel、可靠性与证据缺口

- 完成 Q55-Q74、Q84、Q89-Q90、C08。
- 用 Triton 写并 benchmark 一个 RMSNorm/softmax；至少报告 correctness、HBM bytes 估算、多 shape 和 profiler 指标。
- 做一次 kill-rank/checkpoint 恢复演练，记录 detection、abort、restart、restore 和 loss/data cursor 连续性。
- 纸面演练 NCCL hang、straggler、间歇 OOM 和升级回归，每题要求先保全现场再二分。
- 周末模拟：连续两轮 60 分钟，训练性能一轮、生产事故一轮。

### 第 6 周：岗位定向与全真模拟

- 按目标岗位从第 13 节抽题，每天一轮 45-60 分钟，不提前知道题目。
- 每轮至少包含：项目追问 2 道、公式推导 2 道、故障题 1 道、系统设计 1 道、反例追问 1 道。
- 每次只修复最明显的两个问题，并在下一轮复测同类题；不要用阅读代替输出。
- 最终门槛：通用核心平均≥3.2，岗位重点平均≥3.0，任何安全/正确性题不得为 0，Q75 稳定≥3.5。

---

## 15. 模拟面试评分表

每道重要题可按五个维度各 0-4 分，总分 20：

| 维度 | 0-1 分 | 2 分 | 3-4 分 |
|---|---|---|---|
| 正确性 | 核心错误/只有名词 | 机制基本正确 | 边界、反例和版本差异清楚 |
| 推导 | 无公式/数据流 | 能给主公式 | 能从 shape/bytes/critical path 推导 |
| 观测 | 只看 utilization | 能列常规指标 | 指标能区分竞争假设并定位 rank/phase |
| 实验证据 | 夸大或无数据 | 能引用结果 | 数字准确，说明对照、局限和可复现路径 |
| 生产判断 | 只追求吞吐 | 能讲 trade-off | 包含 SLO、故障、成本、发布和回滚 |

建议判定：

- **17-20**：明显 senior/专项强，能主导设计并防守追问。
- **14-16**：达到多数核心 Infra 岗位的强通过表现。
- **10-13**：概念可用，但推导或生产经验不足，容易在深挖轮失分。
- **0-9**：答案主要依赖背诵，需要回到实验和数据流。

### 面试官统一追问模板

为了避免“背中答案”，模拟面试官在任意题后任选两项：

1. 把 GPU 从 NVLink H100 换成 PCIe V100，结论哪里变？
2. 把 batch/sequence/model size 放大 8 倍，哪个项先成为瓶颈？
3. 给出每 rank tensor shape 和通信 bytes，不要只说 collective 名称。
4. 你会看哪三个指标来区分两个可能根因？
5. 这个结论是实测、文档事实还是推断？如何证伪？
6. 平均值正常但 p99 变差，下一步怎么做？
7. 一个 rank/worker 中途失败，系统状态是否仍一致？
8. 方案提升吞吐 10%，但多占 30% 显存，是否上线？还缺什么信息？

---

## 16. 简历和面试中的证据表达

### 可以直接陈述

- 在 8xV100 PCIe 单节点上建立 NCCL、DDP/FSDP、Megatron 与 vLLM 的可复现实验和 artifact 校验流程。
- 实测拓扑对 512 MiB AllReduce 的影响：PIX 约 11.74 GB/s，PHB/SYS 约 7 GB/s。
- 完成 DDP strong/weak scaling、microbatch 与 CUDA trace 分析；8 卡强扩展 2.82x，并用每 token 梯度 payload 解释效率变化。
- 量化 FSDP 容量/吞吐 trade-off，以及 TP/SP/PP/recompute 的适用边界。
- 构建 open-loop Poisson/SLO benchmark，证明 completed throughput 与 SLO goodput 会在过载区显著分离。
- 通过 DDP reducer trace 量化 bucket/gradient accumulation 的通信次数与 exposed time，验证 `no_sync()` 将 104 个 NCCL kernel 降到 13 个。
- 实现 variable-split MoE All-to-All、原子 checkpoint manifest/failure injection 和 Triton RMSNorm，并为路由、恢复与算子建立独立 correctness oracle。
- 构造 prefix cache 工作集污染实验，证明累计 hit rate 可掩盖关键阶段近 6x 的 p95 TTFT 回归。
- 显式构造 `TP2 x PP2 x DP2` process group 并做 collective-order fault injection，量化 topology-aware mapping，复现 communicator 顺序错配导致的静默数据损坏。
- 用 FSDP rank 0 trace 量化 root/block wrap、backward prefetch 和 all-gather limiter，观测 block wrap 的 25 次 AllGather/13 次 ReduceScatter 与显存/吞吐 trade-off。
- 主动触发 vLLM KV preemption，并通过 metrics time series 关联 cache waterline、running/waiting、TTFT/TPOT 与吞吐；用 cubin resource usage 分析 Triton register/occupancy 上界。
- 诊断 V100/SM70 下 vLLM attention 和 collective backend 回退，并完成真实小模型端到端验证。

### 不应夸大

- 不写“优化最新 vLLM/SGLang”或“生产部署 70B”，本项目是兼容版本上的 synthetic systems benchmark。
- 不写“掌握多机 RDMA/生产 Megatron EP/CP”；本机 MoE 证据是受控 All-to-All toy，不包含真实 router、grouped GEMM 和收敛。
- 不把单节点短跑吞吐称为 time-to-train、模型收敛或生产 SLA。
- 不把 dummy 7B 称为真实 7B 权重质量验证。

完整数字、方法和边界见[实验报告](experiment-report.md)与[工程深挖补充报告](engineering-deep-dive-report.md)。面试前应能从每个简历数字追溯到 summary、raw log 和运行命令。

---

## 17. 官方参考资料

### 当前岗位能力画像

- [OpenAI: Training Performance Engineer](https://openai.com/careers/training-performance-engineer-san-francisco/)
- [OpenAI: Distributed Training Engineer, Sora](https://openai.com/careers/distributed-training-engineer-sora-san-francisco/)
- [OpenAI: ML Systems and Training Architecture](https://openai.com/careers/software-engineer-ml-systems-and-training-architecture-san-francisco/)
- [OpenAI: RL Training Infra](https://openai.com/careers/software-engineer-rl-training-infra-san-francisco/)
- [OpenAI: Platform Systems](https://openai.com/careers/software-engineer-platform-systems-san-francisco/)
- [Anthropic: Careers and current technical role categories](https://www.anthropic.com/careers/jobs?lang=us)

### 训练、通信与性能

- [NVIDIA Megatron Core 0.18 Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.18.0/user-guide/parallelism-guide.html)
- [NVIDIA Megatron Core Distributed Checkpointing](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/api-guide/core/dist_checkpointing.html)
- [PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch `new_group` and process-group ordering](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)
- [PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [TorchTitan FSDP documentation](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVIDIA NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NVIDIA GPU performance-counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
- [Triton Layer Normalization Tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)

### 推理系统

- [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM: Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [SGLang: Benchmark and Profiling](https://docs.sglang.ai/developer_guide/bench_serving)
- [SGLang: Prefill-Decode Disaggregation](https://docs.sglang.ai/backend/pd_disaggregation.html)
- [SGLang: Expert Parallelism](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md)

---

## 18. 最终自检清单

- [ ] 能在 2 分钟讲完项目，并准确说出至少五个关键数字。
- [ ] 能手推 Ring AllReduce、DDP efficiency、PP bubble、KV bytes 和 AdamW memory。
- [ ] 能画 DDP/FSDP/TP/PP/continuous batching 的数据流和关键 collective。
- [ ] 能解释本项目三个负结果：TP 变慢、SP 节省小、DP 非线性。
- [ ] 能解释 completed throughput 与 SLO goodput 分离，并设计 open-loop benchmark。
- [ ] 能用 profiler timeline 区分累计 kernel time、overlap 和 critical path。
- [ ] 能解释 cap 与实际 bucket、`no_sync()` 范围、NCCL channel/transport，以及 collective 中 peer wait 的来源。
- [ ] 能从 manifest/hash/fingerprint 说明语义恢复与逐 bit 恢复的差别，并读懂 collective timeout 日志。
- [ ] 能解释 Triton kernel 的 program mapping、最小字节模型和 launch floor，且不会把有效带宽冒充 HBM counter。
- [ ] 能设计 cache pollution/revisit 对照，并从 Prometheus counter delta 而不是累计 hit rate 判断驱逐。
- [ ] 能手推 `TP2 x PP2 x DP2` rank groups，并解释 `new_group`/collective sequence/stream 生命周期的正确性契约。
- [ ] 能从 25 次 AllGather/13 次 ReduceScatter 还原 FSDP unit 状态机，并解释 PRE、limiter、allocated/reserved 的 trade-off。
- [ ] 能解释 KV preemption 为什么可能让吞吐/TPOT 变差但 mean TTFT 下降，并区分 counter 与 gauge 的采样口径。
- [ ] 能区分 cubin 静态资源、理论 occupancy、achieved occupancy 和最小字节有效带宽。
- [ ] 能为 NCCL hang、straggler、OOM、NaN 和 checkpoint 失败给出可执行二分方案。
- [ ] 能诚实陈述 V100、单机、dummy weights 和 synthetic workload 的证据边界。
- [ ] 能设计 70B 训练/推理方案，但会先询问约束并做容量与通信估算。
- [ ] 至少完成两道编码题和一次限时系统设计，且有测试而非只有主流程。
