# NCCL 专家课程实验结果索引

本文档记录正式实验的脚本、run ID、核心结果和验收状态。冒烟测试和被后续正式实验替代的运行不作为博客定量结论。

## 新版深度课程实验

| 章节 | 正式运行 | 脚本 | 核心结果 | 验收 |
|---|---|---|---|---|
| 01 软件栈与动态链接 | `ch01_stack_version/20260710T065547Z` | `25_ch01_stack_version.sh` | header/runtime/PyTorch 均为 2.22.3；隔离 `LD_LIBRARY_PATH` 使同一 ELF 加载 2.99.99 测试库 | PASS |
| 02 拓扑与 P2P | `ch02_topology/20260710T070553Z` | `26_run_ch02_topology.sh` | 12 个方向 256 MiB median 48.322-48.372 GB/s；disjoint pairs aggregate 96.583 GB/s | PASS |
| 03 异步完成语义 | `ch03_async/20260710T071511Z` | `27_run_ch03_async.sh` | sync API host 74.236 us、stream dependency 3584.432 us；stream-scope 40/40 证明 wait 只阻塞当前 stream | PASS |
| 04 Communicator | `ch04_communicators/20260710T072334Z` | `28_run_ch04_communicators.sh` | 原生 split rank 按 key 重排；重叠 PG、CVD remap 和 duplicate GPU 故障全部验证 | PASS |
| 05 Collective 契约 | `ch05_collective_contracts/20260710T073928Z` | `29_run_ch05_contracts.sh` | 原生 C API 验证 AG/RS rank-offset in-place 和 Reduce 非 root 空指针；PyTorch 14 种 contract 全量逐元素通过 | PASS |
| 06 Group 与顺序 | `ch06_group_ordering/20260710T075642Z` | `30_run_ch06_group_ordering.sh` | GroupEnd 后 CUDA event 0/40 ready；nonblocking init/finalize 完成状态轮询；DETAIL mismatch 与 raw partial-progress/watchdog 分层验证 | PASS |
| 07 Collective 分解 | `ch07_decomposition/20260710T081207Z` | `31_run_ch07_decomposition.sh` | AR 与 RS+AG exact/padding contract 通过；16/256 MiB 稳定点慢 13.1%/10.5%；RS-only steady 64 MiB；Nsight kernel 20 vs 40 | PASS |
| 08 algbw 与 busbw | `ch08_busbw/20260710T082109Z` | `32_ch08_busbw_matrix.py` | 360 条记录正确；N=3 对齐路径验证；独立复算最大误差 algbw 0.00891、busbw 0.00912 GB/s | PASS |
| 09 Benchmark 方法学 | `ch09_benchmark/20260710T150350Z` | `33_ch09_benchmark_methodology.py` | 46 个平衡/镜像配置、920 行正确；I1 对 1 MiB 扰动 7.42%；同核噪声产生 143.94% CV | PASS |
| 10 延迟-带宽模型 | `ch10_latency_model/20260710T151843Z` | `34_ch10_latency_bandwidth_model.py` | 31 size/310 行正确；全局 R2 0.999972 但 MAPE 32.87%；Simple beta 81.426 GB/s | PASS |
| 11 回归分层归因 | `ch11_regression/20260710T152727Z` | `35_ch11_regression_attribution.py` | channel1 大消息保留 19.49% busbw；P2P→SHM 只剩 2.20%；同核 256 KiB CV 208.99% | PASS |
| 12 Ring AllReduce | `ch12_ring/20260710T154018Z` | `36_ch12_ring_steps_channels.py` | 600 行全部正确；四 rank 每 rank 6 send/6 recv；64 MiB ch1→ch12 延迟 4440.330→849.685 us | PASS |
| 13 Tree 与双树 | `ch13_tree/20260710T155522Z` | `37_ch13_tree_ring_hierarchy.py` | 1740 行全部正确；单机 active tree 均为链；N=4 1 GiB Ring/Tree 为 123.45/77.10 GB/s | PASS |
| 14 Simple/LL/LL128 | `ch14_protocol/20260710T161632Z` | `38_ch14_protocol_wire_sync.py` | 1900 行主/对齐记录全正确；1 GiB Ring LL/LL128/Simple 为 42.61/98.37/123.48 GB/s；LL128 非对齐慢约 8% | PASS |
| 15 Channel 与流水 | `ch15_pipeline/20260710T191306Z` | `39_ch15_channels_chunks_buffers.py` | 1060 行全部正确；64 MiB ch1→ch12 为 22.69→118.38 GB/s；4 KiB 请求 12 channels 实际 gridX=1 | PASS |
| 16 Tuning 与插件 | `ch16_tuning/20260710T194200Z` | `40_ch16_tuner_model.py` | 290 行全部正确；源码公式重现 29/29 自动选择；1 MiB auto regret 21.00%；v3 plugin 的 Tree+Simple/gridX=2 路径通过 | PASS |
| 17 Bootstrap 与初始化 | `ch17_init/20260710T200500Z` | `41_ch17_init_state_machine.py` | 100 条阶段计时正确；N=1→4 total median 360→670 ms，kernel init 解释主要增量；非法 ID/重复设备/缺 rank 三类签名通过 | PASS |
| 18 拓扑 XML 与路径 | `ch18_topology_paths/20260711T031500Z` | `42_ch18_topology_xml_paths.py` | 300 行全部正确；NVL/NVB/PIX path matrix 通过；稀疏 NV2 ring 使基础 channel 6→4、64 MiB 慢 39.67%；fixture 物理边界由 x16/x4 反例验证 | PASS |
| 19 Graph Search 与 Channel | `ch19_graph_search/20260711T044500Z` | `43_ch19_graph_search_replay.py` | 360 行全部正确；search/replay/fallback 7/7；自定义 Ring 0-2-1-3 精确执行；基础6与 postset final4/12/16 三层通过 | PASS |
| 20 Transport 与 Connector | `ch20_transport_selection/20260711T061500Z` | `44_ch20_transport_selection.py` | 240 行全部正确、路径8/8；P2P→SHM→NET/Socket fallback 与 connIndex0/1 通过；无 transport 首因通过 | PASS |
| 21 P2P/IPC/SHM 模式 | `ch21_p2p_ipc_shm/20260711T140000Z` | `45_ch21_p2p_ipc_shm_modes.py` | 路径12/12、四卡600行、pair120行正确；direct/CUMEM/IPC/read/CE 与 SHM 三种 CE 精确命中 | PASS |
| 22 Enqueue/Plan/CUDA Graph | `ch22_enqueue_plan/20260711T151500Z` | `46_ch22_enqueue_plan_graph.py` | 190行正确；TRACE work 15行；sequential/grouped/graph Nsight kernel 16/4/20；grouped16 total快54.82% | PASS |
| 23 Device Primitives 与同步 | `ch23_device_primitives/20260711T154500Z` | `47_ch23_device_primitives.py` | 720行正确；TRACE12/12、Nsight9/9、状态模型27/27；Simple buffer64KiB使64MiB busbw 75.78→19.77 | PASS |
| 24 CPU Proxy 与 NET Progress | `ch24_proxy_socket/20260711T170000Z` | `48_ch24_proxy_socket_progress.py` | 150行正确；NET/Socket与Progress/Service/helper线程命中；36个数据面线程pin CPU0使5x64MiB 0.202→3.834s；1ms完成延迟使其增至0.620s | PASS |
| 25 Socket、IB/RoCE 与 Verbs | `ch25_socket_ib_roce/20260711T180000Z` | `49_ch25_socket_ib_roce.py` | Socket fallback 120行正确、路径8/8、interface3/3、IB源码模型29/29；sysfs有mlx4_0但容器无verbs字符设备 | SOCKET/SOURCE PASS；RDMA BLOCKED |
| 26 GPUDirect RDMA 与 MR | `ch26_gdr_registration/20260711T163000Z` | `50_ch26_gdr_registration.py` | 4/4 GPU GDR attr、0/4 DMA-BUF attr；registration60行、参数负对照140行、路径14/14、源码模型30/30；API handle非空但NET注册设备为0 | CAPABILITY/SOURCE PASS；GDRDMA BLOCKED |
| 27 Multi-NIC、Rail 与 PXN | `ch27_multinic_pxn/20260711T170000Z` | `51_ch27_multinic_pxn.py` | 单NIC负对照180行正确、connector18/18、HCA filter3/3、双NIC/Cross-NIC/PXN/merge源码模型39/39 | SINGLE-NIC/SOURCE PASS；MULTI-RAIL BLOCKED |
| 28 CollNet、SHARP、NVLS 与插件 | `ch28_offload_plugins/20260711T173000Z` | `52_ch28_offload_plugins.py` | ABI符号8/8、插件三态3/3、140行负对照正确、路径14/14、强制算法4/4回退Ring、源码模型28/28 | ABI/FALLBACK/SOURCE PASS；OFFLOAD PERF BLOCKED |
| 29 ProcessGroupNCCL、Stream 与 Work | `ch29_process_group_nccl/20260711T183000Z` | `53_ch29_process_group_nccl.py` | cache操作36行、comm创建12/12、Work/Future 160行正确；默认wait 4.9us但GPU未完成，blocking wait 60.95ms；Nsight流分离4/4、源码模型24/24 | PASS |
| 30 DDP Reducer、Bucket 与重叠 | `ch30_ddp_reducer_overlap/20260711T190000Z` | `54_ch30_ddp_reducer_overlap.py` | 320条迭代、1960条bucket-ready；layout 8/8、unused失败1/1；12-bucket四卡均有compute/comm重叠，单bucket均为0；源码模型30/30 | PASS |
| 31 ZeRO/FSDP 状态分片 | `ch31_zero_fsdp_states/20260711T200000Z` | `55_ch31_zero_fsdp_states.py` | 480条迭代、状态因子6/6、数值等价6/6；DDP/ZRO/FSDP collective指纹24/24、源码模型30/30；steady 528.25→144.25MiB | STATE/MEMORY/PATH PASS；FSDP PERF UNSTABLE |
| 32 TP/PP/EP 与多维 Group | `ch32_multidim_parallel/20260711T213000Z` | `56_ch32_multidim_parallel.py` | 336条迭代、2112条通信、864条PP时间线；TP/SP/PP/EP数值与replica通过；Nsight指纹20/20、源码模型39/39；偏斜expert负载112/16 | SHAPE/GROUP/PATH PASS；PERF UNSTABLE |
| 33 Mismatch/Hang/Rank Crash | `ch33_fault_root_cause/20260711T223000Z` | `57_ch33_fault_root_cause.py` | 9/9故障案例、80条因果事件、源码模型37/37；DETAIL三类指纹、raw局部返回、steady watchdog与lazy-init外层deadline分层验证 | ROOT-CAUSE PASS；NETWORK CUT BLOCKED |
| 34 Debug/NVTX/Nsight/Flight Recorder | `ch34_observability/20260711T235500Z` | `58_ch34_observability.py` | 18条计时、32条debug、12条flight、6个kernel与6个NVTX；TUNING大消息proto2但kernel符号仍为RING_LL；源码模型32/32 | EVIDENCE ALIGNMENT PASS |
| 35 生产基线与版本治理 | `ch35_governance/20260712T001500Z` | `59_ch35_production_governance.py` | 15条canary正确；基线116.81GB/s/CV0.81%；channel1回归83.42%，禁P2P退SHM并回归97.81%；policy7/7、模型36/36 | CANARY/POLICY PASS；RUNTIME A/B N/A |
| 36 AllReduce 端到端 Capstone | `ch36_capstone/20260712T003000Z` | `60_ch36_capstone.py` | 64MiB DDP单bucket；预测8/8、调用图35/35、4设备kernel与replica通过；Ring/Simple/12ch/P2P命中；1ch step回归245.93% | END-TO-END PASS |
| 面试进阶实验 | `interview_details/20260717T090915Z` | `61_run_interview_detail_labs.py` | 784条per-rank契约和nccl-tests `#wrong=0`；同80MiB拆分1/10/40/160次顺序关系通过；TP2/4 Decode尺寸对照；FP16取消/溢出反例 | CORRECTNESS/PATH PASS；EAGER REPLAY PERF UNSTABLE |

## 旧版基线实验

以下结果仍然有效，但对应文章需要按 [ARTICLE_STANDARD.md](ARTICLE_STANDARD.md) 补齐源码和实验推导。

| 知识点 | 正式运行 | 脚本 | 核心结果 | 验收 |
|---|---|---|---|---|
| 版本和动态库 | `foundations/20260710T013809Z` | `10_version_matrix.sh` | PyTorch、nccl-tests、系统包均对应 NCCL 2.22.3 + CUDA 12.6 | PASS |
| GPU P2P 矩阵 | `foundations/20260710T013809Z` | `11_p2p_matrix.py` | 12 个方向 median 48.31-48.37 GB/s，最大 CV 0.06%，全部正确 | PASS |
| 异步完成语义 | `foundations/20260710T013809Z` | `12_async_collective.py` | 256 MiB host enqueue 122.84 us，GPU completion 3634.49 us | PASS |
| subgroup communicator | `foundations/20260710T013809Z` | `13_subgroup_communicators.py` | 两个 subgroup 和为 3/7，world 和为 10 | PASS |
| Collective 正确性 | `foundations/20260710T013809Z` | `14_collective_correctness.py` | 七种 collective/P2P 操作在四个 rank 全部通过 reference | PASS |
| 统计性能模型 | `benchmark_statistics/20260710T014347Z` | `16_benchmark_statistics.py` | 小消息 25.86 us；256 MiB 122.38 GB/s；大消息拟合 R2 0.999987 | PASS |
| Algorithm/Protocol | `protocol_algorithm/20260710T014532Z` | `17_protocol_algorithm_sweep.py` | 小消息 Ring+LL 最快，大消息 Ring+Simple 122.51 GB/s | PASS |
| Topology/Graph | `topology_graph/20260710T014639Z` | `18_topology_graph_dump.sh` | 6 条基础 Ring channel、12 coll channels、全部 P2P direct | PASS |
| Tuning 选择 | `tuning_trace/20260710T014751Z` | `19_tuning_trace.sh` | 自动选择 LL -> LL128 -> Simple，与强制矩阵趋势一致 | PASS |
| Nsight 执行链 | `nsys_allreduce/20260710T014759Z` | `20_nsys_allreduce.sh` | 分离 nccl-tests 校验 kernel、NCCL kernel 和 host API | PASS |
| DDP bucket | `ddp_buckets/20260710T015444Z` | `21_run_ddp_bucket_sweep.py` | 1/4/16 MiB cap 均重建为 12x16 MiB；64 MiB 为 3x64 MiB | PASS |
| 故障矩阵 | `fault_matrix/20260710T015721Z` | `22_run_fault_matrix.py` | watchdog、op mismatch、shape mismatch、rank exit 四种签名 | PASS |
| DDP/FSDP 分片 | `sharding_comparison/20260710T020035Z` | `23_run_sharding_comparison.py` | FSDP steady memory 降 78.65%；collective trace 证明 AG/RS | MEMORY PASS / PERF UNSTABLE |
| TP/PP/EP pattern | `parallel_patterns/20260710T020423Z` | `24_run_parallel_patterns.sh` | AR、RS+AG、Send/Recv、AllToAll 在三种 size 全部正确 | PASS |
| P2P/SHM 隔离 | `sweeps/20260709T080909Z` | `09_nccl_param_sweep.py` | 禁 P2P 后 122.56 -> 2.60 GB/s；禁 SHM 基本不变 | PASS |

## 结论使用限制

- FSDP 五个 cycle 的 step-time CV 为 8.83%，超过 5% 阈值。可以使用显存和 collective 路径结论，不把精确 slowdown 百分比作为稳定基线。
- Nsight `.nsys-rep` 和 SQLite 保留在本机，不发布到博客附件；公开的是脱敏后的统计文本。
- nccl-tests JSON 会包含完整进程环境，本课程使用文本原始日志和解析 CSV，避免公开环境信息。
- 所有 GB/s 结果只适用于当前 4xV100 全 NV2、NCCL 2.22.3 环境。

## 硬件阻塞实验

| 知识点 | 缺少条件 | 取得硬件后的验收矩阵 |
|---|---|---|
| IB/RoCE | 至少两台带 HCA 的 GPU 节点 | Socket/IB、GID/MTU、message/rank、median/P95/CV |
| GPUDirect RDMA | 支持 CUDA pointer/DMA-BUF 的 HCA/plugin | GDR on/off、GPU/NIC distance、registration、NIC counters |
| Multi-rail/PXN | 每节点至少双 HCA/port | 单/双 rail、cross-NIC、NUMA affinity、rail utilization |
| CollNet/SHARP | 支持 collective offload 的 fabric/plugin | Ring/Tree/CollNet、node scaling、plugin fallback |
| NVLS | H100/H200 + NVSwitch | Ring/Tree/NVLS/NVLS_TREE、channel、registration、size sweep |
