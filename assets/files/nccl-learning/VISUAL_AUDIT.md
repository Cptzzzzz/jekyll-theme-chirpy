# NCCL 专家课程可视化审计

本清单记录 36 章课程和专题索引的配图决策。技术图必须表达可核对的对象关系、数据流、时序、状态机、内存布局或证据边界；不能用装饰图替代解释，也不能把语义图冒充物理执行路径。

| 页面 | 值得可视化的核心问题 | 图示类型 | 状态 |
|---|---|---|---|
| 专题索引 | 36 章从运行时、算法到生产治理的依赖关系 | 分阶段学习地图 | done |
| 01 版本边界 | 编译版本、ELF SONAME、动态加载与运行库证据链 | 动态链接链路 | done |
| 02 拓扑与 P2P | 物理拓扑、NCCL 图模型、路径等级与实测带宽的边界 | 拓扑建模管线 | done |
| 03 异步完成 | 用户 Stream、NCCL Stream、Event、Host wait 的先后关系 | Stream/Event 时序 | done |
| 04 Rank 与 Communicator | 五层身份和 communicator 创建、split、destroy 生命周期 | 对象关系与生命周期 | done |
| 05 Collective 契约 | 七类 collective 的输入输出和 shard 布局 | 数据流图组 | done |
| 06 Group 与顺序 | communicator 全局序列、Group launch 和异步错误 | 顺序与状态机 | done |
| 07 Collective 分解 | AllReduce 与 RS+AG 的语义、kernel 和显存差异 | 分解数据流 | done |
| 08 algbw/busbw | payload、report bytes、链路流量与报表系数 | 字节口径换算图 | done |
| 09 Benchmark | warmup、barrier、timed region、completion 和统计窗口 | 测量时序 | done |
| 10 延迟带宽模型 | 延迟区、过渡区、饱和区和 Tuner 候选选择 | 分段模型管线 | done |
| 11 回归归因 | 应用、NCCL、transport、topology、system 五层指纹 | 分层归因树 | done |
| 12 Ring AllReduce | ReduceScatter、AllGather 的逐步 chunk 流动 | Ring step 图 | done |
| 13 Tree 与双树 | up/down 方向、双树分流和层次化构图 | 双树结构图 | done |
| 14 Simple/LL/LL128 | 三种线格式、payload/flag 和同步粒度 | 线格式对比 | done |
| 15 Channel/Chunk/Slice | message 到 channel、chunk、slice、step、buffer、block | 粒度分解图 | done |
| 16 Tuning/Plugin | topology graph 到候选代价，再到 plugin override | 选择流水线 | done |
| 17 Bootstrap | rendezvous、AllGather、topology、connect、success 状态 | 初始化状态机 | done |
| 18 XML 与路径 | 物理设备到 XML、节点/边，再到 BFS path | 路径计算管线 | done |
| 19 Graph Search | speed 档位、递归占边、回退和 postset | 回溯状态机 | done |
| 20 Transport | P2P/SHM/NET first-match 与 connector 生命周期 | 选择与生命周期 | done |
| 21 P2P/IPC/SHM | direct pointer、IPC、read/write、SHM copy engine | 数据路径对比 | done |
| 22 Enqueue/Plan | API、task、plan、work batch、upload、kernel/graph | Host enqueue 管线 | done |
| 23 Device Primitive | block/channel、step credit、data/flag 发布 | device 状态机 | done |
| 24 CPU Proxy | GPU、proxy progress、socket helper 三平面协作 | producer/consumer 时序 | done |
| 25 Socket/IB/RoCE | Verbs 对象、QP 建连、CTS、RDMA Write、CQ | Verbs 数据面 | done |
| 26 GDR/MR | GDR read/write、host bounce、MR 两级缓存 | 路径与缓存图 | done |
| 27 Multi-NIC/PXN | rail、cross-NIC、channel-NIC 配对和 PXN 中继 | 多 NIC 路径图 | done |
| 28 CollNet/NVLS | plugin/ABI/device/topology 门控和安全 fallback | 能力门控图 | done |
| 29 ProcessGroupNCCL | Python op、comm cache、Stream/Event、Work/watchdog | 双向 Stream 依赖 | done |
| 30 DDP Reducer | autograd hook、bucket ready order 和计算通信重叠 | backward 时间线 | done |
| 31 ZeRO/FSDP | parameter、gradient、optimizer 在 stage 0-3 的归属 | 状态分片矩阵 | done |
| 32 TP/PP/EP | 多维 rank 坐标、矩阵切分、pipeline 和 token dispatch | 并行维度地图 | done |
| 33 故障根因 | traceback、fingerprint、watchdog、NCCL error 的因果优先级 | 故障证据树 | done |
| 34 Debug/Nsight | op key 对齐日志、flight recorder、NVTX、kernel | 证据对齐图 | done |
| 35 生产治理 | fingerprint、canary、判定、分批发布、回滚 | 发布状态机 | done |
| 36 端到端 Capstone | autograd 到 NVLink 再到 optimizer 的完整链路 | 端到端调用图 | done |

## 验收标准

1. Mermaid 页面必须设置 `mermaid: true`，公式页面继续保留 `math: true`。
2. 图前后必须说明抽象层级及结论边界。
3. Mermaid 必须在真实 Chromium 中生成 SVG，页面不能出现 `Syntax error in graph`。
4. 数学页面必须保持 `mjx-merror = 0`。
5. 桌面页面不得横向溢出；移动端宽图使用图内横向滚动，标签保持可读。
6. 图中文字、rank、方向、count、阶段名必须与文章正文和固定源码版本一致。
