# Megatron 专家面试题库：从架构到大规模训练优化

> 版本日期：2026-07-20。
>
> 本题库同时使用两类证据：本仓库固定的 Megatron-Core `core_v0.9.0`（commit `1afee592e85ac7994887eb5f4ef3998f76384333`）实测，以及 NVIDIA 当前 Megatron-Core 官方文档。凡涉及 CP、生产 MoE、Megatron-FSDP、新版 distributed checkpoint、FP8 等未在本机跑过的能力，均标记为“官方机制/设计题”，不能包装成本机经验。

## 1. 使用方法与专家标准

### 1.1 你需要建立的能力树

| 层级 | 必须掌握 | 面试官验证方式 |
|---|---|---|
| 架构 | Megatron-LM、Megatron-Core、Transformer Engine 的边界；Core 包结构 | 让你从启动脚本画到一个 Transformer layer |
| 张量 | 每种并行下的参数/activation/gradient shape | 临场手推 Column/Row Parallel、SP、CP、EP |
| 通信 | group、collective、bytes、调用频率、关键路径 | 追问 forward/backward 分别通信什么、能否 overlap |
| 调度 | microbatch、1F1B、interleaving、VPP、stage layout | 给 PP/M 要求算 bubble 并定位不均衡 |
| 状态 | contiguous grad buffer、distributed optimizer、checkpoint reshard | 追问 optimizer state 所有权与恢复配置变化 |
| 性能 | MFU、GEMM shape、fusion、precision、topology、straggler | 给 profile 让你提出可证伪优化实验 |
| MoE | router、dispatch、GroupedGEMM、EP、负载均衡、通信 overlap | 从 token 路由一路追到 All-to-All 和尾部专家 |
| 工程 | 初始化、RNG、数据、日志、升级、故障、correctness | 问“为什么没有报错但结果错了” |

### 1.2 每题统一回答模板

每道题按以下顺序组织。正文中的三条“高分回答要点”不是孤立关键词，而是从定义、机制到工程判断的递进参考回答。

1. **前置知识**：回答前必须会的公式、对象和框架接口。
2. **高分回答要点**：先给定义，再推数据流，最后给工程 trade-off 和证据边界。
3. **分析过程**：说明现场应如何从约束推导答案，而不是直接背结论。
4. **主问题得分点**：总分 10 分，明确基础、机制、工程和边界各占多少分。
5. **追问**：面试官用于判断答案是否背诵的问题。
6. **追问分析与参考回答**：同样从对象、公式/数据流和工程验证逐步回答。
7. **追问得分点**：总分 5 分；只有结论、没有推导最多得 2 分。

现场作答仍应遵循六步：先画 rank group、tensor shape、module/stage 和状态所有权；再走 forward/backward/optimizer/checkpoint 数据流；写 collective、bytes、频率和显存；找 critical path 和 overlap；给指标、对照与 correctness oracle；最后说明版本和证据边界。

### 1.3 评分

主问题统一按 10 分评价：

- **0-2 分**：只知道参数名或核心方向错误。
- **3-4 分**：能给定义，但推不出 tensor、通信或状态。
- **5-6 分**：数据流和主公式正确，工程细节不足。
- **7-8 分**：能讨论关键路径、指标、实验和常见反例。
- **9-10 分**：能落到源码/runtime、版本边界和生产决策，并能证伪自己的判断。

追问统一按 5 分评价：对象/前提 1 分，推导或数据流 2 分，工程验证 1 分，反例/边界 1 分。追问答错底层语义时，主问题即使背得完整也不应超过 6 分。

### 1.4 本机证据速查

| 实验 | 核心数字 | 可支持的回答 |
|---|---|---|
| TP1 -> TP8 | 84,626 -> 14,667 tok/s；显存 2.85 -> 0.473 GiB | PCIe 上 TP 首先解决容量；小 GEMM 与层内通信会吞掉算力收益 |
| TP2 SP | 显存省 3.0%，吞吐降 15.9% | SP 收益取决于可切 activation 占比和互连 |
| TP4 SP | 显存省 9.3%，吞吐降 19.3% | 不能把 SP 当免费显存优化 |
| seq4096 SP | 仅省 1.3%-3.8% | 长序列不保证 SP 相对收益单调增加 |
| seq4096 full recompute | 显存省 77.6%-78.1%，吞吐降 27%-30% | SP 与 recompute 优化不同张量集合 |
| PP8/M1 | bubble 87.5%，仅 9,627 tok/s | 少 microbatch 时 PP 可比单卡更慢 |
| PP8/M16 | 42,522 tok/s，PP1 的 2.59x，效率 32.4% | bubble 降低不等于线性扩展 |
| PP8 stage | 参数 max/min=2.76x | 等层切分会被 embedding/output endpoint 打破 |
| 进程组 mapping | SYS TP phase 是 PIX 的 3.05x | rank mapping 必须服从拓扑和通信频率 |

---

## 2. Level 0：Megatron 全局架构（M001-M012）

### M001. Megatron-LM、Megatron-Core 和 Transformer Engine 分别是什么？

**高分回答要点**

- Megatron-LM 是完整训练应用与参考 recipes，包含参数解析、数据、训练循环和模型入口；Megatron-Core 是可复用的模型并行训练库。
- MCore 提供 transformer/model、TP/PP/CP/EP、distributed、optimizer、checkpoint 等核心抽象；Transformer Engine 提供 NVIDIA 优化 layer、attention、FP8 和 fusion 路径。
- “用了 Megatron”必须说清是跑 `pretrain_gpt.py`、调用 MCore API，还是只复用了并行 linear；三者工程责任不同。

**追问**：本项目直接调用 `GPTModel`、`TransformerConfig` 和 `get_forward_backward_func()`，属于 MCore API 实验，不是完整 Megatron-LM 数据训练 recipe。

**前置知识与分析过程**：先区分“应用、训练库、加速后端”三个层级，再沿一次训练调用说明谁负责数据和训练循环、谁负责模型并行、谁提供优化 kernel。最后把自己的项目映射到其中一层，避免用仓库名称替代真实调用范围。

**主问题得分点（10分）**：说清三者定义 3 分；能画调用/依赖关系 3 分；能说明接入和调试责任 2 分；准确陈述本机只直接使用 MCore API 的边界 2 分。

**追问分析与参考回答**：从 import 和控制权判断：脚本自己初始化 `torch.distributed`、构造 synthetic batch、optimizer 和训练循环，只把模型、parallel state 和 PP schedule交给MCore，因此属于“直接集成MCore API”；没有使用Megatron-LM完整参数/数据/训练入口，也没有走TE spec。

**追问得分点（5分）**：指出直接调用对象 1 分；说明训练循环归属 2 分；区分 Megatron-LM recipe 与 TE 路径 1 分；不夸大版本/能力 1 分。

### M002. 请画出 Megatron-Core 的核心 package 地图。

**高分回答要点**

- `parallel_state` 管 group/rank；`tensor_parallel` 管分片 layer/mapping/RNG；`pipeline_parallel` 管 schedule/P2P。
- `transformer` 用 config/spec 组装 layer；`models.gpt` 提供 GPTModel；`distributed/optimizer` 管梯度和模型状态。
- `dist_checkpointing` 管 sharded state 与 reshard；`datasets` 管训练样本；`transformer.moe` 管 router/dispatcher/experts。

**追问**：给一个 loss 异常，要求你说先看 `GPTModel -> TransformerBlock -> TransformerLayer -> Attention/MLP` 哪一层。

**前置知识与分析过程**：先按“全局状态、模型、并行运行时、优化器、持久化”分类 package，再从 loss 反向沿 module tree 找第一个异常 tensor，而不是从目录逐个猜。要同时指出 TP/PP 通信位于哪些包。

**主问题得分点（10分）**：核心 package 与职责 4 分；能画模型调用链 2 分；能画 parallel/optimizer/checkpoint 横向依赖 2 分；给出异常定位顺序 2 分。

**追问分析与参考回答**：先用 loss/logits 判断末端，再在每层挂 hook 或 NVTX 比较 TP1 reference；若进入某层前正常、attention 后异常，就下钻 QKV、attention kernel 与 output projection；若单层都正常，再查 vocab-parallel loss、gradient finalization 和 optimizer。

**追问得分点（5分）**：给出二分位置 1 分；逐层 tensor oracle 2 分；关联 TP collective 1 分；能区分 forward、loss 与 optimizer 问题 1 分。

### M003. `parallel_state.initialize_model_parallel()` 的职责是什么？

**高分回答要点**

- 根据 TP/PP/CP/EP 等维度创建 process groups，维护 global rank 到各维 local rank/world size 的映射。
- 为 tensor、pipeline、data、context、expert 及 embedding 等特殊同步提供 group getter。
- 它建立通信拓扑，不创建模型分片本身；模型 module 根据 config 和当前 rank 决定本地参数/层。

**追问**：为什么 group 创建顺序必须跨 rank 一致？communicator identity 与 collective sequence 不认识业务 tensor 名称。

**前置知识与分析过程**：先列 world rank 与 TP/PP/DP 等 local rank，再说明 `initialize_model_parallel` 如何建立 communicator。分析时区分“创建 group”“保存 handle”“执行 collective”三个阶段。

**主问题得分点（10分）**：group/rank映射 3 分；getter及特殊group职责 2 分；说明模型分片与group初始化的边界 2 分；创建/销毁和顺序风险 3 分。

**追问分析与参考回答**：所有rank必须以相同全局顺序参与 `new_group`，否则同一创建序号可能得到不同成员语义。后续NCCL只按communicator和sequence匹配；即使shape相同也可能静默串错数据，shape不同则可能报错或hang。工程上应使用canonical group plan并记录group id/sequence。

**追问得分点（5分）**：创建顺序 1 分；communicator sequence机制 2 分；静默错误反例 1 分；canonical plan/日志方案 1 分。

### M004. 如何从总 GPU 数推导 DP、TP、PP、CP？EP 为什么不能总是简单相乘？

**高分回答要点**

- dense 常见关系为 `world_size = DP x TP x PP x CP`，每个 rank 在这些正交维上有坐标。
- EP 替换/细分 MoE 区域的数据并行域，且 dense 参数与 expert 参数的 DP group 可能不同；还可能有 expert TP。
- 必须用具体 `parallel_state` group 定义推导，而不是机械写 `x EP`。

**追问**：给 64 GPU、TP4/PP4/CP2，DP=2；再加入 EP8 时要求画 dense-DP 与 expert-DP group。

**前置知识与分析过程**：先对dense模型用正交mesh计算坐标，再单独处理MoE参数所有权。不要在不知道EP与expert-DP定义时机械乘法；应先问expert数量、expert TP、EP嵌入哪个数据域。

**主问题得分点（10分）**：dense公式与DP结果 3 分；能画任一rank四维坐标/group 3 分；说明EP非简单正交的原因 2 分；提到dense/expert参数不同同步域 2 分。

**追问分析与参考回答**：64卡下dense部分TP4xPP4xCP2占32卡一个model replica，DP2。加入EP8时不是把总数变512，而是在MoE层把expert分布到既有rank域；必须依据MCore parallel-state布局确定同一expert在何处复制，从而形成expert-DP。画图时分别标dense attention参数的DP伙伴和expert参数的expert-DP伙伴。

**追问得分点（5分）**：算出DP2 1 分；拒绝机械乘EP 1 分；区分两类参数owner 2 分；说明需核对具体group实现 1 分。

### M005. `TransformerConfig` 为什么不只是一个超参数 dataclass？

**高分回答要点**

- 它同时描述模型 shape、精度、并行度、重计算、fusion 和通信行为，是 module 构造与 runtime 的共同契约。
- 很多合法性依赖是组合约束，例如 hidden/head 对 TP 的可整除性、PP layer 分配、SP 与某些 TP/EP 组合。
- 配置变化可能改变参数 layout、kernel、checkpoint metadata 和数值路径，必须版本化保存。

**追问**：为什么恢复 checkpoint 时只保存权重、不保存训练配置可能造成静默语义变化？

**前置知识与分析过程**：把config字段分为模型数学语义、并行layout、数值精度和性能开关四类。逐类判断改变后是shape不兼容、可reshard、数值路径变化还是仅性能变化。

**主问题得分点（10分）**：说明config是runtime契约 3 分；举出至少三个组合约束 3 分；关联checkpoint/layout 2 分；说明配置版本化和校验 2 分。

**追问分析与参考回答**：权重shape可能仍能加载，但RoPE scaling、norm、attention类型、loss精度或shared embedding配置变化会改变函数；TP/PP改变需要reshard；fusion/precision改变归约顺序。checkpoint应保存并校验模型config fingerprint，对允许变化的并行字段走显式reshard，对数学语义变化默认拒绝。

**追问得分点（5分）**：给出静默语义例子 2 分；区分可reshard与不可变字段 1 分；config fingerprint 1 分；load时fail-fast 1 分。

### M006. `ModuleSpec` 解决了什么架构问题？

**高分回答要点**

- spec 把 Transformer 组件的结构描述与具体实现解耦，可替换 attention、MLP、norm、linear 等子模块。
- 同一 GPTModel 可选择 local PyTorch 实现、Transformer Engine 实现或自定义模块，而不重写整个模型。
- spec 还承载 submodule wiring 和参数，错误替换会影响 checkpoint key/layout、fusion 和并行语义。

**追问**：如何安全接入一个自定义 attention？先实现接口/shape，再做单层 reference、TP 多 rank correctness、checkpoint 和性能回归。

**前置知识与分析过程**：先理解spec描述“构造哪个module及其submodules”，config描述shape/行为。接入时按接口兼容、并行layout、状态持久化、数值、性能五层验证。

**主问题得分点（10分）**：spec作用 3 分；与config/module边界 2 分；checkpoint/并行风险 2 分；完整接入测试矩阵 3 分。

**追问分析与参考回答**：先在TP1用同权重与reference比较forward/backward；再TP2验证QKV/head shard、output projection和grad；测试dropout RNG、recompute与PP；做state-dict往返和旧checkpoint迁移；最后比较kernel、peak、tokens/s并提供feature gate/fallback。

**追问得分点（5分）**：接口/reference 1 分；TP/PP correctness 2 分；checkpoint 1 分；性能与fallback 1 分。

### M007. `GPTModel` 中 `pre_process`、`post_process` 和 `parallel_output` 各有什么作用？

**高分回答要点**

- PP 首 stage `pre_process=True` 负责 embedding/input；末 stage `post_process=True` 负责 output/loss，内层 stage 只持 Transformer block。
- `parallel_output=True` 保持 logits 在 TP vocab 维分片，避免为完整词表做昂贵 AllGather。
- tied embedding/output 跨 PP 首尾时需要额外权重与梯度同步，不能只按普通 layer 参数处理。

**追问**：本机 PP 脚本为何根据 `is_pipeline_first_stage/last_stage` 设置这两个 flag？

**前置知识与分析过程**：先画无PP的embedding->blocks->output/loss，再沿深度切到stages。对每stage列本地module、输入类型和输出layout，最后判断logits是否需要在TP维gather。

**主问题得分点（10分）**：pre/post定义 3 分；stage-local模型与数据流 3 分；parallel_output语义 2 分；tied embedding特殊同步 2 分。

**追问分析与参考回答**：first stage需要token/position并构建embedding，所以`pre_process=True`；last stage需要output layer和loss，所以`post_process=True`；中间stage只接收activation并运行本地blocks。错误地全开会在每rank重复端点参数和计算，且破坏P2P接口。

**追问得分点（5分）**：first/last职责 2 分；中间stage输入 1 分；解释错误后果 1 分；联系stage参数不均衡 1 分。

### M008. 一次完整 Megatron 训练 iteration 有哪些阶段？

**高分回答要点**

- 取 batch/切 microbatches，执行选定 PP schedule 的 forward/backward，产生本地/分片 gradient。
- 完成 TP/SP/CP/embedding/expert 等特殊梯度同步，再做 DP AllReduce 或 ReduceScatter。
- unscale/overflow、global norm/clip、optimizer step、parameter AllGather、scheduler、日志和 checkpoint。

**追问**：为什么“backward 返回”不代表所有通信已经完成？异步 bucket 和 param gather 需要显式 finish/wait。

**前置知识与分析过程**：按iteration时间轴列数据、F/B、模型并行grad finalize、DP sync、optimizer和param sync。对每步注明host调用返回、CUDA stream完成和跨rank完成不是同一时刻。

**主问题得分点（10分）**：完整iteration阶段 4 分；特殊gradient同步 2 分；mixed precision/optimizer顺序 2 分；异步生命周期与checkpoint/logging边界 2 分。

**追问分析与参考回答**：backward hook可能只是把NCCL work排到通信stream，Python autograd返回时GPU仍在执行。optimizer读取main_grad前要`finish_grad_sync`；下一轮forward使用分片更新参数前要等待param AllGather。计时和销毁process group前也要闭环所有Work/event。

**追问得分点（5分）**：host/GPU区别 1 分；grad wait位置 1 分；param wait位置 1 分；计时/teardown风险 2 分。

### M009. 为什么 Megatron 自己管理 microbatch calculator？

**高分回答要点**

- 需要满足 `global_batch = micro_batch x num_microbatches x DP`，并协调 PP schedule 与 gradient accumulation。
- ramp-up batch 会让 global batch 随训练阶段变化，calculator 决定当前 microbatch 数。
- 数目变化会同时改变 bubble、activation 驻留、DP 同步频率和优化语义。

**追问**：TP 不应直接出现在 global batch 公式中；本项目通过改变 accumulation 保持不同 TP 下 global batch 公平。

**前置知识与分析过程**：先区分样本复制维DP和模型分片维TP/PP/CP，再写batch恒等式。分析M变化时同时检查优化语义、PP bubble、activation和step latency。

**主问题得分点（10分）**：batch公式 3 分；calculator/ramp-up职责 2 分；与PP schedule关系 2 分；性能与训练语义trade-off 3 分。

**追问分析与参考回答**：`GBS=MBS x accumulation x DP`。TP只把同一样本的模型计算分到ranks，PP处理同一样本不同层，CP处理同一序列不同chunk，均不增加独立样本。项目中DP从8降到1时把accum从1增到8，从而保持GBS不变。

**追问得分点（5分）**：正确公式 2 分；解释三种模型并行不计样本 2 分；给本机实例 1 分。

### M010. 为什么 model-parallel RNG tracker 是必要的？

**高分回答要点**

- dropout 等操作既要在该复制的维度保持一致，又要在不同数据样本/分片上保持正确独立性。
- TP/SP 下 activation layout 改变，普通全局 seed 容易让各 rank 生成错误的相同或不同 mask。
- checkpoint/recompute 必须恢复 RNG 状态，确保重算 forward 使用与原 forward 一致的随机数。

**追问**：为什么 activation checkpoint 只重跑算子但不处理 RNG 会造成梯度错误？dropout mask 改变了函数本身。

**前置知识与分析过程**：先区分data-parallel seed和model-parallel RNG stream，再列dropout在TP/SP layout上的复制/分片要求。随后分析checkpoint重放需要恢复哪个时刻的RNG状态。

**主问题得分点（10分）**：RNG tracker目的 3 分；TP/SP随机性语义 3 分；recompute状态恢复 2 分；checkpoint/可复现性 2 分。

**追问分析与参考回答**：原forward保存的是使用mask A的activation函数，backward重算若使用mask B，相当于对另一个函数求导。checkpoint wrapper必须保存并恢复CPU/CUDA及model-parallel RNG stream，重算后再恢复外部RNG进度，既保证mask相同又不重复消耗后续随机数。

**追问得分点（5分）**：指出函数改变 2 分；保存/恢复时序 2 分；外部RNG进度 1 分。

### M011. Megatron-Core 的“模型构建”和“训练应用”边界在哪里？

**高分回答要点**

- MCore model API 构建本 rank module 并定义 forward；应用层负责 tokenizer/data iterator、loss closure、optimizer/scheduler、checkpoint 和 job orchestration。
- pipeline schedule 要求应用提供 `forward_step_func`，返回 output 与 loss function，体现 control inversion。
- 专家能区分 library bug、recipe/config bug 和数据/集群 bug，定位入口不同。

**追问**：本机 PP benchmark 为什么自己提供 `forward_step`/`loss_func`，但 schedule 由 MCore 提供？

**前置知识与分析过程**：从dependency inversion理解：schedule只应掌握并行时序，应用掌握数据和任务loss。回答时列双方输入输出与错误归属。

**主问题得分点（10分）**：library/application边界 3 分；callback契约 3 分；故障分类 2 分；结合本机脚本 2 分。

**追问分析与参考回答**：MCore schedule知道stage、microbatch和P2P，但不知道batch字段和业务loss；脚本的`forward_step`从iterator取tokens并调用stage model，`loss_func`只在last stage解释输出。这样同一schedule可服务GPT/T5等任务，同时应用承担数据正确性。

**追问得分点（5分）**：schedule职责 1 分；callback职责 2 分；last-stage loss 1 分；抽象复用价值 1 分。

### M012. 如何描述本项目的 Megatron 版本边界？

**高分回答要点**

- 直接实测只覆盖 `core_v0.9.0` 的 local GPTModel、TP/SP、full recompute 和非交错 1F1B。
- 当前官方 MCore 还包含 CP、Megatron-FSDP、生产 MoE、新 distributed optimizer/checkpoint、custom PP layout、FP8 等演进能力。
- 可以用新版文档回答设计和源码题，但不能声称这些路径在 V100 上跑过或具有报告中的性能。

**追问**：面试中如何说版本？先报 tag/commit/hardware，再说哪些机制跨版本稳定、哪些 API/feature 需要复核。

**前置知识与分析过程**：把陈述分为“本机直接实测、官方当前能力、机制推断”三列。先给可复现版本，再逐功能标证据等级，不以当前文档反向改写旧实验。

**主问题得分点（10分）**：准确版本/commit 2 分；列出实测范围 3 分；列出未实测新版能力 2 分；说明跨版本稳定机制与API差异 3 分。

**追问分析与参考回答**：可回答：“我在8xV100上固定core_v0.9.0实测local GPT的TP/SP、full recompute和非交错1F1B；当前MCore文档中的CP、Megatron-FSDP、生产MoE和FP8我按架构与源码学习，但没有本机性能证据。Column/Row、1F1B等数学机制稳定，具体flag、约束和checkpoint格式按目标commit复核。”

**追问得分点（5分）**：tag/hardware 1 分；实测/未实测区分 2 分；稳定机制 1 分；目标版本复核 1 分。

---

## 3. Level 1：模型抽象与训练数据流（M013-M024）

### M013. 一个 `TransformerLayer` 的主数据流是什么？

**高分回答要点**

- 典型 pre-norm 层是 norm -> self-attention -> residual，再 norm -> MLP -> residual；具体由 spec/config 决定。
- TP/SP/CP 会改变每个边界上的 activation layout，不能只画单卡算子图。
- backward 反向经过同一依赖，并在 parallel linear、attention/context 和 DP buffer 处触发不同通信。

**追问**：让你标出每个 residual 前后的 shape 和是否在 TP rank 间复制。

**前置知识与分析过程**：先确定张量布局记号，例如`[s,b,h]`及SP时`[s/t,b,h]`，再沿norm、attention、residual、MLP逐边标layout。每经过parallel linear都检查是否发生collective或保持分片。

**主问题得分点（10分）**：基本layer顺序 2 分；输入输出shape 2 分；TP/SP/CP layout变化 3 分；backward和通信位置 2 分；spec可变性 1 分。

**追问分析与参考回答**：无SP时residual主干通常在TP ranks复制为`[s,b,h]`；QKV column-parallel后内部head分片，output row-parallel归约回复制hidden再加residual。开SP时norm/residual可保持sequence shard`[s/t,b,h]`，在需要column-parallel输入前AG、row-parallel输出后RS。具体边界依版本实现核对trace。

**追问得分点（5分）**：给shape 1 分；无SP复制语义 1 分；SP shard 1 分；AG/RS位置 1 分；说明实现边界 1 分。

### M014. Megatron attention 中 Q/K/V 如何随 TP 分片？

**高分回答要点**

- 通常按 attention heads 或投影输出维分到 TP ranks，每 rank 计算本地 heads 的 Q/K/V 和 attention。
- output projection 用 row-parallel 汇合各 rank head contribution，forward 需要归约或 SP 下的 ReduceScatter。
- head 数、query groups 与 TP 必须满足 layout 约束；GQA/MQA 可能涉及 KV group 复制或不同分片策略。

**追问**：给 `hidden=8192, heads=64, TP=8`，每 rank query heads=8；再把 KV groups 改成 8，讨论每 rank KV ownership。

**前置知识与分析过程**：先计算head_dim=`hidden/heads`，再按Q heads与KV groups分别判断可整除性和owner。最后检查QKV projection输出shape与attention kernel支持。

**主问题得分点（10分）**：QKV/head分片定义 3 分；output projection通信 2 分；GQA/MQA约束 2 分；给具体shape/owner 2 分；性能含义 1 分。

**追问分析与参考回答**：head_dim=128，MHA时每rank 8个Q/K/V heads。GQA设8个KV groups且TP8时，理想布局为每rank 1个KV group服务本rank 8个Q heads中的对应组；本地Q维`8x128`，KV维`1x128`每种。若groups小于TP，就可能需要复制或更复杂映射，不能负数/零head分片。

**追问得分点（5分）**：head_dim 1 分；Q heads 1 分；KV owner 2 分；groups<TP反例 1 分。

### M015. MHA、GQA、MQA 对 Megatron 并行与显存有什么影响？

**高分回答要点**

- GQA/MQA 减少 KV heads，训练 attention 参数和推理 KV cache 更小，但 Q 与 KV 的 TP 可整除/复制关系更复杂。
- 若 KV groups 少于 TP，某些 rank 可能复制 KV 或使用专门 partition，不能直接假设所有权重都按 TP 等分。
- 性能需看 projection GEMM、attention kernel 支持、通信和负载均衡，不只看参数数目。

**追问**：为什么 TP degree 可能受 `num_query_groups` 而不只是 `num_attention_heads` 限制？

**前置知识与分析过程**：先把参数/activation减少与layout合法性分开。列Q heads和KV groups两个独立分片轴，再分析复制方案和性能代价。

**主问题得分点（10分）**：MHA/GQA/MQA定义 3 分；参数/KV收益 2 分；TP分片约束 3 分；kernel/性能边界 2 分。

**追问分析与参考回答**：Q可按attention heads分，但K/V只有`num_query_groups`组；若TP大于或不能整除groups，简单“每rank若干KV组”失效。框架需复制KV组、改变partition或拒绝配置，都会影响参数节省和通信，因此config validation要同时检查heads与query groups。

**追问得分点（5分）**：指出独立约束 2 分；给不可整除场景 1 分；列复制/拒绝方案 1 分；性能后果 1 分。

### M016. RoPE、learned absolute position 和长上下文 scaling 在架构上如何接入？

**高分回答要点**

- learned absolute position 通常在 embedding 相加；RoPE 在 attention 内对 Q/K 施加位置旋转，不增加 position embedding table。
- PP 首 stage 提供 position ids，但实际 RoPE 计算在持有 attention layer 的各 stage；CP 还需正确处理全局 position offset。
- scaling 改变模型语义和 checkpoint 配置，不是纯 runtime 优化。

**追问**：为什么 CP rank 只持局部 token 仍必须知道全局 position？

**前置知识与分析过程**：先区分位置编码注入点，再分析PP传position、CP切sequence后的global offset。最后说明checkpoint/config语义。

**主问题得分点（10分）**：三类位置编码区别 3 分；RoPE接入attention 2 分；PP/CP position流 3 分；scaling语义边界 2 分。

**追问分析与参考回答**：RoPE相位由token在完整序列中的绝对/规则位置决定；CP rank虽只计算局部chunk，但若把本地索引从0重新编号，跨chunkQ/K相位关系错误。应由全局sequence offset生成position或传递正确position ids，packing时还需按样本边界重置/映射。

**追问得分点（5分）**：RoPE相位语义 2 分；global offset 1 分；错误后果 1 分；packing边界 1 分。

### M017. SwiGLU MLP 的参数与 TP 切分有什么特殊点？

**高分回答要点**

- 第一投影产生 gate 和 value 两路，局部激活后相乘，再经第二投影回 hidden；中间维与参数量口径需考虑两路。
- gate/up 常用 column parallel 在输出维切，down 用 row parallel 在输入维切，激活可保持本地。
- fused SwiGLU/GroupedGEMM 的 shape、对齐和 checkpoint layout 都与普通 GELU MLP 不同。

**追问**：为什么从 GELU `4h` 切到 SwiGLU 常调整 FFN hidden，而不是仍机械使用 4h？

**前置知识与分析过程**：先写GELU与SwiGLU两种FFN参数公式，再按TP轴推local shape。比较参数/FLOPs对齐后才能讨论中间维选择。

**主问题得分点（10分）**：SwiGLU数据流 3 分；两路参数口径 2 分；TP切分 2 分；fusion/shape 2 分；中间维调整原因 1 分。

**追问分析与参考回答**：GELU FFN约有`h*4h + 4h*h=8h^2`权重；SwiGLU第一投影有gate/up两路，若每路仍4h，总量约`12h^2`，显著增参增FLOPs。为匹配dense FFN预算，常把每路中间维降至约`8h/3`并做硬件对齐，具体值依模型recipe。

**追问得分点（5分）**：写两种参数公式 2 分；得出需降中间维 1 分；给约`8h/3`及非固定边界 1 分；对齐/fusion 1 分。

### M018. Vocab-parallel embedding 如何工作？

**高分回答要点**

- vocabulary rows 在 TP ranks 分片；每 rank 对不属于本 shard 的 token mask，查本地表后在 TP group 归并结果。
- 参数显存按 TP 降低，但 forward 有通信；词表 padding/整除会影响本地 shard 大小。
- tied output weight 必须保持输入 embedding 与输出 projection 的 shard/同步语义一致。

**追问**：输入 token 只属于一个 shard，为什么最终每个 rank 的 hidden activation 都能得到正确 embedding？

**前置知识与分析过程**：先写词表切分区间和token mask，再对单token追踪各rank局部lookup值。最后判断使用reduce还是gather及输出layout。

**主问题得分点（10分）**：vocab row分片 2 分；mask/local lookup 2 分；跨rank归并 2 分；padding/整除 2 分；tied权重 2 分。

**追问分析与参考回答**：只有owner rank查到真实向量，其他rank将该token局部结果置零；对TP group做sum-reduce后，每rank都得到owner贡献，即完整embedding hidden。若后续SP需要sequence shard，归并形式可能与layout组合优化，但逻辑oracle仍是每token唯一owner贡献。

**追问得分点（5分）**：owner/mask 2 分；sum语义 2 分；layout优化边界 1 分。

### M019. Vocab-parallel cross entropy 如何避免 gather 全量 logits？

**高分回答要点**

- 每 rank 只持 vocab shard logits；先跨 rank 求 global max 保数值稳定，再求 global exp sum。
- target logit 只在所属 shard 提取，跨 rank归并后计算 loss；backward 直接生成本地 logits gradient。
- 通信规模主要与 token 数相关，而不是完整 `tokens x vocab` AllGather，显著降低峰值。

**追问**：手推 log-sum-exp 的 global max/sum 两阶段，以及 target 不在本 shard 时的 mask。

**前置知识与分析过程**：从单token交叉熵`-z_y+log sum exp(z)`出发，把vocab维拆为shards。为数值稳定先求全局max，再求全局exp和与target logit。

**主问题得分点（10分）**：避免full logits gather 2 分；global max 2 分；global exp sum 2 分；target mask/reduce 2 分；backward本地gradient 2 分。

**追问分析与参考回答**：各rank算local max，AllReduce(MAX)得m；算`sum_j exp(z_j-m)`的local sum并AllReduce(SUM)得S。target owner取`z_y`，非owner置0，再SUM得到全局target logit；loss=`log(S)+m-z_y`。backward每rank对本地softmax减本地target one-hot。

**追问得分点（5分）**：max阶段 1 分；sum阶段 1 分；target owner 1 分；loss公式 1 分；backward 1 分。

### M020. tied embedding 在 PP 首尾 stage 之间如何保持一致？

**高分回答要点**

- 首 stage 输入 embedding 与末 stage output weight 逻辑共享，但物理上可能位于不同 PP ranks。
- 初始化/加载后需要 embedding group 同步参数，backward 后还需归并两端 gradient。
- checkpoint metadata 必须表达共享关系；只让两个 stage 各自 optimizer step 会发生漂移。

**追问**：这也是 `finalize_model_grads()` 不能只做普通 DP gradient sync 的原因之一。

**前置知识与分析过程**：先区分逻辑共享与物理副本，再列参数初始化、forward读取、backward gradient、optimizer四个同步点。判断普通DP group是否包含PP首尾。

**主问题得分点（10分）**：PP首尾共享问题 3 分；参数/grad同步 3 分；checkpoint表达 2 分；optimizer漂移风险 2 分。

**追问分析与参考回答**：普通DP只连接相同model-parallel坐标的replicas，不会连接PP first/last。tied embedding在首尾物理各有一份，需要embedding group初始化/broadcast并在optimizer前AllReduce两端grad；`finalize_model_grads`集中处理此类跨普通DP语义。否则两份权重分别更新后失去tie。

**追问得分点（5分）**：指出DP不覆盖 1 分；embedding group 1 分；grad归并 2 分；漂移后果 1 分。

### M021. Megatron 参数初始化为什么要考虑 TP？

**高分回答要点**

- 分片参数应与未分片 reference 的逻辑初始化分布一致，且不同 TP ranks 持有正确、不重复的 shard。
- CPU initialization、GPU initialization 和 checkpoint load 的内存峰值/速度不同；大模型常用分布式或 meta 初始化。
- seed、partition axis 和 master weight 构造错误会在第一步前造成跨并行配置不一致。

**追问**：如何验证 TP1 与 TP2 初始化等价？gather 逻辑权重后逐 tensor 比较，并跑同输入 forward/backward。

**前置知识与分析过程**：先定义“逻辑全局权重”与“rank local shard”，再区分随机分布等价和逐元素等价。测试需固定seed、partition axis和padding。

**主问题得分点（10分）**：初始化为何依赖TP 3 分；CPU/GPU路径 2 分；seed/shard一致性 2 分；完整oracle 3 分。

**追问分析与参考回答**：用相同master seed构造TP1权重；TP2按真实partition axis生成或加载shards，gather后与TP1逐元素比较。再关闭dropout，用同batch比较logits、loss和gathered gradients；一步optimizer后再比。只比较均值/方差不能发现shard重复或轴切错。

**追问得分点（5分）**：gather逐元素 2 分；同输入forward/backward 1 分；一步更新 1 分；拒绝只比统计量 1 分。

### M022. FP16/BF16 训练中 model params、main params、main grads 各是什么？

**高分回答要点**

- 低精度 model params 用于 forward/backward；optimizer 通常维护 FP32 main params 与 moment states。
- gradients 可在低精度或 FP32 contiguous main-grad buffer 累积，影响每参数 bytes、数值稳定和通信 dtype。
- optimizer step 后要把更新后的 main params 同步/拷回 model params，distributed optimizer 还要 AllGather shards。

**追问**：为什么只说“FP16 参数 2 bytes”会严重低估 Adam 训练显存？

**前置知识与分析过程**：对每参数列model param、model/main grad、FP32 master、Adam m/v，再标dtype与是否分片。最后加入bucket/padding和临时buffer边界。

**主问题得分点（10分）**：三类param/grad定义 3 分；optimizer状态 2 分；通信/更新数据流 3 分；显存公式边界 2 分。

**追问分析与参考回答**：典型FP16 Adam至少含2B model param、可能2/4B grad、4B FP32 master、4B m和4B v，另有buffer/副本，常接近16-20B/param而非2B。distributed optimizer按DP分片其中大部分，但低精度forward参数和临时AG仍存在。

**追问得分点（5分）**：列五类状态 2 分；给数量级 1 分；分片影响 1 分；临时/allocator边界 1 分。

### M023. `forward_step_func` 与 `loss_func` 的接口为何对 PP 很关键？

**高分回答要点**

- schedule 控制何时运行哪个 microbatch，但不知道业务 batch/loss；callback 让应用提供 stage-local forward。
- 只有 last stage 通常计算 loss，其他 stage 返回 activation；loss dict 用于日志和跨 stage 汇总。
- callback 必须处理 first/last stage 输入差异、packed sequence/attention mask 和 gradient scaling。

**追问**：为什么每个 stage 都直接读取完整 batch 既浪费又可能破坏数据迭代语义？

**前置知识与分析过程**：先写schedule对callback的调用次数与stage差异，再明确数据iterator应代表同一逻辑microbatch。分析first/last需要哪些字段。

**主问题得分点（10分）**：callback控制反转 3 分；first/last输入差异 2 分；loss closure 2 分；数据/scale风险 3 分。

**追问分析与参考回答**：若每stage独立推进dataset iterator，可能取得不同样本；中间stage只需上游activation和mask/meta，不需token/label完整IO。通常同model replica由指定rank准备/broadcast必要数据，schedule保证microbatch id一致，last stage只用对应label算loss。

**追问得分点（5分）**：不同样本风险 2 分；字段最小化 1 分；一致microbatch id 1 分；合理broadcast/iterator方案 1 分。

### M024. 如何建立 TP/PP 模型 correctness oracle？

**高分回答要点**

- 小模型上构造 TP1/PP1 reference，gather sharded weights，固定输入、dropout/RNG，比较 logits/loss/grad/一步更新。
- 分开检查 parameter layout、forward、backward、optimizer 与 checkpoint；`loss finite` 只能排除最粗错误。
- 允许浮点容差但报告 max/relative error，并用多 step 检测小偏差累积。

**追问**：同 shape collective 顺序错配可能得到 finite loss，因此还要 tensor checksum 和 collective fingerprint。

**前置知识与分析过程**：把correctness拆为weight layout、forward、backward、optimizer和checkpoint五层；每层选择独立reference与误差口径，不能把最终loss当唯一oracle。

**主问题得分点（10分）**：TP1 reference 2 分；五层验证 3 分；RNG/容差 2 分；多step与静默通信错误 3 分。

**追问分析与参考回答**：对每个logical collective记录group、sequence、op、shape/dtype和输入checksum；业务tensor再记录logical id。若A/B同shape跨rank顺序反转，NCCL可完成且输出finite，但fingerprint的logical id/sequence在首个分歧点暴露。小模型同时比较gathered tensor reference。

**追问得分点（5分）**：指出finite不足 1 分；fingerprint字段 2 分；logical id作用 1 分；reference交叉验证 1 分。

---

## 4. Level 2：Tensor Parallel 与 Sequence Parallel（M025-M040）

### M025. ColumnParallelLinear 如何切参数、输入和输出？

**高分回答要点**

- 逻辑 `Y=XW`，沿 W 输出列切成 `W_i`；每 rank 输入 X 通常完整，计算本地 `Y_i=XW_i`。
- 若下一算子可消费分片输出，就不 gather；需要完整 Y 时才 AllGather。
- backward 本地算 `dW_i`，各 rank 对 X 的梯度 contribution 需在 TP group 归约。

**追问**：写出 `X:[s,b,h]`、`W:[h,4h]`、TP=t 时本地 W/Y shape 和 dX 通信 bytes。

**前置知识与分析过程**：固定矩阵约定`Y=XW`，先选W的partition axis，再推local forward，最后用链式法则推dX/dW与collective。任何TP题都按此顺序，不靠“列并行”名称猜通信。

**主问题得分点（10分）**：切分轴和local shape 3 分；forward是否gather 2 分；backward dX归约 2 分；通信bytes/可组合性 2 分；bias边界 1 分。

**追问分析与参考回答**：`W_i:[h,4h/t]`，`Y_i:[s,b,4h/t]`；`dW_i=X^T dY_i`本地，`dX_i=dY_i W_i^T:[s,b,h]`是各rank contribution，需SUM AllReduce得到完整dX。logical payload为`s*b*h*dtype_bytes`，ring实际链路量再乘collective因子。

**追问得分点（5分）**：W/Y shape 2 分；dW本地 1 分；dX归约 1 分；bytes口径 1 分。

### M026. RowParallelLinear 如何切分？

**高分回答要点**

- 沿 W 输入行切成 `W_i`，输入 X 也按最后一维切；本地输出 contribution 都是完整 hidden shape。
- forward 对本地 contribution 求和，通常 AllReduce；SP 下可改为 ReduceScatter 得到 sequence shard。
- backward 对 dY 的处理取决于 layout，本地即可得到对应输入 shard gradient，dW_i 本地计算。

**追问**：为什么 row-parallel 的 forward 通信与 column-parallel 的 backward 通信位置互补？

**前置知识与分析过程**：沿输入维切W并同步切X，写本地乘积后判断如何恢复逻辑Y；再反推dX shard。与ColumnParallel并排画图最清楚。

**主问题得分点（10分）**：W/X切分 3 分；forward contribution归约 2 分；backward layout 2 分；SP的RS变体 2 分；组合意义 1 分。

**追问分析与参考回答**：RowParallel每rank算`Y_i=X_iW_i`，完整Y是各rank contribution之和，所以forward归约；其dX_i=`dY W_i^T`天然就是对应输入shard。ColumnParallel forward天然产出输出shard，但所有输出shards都依赖同一X，backward dX要汇总。因此两者前后组合可让中间shard不gather。

**追问得分点（5分）**：Row forward原因 2 分；Column backward原因 2 分；组合结论 1 分。

### M027. 两层 MLP 为什么可以只在边界通信，而不是两个 GEMM 都 gather？

**高分回答要点**

- 第一层 column parallel 产生本地 FFN shard，激活是逐元素操作，可直接本地执行。
- 第二层 row parallel 消费同一 shard，只在输出 contribution 汇合时通信。
- 这就是 Megatron TP 的核心：相邻 partition 选择让中间大 activation 不必 gather。

**追问**：若在两层中间插入需要跨 hidden 维全局信息的算子，会破坏该设计。

**前置知识与分析过程**：先判断中间算子是逐元素、只沿本地分片归约，还是需要全局FFN维。只有前两类可直接消费local shard。

**主问题得分点（10分）**：两层partition配对 4 分；激活本地性 2 分；通信边界 2 分；反例与重排方法 2 分。

**追问分析与参考回答**：GELU/SwiGLU在每个元素或成对本地分片上计算，无需其他rank；若插入对完整FFN维做LayerNorm、top-k或全局softmax，就必须先AllGather或设计distributed reduction，增加通信和full activation峰值。应重新选择切分轴或融合成分布式算子。

**追问得分点（5分）**：逐元素判断 1 分；给全局维反例 2 分；通信/峰值后果 1 分；替代设计 1 分。

### M028. Self-attention 的 TP 数据流如何与 MLP 类比？

**高分回答要点**

- QKV projection column-parallel，把 heads 分给 ranks；attention 对本地 heads 独立计算。
- output projection row-parallel，汇总各 rank head contribution。
- softmax 在 sequence/key 维而非 head 维归约，因此 head partition 下通常无需 TP 通信；CP 则改变这一点。

**追问**：标出 QKV projection backward 和 output projection forward 的 TP collective。

**前置知识与分析过程**：把attention拆为QKV ColumnParallel、本地heads attention、output RowParallel，再套M025/M026的autograd规则。

**主问题得分点（10分）**：QKV/head分片 3 分；本地attention 2 分；output projection 2 分；F/B collective 2 分；CP例外 1 分。

**追问分析与参考回答**：QKV ColumnParallel forward无需gather，本地heads完成attention；其backward对输入hidden的dX contributions需TP SUM。output RowParallel forward对各head输出projection contributions做TP AllReduce，开SP时可RS；其backward产生对应输入/head shard。

**追问得分点（5分）**：QKV backward 2 分；output forward 2 分；SP变体 1 分。

### M029. 怎样估算一个 Transformer layer 的 TP 通信量？

**高分回答要点**

- 从 activation 边界而非参数量出发：典型两次 TP 汇合，张量规模与 `sequence x microbatch x hidden x dtype` 相关。
- forward/backward 的 AllReduce 或 RS/AG 算法通信量还乘 `(t-1)/t` 等 collective 因子。
- 调用频率是每层、每 microbatch、forward/backward，位于关键路径；不能用每 step 一次 DP 模型估算。

**追问**：序列翻 8 倍时 TP activation bytes 近似翻 8 倍，但 attention compute 可能增长更快，通信占比未必增加。

**前置知识与分析过程**：从logical tensor元素数计算payload，再乘层数、microbatches和F/B调用次数；最后与GEMM/attention FLOPs增长比较，区分绝对通信和占比。

**主问题得分点（10分）**：activation而非parameter口径 2 分；payload公式 2 分；collective算法因子 2 分；调用频率 2 分；compute/communication比 2 分。

**追问分析与参考回答**：TP边界payload约与`s*b*h*bytes`线性增长；但标准attention score/value FLOPs含`s^2`项，长序列时compute可能增长更快，所以通信毫秒增加而占step比例下降。本机SP吞吐代价随seq增长缩小符合此机制，但需trace验证，不能仅凭复杂度唯一归因。

**追问得分点（5分）**：bytes线性 1 分；attention二次 1 分；占比结论 1 分；本机证据 1 分；需profile边界 1 分。

### M030. `copy/scatter/gather/reduce_from_tensor_model_parallel_region` 等 mapping 的 backward 为什么不同？

**高分回答要点**

- 这些 autograd Function 把 forward layout 变换与 backward 的逆/伴随 collective 成对定义。
- forward identity/copy 可能在 backward AllReduce；forward scatter 对应 backward gather；forward reduce 对应 backward copy。
- 专家应根据数学 Jacobian 推导，而不是只背函数名。

**追问**：让你为一个新 layout primitive 写 forward/backward correctness test，应做 adjoint/gradcheck 和多 rank reference。

**前置知识与分析过程**：把mapping视为线性算子A，backward应实现伴随`A^T`。先数学推导，再实现多rankshape/value和autograd验证。

**主问题得分点（10分）**：mapping成对语义 3 分；至少三组F/B关系 3 分；Jacobian/伴随解释 2 分；测试方案 2 分。

**追问分析与参考回答**：随机小tensor运行distributed primitive，与单进程显式concat/split/sum reference比较forward；给随机上游grad验证`<Ax,y>=<x,A^Ty>`，再做double precision finite-difference/gradcheck可行部分。覆盖非整齐shape拒绝、不同rank和连续性。

**追问得分点（5分）**：reference 1 分；内积伴随测试 2 分；gradcheck 1 分；边界case 1 分。

### M031. 为什么 TP degree 受 hidden、heads、query groups、FFN hidden 和 vocab 约束？

**高分回答要点**

- 分片轴通常要求可整除或明确 padding；不合法会造成 rank shape 不一致或负载不均。
- attention head/query-group 分配、SwiGLU 两路、vocab shard 各有独立约束。
- 即使数学可 padding，过小本地 GEMM/heads 也可能性能不可接受。

**追问**：配置校验通过不等于性能合理，必须列本地 GEMM M/N/K。

**前置知识与分析过程**：逐module列切分轴和整除约束，再把合法global shape除以TP得到local M/N/K、local heads和vocab shard，检查硬件tile。

**主问题得分点（10分）**：五类整除约束 4 分；padding语义 2 分；local GEMM性能 2 分；config fail-fast 2 分。

**追问分析与参考回答**：例如Column FFN的N约为`ffn_hidden/tp`，Row FFN的K被除；即使都整除，N/K只剩很小tile会导致Tensor Core waves不足。还需列`M=s*b`，因为小microbatch同样让GEMM变瘦。以Nsight/TE kernel和achieved FLOPs验证。

**追问得分点（5分）**：列M/N/K 2 分；说明两个被TP切的轴 1 分；tile/wave后果 1 分；profiler验证 1 分。

### M032. 为什么 TP 经常要求放在节点内 NVLink/NVSwitch 域？

**高分回答要点**

- TP 每层多次通信且同步紧，latency/带宽直接进入 layer critical path，难像 DP bucket 跨多层隐藏。
- 跨节点 TP 还会放大 NIC latency、共享 rail contention 与故障耦合。
- 本机将 TP pair 从 PIX 换到 SYS 后，同一 TP phase 慢 3.05x，是 topology placement 的直接证据。

**追问**：超大 layer 单节点仍放不下时可以跨节点 TP，但需量化它与提高 PP/CP 的替代方案。

**前置知识与分析过程**：先按每层TP collective频率判断网络敏感性，再按容量约束找“最低必要TP”。将剩余GPU优先交给PP/DP/CP候选并做成本比较。

**主问题得分点（10分）**：TP关键路径 3 分；节点内映射 2 分；跨节点风险 2 分；替代方案与实验 3 分。

**追问分析与参考回答**：若单层权重/activation在节点内TP上仍放不下，跨节点TP是容量性约束；但应比较增PP能否切深度、CP能否切长序列、FSDP能否切状态。对每候选计算内存、每层/每stage通信、bubble和本地GEMM，而非默认跨节点TP。

**追问得分点（5分）**：承认容量场景 1 分；列三种替代 2 分；成本维度 1 分；最小TP原则 1 分。

### M033. TP 增大为什么会让 GEMM 效率下降？

**高分回答要点**

- partition 后 GEMM 的 N/K 或 head 数变小，Tensor Core tile、wave 数和 kernel occupancy 可能下降。
- collective launch、同步和框架固定开销不会随本地计算同比下降。
- 所以总 FLOPs 分摊到更多 GPU 不等于 step time 按 TP 缩短。

**追问**：profile 时报告本地 GEMM shape、Tensor Core utilization、kernel duration、NCCL exposed time和每 GPU tokens/s。

**前置知识与分析过程**：把理想计算缩放、kernel效率损失和通信固定成本三项写成step模型。逐TP点检查哪项不按`1/t`缩放。

**主问题得分点（10分）**：GEMM变小机制 3 分；固定开销 2 分；同步关键路径 2 分；完整指标与实验 3 分。

**追问分析与参考回答**：先从trace导出每个parallel linear的M/N/K和kernel时间，确认Tensor Core/occupancy；再对NCCL区间做union与compute overlap，算exposed；最后用global tok/s和per-GPU tok/s判断扩展与资源效率。只看`nvidia-smi` busy无法区分小kernel和有效计算。

**追问得分点（5分）**：shape 1 分；Tensor Core指标 1 分；exposed算法 1 分；per-GPU效率 1 分；拒绝busy单指标 1 分。

### M034. Sequence Parallel 到底分了什么？

**高分回答要点**

- SP 将 TP 区域中原本复制的 sequence 维 activation 分片，使 LayerNorm、dropout 等逐 token 操作只处理本地 token shard。
- TP 边界把部分 AllReduce 重写为 ReduceScatter/AllGather，通信总 bytes 量级接近，但中间 activation 不再全复制。
- SP 不把 attention 的完整 context 依赖彻底切开，不能等同 CP。

**追问**：画出 row-parallel output 经 ReduceScatter 后的 `[s/t,b,h]`，以及下个 column-parallel 前何时 AllGather。

**前置知识与分析过程**：先从无SP的复制`[s,b,h]`出发，把AllReduce分解成RS+AG并标sequence切分。再沿norm/dropout/residual到下个需要完整X的linear。

**主问题得分点（10分）**：SP对象 3 分；RS/AG替代 3 分；非attention/CP边界 2 分；显存收益条件 2 分。

**追问分析与参考回答**：RowParallel本地contribution为`[s,b,h]`，RS(SUM)同时归约并按sequence给每rank`[s/t,b,h]`；本地norm/dropout/residual保持该layout。下一ColumnParallel若实现要求每rank完整输入，则AG回`[s,b,h]`后做GEMM；某些融合路径可重排，但数学布局等价。

**追问得分点（5分）**：RS输入/输出shape 2 分；本地算子 1 分；AG位置 1 分；实现可优化边界 1 分。

### M035. 为什么 SP 参数 gradient 可能需要额外 TP 归约？

**高分回答要点**

- LayerNorm 等参数在 TP ranks 复制，但每 rank 只看不同 sequence shard，因此本地参数 gradient 只含局部 tokens。
- backward 后必须在 TP group 汇总这些复制参数的 gradient，才能等价完整序列。
- 本机教学脚本显式对带 `sequence_parallel` 标记的参数 AllReduce；生产 MCore 由对应 runtime 完成。

**追问**：漏掉这一步 loss 仍可能 finite，但 TP ranks 的复制参数会逐步漂移。

**前置知识与分析过程**：判断一个参数是sharded还是replicated，再判断每rank看到相同还是不同token。复制参数若看到不同数据，其grad必须归并。

**主问题得分点（10分）**：SP参数复制语义 3 分；局部token gradient 2 分；TP归约 2 分；静默错误与测试 3 分。

**追问分析与参考回答**：LayerNorm gamma/beta在TP ranks各有副本，SP后每rank只累积本地sequence shard贡献。若不AllReduce，各副本更新方向不同；下一轮即使activation collective正确，模型已不再逻辑等价。测试应每step比较复制参数hash和gathered reference gradient。

**追问得分点（5分）**：复制参数 1 分；不同token贡献 1 分；AllReduce 1 分；漂移后果 1 分；hash/reference 1 分。

### M036. 为什么 TP+EP 的一些配置要求 SP？

**高分回答要点**

- dense TP 区域若保持 token activation 复制，进入 expert routing 前会让 token 被多个 TP ranks 重复路由/通信。
- SP 提供按 token 分片的输入 layout，使每个 token 在目标 routing 域具有清晰 ownership。
- 具体限制随 MCore 版本和 expert-TP 设计变化，回答时应核对当前 config validation。

**追问**：不要只背“官方要求”，要画出没有 SP 时重复 token 如何进入 EP dispatcher。

**前置知识与分析过程**：画dense TP输出在各rank是复制token还是sequence shard，再定义router/dispatcher期望的token ownership。检查同一token是否被重复路由。

**主问题得分点（10分）**：TP复制layout 2 分；EP token ownership 3 分；SP作用 2 分；版本约束和验证 3 分。

**追问分析与参考回答**：无SP的某些TP边界上，每个TP rank都持同一组tokens的hidden表示（hidden内部可能已汇合）；若各rank都进入EP router/dispatcher，同一logical token会被发送多次。SP让token按sequence只属于一个TP rank，再路由一次；expert-TP内部另行处理hidden分片。

**追问得分点（5分）**：画重复token 2 分；SP唯一owner 2 分；expert-TP区别 1 分。

### M037. SP 的通信量为什么“近似不变”但性能仍会变？

**高分回答要点**

- AllReduce 可分解为 ReduceScatter+AllGather，算法总 bytes 量级相近，但 collective 次数、位置和 tensor lifetime 改变。
- RS/AG 的 kernel/protocol、stream 依赖、可 overlap 窗口和小消息 latency 不同。
- SP 还改变 LayerNorm/dropout kernel shape和 allocator 峰值，端到端结果不能只从 bytes 推断。

**追问**：本机 TP2/SP 吞吐降 15.9%，说明 PCIe 上额外同步/布局成本超过少量显存收益。

**前置知识与分析过程**：先比较AllReduce与RS+AG算法bytes，再比较collective个数、时间位置、kernel shape和activation lifetime。最后回到端到端数据，而非从通信量直接判快慢。

**主问题得分点（10分）**：bytes近似关系 2 分；时序/collective差异 3 分；kernel/allocator影响 2 分；实验解释 3 分。

**追问分析与参考回答**：本机seq512小模型中可被SP切的activation只占很小峰值，省约3%；RS/AG和layout转换在7.6GB/s PCIe上增加同步，局部norm等kernel也变小，最终tok/s降15.9%。这支持该shape不划算，不证明SP普遍无效。

**追问得分点（5分）**：给3%与15.9% 1 分；链路机制 2 分；其他kernel因素 1 分；不外推 1 分。

### M038. 为什么 seq=4096 时本项目 SP 相对显存收益反而缩小？

**高分回答要点**

- SP 只切一部分逐 token activation；attention 相关张量、workspace、参数和 allocator 项不一定同比被切。
- 随 sequence 增长，未被 SP 等比例切分的部分增长更快，导致可切部分占总峰值比例下降。
- 实测 TP2 从 seq512 的 3.1% 降到 seq4096 的 1.3%，是否值得应看绝对 GiB 和吞吐代价。

**追问**：下一步应做 memory snapshot 按 allocation 分类，而不是继续假设“长序列必然利好 SP”。

**前置知识与分析过程**：把峰值拆成SP可切、不可切和固定项；观察这些项随sequence的增长阶。用绝对GiB和比例两个口径解释。

**主问题得分点（10分）**：可切范围 2 分；未切项增长 3 分；本机数字 2 分；后续证伪实验 3 分。

**追问分析与参考回答**：在关键peak前后抓PyTorch memory snapshot并按stack/module分类：norm/dropout/residual、attention score/workspace、参数/optimizer和NCCL buffer。对seq512/2K/4K比较每类bytes及lifetime；若attention未切项增长主导，就能解释SP比例下降。再加入CP或高效attention作对照。

**追问得分点（5分）**：snapshot 1 分；分类 2 分；跨sequence对照 1 分；CP/attention证伪 1 分。

### M039. 如何公平 benchmark TP/SP？

**高分回答要点**

- 固定模型、global batch、sequence、dtype、总 GPU；TP 改变 DP 时用 accumulation 保持 batch 语义。
- 报 max-rank step、tokens/s、每 GPU efficiency、allocated/reserved peak、本地参数数和 loss correctness。
- warmup/JIT、三次重复、拓扑 placement 与实际 group 必须保存；再用 trace 分解 GEMM 与 collective。

**追问**：本项目 TP1/DP8 到 TP8/DP1 是固定 GPU 预算下替换 DP，不是固定 DP 的 scaling test。

**前置知识与分析过程**：先定义研究问题：固定总GPU比较并行配置，还是固定DP增加TP做scale。随后锁定GBS/shape和测量窗口，明确吞吐聚合口径。

**主问题得分点（10分）**：公平控制变量 3 分；GBS计算 2 分；指标/correctness 2 分；拓扑/重复/trace 3 分。

**追问分析与参考回答**：8卡始终全用，但TP升高会相应降低DP，所以aggregate tok/s同时反映层内TP成本和失去DP副本吞吐；它回答“固定预算如何选配置”。若要测TP scaling，应固定同一DP replica并从TP1的1卡增加到TP8的8卡，另报speedup/efficiency。

**追问得分点（5分）**：固定预算含义 2 分；混合因素 1 分；正确TP scaling设计 2 分。

### M040. 如何解释本机 TP8 显存降 83.4%，吞吐只剩 TP1 的 17.3%？

**高分回答要点**

- 参数/GEMM 成功分片，所以容量收益真实；但 V100 PCIe 8 卡 bus BW 约 7.6 GB/s，层内 collective 昂贵。
- tiny GPT 本就单卡可放，TP 后本地 GEMM 过小；同时 DP 从 8 降到 1，失去无层内通信的副本吞吐。
- 正确结论是该 workload 上 TP 是容量工具，不是“Megatron TP 实现无效”；换大 hidden/NVSwitch 需重测。

**追问**：若面试官给 H100 NVSwitch 70B，哪些项会改变？链路、GEMM shape、模型可放性和 precision 都变，但分析框架不变。

**前置知识与分析过程**：把本机结果拆为硬件相关量、模型shape相关量和算法不变量。迁移时重新估算容量、local GEMM与通信，而不是缩放旧tok/s。

**主问题得分点（10分）**：解释容量收益 2 分；三类性能损失 3 分；本机数字 2 分；正确外推边界 3 分。

**追问分析与参考回答**：H100 NVSwitch显著提高TP带宽/降低相对通信，70B的hidden与GEMM更大且模型可能迫使TP，BF16/FP8/TE kernel也改变compute；因此TP可能从纯容量工具变成有效加速。但每层同步、local shape和最小必要TP原则仍成立，需重新做TP degree sweep和MFU/peak/trace。

**追问得分点（5分）**：列硬件链路 1 分；模型shape/容量 1 分；precision/kernel 1 分；稳定机制 1 分；重测方案 1 分。

---

## 5. Level 3：Pipeline Parallel 调度与负载均衡（M041-M054）

### M041. `get_forward_backward_func()` 如何选择 pipeline schedule？

**高分回答要点**

- 根据 PP world size、virtual pipeline/interleaving 和 forward-only 等配置选择无 PP、非交错 1F1B 或交错 schedule。
- schedule 拥有 microbatch 执行顺序与 P2P，模型应用只提供 `forward_step_func` 和 stage-local model chunks。
- 专家要追到实际返回函数和输入契约，不把 schedule 当黑盒。

**追问**：本机 `PP=1` 与 `PP>1` 调用同一入口，但实际控制流和 loss 返回位置不同。

**前置知识与分析过程**：先列schedule选择条件，再对PP1、非交错PP、交错VPP分别画调用对象。最后说明callback接口为何保持一致。

**主问题得分点（10分）**：选择条件 3 分；三类schedule差异 3 分；callback契约 2 分；能追到返回函数/源码 2 分。

**追问分析与参考回答**：PP1不需P2P，schedule顺序执行全部microbatch的本地F/B并直接返回loss；PP>1按rank计算warmup/steady/cooldown，只有last stage产生loss dict，其他stage主要传activation/grad。相同入口隐藏分支，但profile与返回值语义不同。

**追问得分点（5分）**：PP1无P2P 1 分；PP三阶段 2 分；last-stage loss 1 分；相同API不同控制流 1 分。

### M042. GPipe 与 1F1B 的主要差异是什么？

**高分回答要点**

- GPipe 先完成所有 forward 再 backward，调度简单但需长期保存更多 microbatch activation。
- 1F1B 在 warmup 后交替 forward/backward，减少 activation 驻留并保持 pipeline steady state。
- 两者基本 bubble 上界都受 stage 数和 microbatch 数影响，但显存与通信时序不同。

**追问**：为什么 1F1B 并不自动消除 bubble？仍有 fill/drain，且 stage imbalance 会扩大有效空洞。

**前置知识与分析过程**：画P stages、M microbatches时空图，比较activation lifetime与fill/drain。再加入不等stage time看慢stage如何传播等待。

**主问题得分点（10分）**：GPipe/1F1B顺序 3 分；activation差异 2 分；bubble来源 2 分；imbalance/通信边界 3 分。

**追问分析与参考回答**：1F1B只在管线填满后交替F/B；开始时后stage无输入、结束时前stage无新工作，`P-1`个stage间距仍存在。若某stage耗时更长，流水节拍由它决定，其他stage即便理论处于steady也会等待P2P。

**追问得分点（5分）**：fill 1 分；drain 1 分；公式/间距 1 分；慢stage传播 2 分。

### M043. 非交错 1F1B 的 warmup、steady、cooldown 各发生什么？

**高分回答要点**

- warmup 中不同 PP rank 先执行若干 forward，把 pipeline 填满；越靠前 stage 通常 warmup forward 更多。
- steady 中每 rank 交替执行一个 forward 与一个 backward，并进行相邻 P2P。
- cooldown 停止新 forward，排空剩余 backward；每 rank 阶段长度不同但 collective/P2P 顺序必须匹配。

**追问**：让你为 PP4/M8 画每个 stage 的时空图，并指出最早的 backward 在何时开始。

**前置知识与分析过程**：先编号F0..F7/B0..B7，按数据依赖让microbatch逐stage前进；last stage完成F0和loss后才能启动B0，再反向传播。

**主问题得分点（10分）**：三阶段定义 3 分；不同rank warmup长度 2 分；正确F/B依赖 3 分；P2P顺序/显存 2 分。

**追问分析与参考回答**：PP4中F0依次到stage3；stage3完成F0后最早可做B0。stage0在收到B0前继续注入若干forward，形成较长warmup；进入steady后各stage交替本地F与B。图必须保证同一microbatch的B只在其F和下游B完成后发生。

**追问得分点（5分）**：F0传播 1 分；最早B0位置 1 分；stage0 warmup 1 分；依赖正确 2 分。

### M044. pipeline bubble 公式如何推导？

**高分回答要点**

- 等 stage time、非交错理想模型下，利用率约 `M/(M+P-1)`，bubble 为 `(P-1)/(M+P-1)`。
- 公式只描述 fill/drain，不含 stage imbalance、P2P、kernel 效率、DP sync 与 optimizer。
- 当 forward/backward time 不等、VPP 或异步通信存在时，应使用更完整 schedule simulation/timeline。

**追问**：PP8/M16 bubble=`7/23=30.4%`，理论利用率 69.6%，为何实测并行效率只有 32.4%？

**前置知识与分析过程**：先在等stage、等F/B假设下推公式，再逐项加入imbalance、P2P、小kernel和同步。理论值只作为上界。

**主问题得分点（10分）**：公式/假设 4 分；正确数值 2 分；至少三项现实损失 3 分；profile验证 1 分。

**追问分析与参考回答**：理论只扣fill/drain；本机PP8 endpoint参数max/min=2.76x，慢stage拉长节拍，PCIe P2P和每stage更少layers带来通信/launch开销，F/B时长也非相等。应采per-stage F/B/P2P/idle trace，用实际service time模拟上界。

**追问得分点（5分）**：bubble计算 1 分；2.76x imbalance 1 分；P2P/小kernel 1 分；真实timeline方案 2 分。

### M045. microbatch 数越多越好吗？

**高分回答要点**

- 更多 M 降低 bubble、增加 steady-state 区间，但若 microbatch size 固定会增大 global batch/step latency。
- 保存 activation、调度/P2P 次数、data iterator 和 optimizer update 间隔也会变化。
- microbatch 过小使 GEMM 低效，过多会受显存和优化语义限制，需与 global batch/accumulation 联合选择。

**追问**：公平比较 M 时应固定 global batch，还是允许变化？取决于研究问题，报告必须明确；本机跨 M 同时改变了 global batch。

**前置知识与分析过程**：先写GBS公式并说明要研究“schedule效率”还是“可用吞吐配置”。再决定调M时是否反向调MBS/DP/accum。

**主问题得分点（10分）**：M收益 2 分；GBS/优化语义 3 分；显存/step latency 2 分；公平实验设计 3 分。

**追问分析与参考回答**：研究纯bubble时固定GBS和每microbatch shape更难同时做到，可固定MBS并报告GBS变化，或调accum/DP构造等价样本量；关键是不能把更大GBS带来的GEMM/摊销收益全归因于bubble。本机结果明确跨M同时增GBS，所以只做机制证据。

**追问得分点（5分）**：识别研究问题 1 分；两种控制方案 2 分；指出混杂因素 1 分；准确限定本机结论 1 分。

### M046. PP stage 之间传输哪些 tensor？

**高分回答要点**

- forward 发送 boundary activation，backward 反向发送该 activation 的 gradient；shape 由 sequence、microbatch、hidden 和 SP/CP layout 决定。
- encoder-decoder、多输出、skip connection 或 packed sequence 可能需要多个 tensor/shape metadata。
- dtype、variable sequence 和通信 batching 决定 P2P buffer 与协议，不能统一按参数 bytes 估算。

**追问**：基础 GPT boundary `[s,b,h]` FP16 每次 bytes 是 `s*b*h*2`，forward/backward 每 microbatch 各一次。

**前置知识与分析过程**：从stage边界module输出定义tensor列表，逐个算numel*dtype；乘方向、microbatches和steps。再考虑SP/CP、variable sequence和metadata。

**主问题得分点（10分）**：F activation/B grad 3 分；shape/bytes 2 分；复杂模型多个tensor 2 分；dtype/变长/buffer 3 分。

**追问分析与参考回答**：单边界每microbatch forward发送`s*b*h*2` bytes，backward发送同shape grad，总logical payload约两倍；整step再乘M和`P-1`边界。实际链路还有协议，SP时s可为local shard，CP/packed sequence需按真实layout和动态shape计算。

**追问得分点（5分）**：单次bytes 1 分；双方向 1 分；M/边界次数 1 分；layout变化 1 分；logical与物理口径 1 分。

### M047. `pre_process/post_process` 如何让每个 PP rank 只构建自己的模型？

**高分回答要点**

- first stage 构建 embedding，last stage 构建 output/loss，中间 stage 仅构建分配到的 Transformer layers。
- 每 rank 的 layer offset/数量由 PP/VPP layout 计算，forward 输入在 first stage 是 token，在其他 stage 是上游 activation。
- 参数和显存不均衡常来自端点模块，而不是 layer 数算错。

**追问**：本机 PP8 最大 stage 39.1M、最小 14.2M 参数，正是 embedding/output 集中导致。

**前置知识与分析过程**：先按layer offset构建stage local module，再额外放置embedding/output/loss。分别统计参数、FLOPs和activation，避免只看layer数。

**主问题得分点（10分）**：pre/post构建 3 分；输入类型 2 分；endpoint不均衡 2 分；参数以外成本与验证 3 分。

**追问分析与参考回答**：PP8等分16层时每stage约2层，但stage0还含vocab embedding，末stage含大output projection，故参数比中间stage高。若tie与vocab-parallel设置不同，首尾分布还会变化。应记录local parameter counts和F/B时间后重切层。

**追问得分点（5分）**：2层基线 1 分；首尾额外模块 2 分；配置影响 1 分；用时间而非参数最终决策 1 分。

### M048. loss 为什么通常只在 pipeline last stage 计算？

**高分回答要点**

- 只有末 stage 持 output projection/logits；loss closure 在那里产生 scalar 与 logits gradient。
- 训练日志需要把 loss 广播/归约到记录 rank，但不应让每个 stage 重复完整词表计算。
- loss scaling 必须与 microbatch 数、DP 和 mixed precision 一致，否则 gradient 大小错误。

**追问**：本机脚本为何把末 stage loss 通过 `all_reduce(MAX)` 带回 rank0？用于结果记录，不是训练 gradient 通信。

**前置知识与分析过程**：区分loss计算、日志传播和gradient生成三条路径。追踪哪个stage持logits、哪个rank写JSON，以及该collective是否参与autograd。

**主问题得分点（10分）**：last-stage loss 3 分；日志/训练区别 3 分；scale与microbatch 2 分；词表/TP边界 2 分。

**追问分析与参考回答**：只有last stage有非零final loss；脚本在autograd/optimizer之后把scalar放tensor并MAX到所有rank，使rank0可写结果。它不连接计算图，也不改变gradient。生产通常reduce loss sum/count或broadcast，MAX只是该教学脚本利用其他rank为0的记录技巧。

**追问得分点（5分）**：非零owner 1 分；不在autograd 2 分；MAX技巧 1 分；生产替代 1 分。

### M049. 如何定位 pipeline stage imbalance？

**高分回答要点**

- 记录每 stage 每 microbatch forward/backward/P2P 时间、空闲区间、参数/activation、GEMM shape和最慢 rank。
- 区分静态层成本、embedding/output、MoE token 动态偏斜、通信拓扑和系统 straggler。
- 用实际 stage service time 重算 schedule critical path，再做非均匀切层对照。

**追问**：参数数只是 proxy；attention/MLP FLOPs、MoE 路由和通信可让参数少的 stage 更慢。

**前置知识与分析过程**：对每stage建立service-time模型：F compute、B compute、P2P、动态MoE和系统等待。先观察再重切，而非按参数静态猜。

**主问题得分点（10分）**：指标列表 3 分；四类根因 3 分；critical path重算 2 分；对照实验 2 分。

**追问分析与参考回答**：采每microbatch的stage F/B start/end、P2P和token counts；求steady-state p50/p95 service time及最慢stage。按实际时间移动一层或用custom layout，再保持GBS/M不变复跑；若bubble/idle和step下降才证明balance优化有效。

**追问得分点（5分）**：per-microbatch trace 2 分；动态因素 1 分；重切对照 1 分；成功指标 1 分。

### M050. Custom Pipeline Layout 解决什么问题？

**高分回答要点**

- 新版 MCore 可显式描述 embedding、decoder、MTP、loss 等在 PP/VPP stage 的分配，处理非对称模型和端点成本。
- layout 应由 profile service time而非等层数生成，并校验 layer 总数、顺序、shared weight 与空 stage。
- 它降低 imbalance，不自动减少 P2P 或 bubble；错误 layout 还会增加跨 stage 依赖。

**追问**：给 PP8 的端点 2.76x imbalance，让你设计重切层实验并定义成功指标。

**前置知识与分析过程**：先用baseline trace估各module cost，再满足层顺序/共享权重约束地搜索layout。不要直接按参数反比移层。

**主问题得分点（10分）**：custom layout语义 3 分；profile驱动 2 分；合法性约束 2 分；指标与风险 3 分。

**追问分析与参考回答**：在stage0/7保留embedding/output但减少decoder层，把层移给中间较快stage；候选layout逐一计算每stage预计F/B与P2P。固定PP8/M16/GBS，比较max stage service、idle/bubble、tok/s和peak；loss/grad与baseline一致。若VPP可用再做正交对照。

**追问得分点（5分）**：端点减层 1 分；固定控制变量 1 分；时间/idle指标 2 分；correctness 1 分。

### M051. Virtual Pipeline Parallelism/Interleaving 如何降低 bubble？

**高分回答要点**

- 每个物理 PP rank 持多个较小 model chunks，schedule 在 virtual stages 间交错，缩短逻辑 stage 粒度。
- 可降低 bubble/改善负载均衡，但增加 P2P 次数、模型 chunk 切换、activation bookkeeping 和 schedule 复杂度。
- 需要 layer 数/layout 可分、足够 microbatches，并检查通信是否抵消收益。

**追问**：VPP 不是增加 GPU；它在同 rank 上重排 chunks 与时序。

**前置知识与分析过程**：把physical PP rank和virtual stage分开画。分析逻辑stage粒度缩小后bubble、P2P次数与activation bookkeeping如何变化。

**主问题得分点（10分）**：VPP定义 3 分；降低bubble机制 2 分；额外成本 3 分；适用约束 2 分。

**追问分析与参考回答**：例如每个物理rank持两个不连续layer chunks，schedule交错执行，使相邻逻辑stage更细、更快轮转；GPU数和物理PP仍不变。代价是更多chunk边界/P2P、切换与activation状态，只有计算窗口足够且layout平衡时才获益。

**追问得分点（5分）**：物理/虚拟区别 2 分；细粒度机制 1 分；成本 1 分；适用条件 1 分。

### M052. PP 中 activation memory 由什么决定？

**高分回答要点**

- 取决于本地 layers、warmup 时并存 microbatches、每层保存 tensor、sequence/microbatch 和 recompute。
- 不同 PP rank warmup 深度不同，所以峰值可能不在参数最多 stage。
- 1F1B、interleaving、deallocate output、offloading 会改变 lifetime，必须看每 rank peak。

**追问**：为什么只用“模型参数/PP”预测显存会错？activation、端点和 optimizer 都不等分。

**前置知识与分析过程**：按常驻、microbatch activation、临时P2P/workspace三类列每rank状态，并结合warmup深度求同时存活数量。

**主问题得分点（10分）**：activation决定项 3 分；rank warmup差异 2 分；端点/optimizer 2 分；测量方法 3 分。

**追问分析与参考回答**：参数只按本地层和端点分布；stage0可能同时保存更多未反传microbatch activation，末stage有logits/loss workspace；optimizer按本地参数，P2P buffer和allocator也非均匀。应在warmup后reset peak并取所有rank max，同时按stage记录snapshot。

**追问得分点（5分）**：列三类非均匀项 2 分；warmup lifetime 1 分；max-rank口径 1 分；snapshot 1 分。

### M053. PP P2P 如何避免死锁和顺序错误？

**高分回答要点**

- 相邻 ranks 必须以 schedule 规定的 send/recv 顺序、shape/dtype 和 communicator 执行，常通过批量双向 P2P 避免环形等待。
- 异步 request/stream 的 lifetime 要覆盖 buffer 使用，发送 buffer 不可过早复用。
- 发生 hang 时对齐 stage、microbatch、F/B direction、sequence number 和 tensor shape。

**追问**：某 stage 因 empty batch 跳过 forward，而邻居仍 recv，会出现什么？控制流必须全局一致。

**前置知识与分析过程**：先写schedule中相邻send/recv配对表，再考虑blocking/async API和buffer lifetime。故障分析找第一个不匹配的microbatch/direction。

**主问题得分点（10分）**：配对顺序 3 分；async lifetime 2 分；hang定位字段 3 分；异常路径一致性 2 分。

**追问分析与参考回答**：下游已post recv但上游跳过send，会一直等待直至timeout；其他stage随后也被背压卡住。empty/skip决策应在进入schedule前全局同步，或仍发送协议规定的empty tensor/metadata。日志记录stage、microbatch、F/B、shape和sequence。

**追问得分点（5分）**：直接hang机制 2 分；传播 1 分；两种修复 1 分；日志字段 1 分。

### M054. 如何解释 PP8/M16 只有 2.59x speedup？

**高分回答要点**

- 理论 bubble 已有 30.4%，相对 PP1 理想上限远低于 8x；stage 参数 max/min 又达 2.76x。
- 每 microbatch 有 P2P、schedule/launch，局部 layer 变少后 GPU kernel 更短；慢 stage 决定节拍。
- 本机无 NVLink，8 卡 SendRecv aggregate/latency 也受 PCIe 限制。

**追问**：下一步需要 per-stage trace+custom layout，而不是继续把 M 增到更大并宣称问题解决。

**前置知识与分析过程**：用理论bubble给上界，再从stage不均衡、P2P和local kernel三项解释上界缺口。每个假设对应一组指标/对照。

**主问题得分点（10分）**：本机数字 3 分；三类损失 3 分；不夸大M收益 2 分；下一实验 2 分。

**追问分析与参考回答**：固定PP8/M16采8rank trace，求最慢stage F/B和其他stage idle；根据service time重切层，再比较tok/s/bubble/peak。单独测相邻P2P并用PP1/小stage kernel判断通信/粒度。若layout后仍慢，再评估VPP或拓扑，而非先增M改变GBS。

**追问得分点（5分）**：固定baseline 1 分；trace 1 分；重切对照 1 分；P2P/kernel分解 1 分；不混GBS 1 分。

---

## 6. Level 4：Megatron DDP、Distributed Optimizer 与 FSDP（M055-M068）

### M055. Megatron DDP 与 PyTorch DDP 的核心实现差异是什么？

**高分回答要点**

- Megatron DDP 围绕 contiguous param/main-grad buffers、bucket groups 和 MCore optimizer/layout 深度集成。
- 支持 gradient AllReduce 或 distributed optimizer 的 ReduceScatter，并提供 grad-reduce/param-gather overlap。
- 还要处理 expert/dense 参数不同 group、PP tied embedding 和模型并行 gradient finalization。

**追问**：本机 TP 教学脚本用了 TorchDDP，因此不能声称测过 MCore DDP overlap/distributed optimizer。

**前置知识与分析过程**：先比较parameter/grad存储模型、reducer触发方式和optimizer集成，再列MCore额外处理的并行域。最后把本机代码实际wrapper作为证据边界。

**主问题得分点（10分）**：contiguous buffer 2 分；bucket/RS/AR 2 分；optimizer overlap 2 分；特殊group finalization 2 分；本机边界 2 分。

**追问分析与参考回答**：查看脚本可见`TorchDDP(model, process_group=data_parallel_group)`和Apex FusedAdam，没有构造MCore DDP config、DistributedOptimizer或调用其start/finish sync。因此实测只说明MCore模型与TorchDDP组合可跑，MCore buffer/overlap需另做实验。

**追问得分点（5分）**：指出实际类 1 分；缺失MCore对象 2 分；可陈述范围 1 分；建议补实验 1 分。

### M056. `param.main_grad` 和 contiguous grad buffer 为什么重要？

**高分回答要点**

- 参数 backward hook 把 gradient 累积到预分配连续 buffer 的 view，减少大量小 allocation/copy。
- 连续 bucket 便于发起大 collective、按 dtype/group 切分，并与 optimizer shard 对齐。
- view/storage lifetime 与 zero_grad/offload/restore 紧耦合，不能随意替换 `.grad` 或 resize buffer。

**追问**：如何检查一个参数的 main_grad offset、bucket ownership 和通信完成状态？

**前置知识与分析过程**：先区分Parameter对象、`.grad`和`.main_grad` view；再从参数到dtype buffer、bucket group和通信handle建立映射。

**主问题得分点（10分）**：main_grad目的 3 分；view/storage语义 2 分；bucket对齐 2 分；lifetime/调试 3 分。

**追问分析与参考回答**：读取param到buffer index/offset metadata，确认`main_grad.storage`与目标grad buffer共享；打印bucket参数列表、range和ready计数。backward后检查该range checksum，`start_grad_sync`后保存Work状态，`finish_grad_sync`后与reference归约结果比较。不要访问已offload/resize storage。

**追问得分点（5分）**：offset/共享storage 2 分；bucket metadata 1 分；ready/Work 1 分；reference checksum 1 分。

### M057. `overlap_grad_reduce` 如何工作？

**高分回答要点**

- backward hook 标记 bucket 中参数 gradient ready；当最后一个所需 grad 就绪时异步发 AllReduce/ReduceScatter。
- 后面更早层的 backward compute 可与已 ready bucket 通信重叠，iteration 末 `finish_grad_sync()` 等待。
- bucket 太大推迟启动，太小增加 launch/协议开销；执行顺序和 unused/conditional parameter 必须正确。

**追问**：需要看 bucket ready timestamp、NCCL union/overlap/exposed time和 step wall，而不是只数 collective。

**前置知识与分析过程**：画backward参数ready顺序，将参数映射到buckets；计算首个/每个bucket何时满足ready，再与通信stream和compute区间相交。

**主问题得分点（10分）**：hook/ready机制 3 分；bucket trade-off 2 分；finish wait 2 分；正确overlap指标 3 分。

**追问分析与参考回答**：记录每bucket最后grad ready时刻和NCCL开始/结束，先对NCCL区间做union，再与非NCCL compute union求intersection；`exposed=NCCL union-overlap`。最终用max-rank step wall验证exposed下降是否转成端到端收益。

**追问得分点（5分）**：ready timestamp 1 分；union 1 分；intersection/exposed 2 分；step wall 1 分。

### M058. gradient accumulation 时何时应该同步 DP gradient？

**高分回答要点**

- 前 `M-1` microbatches 只本地累积，最后一个 microbatch 完成后才触发 DP sync，避免每次重复整套通信。
- MCore schedule/DDP 通过 `no_sync` 或 grad sync enable/disable 协调；loss 要除 accumulation 或统一 scale。
- TP/SP/CP 的某些通信是层计算所必需，不能因 DP accumulation 全部关闭。

**追问**：区分 data-parallel gradient sync 与 tensor-parallel layer collective，否则会错误延迟必要的 dX/activation 汇合。

**前置知识与分析过程**：按通信是否属于“当前layer数学正确性”分类。TP/PP/CP许多通信必须每microbatch发生；DP只在累积完成后同步复制模型grad。

**主问题得分点（10分）**：最后microbatch同步 3 分；loss scaling 2 分；不同并行通信分类 3 分；实现控制点 2 分。

**追问分析与参考回答**：DP `no_sync`只阻止DP bucket AR/RS，TP的row output归约、column dX归约和PP P2P仍是完成该microbatch F/B所必需。把这些也延迟会让后续算子读到分片contribution而非逻辑tensor，结果错误或hang。

**追问得分点（5分）**：DP可延迟 1 分；列两个TP必需通信 2 分；PP/CP例子 1 分；错误后果 1 分。

### M059. `finalize_model_grads()` 为什么存在？

**高分回答要点**

- 普通 DP bucket sync 后，仍可能有 SP LayerNorm 复制参数、PP tied embedding、CP duplication、expert 参数等特殊 gradient。
- finalize 按各自 process group 完成这些同步，并做必要 token/loss scale，确保 optimizer 前状态完整。
- 新并行功能若忘记接入 finalization，常表现为 finite 但不同 rank 参数逐步漂移。

**追问**：要求列出一个 dense GPT 开 SP/PP 后普通 DP reducer 覆盖不到的 gradient。

**前置知识与分析过程**：逐类判断参数是DP复制、TP复制、PP首尾共享还是expert专属。普通DP reducer只覆盖相同model-parallel坐标的复制参数。

**主问题得分点（10分）**：finalize存在原因 3 分；至少三类特殊grad 3 分；正确group 2 分；静默漂移风险 2 分。

**追问分析与参考回答**：SP LayerNorm参数在TP ranks复制但各看不同sequence shard，需TP group归约；PP tied embedding首尾两份需embedding group归约。这两者都不由普通DP group连接。可再举CP复制weights或MoE expert-DP。

**追问得分点（5分）**：SP例子2分；PP例子2分；明确DP为何不覆盖1分。

### M060. Distributed Optimizer 的数据流是什么？

**高分回答要点**

- backward 生成连续 main-grad，DP group ReduceScatter 后每 rank 只持自己的 gradient shard。
- 每 rank 只更新对应 FP32 main params 与 optimizer moments shard。
- 更新后 Parameter AllGather 恢复各 DP rank 下一轮 forward 所需的低精度完整模型参数。

**追问**：它接近 ZeRO-1 的状态分片并结合 gradient RS/param AG；要按实际 MCore版本和配置描述，不只贴阶段标签。

**前置知识与分析过程**：从完整DP Adam状态出发，按时间顺序追grad产生、RS ownership、local update和AG。对每个阶段标哪类状态常驻/临时。

**主问题得分点（10分）**：RS 2 分；optimizer shard 2 分；AG 2 分；buffer/state ownership 2 分；版本/ZeRO标签边界 2 分。

**追问分析与参考回答**：回答应说实际数据流而非只说ZeRO-1：MCore contiguous main grads按DP range做ReduceScatter，rank只更新对应FP32 master/m/v，更新后AllGather低精度模型参数。不同版本对grad常驻、overlap和checkpoint格式细节有变化，因此以目标commit源码为准。

**追问得分点（5分）**：RS->update->AG顺序 3 分；状态owner 1 分；版本边界 1 分。

### M061. 如何推导 distributed optimizer 每参数显存？

**高分回答要点**

- 先列低精度 model param、main grad、FP32 main param、Adam m/v 和可能的 model grad，而后识别哪些复制/按 DP=d 分片。
- 当前官方示例中 FP16 param+grad 从非分布式约 20 B/param 变为 `4+16/d`；BF16 param+FP32 grad 为 `6+12/d`。
- 理论公式不含 fragmentation、bucket padding、activation、临时 AG 和 communication buffers。

**追问**：d 增大后为何下界不是 0？低精度 model param/部分 grad 仍需复制或常驻。

**前置知识与分析过程**：逐项列bytes并写`replicated + sharded/d`，不要直接背总公式；再说明公式外状态。

**主问题得分点（10分）**：状态拆分 3 分；官方三种dtype公式任一正确 2 分；d极限解释 2 分；额外显存边界 3 分。

**追问分析与参考回答**：以FP16 param/grad为例，官方理论为`4+16/d` B/param；d趋无穷时分片main param/m/v等趋零，但forward需要的低精度model param及相关常驻项仍约4B，所以不为零。真实peak还含padding、bucket、AG、activation和allocator。

**追问得分点（5分）**：公式 2 分；极限 1 分；常驻项 1 分；现实额外项 1 分。

### M062. 为什么 Distributed Optimizer 要 ReduceScatter + AllGather，而不是 AllReduce？

**高分回答要点**

- AllReduce 后每 rank 得完整 gradient，无法直接省 optimizer state；RS 同时归约并把 ownership 分给 ranks。
- optimizer step 只更新 shard，随后 AG 新参数以供复制模型 forward。
- 算法 bytes 可与 AllReduce 同量级，但通信位置、buffer lifetime和可 overlap 窗口不同。

**追问**：RS 能与 backward overlap，AG 能与下一 iteration forward overlap，但会扩大 in-flight memory。

**前置知识与分析过程**：先从AllReduce=`RS+AG`理解数学等价，再把AG移到optimizer后参数同步路径。分别找RS和AG可隐藏的compute窗口。

**主问题得分点（10分）**：为何不用AR 3 分；RS ownership 2 分；AG恢复参数 2 分；overlap/peak权衡 3 分。

**追问分析与参考回答**：grad bucket ready后异步RS可与剩余backward重叠；local optimizer完成某param shard后可提前AG，和其他optimizer工作或下一轮较早module compute重叠。多bucket同时full params会提高allocated/reserved，first-use hook仍可能暴露等待。

**追问得分点（5分）**：RS窗口1分；AG窗口1分；first-use wait1分；in-flight peak1分；需trace验证1分。

### M063. `overlap_param_gather` 如何工作，风险是什么？

**高分回答要点**

- optimizer 更新 shard 后提前异步 AllGather后续 module/bucket 参数，forward pre-hook 在真正使用前等待。
- 可把 param sync 隐藏在 optimizer/forward compute 后，但需要 deterministic module order和足够计算窗口。
- 过度预取会增加 full-param buffer峰值；条件执行/多 model chunks 会使 hook/order 更复杂。

**追问**：profile 要看 AG 发起、first-use wait、exposed AG、allocated/reserved peak，而不只看 flag 开启。

**前置知识与分析过程**：对每param bucket画local update完成、AG发起、module first-use和AG完成四个时刻。收益由first-use前完成程度决定。

**主问题得分点（10分）**：pre-hook机制 3 分；deterministic order 2 分；峰值风险 2 分；四类指标 3 分。

**追问分析与参考回答**：记录AG start/end与对应module pre-hook wait；若AG早结束则隐藏成功，若first-use阻塞则该段是exposed。同步采allocator peak和in-flight bucket数，比较limit/prefetch深度。tok/s改善但OOM风险上升也要报告。

**追问得分点（5分）**：四时间点2分；exposed判断1分；显存1分；端到端trade-off1分。

### M064. `overlap_grad_reduce` 与 `overlap_param_gather` 有哪些配置依赖？

**高分回答要点**

- 前者需要 bucket 和异步 grad sync；后者通常依赖 distributed optimizer/参数 shard与 forward pre-hook。
- gradient accumulation、PP rank、bucket size、FP8 post-gather work和 CUDA Graph 可能改变合法组合。
- 开 flag 前先核对版本 config validation；错误组合可能退化为同步路径或改变内存峰值。

**追问**：专家回答应给“如何证明确实 overlap”，而不是复述参数名。

**前置知识与分析过程**：先列每个flag所需的数据结构/异步API，再检查组合是否由config允许。最后用消融与timeline证明，而非看日志显示True。

**主问题得分点（10分）**：两flag依赖 3 分；accum/PP/FP8等约束 2 分；退化/峰值风险 2 分；验证矩阵 3 分。

**追问分析与参考回答**：做baseline、只grad overlap、只param overlap、全开四组，固定GBS/shape；trace bucket ready、RS/AG与compute交集，报告exposed和peak。若全开比单开慢，再看链路/SM争用和in-flight buffers。目标commit的config validation是合法性来源。

**追问得分点（5分）**：四组消融2分；trace指标1分；争用/峰值1分；版本校验1分。

### M065. Distributed Optimizer 与 Megatron-FSDP 的边界是什么？

**高分回答要点**

- distributed optimizer 传统路径主要分片 optimizer/grad ownership，同时 forward 常使用复制的 model params。
- Megatron-FSDP 可按策略进一步分片 parameters、gradients、optimizer states，接近 ZeRO-3，并与 MCore模型/并行集成。
- FSDP 参数按需 AllGather/reshard，容量更强但 communication frequency、prefetch和临时峰值不同。

**追问**：当前官方支持 `optim`、`optim_grads`、`optim_grads_params` 等 sharding strategy；本机没有实测 Megatron-FSDP。

**前置知识与分析过程**：对参数、gradient、optimizer三类状态逐一问是否分片、何时materialize。用数据流而非ZeRO阶段名比较两条runtime。

**主问题得分点（10分）**：Distributed Optimizer边界 3 分；FSDP按需AG/reshard 3 分；容量/通信权衡 2 分；本机证据边界 2 分。

**追问分析与参考回答**：`optim`只切optimizer，`optim_grads`再切grad，`optim_grads_params`连params都切。后者forward前按unit/布局AG参数、之后reshard，通信频率和full-param峰值与传统dist optimizer的step级param同步不同。本机仅跑PyTorch FSDP和MCore0.9模型，未跑Megatron-FSDP。

**追问得分点（5分）**：三策略2分；按需参数AG1分；与dist optim区别1分；边界1分。

### M066. 在 TP/PP/CP/EP 已存在时，DP shard 的“参数”是什么？

**高分回答要点**

- DP 复制的是每个 model-parallel rank 对应的本地模型 shard，不是完整全局模型。
- 同一 TP/PP/CP 坐标、不同 DP rank 的参数逻辑对应，才能做 DP reduce/shard。
- expert 参数可能使用 expert-DP group，dense 参数使用普通 DP group，ownership 不同。

**追问**：画 TP2/PP2/DP2 中 rank0 与哪个 rank 做 DP gradient sync，而不是与任意其他 rank。

**前置知识与分析过程**：先选rank线性化公式并写坐标，再固定TP/PP/CP等model坐标，只变化DP维构成对应组。

**主问题得分点（10分）**：DP复制local shard概念 3 分；坐标/group推导 3 分；expert例外 2 分；拓扑/owner 2 分。

**追问分析与参考回答**：若layout为`rank=((dp*PP)+pp)*TP+tp`，rank0坐标(dp0,pp0,tp0)，对应DP伙伴为rank4(dp1,pp0,tp0)，所以组(0,4)。rank1与5、2与6、3与7分别同步；rank0不应与rank1做DP，因为后者是不同TP shard。

**追问得分点（5分）**：声明layout1分；rank0坐标1分；组(0,4)2分；解释非rank1 1分。

### M067. Megatron 如何计算 global grad norm 和 clipping？

**高分回答要点**

- 每 rank 对自己拥有且不重复计数的 parameter grad 求局部 norm，再按 TP/PP/DP/expert ownership做正确归约。
- shared/tied/复制参数必须避免重复计入；distributed optimizer shard只持部分 grad。
- unscale/overflow 检查应在 clip 前，clip factor 必须对所有相关 ranks 一致。

**追问**：若每 rank 独立 clip 本地 shard，会改变全局优化方向。

**前置知识与分析过程**：先定义全局norm平方和，识别哪些parameter shard唯一、哪些副本应只计一次，再按相应group做SUM。

**主问题得分点（10分）**：global norm公式 2 分；ownership去重 3 分；归约/clip顺序 3 分；mixed precision位置 2 分。

**追问分析与参考回答**：各rank算owned grads的`sumsq`，跨覆盖完整模型的必要groups SUM后开方得到同一global norm，clip系数`min(1,max_norm/(norm+eps))`广播/一致应用。若各rank用local norm，不同shard系数不同，不再等价对全局gradient向量缩放。

**追问得分点（5分）**：sumsq公式1分；去重1分；global reduce1分；统一系数1分；错误后果1分。

### M068. mixed precision overflow 如何跨并行 ranks 保持一致？

**高分回答要点**

- 任一相关 rank 出现 inf/nan，都必须让整个逻辑 optimizer step 跳过并统一更新 loss scale。
- overflow flag 需跨 model/data/expert parallel 域归约；只在 rank0 检查会造成参数版本分叉。
- optimizer step、scheduler、consumed samples 和 checkpoint step 都要与 skip 决策一致。

**追问**：如何构造单 rank 注入 inf 的测试，验证所有 ranks 参数 hash 和 step counter都未推进？

**前置知识与分析过程**：沿unscale->found_inf归约->clip->step->scale update列状态机，定义原子step应推进的所有计数。

**主问题得分点（10分）**：全局overflow语义 3 分；归约域 2 分；step/scheduler/data一致性 3 分；故障测试 2 分。

**追问分析与参考回答**：固定step在一个TP/DP rank的某grad写inf；所有rank做found_inf MAX/SUM后应共同skip optimizer，LR scheduler/iteration consumed samples按框架语义不误推进，loss scale统一下降。比较注入前后每ranklogical parameter checksum及optimizer state hash。

**追问得分点（5分）**：单rank注入1分；全局flag1分；共同skip1分；计数/scale1分；hash oracle1分。

---

## 7. Level 5：Context Parallel、重计算与长序列（M069-M080）

### M069. SP 与 CP 的本质区别是什么？

**高分回答要点**

- SP 主要分片 TP 区域内 LayerNorm/dropout 等可逐 token activation；attention 仍按完整 context 语义执行。
- CP 分片网络输入和全层 activation 的 sequence 维；attention 必须跨 CP ranks交换 KV/中间统计以覆盖全局 context。
- SP 通常与 TP layout绑定；CP 是独立并行维，可与 TP/PP/DP组合。

**追问**：本机只实测 SP，不应把 seq4096 结果称为 CP 性能。

**前置知识与分析过程**：先问“sequence维在哪些module保持分片”，再检查attention是否需要全局K/V。以此区分SP的局部分片和CP的全网络sequence分片。

**主问题得分点（10分）**：SP范围 3 分；CP范围 3 分；attention通信 2 分；组合/证据边界 2 分。

**追问分析与参考回答**：本机config只设置`sequence_parallel=True`并初始化TP/PP，没有`context_parallel_size`或CP attention通信；seq4096测到的是SP对norm/dropout等activation的影响和full recompute。CP需另建CP groups并验证attention K/V交换，因此不能用现有数字外推。

**追问得分点（5分）**：指出实际flag 1 分；缺少CP group 1 分；缺少attention通信 1 分；正确称谓 1 分；补实验 1 分。

### M070. TP2/CP2 时 process group 如何构造？

**高分回答要点**

- 每个 sequence chunk 由一个 TP2 group处理，例如 GPU0-1 与 GPU2-3；对应 TP lane 跨 chunks形成 CP groups，如 `(0,2)`、`(1,3)`。
- CP ranks复制 weights，但分片 tokens/activations，因此 weight gradient需跨 CP 对应域归并，官方实现可把 CP并入数据并行同步域。
- rank mapping应让 TP和CP高频通信位于适合的NVLink/NVSwitch层级。

**追问**：给 TP4/CP2 的 8 rank，手写四个 CP groups。

**前置知识与分析过程**：先形成两个TP4 groups处理两个sequence chunks，再固定TP lane、跨chunk连接CP伙伴。权重复制和grad同步另行标出。

**主问题得分点（10分）**：TP/CP group构造 4 分；token/weight ownership 2 分；grad同步 2 分；拓扑 2 分。

**追问分析与参考回答**：可令TP groups为(0,1,2,3)和(4,5,6,7)，分别处理chunk0/1；CP groups是(0,4)、(1,5)、(2,6)、(3,7)。每个CP pair在同一TP lane上交换context，weights在CP维复制，weight grads需跨相应复制域汇总。

**追问得分点（5分）**：两个TP组1分；四个CP组2分；chunk语义1分；grad复制1分。

### M071. CP attention 为什么必须通信？

**高分回答要点**

- 本地 Q 对应局部 tokens，但 causal/noncausal attention 需要与全序列 K/V交互。
- 可通过 ring P2P传递 KV blocks，或 AllGather/A2A等布局变换；不同方法的内存、通信和 overlap不同。
- softmax 需要在分块计算中保持 global max/sum数值等价，causal mask还影响每 rank有效工作量。

**追问**：只 gather K/V 后本地 attention容易理解，但峰值高；ring可流式计算却增加调度复杂度。

**前置知识与分析过程**：从单个local Q需要所有K/V的数学式出发，再比较一次性materialize和分块在线softmax两种实现。

**主问题得分点（10分）**：全局依赖 3 分；至少两种通信方案 2 分；global softmax正确性 3 分；causal/负载 2 分。

**追问分析与参考回答**：AllGather K/V后每rank直接算local Q对full K/V，简单但峰值约随CP聚合；ring逐块传K/V，维护running max、exp sum和weighted value，用online softmax合并，内存低但有CP-1轮P2P、stream依赖和causal无效块处理。

**追问得分点（5分）**：AG方案1分；ring方案1分；online softmax2分；causal/调度1分。

### M072. 如何选择 CP communication type？

**高分回答要点**

- 比较 sequence、head/GQA布局、CP degree、节点拓扑、attention kernel与是否能 overlap。
- P2P/ring降低 full KV驻留但多轮通信；AllGather简单但峰值大；A2A/hybrid改变 head/sequence layout并依赖硬件层级。
- 具体支持类型和约束随MCore/Transformer Engine版本变化，配置前必须查当前文档与源码校验。

**追问**：给两层网络拓扑时，hierarchical CP 应把哪种通信放节点内/节点间？先按 bytes与轮数建模再回答。

**前置知识与分析过程**：列候选通信的消息bytes、轮数、峰值与可overlap，再映射节点内高速域和节点间NIC。不能只按名字指定。

**主问题得分点（10分）**：选择维度 4 分；各方案trade-off 3 分；版本约束 1 分；hierarchical mapping 2 分。

**追问分析与参考回答**：一般把多轮、延迟敏感或高频交换放节点内NVLink域，把跨节点通信尽量聚合为较少大消息；例如节点内A2A重排、节点间P2P/聚合只是候选。应按目标MCore支持的hybrid模式、GQA layout和实际rail做microbench+端到端验证。

**追问得分点（5分）**：bytes/轮数2分；内外层原则1分；不武断指定1分；版本/实验1分。

### M073. GQA/MQA 为什么会影响 CP 的负载和通信？

**高分回答要点**

- KV heads少于Q heads，KV通信bytes下降，但head到rank的映射可能造成复制或不均匀。
- ring/A2A实现对query groups和TP/CP可整除性有约束，局部attention kernel shape也变化。
- 应分别估算Q compute、KV bytes、softmax和每rank有效token，而不是沿用MHA公式。

**追问**：长上下文容量规划中，训练activation与推理KV cache不要混为同一对象。

**前置知识与分析过程**：分别写Q heads、KV groups和head_dim，再计算训练K/V activation通信；另行说明推理cache跨layer/token长期驻留。

**主问题得分点（10分）**：GQA对KV bytes 3 分；TP/CP映射 3 分；kernel/负载 2 分；训练/推理边界 2 分。

**追问分析与参考回答**：训练CP每layer需让local Q访问全context K/V，bytes与KV groups成正相关；MQA/GQA可降通信但groups<TP/CP时可能复制。推理KV cache则为每已生成token在所有layers保存K/V，容量公式和调度生命周期不同，不能拿训练activation peak直接估算服务并发。

**追问得分点（5分）**：训练bytes1分；groups映射1分；推理长期cache2分；明确不可混用1分。

### M074. full recompute、uniform 和 block 方法分别表示什么？

**高分回答要点**

- full granularity对指定Transformer层/段只保存输入，backward时重跑完整layer forward。
- uniform按固定层数分组checkpoint；block可只对某段/若干层应用，控制重算范围。
- 保存边界、RNG和PP stage局部layer数必须一致；配置语义随版本需核对。

**追问**：本机使用 full+uniform+每组1层，seq4096显存省约78%、吞吐损失约28%-30%。

**前置知识与分析过程**：区分granularity决定“重算什么”，method决定“如何分段”，num_layers决定分组大小。再沿F/B说明保存与重算。

**主问题得分点（10分）**：三个配置维度 4 分；RNG/边界 2 分；显存/FLOPs trade-off 2 分；本机数据 2 分。

**追问分析与参考回答**：full表示checkpoint完整Transformer layer输入，uniform按固定层数均匀分组，num_layers=1意味着逐层checkpoint；backward恢复输入/RNG并重跑该层forward。本机TP2/4 seq4096峰值约省78%，tok/s下降约28%-30%，是该shape/版本结果。

**追问得分点（5分）**：三参数各1分；backward重算1分；准确数字和边界1分。

### M075. selective recompute 为什么可能优于 full recompute？

**高分回答要点**

- 不同activation的bytes/FLOPs重算比不同；选择保存昂贵算子、重算低成本高存储算子可得到更优Pareto点。
- attention、MLP、norm、MoE permutation等模块可有不同策略；新版MCore支持更细粒度模块选择。
- 需用saved tensor/memory snapshot与operator profile选对象，不能只按module名称猜。

**追问**：评价指标是可行最大batch/sequence、tokens/s、MFU、peak和数值正确性。

**前置知识与分析过程**：为每类saved activation计算bytes与重算FLOPs，按`bytes saved / extra FLOPs`排序，再考虑dependency和kernel融合。

**主问题得分点（10分）**：selective原理 3 分；模块候选 2 分；profile选择 3 分；完整指标 2 分。

**追问分析与参考回答**：用saved-tensor hooks/memory snapshot找占峰值的大tensor，用operator profile算重算成本；优先重算高bytes、低FLOPs且易重建的模块，保留昂贵attention/GEMM中间状态视实现而定。做full/selective/off三组，固定GBS和loss验证Pareto。

**追问得分点（5分）**：bytes/FLOPs比2分；profile工具1分；三组对照1分；correctness1分。

### M076. activation offloading 与 recompute 如何比较？

**高分回答要点**

- recompute用额外GPU FLOPs换显存；offload用CPU内存和PCIe/NVLink-C2C带宽换显存。
- offload需安排D2H/H2D预取和pinned buffer，若无法隐藏会直接进入关键路径。
- 可对不同模块细粒度组合，但要限制in-flight offload避免host/GPU双侧峰值。

**追问**：在本机PCIe V100上offload可能比NVLink-C2C平台更难隐藏，不能照搬新硬件结论。

**前置知识与分析过程**：写offload tensor bytes/PCIe带宽得到传输下界，与可用compute窗口比较；再算host和GPU两侧in-flight peak。

**主问题得分点（10分）**：recompute/offload资源交换 3 分；异步预取 2 分；峰值/带宽 2 分；硬件边界与组合 3 分。

**追问分析与参考回答**：V100 PCIe可用带宽远低于NVLink-C2C，若D2H+H2D时间大于相邻layer compute就无法隐藏。应先测pinned异步带宽，限制max in-flight，对指定activation做offload/recompute消融；报告exposed copy、host peak和tok/s，而非依据新平台文档推断。

**追问得分点（5分）**：传输下界1分；compute窗口1分；in-flight/host peak1分；实测方案1分；不外推1分。

### M077. sequence packing 给 Megatron 带来哪些收益与正确性风险？

**高分回答要点**

- 把多个短样本拼成固定长度，减少padding，提高有效tokens和GEMM/attention利用率。
- 必须提供边界/position/cumulative lengths，使attention不能跨样本泄漏，loss mask也要排除padding/边界。
- CP/PP和variable sequence会改变每rank token分配、P2P shape与负载均衡。

**追问**：correctness test应构造两个可辨识样本，验证packed输出与分别运行在有效token上一致。

**前置知识与分析过程**：先明确packing只优化padding，不允许跨样本attention/loss。逐字段检查position、mask、cu_seqlens和loss mask。

**主问题得分点（10分）**：收益 2 分；四类metadata 3 分；并行负载影响 2 分；correctness/恢复 3 分。

**追问分析与参考回答**：构造A/B token分布明显不同，分别运行得到有效token logits；pack后使用block-diagonal attention边界和各自position，拆出对应logits比较。再故意去掉边界应观察交叉污染，从而证明oracle敏感。检查loss只统计各样本目标token。

**追问得分点（5分）**：独立reference1分；边界mask1分；position1分；loss mask1分；负对照1分。

### M078. global batch、microbatch、DP和num_microbatches如何联动？

**高分回答要点**

- 基本关系 `GBS=MBS x DP x num_microbatches`；TP/PP/CP不直接增加独立数据样本数。
- CP将同一sequence分片，不应算成更多batch；PP ranks处理同一microbatch的不同层。
- 修改并行配置恢复训练时，需保持consumed samples/tokens和LR schedule语义。

**追问**：本机TP2/DP4/accum2/MBS2得到GBS16；TP4/DP2/accum4仍是16。

**前置知识与分析过程**：先标哪些维度产生独立样本，写公式后代值。恢复/改并行时再把consumed samples与LR schedule纳入。

**主问题得分点（10分）**：公式 3 分；排除TP/PP/CP 3 分；两组实例 2 分；恢复语义 2 分。

**追问分析与参考回答**：第一组`2*2*4=16`，第二组`2*4*2=16`。TP/PP/CP中的ranks合作处理相同逻辑samples，不乘入GBS。虽然GBS相同，local GEMM、通信、microstep数和浮点顺序不同，所以系统成本不等价。

**追问得分点（5分）**：两次计算2分；维度语义1分；系统不等价2分。

### M079. 长序列时为什么吞吐可能先升后降？

**高分回答要点**

- 从短到中等sequence，GEMM变大、固定通信/launch被摊薄，tokens/s可能上升。
- 更长时attention二次复杂度、HBM/workspace和memory pressure主导，tokens/s下降并接近OOM。
- 本机TP2从seq512约51.5k升至2048约82.5k，再在4096降至54.1k，是典型转折。

**追问**：不能只用tokens/s判断time-to-train，还需samples/s、FLOPs、MFU和目标数据序列分布。

**前置知识与分析过程**：把每step固定开销、线性GEMM和二次attention三项随s的增长写出，再解释tok/s分母/分子共同变化。

**主问题得分点（10分）**：先升机制 3 分；后降机制 3 分；本机转折数字 2 分；指标边界 2 分。

**追问分析与参考回答**：短s时launch/TP通信等固定成本按token摊销高，中等s增大M维使GEMM效率提升；长s时attention`s^2`、workspace和memory压力超过摊销收益。比较配置需同时报tokens/s、sequences/s、executed/useful FLOPs和真实length分布。

**追问得分点（5分）**：固定成本1分；GEMM效率1分；二次attention1分；本机趋势1分；多指标1分。

### M080. 如何为长上下文选择 TP、CP、SP、recompute？

**高分回答要点**

- 先拆参数、线性activation、attention activation/workspace和临时buffer，确定OOM主项。
- 参数/layer放不下才增TP/PP/FSDP；长sequence activation优先评估CP、selective recompute和高效attention；SP补逐token复制项。
- 用最小必要并行度保护GEMM shape和通信，拓扑上TP/CP放快域，并做容量/吞吐Pareto sweep。

**追问**：本机例子若只为放下seq4096，TP2+recompute比增TP或叠SP更直接；这不是所有模型的固定答案。

**前置知识与分析过程**：先做OOM归因，再给每种手段对应的状态对象和代价。按“满足容量的最小通信/计算损失”搜索Pareto。

**主问题得分点（10分）**：状态拆分 3 分；四类手段适用对象 3 分；拓扑/最小并行 2 分；本机与外推边界 2 分。

**追问分析与参考回答**：本机TP2 seq4096 baseline 26.28GiB可放但很高，full recompute降至5.77GiB且损失29.7%；SP仅降到25.95GiB，增TP4会降低参数/activation却使PCIe TP更重。因此若目标只是容量，TP2+recompute更直接；新模型若参数本身不放下或有高速互连，选择会变。

**追问得分点（5分）**：三组数字2分；代价比较1分；结论1分；不外推1分。

---

## 8. Level 6：MoE、Expert Parallel 与负载均衡（M081-M096）

### M081. Dense FFN 与 MoE layer 的核心差异是什么？

**高分回答要点**

- Dense FFN 每个 token 经过同一组参数；MoE router 为 token 选择 top-k experts，只激活总参数的一部分。
- 总参数/optimizer memory可很大，但每 token FLOPs由active experts决定；系统新增routing、permutation、dispatch/combine和负载不均衡。
- MoE扩容量不自动提高吞吐，expert GEMM变小、All-to-All和tail expert都可能成为瓶颈。

**追问**：区分total parameters、active parameters、每token FLOPs和每rank local expert parameters。

**前置知识与分析过程**：从dense FFN参数/FLOPs出发，引入E experts与top-k，分别计算总容量和每token激活计算，再映射EP local experts。

**主问题得分点（10分）**：dense/MoE定义 2 分；四种参数/FLOPs口径 3 分；新增数据路径 3 分；性能非自动提升 2 分。

**追问分析与参考回答**：总参数包含全部E个experts；active参数/FLOPs每token只含top-k experts及shared/dense部分；EP=e时每rank通常只持E/e个local experts（还受expert TP影响）。系统成本另加router、permute、A2A和不均衡，不能用active FLOPs单独预测tok/s。

**追问得分点（5分）**：总/active区别2分；local expert公式1分；top-k影响1分；系统额外成本1分。

### M082. Top-k router 的前向数据流是什么？

**高分回答要点**

- 对每个token hidden计算expert logits/scores，应用softmax或指定scoring，再选择top-k expert及routing probability。
- 构造routing map/counts，按目标rank/local expert排列tokens；expert输出再按probability加权并恢复原token顺序。
- router precision、noise/bias、capacity和load-balancing策略会改变数值与通信分布。

**追问**：top-2中同一token的两个expert位于不同ranks时，token会有两个实例，combine必须按原token id聚合。

**前置知识与分析过程**：给每token保留`token_id, expert_id, probability, source_rank`，按router->instance expansion->dispatch->combine跟踪。

**主问题得分点（10分）**：router计算 2 分；top-k选择/实例 2 分；routing metadata 2 分；dispatch/combine 3 分；数值配置 1 分。

**追问分析与参考回答**：token x选e1/e2后生成两个instances `(x,e1,p1)`和`(x,e2,p2)`，分别送到owner；expert输出返回source后按token_id恢复并算`p1*y1+p2*y2`。若只按接收顺序拼回或遗漏prob，会错位/改变函数。

**追问得分点（5分）**：两个实例1分；metadata1分；返回owner1分；加权聚合1分；错误反例1分。

### M083. router auxiliary loss、z-loss 和无辅助损失平衡分别解决什么？

**高分回答要点**

- auxiliary load-balancing loss鼓励token/概率在experts间均匀，但系数过大可能损害主任务或改变路由。
- z-loss约束router logits规模，改善数值稳定；它不直接保证expert负载均匀。
- aux-loss-free方法可用动态bias等机制调整选择，避免aux梯度干扰，但需要稳定更新和独立监控。

**追问**：不能只看aux loss值；至少看tokens/expert分布、prob mass、drop、top-k overlap和训练质量。

**前置知识与分析过程**：将“训练目标稳定性”和“系统负载均衡”分开。每种loss说明直接约束对象和可能副作用，再列观测。

**主问题得分点（10分）**：三类方法定义 4 分；作用对象 2 分；副作用 2 分；监控/质量 2 分。

**追问分析与参考回答**：aux值低可能因为系数/scale而非负载好；需按layer/step看token counts的CV/max-mean、prob mass、entropy、drop/padding及validation loss。z-loss主要控logit幅度，不能替代负载指标；aux-free bias也需看动态稳定性。

**追问得分点（5分）**：至少三类负载指标2分；概率指标1分；质量1分；指出z-loss边界1分。

### M084. capacity factor、token drop 和 dropless MoE 的trade-off是什么？

**高分回答要点**

- 固定capacity为每expert预留上限，便于静态buffer/GEMM，但热点token可能drop或padding浪费。
- dropless保留全部token，训练语义更完整，但variable splits、最慢expert和峰值buffer更难控制。
- capacity越大不等于更好；显存、通信、padding FLOPs和tail latency都变化。

**追问**：若drop rate均值很低但某些层/step突增，可能造成质量和性能长尾，需按layer/time监控。

**前置知识与分析过程**：写理论平均tokens/expert与capacity，再比较padding、drop和dropless variable-size三种执行。

**主问题得分点（10分）**：capacity定义 2 分；drop/padding 2 分；dropless 2 分；显存/质量/tail trade-off 3 分；监控 1 分。

**追问分析与参考回答**：全局均值会稀释热点层。应记录每层每stepcapacity usage、dropped token ids/比例、max receive和下游loss；按p99告警。性能上热点会扩大recv buffer和最慢expert，质量上特定token系统性drop可能产生偏差。

**追问得分点（5分）**：layer-step切分1分；capacity usage1分；质量后果1分；性能后果1分；p99/告警1分。

### M085. token permutation/unpermutation 为什么是实际热点？

**高分回答要点**

- router输出按token顺序，GroupedGEMM需要按expert连续排列；需要计算offset、gather/scatter和inverse mapping。
- top-k会扩张token instances，probability乘法可放在不同阶段，影响额外访存和临时buffer。
- 小expert或高EP时，permutation kernel可能比GEMM更显著，因此需要fusion和memory-efficient实现。

**追问**：正确性oracle应携带全局token id做dispatch->expert identity->combine roundtrip。

**前置知识与分析过程**：把permutation视为一一/一对多索引变换，先构造forward map再构造inverse map。性能分析再看访存、launch和buffer。

**主问题得分点（10分）**：为何需要permute 2 分；offset/inverse 2 分；top-k/prob位置 2 分；性能热点 2 分；oracle 2 分。

**追问分析与参考回答**：每instance携带唯一`(global_token_id, topk_slot)`，expert函数可暂设为加expert id；dispatch/combine后应恢复原token顺序并得到解析和。这样可发现split、排序、inverse或prob乘法错误，远强于只检查shape/finite。

**追问得分点（5分）**：唯一id1分；可解析expert函数1分；top-k slot1分；逐元素oracle1分；能定位错误阶段1分。

### M086. All-to-All token dispatcher 的完整数据流是什么？

**高分回答要点**

- 先统计每个目标EP rank的send counts，交换counts得到recv splits，再按rank/expert排列hidden和probs。
- `all_to_all_single` variable splits把token instances送到expert owner，接收后再按local expert排序执行GEMM。
- combine时执行逆permutation和反向All-to-All，把结果返回原owner并按top-k probabilities累加。

**追问**：为什么combine splits与dispatch方向相反？当前rank收到多少就需按来源发回多少。

**前置知识与分析过程**：构造send-count矩阵`C[src,dst]`，dispatch时src按行发送，dst按列接收；combine对同一矩阵转置方向返回。

**主问题得分点（10分）**：count交换 2 分；variable split 2 分；local expert排序 2 分；inverse A2A/permute 3 分；metadata 1 分。

**追问分析与参考回答**：dispatch rank i的send splits是`C[i,*]`，recv splits是`C[*,i]`；expert完成后rank i要把收到的每个source段发回，所以combine send splits等于dispatch recv splits，combine recv splits等于dispatch send splits。顺序还需与source segment metadata一致。

**追问得分点（5分）**：矩阵定义1分；dispatch行/列1分；combine交换2分；顺序metadata1分。

### M087. AllGather dispatcher 与 All-to-All dispatcher 如何选择？

**高分回答要点**

- AllGather让各EP rank看到更广token集合并本地选所需部分，逻辑简单但复制/带宽随EP放大。
- All-to-All只把token送到owner，对大EP更省冗余，但variable split、permute和跨节点all-to-all复杂。
- 选择取决于EP、top-k、token数、expert粒度、拓扑和实现kernel；要以端到端profile而非名称决定。

**追问**：官方文档把alltoall作为标准EP>1路径，allgather可用于TP-only、小EP或较大top-k等场景。

**前置知识与分析过程**：分别计算每rank发送的有效token bytes和复制冗余，加入permute/variable split复杂度，再按EP/top-k/拓扑选择。

**主问题得分点（10分）**：两dispatcher数据流 4 分；通信/内存复杂度 2 分；选择条件 2 分；需实测/版本边界 2 分。

**追问分析与参考回答**：AllGather把所有rank token复制到每rank，再筛本地experts，EP大时bytes/peak高但免复杂A2A路由；A2A只发送目标instances，更适合大EP，却需要counts、permute和variable splits。用token规模、EP、top-k和链路做break-even benchmark。

**追问得分点（5分）**：复制与定向发送2分；复杂度1分；适用场景1分；break-even实验1分。

### M088. Flex/DeepEP/HybridEP 等新 dispatcher 优化的是什么？

**高分回答要点**

- 目标是减少跨节点冗余、融合节点内/间dispatch、降低占用SM并利用特定硬件通信能力。
- DeepEP面向Hopper/B200等跨节点fine-grained MoE；HybridEP面向更新NVLink/MNNVL平台，依赖特定TMA/IBGDA/symmetric memory能力。
- 这是当前官方新版能力，本机V100未实测，不能声称相同backend可用或有同等收益。

**追问**：面试时先说硬件和版本前提，再比较NCCL A2A baseline；不要只报热门backend名。

**前置知识与分析过程**：先问GPU架构、NVLink/MNNVL、跨节点网络和MCore/backend版本，再列新dispatcher减少的冗余/SM占用。

**主问题得分点（10分）**：优化目标 3 分；DeepEP/HybridEP定位 2 分；硬件依赖 2 分；baseline/证据边界 3 分。

**追问分析与参考回答**：回答应先说V100不满足这些新backend要求；在H100/B200/GB200目标平台，用同router/token counts对比NCCL alltoall与Flex backend，测A2A exposed、SM占用、tok/s、peak和correctness。若硬件/软件gate不满足就fallback而非强开。

**追问得分点（5分）**：硬件gate1分；V100边界1分；公平baseline1分；五类指标1分；fallback1分。

### M089. EP group、expert-DP group 和expert-TP group如何理解？

**高分回答要点**

- EP group把不同experts放到ranks；同一expert shard在其他复制域上的对应ranks构成expert-DP，用于其gradient同步。
- expert-TP可进一步切单个expert MLP；它可与dense TP不同，fine-grained expert常倾向较小expert-TP保护GEMM shape。
- dense参数和expert参数可能使用不同DP group/optimizer bucket，不能用一个global DP group处理全部参数。

**追问**：给定rank mesh要求分别画dense attention参数和expert参数的同步伙伴。

**前置知识与分析过程**：先为每类参数写owner坐标：dense参数忽略EP expert id，expert参数包含expert id/expert-TP shard。固定所有model坐标只变化复制维得到各自DP组。

**主问题得分点（10分）**：EP/EDP/ETP定义 4 分；dense/expert组差异 3 分；optimizer bucket影响 2 分；避免机械公式 1 分。

**追问分析与参考回答**：dense attention在所有model replicas复制，按普通DP/CP合并域同步；某expert只在持有同一expert id和同一expert-TP shard的replicas间同步，形成expert-DP。先标具体mesh坐标再连线；不能让不同experts互相AllReduce。

**追问得分点（5分）**：dense owner1分；expert id/shard2分；两类group1分；错误归约后果1分。

### M090. TP+EP为什么容易制造复杂的token ownership？

**高分回答要点**

- dense TP ranks可能复制sequence tokens但分片hidden/heads，router通常需要明确每个token实例由谁dispatch。
- SP可按sequence提供token ownership；expert-TP又要求同一expert输入按hidden/layout分发。
- 错误组合会重复dispatch、重复计算或漏grad，必须沿tensor layout逐边验证。

**追问**：官方parallel guide明确提示TP与EP组合时启用SP；高分答案要解释机制而非只背约束。

**前置知识与分析过程**：沿dense TP边界确定token是否复制，再沿router看调用者数量；之后区分token并行和expert hidden TP。

**主问题得分点（10分）**：重复token问题 3 分；SP ownership 2 分；expert-TP 2 分；组合验证 3 分。

**追问分析与参考回答**：若同一logical token在多个dense TP ranks复制且都调用router，会形成多份routing instances和重复A2A。SP在进入MoE前按sequence分配唯一token owner；若expert本身用TP，则owner token可再按expert计算layout分发，这与重复路由不同。

**追问得分点（5分）**：重复路由2分；SP owner1分；expert-TP区别1分；画完整路径1分。

### M091. GroupedGEMM 为什么适合 MoE？

**高分回答要点**

- 每个local expert收到不同token数，形成多个不同M、共享相似K/N的小GEMM；逐expert launch开销和低利用率高。
- GroupedGEMM在一次/少量kernel中调度多个expert matmul，提高launch与SM利用率，并支持不同token counts。
- 收益依赖token分布、expert数、hidden/FFN shape、dtype和硬件；极端热点仍会被单大expert拖尾。

**追问**：应报告每expert M分布、GroupedGEMM kernel时间、padding/有效FLOPs和最慢rank，而非只报总tok/s。

**前置知识与分析过程**：把local experts看成一组`M_i x K x N` GEMMs，先分析M_i分布，再比较逐expert launch与grouped调度。

**主问题得分点（10分）**：GroupedGEMM问题定义 3 分；收益机制 2 分；负载/shape条件 2 分；完整指标 3 分。

**追问分析与参考回答**：收集每step每expert token count即M_i，报告分位数/CV；profile grouped kernel执行时间与实际有效FLOPs，若实现padding则算padding ratio。总tok/s相同也要看最慢EP rank和空/小expert，因为它们决定扩展与浪费。

**追问得分点（5分）**：M_i分布1分；kernel1分；有效/padding FLOPs1分；最慢rank1分；端到端1分。

### M092. shared expert 的作用与系统代价是什么？

**高分回答要点**

- shared expert对所有tokens执行，提供稳定dense capacity，与routed experts输出组合。
- 它增加固定FLOPs和参数，也创造可与EP dispatch/A2A重叠的计算窗口。
- overlap需要独立stream和正确依赖；shared expert过大也会成为新的critical path。

**追问**：如何证明重叠？对shared compute与dispatch/combine做interval union/intersection，并比较exposed A2A和step wall。

**前置知识与分析过程**：画routed与shared两条依赖支路，找到合并点。只有合并前独立部分可并发，之后必须wait。

**主问题得分点（10分）**：shared expert语义 2 分；额外成本 2 分；可重叠窗口 3 分；trace/峰值 3 分。

**追问分析与参考回答**：用不同CUDA streams发A2A和shared MLP，event保证输入ready并在输出combine前汇合。trace对A2A interval与shared GEMM求intersection，exposed A2A下降且step wall下降才算收益；同时检查并发争SM和双分支activation peak。

**追问得分点（5分）**：依赖图1分；stream/event1分；intersection/exposed1分；step wall1分；资源争用/peak1分。

### M093. MoE load imbalance 应观测哪些指标？

**高分回答要点**

- 每layer/step的tokens per expert：max/mean、CV、p95/p99、空expert、top-k pair和跨rank send/recv counts。
- router概率/entropy、aux/z loss、drop/padding、expert GEMM时间、dispatch/combine和rank idle。
- 聚合均值会掩盖动态热点；要按layer、DP replica、sequence类型和时间窗口切分。

**追问**：本机toy的hot expert receive max/mean=6，吞吐下降66.4%，但它没有真实router/quality指标。

**前置知识与分析过程**：将路由质量、计算负载、通信分布和训练质量分四层指标。按layer/step/rank保留分布而非全局平均。

**主问题得分点（10分）**：负载统计 3 分；router/通信/compute关联 3 分；时间维度 2 分；本机数字与边界 2 分。

**追问分析与参考回答**：toy手工构造send splits，证实max/mean=6时最慢expert及combine peer wait使tok/s降66.4%；但没有router logits、aux loss、drop或收敛，不能推生产质量。真实实验需把这些指标与token counts和A2A时间关联。

**追问得分点（5分）**：两个准确数字1分；机制2分；缺失指标1分；不夸大1分。

### M094. 为什么非热点rank的collective kernel也可能很长？

**高分回答要点**

- collective duration包含等待peer到达；热点rank先做更大expert GEMM，其他rank在combine A2A内等待。
- profiler可能把straggler等待归到NCCL kernel，看起来像“网络慢”，实际根因是上游compute imbalance。
- 需对齐rank timeline，把router counts、expert GEMM结束和collective进入时间关联。

**追问**：优化NCCL算法无法修复上游expert负载，必须先区分transport与peer wait。

**前置知识与分析过程**：对齐各rank expert compute结束和collective enter/exit，collective kernel总时长拆为arrival skew与传输/同步。

**主问题得分点（10分）**：peer wait机制 3 分；多rank timeline 3 分；transport指标 2 分；正确优化方向 2 分。

**追问分析与参考回答**：若非热点rank早进入combine而热点rank晚到，前者NCCL kernel长主要是在等待；查看enter-time spread、热点GEMM/token counts。只有所有rank近同时进入而通信仍慢，才进一步查algbw/NIC/algorithm。前一种应优化routing/capacity/expert balance。

**追问得分点（5分）**：enter skew2分；token/GEMM关联1分；网络判据1分；优化归因1分。

### M095. 如何 overlap EP All-to-All 与计算？

**高分回答要点**

- 可把tokens/chunks分批，在一批dispatch/combine时执行另一批expert/shared compute，或做batch-level overlap。
- 需要足够chunk计算隐藏通信，且不能让更多in-flight token buffer造成OOM。
- stream/event、permutation依赖、wgrad延后和PP schedule需共同设计，错误顺序会静默损坏。

**追问**：评价exposed A2A、expert SM utilization、buffer peak、tokens/s和数值一致性，不只看NCCL累计时长。

**前置知识与分析过程**：先按token chunks构造pipeline依赖，再计算每chunk通信与compute是否足够隐藏。评估更多in-flight chunks的内存。

**主问题得分点（10分）**：chunk/batch overlap 3 分；依赖/stream 2 分；计算通信比 2 分；五类指标 3 分。

**追问分析与参考回答**：做no-overlap与不同chunk数对照，固定routing；trace计算A2A union和与expert/shared compute交集，报告exposed。检查SM争用、permute依赖和buffer peak；用token-id oracle与baseline loss/grad确保延后wgrad等调度未改变数学结果。

**追问得分点（5分）**：消融1分；interval指标1分；SM/peak1分；dependency1分；correctness1分。

### M096. 本机MoE实验能证明什么，不能证明什么？

**高分回答要点**

- 证明variable-split All-to-All、token-id roundtrip、简单expert GEMM、hotspot和peer wait的数据路径机制。
- 不能证明MCore真实router、aux loss、capacity/drop、GroupedGEMM、shared expert、DeepEP或训练收敛。
- 面试可称“MoE通信/不均衡原理实验”，不可称“优化了生产Megatron MoE”。

**追问**：扩展顺序应为真实router correctness -> dispatcher对照 -> GroupedGEMM -> overlap -> quality/长时间负载。

**前置知识与分析过程**：用证据成熟度分级：数据路径、完整MCore模块、性能优化、训练质量。每层前一层correctness是后一层前提。

**主问题得分点（10分）**：已证明内容 3 分；六类未证明内容 3 分；准确表述 2 分；扩展路线 2 分。

**追问分析与参考回答**：先将toy替换为MCore router并用token-id/prob oracle；再比较allgather/A2A；启用GroupedGEMM和fusion；最后做overlap与多小时真实数据训练，观察aux/drop/quality。任何阶段性能结论都保留前一阶段reference和feature-off回退。

**追问得分点（5分）**：顺序完整2分；每步oracle1分；性能消融1分；质量/soak1分。

---

## 9. Level 7：性能、融合、精度与通信重叠（M097-M110）

### M097. Megatron训练为什么常用MFU，而不仅是tokens/s？

**高分回答要点**

- tokens/s受模型大小和sequence影响，不能跨配置直接比较硬件利用；MFU用有效模型FLOPs/硬件峰值归一化。
- FLOPs模型要说明attention、embedding、MoE active experts、recompute是否计入，硬件峰值要匹配dtype/Tensor Core。
- MFU仍不等于time-to-quality，也不揭示通信/数据/能效，需与step、loss和profile结合。

**追问**：recompute FLOPs计入分子会提高“硬件利用”却不提高有效训练进度，因此需同时报useful MFU与executed FLOPs口径。

**前置知识与分析过程**：先定义分子有效/执行FLOPs和分母硬件理论峰值，再判断配置变化是否增加重复计算。最后关联time-to-quality。

**主问题得分点（10分）**：MFU定义 3 分；FLOPs/峰值口径 2 分；recompute/MoE边界 3 分；与训练进度区别 2 分。

**追问分析与参考回答**：若full recompute多执行约一段forward，把这些FLOPs计入executed MFU可能显示GPU更忙；但每step有效tokens和模型更新没增加。应同时报useful model FLOPs/step time、executed FLOPs利用率、tok/s及收敛，使优化不会奖励无效重算。

**追问得分点（5分）**：两种分子2分；反直觉结论1分；多指标1分；time-to-quality1分。

### M098. 如何估算 dense Transformer 的训练 FLOPs？

**高分回答要点**

- 从QKV/output、MLP两层和attention score/value matmul逐项按`2MNK`计算forward。
- 训练forward+backward常近似forward的3倍，但embedding、softmax、norm、checkpoint重算和tied权重需明确。
- 常见`~6N tokens`只是大模型/特定假设下近似，长sequence attention项和MoE会偏离。

**追问**：面试官给L/H/S/B时应现场列主项，而不是只背6N。

**前置知识与分析过程**：对每个linear写`2MNK`，attention写QK与PV，按layers/batch累加；backward再按矩阵梯度成本扩展。

**主问题得分点（10分）**：linear主项 3 分；attention二次项 2 分；训练倍数 2 分；近似边界 3 分。

**追问分析与参考回答**：每层列QKV约`2BSH*3H`、output`2BSH*H`、MLP两/三路，以及QK/PV约各`2B*heads*S*S*head_dim`；forward求和，backward通常每matmul有dX/dW两项。最后说明softmax/norm/fusion、MoE和recompute是否计入。

**追问得分点（5分）**：至少四个linear项2分；两attention项1分；backward1分；边界1分。

### M099. local layer spec 与 Transformer Engine spec 如何选择？

**高分回答要点**

- local spec更便于源码理解、兼容/调试；TE spec提供融合layer、optimized attention、FP8等NVIDIA路径。
- 同shape不保证相同kernel、numerics、checkpoint key或硬件支持；V100上很多新TE能力不可用。
- 应做同权重forward/backward correctness、峰值、kernel trace和端到端吞吐对照。

**追问**：本机使用`get_gpt_layer_local_spec()`，不能据此声称测过TE/FP8优化。

**前置知识与分析过程**：先比较spec选择的module实现和硬件依赖，再设计同权重A/B，而不是将“模型结构相同”等同“runtime相同”。

**主问题得分点（10分）**：两spec定位 3 分；kernel/numeric/checkpoint差异 3 分；对照方法 2 分；本机边界 2 分。

**追问分析与参考回答**：脚本显式调用local spec且V100无FP8，所以数据只覆盖PyTorch/local MCore路径。要评估TE需目标硬件安装兼容TE，加载同logical权重，固定GBS比较logits/grad、kernel、peak、MFU和checkpoint key；失败时保留local fallback。

**追问得分点（5分）**：代码证据1分；硬件边界1分；A/B correctness1分；性能指标1分；fallback1分。

### M100. Megatron常见fusion分别消除什么开销？

**高分回答要点**

- bias+activation、bias+dropout+residual、fused softmax、cross entropy、LayerNorm/RMSNorm、rotary等减少launch与中间HBM访存。
- gradient accumulation fusion把weight-gradient accumulation融入GEMM epilogue，减少独立读写/加法kernel。
- fusion收益随shape和硬件变化，必须有fallback/correctness；小shape更容易被launch主导。

**追问**：用roofline解释为什么memory-bound逐元素链更适合fusion，而大GEMM本身主要看Tensor Core效率。

**前置知识与分析过程**：计算算术强度FLOPs/bytes，把多kernel中间读写和launch列出来，再判断fusion是否提高强度或仅减少launch。

**主问题得分点（10分）**：至少五类fusion 3 分；访存/launch机制 3 分；gradient fusion 2 分；正确性/fallback 2 分。

**追问分析与参考回答**：bias、dropout、residual分别成kernel会多次读写同一activation，FLOPs少且受HBM/launch限制，融合成一次遍历收益大；大GEMM已有高算术强度，优化重点是tile、Tensor Core和shape，epilogue fusion只减少边缘访存。用dram bytes/kernel数验证。

**追问得分点（5分）**：强度定义1分；逐元素访存2分；GEMM区别1分；counter验证1分。

### M101. TP communication overlap有哪些常见手段？

**高分回答要点**

- 把AG/RS切分并与GEMM不同chunks流水，或将bulk/dgrad/wgrad阶段与对应collective交叠。
- 需要GEMM足够大、独立stream/event和合适通信chunk；通信kernel占SM也可能与compute争抢。
- sequence parallel layout常提供RS/AG边界，但“可异步”不等于实际critical-path overlap。

**追问**：保存timeline，计算NCCL union、compute intersection和exposed通信，再看step wall与显存峰值。

**前置知识与分析过程**：确定可切chunk和producer/consumer依赖，通信与GEMM放不同stream；再从trace判断真实而非理论overlap。

**主问题得分点（10分）**：overlap手段 3 分；窗口/依赖 2 分；资源争用 2 分；测量 3 分。

**追问分析与参考回答**：对每TP collective记录interval，合并重叠NCCL区间得union，与目标GEMM union求交；`exposed=nccl_union-intersection`。还要看chunk增多是否增加launch/peak，以及max-rank step是否下降；局部交集增加但step不降不算成功。

**追问得分点（5分）**：union1分；intersection1分；exposed1分；step wall1分；peak/争用1分。

### M102. 为什么同时开启多种overlap flag可能变慢？

**高分回答要点**

- TP、DP grad、param gather、EP A2A和PP P2P可能争用同一NIC/PCIe/NVLink、copy engine、SM和stream priority。
- 更多in-flight buffer提高显存峰值/fragmentation；更细chunk增加launch与同步。
- 局部重叠可能把压力移到另一个phase或最慢rank，端到端critical path反而增长。

**追问**：应做单flag、两两组合、全开消融矩阵，保存链路counter和per-group trace。

**前置知识与分析过程**：建立每种通信占用的链路/SM/stream资源表，再用factorial消融识别交互项，避免一次全开无法归因。

**主问题得分点（10分）**：五类通信争用 3 分；内存/launch代价 2 分；消融矩阵 3 分；critical path判断 2 分。

**追问分析与参考回答**：先baseline，再逐个开TP、grad RS、param AG、EP A2A overlap；对收益项做两两和全开。记录每group exposed、NIC/NVLink/PCIe吞吐、SM active、in-flight peak和step。全开回退时根据交互项定位共享链路或SM竞争。

**追问得分点（5分）**：逐flag1分；两两交互1分；资源指标1分；peak1分；归因方案1分。

### M103. CUDA Graph 在Megatron训练中为什么困难？

**高分回答要点**

- graph要求稳定control flow、shape、地址和stream依赖；pipeline microbatch、MoE variable token counts、动态loss/数据都增加变化。
- NCCL capture、RNG、optimizer、FP8 metadata和参数AllGather必须使用兼容路径。
- 收益主要是降低CPU launch gap；若已被大GEMM/NCCL主导，端到端收益有限。

**追问**：先用Nsight确认CPU gap，再逐scope capture并验证replay数值和buffer地址，不要因“支持graph”就全迭代捕获。

**前置知识与分析过程**：先判断性能是否CPU launch-bound，再检查shape/control/address/stream稳定性。按module或microbatch逐步扩大capture scope。

**主问题得分点（10分）**：graph约束 4 分；Megatron动态难点 3 分；收益判断 1 分；渐进验证 2 分。

**追问分析与参考回答**：Nsight若显示GPU kernel间CPU空洞明显才值得；从固定shape attention/MLP scope开始，固定allocator地址和RNG/NCCL capture，重复replay比较loss/grad。MoE variable tokens和动态PP先保留graph外，逐步扩大并监控graph cache/内存。

**追问得分点（5分）**：CPU gap1分；最小scope1分；地址/RNG/NCCL1分；数值1分；动态fallback1分。

### M104. FP8训练需要解决哪些问题？

**高分回答要点**

- 需要选择FP8格式、scale/amax历史、何处quantize/dequantize，并保留更高精度master/敏感算子。
- Tensor Core吞吐和通信/显存可受益，但overflow/underflow、scale同步和收敛需验证。
- Transformer Engine、硬件架构和MCore版本决定支持范围；V100没有FP8 Tensor Core路径。

**追问**：专家不会把“权重FP8”与“optimizer state FP8”混为一谈，也不会只跑一步loss finite作为精度结论。

**前置知识与分析过程**：为forward weight/activation、gradient、master param和optimizer states分别标dtype；再画scale/amax生成与同步。

**主问题得分点（10分）**：FP8格式/scale 3 分；敏感状态高精度 2 分；硬件/TE依赖 2 分；数值验证 3 分。

**追问分析与参考回答**：权重/activation进入FP8 GEMM不意味着FP32 master和Adam m/v也FP8；scale/amax决定量化范围并可能跨group同步。验证需多层activation/grad误差、overflow、数百步loss与最终quality，对照BF16；V100无对应Tensor Core路径。

**追问得分点（5分）**：dtype分层2分；scale/amax1分；长期验证1分；V100边界1分。

### M105. 如何从GEMM shape判断一个并行配置可能低效？

**高分回答要点**

- 列每rank的M=`seq_local*microbatch`、N/K由hidden/FFN/TP切分，检查Tensor Core对齐、tile数和wave量。
- TP/EP过大使N/K或expert token M过小；PP过大使每rankkernel数量少、launch gap占比高。
- 用cublas/TE kernel name、achieved FLOPs、Tensor Core指令和duration验证，而非只看GPU utilization。

**追问**：同tok/s下每GPU MFU下降可能意味着加GPU后工作切得过碎。

**前置知识与分析过程**：从global shape和并行轴计算每rankM/N/K，再检查对齐、tiles/waves和launch数量；将kernel效率与通信分开。

**主问题得分点（10分）**：M/N/K推导 3 分；TP/EP/PP碎片化 3 分；硬件指标 2 分；配置决策 2 分。

**追问分析与参考回答**：aggregate tok/s可因更多GPU维持，但每GPU处理更小GEMM，achieved FLOPs/MFU下降，成本效率变差。应报per-GPU tok/s、GEMM shape、waves、Tensor Core指令和NCCL；尝试减TP/EP或增MBS，在容量允许下恢复shape。

**追问得分点（5分）**：aggregate/per-GPU区别1分；shape2分；指标1分；调整方向1分。

### M106. 如何做topology-aware的并行映射？

**高分回答要点**

- 按通信频率、bytes和exposed fraction给TP/CP/EP/PP/DP edge加权，再映射GPU/NIC/NUMA/故障域。
- 高频TP/CP优先高速域，EP需考虑all-to-all rail，PP相邻stage需稳定低延迟，DP可跨节点但梯度流量大。
- 同时考虑共享链路contention和故障域，不能只最小化pair距离。

**追问**：本机SYS TP phase慢3.05x，但三类各一次并发两mapping接近，说明目标函数必须包含真实调用频率。

**前置知识与分析过程**：构造通信图，每edge权重=`bytes*frequency*exposed sensitivity`，再加入共享链路和故障域约束。

**主问题得分点（10分）**：各并行placement原则 3 分；加权成本 3 分；contention/故障域 2 分；本机反例 2 分。

**追问分析与参考回答**：两mapping并发实验都恰含PIX/PHB/SYS各一流，所以物理流量multiset相同；真实TP每层多次而DP通常每step bucket，权重不同。planner需用schedule频率与critical path而非一次对称microbench选择。

**追问得分点（5分）**：解释multiset2分；调用频率1分；critical path1分；planner成本1分。

### M107. Megatron显存分析为什么必须按生命周期拆分？

**高分回答要点**

- 常驻：本地param shard、main param、optimizer；周期：grad buffer、AG full param；microbatch：activation/recompute；临时：workspace/communication。
- PP/overlap/recompute改变并存时间，allocator reserved/fragmentation又与active tensor bytes不同。
- 记录每rankmax allocated/reserved、memory snapshot、OOM时allocation request和parallel coordinate。

**追问**：只用参数公式能判断“理论放得下”，不能保证runtime不在prefetch+activation重叠处OOM。

**前置知识与分析过程**：将状态分常驻、iteration、microbatch和瞬态，画live intervals并找并集峰值；再区分allocated、reserved和外部库内存。

**主问题得分点（10分）**：四类状态 3 分；lifetime峰值 3 分；allocator口径 2 分；OOM诊断 2 分。

**追问分析与参考回答**：参数公式漏掉AG full params、PP warmup activations、communication/workspace和checkpoint staging的同时存活。用rank-local memory snapshot定位OOM时刻，记录requested/allocated/reserved及module/stream；关闭prefetch/recompute/overlap做因果对照。

**追问得分点（5分）**：瞬态并存2分；snapshot1分；三内存口径1分；消融1分。

### M108. 如何使用straggler信息定位慢rank？

**高分回答要点**

- 先按DP/TP/PP/CP/EP坐标聚合step、F/B、collective和data time，看异常是否沿某process group传播。
- 关联GPU clocks/ECC/thermal、CPU/NUMA、NIC counters、data shard和MoE token counts。
- collective中所有rank变长不代表所有rank故障，最晚进入者可能是根因。

**追问**：PP某stage compute慢会让上下游在P2P等待；EP热点会让其他rank在combine等待。

**前置知识与分析过程**：先按并行坐标聚合异常，再对共同group找最晚enter rank；不要把等待者误判为根因。

**主问题得分点（10分）**：坐标聚合 3 分；系统指标 2 分；peer-wait归因 3 分；故障域 2 分。

**追问分析与参考回答**：PP看每stage compute结束与send/recv enter；最慢stage之前/之后会形成等待链。EP看token counts和expert GEMM end；早进combine的rank NCCL长是等待热点。关联clock/NIC/data后才能区分软件负载与坏卡。

**追问得分点（5分）**：PP时间线1分；EP时间线1分；最晚enter1分；系统交叉证据1分；不误归NCCL1分。

### M109. 一个可信的Megatron性能benchmark需要什么？

**高分回答要点**

- 固定commit/container/driver/TE/NCCL、模型/config、global batch、sequence、数据shape、拓扑和环境变量。
- 分离初始化/JIT/autotune/warmup，取max-rank wall，做多次独立启动并保存loss/correctness、方差和原始log/trace。
- 同时报tok/s、samples/s、MFU、每GPU效率、peak、collective breakdown和数据/checkpoint pause。

**追问**：性能提升若来自改变global batch或跳过optimizer/checkpoint，必须单独标注，不能称公平训练优化。

**前置知识与分析过程**：先定义被测边界是纯F/B还是完整iteration/训练job，列固定变量和计入项。再制定warmup、重复与correctness。

**主问题得分点（10分）**：环境/配置固定 3 分；计时与max-rank 2 分；指标 2 分；correctness/artifact 3 分。

**追问分析与参考回答**：对优化A/B保存完整config diff，确保GBS、sequence、optimizer step、precision和数据相同；若单独测F/B就明确排除optimizer/checkpoint，另报端到端job。每次独立启动、warmup后同步计时，保存raw JSON/trace和loss。

**追问得分点（5分）**：边界声明1分；固定变量2分；重复/同步1分；artifact/correctness1分。

### M110. V100实验对今天Megatron专家能力有什么价值与局限？

**高分回答要点**

- 可真实学习TP/SP/PP数据流、NCCL拓扑、recompute、benchmark和失败定位，慢PCIe还放大通信代价。
- 不能验证BF16/FP8、FlashAttention新路径、NVSwitch/IB、多机EP/CP或最新版TE/MCore绝对性能。
- 专家价值在可迁移的tensor/通信/状态分析方法，并能明确新硬件需要重新校准哪些数字。

**追问**：回答“原理一样所以数字也适用”或“老卡完全无价值”都不合格。

**前置知识与分析过程**：把算法不变量、软件API和硬件性能量三层分开；逐层说明可迁移与需重测内容。

**主问题得分点（10分）**：可学机制 3 分；硬件限制 3 分；可迁移方法 2 分；诚实边界 2 分。

**追问分析与参考回答**：Column/Row数学、1F1B依赖、group顺序和benchmark方法可迁移；NVLink/NIC、BF16/FP8、attention/TE kernel和最佳TP degree必须在新平台重测。V100通信慢使瓶颈清晰，但绝对tok/s不能代表H100/B200。

**追问得分点（5分）**：稳定机制2分；需重测量2分；不走极端1分。

---

## 10. Level 8：Checkpoint、数据、可靠性与可观测性（M111-M124）

### M111. 一个可继续训练的 Megatron checkpoint 必须保存什么？

**高分回答要点**

- model shards、optimizer states、master params、scheduler、loss scaler、iteration/consumed samples/tokens。
- CPU/CUDA/model-parallel RNG、data iterator/sampler状态和并行/config metadata。
- manifest/checksum/version/完成标记，保证partial checkpoint不可见并支持兼容性判断。

**追问**：只恢复model/optimizer可语义继续但不保证bitwise replay；runtime/reducer/kernel归约顺序也可能变化。

**前置知识与分析过程**：按数学状态、优化状态、随机/数据状态和提交元数据四类列清单，再定义恢复目标是加载、step replay还是收敛等价。

**主问题得分点（10分）**：四类状态 5 分；原子/校验 2 分；恢复级别 2 分；并行metadata 1 分。

**追问分析与参考回答**：model/optimizer exact只保证保存点参数与动量；若RNG/data cursor不同，下一batch/dropout变；即便全恢复，重建bucket或换TP使浮点归约顺序不同。应分别验证保存点hash、下一步loss/grad及长期quality，并声明目标。

**追问得分点（5分）**：RNG/data1分；runtime顺序1分；三层验证2分；目标声明1分。

### M112. Megatron distributed checkpoint 的 sharded state dict 表达什么？

**高分回答要点**

- 每个local tensor shard附带global shape、global offset、axis/replica id和dtype等元数据。
- 保存层将logical tensor与当前TP/PP/DP布局解耦，load策略据metadata构造目标shard。
- optimizer state需与参数shard映射一致，不能靠rank文件名猜ownership。

**追问**：要求你为一个TP2 ColumnParallel权重写两个shard的global offset，并说明TP4恢复如何切分。

**前置知识与分析过程**：先声明global tensor shape和partition axis，local shard用global offset/shape描述；load时用目标layout求区间交集。

**主问题得分点（10分）**：metadata字段 3 分；逻辑/物理解耦 2 分；optimizer映射 2 分；具体offset推导 3 分。

**追问分析与参考回答**：若W `[h,4h]`按axis1切TP2，shard0 offset`[0,0]` shape`[h,2h]`，shard1 offset`[0,2h]`同shape；TP4目标区间每份宽h，loader从相应源shard读取前/后半。padding时还需logical vs padded shape。

**追问得分点（5分）**：global shape1分；TP2 offsets2分；TP4切法1分；padding边界1分。

### M113. 为什么 distributed checkpoint 能跨 TP/PP/DP 配置 reshard？

**高分回答要点**

- checkpoint记录logical global tensor，而不是把“rank0文件”永久绑定给rank0。
- load时根据新parallel layout读取、切分/拼接并路由目标shards；PP变化还会改变layer owner。
- 可恢复性受tensor metadata、模型结构和optimizer格式限制，必须先做兼容性plan与peak/IO估算。

**追问**：8 rank保存、4 rank恢复时应验证权重checksum、optimizer states、第一步loss/grad和读取放大。

**前置知识与分析过程**：建立源global shard map与目标owner map，先验证可覆盖性，再规划read/redistribute；PP变化还映射layer owner。

**主问题得分点（10分）**：reshard机制 3 分；PP/TP/DP变化 3 分；资源/兼容风险 2 分；验证 2 分。

**追问分析与参考回答**：保存后从metadata重建logical tensor checksum；4rank目标按新layout加载重叠ranges，optimizer按参数key/状态类型重分。记录每rankread bytes、network shuffle和peak；加载后比较logical model/optimizer，固定batch做第一步loss/grad/更新。

**追问得分点（5分）**：range映射1分；optimizer1分；IO/peak1分；四类correctness2分。

### M114. distributed optimizer checkpoint 为什么比模型权重更难兼容？

**高分回答要点**

- 一个参数对应FP32 main param、m/v等多个shards，并受DP ownership、flatten/bucket layout和parameter key影响。
- MCore版本升级可能改变flatten metadata或optimizer格式；模型权重可兼容不代表optimizer可直接加载。
- 当前官方文档区分偏速度的`dp_reshardable`和更灵活但较慢的`fully_reshardable`等路径，应按版本核对。

**追问**：不能恢复optimizer时可加载model并重建optimizer，但训练轨迹/收敛会变化，必须业务确认。

**前置知识与分析过程**：对每参数列master/m/v及flatten range和owner，比较源/目标版本metadata是否足以重建。区分“不能加载”和“可丢弃继续”。

**主问题得分点（10分）**：复杂性来源 4 分；格式/版本 2 分；性能/灵活性trade-off 2 分；fallback语义 2 分。

**追问分析与参考回答**：重建optimizer会丢moment和可能的master状态，相当于改变优化器历史；即使model loss连续，更新方向/学习率阶段可能不匹配。只能在明确接受训练轨迹变化时使用，并考虑warm restart、LR和质量监控。

**追问得分点（5分）**：丢失状态2分；轨迹后果1分；业务确认1分；缓解/监控1分。

### M115. async checkpoint 怎样降低 pause，又引入哪些一致性问题？

**高分回答要点**

- training thread先冻结/复制一致state snapshot，再由后台线程/进程执行序列化和IO，使GPU继续后续step。
- 若后台读取仍在被optimizer修改的storage，会写出混合step；需要staging buffer、lifetime和完成协议。
- 控制in-flight checkpoints、host内存、带宽竞争、失败传播和job退出flush；manifest只能在全部shards完成后发布。

**追问**：当前官方异步策略正在向NVRx演进；版本迁移与依赖属于运维设计，不是恒定API事实。

**前置知识与分析过程**：画训练storage与staging snapshot的所有权，标copy完成、后台IO、commit和下次checkpoint。再分析失败/退出。

**主问题得分点（10分）**：snapshot一致性 3 分；后台生命周期 2 分；资源/backpressure 2 分；版本迁移 3 分。

**追问分析与参考回答**：目标commit应锁定async backend和依赖，启动时验证能力；升级做旧/新checkpoint load和pause对照。job退出前flush并传播后台错误，不能因训练继续就忽略失败；文档中的未来默认值不能替代当前部署测试。

**追问得分点（5分）**：锁版本1分；一致snapshot1分；flush/error1分；兼容测试1分；不把路线图当现状1分。

### M116. 如何设计原子 checkpoint commit 协议？

**高分回答要点**

- 每rank写临时/唯一版本shard并fsync/上传，计算size/hash；协调者确认期望shards齐全。
- 最后原子发布manifest/COMMITTED标记，loader只枚举已commit且checksum通过的版本。
- 处理重试、重复rank、迟到writer、对象存储无rename、GC与reader并发。

**追问**：本机failure实验已证明partial版本不应可见，但没有覆盖MCore DCP/对象存储/异步reshard。

**前置知识与分析过程**：先定义writer、coordinator、reader和GC状态机，按本地文件系统与对象存储不同原语设计commit。

**主问题得分点（10分）**：临时写/校验 3 分；manifest最后发布 2 分；重试/GC竞态 3 分；证据边界 2 分。

**追问分析与参考回答**：本机协议验证缺rank shard时manifest不发布、reader忽略partial；扩展到DCP需真实sharded metadata和optimizer，扩展对象存储需用不可变version key+原子/一致manifest而非rename假设，再做异步writer崩溃和reshard reader测试。

**追问得分点（5分）**：本机已证1分；三项未证2分；对象存储方案1分；故障测试1分。

### M117. 如何追求跨并行配置的可复现性？

**高分回答要点**

- 恢复model/optimizer/RNG/data position，固定kernel/determinism、microbatch/global batch和loss scaling。
- TP/PP/DP改变会改变浮点归约树、kernel shape和操作顺序，bitwise一致通常不现实，应定义数值/训练等价标准。
- 用初始logits/loss、gradient norm/checksum、若干step loss分布与最终quality分层验证。

**追问**：专家会区分state exact、step replay exact和convergence equivalent。

**前置知识与分析过程**：为三个级别分别定义observable和容差，从最严格到生产可接受逐级放宽。

**主问题得分点（10分）**：三层定义 4 分；并行浮点差异 2 分；验证指标 2 分；目标选择 2 分。

**追问分析与参考回答**：state exact比较保存点tensor/hash；step replay exact比较同batch RNG下下一步activation/grad/update，换并行常难bitwise；convergence equivalent比较多seed loss曲线和最终quality统计。项目/发布需事先选择门槛，不能失败后临时改标准。

**追问得分点（5分）**：三层各1分；并行差异1分；预定义门槛1分。

### M118. 一个Megatron多卡job hang，如何定位？

**高分回答要点**

- 先冻结现场：stack、NCCL flight recorder/debug、每rank最后logical step/stage/microbatch/collective和process-group坐标。
- 判断是数据/compute未进入collective、collective op/shape/order错配、P2P schedule还是硬件/网络。
- 缩小world/group，比较第一个sequence fingerprint分歧，设置有限timeout并清理整个job。

**追问**：同shape顺序错配可能不hang而静默错误，所以无timeout不代表通信正确。

**前置知识与分析过程**：先保全所有rank最后事件，再按“未进入、进入错communicator/sequence、transport故障”三类二分。

**主问题得分点（10分）**：现场 3 分；三类根因 3 分；最小复现 2 分；cleanup 2 分。

**追问分析与参考回答**：为每collective记录group id、logical id、sequence、op、shape/dtype和checksum；若A/B同shape错序，NCCL可按sequence完成但logical checksum错。watchdog用于活性，fingerprint/oracle用于正确性，两者都需要。

**追问得分点（5分）**：活性/正确性区别1分；fingerprint字段2分；checksum1分；watchdog边界1分。

### M119. 出现NaN/Inf时如何按Megatron数据流二分？

**高分回答要点**

- 找第一个step/rank/layer/tensor，区分forward activation、loss、backward grad、collective后grad和optimizer后param。
- 记录amax/norm、loss scale/overflow、router logits、attention softmax、学习率和数据样本。
- 单卡/TP1、关闭FP8/fusion/overlap、固定batch逐项消融，并比较reference。

**追问**：所有rank同时NaN可能由共享数据/超参；单TP/EP rank先NaN可能由shard/kernel/路由，但都需证据。

**前置知识与分析过程**：用first-bad原则按step/rank/layer/phase定位，再关闭复杂特性逐项恢复。每次只改一个变量。

**主问题得分点（10分）**：first-bad定位 3 分；关键数值指标 2 分；消融路径 3 分；rank pattern只作假设 2 分。

**追问分析与参考回答**：全rank同一layer同时坏先检查输入、LR/loss scale和共同kernel；单shard先检查partition、amax、collective前后与router count。都要用hook抓首个非finite tensor，并与TP1/BF16/no-fusion reference比较，不能仅凭分布下结论。

**追问得分点（5分）**：两个假设1分；first-bad2分；reference消融1分；不武断归因1分。

### M120. MCore/TE/PyTorch升级前后如何做兼容验证？

**高分回答要点**

- 固定旧baseline artifact；矩阵验证checkpoint model/optimizer load、短步loss/grad、吞吐/peak和关键kernel/collective。
- 检查参数默认值、ModuleSpec、state-dict key/layout、precision/fusion/backend和弃用项。
- canary小规模再全规模，设数值/性能阈值、回滚镜像和checkpoint前向/后向兼容策略。

**追问**：不能只看import成功或第一步loss finite；optimizer格式可能在真正resume时失败。

**前置知识与分析过程**：列升级影响面并设计从静态load到短步、性能、全规模canary的分层门禁。

**主问题得分点（10分）**：兼容矩阵 3 分；默认/config diff 2 分；数值/性能门槛 3 分；回滚 2 分。

**追问分析与参考回答**：用生产checkpoint分别测试model-only和model+optimizer restore，完成至少数步optimizer和再保存/再加载；比较loss/grad/hash、tok/s/peak/kernel。只有这些通过才canary，且保留旧镜像和旧格式checkpoint回滚。

**追问得分点（5分）**：optimizer resume2分；roundtrip1分；性能/数值1分；回滚1分。

### M121. 数据加载如何与DP/CP/PP并行保持一致？

**高分回答要点**

- 不同DP replicas消费不同样本；同一model-parallel replica内TP/PP/CP ranks必须处理同一逻辑batch/sequence shards。
- 通常由指定rank读取/构造batch后在组内broadcast或按规则一致生成，避免每rank重复IO与错样本。
- sampler需保存consumed samples、epoch/shard/packing状态，worker故障恢复不能跳/重数据而不知情。

**追问**：CP ranks持同一sequence不同chunk，不是独立样本；把它们当DP会直接破坏attention语义。

**前置知识与分析过程**：先按rank坐标决定数据ownership，再设计哪个rank读、如何broadcast/scatter，以及sampler checkpoint。

**主问题得分点（10分）**：DP样本分片 2 分；model-parallel一致batch 3 分；IO方案 2 分；恢复状态 3 分。

**追问分析与参考回答**：同一TP/PP/CP model replica必须共享样本id；CP只按sequence offset切token，attention仍组成同一样本。若各CP rank从sampler取不同样本，交换K/V时会混合不同序列。应由replica数据owner准备并按layout分发，保存统一consumed position。

**追问得分点（5分）**：同一样本2分；错误混合1分；owner分发1分；cursor1分。

### M122. 生产Megatron训练应记录哪些指标？

**高分回答要点**

- 训练：loss、LR、grad norm、loss scale、tokens/s、samples/s、MFU、有效/丢弃tokens。
- 分布式：每parallel group collective/P2P、overlap/exposed、rank step分布、PP bubble/stage、MoE expert负载。
- 系统：GPU clocks/SM/HBM、memory peak/fragmentation、CPU/data/NIC/storage、checkpoint pause、错误/重启。

**追问**：指标需带model/step/rank坐标标签但控制基数；高频per-layer信息可采样或写trace。

**前置知识与分析过程**：按训练语义、并行runtime和系统资源三层设计指标，再为每项决定counter/gauge/histogram和标签。

**主问题得分点（10分）**：三层指标 5 分；MoE/PP专项 2 分；标签/基数 2 分；告警关联 1 分。

**追问分析与参考回答**：在线聚合保留job/model、parallel coordinate和有限phase标签；不要把每parameter/token id做时序标签。per-layer/token分布写采样trace或结构化日志，告警时关联step、rank group和硬件，以便从慢rank回到具体parallel域。

**追问得分点（5分）**：标签选择1分；基数风险1分；聚合/trace分层2分；故障关联1分。

### M123. 如何profile Megatron而不把profiler开销当性能结论？

**高分回答要点**

- 稳态非profiler runs给绝对tokens/s/peak；短窗口trace只解释事件结构。
- 用NVTX标iteration/microbatch/stage/module，采多rank或至少关键ranks，做时间对齐和interval union。
- Nsight Systems看CPU/CUDA/NCCL时间线，Nsight Compute看选定kernel counters；权限/回放开销必须记录。

**追问**：累计NCCL duration不能直接加到compute；collective包含peer wait且跨stream可重叠。

**前置知识与分析过程**：定义稳态性能与profile证据不同用途，时间线统一用interval union/intersection；多rank需时钟/step对齐。

**主问题得分点（10分）**：两类run分离 3 分；NVTX/多rank 2 分；union/peer wait 3 分；工具边界 2 分。

**追问分析与参考回答**：同stream事件可直接排序，跨stream先做每类区间union；compute与NCCL交集是overlap，不能将累计sum相加。再比较各rank进入collective时刻区分peer wait。绝对tok/s来自无profiler重复run。

**追问得分点（5分）**：union1分；intersection1分；多rank enter1分；无profiler基线1分；工具分工1分。

### M124. 如何建设Megatron性能回归CI？

**高分回答要点**

- 分层：单module correctness/kernel microbench、小规模TP/PP/DP集成、固定节点性能、周期性多节点soak。
- 固定环境和拓扑，重复并用统计阈值；同时监控tokens/s、MFU、peak、NCCL、loss/grad与checkpoint。
- 自动保存config diff、trace-on-regression和首个坏commit二分信息；区分噪声、硬件坏卡和代码回归。

**追问**：只测单个热门大配置会让定位成本极高，应为每个并行维保留最小可解释case。

**前置知识与分析过程**：按unit、single-axis integration、combination、scale四层构建测试金字塔；每层同时有correctness和performance阈值。

**主问题得分点（10分）**：四层CI 4 分；环境/统计 2 分；多指标 2 分；自动诊断/二分 2 分。

**追问分析与参考回答**：保留TP2 linear、PP2 schedule、DP2 optimizer、EP2 dispatcher等小case，能在分钟级定位；组合case验证交互，大规模soak只周期运行。回归自动抓config diff和短trace，并用重复/控制节点过滤噪声。

**追问得分点（5分）**：小case2分；组合/规模分层1分；统计噪声1分；自动trace/二分1分。

---

## 11. Level 9：专家系统设计、源码与事故题（M125-M140）

### M125. 用64张H100训练LLaMA类70B，你如何选择并行配置？

**高分回答要点**

- 先要模型层/hidden/heads/GQA、sequence、GBS、精度、节点拓扑与checkpoint SLA；做参数/optimizer/activation容量预算。
- 从DP only起，容量不足引入最小TP，深度扩展用PP，长context用CP，状态容量用distributed optimizer/FSDP。
- 例如TP4/PP4/CP2/DP2只是候选；必须验证head/query-group整除、每stage层、microbatches/bubble和节点映射。

**追问**：给出每rank本地layers/heads、DP group、global batch公式、关键collective及至少五个验收指标。

**前置知识与分析过程**：先问完整约束并做参数/optimizer/activation预算，再枚举合法TP/PP/CP，最后用GBS和拓扑填DP及M。配置是推导结果，不是起点。

**主问题得分点（10分）**：主动澄清 2 分；容量/shape计算 2 分；并行与拓扑 3 分；schedule/GBS 1 分；验收/容错 2 分。

**追问分析与参考回答**：以候选TP4/PP4/CP2/DP2为例，80层若均匀约20层/PP stage但需按端点重平衡，64 Q heads每TP rank16；DP伙伴固定其余坐标。`GBS=MBS*accum*DP`。关键通信含每层TP、CP attention、PP P2P、DP RS/AG；验收tok/s、MFU、peak、PP idle、exposed通信、loss/grad和checkpoint。

**追问得分点（5分）**：layers/heads1分；DP group语义1分；GBS1分；通信1分；>=5指标1分。

### M126. 设计405B dense模型的Megatron训练方案。

**高分回答要点**

- 大参数与深度通常需要TP+PP+distributed optimizer/FSDP，8K+context再加CP；先算每rankmodel states与activation。
- TP限制在高速域，PP跨节点；用VPP/custom layout平衡端点，M足够覆盖bubble。
- 设计distributed checkpoint reshard、async save、故障域和多小时soak，性能目标用MFU/time-to-train而非理论GPU数。

**追问**：为什么不能直接照抄官方示例？硬件数量、网络、GBS、模型GQA/词表和软件版本可能不同。

**前置知识与分析过程**：从模型state容量、单layer容量、深度、context四个约束分别决定FSDP/TP/PP/CP最低度数，再优化效率。

**主问题得分点（10分）**：四类容量约束 3 分；组合并行 3 分；PP/VPP平衡 1 分；checkpoint/soak 2 分；不照抄 1 分。

**追问分析与参考回答**：官方recipe只给一个已知平台起点；实际需重算heads/query groups整除、本地GEMM、PP层数/端点、网络rail和GBS。先小规模验证同配置语义，再扫最低TP/CP与M/VPP，最后多节点soak和DCP恢复。

**追问得分点（5分）**：列四个变化项2分；重算shape1分；渐进验证1分；soak/恢复1分。

### M127. 设计128K context训练，如何在CP、recompute和offload之间选择？

**高分回答要点**

- 估算attention与线性activation，明确attention kernel是否支持目标mask/GQA/packing；先确定最低CP。
- TP保护layer容量但避免过大；CP按拓扑选通信类型；对剩余峰值做selective recompute，offload只在可隐藏带宽时加入。
- 扫context/CP、记录每rankpeak、attention/CP通信、MFU、有效tokens和数值correctness。

**追问**：CP分片sequence不减少全局attention语义，softmax/causal边界必须等价reference。

**前置知识与分析过程**：先算128K下attention/activation，再选支持目标mask/GQA的kernel和CP算法；之后分配recompute/offload。

**主问题得分点（10分）**：容量模型 2 分；CP/TP选择 2 分；attention正确性 2 分；recompute/offload 2 分；验证 2 分。

**追问分析与参考回答**：用短序列小模型构造CP1 reference，CP2/4需对每local Q合并全局K/V并保持online softmax max/sum和causal mask；比较logits/grad。性能再看CP通信、peak、MFU和packing有效tokens，不能用“每rank序列变短”推断数学计算被截断。

**追问得分点（5分）**：全局K/V1分；softmax1分；causal1分；reference1分；性能指标1分。

### M128. 设计一个大规模DeepSeek类MoE训练方案。

**高分回答要点**

- 明确experts/top-k/shared expert/MTP/MLA、active参数、tokens/step和硬件；设计EP、expert-TP、dense TP、PP/DP/CP mesh。
- dispatcher先以NCCL A2A baseline，再按硬件评估DeepEP/HybridEP；GroupedGEMM、permute/router fusion和shared overlap逐项消融。
- 监控per-layer expert负载/drop、A2A exposed、expert GEMM、质量loss和checkpoint/upcycling。

**追问**：为什么fine-grained MoE常倾向expert-TP=1？保护小expert GEMM shape并减少层内通信，但容量可能迫使分片。

**前置知识与分析过程**：先算每expert参数和每step tokens/expert的GEMM M，再决定EP放experts还是ETP切单expert。然后考虑dispatcher硬件。

**主问题得分点（10分）**：模型/active预算 2 分；mesh设计 3 分；dispatcher/优化 2 分；负载/质量/checkpoint 3 分。

**追问分析与参考回答**：fine-grained expert本就token M小，再ETP会把N/K切小并加入TP collective，效率差；优先ETP1、增EP分布不同experts。若单expert参数/activation超单卡容量才加ETP，并用GroupedGEMM与高速域缓解。

**追问得分点（5分）**：M/N/K变小2分；通信1分；容量例外1分；优化/拓扑1分。

### M129. 一个Megatron job间歇OOM，你如何定位？

**高分回答要点**

- 记录OOM rank的TP/PP/CP/EP坐标、phase/microbatch、requested bytes、allocated/reserved和memory snapshot。
- 判断静态端点/stage不均衡、长样本、MoE热点、overlap预取、checkpoint staging、fragmentation或泄漏。
- 复现固定batch，关overlap/async、限制token/capacity，逐项对照；修复后做长时间高水位回归。

**追问**：平均显存正常无意义，PP warmup、expert hotspot和param AG会产生rank/step瞬时峰值。

**前置知识与分析过程**：先记录OOM坐标/时刻/requested bytes，再按静态不均衡、数据动态、overlap瞬态和fragmentation分类。

**主问题得分点（10分）**：现场字段 3 分；四类根因 3 分；单变量消融 2 分；修复/soak 2 分。

**追问分析与参考回答**：PP warmup看并存microbatches，MoE看该step expert counts，param AG看in-flight buckets；memory snapshot区分active与reserved fragments。固定触发batch，分别关overlap、限capacity、开recompute，找到峰值下降的因果项后做长时间p99水位验证。

**追问得分点（5分）**：三类瞬态各1分；snapshot1分；固定batch消融/soak1分。

### M130. MFU突然下降但NCCL带宽看起来正常，怎么查？

**高分回答要点**

- 分解data wait、CPU launch gap、GEMM shape/kernel、PP bubble/stage、MoE负载和collective exposed，而非只看带宽。
- 配置变化可能增TP/PP、减microbatch、关闭fusion/TE、触发recompute或改变sequence分布。
- 用旧/新trace与config diff定位第一个变化phase，跑单rank/module microbench区分kernel与系统。

**追问**：collective algbw正常仍可能等待peer更久；需要进入collective时间与exposed wall。

**前置知识与分析过程**：从step分解开始，不把MFU下降等同通信。比较旧/新config和各phase时间，再找first changed phase。

**主问题得分点（10分）**：完整分解 4 分；config/kernel变化 2 分；peer-wait 2 分；最小对照 2 分。

**追问分析与参考回答**：记录每rankcollective enter/exit；带宽工具正常但enter spread增大说明上游data/GEMM/stage/MoE变慢。若同时进入且duration增大才查网络。用旧batch/旧config单模块复跑，逐项恢复新变化。

**追问得分点（5分）**：enter spread2分；网络判据1分；config diff1分；最小复现1分。

### M131. PP扩展很差，你的前三个实验是什么？

**高分回答要点**

- 固定GBS扫M，验证bubble与局部GEMM；采per-stage F/B/P2P/idle timeline定位最慢stage。
- 统计每stage层/参数/FLOPs/activation，做custom layout或VPP对照。
- 测相邻stage P2P拓扑/bytes并重排ranks，确认通信与stage imbalance各占多少。

**追问**：本机PP8/M16仍只有32.4%效率，优先补stage profile，不应只继续增加M。

**前置知识与分析过程**：前三个实验必须分别隔离bubble、imbalance和P2P/拓扑，且固定GBS或明确变化。

**主问题得分点（10分）**：三实验顺序 5 分；控制变量 2 分；指标 2 分；本机证据 1 分。

**追问分析与参考回答**：一是固定PP扫M建立bubble曲线；二是固定PP/M采per-stage F/B/P2P/idle并重切层；三是测相邻P2P和重排rank。PP8/M16理论利用69.6%但实效32.4%、参数2.76x，说明第二实验优先级高。

**追问得分点（5分）**：三个实验各1分；两个本机数字1分；优先级理由1分。

### M132. TP1与TP4 loss缓慢分叉，如何判断是正常浮点差异还是bug？

**高分回答要点**

- 固定权重/输入/RNG，逐层比较gather后的forward activation、loss、backward grad和一步param。
- 关闭dropout/fusion/mixed precision，再逐项恢复；检查SP复制参数grad、vocab CE和collective顺序。
- 量化误差增长与训练quality，多seed验证；finite和初始loss接近不足以判定正确。

**追问**：若第一层输出已差很大，先查weight partition/init/layout；若optimizer后才差，查grad sync/main params/clip。

**前置知识与分析过程**：建立阶段性oracle并找first divergence；先关闭随机/低精度/融合，再逐层恢复复杂度。

**主问题得分点（10分）**：逐层二分 3 分；消融 2 分；通信/复制grad检查 2 分；容差/质量 3 分。

**追问分析与参考回答**：第一层前先gather比较logical weights和输入；第一层输出差查partition、RNG、collective。全部F/B相近而step后差则比较global norm/clip、main grad RS、master update和param AG。每阶段报告误差而非最终猜因。

**追问得分点（5分）**：first divergence2分；前向路径1分；optimizer路径1分；误差口径1分。

### M133. 只有部分ranks卡在NCCL，如何从并行坐标定位？

**高分回答要点**

- 列卡住ranks属于哪些共同TP/PP/CP/EP/DP groups，找最小共同communicator和缺席peer。
- 对齐最后stage/microbatch/collective sequence/shape，检查某rank是否在data、OOM、NaN分支或不同schedule。
- group内最小复现并启用fingerprint/timeout；不要先kill单rank导致现场丢失。

**追问**：某TP group卡而其他DP replicas正常，优先查该model replica/节点，而非全局训练逻辑。

**前置知识与分析过程**：把卡住rank映射到所有groups，求最小共同故障域；对缺席peer找其最后事件。

**主问题得分点（10分）**：坐标交集 3 分；sequence/shape 3 分；现场/timeout 2 分；定位故障域 2 分。

**追问分析与参考回答**：只有一个TP group卡说明相同训练代码在其他DP replica完成，优先比较该组节点GPU、data shard和last collective fingerprint；找哪个rank没进入或op不同。仍保留全局逻辑可能受数据分支触发的反例。

**追问得分点（5分）**：group交集1分；对照其他replica1分；last event2分；数据分支反例1分。

### M134. checkpoint pause突然从30秒变成5分钟，如何定位？

**高分回答要点**

- 分解state materialization、GPU->CPU、serialization、每rankwrite、metadata/commit和barrier最慢rank。
- 检查checkpoint格式/optimizer reshardability、async是否退化同步、storage throttling、小文件数和某rank shard偏大。
- 对比每rankbytes/throughput、host memory、IO queue和训练干扰；做model-only/optim-only与本地盘对照。

**追问**：所有rank在barrier等5分钟不代表barrier慢，可能只有一个writer/reshard慢。

**前置知识与分析过程**：将checkpoint分phase并对每rank打时间戳/bytes，barrier只表示汇合点。找最后到达者。

**主问题得分点（10分）**：phase分解 4 分；rank skew 2 分；格式/存储假设 2 分；对照 2 分。

**追问分析与参考回答**：记录materialize、D2H、serialize、write、metadata、commit结束时刻；barrier等待等于最慢rank前置phase。若某rank shard大/存储慢，其他rank都在barrier显示长。model-only、optim-only、本地盘对照定位数据量、reshard或远端存储。

**追问得分点（5分）**：phase timestamps2分；last arriver1分；三组对照1分；不归因barrier1分。

### M135. 从`pretrain_gpt.py`如何追到一个TP MLP的NCCL调用？

**高分回答要点**

- 训练入口构建config/model provider，得到GPTModel与layer spec；forward进入TransformerBlock/Layer/MLP。
- MLP调用Column/RowParallelLinear，autograd mapping或linear Function在相应forward/backward触发TP collective。
- 同时追`parallel_state`取得group，记录tensor shape/stream；最后用trace/kernel验证静态阅读。

**追问**：要求现场使用`rg`找symbol definition/call site，而不是凭目录猜。

**前置知识与分析过程**：从入口的model provider和spec对象开始，用symbol调用链向下追，遇到autograd Function再追backward和group getter。

**主问题得分点（10分）**：完整调用链 5 分；collective/group位置 2 分；shape/stream 2 分；动态验证 1 分。

**追问分析与参考回答**：用`rg 'get_gpt_layer.*spec|ColumnParallelLinear|reduce_from_tensor_model_parallel'`找定义和call site；在`GPTModel->TransformerBlock->Layer->MLP->TP layers/mappings`链上记录输入shape。最后用profiler kernel name/NVTX确认该路径实际执行，而非只静态阅读。

**追问得分点（5分）**：rg命令1分；调用链2分；group/shape1分；trace验证1分。

### M136. 如何在MCore中接入一个新Transformer子模块？

**高分回答要点**

- 定义module接口和ModuleSpec wiring，扩TransformerConfig及validation，明确TP/PP/checkpoint layout。
- 单卡reference后做TP/SP/PP correctness、RNG/recompute、mixed precision与state-dict roundtrip。
- 添加kernel/perf benchmark、fallback/feature gate、文档recipe和CI矩阵。

**追问**：若新模块含跨token状态，必须重新评估CP/sequence packing和pipeline boundary，不能只替换Python类。

**前置知识与分析过程**：从模块数学依赖推导各并行维合法layout，再设计spec/config/checkpoint和测试矩阵。

**主问题得分点（10分）**：接口/spec 2 分；并行layout 3 分；checkpoint/RNG 2 分；CI/perf/fallback 3 分。

**追问分析与参考回答**：跨token模块在CP下需交换全局状态或改变partition，packing需阻断样本间状态，PP切点可能需发送额外state tensor。先做单卡reference，再逐CP/PP验证shape与grad，定义sharded state metadata并提供unsupported config fail-fast。

**追问得分点（5分）**：CP1分；packing1分；PP boundary1分；checkpoint1分；fail-fast1分。

### M137. 你会如何review一个“开启通信overlap提速15%”的PR？

**高分回答要点**

- 检查基线公平、硬件/拓扑、GBS、warmup/重复、端到端step与peak；排除异步工作漏出计时窗口。
- 审核stream/event/Work lifetime、buffer复用、collective顺序、gradient accumulation和异常路径。
- 要求多shape/规模、correctness多step、trace证明exposed下降，以及关闭flag无行为回归。

**追问**：局部kernel提前返回但iteration末wait更久，不是端到端提速。

**前置知识与分析过程**：review分correctness、测量、可扩展性三条线；先验证异步work被完整计时，再看依赖和资源。

**主问题得分点（10分）**：公平测量 3 分；stream/Work正确性 3 分；trace证据 2 分；回归矩阵 2 分。

**追问分析与参考回答**：计时窗口末必须wait/synchronize目标work，以max-rank iteration wall为准；trace看通信是否只是移到末尾。检查buffer复用和异常控制流，多step compare参数。只有exposed和step均降、peak可接受才接受15%claim。

**追问得分点（5分）**：wait计时1分；max-rank1分；末尾暴露1分；correctness1分；peak1分。

### M138. 面试官说“你没跑过多机，怎么能说懂Megatron？”你如何回答？

**高分回答要点**

- 明确不伪造多机经验；给出本机真实TP/SP/PP/recompute、process-group顺序、NCCL拓扑和failure证据。
- 展示可迁移方法：tensor/bytes/critical path、rank mapping、trace、correctness和版本边界。
- 对多机部分给具体设计、指标与实验计划，同时指出未验证的RDMA/rail/SHARP/多机EP结论。

**追问**：可信专家不是“所有都跑过”，而是能精确说明证据等级并快速设计验证。

**前置知识与分析过程**：把回答分本机事实、稳定机制、多机待验证计划；每项都给artifact或指标。

**主问题得分点（10分）**：不伪造 2 分；本机证据 3 分；可迁移推导 2 分；多机实验设计 3 分。

**追问分析与参考回答**：列真实tag、硬件和TP/SP/PP/recompute数字；多机明确未测RDMA/rail/EP，给NCCL baseline、rank mapping、multi-rank trace、failure/soak和DCP计划。这样展示方法深度而非用“懂原理”回避证据。

**追问得分点（5分）**：版本事实1分；真实数字1分；未测项1分；具体计划2分。

### M139. 请用两分钟介绍你的Megatron经验。

**参考骨架**

- 架构：直接使用MCore `GPTModel/TransformerConfig/ModuleSpec/parallel_state/schedule`，不是手写TP/PP模拟器。
- 证据：固定8xV100，TP1->8容量提升但吞吐84.6k->14.7k；seq4096 recompute省约78%；PP8/M16为2.59x且stage 2.76x不均衡。
- 深挖：能解释Column/Row layout、SP grad、1F1B、rank groups、communication critical path和下一步profile。
- 边界：版本0.9.0、单机local spec；CP/生产MoE/FSDP/FP8按当前官方机制掌握但未冒充实测。

**追问**：面试官会任选一个数字要求你给命令、shape、global batch、parser和raw artifact。

**前置知识与分析过程**：项目介绍按问题、方法、结果、机制、边界组织；选3-5个能防守的数字，不流水账。

**主问题得分点（10分）**：架构定位 2 分；关键数字 3 分；机制 2 分；深挖/边界 3 分。

**追问分析与参考回答**：例如TP8数字应能说runner、12层hidden768、GBS与TP/DP/accum、三次重复、summary/raw路径；PP数字同理。parser以max-rank step算tok/s，loss finite只是基础oracle。不能给这些时简历数字可信度低。

**追问得分点（5分）**：命令/config1分；GBS1分；parser1分；artifact1分；correctness/边界1分。

### M140. 什么表现才算“Megatron专家”，而不是熟练用户？

**高分回答要点**

- 能从模型公式推导每ranktensor/参数/gradient/通信，不依赖复制recipe。
- 能从parallel config追到process group、schedule、buffer、optimizer和checkpoint state ownership。
- 能用profile和correctness oracle定位性能/数值/故障，并改ModuleSpec/runtime而非只调flag。
- 能区分版本、硬件与证据边界，设计可扩展系统并知道何时不该增加某种并行度。

**最终追问**：随机抽M025、M043、M060、M086、M113、M135，若都能从公式讲到源码和实验，才接近专家表现。

**前置知识与分析过程**：专家标准按张量、调度、状态、MoE、checkpoint、源码六条独立能力交叉验证，避免单方向熟练冒充全栈。

**主问题得分点（10分）**：四项能力定义各2分共8分；证据/版本自律2分。只会调参不超过4分，只会论文公式不超过6分。

**追问分析与参考回答**：M025应推W/Y/dX，M043画时空图，M060走RS-update-AG，M086写count矩阵和逆split，M113做global range reshard，M135从入口追到mapping/NCCL。每题还要给oracle和profile；任两条无法展开说明能力仍有结构性缺口。

**追问得分点（5分）**：六题各层级对应2分；公式到源码1分；oracle/profile1分；能识别自身缺口1分。

---

## 12. 十二组上机实验

| 实验 | 设置 | 必看指标 | 你应掌握的内容 |
|---|---|---|---|
| E01 单卡源码跟踪 | 2层GPT，local spec，单step | module调用、tensor shape、kernel | 从GPTModel走到Attention/MLP/loss |
| E02 TP正确性 | TP1 vs TP2，同权重/输入/RNG | logits/grad/step误差、shard checksum | Column/Row、vocab CE和autograd mapping |
| E03 TP性能 | TP1/2/4/8，固定GBS/GPU | GEMM shape、NCCL exposed、MFU、peak | 容量与效率的Pareto边界 |
| E04 SP与重计算 | seq512/2K/4K，SP/full/selective | allocation分类、AG/RS、recompute FLOPs | 两者优化不同activation集合 |
| E05 PP时间线 | PP2/4/8、M1/4/16 | 每stage F/B/P2P/idle、bubble | warmup/steady/cooldown与慢stage |
| E06 PP重切层 | 等层 vs profile驱动custom layout/VPP | max stage time、bubble、peak | 参数proxy与真实service time区别 |
| E07 3D训练 | TP2xPP2xDP2，固定GBS | group/sequence、loss、各类通信 | 组合并行runtime与rank mapping |
| E08 Distributed Optimizer | baseline vs RS/AG + overlap | bytes/param、bucket、AG/RS exposed、peak | main-grad/param shard状态机 |
| E09 CP长上下文 | CP1/2/4，固定全局sequence | KV通信、attention correctness、peak/MFU | SP与CP边界及通信类型 |
| E10 真实MoE | top2/8 experts，EP1/2/4 | expert分布、A2A、GroupedGEMM、drop | router->dispatch->expert->combine |
| E11 DCP reshard | TP2/PP2保存，TP1/PP4恢复 | IO、peak、checksum、resume loss | sharded metadata与optimizer兼容 |
| E12 故障注入 | kill rank、错collective、慢data、坏shard | detection/abort/cleanup/restore | 从现场到根因的工程闭环 |

### 实验完成标准

每组必须保存：完整命令与commit、每rank原始log/JSON、三次稳态重复、correctness oracle、至少一个负结果、parser公式和环境清理检查。没有这些产物，只能说“运行过”，不能说“做过性能工程”。

---

## 13. 六条连续深挖链

### 13.1 TP深挖链

`为什么TP` -> `Column/Row shape` -> `forward/backward collective` -> `每层bytes` -> `SP如何改layout` -> `GEMM为何变小` -> `如何profile overlap` -> `为什么本机TP8变慢`。

### 13.2 PP深挖链

`为什么PP` -> `1F1B三阶段` -> `bubble公式` -> `microbatch trade-off` -> `P2P shape` -> `stage imbalance` -> `VPP/custom layout` -> `PP hang如何定位`。

### 13.3 Distributed Optimizer深挖链

`为什么省显存` -> `main_grad buffer` -> `ReduceScatter ownership` -> `optimizer shard` -> `param AllGather` -> `双向overlap` -> `峰值为何上升` -> `checkpoint如何reshard`。

### 13.4 MoE深挖链

`top-k router` -> `load balancing` -> `token permutation` -> `variable-split A2A` -> `GroupedGEMM` -> `tail expert` -> `shared/communication overlap` -> `DeepEP硬件边界`。

### 13.5 长上下文深挖链

`OOM组成` -> `SP与CP` -> `CP groups` -> `attention全局依赖` -> `communication type` -> `GQA影响` -> `selective recompute/offload` -> `128K系统设计`。

### 13.6 故障深挖链

`step变慢/卡住` -> `并行坐标` -> `最后logical event` -> `collective sequence/shape` -> `peer wait还是transport` -> `最小复现` -> `故障清理` -> `checkpoint恢复与防复发CI`。

---

## 14. 四周训练计划

### 第1周：架构和TP

- 闭卷完成M001-M040；每天手推两组Column/Row/attention/vocab CE shape。
- 跑E01-E03，从一个NCCL kernel反查到MCore调用点。
- 周末60分钟模拟：架构15分钟、TP推导25分钟、实验答辩20分钟。

### 第2周：PP、DP与状态

- 完成M041-M068，手画PP4/M8和PP8/M16时空图。
- 跑E05和E08；至少解释一个overlap没有提速的负结果。
- 源码阅读`schedules.py`、`p2p_communication.py`、`distributed_data_parallel.py`、`distrib_optimizer.py`。

### 第3周：CP、MoE与性能

- 完成M069-M110，手推TP2/CP2与一个EP mesh。
- 若硬件/版本不支持E09/E10，做最小toy并明确证据边界，不伪造MCore生产能力。
- 模拟面试必须连续走完TP、MoE、长上下文三条深挖链。

### 第4周：Checkpoint、系统设计与高压答辩

- 完成M111-M140，跑E11/E12或写出可执行实验协议。
- 每天一次70B/405B/MoE/128K设计，强制先问约束再给配置。
- 最终门槛：随机30题平均>=3.2；M025/M043/M060/M086/M113/M135均>=3.5；两分钟项目介绍稳定且数字无误。

---

## 15. 源码阅读路径

按下面顺序阅读本仓库固定的MCore 0.9.0，边读边画tensor和group：

1. `megatron/core/parallel_state.py`
2. `megatron/core/transformer/transformer_config.py`
3. `megatron/core/transformer/spec_utils.py`
4. `megatron/core/models/gpt/gpt_model.py`
5. `megatron/core/transformer/transformer_block.py`
6. `megatron/core/transformer/transformer_layer.py`
7. `megatron/core/transformer/attention.py`
8. `megatron/core/transformer/mlp.py`
9. `megatron/core/tensor_parallel/layers.py`
10. `megatron/core/tensor_parallel/mappings.py`
11. `megatron/core/pipeline_parallel/schedules.py`
12. `megatron/core/pipeline_parallel/p2p_communication.py`
13. `megatron/core/distributed/distributed_data_parallel.py`
14. `megatron/core/distributed/param_and_grad_buffer.py`
15. `megatron/core/optimizer/distrib_optimizer.py`

阅读每个文件时回答四个问题：它拥有哪个状态；输入输出layout是什么；调用哪个process group/stream；异常时哪个log/trace能证明执行到了这里。

---

## 16. 官方资料

- [NVIDIA Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-Core Transformer package and ModuleSpec](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/transformer.html)
- [Megatron-Core parallel state API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html)
- [Megatron-Core pipeline parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.html)
- [Custom Pipeline Model Parallel Layout](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/pipeline_parallel_layout.html)
- [Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron-Core Distributed Optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)
- [Megatron-Core DDP API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.distributed.distributed_data_parallel.html)
- [Megatron-Core Distributed Checkpointing](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html)
- [Megatron-Core MoE Guide](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html)

---

## 17. 最终自检清单

- [ ] 能在白板上画Megatron-Core包结构和一个GPT layer的数据流。
- [ ] 能手推Column/Row Parallel的forward/backward tensor shape和collective。
- [ ] 能解释SP的RS/AG layout、复制参数gradient以及它与CP的区别。
- [ ] 能画非交错1F1B的warmup/steady/cooldown并计算bubble。
- [ ] 能从per-stage timeline设计custom layout/VPP对照。
- [ ] 能解释main_grad、bucket、ReduceScatter、optimizer shard和Parameter AllGather完整状态机。
- [ ] 能推导distributed optimizer每参数bytes并说明理论公式未包含什么。
- [ ] 能画TP/PP/DP/CP/EP groups，说明dense与expert参数的不同同步域。
- [ ] 能从router一路讲到permutation、A2A、GroupedGEMM、combine和load imbalance。
- [ ] 能区分AllGather/A2A/Flex dispatcher，并主动说明硬件和版本前提。
- [ ] 能为70B、405B、128K和大MoE先做容量/通信估算再给并行配置。
- [ ] 能从NCCL hang、NaN、OOM、低MFU、PP慢和checkpoint慢中保全现场并二分。
- [ ] 能说明sharded state dict如何跨TP/PP/DP reshard，以及optimizer为什么更复杂。
- [ ] 能从`pretrain_gpt.py`追到TP linear的collective调用，并用trace验证。
- [ ] 能准确引用本机TP/SP/PP/recompute数字，同时不冒充跑过CP、生产MoE、Megatron-FSDP或FP8。
