# NCCL 实验与博客证据标准

任何实验只有同时给出正确性、性能、路径和源码证据，才可以写成“验证了某个结论”。

## 1. 实验目录

每次实验使用 UTC 时间戳目录：

```text
logs/<experiment>/<YYYYmmddTHHMMSSZ>/
├── manifest.txt       完整环境、命令、git commit 和环境变量
├── raw/               未修改的 stdout/stderr 和 profiler 文件
├── summary.csv        机器可读结果
├── summary.md         人类可读结果、结论和限制
└── artifacts/         topology XML、nsys report 等附件
```

原始日志不可手工修改。解析脚本只生成新的 summary 文件。

## 2. 四类必需证据

### 正确性证据

- collective 脚本必须用 CPU 公式生成 expected result 并自动断言。
- nccl-tests 必须记录 out-of-place 和 in-place 的 `#wrong`。
- 任一 `#wrong != 0` 时，该性能数据无效。

### 性能证据

- 至少包含消息大小、rank 数、collective、datatype、in-place 模式和拓扑。
- 正式基线至少十次独立 cycle，报告 median、P95、min、max 和 CV。
- CV 超过 5% 时标记为不稳定，先调查噪声，不直接比较优化幅度。
- 小消息报告 latency，大消息同时报告 algbw 和 busbw。

### 路径证据

- 保存 `NCCL_DEBUG` 日志，证明 algorithm、protocol、channel 和 transport。
- 保存 `nvidia-smi topo -m`；多机场景同时保存 GPU/NIC/NUMA 拓扑。
- 对执行链结论保存 Nsight Systems 或等价 timeline。

### 源码证据

- 记录运行时 NCCL 版本和对应源码 tag/commit。
- 正文直接展示关键源码连续片段，不能只给文件路径或函数名。
- 同时提供带课程注释的版本，分析输入、输出、状态修改和条件分支。
- 把实验变量映射到具体源码分支，并用日志或 profiler 证明实际命中。
- `master` 新代码只能解释新版本设计，不能替代运行版本源码证据。

## 3. 控制变量

每组实验只改变一个主要变量。固定并记录：

```text
GPU 型号和 GPU 集合
rank 数与 rank/device 映射
CPU/NUMA 绑定
NCCL/CUDA/PyTorch 版本
collective、datatype、message size
warmup、iterations、blocking/in-place 模式
全部 NCCL_* 和 TORCH_NCCL_* 环境变量
GPU 时钟、P-state 和并发 workload
```

强制 `NCCL_ALGO`、`NCCL_PROTO`、channel 或禁用 transport 只用于诊断实验，不直接作为生产推荐。

## 4. 博客固定结构

每篇博客按以下顺序：

1. 要回答的问题和可证伪假设。
2. 前置知识和数学模型。
3. 当前机器上预测会发生什么。
4. 实验脚本、参数矩阵和控制变量。
5. 与实验相关的 NCCL/nccl-tests/PyTorch 关键源码。
6. 原始结果和结构化表格。
7. 正确性、性能、路径和源码四类证据。
8. 从结果推出的结论。
9. 结论的硬件、版本和 workload 边界。
10. 生产排障与调优方法。

源码展示和完成定义遵循 [ARTICLE_STANDARD.md](ARTICLE_STANDARD.md)。

## 5. 禁止事项

- 没有实际运行时不得写“实验验证”。
- 没有 HCA 时不得推断 RDMA、RoCE、GDR 或 multi-rail 性能。
- 没有 NVSwitch/NVLS 硬件时不得根据源码推断实际加速比。
- 不用一次运行的均值代表稳定基线。
- 不把相关性写成因果关系；必须通过隔离实验或源码证明。
- 不把 teardown 阶段的 abort/close 日志单独判定为训练失败。
