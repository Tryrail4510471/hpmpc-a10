# Softmax 通信与 FFN GPU 矩阵乘法流水线分析

## 环境与数据来源

- 集群：3 台 NVIDIA A10 24GB，Ubuntu 22.04，8 核 CPU / 32GB RAM
- 网络：15Gbps 内网，PROTOCOL=5（Trio 半诚实 3PC）
- GPU：sm_86，CUDA 12.x；定点数 Q12.20
- Softmax 数据：`tests/results/softmax_3pc_20260715/softmax_3pc_test_report.md`
- LayerNorm 数据：`tests/results/layernorm_3pc_20260715/layernorm_3pc_test_report.md`
- FFN 数据：`results/ffn_profile.md` 与 `results/ffn_logs/T4_p*.log`

## 前置数据回填

| 指标 | 实测值 | 口径 |
|---|---:|---|
| FFN 端到端延迟 | 3747.905 ms/sample | 单 worker，batch=1，完整 ReLU FFN+LN |
| Softmax 通信量 | 241.321 MB/sample/node | 48-worker T3，batch=2 |
| Softmax 延迟 | 1457.325 ms/batch | T3 最大 worker latency；约 728.663 ms/sample |
| LayerNorm 通信量 | 51.738 MB/sample/node | 48-worker T2，batch=4 |
| LayerNorm 延迟 | 525.462 ms/batch | T2 最大 worker latency；约 131.366 ms/sample |
| GPU 空闲采样 | 81.5% | 三机 `SM <= 20%` 的采样比例 |
| 两段 Linear live 时间 | 6055.875 ms | 8-worker：4485.250 + 1570.625 ms |
| 两段 Linear live 时间（单 worker） | 2994 ms | P1 T4：1627 + 1367 ms |

`Linear live` 包含输入 limb 截断、CUTLASS、GEMM 后通信和精确截断，不是 CUDA event 测得的纯 kernel 时间。Softmax 的 `1457.325 ms` 同样是完整协议延迟，而非仅 socket 时间。因此用“Softmax 完整延迟 < Linear live 时间”判断重叠空间是保守的：通信子集不会比完整 Softmax 更长。

## 时序分析

8-worker 实测粗粒度时间轴：

```text
时间(ms)  0               1457                         4485       6056
           |----------------|----------------------------|----------|
GPU/FFN N  |================ Linear-1 ===================|ReLU|=L2==|
CPU/NET    |--- Softmax communication, batch N+1 --------|
           |<---- Softmax 完整协议 1457.325 ms ---------->|
```

单 worker 对照：

```text
时间(ms)  0               1457                         2994
           |----------------|----------------------------|
GPU/FFN N  |================ Linear-1 + Linear-2 =========|
CPU/NET    |--- Softmax, batch N+1 -----------------------|
```

定量结论：

```text
Softmax total / Linear live = 1457.325 / 6055.875 = 24.06%
理论可隐藏上限             = min(1457.325, 6055.875) = 1457.325 ms
粗粒度成对任务理论加速比   = (6055.875 + 1457.325) / 6055.875 = 1.241x
```

Softmax 总延迟小于 Linear live 时间，GPU 空闲 81.5% 又明显高于 15%，所以不触发“放弃流水线”的降级决策。

## 代码审计与安全边界

仓库中没有独立的“ReLU 异步流水线”实现。现有机制是：

1. `core/networking/server.hpp` 和 `client.hpp` 使用后台 pthread 处理 socket。
2. `programs/functions/GEMM.hpp` 的 `prepare_GEMM` 先执行 `T::GEMM`，再把乘法消息追加到全局发送缓冲区。
3. `core/cuda/include/gemm.cuh` 每次 CUTLASS 调用后执行 `cudaDeviceSynchronize()`。
4. `core/networking/buffers.h` 中发送/接收缓冲、轮次计数和索引均为进程级全局变量，没有针对两个协议线程的隔离或互斥。

因此，直接在 FID=202 内写：

```cpp
std::async([&] { softmax_mpc_forward(...); });
gemm_fixed(...);
```

是不正确的。两个线程会同时修改相同的发送缓冲和通信轮次，三方消息顺序可能不一致，从而产生错误结果或死锁。

## 实现方案

采用进程隔离的 inter-batch 双流水线：

```text
3PC session A, BASE_PORT=31000: FFN Linear(batch N), USE_CUDA_GEMM=2
3PC session B, BASE_PORT=21000: Softmax(batch N+1), CPU + network
```

两个 session 各自拥有通信缓冲区、socket、协议轮次和输出文件；只共享物理 NIC、CPU 与 A10。串行基线先执行 session B 再执行 session A；流水线实验同时启动六个进程（三台机器各两个）。这对应 steady-state 的相邻 batch 调度，不改变单个 MPC 协议的消息顺序。

修改文件：

- `experiments/pipeline_comparison.sh`：构建两个不同 `BASE_PORT` 的可执行文件，生成 batch=1/4/8/16/32 输入，执行串行和流水线模式，各重复 5 次，输出原始日志、`timings.csv` 和 `summary.csv`。
- `results/pipeline_timing_analysis.md`：记录实测输入、时序、线程安全判断和理论收益。

## 实验命令

在 P0 执行：

```bash
cd /home/user/hpmpc
bash experiments/pipeline_comparison.sh
```

仅构建和同步：

```bash
bash experiments/pipeline_comparison.sh --prepare-only
```

复用已构建的两个 session：

```bash
bash experiments/pipeline_comparison.sh --run-only --repeats 5 --batches 1,4,8,16,32
```

主评价指标是 `serial_mean_ms / pipeline_mean_ms`。理论值 1.241x 只是上限；实际值会受到两个 session 同时占用 NIC、FFN 的精确截断通信以及内存带宽竞争影响。

## 真实三机冒烟结果

完成两套可执行文件编译后，在 batch=1 上分别执行一次串行和流水线模式：

| batch | repeat | 串行 wall | 流水线 wall | 加速比 | 节省时间 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4484 ms | 3790 ms | 1.1831x | 15.477% |

- 第一轮端口冒烟的串行/流水线 FFN 最大 protocol chrono 分别为 3558.242/3477.239 ms；加入输出哈希校验后的复测仍正常完成。
- batch=1 Softmax protocol chrono 约 460 ms，能够被 FFN Linear 阶段覆盖。
- 串行和流水线共 12 个三方日志未发现 timeout、exception 或连接错误。
- 输出文件尺寸符合预期：Softmax 786,448 bytes；FFN Q20 393,220 bytes。
- 哈希校验复测中，串行与流水线的 Softmax SHA-256 均为 `7104933ada03c3d781e2d7781db26597aca9a4527b0ee302c0a33e338c619028`，FFN SHA-256 均为 `4a2c0aa835857804659b0783a6c9cd54f5d08763e4a642fd754ad5a35c6d6f70`。

这是一轮调度与端口隔离冒烟，不替代 batch=1/4/8/16/32 各 5 次的正式统计。正式脚本会记录每轮两个输出的 SHA-256，并要求同一 batch 的串行/流水线输出哈希完全一致。
