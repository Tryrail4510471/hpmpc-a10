# FID=202 BERT FFN 真实三机测试与性能报告

## 环境

- 集群：3 台 NVIDIA A10 24GB，Ubuntu 22.04，8 核 CPU，28 GiB 可见 RAM
- 网络：15 Gbps 内网；P0/P1/P2 为 `172.24.178.175/176/177`
- 公网管理地址：`47.99.65.0 / 47.97.60.170 / 116.62.10.248`
- 框架：hpmpc，`PROTOCOL=5`（Trio 半诚实 3PC）
- GPU：sm_86，CUDA 12.8，CUTLASS
- 定点配置：`BITLENGTH=32 DATTYPE=512 FRACTIONAL=20 TRUNC_APPROACH=2`
- Owner：`MODELOWNER=0 DATAOWNER=1`
- 日期：2026-07-16（Asia/Shanghai）

## 结论

当前 **ReLU 占位版** FID=202 已在真实三机上完成 core 与 full-chain 数值闭环：T1、T2、T3 对相同 ReLU 明文函数的最大绝对误差分别为 `0.006078`、`0.008660`、`0.052199`，均低于 0.1，且没有 NaN/Inf。

原生 HuggingFace `BertIntermediate` 使用 GELU。当前 ReLU full-chain 对 GELU reference 的最大误差为 `1.727454`，因此不能声称已通过“与原生 HuggingFace GELU 输出一致”的验收；这是算子定义差异，不是 MPC 随机误差。

性能验收未完全通过：8-worker 单 constellation 的活跃窗口网络利用率为 `37.278%`，低于 40%；48-worker full split 因私有 limb 权重预计需要约 `54.03 GiB/node`，超过节点内存，未冒险启动。

## 为数值正确性完成的修复

1. `MODELOWNER`/`DATAOWNER` 不再硬编码为 P0。
2. HuggingFace `[out,in]` 权重在 owner 侧转置为 CUTLASS 所需 `[in,out]`。
3. 32-bit Q20 GEMM 改为 10-bit limb：`HH + ((HL+LH)>>10)`，避免 Q40 点积在截断前丢失高位。
4. 支持 `core` 与 `full` 输出模式，以及按 `process_offset` 的 worker 分片输出。
5. LayerNorm 在 `x/2` 上计算统计量并使用 `epsilon/4`，避免约 `|x|>45.25` 时单元素平方超出 Q12.20 范围。
6. 使用框架原生 `start_layer_stats/stop_layer_stats` 记录各子模块时间和通信。

## 数值测试

| Case | 形状 | MPC 路径 | 对比 reference | max abs | mean abs | RMSE | 结果 |
|---|---:|---|---|---:|---:|---:|---|
| T1 | `[1,4,768]` | Linear-ReLU-Linear | ReLU core | 0.006078 | 0.000381 | 0.000509 | PASS |
| T2 | `[2,128,768]` | Linear-ReLU-Linear | ReLU core | 0.008660 | 0.000391 | 0.000530 | PASS |
| T3 | `[2,128,768]` | Linear-ReLU-Linear-Residual-LN | ReLU full | 0.052199 | 0.000576 | 0.001424 | PASS |
| T3-contract | `[2,128,768]` | ReLU full | HuggingFace GELU full | 1.727454 | 0.091225 | 0.120302 | FAIL（定义不同） |
| T4 | `[1,128,768]` | 完整 ReLU full | ReLU full | 0.052199 | 0.000578 | 0.001491 | PASS |

T1 在修复前最大误差为 `24.278652`；完成 CUTLASS 转置和 limb-GEMM 后下降到 `0.006078`。权重导出文件 SHA-256 为 `a18d05211dc40e11e3d1a71dba1a826ab047c2213f2b9a89c4aec356947901b2`，Q20 导出重载误差为 `4.7684e-07`。

## 延迟与通信

真实三机单 worker 结果：

| Case | batch | 最大 protocol chrono | ms/sample | 总线传输量（发送只计一次） | MB/sample |
|---|---:|---:|---:|---:|---:|
| T1 core | 1 | 3809.149 ms | 3809.149 | 614.209 MB | 614.209 |
| T2 core | 2 | 3582.985 ms | 1791.493 | 762.963 MB | 381.481 |
| T3 full | 2 | 3874.000 ms | 1937.000 | 891.023 MB | 445.511 |
| T4 full | 1 | 3747.905 ms | 3747.905 | 747.906 MB | 747.906 |

通信量包含在线分享模型。T4 中模型分享约 `604.3 MB`，若模型已预分享，推理阶段线传输约 `143.606 MB/sample`。

8-worker 单 constellation 性能点：

| 指标 | 实测值 |
|---|---:|
| 最大 protocol chrono | 7127.597 ms/sample |
| controller wall（P0） | 9564 ms |
| 总线传输量 | 4981.930 MB/sample |
| 去除重复模型分享后的传输量 | 147.530 MB/sample |
| 相对单-worker 加速 | 0.526×（负扩展） |

负扩展主要来自每个 worker 重复私密分享完整权重，以及 8 个进程争用同一 A10。

## 子模块耗时饼图数据

以下采用 8-worker 单 constellation 中 P1 各 worker 的平均 live 时间；总和 `6942.020 ms`，与该配置关键路径相符。

| 子模块 | 平均时间 (ms) | 饼图占比 (%) |
|---|---:|---:|
| InputShare | 0.127 | 0.002 |
| ModelShare | 628.525 | 9.054 |
| Linear 768→3072 | 4485.250 | 64.610 |
| ReLU | 2.367 | 0.034 |
| Linear 3072→768 | 1570.625 | 22.625 |
| Residual | 0.008 | <0.001 |
| LayerNorm | 233.463 | 3.363 |
| LayerNorm affine | 21.641 | 0.312 |
| OutputReveal | 0.014 | <0.001 |

两层 GEMM 合计占 `87.235%`，是下一阶段优化重点。

## GPU 利用率时序与空闲窗口

采样间隔 1 秒，空闲定义为 `SM <= 20%`。

| 节点 | SM 时序 (%) | 平均 SM | 空闲采样比例 | 连续空闲窗口 |
|---|---|---:|---:|---|
| P0 | `0,0,0,0,0,58,96,59,0,0` | 21.3% | 70.0% | 16:46:35–39；16:46:43–44 |
| P1 | `0,0,0,0,0,12,0,0,12` | 2.7% | 100.0% | 整个 16:46:35–43 |
| P2 | `0,0,0,0,0,70,75,0` | 18.1% | 75.0% | 16:46:37–41；16:46:44 |
| 集群采样 | 27 个样本 | 14.1% | 81.5% | 三机均存在超过总时长 20% 的空闲窗口 |

GPU 空闲主要发生在模型分享、A/B 转换、精确截断和 Reveal 阶段；当前 CUTLASS kernel 呈短脉冲而非持续占用。

## 网络利用率

按 `eth0` 的 RX+TX 计数器和 15 Gbps 额定带宽计算：

| 节点 | RX+TX | controller wall | 平均利用率 |
|---|---:|---:|---:|
| P0 | 5081.244 MB | 9.564 s | 28.335% |
| P1 | 90.922 MB | 8.903 s | 0.545% |
| P2 | 5237.755 MB | 7.370 s | 37.903% |

以框架记录的 `4981.930 MB` 单向线传输量和最大 protocol chrono `7.127597 s` 计算，活跃窗口网络利用率为：

```text
4981.930 MB × 8 / (7.127597 s × 15 Gbps) = 37.278%
```

因此 `>40%` 条件 **未通过**，差 2.722 个百分点。继续增加 constellation 会重复模型副本；16-worker 尝试还出现角色排列收尾无法退出，不作为论文数据。

## 48-worker full split 内存约束

limb-GEMM 为两个矩阵各保存 high/low 分享，共 `9,442,560` 个 Trio arithmetic shares。每个 share 约 128 bytes：

```text
1 worker: 1.126 GiB model shares
48 workers: 54.031 GiB/node
```

这尚未包含 CUDA 解包缓冲、激活、通信队列和 context，而节点只有 28 GiB 可见 RAM。脚本默认拒绝该配置，除非显式设置 `HPMPC_ALLOW_OVERSUBSCRIBED_SPLIT=1`。

## 验收状态

- [x] T1–T3 对当前 ReLU 算子合同误差均 `<0.1`
- [ ] 对原生 HuggingFace GELU 输出误差 `<0.1`（需实现 MPC GELU）
- [x] 真实三机 batch=1 延迟已记录：`3747.905 ms/sample`（单 worker）
- [x] 总通信量已记录：`747.906 MB/sample`（单 worker，包含模型分享）
- [x] GPU 利用率曲线已记录，空闲窗口超过 20%
- [ ] 网络利用率 `>40%`（实测 `37.278%`）
- [x] 无 NaN/Inf

## 原始产物

- `results/ffn_logs/T1_metrics.json` 至 `T4_metrics.json`
- `results/ffn_logs/T{1,2,3}_p{0,1,2}.log`
- `results/ffn_logs/T4_p{0,1,2}.log`
- `results/ffn_logs/T4_p{0,1,2}_gpu_dmon.log`
- `results/ffn_logs/T4_constellation_p{0,1,2}.log`
- `results/ffn_logs/T4_constellation_p{0,1,2}_gpu_dmon.log`
