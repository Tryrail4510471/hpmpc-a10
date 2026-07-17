# FID=203 BERT-base 论文实验结果

**状态：PENDING_REAL_FID203_EXPERIMENTS**

当前文件是完整性门禁尚未通过时的状态页，不包含任何本项目占位实测值。
`experiments/collect_results.py` 仅在以下真实证据全部存在并通过哈希校验后才会覆盖本文件：

- MRPC、RTE 各3次完整验证集 FID=203 推理及三方日志；
- serial、pipeline 在 batch=1/4/8/16/32/64 下各5次真实运行，或带三方日志的明确 OOM；
- 1/10/15Gbps 各3次运行，以及 `tc qdisc` 已恢复的确认记录。

预定实验配置：3×NVIDIA A10 24GB，Ubuntu 22.04，15Gbps 内网，hpmpc Trio 半诚实3PC，
FID=203，BERT-base 12层，DATTYPE=512，FRACTIONAL=20，PROCESS_NUM=1，SPLITROLES=0。

| 方法 | 协议 | MRPC(%) | RTE(%) | 延迟(s) | 通信(GB) | NIC利用率 |
|---|---|---:|---:|---:|---:|---:|
| PUMA | 3PC RSS | — | 70.0 | 33.913 | 10.773 | — |
| 本文（无流水线） | Trio 3PC | PENDING | PENDING | PENDING | PENDING | PENDING |
| 本文（流水线） | Trio 3PC | PENDING | PENDING | PENDING | PENDING | PENDING |

PUMA 行是其论文公开的 BERT-base/RTE、序列长度128数据，不是本项目实测值；PUMA未报告MRPC。
本文两行在 truth gate 通过前不得用于论文。
