# BERT-base MPC论文Table模板

状态：`PENDING_REAL_FID203_EXPERIMENTS`

本模板只能由 `experiments/collect_results.py` 在MRPC/RTE准确率、完整性能矩阵和三点带宽实验的
证据门禁全部通过后填充。FID=203单样本smoke结果不得写入本文行。

| 方法 | 协议 | MRPC(%) | RTE(%) | 延迟(s) | 通信(GB) | NIC利用率 |
|---|---|---:|---:|---:|---:|---:|
| PUMA | 3PC RSS | — | 70.0 | 33.913 | 10.773 | — |
| 本文（无流水线） | Trio 3PC | PENDING | PENDING | PENDING | PENDING | PENDING |
| 本文（流水线） | Trio 3PC | PENDING | PENDING | PENDING | PENDING | PENDING |

PUMA行为论文公开的BERT-base/RTE、序列长度128数据，不是本项目实测值；PUMA未报告MRPC。
本文配置为3×A10、FID=203、12层、Q12.20、`PROCESS_NUM=1`、`SPLITROLES=0`。
