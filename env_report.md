# HPMPC 三机环境验收报告

- 原始验收时间：2026-07-15（Asia/Shanghai）
- 公网 IP 复核时间：2026-07-17（Asia/Shanghai）
- 验收结论：**PASSED，可以进入开发阶段**

## 环境说明（每次提交前附上）

- 集群：3 台 NVIDIA A10 24GB，Ubuntu 22.04，8 vCPU，标称 32GB RAM（系统可见约 28GiB）
- 网络：三机同一内网，目标带宽 15Gbps
- 框架：hpmpc（github.com/chart21/hpmpc），`PROTOCOL=5`（Trio 半诚实 3PC）
- GPU 架构：`sm_86`，CUDA 12.x
- CPU：Intel Xeon Platinum 8369B，支持 AVX-512
- 推荐编译参数：`DATTYPE=512 PROCESS_NUM=8 SPLITROLES=1 FRACTIONAL=20 USE_CUDA_GEMM=2`
- 目标：实现 BERT-base 的 Softmax、LayerNorm、FFN 私有推理，投 SCI 4 区
- 同步方式：代码主仓库位于 p0，构建产物通过 rsync 同步至 p1、p2

## 1. 三机一致性

| 项目 | p0 | p1 | p2 |
|---|---|---|---|
| 公网 IP | 47.99.78.41 | 118.178.242.191 | 101.37.149.54 |
| 内网 IP | 172.24.178.175 | 172.24.178.176 | 172.24.178.177 |
| 操作系统 | Ubuntu 22.04.5 LTS | Ubuntu 22.04.5 LTS | Ubuntu 22.04.5 LTS |
| Kernel | 5.15.0-177-generic | 5.15.0-177-generic | 5.15.0-177-generic |
| GCC | 11.4.0 | 11.4.0 | 11.4.0 |
| hpmpc SHA | `b39e34ae7ccc80f87d09dbc050b9017cb7cd5838` | 相同 | 相同 |
| PIGEON SHA | `09ee6ba9a6c4c450930e4179a847a819d02561ed` | 相同 | 相同 |
| Pygeon SHA | `a36038042cf5f3fb293e92a1f72bcb3f20d6d18b` | 相同 | 相同 |
| CUTLASS SHA | `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e` | 相同 | 相同 |
| Git 状态 | clean | clean | clean |

2026-07-17 公网地址变更后重新执行严格 SSH 主机密钥校验。新 P0/P1/P2 的 ED25519 指纹分别与
变更前对应角色完全一致，主机名仍为 `p0/p1/p2`；内网地址、A10 GPU、CUDA 12.8 与仓库路径均未改变。
公网地址只用于管理面 SSH，Trio 3PC 仍使用上表中的 `172.24.178.175/176/177`。

三机 split-roles 可执行文件数量均为 18，文件集合 SHA-256 均为：

```text
80ade4ccf8baff8ef014b441f773e3a56ab06b8c6a44b0ce3958bdb96306f270
```

## 2. Step 1：基础单元测试

结果：**PASSED**。

p0 使用以下参数重新生成 Trio split-roles 基础测试（`FUNCTION_IDENTIFIER=54`）：

```bash
make clean
PATH=/usr/local/cuda/bin:$PATH make -j8 \
  PARTY=all FUNCTION_IDENTIFIER=54 PROTOCOL=5 \
  DATTYPE=512 PROCESS_NUM=8 SPLITROLES=1 FRACTIONAL=20 \
  USE_CUDA_GEMM=2 NVCC=/usr/local/cuda/bin/nvcc
```

- 编译退出码：0
- 18 个可执行文件全部生成并成功链接 CUDA 对象
- 构建产物通过 rsync 同步到 p1、p2，三机集合哈希一致
- 单元测试结果：每个测试实例均为 `Passed 5 out of 5 tests.`

## 3. Step 2：p0 本机三方模拟

结果：**PASSED**。

```bash
timeout 180 scripts/run.sh -p all -s 1
```

- 退出码：0
- 成功连接：144 次
- `Passed 5 out of 5 tests`：144 次
- 汇总：`No errors in Split roles execution`
- 超时或残余错误：0

## 4. Step 3：真实三机 Trio 3PC

结果：**PASSED**，三机均无超时。

任务给出的 `host-p0`、`host-p2`、`host-p3` 在三机上无法解析，并且编号与实际 p0/p1/p2 不一致。测试使用实际内网地址：

```bash
# p0
scripts/run.sh -p 0 -s 1 \
  -a 172.24.178.175 -b 172.24.178.176 -c 172.24.178.177

# p1
scripts/run.sh -p 1 -s 1 \
  -a 172.24.178.175 -b 172.24.178.176 -c 172.24.178.177

# p2
scripts/run.sh -p 2 -s 1 \
  -a 172.24.178.175 -b 172.24.178.176 -c 172.24.178.177
```

每条命令外层均设置了 180 秒超时。

| 主机 | 脚本退出码 | 成功连接 | 5/5 测试通过 | split-roles 汇总 | 超时/残余错误 |
|---|---:|---:|---:|---|---:|
| p0 | 0 | 48 | 48 | No errors | 0 |
| p1 | 0 | 48 | 48 | No errors | 0 |
| p2 | 0 | 48 | 48 | No errors | 0 |

日志保存在各服务器的 `/tmp/hpmpc-real3pc.log`。

## 5. Step 4：CUTLASS GPU 模块

结果：**PASSED**。

p0 从 clean 状态执行：

```bash
cd ~/hpmpc/core/cuda
make clean
PATH=/usr/local/cuda/bin:$PATH make -j8 arch=sm_86 \
  CUDA_PATH=/usr/local/cuda \
  CUTLASS_PATH=/home/user/cutlass
```

- 退出码：0
- `gemm_cutlass_int.o`：成功生成
- `conv_cutlass_int_NCHW.o`：成功生成
- `conv_cutlass_int_CHWN.o`：成功生成
- NVCC 编译错误：0

## 6. GPU 与 CUDA 摘要

| 主机 | GPU | 显存 | Compute Capability | Driver | CUDA Toolkit | 温度（采样时） | 功率上限 |
|---|---|---:|---:|---|---|---:|---:|
| p0 | NVIDIA A10 | 23028 MiB | 8.6 | 580.126.09 | 12.8 | 28°C | 150W |
| p1 | NVIDIA A10 | 23028 MiB | 8.6 | 580.126.09 | 12.8 | 27°C | 150W |
| p2 | NVIDIA A10 | 23028 MiB | 8.6 | 580.126.09 | 12.8 | 33°C | 150W |

NVCC 构建信息三机一致：

```text
Build cuda_12.8.r12.8/compiler.35583870_0
```

## 7. iperf3 六方向带宽

测试使用内网地址、TCP、8 条并行流、5 秒有效采样（忽略首秒 warm-up）。验收以接收端吞吐为准。

| 方向 | 发送端 | 接收端 | 重传次数 | >12Gbps |
|---|---:|---:|---:|---|
| p0 → p1 | 13.495 Gbps | **13.557 Gbps** | 97,139 | 是 |
| p1 → p0 | 13.046 Gbps | **13.067 Gbps** | 79,842 | 是 |
| p0 → p2 | 12.847 Gbps | **12.506 Gbps** | 70,202 | 是 |
| p2 → p0 | 12.606 Gbps | **12.634 Gbps** | 63,414 | 是 |
| p1 → p2 | 12.582 Gbps | **12.563 Gbps** | 47,195 | 是 |
| p2 → p1 | 12.747 Gbps | **13.014 Gbps** | 66,076 | 是 |

- 最低接收端带宽：**12.506 Gbps**
- 最高接收端带宽：**13.557 Gbps**
- 六方向全部超过 12Gbps 验收线
- 8 流饱和测试出现较多 TCP 重传；当前不影响带宽验收，但正式性能实验应记录该现象，并考虑检查拥塞控制、socket buffer 和虚拟交换网络抖动

## 8. 最终验收

| 验收项 | 条件 | 结果 |
|---|---|---|
| 单元测试 | PASSED | **通过** |
| 本机三方模拟 | 无错误 | **通过** |
| 真实三机 3PC | 无超时 | **通过** |
| 两两带宽 | 六方向 >12Gbps | **通过** |
| CUTLASS | sm_86 编译无错误 | **通过** |

**最终结论：三机环境一致，Trio 3PC 通信与 CUTLASS GPU 工具链均可用，可以进入 Softmax、LayerNorm 和 FFN 开发阶段。**
