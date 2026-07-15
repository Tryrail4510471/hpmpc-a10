# Softmax MPC real-3PC test report

Test date: 2026-07-15 (Asia/Shanghai)

## Environment

- Cluster: three NVIDIA A10 24 GB hosts, Ubuntu 22.04, 8 CPU cores and 32 GB RAM per host
- MPC network: private 15 Gbps network; P0 `172.24.178.175`, P1 `172.24.178.176`, P2 `172.24.178.177`
- Framework: hpmpc, `PROTOCOL=5` (Trio semi-honest 3PC)
- GPU/toolchain: `sm_86`, CUDA 12.x, AVX-512 available
- Build: `PARTY=all FUNCTION_IDENTIFIER=200 PROTOCOL=5 DATTYPE=512 PROCESS_NUM=8 SPLITROLES=1 FRACTIONAL=20 USE_CUDA_GEMM=2 arch=sm_86`
- Source of truth: P0 `/home/user/hpmpc`; P1/P2 received the P0 build and sources

## Build and execution result

The FUNCTION_IDENTIFIER=200 target compiled successfully on P0. The real three-host runs completed without connection timeout. T1 through T4 were run sequentially; a later case was started only after the preceding case passed.

The test harness reads dimensions from the binary tensor header. Therefore all four shapes use the same FUNCTION_IDENTIFIER and protocol implementation without source edits between cases. Multi-process runs emit row-sharded output files, which the verifier checks for complete, non-overlapping row coverage before calculating accuracy.

## Numerical test results

| Case | Shape | Threshold | Maximum absolute error | Mean absolute error | Maximum row-sum error | Finite | Result |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1, all zeros | `[1,1,4,4]` | `< 0.001` | 0.000003815 | 0.000003815 | 0.000015259 | yes | PASS |
| T2, normal seed 42 | `[1,1,8,8]` | `< 0.01` | 0.000123322 | 0.000016362 | 0.000014305 | yes | PASS |
| T3, BERT size seed 42 | `[2,12,128,128]` | `< 0.01` | 0.000698715 | 0.000003266 | 0.000173569 | yes | PASS |
| T4, one `+10` per row | `[1,1,128,128]` | `< 0.01` | 0.000901520 | 0.000014927 | 0.000107765 | yes | PASS |

Additional checks:

- T1 maximum error from the expected uniform value 0.25: `0.000003815`.
- T4 minimum output at the dominant location: `0.995168686`; maximum output at any other location: `0.000037193`.
- Every result was finite and nonnegative. No NaN or infinity was observed.

Raw logs and JSON metrics are stored on P0 under:

```text
/home/user/hpmpc/tests/results/softmax_3pc_20260715/T1_optimized/
/home/user/hpmpc/tests/results/softmax_3pc_20260715/T2_optimized/
/home/user/hpmpc/tests/results/softmax_3pc_20260715/T3_optimized/
/home/user/hpmpc/tests/results/softmax_3pc_20260715/T4_optimized/
```

Each directory contains `p0.log`, `p1.log`, `p2.log`, and `metrics.json`.

## T3 performance results

Framework communication is the sum of the online send and receive counters printed by hpmpc for all workers on that physical host. NIC utilization uses host `eth0` byte-counter deltas over controller-observed wall time:

```text
utilization = (RX bytes + TX bytes) * 8 / (wall seconds * 15e9) * 100
```

This is aggregate full-duplex utilization and is more generous than measuring either direction separately.

| Mode | Workers / shards | P0/P1/P2 framework communication (MB) | P0/P1/P2 max worker latency (ms) | Wall time (ms) | P0/P1/P2 NIC utilization | Acceptance |
|---|---:|---:|---:|---:|---:|---|
| `PROCESS_NUM=8,SPLITROLES=0` | 8 | 232.906 / 317.733 / 547.493 | 360.455 / 356.889 / 358.747 | 707.624 | 17.65% / 24.96% / 43.10% | FAIL: only P2 exceeds 40% utilization |
| `split2` diagnostic | 16 | 254.826 / 456.373 / 456.373 | 430.644 / 430.628 / 435.959 | 792.580 | 17.34% / 32.50% / 32.49% | FAIL: utilization < 40% |
| Requested `PROCESS_NUM=8,SPLITROLES=1` | 48 | 482.641 / 482.641 / 482.641 | 1446.559 / 1453.443 / 1457.325 | 1809.095 | 15.19% / 15.22% / 15.21% | FAIL: utilization < 40% |

For the requested 48-shard run, the measured NIC byte totals were 515.229 MB, 516.425 MB, and 515.930 MB. These include TCP/transport and control overhead, whereas the hpmpc online counters report 482.641 MB per node for the two-sample T3 batch, or 241.321 MB per sample. The protocol-level communication acceptance condition (`< 500 MB per sample`) passes; the observed NIC utilization condition does not.

The 16-shard diagnostic also produced all 16 output parts, reconstructed every T3 row exactly once, and retained the same maximum absolute error (`0.000698715`). It demonstrates that the low utilization is not caused by missing work or invalid output.

## Acceptance decision

| Requirement | Result |
|---|---|
| T1-T4 below their error thresholds | PASS |
| No NaN/INF | PASS |
| T3 communication < 500 MB per sample | PASS for requested `SPLITROLES=1` using hpmpc online counters (241.321 MB/sample/node; 482.641 MB for the two-sample batch) |
| Network utilization > 40% on the 15 Gbps network | PASS by user waiver on 2026-07-15; measured value remains 15.22% maximum |

Overall result: **ACCEPTED WITH USER WAIVER**. Numerical correctness, stability, real three-host communication, and the hpmpc communication-volume bound pass. The network-utilization measurement remains below the original threshold, but the user explicitly marked this item passed on 2026-07-15.

The next engineering target is the communication schedule rather than the fixed-point approximation: reduce per-round synchronization and/or batch independent rows into larger sends so that the 15 Gbps link remains occupied. The DReLU/maximum comparison tree is the first profile target.
